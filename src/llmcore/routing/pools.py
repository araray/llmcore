# src/llmcore/routing/pools.py
"""Pools: sets of interchangeable targets, with selection and failover.

A **pool** means "any of these can serve this request". That is the whole
definition, and it is why pools are kept separate from lanes (spec §1.1): a
failover set needs no classifier, and a named destination with one target
needs no failover set.

Selection answers "which member, right now?" and is split into two steps that
should not be conflated:

1. **Eligibility** — is this target usable at all? Cooldowns, auth failures
   and a context window too small for the prompt are all facts, independent of
   strategy. An ineligible target is recorded with the reason it was skipped,
   because "why did it pick the slow one?" is only answerable if the record
   says the fast one had twelve seconds of 429 cooldown left.
2. **Ordering** — among the eligible, which first? This is the strategy, and
   it is advisory. Every strategy produces a full ordering rather than a single
   pick, so the attempt loop can walk down it when the first choice fails.

Order tiers (``?order=`` on a member) cut across both: a tier-2 target is only
considered once every tier-1 target is ineligible. This is borrowed from
LiteLLM and is the honest way to express "use my own GPU, and only fall back
to a paid API if it is down".
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from .cards import context_window, estimate_cost_usd
from .models import Candidate, RoutingRequest, SelectionStrategy, Target, TargetHealth

if TYPE_CHECKING:
    from .state import InMemoryRoutingState

logger = logging.getLogger(__name__)

__all__ = [
    "POOL_MEMBER_PARAMS",
    "Pool",
    "select",
]

#: Parameters on a pool member that configure *routing* rather than the model,
#: and are stripped from the target before it reaches a provider.
#:
#: Carrying them in the spec string keeps a pool declarable as a plain list of
#: strings (``"openai:gpt-5.4?weight=3&order=1"``), which matters because a
#: pool has to be expressible in TOML, in an env var and over the bridge. The
#: strip is not optional: forwarding ``weight`` to a vendor would be a 400.
POOL_MEMBER_PARAMS: frozenset[str] = frozenset({"weight", "order"})

#: Reasons a candidate was skipped. Strings rather than an enum because they
#: are carried in events and read by humans, and several are parameterised.
_SKIP_COOLDOWN = "cooling down for {remaining:.0f}s after {failure}"
_SKIP_UNUSABLE = "marked unusable for this process ({failure})"
_SKIP_WINDOW = "context window {window} < {needed} estimated tokens"
_SKIP_TRIED = "already tried for this request"
_SKIP_TIER = "order tier {order} held back while tier {active} has candidates"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class Pool:
    """An ordered set of interchangeable targets.

    Args:
        name: The pool's name, as used in ``pool="main"`` and in events.
        targets: Members, in declared order. ``priority`` honours this order,
            and every other strategy falls back to it for ties and unknowns —
            so the declared order is never meaningless.
        strategy: How to order eligible members. ``None`` defers to
            :attr:`~llmcore.routing.settings.RoutingSettings.default_strategy`,
            which is itself a user config value.
        max_attempts: Distinct targets to try per request. ``None`` defers to
            settings. Capping this matters: an unbounded pool walk on a
            vendor-wide outage turns one failed request into fifteen paid
            attempts.
        affinity: ``"session"`` pins a session to its first target (the
            default where a session id exists), ``"none"`` distributes every
            request. See :meth:`affinity_is_sticky`.
        weights: Per-target weight for ``weighted``, keyed by
            :attr:`~llmcore.routing.models.Target.key`. Normally supplied as
            ``?weight=`` on the member and extracted by :meth:`from_config`.
        orders: Per-target order tier, same keying. Lower tiers are preferred
            outright; the default tier is ``0``.
    """

    name: str
    targets: tuple[Target, ...]
    strategy: SelectionStrategy | None = None
    max_attempts: int | None = None
    affinity: str = "session"
    weights: Mapping[str, float] = field(default_factory=dict)
    orders: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError(f"Pool '{self.name}' has no targets.")

    # -- construction -----------------------------------------------------

    @classmethod
    def from_config(cls, name: str, raw: Mapping[str, Any] | Sequence[Any] | str) -> Pool:
        """Build a pool from its config section.

        Accepts three shapes, because all three turn up in practice:

        * a mapping — the documented ``[routing.pools.<name>]`` form;
        * a bare list of specs — ``cheap = ["a", "b"]``, which is what people
          write before they need a strategy;
        * a single spec string — a one-member pool, which is a legitimate way
          to name a target.
        """
        if isinstance(raw, str):
            raw = {"targets": [raw]}
        elif isinstance(raw, Sequence) and not isinstance(raw, Mapping):
            raw = {"targets": list(raw)}

        specs = raw.get("targets") or raw.get("members") or []
        if isinstance(specs, str):
            specs = [specs]
        if not specs:
            raise ValueError(
                f"Pool '{name}' declares no targets. Add e.g. "
                f'targets = ["openai:gpt-5.4", "anthropic:claude-opus-5-5"].'
            )

        targets: list[Target] = []
        weights: dict[str, float] = {}
        orders: dict[str, int] = {}
        for spec in specs:
            target = Target.parse(spec) if isinstance(spec, str) else spec
            routing_params = {k: v for k, v in target.params.items() if k in POOL_MEMBER_PARAMS}
            if routing_params:
                # Strip before the target can reach a provider.
                target = replace(
                    target,
                    params={
                        k: v for k, v in target.params.items() if k not in POOL_MEMBER_PARAMS
                    },
                )
            if "weight" in routing_params:
                try:
                    weights[target.key] = max(0.0, float(routing_params["weight"]))
                except (TypeError, ValueError):
                    logger.warning(
                        "Pool '%s': ignoring non-numeric weight %r on %s",
                        name,
                        routing_params["weight"],
                        target.key,
                    )
            if "order" in routing_params:
                try:
                    orders[target.key] = int(routing_params["order"])
                except (TypeError, ValueError):
                    logger.warning(
                        "Pool '%s': ignoring non-integer order %r on %s",
                        name,
                        routing_params["order"],
                        target.key,
                    )
            targets.append(target)

        # Explicit tables win over the inline form, for users who prefer them.
        for key, value in (raw.get("weights") or {}).items():
            try:
                weights[Target.parse(key).key] = max(0.0, float(value))
            except (TypeError, ValueError):
                logger.warning("Pool '%s': ignoring non-numeric weight for %r", name, key)
        for key, value in (raw.get("orders") or {}).items():
            try:
                orders[Target.parse(key).key] = int(value)
            except (TypeError, ValueError):
                logger.warning("Pool '%s': ignoring non-integer order for %r", name, key)

        strategy_raw = raw.get("strategy")
        strategy: SelectionStrategy | None = None
        if strategy_raw:
            try:
                strategy = SelectionStrategy(str(strategy_raw).strip().lower())
            except ValueError:
                logger.warning(
                    "Pool '%s': unknown strategy %r; deferring to routing.default_strategy. "
                    "Known strategies: %s",
                    name,
                    strategy_raw,
                    ", ".join(s.value for s in SelectionStrategy),
                )

        max_attempts = raw.get("max_attempts")
        affinity = str(raw.get("affinity", "session")).strip().lower()
        if affinity not in ("session", "none"):
            logger.warning(
                "Pool '%s': unknown affinity %r; using 'session'. Valid: session, none.",
                name,
                affinity,
            )
            affinity = "session"

        return cls(
            name=name,
            targets=tuple(targets),
            strategy=strategy,
            max_attempts=int(max_attempts) if max_attempts is not None else None,
            affinity=affinity,
            weights=weights,
            orders=orders,
        )

    # -- queries ----------------------------------------------------------

    @property
    def affinity_is_sticky(self) -> bool:
        """Whether a session should keep the target it started on.

        Default-on, for a reason the request did not raise (spec §3.5):
        changing model mid-conversation throws away the cached prompt prefix,
        shifts output style under few-shot expectations, and invalidates
        preserved reasoning blocks. A single 429 should move *this turn*, not
        the conversation.
        """
        return self.affinity == "session"

    def weight_of(self, target: Target) -> float:
        """Weight for ``target``, defaulting to 1.0."""
        return float(self.weights.get(target.key, 1.0))

    def order_of(self, target: Target) -> int:
        """Order tier for ``target``, defaulting to 0 (the first tier)."""
        return int(self.orders.get(target.key, 0))

    def __contains__(self, target: object) -> bool:
        if isinstance(target, Target):
            return any(member.key == target.key for member in self.targets)
        if isinstance(target, str):
            key = Target.parse(target).key
            return any(member.key == key for member in self.targets)
        return False


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


def _eligibility(
    target: Target,
    health: TargetHealth,
    *,
    request: RoutingRequest | None,
    exclude: frozenset[str],
    min_context_tokens: int | None,
    provider_type_for: Callable[[Target], str] | None,
    now: datetime,
) -> tuple[bool, str | None, float]:
    """Decide whether ``target`` can be tried, and say why not if it cannot."""
    if target.key in exclude:
        return False, _SKIP_TRIED, 0.0

    if health.unusable:
        return False, _SKIP_UNUSABLE.format(failure=health.last_failure or "auth failure"), float(
            "inf"
        )

    remaining = health.cooldown_remaining(now)
    if remaining > 0:
        return (
            False,
            _SKIP_COOLDOWN.format(remaining=remaining, failure=health.last_failure or "a failure"),
            remaining,
        )

    # A pre-call window check: the card already knows this will fail, so
    # spending a request to be told so is pure waste. Unknown windows are
    # never a reason to skip -- plenty of routable models ship no card.
    needed = min_context_tokens
    if needed is None and request is not None:
        needed = request.approx_tokens
    if needed:
        provider_type = provider_type_for(target) if provider_type_for else None
        window = context_window(target, provider_type=provider_type)
        if window is not None and window < needed:
            return False, _SKIP_WINDOW.format(window=window, needed=needed), 0.0

    return True, None, 0.0


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def _by_priority(pool: Pool, eligible: list[tuple[Target, TargetHealth]]) -> list[Target]:
    """Declared order. Predictable, and the default for that reason."""
    return [target for target, _ in eligible]


def _by_round_robin(
    pool: Pool, eligible: list[tuple[Target, TargetHealth]], cursor: int
) -> list[Target]:
    """Declared order, rotated by a per-pool cursor."""
    targets = [target for target, _ in eligible]
    if not targets:
        return targets
    offset = cursor % len(targets)
    return targets[offset:] + targets[:offset]


def _by_weighted(
    pool: Pool, eligible: list[tuple[Target, TargetHealth]], rng: random.Random
) -> list[Target]:
    """Weighted random order.

    The *whole* list is shuffled by weight rather than just the head, so the
    fallback order after a failure stays weighted too. Zero-weight members
    sort last instead of being dropped: a weight of 0 means "do not normally
    pick this", and it should still be reachable when everything else is
    down — that is what makes it a usable way to park a spare.
    """
    scored: list[tuple[float, int, Target]] = []
    for index, (target, _) in enumerate(eligible):
        weight = pool.weight_of(target)
        if weight <= 0:
            score = float("inf")
        else:
            # Efraimidis-Spirakis: one exponential draw per item gives a
            # correctly weighted permutation in a single pass.
            score = rng.expovariate(1.0) / weight
        scored.append((score, index, target))
    scored.sort(key=lambda row: (row[0], row[1]))
    return [target for _, _, target in scored]


def _by_lowest_latency(pool: Pool, eligible: list[tuple[Target, TargetHealth]]) -> list[Target]:
    """Lowest smoothed latency first; unmeasured targets explored first.

    An unmeasured target sorts ahead of measured ones, which looks wrong and
    is deliberate: the strategy cannot prefer low latency without a
    measurement, and one call is all it takes to get one. Exploration is
    bounded by construction — after that call the target is ranked on its
    merits forever.
    """
    def sort_key(row: tuple[Target, TargetHealth]) -> tuple[int, float, int]:
        target, health = row
        index = pool.targets.index(target) if target in pool.targets else 0
        if health.ewma_latency_seconds is None:
            return (0, 0.0, index)
        return (1, health.ewma_latency_seconds, index)

    return [target for target, _ in sorted(eligible, key=sort_key)]


def _by_lowest_cost(
    pool: Pool,
    eligible: list[tuple[Target, TargetHealth]],
    *,
    request: RoutingRequest | None,
    provider_type_for: Callable[[Target], str] | None,
) -> tuple[list[Target], dict[str, float]]:
    """Cheapest estimated spend first, with unpriced targets after priced ones.

    The estimate uses the prompt's own token count and the card's prices, so
    it is a real comparison rather than a static ranking — a long prompt and a
    short one can legitimately choose different members when input and output
    prices differ.

    Unpriced targets rank *after* every priced one, in declared order. Ranking
    an unknown price as zero would make a model with no card beat a model
    known to be free, which is how a cost strategy ends up being the most
    expensive thing in the system.
    """
    input_tokens = request.approx_tokens if request is not None else 1_000
    # A guess, and labelled as one: output length is unknown before the call.
    # It is held constant across candidates, so it cannot change their order
    # unless their *output* prices differ, which is exactly when it should.
    output_tokens = 512
    scores: dict[str, float] = {}
    priced: list[tuple[float, int, Target]] = []
    unpriced: list[tuple[int, Target]] = []
    for index, (target, _) in enumerate(eligible):
        provider_type = provider_type_for(target) if provider_type_for else None
        cost = estimate_cost_usd(
            target,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider_type=provider_type,
        )
        if cost is None:
            unpriced.append((index, target))
        else:
            scores[target.key] = cost
            priced.append((cost, index, target))
    priced.sort(key=lambda row: (row[0], row[1]))
    ordered = [target for _, _, target in priced] + [target for _, target in unpriced]
    return ordered, scores


def _by_least_busy(pool: Pool, eligible: list[tuple[Target, TargetHealth]]) -> list[Target]:
    """Fewest in-flight requests first, declared order breaking ties."""
    def sort_key(row: tuple[Target, TargetHealth]) -> tuple[int, int]:
        target, health = row
        index = pool.targets.index(target) if target in pool.targets else 0
        return (health.in_flight, index)

    return [target for target, _ in sorted(eligible, key=sort_key)]


def _by_most_credits(
    pool: Pool, eligible: list[tuple[Target, TargetHealth]]
) -> tuple[list[Target], dict[str, float]]:
    """Three bands: known funds, then unknown, then known-empty.

    Most vendors expose no balance at all (spec §3.4), so "unknown" is the
    common case and must not be read as empty — doing so would demote every
    provider that simply has no endpoint.

    The band order is the part worth stating, because the obvious
    implementation gets it wrong. Sorting all known balances descending puts a
    confirmed **zero** ahead of an unknown, which is backwards: an unknown
    account might have money, whereas one we have measured at zero certainly
    does not. So an empty wallet ranks *last*, behind the providers we simply
    cannot see into.

    What makes the strategy useful even with no probe anywhere is that an
    observed insufficient-credit failure records a balance of zero, so the
    signal arrives from failure rather than from polling — and once the
    cooldown lapses, that target is correctly the last resort instead of the
    first choice.
    """
    scores: dict[str, float] = {}
    funded: list[tuple[float, int, Target]] = []
    unknown: list[tuple[int, Target]] = []
    empty: list[tuple[int, Target]] = []
    for index, (target, health) in enumerate(eligible):
        balance = health.balance
        if balance is not None and balance.is_known and balance.amount is not None:
            scores[target.key] = balance.amount
            if balance.amount > 0:
                funded.append((balance.amount, index, target))
            else:
                empty.append((index, target))
        else:
            unknown.append((index, target))
    funded.sort(key=lambda row: (-row[0], row[1]))
    ordered = (
        [target for _, _, target in funded]
        + [target for _, target in unknown]
        + [target for _, target in empty]
    )
    return ordered, scores


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def select(
    pool: Pool,
    *,
    state: InMemoryRoutingState,
    strategy: SelectionStrategy | None = None,
    request: RoutingRequest | None = None,
    exclude: frozenset[str] | set[str] | None = None,
    min_context_tokens: int | None = None,
    provider_type_for: Callable[[Target], str] | None = None,
    rng: random.Random | None = None,
    cursor: int = 0,
    now: datetime | None = None,
) -> tuple[Target | None, tuple[Candidate, ...]]:
    """Order ``pool``'s members for this request.

    Args:
        pool: The pool to select from.
        state: Health store, read for cooldowns, latency, in-flight and balance.
        strategy: Override the pool's strategy — a per-request override, since
            config is a warm-up and not a cage. Falls back to the pool's own
            strategy, then to ``priority``.
        request: The request, used for the token estimate that drives
            ``lowest_cost`` and the pre-call window check.
        exclude: Target keys already tried for this request.
        min_context_tokens: Override the estimated token requirement, for a
            caller that knows better (a retry after a real
            ``ContextLengthError`` knows the true number).
        provider_type_for: Maps a target to its provider *type*, for card
            lookup when the target names a configured instance.
        rng: Injectable randomness, so ``weighted`` is testable.
        cursor: Rotation counter for ``round_robin``.
        now: Injectable clock.

    Returns:
        ``(chosen, candidates)``. ``chosen`` is ``None`` when every member was
        ineligible; ``candidates`` always describes every member and why it was
        or was not usable, in final order, so the caller can report the
        decision without recomputing it.
    """
    now = now or _utcnow()
    exclude = frozenset(exclude or ())
    strategy = strategy or pool.strategy or SelectionStrategy.PRIORITY

    rows: list[tuple[Target, TargetHealth, bool, str | None, float]] = []
    for target in pool.targets:
        health = state.health_sync(target.key)
        ok, reason, cooldown = _eligibility(
            target,
            health,
            request=request,
            exclude=exclude,
            min_context_tokens=min_context_tokens,
            provider_type_for=provider_type_for,
            now=now,
        )
        rows.append((target, health, ok, reason, cooldown))

    # Order tiers gate eligibility: hold back a higher tier entirely while a
    # lower one still has something usable.
    eligible_rows = [row for row in rows if row[2]]
    if eligible_rows and pool.orders:
        active_tier = min(pool.order_of(row[0]) for row in eligible_rows)
        held_back = {
            row[0].key: pool.order_of(row[0])
            for row in eligible_rows
            if pool.order_of(row[0]) > active_tier
        }
        if held_back:
            rows = [
                (
                    target,
                    health,
                    False,
                    _SKIP_TIER.format(order=held_back[target.key], active=active_tier),
                    cooldown,
                )
                if target.key in held_back
                else (target, health, ok, reason, cooldown)
                for target, health, ok, reason, cooldown in rows
            ]
            eligible_rows = [row for row in rows if row[2]]

    eligible = [(row[0], row[1]) for row in eligible_rows]
    scores: dict[str, float] = {}

    if strategy is SelectionStrategy.ROUND_ROBIN:
        ordered = _by_round_robin(pool, eligible, cursor)
    elif strategy is SelectionStrategy.WEIGHTED:
        ordered = _by_weighted(pool, eligible, rng or random)
        scores = {target.key: pool.weight_of(target) for target, _ in eligible}
    elif strategy is SelectionStrategy.LOWEST_LATENCY:
        ordered = _by_lowest_latency(pool, eligible)
        scores = {
            target.key: health.ewma_latency_seconds
            for target, health in eligible
            if health.ewma_latency_seconds is not None
        }
    elif strategy is SelectionStrategy.LOWEST_COST:
        ordered, scores = _by_lowest_cost(
            pool, eligible, request=request, provider_type_for=provider_type_for
        )
    elif strategy is SelectionStrategy.LEAST_BUSY:
        ordered = _by_least_busy(pool, eligible)
        scores = {target.key: float(health.in_flight) for target, health in eligible}
    elif strategy is SelectionStrategy.MOST_CREDITS:
        ordered, scores = _by_most_credits(pool, eligible)
    else:
        ordered = _by_priority(pool, eligible)

    rank = {target.key: index for index, target in enumerate(ordered)}
    candidates = sorted(
        (
            Candidate(
                target=target,
                eligible=ok,
                reason=reason,
                score=scores.get(target.key),
                cooldown_remaining=cooldown,
            )
            for target, _, ok, reason, cooldown in rows
        ),
        # Eligible first, in strategy order; the rest keep declared order so a
        # report reads the way the config does.
        key=lambda candidate: (
            0 if candidate.eligible else 1,
            rank.get(candidate.target.key, len(ordered)),
        ),
    )
    return (ordered[0] if ordered else None), tuple(candidates)
