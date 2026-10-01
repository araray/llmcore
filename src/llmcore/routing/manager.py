# src/llmcore/routing/manager.py
"""The routing manager: the one place the five layers meet.

Everything else in this package is deliberately independent — a pool knows
nothing about classifiers, a transform knows nothing about pools. This module
is where they compose, in a fixed order that the rest of the design depends on:

1. **Classify** — a chain of classifiers names a lane (or nobody has an
   opinion, and the default pool serves).
2. **Resolve** — the lane becomes a pool or a single target; a pool selects a
   member; parameters are layered.
3. **Transform** — the chosen target is shown to the transform chain, which may
   rewrite the prompt, *change the destination*, or refuse outright.
4. **Attempt** — call it; on failure, classify the failure and decide whether
   to retry the same target, move to a peer, or stop.
5. **Cascade** — if configured, verify the answer and escalate only if it fell
   short.

Step 3 coming *after* step 2 is the part worth defending: a transform's whole
value is that it sees where the prompt is about to go, so it cannot run before
a target is chosen. The cost is that a ``constrain`` action has to re-resolve
against a different pool, which this module does explicitly and once.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Mapping, Sequence

from ..exceptions import (
    ConfigError,
    ContextLengthError,
    NoTargetAvailableError,
    PromptBlockedError,
)
from . import events
from .cards import context_window
from .classifiers import ClassifierChain, build_classifier
from .lanes import Lane, parse_lanes
from .models import (
    Candidate,
    Classification,
    FailureKind,
    Outcome,
    RoutingPlan,
    RoutingRequest,
    SelectionStrategy,
    Target,
    TargetHealth,
    TransformAction,
    TransformResult,
    Verdict,
    classify_failure,
)
from .pools import Pool, select
from .settings import ClassifierBias, RoutingSettings, UnknownVerdictPolicy
from .state import InMemoryRoutingState
from .transforms import TransformChain, build_transform
from .verifiers import build_verifier

logger = logging.getLogger(__name__)

__all__ = ["RoutingManager", "RoutingResult"]

#: Signature of the callable that actually performs a provider call. The
#: manager never builds a chat request itself — it is given a runner, which
#: keeps routing independent of the chat API and makes it reusable for
#: embeddings, media or anything else with a provider and a model.
Runner = Callable[[Any, Target, dict[str, Any]], Awaitable[Any]]


@dataclass(slots=True)
class RoutingResult:
    """What a routed call returns, with the decision attached.

    Attributes:
        value: Whatever the runner returned.
        target: The target that actually served the request.
        plan: The decision, for logging and ``explain()``.
        attempts: Every outcome recorded, in order — including the failures
            that led here. A caller that wants to know it was served by the
            third choice can see it.
        verdict: The cascade's judgement, when one ran.
        rungs_used: How many cascade rungs were spent.
    """

    value: Any
    target: Target
    plan: RoutingPlan
    attempts: tuple[Outcome, ...] = ()
    verdict: Verdict | None = None
    rungs_used: int = 1

    @property
    def failed_over(self) -> bool:
        """Whether anything had to fail before this answer arrived."""
        return any(not outcome.ok for outcome in self.attempts)


class RoutingManager:
    """Routes a request through pools, lanes, transforms and cascades.

    Args:
        provider_manager: The :class:`~llmcore.providers.manager.ProviderManager`
            that resolves a target to a live provider.
        config: An llmcore config object (anything with ``get(key, default)``).
        settings: Pre-resolved settings. Normally built from *config*; passed
            explicitly in tests and by callers that have already resolved.
        state: Health store. Defaults to an in-process one.
        pools: Pre-built pools, overriding config.
        lanes: Pre-built lanes, overriding config.
        ask: ``async (prompt, target) -> str``, used by the ``llm`` classifier
            and verifier. Injected by :class:`~llmcore.api.LLMCore` so routing
            does not import the chat API, which would be circular.
    """

    def __init__(
        self,
        provider_manager: Any,
        *,
        config: Any = None,
        settings: RoutingSettings | None = None,
        state: InMemoryRoutingState | None = None,
        pools: Mapping[str, Pool] | None = None,
        lanes: Mapping[str, Lane] | None = None,
        ask: Callable[..., Awaitable[str]] | None = None,
    ) -> None:
        self._providers = provider_manager
        self._config = config
        self._ask = ask
        get = getattr(config, "get", None) if config is not None else None
        self._settings = settings or RoutingSettings.from_config(get)
        self._state = state or InMemoryRoutingState(self._settings.cooldowns)

        self._pools: dict[str, Pool] = dict(pools) if pools is not None else self._load_pools(get)
        self._lanes: dict[str, Lane] = (
            dict(lanes) if lanes is not None else parse_lanes(_read(get, "routing.lanes", {}))
        )
        self._profiles: dict[str, dict[str, Any]] = {
            str(name).strip().lower(): dict(values or {})
            for name, values in (_read(get, "routing.profiles", {}) or {}).items()
        }
        self._cascades: dict[str, dict[str, Any]] = {
            str(name).strip().lower(): dict(values or {})
            for name, values in (_read(get, "routing.cascade.rungs", {}) or {}).items()
        }
        if not self._cascades:
            rungs = _read(get, "routing.cascade.default.rungs", None) or _read(
                get, "routing.cascade.default", None
            )
            if rungs:
                self._cascades["default"] = (
                    {"rungs": rungs} if isinstance(rungs, (list, tuple)) else dict(rungs)
                )

        self._classifier_config = dict(_read(get, "routing.classifier", {}) or {})
        self._transform_config = dict(_read(get, "routing.transforms", {}) or {})
        self._chain: ClassifierChain | None = None
        self._transform_chain: TransformChain | None = None
        self._verifier: Any = None
        self._round_robin_cursors: dict[str, int] = {}
        self._session_pins: dict[str, str] = {}
        self._rng = random.Random()

    # ------------------------------------------------------------------
    # Construction from config
    # ------------------------------------------------------------------

    def _load_pools(self, get: Any) -> dict[str, Pool]:
        raw = _read(get, "routing.pools", {}) or {}
        pools: dict[str, Pool] = {}
        for name, section in dict(raw).items():
            key = str(name).strip().lower()
            try:
                pools[key] = Pool.from_config(key, section)
            except (ValueError, TypeError) as exc:
                # One malformed pool must not cost the others. Routing degrades
                # to the remaining pools, which is strictly better than a
                # process that will not start because of a typo.
                logger.error("Ignoring malformed pool '%s': %s", key, exc)
        return pools

    @property
    def settings(self) -> RoutingSettings:
        """The resolved settings this manager was built with."""
        return self._settings

    @property
    def state(self) -> InMemoryRoutingState:
        return self._state

    def pools(self) -> dict[str, list[str]]:
        """Pool name → member spec strings, for reporting."""
        return {
            name: [target.spec() for target in pool.targets] for name, pool in self._pools.items()
        }

    def lanes(self) -> dict[str, str]:
        """Lane name → destination, for reporting and ``/v1/models``."""
        return {name: lane.destination for name, lane in self._lanes.items()}

    def profiles(self) -> dict[str, dict[str, Any]]:
        """Parameter profiles, for reporting."""
        return {name: dict(values) for name, values in self._profiles.items()}

    def health(self) -> dict[str, dict[str, Any]]:
        """Per-target health, for ``llm.routing.health()``.

        A plain dict rather than the dataclasses, because this is the thing a
        human or a CLI reads when a pool is misbehaving and it should not
        require importing llmcore's types to inspect.
        """
        now = datetime.now(timezone.utc)
        out: dict[str, dict[str, Any]] = {}
        for key, record in self._state.snapshot_sync().items():
            out[key] = {
                "available": record.is_available(now),
                "unusable": record.unusable,
                "cooldown_remaining": record.cooldown_remaining(now),
                "consecutive_failures": record.consecutive_failures,
                "successes": record.total_successes,
                "failures": record.total_failures,
                "ewma_latency_seconds": record.ewma_latency_seconds,
                "in_flight": record.in_flight,
                "last_failure": str(record.last_failure) if record.last_failure else None,
                "last_error": record.last_error,
                "balance": (
                    {"amount": record.balance.amount, "unit": record.balance.unit}
                    if record.balance is not None
                    else None
                ),
            }
        return out

    # ------------------------------------------------------------------
    # Lazily built chains
    # ------------------------------------------------------------------

    def classifier_chain(self, settings: RoutingSettings | None = None) -> ClassifierChain:
        """Build (once) the configured classifier chain."""
        if self._chain is not None:
            return self._chain
        settings = settings or self._settings
        config = dict(self._classifier_config)
        config.setdefault("provider_getter", self._typesafe_provider)
        if self._ask is not None:
            config.setdefault("ask", self._ask)
        if settings.magic_pattern:
            config.setdefault("magic_pattern", settings.magic_pattern)

        built = [
            classifier
            for classifier in (
                build_classifier(name, config=config, lanes=self._lanes)
                for name in settings.classifier_chain
            )
            if classifier is not None
        ]
        self._chain = ClassifierChain(built, min_confidence=settings.min_confidence)
        if settings.classifier_chain and not built:
            logger.warning(
                "routing.classifier.chain names %d classifier(s) but none could be built; "
                "requests will fall through to the default pool.",
                len(settings.classifier_chain),
            )
        return self._chain

    def transform_chain(self, settings: RoutingSettings | None = None) -> TransformChain:
        """Build (once) the configured transform chain.

        ``strip_magic`` is appended automatically whenever ``magic_string`` is
        classifying, so a user cannot configure the leak by forgetting it: a
        marker that reaches a provider is llmcore's internals showing up in
        someone's context window.
        """
        if self._transform_chain is not None:
            return self._transform_chain
        settings = settings or self._settings
        names = list(settings.transforms)
        if "magic_string" in settings.classifier_chain and "strip_magic" not in names:
            names.append("strip_magic")

        config = dict(self._transform_config)
        if settings.magic_pattern:
            config.setdefault("magic_pattern", settings.magic_pattern)
        built = [
            transform
            for transform in (build_transform(name, config=config) for name in names)
            if transform is not None
        ]
        self._transform_chain = TransformChain(
            built, fail_closed=_as_bool(self._transform_config.get("fail_closed", True))
        )
        return self._transform_chain

    def verifier(self, settings: RoutingSettings | None = None) -> Any:
        """Build (once) the configured cascade verifier."""
        if self._verifier is not None:
            return self._verifier
        settings = settings or self._settings
        name = str(
            self._cascades.get("default", {}).get("verifier")
            or _read(getattr(self._config, "get", None), "routing.cascade.verifier", "")
            or ""
        ).strip()
        if not name:
            return None
        config = dict(self._cascades.get("default", {}))
        config.setdefault("threshold", settings.cascade_threshold)
        config.setdefault("provider_getter", self._typesafe_provider)
        if self._ask is not None:
            config.setdefault("ask", self._ask)
        self._verifier = build_verifier(name, config=config)
        return self._verifier

    def _typesafe_provider(self) -> Any:
        """Resolve the TypeSafe provider, or ``None`` if it is unreachable."""
        try:
            return self._providers.resolve_target(
                Target(provider="typesafe"), autoprovision=self._settings.autoprovision
            )
        except Exception as exc:
            logger.debug("TypeSafe provider is unavailable for routing: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    async def plan(
        self,
        request: RoutingRequest,
        *,
        settings: RoutingSettings | None = None,
        target: str | Target | None = None,
        pool: str | None = None,
        lane: str | None = None,
        profile: str | None = None,
        exclude: frozenset[str] | None = None,
        min_context_tokens: int | None = None,
    ) -> RoutingPlan:
        """Decide where a request should go, without sending it.

        This is what ``llm.routing.explain()`` returns, and it is part of the
        first implementation on purpose: opaque routing is a support burden,
        and the first question anyone asks is "why did it pick that?".
        """
        settings = settings or self._settings
        notes: list[str] = []

        # 1. An explicit target bypasses everything. Pinning a target is the
        #    documented way to take routing out of the picture entirely, which
        #    matters for reproducing a result.
        if target is not None:
            chosen = Target.parse(target) if isinstance(target, str) else target
            chosen = self._apply_params(chosen, lane=None, profile=profile)
            return RoutingPlan(
                chosen=chosen,
                candidates=(Candidate(target=chosen, eligible=True, reason="pinned by caller"),),
                notes=("target pinned by the caller; classification and pools were skipped",),
                estimated_cost_usd=self._estimate(chosen, request),
            )

        classification: Classification | None = None
        lane_name = lane

        # 2. Classify, unless a lane or pool was named outright.
        if lane_name is None and pool is None and settings.enabled:
            chain = self.classifier_chain(settings)
            if chain:
                classification = await chain.classify(request)
                if classification is not None:
                    if classification.target is not None:
                        chosen = self._apply_params(
                            classification.target, lane=None, profile=profile
                        )
                        plan = RoutingPlan(
                            chosen=chosen,
                            classification=classification,
                            candidates=(
                                Candidate(
                                    target=chosen,
                                    eligible=True,
                                    reason=f"suggested by the '{classification.source}' classifier",
                                ),
                            ),
                            estimated_cost_usd=self._estimate(chosen, request),
                        )
                        events.emit_classified(plan)
                        return plan
                    lane_name = self._biased_lane(classification, settings, notes)

        if lane_name is None:
            lane_name = settings.default_lane

        # 3. Lane → pool or target.
        lane_obj = self._lanes.get(lane_name) if lane_name else None
        if lane_name and lane_obj is None:
            notes.append(
                f"classifier chose lane '{lane_name}', which is not configured; "
                f"falling back to the default pool"
            )
            logger.warning(
                "Lane '%s' is not configured. Configured lanes: %s",
                lane_name,
                ", ".join(sorted(self._lanes)) or "(none)",
            )
            lane_name = None

        pool_name = pool
        if lane_obj is not None:
            if lane_obj.target is not None:
                chosen = self._apply_params(lane_obj.target, lane=lane_obj, profile=profile)
                plan = RoutingPlan(
                    chosen=chosen,
                    lane=lane_name,
                    classification=classification,
                    candidates=(
                        Candidate(
                            target=chosen, eligible=True, reason=f"lane '{lane_name}' target"
                        ),
                    ),
                    notes=tuple(notes),
                    estimated_cost_usd=self._estimate(chosen, request),
                )
                events.emit_classified(plan)
                events.emit_selected(plan)
                return plan
            pool_name = pool_name or lane_obj.pool

        pool_name = pool_name or settings.default_pool
        if not pool_name:
            # No routing configured at all: this is the "config is a preset
            # layer" case, and the provider manager's own default serves.
            return RoutingPlan(
                chosen=None,
                lane=lane_name,
                classification=classification,
                notes=tuple(
                    notes + ["no pool, lane or default_pool configured; using llmcore's default provider"]
                ),
            )

        pool_obj = self._pools.get(pool_name)
        if pool_obj is None:
            raise ConfigError(
                f"Pool '{pool_name}' is not configured. Configured pools: "
                f"{', '.join(sorted(self._pools)) or '(none)'}."
            )

        # 4. Session affinity, before selection, so a pinned target is simply
        #    the first candidate rather than a special case in the loop.
        pinned = self._pinned_target(request, pool_obj, settings)

        chosen, candidates = select(
            pool_obj,
            state=self._state,
            strategy=self._strategy_for(pool_obj, settings),
            request=request,
            exclude=exclude,
            min_context_tokens=min_context_tokens,
            provider_type_for=self._provider_type_for,
            rng=self._rng,
            cursor=self._round_robin_cursors.get(pool_name, 0),
            now=datetime.now(timezone.utc),
        )
        if pinned is not None:
            still_eligible = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.target.key == pinned and candidate.eligible
                ),
                None,
            )
            if still_eligible is not None:
                chosen = still_eligible.target
                notes.append(f"session affinity kept this conversation on {pinned}")
            else:
                notes.append(
                    f"session affinity wanted {pinned}, which is not currently usable; "
                    f"moving the pin"
                )

        if chosen is not None:
            self._round_robin_cursors[pool_name] = (
                self._round_robin_cursors.get(pool_name, 0) + 1
            )
            chosen = self._apply_params(chosen, lane=lane_obj, profile=profile)

        plan = RoutingPlan(
            chosen=chosen,
            pool=pool_name,
            lane=lane_name,
            strategy=self._strategy_for(pool_obj, settings),
            classification=classification,
            candidates=candidates,
            notes=tuple(notes),
            estimated_cost_usd=self._estimate(chosen, request) if chosen else None,
        )
        if classification is not None:
            events.emit_classified(plan)
        events.emit_selected(plan)
        return plan

    def _biased_lane(
        self, classification: Classification, settings: RoutingSettings, notes: list[str]
    ) -> str | None:
        """Apply the configured bias to a classifier's lane choice.

        The two error directions are not symmetric: routing too cheap produces
        a bad answer, routing too expensive only costs money. So when a
        classifier is near its floor, ``bias`` decides which way to lean —
        and ``quality`` is the default because a wrong answer is the more
        expensive mistake in every setting except a deliberately frugal one.

        The bias only moves a *borderline* decision. A confident classifier is
        not second-guessed, because overriding a confident classifier would
        make the whole chain pointless.
        """
        lane = classification.lane
        if lane is None or settings.bias is ClassifierBias.NEUTRAL:
            return lane
        confidence = classification.confidence
        if confidence is None or confidence >= settings.min_confidence + 0.2:
            return lane

        ranked = [name for name in self._lanes if name in (classification.scores or {})]
        if len(ranked) < 2:
            return lane
        ordered = sorted(ranked, key=lambda name: -classification.scores[name])
        if len(ordered) < 2 or ordered[0] != lane:
            return lane
        runner_up = ordered[1]
        gap = classification.scores[lane] - classification.scores[runner_up]
        if gap > 0.15:
            return lane

        # Borderline. Prefer whichever of the two the bias points at, using
        # the lanes' own estimated cost as the ordering.
        cheaper, dearer = self._cheaper_of(lane, runner_up)
        pick = cheaper if settings.bias is ClassifierBias.COST else dearer
        if pick != lane:
            notes.append(
                f"bias={settings.bias.value} moved a borderline decision from '{lane}' "
                f"to '{pick}' (gap {gap:.2f})"
            )
        return pick

    def _cheaper_of(self, first: str, second: str) -> tuple[str, str]:
        """Order two lanes by estimated cost, cheapest first.

        Where a cost cannot be estimated the declared order of the lane table
        decides, which at least is deterministic and inspectable.
        """
        costs: dict[str, float] = {}
        for name in (first, second):
            lane = self._lanes.get(name)
            targets: Sequence[Target] = ()
            if lane is None:
                continue
            if lane.target is not None:
                targets = (lane.target,)
            elif lane.pool and lane.pool in self._pools:
                targets = self._pools[lane.pool].targets
            estimates = [
                value
                for value in (
                    self._estimate(target, None, output_tokens=512) for target in targets
                )
                if value is not None
            ]
            if estimates:
                costs[name] = min(estimates)
        if first in costs and second in costs:
            return (
                (first, second) if costs[first] <= costs[second] else (second, first)
            )
        order = list(self._lanes)
        if first in order and second in order and order.index(first) < order.index(second):
            return first, second
        return second, first

    def _strategy_for(self, pool: Pool, settings: RoutingSettings) -> SelectionStrategy:
        return pool.strategy or settings.default_strategy

    def _pinned_target(
        self, request: RoutingRequest, pool: Pool, settings: RoutingSettings
    ) -> str | None:
        """The target this session is pinned to, if affinity applies."""
        if not request.session_id:
            return None
        if settings.affinity != "session" or not pool.affinity_is_sticky:
            return None
        return self._session_pins.get(f"{pool.name}:{request.session_id}")

    def _remember_pin(self, request: RoutingRequest, pool_name: str | None, target: Target) -> None:
        if request.session_id and pool_name:
            self._session_pins[f"{pool_name}:{request.session_id}"] = target.key

    def _provider_type_for(self, target: Target) -> str:
        """Map a target to its provider *type*, for model-card lookups.

        A configured instance may be called anything (``my-openai``), and a
        card is filed under the type, so the type has to come from the live
        provider rather than from the target's text.
        """
        try:
            provider = self._providers.resolve_target(
                target, autoprovision=False, cache=False
            )
            return str(provider.get_name())
        except Exception:
            return target.provider

    def _estimate(
        self, target: Target | None, request: RoutingRequest | None, *, output_tokens: int = 512
    ) -> float | None:
        if target is None:
            return None
        from .cards import estimate_cost_usd

        return estimate_cost_usd(
            target,
            input_tokens=request.approx_tokens if request is not None else 1_000,
            output_tokens=output_tokens,
            provider_type=self._provider_type_for(target),
        )

    def _apply_params(
        self, target: Target, *, lane: Lane | None, profile: str | None
    ) -> Target:
        """Layer parameters onto a target, lowest precedence first.

        Spec §6.1: target params, then lane params, then a named profile. The
        per-call ``**kwargs`` are applied by the caller afterwards, since they
        are the highest layer and the manager never sees them.
        """
        params = dict(target.params)
        if lane is not None and lane.params:
            params.update(lane.params)
        if profile:
            values = self._profiles.get(profile.strip().lower())
            if values is None:
                logger.warning(
                    "Profile '%s' is not configured; ignoring it. Configured profiles: %s",
                    profile,
                    ", ".join(sorted(self._profiles)) or "(none)",
                )
            else:
                params.update(values)
        return replace(target, params=params) if params != target.params else target

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(
        self,
        request: RoutingRequest,
        runner: Runner,
        *,
        settings: RoutingSettings | None = None,
        target: str | Target | None = None,
        pool: str | None = None,
        lane: str | None = None,
        profile: str | None = None,
        call_params: Mapping[str, Any] | None = None,
        cascade: str | None = None,
    ) -> RoutingResult:
        """Route and call, failing over as the failure taxonomy dictates.

        Args:
            request: The request, as classifiers and transforms will see it.
            runner: ``async (provider, target, params) -> value``.
            settings: Resolved settings for this call.
            target: Pin a target and bypass routing.
            pool: Route through this pool.
            lane: Route to this lane, skipping classification.
            profile: Apply a named parameter profile.
            call_params: Per-call parameters — the highest precedence layer.
            cascade: Name of a cascade to run after a successful answer.

        Returns:
            A :class:`RoutingResult`.

        Raises:
            PromptBlockedError: A transform refused to send the prompt.
            NoTargetAvailableError: Every candidate was unusable, with the
                per-candidate reasons attached.
        """
        settings = settings or self._settings
        attempts: list[Outcome] = []
        tried: set[str] = set()
        reasons: dict[str, str] = {}
        last_error: BaseException | None = None
        min_context_tokens: int | None = None
        constrained_pool: str | None = None
        current_request = request

        for attempt_index in range(1, max(1, settings.max_attempts) + 1):
            plan = await self.plan(
                current_request,
                settings=settings,
                target=target,
                pool=constrained_pool or pool,
                lane=lane,
                profile=profile,
                exclude=frozenset(tried),
                min_context_tokens=min_context_tokens,
            )
            for candidate in plan.candidates:
                if not candidate.eligible and candidate.reason:
                    reasons.setdefault(candidate.target.spec(), candidate.reason)

            if plan.chosen is None:
                if plan.pool is None and not reasons:
                    # Nothing configured: hand back to the caller's default.
                    provider = self._providers.get_provider()
                    value = await runner(provider, Target(provider=provider.get_name()), dict(call_params or {}))
                    return RoutingResult(
                        value=value,
                        target=Target(provider=provider.get_name()),
                        plan=plan,
                        attempts=tuple(attempts),
                    )
                break

            chosen = plan.chosen

            # Transforms see the chosen target, which is the whole point.
            transformed_request, transform_results = await self.transform_chain(settings).apply(
                current_request, chosen
            )
            for result in transform_results:
                events.emit_transformed(chosen.key, result)

            blocked = next(
                (r for r in transform_results if r.action is TransformAction.BLOCK), None
            )
            if blocked is not None:
                raise PromptBlockedError(
                    blocked.reason or "A transform blocked this prompt.",
                    transform=blocked.source or None,
                    findings=[(f.kind, f.where, f.hashed) for f in blocked.findings],
                )

            constrain = next(
                (
                    r
                    for r in transform_results
                    if r.action is TransformAction.CONSTRAIN and r.constrain_to_pool
                ),
                None,
            )
            if constrain is not None and constrain.constrain_to_pool != plan.pool:
                if constrain.constrain_to_pool not in self._pools:
                    raise PromptBlockedError(
                        f"A transform required pool '{constrain.constrain_to_pool}', which is "
                        f"not configured; refusing to send this prompt to {chosen.key} instead.",
                        transform=constrain.source or None,
                        findings=[(f.kind, f.where, f.hashed) for f in constrain.findings],
                    )
                logger.info(
                    "routing: %s constrained this request to pool '%s'",
                    constrain.source or "a transform",
                    constrain.constrain_to_pool,
                )
                constrained_pool = constrain.constrain_to_pool
                current_request = transformed_request
                # Re-plan against the constrained pool. This does not count as
                # an attempt: nothing was called.
                target = None
                lane = None
                continue

            current_request = transformed_request
            params = self._merge_params(chosen, call_params)

            provider = self._providers.resolve_target(
                chosen,
                autoprovision=settings.autoprovision,
                cache=settings.autoprovision_cache,
            )

            outcome, value, error = await self._attempt(
                runner,
                provider,
                chosen,
                params,
                settings=settings,
                plan=plan,
                attempt=attempt_index,
            )
            attempts.append(outcome)
            tried.add(chosen.key)

            if outcome.ok:
                self._remember_pin(current_request, plan.pool, chosen)
                result = RoutingResult(
                    value=value, target=chosen, plan=plan, attempts=tuple(attempts)
                )
                if settings.cascade_enabled:
                    result = await self._cascade(
                        current_request, result, runner, settings=settings, name=cascade
                    )
                return result

            last_error = error
            kind = outcome.failure or FailureKind.UNKNOWN
            reasons[chosen.spec()] = f"{kind}: {outcome.error or 'failed'}"

            if not settings.should_failover(kind):
                assert error is not None
                raise error

            if kind is FailureKind.CONTEXT_LENGTH and settings.reroute_on_context_overflow:
                # A context overflow is a routing signal, not an error: the
                # card knows every model's window, so the prompt can go to a
                # bigger one instead of failing.
                min_context_tokens = self._required_window(error, current_request, chosen)
                logger.info(
                    "routing: %s cannot hold this prompt; looking for a window of at least "
                    "%s tokens",
                    chosen.key,
                    min_context_tokens,
                )

            if attempt_index < settings.max_attempts:
                events.emit_failover(
                    from_target=chosen.key,
                    to_target="(selecting)",
                    failure=str(kind),
                    attempt=attempt_index,
                    pool=plan.pool,
                )

        events.emit_exhausted(
            pool=constrained_pool or pool,
            attempts=len(attempts),
            candidates=[{"target": spec, "reason": reason} for spec, reason in reasons.items()],
        )
        if last_error is not None and not reasons:
            raise last_error
        raise NoTargetAvailableError(
            "Routing exhausted every candidate.",
            pool=constrained_pool or pool,
            lane=lane,
            candidates=sorted(reasons.items()),
            last_error=last_error,
        )

    async def _attempt(
        self,
        runner: Runner,
        provider: Any,
        target: Target,
        params: dict[str, Any],
        *,
        settings: RoutingSettings,
        plan: RoutingPlan,
        attempt: int,
    ) -> tuple[Outcome, Any, BaseException | None]:
        """Call one target, retrying it in place where that is sensible.

        A 5xx or a dropped connection is worth one more go at the *same*
        target: it is usually a single bad node behind a load balancer, and
        moving to a peer would throw away a warm prompt cache for nothing.
        Every other failure moves on immediately.
        """
        retries = settings.retry_same_attempts
        last_error: BaseException | None = None

        for same_attempt in range(retries + 1):
            started = time.perf_counter()
            try:
                async with self._state.in_flight(target.key):
                    value = await runner(provider, target, params)
            except BaseException as exc:  # noqa: BLE001 - classified immediately below
                elapsed = time.perf_counter() - started
                kind, retry_after = classify_failure(exc)
                last_error = exc
                outcome = Outcome(
                    target_key=target.key,
                    ok=False,
                    latency_seconds=elapsed,
                    failure=kind,
                    retry_after_seconds=retry_after,
                    error=str(exc)[:300],
                )
                await self._state.record(target.key, outcome)
                events.emit_attempt(outcome, attempt=attempt, pool=plan.pool)
                if kind.retry_same and same_attempt < retries:
                    logger.info(
                        "routing: %s returned %s; retrying the same target once before moving on",
                        target.key,
                        kind,
                    )
                    continue
                return outcome, None, exc

            elapsed = time.perf_counter() - started
            outcome = Outcome(
                target_key=target.key, ok=True, latency_seconds=elapsed
            )
            await self._state.record(target.key, outcome)
            events.emit_attempt(outcome, attempt=attempt, pool=plan.pool)
            return outcome, value, None

        # Unreachable: the loop either returns or continues.
        return (
            Outcome(target_key=target.key, ok=False, failure=FailureKind.UNKNOWN),
            None,
            last_error,
        )

    def _merge_params(
        self, target: Target, call_params: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        """Per-call parameters win over everything the plan decided."""
        params = dict(target.params)
        params.update({k: v for k, v in (call_params or {}).items() if v is not None})
        return params

    def _required_window(
        self, error: BaseException | None, request: RoutingRequest, target: Target
    ) -> int:
        """How big a context window the next target needs.

        Prefers the number the provider itself reported, because an estimate
        that is too small would route to another target that also cannot hold
        the prompt — turning one clear error into several.
        """
        if isinstance(error, ContextLengthError):
            actual = getattr(error, "actual", None)
            if actual:
                return int(actual)
            limit = getattr(error, "limit", None)
            if limit:
                return int(limit) + 1
        current = context_window(target, provider_type=self._provider_type_for(target))
        if current:
            return current + 1
        return max(request.approx_tokens, 1)

    # ------------------------------------------------------------------
    # Cascade
    # ------------------------------------------------------------------

    async def _cascade(
        self,
        request: RoutingRequest,
        result: RoutingResult,
        runner: Runner,
        *,
        settings: RoutingSettings,
        name: str | None,
    ) -> RoutingResult:
        """Verify the answer and escalate only if it fell short."""
        verifier = self.verifier(settings)
        if verifier is None:
            return result
        config = self._cascades.get((name or "default").strip().lower()) or {}
        rungs = [str(rung) for rung in (config.get("rungs") or [])]
        if not rungs:
            return result

        text = _as_text(result.value)
        if text is None:
            logger.debug("Cascade cannot verify a %s result; keeping it.", type(result.value))
            return result

        verdict = await verifier.verify(request, text)
        result.verdict = verdict
        if verdict.sufficient is True:
            return result
        if verdict.sufficient is None:
            if settings.on_unknown_verdict is UnknownVerdictPolicy.ACCEPT:
                logger.debug(
                    "Cascade verifier could not judge the answer (%s); accepting it per "
                    "routing.cascade.on_unknown.",
                    verdict.rationale,
                )
                return result
            logger.info(
                "Cascade verifier could not judge the answer; escalating per "
                "routing.cascade.on_unknown = escalate."
            )

        # Escalate. The rung we were already served by is skipped, so a cascade
        # whose first rung is also the default pool does not pay twice for the
        # same answer.
        max_rungs = max(1, min(settings.cascade_max_rungs, len(rungs)))
        used = result.rungs_used
        for rung in rungs:
            if used >= max_rungs + 1:
                break
            try:
                escalated = await self.execute(
                    request,
                    runner,
                    settings=replace(settings, cascade_enabled=False),
                    **_rung_kwargs(rung),
                )
            except Exception as exc:
                logger.warning("Cascade rung %r failed (%s); keeping the previous answer.", rung, exc)
                continue
            if escalated.target.key == result.target.key:
                continue
            used += 1
            escalated.attempts = tuple(result.attempts) + tuple(escalated.attempts)
            escalated.rungs_used = used
            text = _as_text(escalated.value)
            if text is None:
                return escalated
            escalated.verdict = await verifier.verify(request, text)
            if escalated.verdict.sufficient is not False:
                return escalated
            result = escalated
        return result

    # ------------------------------------------------------------------
    # Balances
    # ------------------------------------------------------------------

    async def probe_balances(self) -> dict[str, dict[str, Any]]:
        """Ask every reachable target's provider for its remaining balance.

        Only a handful of vendors expose one (spec §3.4), so most entries come
        back ``None``. That is reported as *unknown* rather than omitted,
        because "we asked and nobody knows" and "we did not ask" are different
        facts and the distinction is the whole point of the
        :class:`~llmcore.routing.protocols.BalanceProbe` protocol.
        """
        out: dict[str, dict[str, Any]] = {}
        seen: set[str] = set()
        for pool in self._pools.values():
            for target in pool.targets:
                if target.key in seen:
                    continue
                seen.add(target.key)
                entry: dict[str, Any] = {"known": False, "amount": None, "unit": None}
                try:
                    provider = self._providers.resolve_target(
                        target, autoprovision=False, cache=False
                    )
                except Exception as exc:
                    entry["error"] = str(exc)[:200]
                    out[target.key] = entry
                    continue
                probe = getattr(provider, "remaining_balance", None)
                if probe is None:
                    entry["error"] = f"{provider.get_name()} exposes no balance endpoint"
                    out[target.key] = entry
                    continue
                try:
                    balance = await probe()
                except Exception as exc:
                    entry["error"] = str(exc)[:200]
                    out[target.key] = entry
                    continue
                if balance is not None:
                    await self._state.record_balance(target.key, balance)
                    entry = {
                        "known": balance.is_known,
                        "amount": balance.amount,
                        "unit": balance.unit,
                    }
                out[target.key] = entry
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rung_kwargs(rung: str) -> dict[str, Any]:
    """Interpret a cascade rung as ``pool:``, ``lane:`` or a target spec."""
    text = rung.strip()
    lowered = text.lower()
    if lowered.startswith("pool:"):
        return {"pool": text[5:].strip()}
    if lowered.startswith("lane:"):
        return {"lane": text[5:].strip()}
    return {"target": text}


def _as_text(value: Any) -> str | None:
    """Pull judgeable text out of whatever the runner returned."""
    if isinstance(value, str):
        return value
    for attribute in ("content", "text", "message"):
        found = getattr(value, attribute, None)
        if isinstance(found, str):
            return found
        if found is not None:
            nested = getattr(found, "content", None)
            if isinstance(nested, str):
                return nested
    return None


def _read(get: Any, key: str, default: Any) -> Any:
    if get is None:
        return default
    try:
        return get(key, default)
    except Exception:  # pragma: no cover - defensive against odd config objects
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")
