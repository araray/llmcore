# src/llmcore/routing/models.py
"""Core types for the routing subsystem (see ``docs/ROUTING_SUBSYSTEM_SPEC.md``).

Five layers compose here, and the types are deliberately separate per layer so
each is usable without the others:

* :class:`Target` — which provider, model and parameters.
* :class:`Pool` state — :class:`TargetHealth`, :class:`Outcome`, :class:`FailureKind`.
* :class:`Classification` — what kind of request this is.
* :class:`Verdict` — whether a cheap answer was good enough.
* :class:`TransformResult` — what must not leave this machine.

:class:`RoutingPlan` ties them together for ``llm.routing.explain()``, because
opaque routing is a support burden: the first question anyone asks is "why did
it pick that?", and answering it must not require reading logs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Any
from urllib.parse import parse_qsl, urlencode

__all__ = [
    "Balance",
    "Classification",
    "FailureKind",
    "Finding",
    "Outcome",
    "RoutingPlan",
    "RoutingRequest",
    "SelectionStrategy",
    "Target",
    "TargetHealth",
    "TransformAction",
    "TransformResult",
    "Verdict",
    "classify_failure",
]


# ---------------------------------------------------------------------------
# Layer 1 — Target
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Target:
    """One routable destination: a provider, a model, and parameters.

    Addressable as a **spec string** so a target works in config, in a CLI flag,
    in an environment variable and over the bridge — anywhere a dataclass
    cannot go::

        anthropic:claude-opus-5-5?effort=high
        vllm:Qwen/Qwen3-30B#my-box
        ollama:llama3.3:70b

    Attributes:
        provider: A ``PROVIDER_MAP`` type or a configured instance name.
        model: Model id; ``None`` uses the provider's ``default_model``.
        params: Per-call parameters (``effort``, ``temperature``, ...).
        instance: Pin a specific configured instance.
        label: Optional friendly name for metrics and logs.
    """

    provider: str
    model: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    instance: str | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("Target.provider is required.")

    @property
    def key(self) -> str:
        """Stable identity for health tracking and metrics.

        Parameters are **excluded**: the same model behind the same credential
        shares a rate limit and a balance whether or not a caller asked for more
        reasoning effort, so health must be tracked per endpoint rather than per
        parameter combination.
        """
        base = f"{self.instance or self.provider}:{self.model or ''}"
        return base.rstrip(":")

    def spec(self) -> str:
        """Render this target as a spec string."""
        out = f"{self.provider}:{self.model}" if self.model else self.provider
        if self.instance:
            out += f"#{self.instance}"
        if self.params:
            out += "?" + urlencode(
                {k: _unparse_scalar(v) for k, v in sorted(self.params.items())}
            )
        return out

    def with_params(self, **params: Any) -> Target:
        """Return a copy with *params* merged over the existing ones."""
        merged = {**self.params, **{k: v for k, v in params.items() if v is not None}}
        return replace(self, params=merged)

    # --- parsing -------------------------------------------------------

    #: ``provider ":" [model] ["#" instance] ["?" params]``
    _SPEC = re.compile(
        r"^(?P<provider>[^:?#]+)"
        r"(?::(?P<model>[^?#]*))?"
        r"(?:#(?P<instance>[^?]*))?"
        r"(?:\?(?P<params>.*))?$"
    )

    @classmethod
    def parse(cls, spec: str | Target) -> Target:
        """Parse a spec string into a :class:`Target`.

        The provider is everything before the **first** colon and the model is
        the remainder, because model names routinely contain ``:`` and ``/``
        (``ollama:llama3.3:70b``, ``vllm:Qwen/Qwen3-30B``). Splitting on the
        last colon would silently mangle both.

        Raises:
            ValueError: If *spec* is empty or has no provider.
        """
        if isinstance(spec, Target):
            return spec
        text = (spec or "").strip()
        if not text:
            raise ValueError("A target spec cannot be empty.")
        match = cls._SPEC.match(text)
        if match is None:  # pragma: no cover - the regex accepts any non-empty head
            raise ValueError(f"Could not parse target spec {spec!r}.")

        provider = (match.group("provider") or "").strip()
        if not provider:
            raise ValueError(f"Target spec {spec!r} has no provider.")
        model = (match.group("model") or "").strip() or None
        instance = (match.group("instance") or "").strip() or None
        raw_params = match.group("params") or ""
        params = {k: _parse_scalar(v) for k, v in parse_qsl(raw_params, keep_blank_values=True)}
        return cls(provider=provider, model=model, params=params, instance=instance)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.label or self.spec()


def _parse_scalar(value: str) -> Any:
    """Coerce a query-string value to bool/int/float where unambiguous.

    Left as a string otherwise: provider parameters are vendor-defined, and
    guessing a type for something like a model name or a stop sequence would be
    worse than passing the caller's text through.
    """
    lowered = value.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("none", "null"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _unparse_scalar(value: Any) -> str:
    """Render a parameter value for a spec string."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "none"
    return str(value)


# ---------------------------------------------------------------------------
# Layer 2 — Pool health
# ---------------------------------------------------------------------------


class SelectionStrategy(StrEnum):
    """How a pool chooses among healthy targets.

    Attributes:
        PRIORITY: First healthy target in declared order.
        ROUND_ROBIN: Next healthy target, cycling.
        WEIGHTED: Weighted random over healthy targets.
        LOWEST_LATENCY: Lowest observed EWMA latency.
        LOWEST_COST: Cheapest for the estimated token spend, from model cards.
        LEAST_BUSY: Fewest in-flight requests.
        MOST_CREDITS: Most remaining balance, where the vendor reports one.
    """

    PRIORITY = "priority"
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    LOWEST_LATENCY = "lowest_latency"
    LOWEST_COST = "lowest_cost"
    LEAST_BUSY = "least_busy"
    MOST_CREDITS = "most_credits"


class FailureKind(StrEnum):
    """Why a call failed, which decides what routing does next.

    The distinctions matter more than the names. Treating a rate limit, an empty
    wallet and an over-long prompt alike is how naive failover either loops or
    burns money:

    Attributes:
        RATE_LIMIT: 429. Cool down briefly, honour ``Retry-After``, try a peer.
        INSUFFICIENT_CREDIT: 402/403-with-billing. Cool down for **minutes** —
            it will not fix itself in seconds — and treat the balance as zero.
        TIMEOUT: Cool down briefly and try a peer.
        SERVER: 5xx or connection. Retry the *same* target once, then move.
        AUTH: 401/403. Mark unusable for the process; a bad key does not heal.
        CONTEXT_LENGTH: The prompt overflowed. **Not** a health problem — route
            to a larger window instead.
        BAD_REQUEST: 400. Fails everywhere; do not fail over.
        REFUSAL: Content policy. Whether to fail over is configurable
            (``routing.on_refusal``) and overridable per request; the shipped
            default is not to, because retrying elsewhere is "shop until
            someone says yes" — but that is a deployment's call, not ours.
        UNKNOWN: Unclassified; treated conservatively like ``SERVER``.
    """

    RATE_LIMIT = "rate_limit"
    INSUFFICIENT_CREDIT = "insufficient_credit"
    TIMEOUT = "timeout"
    SERVER = "server"
    AUTH = "auth"
    CONTEXT_LENGTH = "context_length"
    BAD_REQUEST = "bad_request"
    REFUSAL = "refusal"
    UNKNOWN = "unknown"

    @property
    def failover_is_pointless(self) -> bool:
        """Whether another target would certainly fail the same way.

        This is a *fact about the failure*, not a policy: a malformed request is
        malformed everywhere. Whether to fail over on the cases that are merely
        *questionable* — a content refusal, say — is a policy decision and lives
        in :class:`~llmcore.routing.settings.RoutingSettings`, because the
        answer differs per deployment and must be overridable per request.
        """
        return self is FailureKind.BAD_REQUEST

    @property
    def affects_health(self) -> bool:
        """Whether this failure should cool the target down.

        ``CONTEXT_LENGTH``, ``BAD_REQUEST`` and ``REFUSAL`` are properties of
        the *request*, not of the target, so penalising the target for them
        would take a healthy endpoint out of rotation for someone else's
        oversized prompt or disallowed question.
        """
        return self not in (
            FailureKind.CONTEXT_LENGTH,
            FailureKind.BAD_REQUEST,
            FailureKind.REFUSAL,
        )

    @property
    def retry_same(self) -> bool:
        """Whether retrying the same target once is sensible first."""
        return self in (FailureKind.SERVER, FailureKind.UNKNOWN)


#: Default cooldown per failure kind, in seconds. ``None`` means "for the life
#: of the process" (a credential does not fix itself), ``0`` means no cooldown.
DEFAULT_COOLDOWNS: dict[FailureKind, float | None] = {
    FailureKind.RATE_LIMIT: 20.0,
    # Minutes, not seconds: a topped-up wallet is a human action.
    FailureKind.INSUFFICIENT_CREDIT: 600.0,
    FailureKind.TIMEOUT: 15.0,
    FailureKind.SERVER: 10.0,
    FailureKind.AUTH: None,
    FailureKind.CONTEXT_LENGTH: 0.0,
    FailureKind.BAD_REQUEST: 0.0,
    FailureKind.REFUSAL: 0.0,
    FailureKind.UNKNOWN: 10.0,
}


@dataclass(frozen=True, slots=True)
class Outcome:
    """The result of one attempt against one target."""

    target_key: str
    ok: bool
    latency_seconds: float | None = None
    failure: FailureKind | None = None
    retry_after_seconds: float | None = None
    cost_usd: float | None = None
    error: str | None = None
    at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class TargetHealth:
    """Rolling health for one target.

    Mutable because it is updated on every attempt. ``unusable`` is separate
    from ``cooldown_until`` so an authentication failure is permanent for the
    process rather than expiring quietly and failing again.
    """

    target_key: str
    consecutive_failures: int = 0
    total_failures: int = 0
    total_successes: int = 0
    cooldown_until: datetime | None = None
    unusable: bool = False
    ewma_latency_seconds: float | None = None
    in_flight: int = 0
    last_failure: FailureKind | None = None
    last_error: str | None = None
    balance: Balance | None = None

    #: Smoothing for the latency average. 0.3 reacts within a few requests
    #: without letting one slow call evict a target that is usually fast.
    _ALPHA = 0.3

    def is_available(self, now: datetime | None = None) -> bool:
        """Whether this target may be selected."""
        if self.unusable:
            return False
        if self.cooldown_until is None:
            return True
        return (now or datetime.now(timezone.utc)) >= self.cooldown_until

    def cooldown_remaining(self, now: datetime | None = None) -> float:
        """Seconds until this target is available again; 0 when it already is."""
        if self.unusable:
            return float("inf")
        if self.cooldown_until is None:
            return 0.0
        delta = (self.cooldown_until - (now or datetime.now(timezone.utc))).total_seconds()
        return max(0.0, delta)

    def record(
        self,
        outcome: Outcome,
        *,
        cooldowns: dict[FailureKind, float | None] | None = None,
        now: datetime | None = None,
    ) -> None:
        """Fold *outcome* into this health record."""
        moment = now or datetime.now(timezone.utc)
        table = cooldowns or DEFAULT_COOLDOWNS

        if outcome.latency_seconds is not None:
            if self.ewma_latency_seconds is None:
                self.ewma_latency_seconds = outcome.latency_seconds
            else:
                self.ewma_latency_seconds = (
                    self._ALPHA * outcome.latency_seconds
                    + (1 - self._ALPHA) * self.ewma_latency_seconds
                )

        if outcome.ok:
            self.total_successes += 1
            self.consecutive_failures = 0
            # A success clears a cooldown: whatever was wrong has passed, and
            # keeping the target benched would waste a working endpoint.
            self.cooldown_until = None
            self.last_failure = None
            self.last_error = None
            return

        self.total_failures += 1
        self.consecutive_failures += 1
        self.last_failure = outcome.failure
        self.last_error = outcome.error
        kind = outcome.failure or FailureKind.UNKNOWN

        if kind is FailureKind.AUTH:
            self.unusable = True
            return
        if kind is FailureKind.INSUFFICIENT_CREDIT:
            # The vendor just told us the wallet is empty; that is better
            # evidence than any balance endpoint.
            self.balance = Balance(amount=0.0, unit="usd", as_of=moment)

        if not kind.affects_health:
            return

        seconds = outcome.retry_after_seconds
        if seconds is None:
            seconds = table.get(kind, 10.0)
        if seconds:
            self.cooldown_until = moment + timedelta(seconds=float(seconds))


@dataclass(frozen=True, slots=True)
class Balance:
    """Remaining balance for a provider, where it reports one.

    ``None`` from a probe means **unknown**, which is deliberately distinct from
    zero: most vendors publish no balance endpoint at all, and ranking an
    unknown target as broke would quietly demote perfectly good providers.
    """

    amount: float | None
    unit: str = "usd"
    currency: str | None = None
    as_of: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_known(self) -> bool:
        """Whether an amount was actually reported."""
        return self.amount is not None


# ---------------------------------------------------------------------------
# Layer 3 — Classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RoutingRequest:
    """What a classifier, verifier or transform gets to look at.

    Deliberately a copy rather than the live message list, so a classifier
    cannot mutate the request it is being asked about.

    ``prompt`` is optional because ``chat(messages=[...])`` is a legitimate
    call shape with no prompt at all, and a routing request that could not
    represent it would simply not work for half of llmcore's callers.
    """

    prompt: str | None = None
    messages: tuple[dict[str, Any], ...] = ()
    system: str | None = None
    session_id: str | None = None
    hints: dict[str, Any] = field(default_factory=dict)
    tools: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def approx_tokens(self) -> int:
        """A cheap character-based estimate, for heuristics only.

        Not a tokenizer: heuristic classifiers need an order of magnitude, and
        loading a tokenizer to decide whether to load a model would defeat the
        purpose.
        """
        total = len(self.prompt or "") + len(self.system or "")
        for message in self.messages:
            content = message.get("content")
            if isinstance(content, str):
                total += len(content)
        return total // 4


@dataclass(frozen=True, slots=True)
class Classification:
    """A classifier's opinion about a request.

    All fields are optional because classifiers disagree about what they can
    say: a prompt router names a lane, a difficulty scorer names an effort, and
    a preference router may name an outright target.

    Returning ``None`` from a classifier instead of an empty
    :class:`Classification` is how a chain stays composable — it means *no
    opinion*, so the next classifier gets a turn.
    """

    lane: str | None = None
    effort: str | None = None
    target: Target | None = None
    confidence: float | None = None
    rationale: str | None = None
    source: str = ""
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """Whether this carries no routing information at all."""
        return self.lane is None and self.effort is None and self.target is None


# ---------------------------------------------------------------------------
# Layer 4 — Cascade
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Verdict:
    """Whether a response was good enough to accept.

    ``sufficient=None`` means *could not judge*, which must not silently count
    as either pass or fail — escalating on every unjudgeable answer inverts the
    cost saving the cascade exists for, and accepting silently defeats it.
    """

    sufficient: bool | None
    score: float | None = None
    rationale: str | None = None
    source: str = ""


# ---------------------------------------------------------------------------
# Layer 5 — Transforms
# ---------------------------------------------------------------------------


class TransformAction(StrEnum):
    """What a transform decided.

    Attributes:
        ALLOW: Nothing of concern; send as-is.
        REDACT: Send the rewritten text.
        CONSTRAIN: Restrict routing to a safer pool — the actual guarantee.
        BLOCK: Refuse to send at all.
    """

    ALLOW = "allow"
    REDACT = "redact"
    CONSTRAIN = "constrain"
    BLOCK = "block"


@dataclass(frozen=True, slots=True)
class Finding:
    """One thing a transform detected.

    Carries a **hash** of the matched text, never the text: a privacy feature
    whose audit log contains the identifiers it found would be the leak it
    exists to prevent.
    """

    kind: str
    where: str
    hashed: str
    start: int | None = None
    end: int | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class TransformResult:
    """The outcome of applying a transform."""

    action: TransformAction = TransformAction.ALLOW
    prompt: str | None = None
    messages: tuple[dict[str, Any], ...] | None = None
    system: str | None = None
    findings: tuple[Finding, ...] = ()
    constrain_to_pool: str | None = None
    reason: str | None = None
    source: str = ""


# ---------------------------------------------------------------------------
# Explain
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Candidate:
    """One target considered during selection, and what became of it."""

    target: Target
    eligible: bool
    reason: str | None = None
    score: float | None = None
    cooldown_remaining: float = 0.0


@dataclass(frozen=True, slots=True)
class RoutingPlan:
    """The decision ``llm.routing.explain()`` returns.

    Exists so "why did it pick that?" is answerable without reading logs, which
    is the first question anyone asks of a router.
    """

    chosen: Target | None
    pool: str | None = None
    lane: str | None = None
    strategy: SelectionStrategy | None = None
    classification: Classification | None = None
    candidates: tuple[Candidate, ...] = ()
    transforms: tuple[TransformResult, ...] = ()
    estimated_cost_usd: float | None = None
    notes: tuple[str, ...] = ()

    def summary(self) -> str:
        """One readable line, for CLIs and logs."""
        parts = []
        if self.lane:
            parts.append(f"lane={self.lane}")
        if self.pool:
            parts.append(f"pool={self.pool}")
        if self.strategy:
            parts.append(f"strategy={self.strategy}")
        parts.append(f"chosen={self.chosen.spec() if self.chosen else 'none'}")
        if self.estimated_cost_usd is not None:
            parts.append(f"est=${self.estimated_cost_usd:.6f}")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------

#: Substrings that identify an insufficient-balance refusal. Collected from the
#: live error bodies of the providers llmcore ships, because every vendor spells
#: this differently and several use 403 rather than 402.
_CREDIT_MARKERS: tuple[str, ...] = (
    "not_enough_credits",
    "insufficient",
    "credit balance is too low",
    "paid_plan_required",
    "payment_required",
    "quota",
    "billing",
    "out of credits",
    "exceeded your current quota",
)

_REFUSAL_MARKERS: tuple[str, ...] = (
    "content_policy",
    "content policy",
    "safety",
    "nsfw",
    "refus",
)

_CONTEXT_MARKERS: tuple[str, ...] = (
    "context_length",
    "context length",
    "maximum context",
    "too many tokens",
    "reduce the length",
)


def classify_failure(exc: BaseException) -> tuple[FailureKind, float | None]:
    """Classify *exc* into a :class:`FailureKind` and an optional retry delay.

    Reads llmcore's own exception types first, then falls back to HTTP status
    and message text. The text fallback exists because providers disagree about
    status codes for the same condition — an empty wallet arrives as 402 on
    Replicate, 403 on Higgsfield and ElevenLabs, and 400 on Anthropic.

    Returns:
        ``(kind, retry_after_seconds)``; the delay is ``None`` when the provider
        did not say.
    """
    from ..exceptions import ContextLengthError, ProviderError

    if isinstance(exc, ContextLengthError):
        return FailureKind.CONTEXT_LENGTH, None

    status = getattr(exc, "status_code", None)
    text = str(exc).lower()
    retry_after = getattr(exc, "retry_after", None)
    try:
        retry_after = float(retry_after) if retry_after is not None else None
    except (TypeError, ValueError):
        retry_after = None

    if any(marker in text for marker in _CONTEXT_MARKERS):
        return FailureKind.CONTEXT_LENGTH, None

    # Billing is checked before auth: a valid key on an empty account commonly
    # answers 403, and calling that an auth failure would bench the target for
    # the whole process over a problem that a top-up fixes.
    if any(marker in text for marker in _CREDIT_MARKERS) or status == 402:
        return FailureKind.INSUFFICIENT_CREDIT, retry_after

    if status == 429:
        return FailureKind.RATE_LIMIT, retry_after
    if status in (401, 403):
        return FailureKind.AUTH, None
    if status == 400:
        return FailureKind.BAD_REQUEST, None
    if status is not None and 500 <= int(status) < 600:
        return FailureKind.SERVER, retry_after

    if isinstance(exc, TimeoutError) or "timeout" in text or "timed out" in text:
        return FailureKind.TIMEOUT, retry_after
    if any(marker in text for marker in _REFUSAL_MARKERS):
        return FailureKind.REFUSAL, None
    if "rate limit" in text or "too many requests" in text:
        return FailureKind.RATE_LIMIT, retry_after
    if "connect" in text or "connection" in text:
        return FailureKind.SERVER, retry_after

    if isinstance(exc, ProviderError) and getattr(exc, "retryable", False):
        return FailureKind.SERVER, retry_after
    return FailureKind.UNKNOWN, retry_after
