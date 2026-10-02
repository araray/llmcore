# src/llmcore/routing/settings.py
"""Resolved routing settings, and the layering that produces them.

**Config is a warm-up, not a cage.** It exists so a caller does not have to
re-specify everything on every call — it is paramount as a starting point, and
it does not freeze the runtime. The moment something needs to change, it is
passed as an API parameter or an environment variable.

Every knob here therefore resolves through the same three layers, lowest
precedence first:

1. ``default_config.toml`` / the user's config file — the starting point;
2. **environment** — confy already maps ``LLMCORE_ROUTING__<KEY>`` onto
   ``routing.<key>``, so anything in config is env-overridable without this
   module doing anything special;
3. **the request** — a per-call override, which always wins.

That ordering is why policy lives here rather than in
:mod:`llmcore.routing.models`: a :class:`~llmcore.routing.models.FailureKind`
describes what happened and cannot change, while what routing *does* about it
differs per deployment and per call. Nothing in this module may be read
directly from config at call time — resolve once, then override — so that a
per-request change cannot be silently ignored by code that went back to the
config file.

:class:`RoutingSettings` is therefore immutable and cheap to copy — a request
that overrides one knob gets a whole resolved snapshot rather than a
config-plus-patches object that later code has to re-resolve.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from .models import DEFAULT_COOLDOWNS, FailureKind, SelectionStrategy

__all__ = ["ClassifierBias", "RefusalPolicy", "RoutingSettings", "UnknownVerdictPolicy"]

from enum import StrEnum


class RefusalPolicy(StrEnum):
    """What to do when a provider refuses on content grounds.

    Attributes:
        FAIL: Surface the refusal. The shipped default — retrying elsewhere is
            "shop until someone says yes".
        FAILOVER: Try the next target in the pool.
    """

    FAIL = "fail"
    FAILOVER = "failover"


class UnknownVerdictPolicy(StrEnum):
    """What a cascade does when its verifier cannot judge an answer.

    Attributes:
        ACCEPT: Keep the cheap answer. The default — escalating on every
            unjudgeable answer inverts the saving the cascade exists for.
        ESCALATE: Move to the next rung.
    """

    ACCEPT = "accept"
    ESCALATE = "escalate"


class ClassifierBias(StrEnum):
    """Which way to lean when a classifier is unsure.

    The two error directions are not symmetric: routing too cheap produces a
    bad answer, routing too expensive only costs money.

    Attributes:
        QUALITY: Prefer the stronger lane. The default.
        COST: Prefer the cheaper lane.
        NEUTRAL: Take the classifier at its word.
    """

    QUALITY = "quality"
    COST = "cost"
    NEUTRAL = "neutral"


@dataclass(frozen=True, slots=True)
class RoutingSettings:
    """A fully resolved snapshot of routing policy.

    Attributes:
        enabled: Whether pools/lanes are consulted at all.
        autoprovision: Build an instance for an unconfigured provider when a
            credential is discoverable. ``False`` restores an allow-list.
        autoprovision_cache: Keep autoprovisioned instances for the process.
        default_pool: Pool used when none is named.
        default_strategy: Strategy for a pool that declares none.
        default_lane: Lane used when a classifier has no opinion.
        max_attempts: Distinct targets tried per request.
        retry_same_attempts: Extra tries against the *same* target for failures
            where that is sensible (5xx, connection).
        on_refusal: See :class:`RefusalPolicy`.
        on_unsupported_param: ``"drop"`` or ``"error"``. ``drop`` is the default
            because a pool whose members have different parameter surfaces is
            the normal case, and erroring would make such pools unusable.
        reroute_on_context_overflow: Treat a context-length failure as a routing
            signal and look for a larger window.
        affinity: ``"session"`` pins a conversation to its first target.
        cooldowns: Per-failure cooldown seconds; ``None`` means for the process.
        cascade_enabled: Off by default; it trades latency and an extra call.
        cascade_max_rungs: Cap on escalation steps.
        cascade_threshold: Verifier score below which a rung escalates.
        on_unknown_verdict: See :class:`UnknownVerdictPolicy`.
        classifier_chain: Classifier names, evaluated cheapest-first.
        min_confidence: Floor below which a classification is ignored.
        bias: See :class:`ClassifierBias`.
        magic_pattern: Regex for an in-prompt routing directive, or ``None``
            for the built-in
            :data:`~llmcore.routing.classifiers.free.DEFAULT_MAGIC_PATTERN`.
            A custom pattern must provide named groups ``key`` and ``value``
            (``[[lane:deep]]`` → key ``lane``, value ``deep``); one without
            them is rejected at build time with a message saying so, rather
            than raising from inside the classifier on a live request.
            Matches are **stripped before egress** either way.
        transforms: Transform names to apply before a call.
        explain_only: Plan without calling — used by ``routing.explain()``.
    """

    enabled: bool = True
    autoprovision: bool = True
    autoprovision_cache: bool = True

    default_pool: str | None = None
    default_strategy: SelectionStrategy = SelectionStrategy.PRIORITY
    default_lane: str | None = None

    max_attempts: int = 3
    retry_same_attempts: int = 1
    on_refusal: RefusalPolicy = RefusalPolicy.FAIL
    on_unsupported_param: str = "drop"
    reroute_on_context_overflow: bool = True
    affinity: str = "session"
    cooldowns: dict[FailureKind, float | None] = field(
        default_factory=lambda: dict(DEFAULT_COOLDOWNS)
    )

    cascade_enabled: bool = False
    cascade_max_rungs: int = 2
    cascade_threshold: float = 0.7
    on_unknown_verdict: UnknownVerdictPolicy = UnknownVerdictPolicy.ACCEPT

    classifier_chain: tuple[str, ...] = ()
    min_confidence: float = 0.55
    bias: ClassifierBias = ClassifierBias.QUALITY
    magic_pattern: str | None = None

    transforms: tuple[str, ...] = ()
    explain_only: bool = False

    # ------------------------------------------------------------------
    # Layer 1 + 2: config, with env already folded in by confy
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, get: Callable[[str, Any], Any] | None = None) -> RoutingSettings:
        """Build settings from an llmcore config accessor.

        *get* is a ``config.get(key, default)`` callable. Environment overrides
        need no handling here: confy resolves ``LLMCORE_ROUTING__MAX_ATTEMPTS``
        onto ``routing.max_attempts`` before this sees it, so config and env are
        already one layer by the time we read a key.
        """
        read = get or (lambda _k, d=None: d)
        defaults = cls()

        cooldowns = dict(DEFAULT_COOLDOWNS)
        configured = read("routing.cooldowns", None) or {}
        for name, seconds in dict(configured).items():
            try:
                kind = FailureKind(str(name))
            except ValueError:
                continue
            cooldowns[kind] = None if seconds is None else float(seconds)

        return cls(
            enabled=_as_bool(read("routing.enabled", defaults.enabled)),
            autoprovision=_as_bool(read("routing.autoprovision", defaults.autoprovision)),
            autoprovision_cache=_as_bool(
                read("routing.autoprovision_cache", defaults.autoprovision_cache)
            ),
            default_pool=read("routing.default_pool", defaults.default_pool) or None,
            default_strategy=_as_enum(
                SelectionStrategy,
                read("routing.default_strategy", defaults.default_strategy),
                defaults.default_strategy,
            ),
            default_lane=read("routing.default_lane", defaults.default_lane) or None,
            max_attempts=_as_int(read("routing.max_attempts", defaults.max_attempts), 1),
            retry_same_attempts=_as_int(
                read("routing.retry_same_attempts", defaults.retry_same_attempts), 0
            ),
            on_refusal=_as_enum(
                RefusalPolicy, read("routing.on_refusal", defaults.on_refusal), defaults.on_refusal
            ),
            on_unsupported_param=str(
                read("routing.on_unsupported_param", defaults.on_unsupported_param)
            ).lower(),
            reroute_on_context_overflow=_as_bool(
                read(
                    "routing.reroute_on_context_overflow",
                    defaults.reroute_on_context_overflow,
                )
            ),
            affinity=str(read("routing.affinity", defaults.affinity)).lower(),
            cooldowns=cooldowns,
            cascade_enabled=_as_bool(
                read("routing.cascade.enabled", defaults.cascade_enabled)
            ),
            cascade_max_rungs=_as_int(
                read("routing.cascade.max_rungs", defaults.cascade_max_rungs), 1
            ),
            cascade_threshold=_as_float(
                read("routing.cascade.threshold", defaults.cascade_threshold)
            ),
            on_unknown_verdict=_as_enum(
                UnknownVerdictPolicy,
                read("routing.cascade.on_unknown", defaults.on_unknown_verdict),
                defaults.on_unknown_verdict,
            ),
            classifier_chain=tuple(read("routing.classifier.chain", ()) or ()),
            min_confidence=_as_float(
                read("routing.classifier.min_confidence", defaults.min_confidence)
            ),
            bias=_as_enum(
                ClassifierBias, read("routing.classifier.bias", defaults.bias), defaults.bias
            ),
            magic_pattern=read("routing.classifier.magic_pattern", defaults.magic_pattern)
            or None,
            transforms=tuple(read("routing.transforms.chain", ()) or ()),
        )

    # ------------------------------------------------------------------
    # Layer 3: the request
    # ------------------------------------------------------------------

    def override(self, **overrides: Any) -> RoutingSettings:
        """Return a copy with per-request *overrides* applied.

        ``None`` values are ignored so a caller can pass every keyword through
        from a signature full of optional arguments without having to filter
        them, which is how ``chat()`` forwards its routing kwargs.

        Unknown keys raise rather than being dropped: a misspelled override
        that silently did nothing would be a routing bug nobody could see.

        Raises:
            TypeError: If an override names a setting that does not exist.
        """
        clean: dict[str, Any] = {}
        fields = set(self.__dataclass_fields__)
        for key, value in overrides.items():
            if value is None:
                continue
            if key not in fields:
                raise TypeError(
                    f"Unknown routing override {key!r}. Valid overrides: "
                    f"{', '.join(sorted(fields))}."
                )
            clean[key] = _coerce_field(key, value, getattr(self, key))
        return replace(self, **clean) if clean else self

    def cooldown_for(self, kind: FailureKind) -> float | None:
        """Cooldown seconds for *kind*, honouring configuration."""
        return self.cooldowns.get(kind, DEFAULT_COOLDOWNS.get(kind, 10.0))

    def should_failover(self, kind: FailureKind) -> bool:
        """Whether *kind* should move to another target, under this policy."""
        if kind.failover_is_pointless:
            return False
        if kind is FailureKind.REFUSAL:
            return self.on_refusal is RefusalPolicy.FAILOVER
        return True


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _as_int(value: Any, minimum: int) -> int:
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return minimum


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_enum(enum_cls: type, value: Any, fallback: Any) -> Any:
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(str(value).strip().lower())
    except ValueError:
        return fallback


def _coerce_field(key: str, value: Any, current: Any) -> Any:
    """Coerce a per-request override to the field's type.

    Overrides arrive from CLI flags, env strings and the bridge as text, so
    ``max_attempts="5"`` and ``cascade_enabled="true"`` have to work.
    """
    if isinstance(current, bool):
        return _as_bool(value)
    if isinstance(current, SelectionStrategy):
        return _as_enum(SelectionStrategy, value, current)
    if isinstance(current, RefusalPolicy):
        return _as_enum(RefusalPolicy, value, current)
    if isinstance(current, UnknownVerdictPolicy):
        return _as_enum(UnknownVerdictPolicy, value, current)
    if isinstance(current, ClassifierBias):
        return _as_enum(ClassifierBias, value, current)
    if isinstance(current, int) and not isinstance(current, bool):
        return _as_int(value, 0)
    if isinstance(current, float):
        return _as_float(value)
    if isinstance(current, tuple):
        if isinstance(value, str):
            return tuple(v.strip() for v in value.split(",") if v.strip())
        return tuple(value)
    return value
