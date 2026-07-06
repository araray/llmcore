# src/llmcore/moderation/models.py
"""
Value objects for the moderation gateway (plan SF-1 / DDS-06).

This module defines the small, immutable records that flow through the
moderation pipeline:

* :class:`ModerationAction` — what a policy wants done with a piece of text.
* :class:`ModerationResult` — the raw verdict a gateway returns for one text.
* :class:`ModerationDecision` — the policy-evaluated outcome that callers
  (e.g. wairu's ``ToolDispatcher``) act on.

Gateways report *facts* (scores, provider flags); policies turn facts into
*decisions*. Keeping the two separated lets one gateway serve many policies
(per-session opt-in, different thresholds per surface, ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "ModerationAction",
    "ModerationDecision",
    "ModerationResult",
]


class ModerationAction(str, Enum):
    """Action a moderation policy can take on a piece of text.

    Attributes:
        ALLOW: Content passes; proceed normally.
        WARN: Content is suspicious; proceed but surface a warning.
        BLOCK: Content must not proceed.
    """

    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"


@dataclass(frozen=True)
class ModerationResult:
    """Raw verdict a :class:`~llmcore.moderation.gateway.ModerationGateway` returns.

    Attributes:
        flagged: Provider-level overall verdict (the provider's own
            thresholds tripped for at least one category).
        categories: Per-category confidence scores in ``[0.0, 1.0]``.
            Category names are normalized to snake_case (e.g.
            ``"self_harm_intent"``, ``"harassment_threatening"``).
        flagged_categories: Categories the *provider* asserted as violations
            (independent of any local policy thresholds).
        action_hint: Optional gateway-suggested action. Policies may use it
            as a tie-breaker; it never overrides policy thresholds.
        provider: Name of the gateway that produced this result.
        model: Moderation model identifier, when known.
    """

    flagged: bool
    categories: dict[str, float] = field(default_factory=dict)
    flagged_categories: tuple[str, ...] = ()
    action_hint: ModerationAction | None = None
    provider: str = ""
    model: str = ""


@dataclass(frozen=True)
class ModerationDecision:
    """Policy-evaluated moderation outcome.

    Attributes:
        allowed: ``True`` when the content may proceed (``ALLOW`` or
            ``WARN``); ``False`` when it must be stopped (``BLOCK``).
        action: The concrete action the policy selected.
        reason: Human-readable explanation (category list, fail-safe note,
            ...); suitable for logs and user-facing errors.
        categories: Per-category scores copied from the gateway result
            (empty when the gateway errored or returned no scores).
        triggered_categories: Categories that tripped the policy (its own
            thresholds and/or provider flags, per policy configuration).
        context: Caller-supplied surface tag (e.g. ``"input"``,
            ``"output"``, ``"tool"``), echoed back for observability.
        fail_safe: ``True`` when this decision came from the fail-safe
            path (gateway exception while moderation was enabled) rather
            than from an actual gateway verdict.
    """

    allowed: bool
    action: ModerationAction
    reason: str
    categories: dict[str, float] = field(default_factory=dict)
    triggered_categories: tuple[str, ...] = ()
    context: str = ""
    fail_safe: bool = False
