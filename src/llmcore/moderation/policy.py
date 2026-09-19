# src/llmcore/moderation/policy.py
"""
Moderation policy: category thresholds -> :class:`ModerationDecision`.

The policy is deliberately pure and synchronous: it receives the raw
:class:`~llmcore.moderation.models.ModerationResult` a gateway produced and
turns it into a decision. Two knobs matter most:

* **Per-category thresholds** — ``thresholds["violence"] = 0.8`` means the
  ``violence`` category triggers at a score of 0.8 or above *regardless* of
  the provider's own flag for that category. Categories without a local
  threshold fall back to the provider's flag.
* **Fail-safe** (``fail_safe=True``, the default) — when the gateway raises
  while moderation is enabled, :meth:`decision_for_error` returns **BLOCK**.
  A broken safety net must never silently become an open door. Setting
  ``fail_safe=False`` (fail-open) is an explicit, logged opt-out.

Every decision the policy produces is also emitted on the shared event
spine (``source="llmcore"``, ``type="moderation.decision"``) when at least
one sink is registered — near-free otherwise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .. import shared_events
from .models import ModerationAction, ModerationDecision, ModerationResult

__all__ = ["ModerationPolicy"]

logger = logging.getLogger(__name__)


@dataclass
class ModerationPolicy:
    """Turns gateway results into moderation decisions.

    Attributes:
        thresholds: Per-category score thresholds (inclusive). A category
            listed here triggers purely on its score; the provider flag for
            that category is ignored (this is how operators *raise* or
            *lower* sensitivity per category).
        default_threshold: Optional threshold applied to categories not
            listed in ``thresholds``. When ``None`` (default), unlisted
            categories defer to the provider's own flag.
        default_action: What to do when content triggers — ``BLOCK``
            (default) or ``WARN``. ``WARN`` decisions are still *allowed*.
        fail_safe: When ``True`` (default), a gateway exception yields a
            BLOCK decision. When ``False``, the error is logged and the
            content is allowed (fail-open).
    """

    thresholds: dict[str, float] = field(default_factory=dict)
    default_threshold: float | None = None
    default_action: ModerationAction = ModerationAction.BLOCK
    fail_safe: bool = True

    def __post_init__(self) -> None:
        # Accept plain strings from config ("block"/"warn"/"allow").
        if not isinstance(self.default_action, ModerationAction):
            self.default_action = ModerationAction(str(self.default_action).lower())

    def evaluate(self, result: ModerationResult, *, context: str = "") -> ModerationDecision:
        """Evaluate a gateway result under this policy.

        Args:
            result: The raw gateway verdict.
            context: Surface tag echoed into the decision and event.

        Returns:
            The resulting decision (also emitted on the event spine).
        """
        triggered: list[str] = []
        for category, score in result.categories.items():
            threshold = self.thresholds.get(category, self.default_threshold)
            if threshold is not None:
                if score >= threshold:
                    triggered.append(category)
            elif category in result.flagged_categories:
                triggered.append(category)

        # Provider-asserted categories with no score entry still count when
        # they have no local threshold overriding them.
        for category in result.flagged_categories:
            if category not in result.categories and category not in triggered:
                threshold = self.thresholds.get(category, self.default_threshold)
                if threshold is None:
                    triggered.append(category)

        if not triggered and result.flagged and not result.categories:
            # Provider says "flagged" but gave us nothing per-category to
            # re-score against local thresholds: trust the provider.
            triggered.append("provider_flagged")

        if triggered:
            action = self.default_action
            reason = "moderation triggered: " + ", ".join(sorted(triggered))
        else:
            action = ModerationAction.ALLOW
            reason = "no category triggered"

        decision = ModerationDecision(
            allowed=action is not ModerationAction.BLOCK,
            action=action,
            reason=reason,
            categories=dict(result.categories),
            triggered_categories=tuple(sorted(triggered)),
            context=context,
        )
        self._emit(decision, provider=result.provider, model=result.model)
        return decision

    def decision_for_error(self, error: BaseException, *, context: str = "") -> ModerationDecision:
        """Build the decision for a gateway exception (the fail-safe seam).

        Args:
            error: The exception the gateway raised.
            context: Surface tag echoed into the decision and event.

        Returns:
            A BLOCK decision when ``fail_safe`` is true (default); an ALLOW
            decision (fail-open, explicitly logged) otherwise.
        """
        if self.fail_safe:
            decision = ModerationDecision(
                allowed=False,
                action=ModerationAction.BLOCK,
                reason=f"moderation gateway error (fail-safe block): {error}",
                context=context,
                fail_safe=True,
            )
        else:
            logger.warning(
                "Moderation gateway error ignored (fail_safe=False, context=%r): %s",
                context,
                error,
            )
            decision = ModerationDecision(
                allowed=True,
                action=ModerationAction.ALLOW,
                reason=f"moderation gateway error ignored (fail_safe=False): {error}",
                context=context,
                fail_safe=True,
            )
        self._emit(decision, provider="", model="")
        return decision

    @staticmethod
    def _emit(decision: ModerationDecision, *, provider: str, model: str) -> None:
        """Emit *decision* on the shared event spine (no-op without sinks)."""
        if not shared_events.has_sinks():
            return
        shared_events.emit(
            shared_events.new_event(
                "llmcore",
                "moderation.decision",
                {
                    "allowed": decision.allowed,
                    "action": decision.action.value,
                    "reason": decision.reason,
                    "categories": decision.categories,
                    "triggered_categories": list(decision.triggered_categories),
                    "context": decision.context,
                    "fail_safe": decision.fail_safe,
                    "provider": provider,
                    "model": model,
                },
            )
        )
