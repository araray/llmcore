# src/llmcore/routing/events.py
"""Routing events.

Spec §3.6 calls observability non-negotiable here, and the reason is specific:
routing silently changes which vendor served a request, so without a record
the question *"why did this answer look different today?"* has no answer at
all. Every decision therefore emits an event — including the decisions that
chose nothing.

Events ride the existing :mod:`llmcore.shared_events` spine, so a sink already
collecting llmcore events gets routing for free. Emission is guarded by
``has_sinks()`` and the payload is built lazily, so routing costs nothing when
nobody is listening.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .. import shared_events

if TYPE_CHECKING:
    from .models import Outcome, RoutingPlan

logger = logging.getLogger(__name__)

__all__ = [
    "ROUTING_ATTEMPT",
    "ROUTING_CLASSIFIED",
    "ROUTING_EXHAUSTED",
    "ROUTING_FAILOVER",
    "ROUTING_SELECTED",
    "ROUTING_TRANSFORMED",
    "emit_attempt",
    "emit_classified",
    "emit_exhausted",
    "emit_failover",
    "emit_routing_event",
    "emit_selected",
    "emit_transformed",
]

#: A classifier chain produced a lane (or declined to).
ROUTING_CLASSIFIED = "routing.classified"
#: A transform inspected the request and may have rewritten or constrained it.
ROUTING_TRANSFORMED = "routing.transformed"
#: A target was selected from a pool, with the candidates that were not.
ROUTING_SELECTED = "routing.selected"
#: An attempt against one target finished, successfully or not.
ROUTING_ATTEMPT = "routing.attempt"
#: An attempt failed and routing moved to a different target.
ROUTING_FAILOVER = "routing.failover"
#: Every candidate was tried or skipped and none served the request.
ROUTING_EXHAUSTED = "routing.exhausted"


def emit_routing_event(event_type: str, payload: dict[str, Any]) -> None:
    """Emit one routing event on the shared spine.

    Never raises: the spine already swallows sink errors, and a failure to
    describe a routing decision must not fail the request that decision was
    made for.
    """
    if not shared_events.has_sinks():
        return
    try:
        shared_events.emit(shared_events.new_event("llmcore", event_type, payload))
    except Exception:  # pragma: no cover - defensive
        logger.debug("routing event %s could not be emitted", event_type, exc_info=True)


def emit_classified(plan: RoutingPlan) -> None:
    """Record which lane a classifier chain picked, and on what evidence."""
    if not shared_events.has_sinks():
        return
    classification = plan.classification
    emit_routing_event(
        ROUTING_CLASSIFIED,
        {
            "lane": plan.lane,
            "classifier": classification.source if classification else None,
            "confidence": classification.confidence if classification else None,
            "rationale": classification.rationale if classification else None,
            "scores": dict(classification.scores) if classification else {},
        },
    )


def emit_transformed(target_key: str, result: Any) -> None:
    """Record a transform's verdict.

    Findings carry a hash of what was detected, never the value — the whole
    point of the PII path would be undone by logging the PII.
    """
    if not shared_events.has_sinks():
        return
    emit_routing_event(
        ROUTING_TRANSFORMED,
        {
            "target": target_key,
            "transform": getattr(result, "source", None),
            "action": str(getattr(result, "action", "")),
            "reason": getattr(result, "reason", None),
            "constrain_to_pool": getattr(result, "constrain_to_pool", None),
            "findings": [
                {
                    "kind": finding.kind,
                    "hashed": finding.hashed,
                    "where": finding.where,
                    "confidence": finding.confidence,
                }
                for finding in getattr(result, "findings", ())
            ],
        },
    )


def emit_selected(plan: RoutingPlan) -> None:
    """Record the chosen target *and why each other candidate was not*.

    The skipped candidates are the useful half: "it picked the slow one" is
    only diagnosable if the record says the fast one was in a 429 cooldown
    with 12 seconds left.
    """
    if not shared_events.has_sinks():
        return
    emit_routing_event(
        ROUTING_SELECTED,
        {
            "pool": plan.pool,
            "lane": plan.lane,
            "strategy": str(plan.strategy) if plan.strategy else None,
            "chosen": plan.chosen.spec() if plan.chosen else None,
            "estimated_cost_usd": plan.estimated_cost_usd,
            "candidates": [
                {
                    "target": candidate.target.spec(),
                    "eligible": candidate.eligible,
                    "reason": candidate.reason,
                    "score": candidate.score,
                    "cooldown_remaining": candidate.cooldown_remaining,
                }
                for candidate in plan.candidates
            ],
            "notes": list(plan.notes),
        },
    )


def emit_attempt(outcome: Outcome, *, attempt: int, pool: str | None = None) -> None:
    """Record the result of one attempt against one target."""
    if not shared_events.has_sinks():
        return
    emit_routing_event(
        ROUTING_ATTEMPT,
        {
            "target": outcome.target_key,
            "attempt": attempt,
            "ok": outcome.ok,
            "pool": pool,
            "latency_seconds": outcome.latency_seconds,
            "failure": str(outcome.failure) if outcome.failure else None,
            "retry_after_seconds": outcome.retry_after_seconds,
            "cost_usd": outcome.cost_usd,
            "error": outcome.error,
        },
    )


def emit_failover(
    *,
    from_target: str,
    to_target: str,
    failure: str,
    attempt: int,
    pool: str | None = None,
) -> None:
    """Record a move from one target to another, and what prompted it.

    Spec §12 lists "failover hides a broken primary" as a risk; this event is
    the mitigation. Persistent failover also logs a warning, because a pool
    quietly running on its second choice for a week is a problem, not a
    success.
    """
    logger.info(
        "routing: failing over from %s to %s after %s (attempt %d)",
        from_target,
        to_target,
        failure,
        attempt,
    )
    emit_routing_event(
        ROUTING_FAILOVER,
        {
            "from": from_target,
            "to": to_target,
            "failure": failure,
            "attempt": attempt,
            "pool": pool,
        },
    )


def emit_exhausted(*, pool: str | None, attempts: int, candidates: list[dict[str, Any]]) -> None:
    """Record that routing ran out of usable targets."""
    emit_routing_event(
        ROUTING_EXHAUSTED,
        {"pool": pool, "attempts": attempts, "candidates": candidates},
    )
