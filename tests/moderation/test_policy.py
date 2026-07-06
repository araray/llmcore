# tests/moderation/test_policy.py
"""
Tests for llmcore.moderation policy evaluation, fail-safe, and event emission (SF-1).

Covers:
- Per-category thresholds (trigger at/above, override provider flags below)
- default_threshold for unlisted categories; provider-flag fallback
- default_action block vs warn semantics
- Fail-safe: gateway exception => BLOCK when enabled; fail-open opt-out
- moderate() one-call seam (check + evaluate + exception routing)
- ModerationDecision events on the shared event spine
"""

from __future__ import annotations

import pytest

from llmcore import shared_events
from llmcore.moderation import (
    ModerationAction,
    ModerationDecision,
    ModerationPolicy,
    ModerationResult,
    NoopGateway,
    moderate,
)


@pytest.fixture(autouse=True)
def _clean_spine():
    """Run every test with an empty sink registry, restoring it afterwards."""
    saved = list(shared_events._sinks)
    for sink in saved:
        shared_events.unregister_sink(sink)
    yield
    for sink in list(shared_events._sinks):
        shared_events.unregister_sink(sink)
    for sink in saved:
        shared_events.register_sink(sink)


class _BoomGateway:
    """Gateway whose backend is down."""

    async def check(self, text: str, *, context: str = "") -> ModerationResult:
        raise RuntimeError("moderation backend unreachable")


# =============================================================================
# THRESHOLD EVALUATION
# =============================================================================


def test_score_at_threshold_triggers_block():
    policy = ModerationPolicy(thresholds={"violence": 0.8})
    result = ModerationResult(flagged=False, categories={"violence": 0.8})
    decision = policy.evaluate(result)
    assert decision.allowed is False
    assert decision.action is ModerationAction.BLOCK
    assert decision.triggered_categories == ("violence",)
    assert "violence" in decision.reason


def test_score_below_threshold_allows_even_when_provider_flagged():
    """A local threshold overrides the provider's flag for that category."""
    policy = ModerationPolicy(thresholds={"violence": 0.9})
    result = ModerationResult(
        flagged=True,
        categories={"violence": 0.5},
        flagged_categories=("violence",),
    )
    decision = policy.evaluate(result)
    assert decision.allowed is True
    assert decision.action is ModerationAction.ALLOW
    assert decision.triggered_categories == ()


def test_unlisted_category_defers_to_provider_flag():
    policy = ModerationPolicy(thresholds={"violence": 0.9})
    result = ModerationResult(
        flagged=True,
        categories={"violence": 0.5, "hate": 0.4},
        flagged_categories=("hate",),
    )
    decision = policy.evaluate(result)
    assert decision.allowed is False
    assert decision.triggered_categories == ("hate",)


def test_default_threshold_applies_to_unlisted_categories():
    policy = ModerationPolicy(thresholds={"violence": 0.9}, default_threshold=0.3)
    result = ModerationResult(flagged=False, categories={"violence": 0.5, "hate": 0.4})
    decision = policy.evaluate(result)
    # violence has an explicit higher threshold (not tripped); hate trips the default.
    assert decision.triggered_categories == ("hate",)
    assert decision.allowed is False


def test_provider_flagged_category_without_score_triggers():
    policy = ModerationPolicy()
    result = ModerationResult(flagged=True, flagged_categories=("sexual_minors",))
    decision = policy.evaluate(result)
    assert decision.allowed is False
    assert decision.triggered_categories == ("sexual_minors",)


def test_provider_flagged_without_any_category_info_blocks():
    policy = ModerationPolicy()
    result = ModerationResult(flagged=True)
    decision = policy.evaluate(result)
    assert decision.allowed is False
    assert decision.triggered_categories == ("provider_flagged",)


def test_clean_result_allows():
    policy = ModerationPolicy(thresholds={"violence": 0.8})
    result = ModerationResult(flagged=False, categories={"violence": 0.1, "hate": 0.0})
    decision = policy.evaluate(result, context="input")
    assert decision.allowed is True
    assert decision.action is ModerationAction.ALLOW
    assert decision.context == "input"
    assert decision.fail_safe is False
    assert decision.categories == {"violence": 0.1, "hate": 0.0}


def test_default_action_warn_allows_but_reports():
    policy = ModerationPolicy(thresholds={"violence": 0.5}, default_action=ModerationAction.WARN)
    result = ModerationResult(flagged=False, categories={"violence": 0.9})
    decision = policy.evaluate(result)
    assert decision.allowed is True
    assert decision.action is ModerationAction.WARN
    assert decision.triggered_categories == ("violence",)


def test_default_action_accepts_config_string():
    policy = ModerationPolicy(default_action="warn")  # type: ignore[arg-type]
    assert policy.default_action is ModerationAction.WARN


# =============================================================================
# FAIL-SAFE
# =============================================================================


def test_gateway_error_blocks_when_fail_safe():
    policy = ModerationPolicy()  # fail_safe defaults to True
    decision = policy.decision_for_error(TimeoutError("gateway down"), context="input")
    assert decision.allowed is False
    assert decision.action is ModerationAction.BLOCK
    assert decision.fail_safe is True
    assert "gateway down" in decision.reason
    assert decision.context == "input"


def test_gateway_error_allows_when_fail_open():
    policy = ModerationPolicy(fail_safe=False)
    decision = policy.decision_for_error(TimeoutError("gateway down"))
    assert decision.allowed is True
    assert decision.action is ModerationAction.ALLOW
    assert decision.fail_safe is True
    assert "fail_safe=False" in decision.reason


async def test_moderate_routes_gateway_exception_to_fail_safe_block():
    decision = await moderate(_BoomGateway(), ModerationPolicy(), "anything", context="input")
    assert decision.allowed is False
    assert decision.action is ModerationAction.BLOCK
    assert decision.fail_safe is True


async def test_moderate_fail_open_policy_allows_on_gateway_exception():
    decision = await moderate(_BoomGateway(), ModerationPolicy(fail_safe=False), "anything")
    assert decision.allowed is True


async def test_moderate_with_noop_gateway_allows():
    decision = await moderate(NoopGateway(), ModerationPolicy(), "anything", context="output")
    assert decision.allowed is True
    assert decision.action is ModerationAction.ALLOW
    assert decision.context == "output"


async def test_noop_gateway_never_flags():
    result = await NoopGateway().check("terrible content", context="input")
    assert result.flagged is False
    assert result.categories == {}


# =============================================================================
# EVENT EMISSION (shared event spine)
# =============================================================================


def test_evaluate_emits_moderation_decision_event():
    events: list[shared_events.UnifiedEvent] = []
    shared_events.register_sink(events.append)

    policy = ModerationPolicy(thresholds={"violence": 0.5})
    result = ModerationResult(
        flagged=True,
        categories={"violence": 0.9},
        provider="openai",
        model="omni-moderation-latest",
    )
    with shared_events.correlation_context() as cid:
        decision = policy.evaluate(result, context="input")

    assert decision.allowed is False
    assert len(events) == 1
    event = events[0]
    assert event.source == "llmcore"
    assert event.type == "moderation.decision"
    assert event.correlation_id == cid
    assert event.payload["allowed"] is False
    assert event.payload["action"] == "block"
    assert event.payload["triggered_categories"] == ["violence"]
    assert event.payload["categories"] == {"violence": 0.9}
    assert event.payload["context"] == "input"
    assert event.payload["fail_safe"] is False
    assert event.payload["provider"] == "openai"
    assert event.payload["model"] == "omni-moderation-latest"


def test_fail_safe_decision_emits_event():
    events: list[shared_events.UnifiedEvent] = []
    shared_events.register_sink(events.append)

    ModerationPolicy().decision_for_error(RuntimeError("boom"), context="tool")

    assert len(events) == 1
    assert events[0].type == "moderation.decision"
    assert events[0].payload["allowed"] is False
    assert events[0].payload["fail_safe"] is True
    assert events[0].payload["context"] == "tool"


def test_no_sink_registered_is_a_noop():
    policy = ModerationPolicy()
    decision = policy.evaluate(ModerationResult(flagged=False))
    assert isinstance(decision, ModerationDecision)
    assert decision.allowed is True
