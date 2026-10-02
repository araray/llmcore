# tests/agents/test_capability_checker_model_cards.py
"""The capability pre-check must know about models released after 2024.

`CapabilityChecker` resolved models against a static table of 13 entries
(gpt-4-turbo, claude-3-opus, gemini-pro...), returned `compatible=False`
for anything absent from it, and `strict_mode` defaults to True — so
`EnhancedAgentManager.run()` refused to run any current model out of the
box, while a model card existed for each of them.

`agents.capability_check.use_model_cards` has always defaulted to True and
promised exactly this lookup; nothing read it.
"""

from __future__ import annotations

import pytest

from llmcore.agents.routing.capability_checker import (
    Capability,
    CapabilityChecker,
    IssueSeverity,
    ModelInfo,
    _capability_from_card,
)


class FakeCaps:
    def __init__(self, **flags):
        for k, v in flags.items():
            setattr(self, k, v)

    def __getattr__(self, item):  # unset flags read as False, as pydantic would
        return False


class FakeContext:
    def __init__(self, max_input_tokens=0, max_output_tokens=0):
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens


class FakeCard:
    def __init__(self, model_id="m", provider="p", caps=None, ctx=None, model_type="chat"):
        self.model_id = model_id
        self.provider = provider
        self.capabilities = caps
        self.context = ctx
        self.model_type = model_type
        self.pricing = None


class TestCardTranslation:
    def test_a_filled_card_yields_known_capabilities(self):
        info = _capability_from_card(
            FakeCard(caps=FakeCaps(tool_use=True, vision=True),
                     ctx=FakeContext(200_000, 64_000))
        )
        assert info.capabilities_known is True
        assert info.supports_tools and info.supports_vision
        assert info.context_window == 200_000

    def test_streaming_alone_does_not_count_as_filled_in(self):
        # Card generators default streaming to True. Treating such a card
        # as authoritative makes the checker assert that a frontier model
        # cannot call tools.
        info = _capability_from_card(
            FakeCard(caps=FakeCaps(streaming=True), ctx=FakeContext(128_000))
        )
        assert info.capabilities_known is False
        assert Capability.STREAMING in info.capabilities

    def test_context_window_is_trusted_even_on_a_stub_card(self):
        # It is a number, not a defaulted flag.
        info = _capability_from_card(
            FakeCard(caps=FakeCaps(streaming=True), ctx=FakeContext(128_000))
        )
        assert info.context_window == 128_000

    def test_a_card_with_nothing_useful_is_rejected(self):
        assert _capability_from_card(FakeCard(caps=FakeCaps(), ctx=FakeContext(0))) is None


class TestUnknownIsNotIncompatible:
    def test_unknown_model_is_allowed_with_a_warning(self):
        checker = CapabilityChecker(use_model_cards=False)
        result = checker.check_compatibility("no-such-model-anywhere", requires_tools=True)
        assert result.compatible is True
        assert result.issues and all(
            i.severity == IssueSeverity.WARNING for i in result.issues
        )

    def test_static_table_models_still_resolve(self):
        checker = CapabilityChecker()
        assert checker.get_model_info("gpt-4o") is not None


class TestRealNegativesStillFail:
    def test_a_card_that_states_no_tools_is_an_error(self):
        registry = {
            "no-tools": ModelInfo(
                name="no-tools", provider="p",
                capabilities={Capability.JSON_MODE}, context_window=100_000,
                max_output_tokens=1_000, capabilities_known=True,
            )
        }
        result = CapabilityChecker(model_registry=registry).check_compatibility(
            "no-tools", requires_tools=True
        )
        assert result.compatible is False
        assert any(i.severity == IssueSeverity.ERROR for i in result.issues)

    def test_unverified_capabilities_warn_rather_than_error(self):
        registry = {
            "stub": ModelInfo(
                name="stub", provider="p", capabilities=set(),
                context_window=100_000, max_output_tokens=1_000,
                capabilities_known=False,
            )
        }
        result = CapabilityChecker(model_registry=registry).check_compatibility(
            "stub", requires_tools=True
        )
        assert result.compatible is True
        assert any(i.severity == IssueSeverity.WARNING for i in result.issues)

    def test_context_window_shortfall_is_still_an_error(self):
        registry = {
            "small": ModelInfo(
                name="small", provider="p", capabilities={Capability.TOOLS},
                context_window=8_000, max_output_tokens=1_000,
            )
        }
        result = CapabilityChecker(model_registry=registry).check_compatibility(
            "small", requires_tools=True, min_context_window=200_000
        )
        assert result.compatible is False


class TestAgainstTheRealRegistry:
    """The behaviour that was broken, exercised end to end."""

    @pytest.mark.parametrize("model", ["claude-opus-5-5", "gpt-5.6-sol"])
    def test_current_models_are_no_longer_refused(self, model):
        checker = CapabilityChecker()
        result = checker.check_compatibility(model, requires_tools=True)
        assert result.compatible is True, (
            f"{model} must not be refused; a model card exists for it"
        )

    def test_current_models_resolve_through_the_cards(self):
        checker = CapabilityChecker()
        assert checker.get_model_info("claude-opus-5-5") is not None

    def test_disabling_model_cards_falls_back_to_the_static_table(self):
        checker = CapabilityChecker(use_model_cards=False)
        assert checker.get_model_info("claude-opus-5-5") is None
