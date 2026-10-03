"""An unfilled capability block means unknown, not False."""

from __future__ import annotations

from llmcore.api import _card_capability
from llmcore.model_cards.schema import ModelCapabilities, ModelCard, ModelContext


def card(**flags) -> ModelCard:
    return ModelCard(
        model_id="m", display_name="m", provider="anthropic", model_type="chat",
        context=ModelContext(max_input_tokens=1000),
        capabilities=ModelCapabilities(**flags),
    )


class TestIsPopulated:
    def test_streaming_alone_proves_nothing(self):
        # Generators default streaming to True and leave the rest False.
        assert ModelCapabilities(streaming=True).is_populated() is False

    def test_an_all_false_block_is_unpopulated(self):
        assert ModelCapabilities(streaming=False).is_populated() is False

    def test_any_real_flag_counts(self):
        assert ModelCapabilities(tool_use=True).is_populated() is True
        assert ModelCapabilities(vision=True).is_populated() is True


class TestReporting:
    def test_unfilled_block_reports_unknown(self):
        assert _card_capability(card(streaming=True), "tool_use") is None

    def test_a_real_negative_is_still_false(self):
        # json_mode set proves the block was filled in, so tool_use=False
        # is a genuine negative.
        assert _card_capability(card(json_mode=True), "tool_use") is False

    def test_a_real_positive_is_true(self):
        assert _card_capability(card(tool_use=True), "tool_use") is True

    def test_missing_block_is_unknown(self):
        bare = ModelCard(model_id="m", display_name="m", provider="p",
                         model_type="chat",
                         context=ModelContext(max_input_tokens=1000))
        assert _card_capability(bare, "tool_use") is None


class TestModelDetailsDefault:
    def test_unset_capabilities_default_to_unknown(self):
        from llmcore.models import ModelDetails

        details = ModelDetails(id="m", provider_name="p")
        assert details.supports_tools is None
        assert details.supports_vision is None

    def test_none_is_falsy_so_truthiness_checks_are_unchanged(self):
        from llmcore.models import ModelDetails

        details = ModelDetails(id="m", provider_name="p")
        assert not details.supports_tools
