# tests/providers/test_openai_model_card_context_length.py
"""
Tests for model card registry integration in OpenAIProvider.get_max_context_length().

Bug: OpenAI-compatible providers (deepseek, mistral, xai, groq, together)
route through OpenAIProvider, whose get_max_context_length() only knows
about GPT models. Non-GPT models fall to a 4096 fallback, even when the
model card registry has the correct context length (e.g. 131072 for
deepseek-chat).

Fix: get_max_context_length() now consults the ModelCardRegistry before
falling back to 4096. This resolves cascading failures:
  - Context budget mis-sized (3596 instead of ~130k)
  - Truncation failures when content exceeds tiny budget
  - LLM losing history → repeating same tool calls
  - Premature solver completion

Also verifies that get_models_details() enriches tool support from model
cards instead of relying on GPT substring heuristics.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_openai_provider(
    config: dict,
    *,
    instance_name: str | None = None,
) -> "OpenAIProvider":
    """Construct an OpenAIProvider with tiktoken and AsyncOpenAI mocked."""
    if instance_name:
        config.setdefault("_instance_name", instance_name)
    with (
        patch("llmcore.providers.openai_provider.AsyncOpenAI") as _mock_client,
        patch("llmcore.providers.openai_provider.tiktoken") as mock_tiktoken,
    ):
        mock_tiktoken.encoding_for_model.return_value = MagicMock()
        mock_tiktoken.get_encoding.return_value = MagicMock()
        from llmcore.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(config)


def _deepseek_provider() -> "OpenAIProvider":
    """Create a provider configured as DeepSeek."""
    return _make_openai_provider(
        {
            "api_key": "test-key",
            "base_url": "https://api.deepseek.com",
            "default_model": "deepseek-chat",
        },
        instance_name="deepseek",
    )


def _xai_provider() -> "OpenAIProvider":
    """Create a provider configured as xAI."""
    return _make_openai_provider(
        {
            "api_key": "test-key",
            "base_url": "https://api.x.ai/v1",
            "default_model": "grok-2",
        },
        instance_name="xai",
    )


def _mistral_provider() -> "OpenAIProvider":
    """Create a provider configured as Mistral."""
    return _make_openai_provider(
        {
            "api_key": "test-key",
            "base_url": "https://api.mistral.ai/v1",
            "default_model": "mistral-large-latest",
        },
        instance_name="mistral",
    )


# ---------------------------------------------------------------------------
# Tests: Context length from model cards
# ---------------------------------------------------------------------------


class TestModelCardContextLength:
    """Verify model card registry is consulted for non-GPT context lengths."""

    def test_deepseek_chat_uses_model_card(self):
        """deepseek-chat should resolve to 131072, not 4096."""
        provider = _deepseek_provider()
        limit = provider.get_max_context_length("deepseek-chat")
        # The deepseek-chat model card has max_input_tokens=131072
        assert limit == 131072, (
            f"Expected 131072 from model card, got {limit}. "
            "Model card registry lookup may have failed."
        )

    def test_deepseek_reasoner_uses_model_card(self):
        """deepseek-reasoner should resolve from model card, not fallback."""
        provider = _deepseek_provider()
        limit = provider.get_max_context_length("deepseek-reasoner")
        # Should be the value from the deepseek-reasoner model card
        assert limit > 4096, (
            f"Expected context length > 4096 from model card, got {limit}. "
            "Falling back to hardcoded 4096 — model card not consulted."
        )

    def test_deepseek_default_model_uses_model_card(self):
        """Default model (no argument) should also consult model card."""
        provider = _deepseek_provider()
        limit = provider.get_max_context_length()
        assert limit == 131072

    def test_xai_grok2_uses_model_card(self):
        """xAI grok-2 should resolve from model card."""
        provider = _xai_provider()
        limit = provider.get_max_context_length("grok-2")
        assert limit > 4096, (
            f"Expected context length > 4096 from model card for grok-2, got {limit}"
        )

    def test_gpt4o_still_works(self):
        """GPT models must still resolve via the fast hardcoded path."""
        provider = _make_openai_provider(
            {"api_key": "test-key", "default_model": "gpt-4o"},
            instance_name="openai",
        )
        assert provider.get_max_context_length("gpt-4o") == 128000
        assert provider.get_max_context_length("gpt-4") == 8000  # per DEFAULT_OPENAI_TOKEN_LIMITS
        assert (
            provider.get_max_context_length("gpt-3.5-turbo") == 16000
        )  # per DEFAULT_OPENAI_TOKEN_LIMITS

    def test_unknown_model_still_falls_to_4096(self):
        """A model with no card and no heuristic match gets 4096."""
        provider = _make_openai_provider(
            {"api_key": "test-key", "default_model": "gpt-4o"},
            instance_name="openai",
        )
        limit = provider.get_max_context_length("totally-unknown-model-xyz")
        assert limit == 4096

    def test_model_card_registry_exception_handled_gracefully(self):
        """If model card registry throws, fall back to 4096 without crashing."""
        provider = _deepseek_provider()
        with patch(
            "llmcore.providers.openai_provider.get_model_card_registry",
            side_effect=RuntimeError("registry broken"),
        ):
            limit = provider.get_max_context_length("deepseek-chat")
            assert limit == 4096

    def test_model_card_returns_none_falls_to_4096(self):
        """If model card not found for provider/model pair, fall back to 4096."""
        provider = _deepseek_provider()
        mock_registry = MagicMock()
        mock_registry.get.return_value = None  # No card found
        with patch(
            "llmcore.providers.openai_provider.get_model_card_registry",
            return_value=mock_registry,
        ):
            limit = provider.get_max_context_length("nonexistent-model")
            assert limit == 4096


# ---------------------------------------------------------------------------
# Tests: get_models_details tool support enrichment
# ---------------------------------------------------------------------------


class TestModelDetailsToolSupport:
    """Verify get_models_details enriches tool support from model cards."""

    @pytest.mark.asyncio
    async def test_deepseek_model_details_include_tool_support(self):
        """deepseek-chat should be marked as supports_tools=True."""
        provider = _deepseek_provider()

        # Mock the OpenAI client models.list() response
        mock_model = MagicMock()
        mock_model.id = "deepseek-chat"
        mock_model.owned_by = "deepseek"
        mock_model.created = 1700000000

        mock_response = MagicMock()
        mock_response.data = [mock_model]
        provider._client.models.list = MagicMock(return_value=mock_response)

        # Make the mock awaitable
        import asyncio

        async def mock_list():
            return mock_response

        provider._client.models.list = mock_list

        details = await provider.get_models_details()
        assert len(details) == 1
        assert details[0].id == "deepseek-chat"
        assert details[0].supports_tools is True
        assert details[0].context_length == 131072

    @pytest.mark.asyncio
    async def test_gpt_model_still_gets_tool_support(self):
        """GPT models should still get supports_tools=True via heuristic."""
        provider = _make_openai_provider(
            {"api_key": "test-key", "default_model": "gpt-4o"},
            instance_name="openai",
        )

        mock_model = MagicMock()
        mock_model.id = "gpt-4o"
        mock_model.owned_by = "openai"
        mock_model.created = 1700000000

        async def mock_list():
            mock_response = MagicMock()
            mock_response.data = [mock_model]
            return mock_response

        provider._client.models.list = mock_list

        details = await provider.get_models_details()
        assert details[0].supports_tools is True
