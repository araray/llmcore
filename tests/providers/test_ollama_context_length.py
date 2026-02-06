"""
Tests for OllamaProvider.get_max_context_length() model card fallback.

Validates that when a model is NOT in the hardcoded DEFAULT_OLLAMA_TOKEN_LIMITS
dict, the provider falls back to the model card registry before using the 4096
default.

The root cause of the "Unknown context length for Ollama model 'qwen3:4b'"
warning was that qwen3 isn't in the hardcoded dict, and the model card registry
(which has the correct value of 262144) was never consulted.
"""

from unittest.mock import MagicMock, patch

import pytest

# The ollama library may not be installed in CI. We test get_max_context_length
# which doesn't need the actual client, so we can mock the library availability
# flag and the __init__ to skip client setup.


@pytest.fixture
def ollama_provider():
    """Create an OllamaProvider instance with mocked ollama library.

    We need to mock several imports because the ollama and tiktoken packages
    may not be installed in CI.
    """
    with (
        patch("llmcore.providers.ollama_provider.ollama_available", True),
        patch("llmcore.providers.ollama_provider.tiktoken_available", False),
        patch("llmcore.providers.ollama_provider.AsyncClient", MagicMock()),
    ):
        from llmcore.providers.ollama_provider import OllamaProvider

        config = {
            "default_model": "qwen3:4b",
            "host": "http://localhost:11434",
        }
        return OllamaProvider(config)


class TestOllamaContextLengthFromModelCard:
    """OllamaProvider should consult model card registry for unknown models."""

    def test_hardcoded_model_uses_dict(self, ollama_provider):
        """Models in DEFAULT_OLLAMA_TOKEN_LIMITS use the hardcoded value."""
        assert ollama_provider.get_max_context_length("llama3") == 8000

    def test_hardcoded_model_with_tag(self, ollama_provider):
        """Models with tags (e.g. llama3:8b) also match the hardcoded dict."""
        assert ollama_provider.get_max_context_length("llama3:8b") == 8000

    def test_qwen3_4b_in_hardcoded_dict(self, ollama_provider):
        """qwen3:4b should resolve from hardcoded dict (262144) without
        needing model card fallback — the original bug was a 4096 fallback."""
        assert ollama_provider.get_max_context_length("qwen3:4b") == 262144

    def test_qwen3_base_name_in_hardcoded_dict(self, ollama_provider):
        """qwen3 base name should also resolve from hardcoded dict."""
        assert ollama_provider.get_max_context_length("qwen3") == 262144

    def test_unknown_model_falls_back_to_model_card(self, ollama_provider):
        """Models not in hardcoded dict should consult model card registry."""
        mock_registry = MagicMock()
        mock_registry.get_context_length.return_value = 262144

        with patch(
            "llmcore.model_cards.get_model_card_registry",
            return_value=mock_registry,
        ):
            # Use a model NOT in DEFAULT_OLLAMA_TOKEN_LIMITS
            result = ollama_provider.get_max_context_length("deepseek-r1:7b")

        assert result == 262144
        mock_registry.get_context_length.assert_called_once_with(
            "ollama", "deepseek-r1:7b", default=0,
        )

    def test_model_card_returns_zero_falls_to_default(self, ollama_provider):
        """When model card has no entry (returns 0), fall back to 4096."""
        mock_registry = MagicMock()
        mock_registry.get_context_length.return_value = 0

        with patch(
            "llmcore.model_cards.get_model_card_registry",
            return_value=mock_registry,
        ):
            result = ollama_provider.get_max_context_length("totally-unknown-model")

        assert result == 4096

    def test_model_card_import_error_falls_to_default(self, ollama_provider):
        """If model card registry fails to import, fall back to 4096."""
        with patch(
            "llmcore.model_cards.get_model_card_registry",
            side_effect=ImportError("no model_cards module"),
        ):
            # Use a model NOT in DEFAULT_OLLAMA_TOKEN_LIMITS
            result = ollama_provider.get_max_context_length("deepseek-r1:7b")

        assert result == 4096

    def test_base_model_name_strips_tag(self, ollama_provider):
        """Hardcoded dict lookup tries both full name and base name."""
        assert ollama_provider.get_max_context_length("gemma3:4b") == 128000

    def test_default_model_used_when_none(self, ollama_provider):
        """When model=None, uses provider's default_model."""
        # Override default_model to something NOT in the hardcoded dict
        ollama_provider.default_model = "deepseek-r1:7b"

        mock_registry = MagicMock()
        mock_registry.get_context_length.return_value = 262144

        with patch(
            "llmcore.model_cards.get_model_card_registry",
            return_value=mock_registry,
        ):
            result = ollama_provider.get_max_context_length(None)

        assert result == 262144
        mock_registry.get_context_length.assert_called_once_with(
            "ollama", "deepseek-r1:7b", default=0,
        )
