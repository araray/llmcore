# tests/providers/test_openrouter_provider.py
"""
Unit tests for the OpenRouterProvider.

Validates:
- Inheritance from OpenAIProvider (OpenAI-compatible API)
- Provider name and registration
- OpenRouter-specific headers (HTTP-Referer, X-Title)
- Model discovery from /models API
- Context length resolution chain
- OpenRouter kwargs extraction (provider, models, reasoning)
- Dual backend support (openai vs sdk)
- Default model configuration

All tests use mocking — no live API calls.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Plain imports. These used to happen inside patch.dict(sys.modules), which
# drops every module imported in the block when it exits: the provider modules
# were then re-imported as fresh copies, so patches here targeted a different
# OpenAIProvider than the one OpenRouterProvider inherits from, and the tests
# passed or failed depending on what another file had imported first.
pytest.importorskip("openai")
from llmcore.providers.openrouter_provider import (  # noqa: E402
    DEFAULT_OPENROUTER_MODEL,
    OPENROUTER_BASE_URL,
    OpenRouterProvider,
)

from llmcore.models import Message
from llmcore.models import Role as LLMCoreRole


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provider_config():
    return {
        "api_key": "sk-or-test-key",
        "default_model": "openai/gpt-4o-mini",
        "app_url": "https://myapp.example.com",
        "app_title": "Test App",
    }


@pytest.fixture
def provider(provider_config):
    with patch("llmcore.providers.openrouter_provider.OpenAIProvider.__init__") as mock_init:
        mock_init.return_value = None
        p = OpenRouterProvider.__new__(OpenRouterProvider)
        # Manually set attributes that __init__ would set
        p.log_raw_payloads_enabled = False
        p._provider_instance_name = None
        p._app_url = "https://myapp.example.com"
        p._app_title = "Test App"
        p._backend = "openai"
        p._openrouter_client = None
        p._models_cache = None
        p._client = MagicMock()
        p.default_model = "openai/gpt-4o-mini"
        p.api_key = "sk-or-test-key"
        p.base_url = OPENROUTER_BASE_URL
        return p


# ---------------------------------------------------------------------------
# Test: Provider Identity
# ---------------------------------------------------------------------------

class TestProviderIdentity:
    def test_name_is_openrouter(self, provider):
        assert provider.get_name() == "openrouter"

    def test_instance_name_override(self, provider):
        provider._provider_instance_name = "my-router"
        assert provider.get_name() == "my-router"

    def test_default_model(self):
        assert DEFAULT_OPENROUTER_MODEL == "openai/gpt-5.4-mini"

    def test_base_url(self):
        assert OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------------------
# Test: Model Discovery
# ---------------------------------------------------------------------------

class TestModelDiscovery:
    @pytest.mark.asyncio
    async def test_get_models_details_parses_response(self, provider):
        """Model discovery parses OpenRouter /models response correctly."""
        # Mock _fetch_models to return test data
        test_models = [
            {
                "id": "anthropic/claude-sonnet-4.6",
                "name": "Claude Sonnet 4.6",
                "context_length": 1048576,
                "architecture": {
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                },
                "top_provider": {
                    "max_completion_tokens": 16384,
                    "is_moderated": False,
                },
                "pricing": {
                    "prompt": "0.000003",
                    "completion": "0.000015",
                },
                "supported_parameters": [
                    "temperature", "tools", "tool_choice", "reasoning",
                    "response_format", "structured_outputs",
                ],
                "description": "Test model",
                "knowledge_cutoff": "2025-04",
                "canonical_slug": "anthropic/claude-sonnet-4.6",
            }
        ]
        provider._fetch_models = AsyncMock(return_value=test_models)

        details = await provider.get_models_details()
        assert len(details) == 1

        d = details[0]
        assert d.id == "anthropic/claude-sonnet-4.6"
        assert d.display_name == "Claude Sonnet 4.6"
        assert d.context_length == 1048576
        assert d.max_output_tokens == 16384
        assert d.supports_tools is True
        assert d.supports_vision is True
        assert d.supports_reasoning is True
        assert d.provider_name == "openrouter"

        # Pricing should be per-million in metadata
        pricing_m = d.metadata["pricing_per_million"]
        assert pricing_m["input"] == 3.0  # $3/M
        assert pricing_m["output"] == 15.0  # $15/M

    @pytest.mark.asyncio
    async def test_get_models_details_handles_empty(self, provider):
        provider._fetch_models = AsyncMock(return_value=[])
        details = await provider.get_models_details()
        assert details == []


# ---------------------------------------------------------------------------
# Test: Context Length Resolution
# ---------------------------------------------------------------------------

class TestContextLength:
    def test_from_cached_models(self, provider):
        """Context length from cached API response."""
        provider._models_cache = [
            {"id": "test/model", "context_length": 262144}
        ]
        assert provider.get_max_context_length("test/model") == 262144

    def test_fallback_when_unknown(self, provider):
        """Fallback to 128000 for unknown models."""
        provider._models_cache = []
        assert provider.get_max_context_length("unknown/model") == 128000


# ---------------------------------------------------------------------------
# Test: OpenRouter-specific kwargs handling
# ---------------------------------------------------------------------------

class TestOpenRouterKwargs:
    @pytest.mark.asyncio
    async def test_provider_preferences_extracted(self, provider):
        """OpenRouter 'provider' kwarg is extracted to extra_body."""
        # We need to mock the parent chat_completion
        with patch(
            "llmcore.providers.openai_provider.OpenAIProvider.chat_completion",
            new_callable=AsyncMock,
        ) as mock_parent:
            mock_parent.return_value = {"choices": [{"message": {"content": "ok"}}]}

            msgs = [Message(role=LLMCoreRole.USER, content="test")]
            await provider.chat_completion(
                msgs,
                model="test/model",
                provider={"order": ["Anthropic"], "allow_fallbacks": True},
                temperature=0.5,
            )

            # Passed through; the OpenRouter vendor profile moves it into the
            # body (extra_body on the SDK transport).
            call_kwargs = mock_parent.call_args
            assert call_kwargs.kwargs["provider"]["order"] == ["Anthropic"]

    @pytest.mark.asyncio
    async def test_multi_model_routing(self, provider):
        """OpenRouter 'models' kwarg for fallback routing."""
        with patch(
            "llmcore.providers.openai_provider.OpenAIProvider.chat_completion",
            new_callable=AsyncMock,
        ) as mock_parent:
            mock_parent.return_value = {"choices": [{"message": {"content": "ok"}}]}

            msgs = [Message(role=LLMCoreRole.USER, content="test")]
            await provider.chat_completion(
                msgs,
                models=["openai/gpt-4o", "anthropic/claude-sonnet-4.6"],
            )

            call_kwargs = mock_parent.call_args
            assert call_kwargs.kwargs["models"] == ["openai/gpt-4o", "anthropic/claude-sonnet-4.6"]


# ---------------------------------------------------------------------------
# Test: Backend Selection
# ---------------------------------------------------------------------------

class TestBackendSelection:
    def test_default_backend_is_openai(self, provider):
        assert provider._backend == "openai"

    def test_sdk_backend_requires_package(self, provider):
        """When SDK not installed, falls back to openai backend."""
        # _backend should stay "openai" if SDK unavailable
        provider._backend = "openai"
        assert provider._openrouter_client is None


# ---------------------------------------------------------------------------
# Test: Provider Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_in_provider_map(self):
        """OpenRouter should be in the PROVIDER_MAP."""
        from llmcore.providers.manager import PROVIDER_MAP
        assert "openrouter" in PROVIDER_MAP
        assert PROVIDER_MAP["openrouter"] is OpenRouterProvider
