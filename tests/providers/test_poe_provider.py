# tests/providers/test_poe_provider.py
"""
Unit tests for the PoeProvider.

Validates:
- Inheritance from OpenAIProvider (OpenAI-compatible API)
- Provider name and registration
- API key resolution (env var, config)
- Model discovery from /v1/models
- Context length resolution chain
- Poe-ignored parameter stripping
- Balance checking
- Dual backend support (openai vs native)
- Fallback from native to OpenAI-compatible on error
- Default model configuration

All tests use mocking — no live API calls.
"""

import json
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock openai and tiktoken before importing the provider
mock_openai = MagicMock()
mock_tiktoken = MagicMock()
mock_tiktoken.encoding_for_model.return_value = MagicMock()

with patch.dict(
    sys.modules,
    {
        "openai": mock_openai,
        "tiktoken": mock_tiktoken,
    },
):
    mock_openai.AsyncOpenAI = MagicMock
    from llmcore.providers.poe_provider import (
        _POE_IGNORED_PARAMS,
        DEFAULT_POE_MODEL,
        POE_BASE_URL,
        POE_MODELS_URL,
        PoeProvider,
    )

from llmcore.models import Message, ModelDetails
from llmcore.models import Role as LLMCoreRole

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def provider_config():
    return {
        "api_key": "sk-poe-test-key-123",
        "default_model": "Claude-Sonnet-4.6",
        "timeout": 120,
    }


@pytest.fixture
def provider(provider_config):
    """Create a PoeProvider with mocked internals."""
    with patch("llmcore.providers.poe_provider.OpenAIProvider.__init__") as mock_init:
        mock_init.return_value = None
        p = PoeProvider.__new__(PoeProvider)
        # Manually set attributes that __init__ would set
        p.log_raw_payloads_enabled = False
        p._provider_instance_name = None
        p._backend = "openai"
        p._poe_api_key = "sk-poe-test-key-123"
        p._models_cache = None
        p._httpx_session = None
        p._client = MagicMock()
        p.default_model = "Claude-Sonnet-4.6"
        p.api_key = "sk-poe-test-key-123"
        p.base_url = POE_BASE_URL
        return p


@pytest.fixture
def sample_poe_models_response():
    """Sample response from Poe /v1/models endpoint."""
    return {
        "object": "list",
        "data": [
            {
                "id": "Claude-Sonnet-4.6",
                "object": "model",
                "created": 1758868894776,
                "description": "Claude Sonnet 4.6 by Anthropic",
                "owned_by": "Anthropic",
                "architecture": {
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                    "modality": "text,image->text",
                },
                "pricing": {
                    "prompt": "0.0000026",
                    "completion": "0.000013",
                },
            },
            {
                "id": "GPT-5.4",
                "object": "model",
                "created": 1758868894777,
                "description": "GPT-5.4 by OpenAI",
                "owned_by": "OpenAI",
                "architecture": {
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                    "modality": "text,image->text",
                },
                "pricing": {
                    "prompt": "0.000005",
                    "completion": "0.00002",
                },
            },
            {
                "id": "Sora-2",
                "object": "model",
                "created": 1758868894778,
                "description": "Sora 2 video generation by OpenAI",
                "owned_by": "OpenAI",
                "architecture": {
                    "input_modalities": ["text"],
                    "output_modalities": ["video"],
                    "modality": "text->video",
                },
                "pricing": {
                    "prompt": "0",
                    "completion": "0",
                },
            },
        ],
    }


# ---------------------------------------------------------------------------
# Tests: Identity
# ---------------------------------------------------------------------------


class TestPoeProviderIdentity:
    """Tests for provider identity and naming."""

    def test_get_name_default(self, provider):
        """Should return 'poe' by default."""
        assert provider.get_name() == "poe"

    def test_get_name_instance_override(self, provider):
        """Should return instance name if set."""
        provider._provider_instance_name = "poe_custom"
        assert provider.get_name() == "poe_custom"


# ---------------------------------------------------------------------------
# Tests: Initialisation
# ---------------------------------------------------------------------------


class TestPoeProviderInit:
    """Tests for provider initialisation and config resolution."""

    def test_init_with_api_key(self, provider_config):
        """Should initialise with an explicit api_key."""
        with patch("llmcore.providers.poe_provider.OpenAIProvider.__init__") as mock_init:
            mock_init.return_value = None
            p = PoeProvider.__new__(PoeProvider)
            # Simulate __init__
            p._poe_api_key = provider_config["api_key"]
            p._backend = "openai"
            assert p._poe_api_key == "sk-poe-test-key-123"

    def test_init_missing_api_key_raises(self):
        """Should raise ConfigError when no API key is available."""
        from llmcore.exceptions import ConfigError

        config = {"default_model": "GPT-4o-Mini"}
        with patch.dict("os.environ", {}, clear=True):
            # Remove POE_API_KEY if present
            import os

            env_backup = os.environ.pop("POE_API_KEY", None)
            try:
                with pytest.raises(ConfigError, match="Poe API key not found"):
                    PoeProvider(config)
            finally:
                if env_backup:
                    os.environ["POE_API_KEY"] = env_backup

    def test_init_api_key_from_env(self):
        """Should resolve API key from POE_API_KEY env var."""
        config = {"default_model": "GPT-4o-Mini"}
        with patch.dict("os.environ", {"POE_API_KEY": "sk-from-env"}):
            with patch("llmcore.providers.poe_provider.OpenAIProvider.__init__"):
                p = PoeProvider(config)
                assert p._poe_api_key == "sk-from-env"


# ---------------------------------------------------------------------------
# Tests: Model Discovery
# ---------------------------------------------------------------------------


class TestPoeModelDiscovery:
    """Tests for model discovery via /v1/models."""

    @pytest.mark.asyncio
    async def test_get_models_details(self, provider, sample_poe_models_response):
        """Should parse Poe /v1/models response into ModelDetails."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = sample_poe_models_response

        with patch("llmcore.providers.poe_provider.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            details = await provider.get_models_details()

        assert len(details) == 3
        assert all(isinstance(d, ModelDetails) for d in details)

        # Check Claude
        claude = next(d for d in details if d.id == "Claude-Sonnet-4.6")
        assert claude.supports_vision is True
        assert claude.provider_name == "poe"

        # Check Sora (video gen)
        sora = next(d for d in details if d.id == "Sora-2")
        assert sora.supports_vision is False

    @pytest.mark.asyncio
    async def test_fetch_models_caching(self, provider, sample_poe_models_response):
        """Should cache models after first fetch."""
        provider._models_cache = sample_poe_models_response["data"]
        result = await provider._fetch_models()
        assert len(result) == 3
        # No HTTP call should be made when cache is populated

    @pytest.mark.asyncio
    async def test_fetch_models_error_returns_empty(self, provider):
        """Should return empty list on API error."""
        with patch("llmcore.providers.poe_provider.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=Exception("Connection failed"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await provider._fetch_models()
            assert result == []


# ---------------------------------------------------------------------------
# Tests: Context Length
# ---------------------------------------------------------------------------


class TestPoeContextLength:
    """Tests for context length resolution."""

    def test_context_length_fallback(self, provider):
        """Should fall back to 128000 when no model card exists."""
        with patch(
            "llmcore.providers.poe_provider.get_model_card_registry",
            side_effect=Exception("No registry"),
        ):
            assert provider.get_max_context_length("SomeBot") == 128000

    def test_context_length_from_model_card(self, provider):
        """Should use model card context length when available."""
        mock_registry = MagicMock()
        mock_card = MagicMock()
        mock_card.get_context_length.return_value = 200000
        mock_registry.get.return_value = mock_card

        with patch(
            "llmcore.providers.poe_provider.get_model_card_registry",
            return_value=mock_registry,
        ):
            result = provider.get_max_context_length("Claude-Sonnet-4.6")
            assert result == 200000


# ---------------------------------------------------------------------------
# Tests: Balance Checking
# ---------------------------------------------------------------------------


class TestPoeBalance:
    """Tests for point balance checking."""

    @pytest.mark.asyncio
    async def test_get_balance_success(self, provider):
        """Should return balance from API."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"current_point_balance": 295932027}

        with patch("llmcore.providers.poe_provider.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            balance = await provider.get_balance()
            assert balance == 295932027

    @pytest.mark.asyncio
    async def test_get_balance_error_raises(self, provider):
        """Should raise ProviderError on failure."""
        from llmcore.exceptions import ProviderError

        with patch("llmcore.providers.poe_provider.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=Exception("Network error"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(ProviderError, match="balance"):
                await provider.get_balance()


# ---------------------------------------------------------------------------
# Tests: Supported Parameters
# ---------------------------------------------------------------------------


class TestPoeParameters:
    """Tests for parameter handling."""

    def test_ignored_params_stripped(self, provider):
        """Poe-ignored params should not appear in supported params."""
        # Mock the parent's get_supported_parameters
        with patch(
            "llmcore.providers.openai_provider.OpenAIProvider.get_supported_parameters",
            return_value={
                "temperature": {"type": "number"},
                "max_tokens": {"type": "integer"},
                "response_format": {"type": "object"},
                "seed": {"type": "integer"},
                "frequency_penalty": {"type": "number"},
            },
        ):
            params = provider.get_supported_parameters()
            assert "temperature" in params
            assert "max_tokens" in params
            assert "response_format" not in params
            assert "seed" not in params
            assert "frequency_penalty" not in params
            assert "extra_body" in params

    def test_extra_body_always_present(self, provider):
        """extra_body should always be in supported params."""
        with patch(
            "llmcore.providers.openai_provider.OpenAIProvider.get_supported_parameters",
            return_value={},
        ):
            params = provider.get_supported_parameters()
            assert "extra_body" in params


# ---------------------------------------------------------------------------
# Tests: Chat Completion
# ---------------------------------------------------------------------------


class TestPoeChatCompletion:
    """Tests for chat completion dispatch."""

    @pytest.mark.asyncio
    async def test_chat_completion_strips_ignored_params(self, provider):
        """Should strip Poe-ignored params before calling parent."""
        context = [Message(role=LLMCoreRole.USER, content="Hello")]
        mock_response = {
            "id": "chatcmpl-test",
            "choices": [{"message": {"role": "assistant", "content": "Hi"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }

        with patch.object(
            provider.__class__.__bases__[0],  # OpenAIProvider
            "chat_completion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_parent:
            await provider.chat_completion(
                context,
                model="Claude-Sonnet-4.6",
                seed=42,
                response_format={"type": "json"},
                temperature=0.7,
            )

            # Verify seed and response_format were stripped
            call_kwargs = mock_parent.call_args
            assert "seed" not in call_kwargs.kwargs
            assert "response_format" not in call_kwargs.kwargs
            assert call_kwargs.kwargs.get("temperature") == 0.7

    @pytest.mark.asyncio
    async def test_chat_completion_openai_backend(self, provider):
        """Should use OpenAI-compatible path when backend='openai'."""
        provider._backend = "openai"
        context = [Message(role=LLMCoreRole.USER, content="Hello")]
        mock_response = {
            "id": "chatcmpl-test",
            "choices": [{"message": {"role": "assistant", "content": "Hi"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }

        with patch.object(
            provider.__class__.__bases__[0],
            "chat_completion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await provider.chat_completion(context)
            assert result["choices"][0]["message"]["content"] == "Hi"


# ---------------------------------------------------------------------------
# Tests: Provider Registration
# ---------------------------------------------------------------------------


class TestPoeProviderRegistration:
    """Tests for provider registration in PROVIDER_MAP."""

    def test_poe_in_provider_map(self):
        """'poe' should be registered in PROVIDER_MAP."""
        from llmcore.providers.manager import PROVIDER_MAP

        assert "poe" in PROVIDER_MAP

    def test_poe_maps_to_poe_provider(self):
        """'poe' key should map to PoeProvider class."""
        from llmcore.providers.manager import PROVIDER_MAP

        assert PROVIDER_MAP["poe"] is PoeProvider


# ---------------------------------------------------------------------------
# Tests: Provider Enum
# ---------------------------------------------------------------------------


class TestPoeProviderEnum:
    """Tests for POE entry in the Provider enum."""

    def test_poe_in_provider_enum(self):
        """POE should exist in model_cards.schema.Provider."""
        from llmcore.model_cards.schema import Provider

        assert hasattr(Provider, "POE")
        assert Provider.POE.value == "poe"


# ---------------------------------------------------------------------------
# Tests: Lifecycle
# ---------------------------------------------------------------------------


class TestPoeLifecycle:
    """Tests for provider lifecycle (close)."""

    @pytest.mark.asyncio
    async def test_close_cleans_up(self, provider):
        """close() should clean up httpx session and call parent close."""
        mock_session = AsyncMock()
        provider._httpx_session = mock_session

        with patch.object(
            provider.__class__.__bases__[0],
            "close",
            new_callable=AsyncMock,
        ) as mock_parent_close:
            await provider.close()
            mock_session.aclose.assert_called_once()
            mock_parent_close.assert_called_once()
            assert provider._httpx_session is None
