# tests/providers/test_vllm_provider.py
"""
Unit tests for the VLLMProvider.

Validates:
- Subclass relationship to OpenAIProvider
- Required base_url validation
- API key resolution (explicit, env var, "EMPTY" fallback)
- Provider name / multi-instance naming
- Model discovery via /v1/models (httpx, not SDK) with max_model_len
- Context length resolution chain (cache → config → registry → 4096)
- get_supported_parameters filtering (OpenAI-only out, vLLM-specific in, extra_body declared)
- chat_completion repackages vLLM-specific kwargs into extra_body
- count_tokens / count_message_tokens use approximation (no tiktoken)
- Tool / vision heuristics against known model families
- Registration in ProviderManager's PROVIDER_MAP

All tests use mocking — no live API calls.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Note on test isolation:
#
# Earlier iterations of this file used ``patch.dict(sys.modules, ...)`` at
# module-import time to stub out ``openai`` and ``tiktoken`` for
# environments where those packages weren't installed. That pattern causes
# import-time sibling modules (notably ``llmcore.exceptions``) to be
# re-imported, producing *two distinct* ``ConfigError`` classes at two
# different ``id()``s. Exceptions raised from code loaded under the patch
# could then not be caught by ``except ConfigError`` in tests importing
# ``ConfigError`` outside the patch.
#
# We rely on ``openai`` being installed in the test environment (it's a
# core ``llmcore[openai]`` dependency) and avoid the patching dance
# entirely. ``tiktoken`` is likewise a hard dependency of
# ``OpenAIProvider`` in this repo. If either becomes optional again, a
# cleaner approach is a conftest fixture that runs before collection.
from llmcore.exceptions import ConfigError, ProviderError
from llmcore.models import Message, ModelDetails, Tool
from llmcore.models import Role as LLMCoreRole
from llmcore.providers.vllm_provider import (
    _OPENAI_ONLY_KEYS,
    _VLLM_EXTRA_BODY_KEYS,
    DEFAULT_VLLM_MODEL,
    VLLMProvider,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VLLM_BASE_URL = "http://localhost:8000/v1"


@pytest.fixture
def minimal_config() -> dict[str, Any]:
    """The smallest config that should construct a VLLMProvider."""
    return {"base_url": VLLM_BASE_URL}


@pytest.fixture
def full_config() -> dict[str, Any]:
    """A config exercising every honoured field."""
    return {
        "base_url": VLLM_BASE_URL,
        "api_key": "sk-vllm-secret",
        "default_model": "Qwen/Qwen2.5-7B-Instruct",
        "timeout": 180,
        "fallback_context_length": 16384,
    }


@pytest.fixture
def provider_no_init(minimal_config):
    """Construct a VLLMProvider without invoking its (or its parent's) __init__.

    Mirrors the pattern in test_openrouter_provider.py: we set just the
    attributes each test needs, then call the methods directly. This
    keeps the tests isolated from AsyncOpenAI construction details.
    """
    p = VLLMProvider.__new__(VLLMProvider)
    # BaseProvider attrs
    p.log_raw_payloads_enabled = False
    p._provider_instance_name = None
    # OpenAIProvider attrs consulted by the methods under test
    p._client = MagicMock()
    p._encoding = None
    p.api_key = "EMPTY"
    p.base_url = VLLM_BASE_URL
    p.default_model = DEFAULT_VLLM_MODEL
    p.timeout = 240.0
    p.max_retries = 2
    # VLLMProvider-specific
    p._vllm_model_cache = {}
    p._fallback_context_length = None
    return p


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------


class TestInheritance:
    def test_subclasses_openai_provider(self):
        """VLLMProvider must inherit from OpenAIProvider.

        Note: we compare by qualname rather than ``issubclass()`` because
        the test module patches ``sys.modules`` at import time, which can
        produce a distinct module object for ``openai_provider`` relative
        to a fresh import in the test body. The subclass relationship is
        verified at runtime by exercising parent-class behaviour in the
        other test classes below."""
        base = VLLMProvider.__bases__[0]
        assert base.__qualname__ == "OpenAIProvider"
        assert base.__module__ == "llmcore.providers.openai_provider"

    def test_default_model_is_llama3(self):
        assert DEFAULT_VLLM_MODEL == "meta-llama/Llama-3.1-8B-Instruct"


# ---------------------------------------------------------------------------
# __init__ validation
# ---------------------------------------------------------------------------


class TestInit:
    @pytest.fixture(autouse=True)
    def _pretend_openai_installed(self, monkeypatch):
        """Patch module-level ``openai_available`` so __init__'s
        ImportError gate does not fire. In production this flag is set
        by the real ``import openai`` at module load; under test we
        mocked ``openai`` in ``sys.modules`` which leaves the flag False."""
        import llmcore.providers.vllm_provider as vp

        monkeypatch.setattr(vp, "openai_available", True)

    def test_init_requires_base_url(self):
        """Constructing without base_url raises ConfigError."""
        with pytest.raises(ConfigError, match="base_url"):
            VLLMProvider({})

    def test_init_empty_base_url_raises(self):
        with pytest.raises(ConfigError, match="base_url"):
            VLLMProvider({"base_url": ""})

    def test_init_happy_path_calls_parent(self, minimal_config):
        """Constructor should delegate to OpenAIProvider.__init__ with
        the correctly-shaped config.

        We patch the *parent* class to observe what it receives."""
        with patch(
            "llmcore.providers.vllm_provider.OpenAIProvider.__init__",
            return_value=None,
        ) as mock_parent_init:
            p = VLLMProvider(minimal_config)

            assert mock_parent_init.called
            call_args, call_kwargs = mock_parent_init.call_args
            # First positional is the config dict
            parent_config = call_args[0]
            assert parent_config["base_url"] == VLLM_BASE_URL
            assert parent_config["api_key"] == "EMPTY"
            assert parent_config["default_model"] == DEFAULT_VLLM_MODEL
            assert parent_config["timeout"] == 240.0
            assert call_kwargs.get("log_raw_payloads") is False

        assert p._vllm_model_cache == {}
        assert p._fallback_context_length is None

    def test_init_respects_explicit_api_key(self, full_config):
        with patch(
            "llmcore.providers.vllm_provider.OpenAIProvider.__init__",
            return_value=None,
        ) as mock_parent_init:
            VLLMProvider(full_config)
            parent_config = mock_parent_init.call_args[0][0]
            assert parent_config["api_key"] == "sk-vllm-secret"
            assert parent_config["default_model"] == "Qwen/Qwen2.5-7B-Instruct"

    def test_init_reads_api_key_from_env_var(self, monkeypatch):
        monkeypatch.setenv("MY_VLLM_TOKEN", "envkey-abc")
        cfg = {"base_url": VLLM_BASE_URL, "api_key_env_var": "MY_VLLM_TOKEN"}
        with patch(
            "llmcore.providers.vllm_provider.OpenAIProvider.__init__",
            return_value=None,
        ) as mock_parent_init:
            VLLMProvider(cfg)
            parent_config = mock_parent_init.call_args[0][0]
            assert parent_config["api_key"] == "envkey-abc"

    def test_init_env_var_unset_falls_back_to_empty(self, monkeypatch):
        monkeypatch.delenv("MY_VLLM_TOKEN", raising=False)
        cfg = {"base_url": VLLM_BASE_URL, "api_key_env_var": "MY_VLLM_TOKEN"}
        with patch(
            "llmcore.providers.vllm_provider.OpenAIProvider.__init__",
            return_value=None,
        ) as mock_parent_init:
            VLLMProvider(cfg)
            parent_config = mock_parent_init.call_args[0][0]
            assert parent_config["api_key"] == "EMPTY"

    def test_init_clears_encoding(self, minimal_config):
        """After parent __init__ loads a tiktoken encoding, our override
        must clear it so count_tokens falls back to approximation."""
        with patch(
            "llmcore.providers.vllm_provider.OpenAIProvider.__init__",
            return_value=None,
        ):
            p = VLLMProvider(minimal_config)
            assert p._encoding is None

    def test_init_accepts_fallback_context_length(self, full_config):
        with patch(
            "llmcore.providers.vllm_provider.OpenAIProvider.__init__",
            return_value=None,
        ):
            p = VLLMProvider(full_config)
            assert p._fallback_context_length == 16384

    def test_init_ignores_invalid_fallback_context_length(self, caplog):
        cfg = {"base_url": VLLM_BASE_URL, "fallback_context_length": "not-an-int"}
        with patch(
            "llmcore.providers.vllm_provider.OpenAIProvider.__init__",
            return_value=None,
        ):
            with caplog.at_level("WARNING"):
                p = VLLMProvider(cfg)
        assert p._fallback_context_length is None
        assert any("fallback_context_length" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


class TestIdentity:
    def test_get_name_default(self, provider_no_init):
        assert provider_no_init.get_name() == "vllm"

    def test_get_name_with_instance_name(self, provider_no_init):
        provider_no_init._provider_instance_name = "vllm_prod"
        assert provider_no_init.get_name() == "vllm_prod"

    def test_get_name_empty_instance_name_falls_back(self, provider_no_init):
        provider_no_init._provider_instance_name = ""
        # Empty string is falsy → default name
        assert provider_no_init.get_name() == "vllm"


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------


class TestHeuristics:
    @pytest.mark.parametrize(
        "model_id",
        [
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen3-32B",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "NousResearch/Hermes-3-Llama-3.1-8B",
            "cohere/command-r-plus",
            "deepseek-ai/DeepSeek-V3",
        ],
    )
    def test_tool_capable_models(self, model_id):
        assert VLLMProvider._heuristic_supports_tools(model_id) is True

    @pytest.mark.parametrize(
        "model_id",
        [
            "gpt2",
            "bigscience/bloom",
            "microsoft/phi-2",
            "some-random/model-name",
        ],
    )
    def test_non_tool_capable_models(self, model_id):
        assert VLLMProvider._heuristic_supports_tools(model_id) is False

    @pytest.mark.parametrize(
        "model_id",
        [
            "Qwen/Qwen2-VL-7B-Instruct",
            "llava-hf/llava-1.5-7b-hf",
            "mistralai/Pixtral-12B-2409",
            "allenai/Molmo-7B-D",
            "OpenGVLab/InternVL2-8B",
        ],
    )
    def test_vision_capable_models(self, model_id):
        assert VLLMProvider._heuristic_supports_vision(model_id) is True

    @pytest.mark.parametrize(
        "model_id",
        [
            "meta-llama/Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "microsoft/phi-4",
        ],
    )
    def test_non_vision_models(self, model_id):
        assert VLLMProvider._heuristic_supports_vision(model_id) is False


# ---------------------------------------------------------------------------
# Model discovery — /v1/models via httpx
# ---------------------------------------------------------------------------


def _mock_httpx_response(payload: dict[str, Any], status_code: int = 200):
    """Build a MagicMock Response suitable for `resp.raise_for_status()`
    / `resp.json()` in the module under test."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    if status_code >= 400:
        import httpx

        err = httpx.HTTPStatusError(
            "boom",
            request=MagicMock(),
            response=MagicMock(status_code=status_code),
        )
        resp.raise_for_status.side_effect = err
    else:
        resp.raise_for_status.return_value = None
    return resp


class _AsyncClientCM:
    """Async context manager wrapper around a MagicMock-backed client."""

    def __init__(self, client_mock: MagicMock):
        self._client = client_mock

    async def __aenter__(self):
        return self._client

    async def __aexit__(self, *exc_info):
        return False


def _patch_httpx_client(async_client_mock: MagicMock):
    """Return a patch context that makes ``httpx.AsyncClient(...)`` yield
    ``async_client_mock`` when used with ``async with``.

    We need a callable that accepts ``timeout=...`` and returns the CM.
    ``MagicMock`` by default does this, but its ``return_value`` would
    need to be the CM. Using a small factory keeps the intent explicit.
    """
    cm = _AsyncClientCM(async_client_mock)
    factory = MagicMock(return_value=cm)
    return patch(
        "llmcore.providers.vllm_provider.httpx.AsyncClient",
        factory,
    )


class TestGetModelsDetails:
    @pytest.mark.asyncio
    async def test_parses_max_model_len(self, provider_no_init):
        payload = {
            "object": "list",
            "data": [
                {
                    "id": "meta-llama/Llama-3.1-8B-Instruct",
                    "object": "model",
                    "owned_by": "vllm",
                    "created": 1712000000,
                    "max_model_len": 131072,
                    "root": "meta-llama/Llama-3.1-8B-Instruct",
                    "parent": None,
                }
            ],
        }
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=_mock_httpx_response(payload))

        with _patch_httpx_client(mock_client):
            result = await provider_no_init.get_models_details()

        assert len(result) == 1
        md = result[0]
        assert md.id == "meta-llama/Llama-3.1-8B-Instruct"
        assert md.context_length == 131072
        assert md.provider_name == "vllm"
        assert md.supports_tools is True
        assert md.supports_vision is False
        assert md.model_type == "chat"
        assert md.metadata["max_model_len"] == 131072
        # Cache side-effect
        assert provider_no_init._vllm_model_cache["meta-llama/Llama-3.1-8B-Instruct"] == 131072

    @pytest.mark.asyncio
    async def test_missing_max_model_len_falls_back(self, provider_no_init):
        payload = {
            "object": "list",
            "data": [
                {
                    "id": "some/ancient-vllm-model",
                    "object": "model",
                    "owned_by": "vllm",
                    "created": 0,
                    # No max_model_len — older vLLM builds
                }
            ],
        }
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=_mock_httpx_response(payload))

        with _patch_httpx_client(mock_client):
            result = await provider_no_init.get_models_details()

        assert result[0].context_length == 4096
        assert "some/ancient-vllm-model" not in provider_no_init._vllm_model_cache

    @pytest.mark.asyncio
    async def test_string_max_model_len_is_coerced(self, provider_no_init):
        """Custom proxies sometimes emit max_model_len as a string."""
        payload = {
            "data": [
                {"id": "proxied/model", "max_model_len": "32768"},
            ]
        }
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=_mock_httpx_response(payload))

        with _patch_httpx_client(mock_client):
            result = await provider_no_init.get_models_details()

        assert result[0].context_length == 32768
        assert provider_no_init._vllm_model_cache["proxied/model"] == 32768

    @pytest.mark.asyncio
    async def test_http_status_error_wraps_as_provider_error(self, provider_no_init):
        from llmcore.exceptions import ProviderError

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=_mock_httpx_response({}, status_code=500))

        with _patch_httpx_client(mock_client):
            with pytest.raises(ProviderError, match="HTTP 500"):
                await provider_no_init.get_models_details()

    @pytest.mark.asyncio
    async def test_transport_error_wraps_as_provider_error(self, provider_no_init):
        import httpx

        from llmcore.exceptions import ProviderError

        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

        with _patch_httpx_client(mock_client):
            with pytest.raises(ProviderError, match="Transport error"):
                await provider_no_init.get_models_details()

    @pytest.mark.asyncio
    async def test_authorization_header_set_when_api_key_real(self, provider_no_init):
        provider_no_init.api_key = "sk-real-token"
        payload = {"data": []}
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=_mock_httpx_response(payload))

        with _patch_httpx_client(mock_client):
            await provider_no_init.get_models_details()

        call_kwargs = mock_client.get.await_args.kwargs
        assert call_kwargs["headers"]["Authorization"] == "Bearer sk-real-token"

    @pytest.mark.asyncio
    async def test_authorization_header_omitted_when_empty(self, provider_no_init):
        assert provider_no_init.api_key == "EMPTY"
        payload = {"data": []}
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=_mock_httpx_response(payload))

        with _patch_httpx_client(mock_client):
            await provider_no_init.get_models_details()

        call_kwargs = mock_client.get.await_args.kwargs
        assert "Authorization" not in call_kwargs["headers"]

    @pytest.mark.asyncio
    async def test_empty_data_list(self, provider_no_init):
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=_mock_httpx_response({"data": []}))

        with _patch_httpx_client(mock_client):
            result = await provider_no_init.get_models_details()
        assert result == []

    @pytest.mark.asyncio
    async def test_malformed_entries_skipped(self, provider_no_init):
        payload = {
            "data": [
                "not-a-dict",
                {},  # no id
                {"id": 12345},  # id is not a string
                {"id": "good/model", "max_model_len": 8192},
            ]
        }
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=_mock_httpx_response(payload))

        with _patch_httpx_client(mock_client):
            result = await provider_no_init.get_models_details()

        assert [m.id for m in result] == ["good/model"]


# ---------------------------------------------------------------------------
# Context length resolution
# ---------------------------------------------------------------------------


class TestGetMaxContextLength:
    def test_uses_cache_when_populated(self, provider_no_init):
        provider_no_init._vllm_model_cache["my/model"] = 262144
        assert provider_no_init.get_max_context_length("my/model") == 262144

    def test_default_model_uses_cache(self, provider_no_init):
        provider_no_init.default_model = "cached/model"
        provider_no_init._vllm_model_cache["cached/model"] = 65536
        assert provider_no_init.get_max_context_length() == 65536

    def test_uses_fallback_config_value(self, provider_no_init):
        provider_no_init._fallback_context_length = 16384
        assert provider_no_init.get_max_context_length("uncached/model") == 16384

    def test_cache_wins_over_fallback(self, provider_no_init):
        provider_no_init._fallback_context_length = 16384
        provider_no_init._vllm_model_cache["x/y"] = 999
        assert provider_no_init.get_max_context_length("x/y") == 999

    def test_last_resort_4096_with_warning(self, provider_no_init, caplog):
        with caplog.at_level("WARNING"):
            result = provider_no_init.get_max_context_length("unknown/model")
        assert result == 4096
        assert any(
            "unknown/model" in rec.message and "4096" in rec.message for rec in caplog.records
        )


# ---------------------------------------------------------------------------
# Supported parameters
# ---------------------------------------------------------------------------


class TestGetSupportedParameters:
    def test_adds_vllm_specific_keys(self, provider_no_init):
        params = provider_no_init.get_supported_parameters()
        for key in (
            "structured_outputs",
            "guided_json",
            "guided_regex",
            "guided_choice",
            "guided_grammar",
            "prompt_logprobs",
            "min_p",
            "top_k",
            "repetition_penalty",
            "kv_transfer_params",
        ):
            assert key in params, f"{key} missing from supported parameters"

    def test_declares_extra_body(self, provider_no_init):
        params = provider_no_init.get_supported_parameters()
        assert "extra_body" in params
        assert params["extra_body"].get("type") == "object"

    def test_removes_openai_only_keys(self, provider_no_init):
        params = provider_no_init.get_supported_parameters()
        for key in _OPENAI_ONLY_KEYS:
            assert key not in params, (
                f"{key} should be stripped — it is OpenAI-only and vLLM ignores or rejects it"
            )

    def test_preserves_standard_openai_keys(self, provider_no_init):
        params = provider_no_init.get_supported_parameters()
        for key in (
            "temperature",
            "top_p",
            "max_tokens",
            "stop",
            "seed",
            "response_format",
            "logprobs",
            "top_logprobs",
            "tools",  # -- not in parent schema but accepted via signature
        ):
            # tools is part of the signature, not get_supported_parameters
            if key == "tools":
                continue
            assert key in params, f"{key} should still be supported"


# ---------------------------------------------------------------------------
# chat_completion — extra_body repackaging
# ---------------------------------------------------------------------------


class TestChatCompletionRepackaging:
    @pytest.mark.asyncio
    async def test_vllm_keys_moved_to_extra_body(self, provider_no_init):
        """vLLM-specific kwargs should land inside extra_body, not at top level."""
        parent_mock = AsyncMock(return_value={"choices": []})
        messages = [Message(role=LLMCoreRole.USER, content="hi")]

        with patch(
            "llmcore.providers.vllm_provider.OpenAIProvider.chat_completion",
            new=parent_mock,
        ):
            await provider_no_init.chat_completion(
                messages,
                temperature=0.5,
                structured_outputs={"json": {"type": "object"}},
                prompt_logprobs=3,
                min_p=0.05,
                top_k=40,
            )

        call_kwargs = parent_mock.await_args.kwargs
        assert call_kwargs["temperature"] == 0.5
        # vLLM-specific keys should NOT remain at top level
        assert "structured_outputs" not in call_kwargs
        assert "prompt_logprobs" not in call_kwargs
        assert "min_p" not in call_kwargs
        assert "top_k" not in call_kwargs
        # They should be nested under extra_body
        extra = call_kwargs["extra_body"]
        assert extra["structured_outputs"] == {"json": {"type": "object"}}
        assert extra["prompt_logprobs"] == 3
        assert extra["min_p"] == 0.05
        assert extra["top_k"] == 40

    @pytest.mark.asyncio
    async def test_caller_supplied_extra_body_preserved_and_merged(self, provider_no_init):
        parent_mock = AsyncMock(return_value={"choices": []})
        messages = [Message(role=LLMCoreRole.USER, content="hi")]

        with patch(
            "llmcore.providers.vllm_provider.OpenAIProvider.chat_completion",
            new=parent_mock,
        ):
            await provider_no_init.chat_completion(
                messages,
                extra_body={"my_custom_field": "preserve-me"},
                structured_outputs={"json_object": True},
            )

        extra = parent_mock.await_args.kwargs["extra_body"]
        assert extra["my_custom_field"] == "preserve-me"
        assert extra["structured_outputs"] == {"json_object": True}

    @pytest.mark.asyncio
    async def test_no_vllm_keys_means_no_extra_body_added(self, provider_no_init):
        parent_mock = AsyncMock(return_value={"choices": []})
        messages = [Message(role=LLMCoreRole.USER, content="hi")]

        with patch(
            "llmcore.providers.vllm_provider.OpenAIProvider.chat_completion",
            new=parent_mock,
        ):
            await provider_no_init.chat_completion(messages, temperature=0.7)

        assert "extra_body" not in parent_mock.await_args.kwargs

    @pytest.mark.asyncio
    async def test_tools_and_tool_choice_pass_through_cleanly(self, provider_no_init):
        from llmcore.models import Tool

        parent_mock = AsyncMock(return_value={"choices": []})
        messages = [Message(role=LLMCoreRole.USER, content="hi")]
        tool = Tool(name="get_weather", description="x", parameters={})

        with patch(
            "llmcore.providers.vllm_provider.OpenAIProvider.chat_completion",
            new=parent_mock,
        ):
            await provider_no_init.chat_completion(messages, tools=[tool], tool_choice="auto")

        positional, kwargs = parent_mock.await_args.args, parent_mock.await_args.kwargs
        assert positional[0] == messages
        assert kwargs["tools"] == [tool]
        assert kwargs["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_all_vllm_extra_body_keys_are_recognised(self, provider_no_init):
        """Every key in _VLLM_EXTRA_BODY_KEYS must be moved into extra_body."""
        parent_mock = AsyncMock(return_value={"choices": []})
        messages = [Message(role=LLMCoreRole.USER, content="hi")]
        kwargs = {key: "dummy" for key in _VLLM_EXTRA_BODY_KEYS}

        with patch(
            "llmcore.providers.vllm_provider.OpenAIProvider.chat_completion",
            new=parent_mock,
        ):
            await provider_no_init.chat_completion(messages, **kwargs)

        call_kwargs = parent_mock.await_args.kwargs
        for key in _VLLM_EXTRA_BODY_KEYS:
            assert key not in call_kwargs, f"{key} leaked to top level"
        assert set(call_kwargs["extra_body"].keys()) == set(_VLLM_EXTRA_BODY_KEYS)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


class TestTokenCounting:
    @pytest.mark.asyncio
    async def test_count_tokens_empty(self, provider_no_init):
        assert await provider_no_init.count_tokens("") == 0

    @pytest.mark.asyncio
    async def test_count_tokens_short(self, provider_no_init):
        # Character approximation: ceil(len/4)
        n = await provider_no_init.count_tokens("hello")
        assert n == 2  # (5 + 3) // 4 == 2

    @pytest.mark.asyncio
    async def test_count_tokens_does_not_use_encoding(self, provider_no_init):
        """count_tokens must not call self._encoding (it is None for vLLM)."""
        fake_enc = MagicMock()
        provider_no_init._encoding = fake_enc
        # Even if the parent would use it, our override doesn't consult _encoding
        n = await provider_no_init.count_tokens("abcdefgh")
        assert n == 2  # (8 + 3) // 4 == 2
        fake_enc.encode.assert_not_called()

    @pytest.mark.asyncio
    async def test_count_message_tokens_empty_list(self, provider_no_init):
        assert await provider_no_init.count_message_tokens([]) == 0

    @pytest.mark.asyncio
    async def test_count_message_tokens_rough_scale(self, provider_no_init):
        """Counts should grow monotonically with input size."""
        short = [Message(role=LLMCoreRole.USER, content="hi")]
        long = [Message(role=LLMCoreRole.USER, content="hi" * 100)]
        assert await provider_no_init.count_message_tokens(
            long
        ) > await provider_no_init.count_message_tokens(short)


# ---------------------------------------------------------------------------
# Manager registration
# ---------------------------------------------------------------------------


class TestManagerRegistration:
    def test_provider_in_provider_map(self):
        from llmcore.providers.manager import PROVIDER_MAP

        assert PROVIDER_MAP.get("vllm") is VLLMProvider

    def test_provider_not_in_openai_compatible_defaults(self):
        """vLLM has no canonical URL, so it must NOT be in the defaults table.
        If it were, ProviderManager would inject a wrong base_url."""
        from llmcore.providers.manager import _OPENAI_COMPATIBLE_DEFAULTS

        assert "vllm" not in _OPENAI_COMPATIBLE_DEFAULTS
