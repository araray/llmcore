# tests/providers/test_kimi_provider.py
"""
Tests for the Kimi (Moonshot AI) provider implementation.

Covers:
- Initialization and configuration (api key, base_url, thinking/keep defaults).
- Model-family capability gating (thinking param, keep, fixed sampling).
- Thinking-mode resolution (default, dict/str/bool overrides, keep handling).
- Message payload building (multimodal image/video, content_parts, tool_calls,
  reasoning_content preservation, partial mode, tool results).
- Chat completion request shaping (thinking via extra_body, sampling stripping
  for k2.x, tools/tool_choice, stream_options) using a mocked OpenAI client.
- Response extraction (content, reasoning, tool calls, usage/cached, finish).
- Streaming delta extraction (content + reasoning).
- Token counting via the estimate endpoint (mocked httpx) with fallback.
- Auxiliary REST helpers (balance, file upload) and model discovery.
- Error mapping (context length, 401, 404, 429).
- Context-length lookup and supported-parameter validation.
- close() lifecycle.

All tests are mock-based; no live Moonshot endpoint is contacted.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmcore.exceptions import ConfigError
from llmcore.models import Message, Role, Tool

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_CONFIG: dict[str, Any] = {
    "api_key": "sk-test-key-000",
    "default_model": "kimi-k2.6",
    "timeout": 30,
    "thinking": "enabled",
    "thinking_keep": "all",
}


def _make_provider(cfg: dict[str, Any] | None = None):
    """Construct a KimiProvider with both clients mocked out."""
    with patch("llmcore.providers.kimi_provider.AsyncOpenAI") as MockClient, patch(
        "llmcore.providers.kimi_provider.httpx"
    ) as mock_httpx:
        MockClient.return_value = MagicMock()
        mock_httpx.AsyncClient.return_value = MagicMock()
        from llmcore.providers.kimi_provider import KimiProvider

        p = KimiProvider((cfg or MINIMAL_CONFIG).copy(), log_raw_payloads=False)
        return p


@pytest.fixture
def provider():
    return _make_provider()


@pytest.fixture
def v1_provider():
    cfg = MINIMAL_CONFIG.copy()
    cfg["default_model"] = "moonshot-v1-128k"
    return _make_provider(cfg)


# ---------------------------------------------------------------------------
# Capability gating (pure functions)
# ---------------------------------------------------------------------------


class TestCapabilityGating:
    def test_model_family(self):
        from llmcore.providers.kimi_provider import _model_family

        assert _model_family("kimi-k2.6") == "k2.6"
        assert _model_family("kimi-k2.5") == "k2.5"
        assert _model_family("kimi-k2-thinking") == "k2-thinking"
        assert _model_family("kimi-k2-thinking-turbo") == "k2-thinking"
        assert _model_family("kimi-k2-0905-preview") == "k2"
        assert _model_family("moonshot-v1-128k") == "moonshot-v1"
        assert _model_family("moonshot-v1-32k-vision-preview") == "moonshot-v1"
        assert _model_family("something-else") == "unknown"

    def test_thinking_param_gate(self):
        from llmcore.providers.kimi_provider import _model_supports_thinking_param

        assert _model_supports_thinking_param("kimi-k2.6") is True
        assert _model_supports_thinking_param("kimi-k2.5") is True
        assert _model_supports_thinking_param("kimi-k2-thinking") is False
        assert _model_supports_thinking_param("moonshot-v1-128k") is False

    def test_keep_gate(self):
        from llmcore.providers.kimi_provider import _model_supports_thinking_keep

        assert _model_supports_thinking_keep("kimi-k2.6") is True
        assert _model_supports_thinking_keep("kimi-k2.5") is False

    def test_fixed_sampling_gate(self):
        from llmcore.providers.kimi_provider import _model_has_fixed_sampling

        assert _model_has_fixed_sampling("kimi-k2.6") is True
        assert _model_has_fixed_sampling("kimi-k2.5") is True
        assert _model_has_fixed_sampling("kimi-k2-thinking") is True
        assert _model_has_fixed_sampling("moonshot-v1-128k") is False


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    def test_basic_init(self, provider):
        assert provider.get_name() == "kimi"
        assert provider.default_model == "kimi-k2.6"
        assert provider._default_thinking == "enabled"
        assert provider._default_keep == "all"
        assert provider._base_url == "https://api.moonshot.ai/v1"

    def test_origin_strips_v1(self, provider):
        assert provider._origin() == "https://api.moonshot.ai"

    def test_instance_name_override(self):
        cfg = MINIMAL_CONFIG.copy()
        cfg["_instance_name"] = "kimi_prod"
        p = _make_provider(cfg)
        assert p.get_name() == "kimi_prod"

    def test_keep_null_default(self):
        cfg = MINIMAL_CONFIG.copy()
        cfg.pop("thinking_keep")
        p = _make_provider(cfg)
        assert p._default_keep is None

    def test_invalid_keep_defaults_null(self):
        cfg = MINIMAL_CONFIG.copy()
        cfg["thinking_keep"] = "bogus"
        p = _make_provider(cfg)
        assert p._default_keep is None

    def test_invalid_thinking_defaults_enabled(self):
        cfg = MINIMAL_CONFIG.copy()
        cfg["thinking"] = "bogus"
        p = _make_provider(cfg)
        assert p._default_thinking == "enabled"

    def test_custom_base_url(self):
        cfg = MINIMAL_CONFIG.copy()
        cfg["base_url"] = "https://api.moonshot.cn/v1"
        p = _make_provider(cfg)
        assert p._base_url == "https://api.moonshot.cn/v1"
        assert p._origin() == "https://api.moonshot.cn"

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
        with patch("llmcore.providers.kimi_provider.AsyncOpenAI"), patch(
            "llmcore.providers.kimi_provider.httpx"
        ):
            from llmcore.providers.kimi_provider import KimiProvider

            with pytest.raises(ConfigError):
                KimiProvider({"default_model": "kimi-k2.6"})

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("MOONSHOT_API_KEY", "sk-env-key")
        with patch("llmcore.providers.kimi_provider.AsyncOpenAI"), patch(
            "llmcore.providers.kimi_provider.httpx"
        ):
            from llmcore.providers.kimi_provider import KimiProvider

            p = KimiProvider({"default_model": "kimi-k2.6"})
            assert p._api_key == "sk-env-key"


# ---------------------------------------------------------------------------
# Thinking resolution
# ---------------------------------------------------------------------------


class TestThinkingResolution:
    def test_default_enabled_with_keep(self, provider):
        assert provider._resolve_thinking("kimi-k2.6", {}) == {
            "type": "enabled",
            "keep": "all",
        }

    def test_non_thinking_model_returns_none(self, provider):
        assert provider._resolve_thinking("moonshot-v1-128k", {}) is None

    def test_override_disabled_string(self, provider):
        assert provider._resolve_thinking("kimi-k2.6", {"thinking": "disabled"}) == {
            "type": "disabled"
        }

    def test_override_bool_false(self, provider):
        assert provider._resolve_thinking("kimi-k2.6", {"thinking": False}) == {
            "type": "disabled"
        }

    def test_override_dict(self, provider):
        out = provider._resolve_thinking(
            "kimi-k2.6", {"thinking": {"type": "enabled", "keep": "all"}}
        )
        assert out == {"type": "enabled", "keep": "all"}

    def test_keep_dropped_for_k25(self, provider):
        out = provider._resolve_thinking(
            "kimi-k2.5", {"thinking": {"type": "enabled", "keep": "all"}}
        )
        assert out == {"type": "enabled"}  # keep not allowed on k2.5

    def test_keep_dropped_when_disabled(self, provider):
        out = provider._resolve_thinking(
            "kimi-k2.6", {"thinking": {"type": "disabled", "keep": "all"}}
        )
        assert out == {"type": "disabled"}  # keep only meaningful with enabled

    def test_explicit_keep_kwarg(self, provider):
        out = provider._resolve_thinking(
            "kimi-k2.6", {"thinking": "enabled", "thinking_keep": "all"}
        )
        assert out == {"type": "enabled", "keep": "all"}

    def test_pops_kwargs(self, provider):
        kw = {"thinking": "disabled", "thinking_keep": "all"}
        provider._resolve_thinking("kimi-k2.6", kw)
        assert "thinking" not in kw and "thinking_keep" not in kw


# ---------------------------------------------------------------------------
# Message payload building
# ---------------------------------------------------------------------------


class TestMessagePayload:
    def test_basic_user(self, provider):
        msg = Message(role=Role.USER, content="hi")
        assert provider._build_message_payload(msg) == {"role": "user", "content": "hi"}

    def test_inline_images_data_uri_and_msref(self, provider):
        msg = Message(
            role=Role.USER,
            content="Describe",
            metadata={"inline_images": ["data:image/png;base64,AAA", "ms://file-1"]},
        )
        out = provider._build_message_payload(msg)
        assert out["role"] == "user"
        assert out["content"][0] == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AAA"},
        }
        assert out["content"][1] == {
            "type": "image_url",
            "image_url": {"url": "ms://file-1"},
        }
        assert out["content"][-1] == {"type": "text", "text": "Describe"}

    def test_inline_video(self, provider):
        msg = Message(
            role=Role.USER,
            content="What happens?",
            metadata={"inline_videos": ["ms://vid-1"]},
        )
        out = provider._build_message_payload(msg)
        assert out["content"][0] == {
            "type": "video_url",
            "video_url": {"url": "ms://vid-1"},
        }

    def test_inline_image_dict_url(self, provider):
        msg = Message(
            role=Role.USER,
            content="x",
            metadata={"inline_images": [{"url": "ms://f"}]},
        )
        out = provider._build_message_payload(msg)
        assert out["content"][0] == {"type": "image_url", "image_url": {"url": "ms://f"}}

    def test_content_parts_passthrough(self, provider):
        parts = [{"type": "text", "text": "pre"}]
        msg = Message(role=Role.USER, content="ignored", metadata={"content_parts": parts})
        out = provider._build_message_payload(msg)
        assert out["content"] is parts

    def test_tool_result_message(self, provider):
        msg = Message(role=Role.TOOL, content="42", tool_call_id="call_1")
        out = provider._build_message_payload(msg)
        assert out["role"] == "tool"
        assert out["tool_call_id"] == "call_1"

    def test_assistant_tool_calls_and_reasoning(self, provider):
        msg = Message(
            role=Role.ASSISTANT,
            content="",
            metadata={
                "tool_calls": [
                    {"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ],
                "reasoning_content": "because",
            },
        )
        out = provider._build_message_payload(msg)
        assert out["content"] is None
        assert out["tool_calls"][0]["id"] == "t1"
        assert out["reasoning_content"] == "because"

    def test_assistant_partial_mode(self, provider):
        msg = Message(role=Role.ASSISTANT, content="The story so far", metadata={"partial": True})
        out = provider._build_message_payload(msg)
        assert out["partial"] is True

    def test_name_field(self, provider):
        msg = Message(role=Role.USER, content="hi", metadata={"name": "alice"})
        out = provider._build_message_payload(msg)
        assert out["name"] == "alice"


# ---------------------------------------------------------------------------
# Chat completion request shaping (mocked client)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestChatCompletion:
    async def _call(self, provider, model, **kwargs):
        captured: dict[str, Any] = {}

        async def fake_create(**call_kwargs):
            captured.update(call_kwargs)
            resp = MagicMock()
            resp.model_dump.return_value = {"choices": [{"message": {"content": "ok"}}]}
            return resp

        provider._client.chat.completions.create = AsyncMock(side_effect=fake_create)
        ctx = [Message(role=Role.USER, content="hello")]
        await provider.chat_completion(ctx, model=model, **kwargs)
        return captured

    async def test_thinking_injected_via_extra_body(self, provider):
        captured = await self._call(provider, "kimi-k2.6")
        assert captured["extra_body"]["thinking"] == {"type": "enabled", "keep": "all"}

    async def test_no_thinking_for_v1(self, provider):
        captured = await self._call(provider, "moonshot-v1-128k")
        assert "extra_body" not in captured or "thinking" not in captured.get("extra_body", {})

    async def test_sampling_stripped_for_k26(self, provider):
        captured = await self._call(provider, "kimi-k2.6", temperature=0.2, top_p=0.5)
        assert "temperature" not in captured
        assert "top_p" not in captured

    async def test_sampling_allowed_for_v1(self, provider):
        captured = await self._call(provider, "moonshot-v1-128k", temperature=0.2)
        assert captured["temperature"] == 0.2

    async def test_prompt_cache_key_in_extra_body(self, provider):
        captured = await self._call(provider, "kimi-k2.6", prompt_cache_key="sess-1")
        assert captured["extra_body"]["prompt_cache_key"] == "sess-1"

    async def test_tools_and_tool_choice(self, provider):
        tool = Tool(name="get_weather", description="w", parameters={"type": "object"})
        captured = await self._call(provider, "kimi-k2.6", tools=[tool], tool_choice="auto")
        assert captured["tools"][0]["type"] == "function"
        assert captured["tools"][0]["function"]["name"] == "get_weather"
        assert captured["tool_choice"] == "auto"

    async def test_stream_options_injected(self, provider):
        captured = await self._call(provider, "kimi-k2.6", stream=True)
        assert captured["stream_options"] == {"include_usage": True}

    async def test_unsupported_param_raises(self, provider):
        ctx = [Message(role=Role.USER, content="hi")]
        with pytest.raises(ValueError):
            await provider.chat_completion(ctx, model="kimi-k2.6", bogus_param=1)


# ---------------------------------------------------------------------------
# Response extraction
# ---------------------------------------------------------------------------


class TestResponseExtraction:
    RESP = {
        "choices": [
            {
                "message": {
                    "content": "Hello",
                    "reasoning_content": "let me think",
                    "tool_calls": [
                        {
                            "id": "a",
                            "type": "function",
                            "function": {"name": "g", "arguments": '{"x": 1}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
            "cached_tokens": 2,
            "completion_tokens_details": {"reasoning_tokens": 2},
        },
    }

    def test_content(self, provider):
        assert provider.extract_response_content(self.RESP) == "Hello"

    def test_reasoning(self, provider):
        assert provider.extract_reasoning_content(self.RESP) == "let me think"

    def test_reasoning_absent(self, provider):
        assert provider.extract_reasoning_content({"choices": [{"message": {}}]}) is None

    def test_tool_calls(self, provider):
        calls = provider.extract_tool_calls(self.RESP)
        assert len(calls) == 1
        assert calls[0].name == "g"
        assert calls[0].arguments == {"x": 1}

    def test_tool_calls_invalid_json(self, provider):
        resp = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "a", "type": "function", "function": {"name": "g", "arguments": "{bad"}}
                        ]
                    }
                }
            ]
        }
        calls = provider.extract_tool_calls(resp)
        assert calls[0].arguments == {"_raw": "{bad"}

    def test_usage(self, provider):
        u = provider.extract_usage_details(self.RESP)
        assert u["cached_tokens"] == 2
        assert u["reasoning_tokens"] == 2

    def test_finish_reason(self, provider):
        assert provider.extract_finish_reason(self.RESP) == "tool_calls"

    def test_empty_response(self, provider):
        assert provider.extract_response_content({"choices": []}) == ""

    def test_delta_content(self, provider):
        chunk = {"choices": [{"delta": {"content": "hi"}}]}
        assert provider.extract_delta_content(chunk) == "hi"

    def test_delta_reasoning(self, provider):
        chunk = {"choices": [{"delta": {"reasoning_content": "mm"}}]}
        assert provider.extract_delta_reasoning_content(chunk) == "mm"


# ---------------------------------------------------------------------------
# Token counting / estimate endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTokenCounting:
    async def test_estimate_endpoint_used(self, provider):
        resp = MagicMock()
        resp.json.return_value = {"data": {"total_tokens": 42}}
        resp.raise_for_status = MagicMock()
        provider._http.post = AsyncMock(return_value=resp)
        n = await provider.count_tokens("hello world", model="kimi-k2.6")
        assert n == 42

    async def test_count_message_tokens_estimate(self, provider):
        resp = MagicMock()
        resp.json.return_value = {"data": {"total_tokens": 100}}
        resp.raise_for_status = MagicMock()
        provider._http.post = AsyncMock(return_value=resp)
        n = await provider.count_message_tokens(
            [Message(role=Role.USER, content="hi")], model="kimi-k2.6"
        )
        assert n == 100

    async def test_estimate_error_field_returns_none(self, provider):
        resp = MagicMock()
        resp.json.return_value = {"error": {"message": "bad"}}
        resp.raise_for_status = MagicMock()
        provider._http.post = AsyncMock(return_value=resp)
        out = await provider.estimate_tokens([{"role": "user", "content": "x"}])
        assert out is None

    async def test_count_tokens_empty(self, provider):
        assert await provider.count_tokens("") == 0

    async def test_fallback_heuristic_when_no_http(self, provider):
        provider._http = None
        provider._encoding = None
        n = await provider.count_tokens("abcdefghij")
        assert n >= 1


# ---------------------------------------------------------------------------
# Auxiliary REST helpers + discovery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAuxiliary:
    async def test_check_balance(self, provider):
        resp = MagicMock()
        resp.json.return_value = {"data": {"available_balance": 49.5}}
        resp.raise_for_status = MagicMock()
        provider._http.get = AsyncMock(return_value=resp)
        bal = await provider.check_balance()
        assert bal["available_balance"] == 49.5

    async def test_upload_file(self, provider):
        resp = MagicMock()
        resp.json.return_value = {"id": "file-xyz"}
        resp.raise_for_status = MagicMock()
        provider._http.post = AsyncMock(return_value=resp)
        fid = await provider.upload_file(b"\x00\x01", "v.mp4", purpose="video")
        assert fid == "file-xyz"

    async def test_get_models_details_rich(self, provider):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "data": [
                {
                    "id": "kimi-k2.6",
                    "owned_by": "moonshot",
                    "created": 1700000000,
                    "context_length": 262144,
                    "supports_image_in": True,
                    "supports_video_in": True,
                    "supports_reasoning": True,
                },
                {
                    "id": "moonshot-v1-8k",
                    "context_length": 8192,
                    "supports_image_in": False,
                    "supports_video_in": False,
                    "supports_reasoning": False,
                },
            ]
        }
        provider._http.get = AsyncMock(return_value=resp)
        details = await provider.get_models_details()
        by_id = {d.id: d for d in details}
        assert by_id["kimi-k2.6"].context_length == 262144
        assert by_id["kimi-k2.6"].supports_vision is True
        assert by_id["kimi-k2.6"].supports_reasoning is True
        assert by_id["kimi-k2.6"].metadata["supports_video_in"] is True
        assert by_id["moonshot-v1-8k"].supports_vision is False


# ---------------------------------------------------------------------------
# Context length + supported parameters
# ---------------------------------------------------------------------------


class TestContextAndParams:
    def test_known_context(self, provider):
        assert provider.get_max_context_length("kimi-k2.6") == 262144
        assert provider.get_max_context_length("moonshot-v1-8k") == 8192
        assert provider.get_max_context_length("moonshot-v1-128k") == 131072

    def test_unknown_family_heuristic(self, provider):
        assert provider.get_max_context_length("kimi-k2.7-future") == 262144

    def test_supported_parameters_includes_thinking(self, provider):
        params = provider.get_supported_parameters("kimi-k2.6")
        assert "thinking" in params
        assert "thinking_keep" in params
        assert "prompt_cache_key" in params


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestLifecycle:
    async def test_close(self, provider):
        provider._client.close = AsyncMock()
        provider._http.aclose = AsyncMock()
        await provider.close()
        assert provider._client is None
        assert provider._http is None
