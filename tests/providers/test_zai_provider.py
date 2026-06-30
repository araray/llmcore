# tests/providers/test_zai_provider.py
"""
Tests for the Z.ai (GLM) provider implementation.

Covers:
- Initialization and configuration
- Thinking mode parameter resolution
- Open-interval temperature/top_p clamping
- Message payload building (including reasoning_content preservation)
- Chat completion (mocked), including extra_body routing of platform extras
- Response extraction (content, reasoning, tools, usage/cache)
- Streaming delta extraction
- Embeddings (mocked)
- Token counting
- Context length lookup
- Provider registration / aliases
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmcore.models import Message, Role, Tool

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_CONFIG: dict[str, Any] = {
    "api_key": "test-zai-key-000",
    "default_model": "glm-5.2",
    "timeout": 30,
    "thinking": "enabled",
    "reasoning_effort": "high",
}


@pytest.fixture
def provider():
    """Create a ZaiProvider with mocked OpenAI client."""
    with patch("llmcore.providers.zai_provider.AsyncOpenAI") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance
        from llmcore.providers.zai_provider import ZaiProvider

        p = ZaiProvider(MINIMAL_CONFIG.copy(), log_raw_payloads=False)
        p._mock_client_cls = MockClient
        return p


@pytest.fixture
def disabled_thinking_provider():
    """Provider with thinking disabled by default."""
    cfg = MINIMAL_CONFIG.copy()
    cfg["thinking"] = "disabled"
    with patch("llmcore.providers.zai_provider.AsyncOpenAI"):
        from llmcore.providers.zai_provider import ZaiProvider

        return ZaiProvider(cfg, log_raw_payloads=False)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    def test_basic_init(self, provider):
        assert provider.get_name() == "zai"
        assert provider.default_model == "glm-5.2"
        assert provider.default_embedding_model == "embedding-3"
        assert provider._default_thinking == "enabled"
        assert provider._default_reasoning_effort == "high"

    def test_init_with_instance_name(self):
        cfg = MINIMAL_CONFIG.copy()
        cfg["_instance_name"] = "my-zai"
        with patch("llmcore.providers.zai_provider.AsyncOpenAI"):
            from llmcore.providers.zai_provider import ZaiProvider

            p = ZaiProvider(cfg)
            assert p.get_name() == "my-zai"

    def test_init_disabled_thinking(self, disabled_thinking_provider):
        assert disabled_thinking_provider._default_thinking == "disabled"

    def test_init_effort_passthrough(self):
        for raw in ["none", "minimal", "low", "medium", "high", "xhigh", "max"]:
            cfg = MINIMAL_CONFIG.copy()
            cfg["reasoning_effort"] = raw
            with patch("llmcore.providers.zai_provider.AsyncOpenAI"):
                from llmcore.providers.zai_provider import ZaiProvider

                p = ZaiProvider(cfg)
                assert p._default_reasoning_effort == raw

    def test_init_invalid_effort_defaults_high(self):
        cfg = MINIMAL_CONFIG.copy()
        cfg["reasoning_effort"] = "bogus"
        with patch("llmcore.providers.zai_provider.AsyncOpenAI"):
            from llmcore.providers.zai_provider import ZaiProvider

            assert ZaiProvider(cfg)._default_reasoning_effort == "high"

    def test_init_invalid_thinking_defaults_enabled(self):
        cfg = MINIMAL_CONFIG.copy()
        cfg["thinking"] = "bogus"
        with patch("llmcore.providers.zai_provider.AsyncOpenAI"):
            from llmcore.providers.zai_provider import ZaiProvider

            assert ZaiProvider(cfg)._default_thinking == "enabled"

    def test_init_missing_api_key_raises(self):
        cfg = {"default_model": "glm-5.2"}
        with patch("llmcore.providers.zai_provider.AsyncOpenAI"):
            with patch.dict("os.environ", {}, clear=True):
                from llmcore.exceptions import ConfigError
                from llmcore.providers.zai_provider import ZaiProvider

                with pytest.raises(ConfigError, match="API key not found"):
                    ZaiProvider(cfg)

    def test_init_api_key_from_env(self):
        cfg = {"default_model": "glm-5.2"}
        with patch("llmcore.providers.zai_provider.AsyncOpenAI"):
            with patch.dict("os.environ", {"ZAI_API_KEY": "env-key"}):
                from llmcore.providers.zai_provider import ZaiProvider

                p = ZaiProvider(cfg)
                assert p.default_model == "glm-5.2"

    def test_init_china_region_base_url(self):
        cfg = MINIMAL_CONFIG.copy()
        cfg["region"] = "china"
        with patch("llmcore.providers.zai_provider.AsyncOpenAI") as MockCls:
            from llmcore.providers.zai_provider import ZaiProvider

            ZaiProvider(cfg)
            _, kwargs = MockCls.call_args
            assert "bigmodel.cn" in kwargs["base_url"]

    def test_init_default_overseas_base_url(self):
        with patch("llmcore.providers.zai_provider.AsyncOpenAI") as MockCls:
            from llmcore.providers.zai_provider import ZaiProvider

            ZaiProvider(MINIMAL_CONFIG.copy())
            _, kwargs = MockCls.call_args
            assert kwargs["base_url"] == "https://api.z.ai/api/paas/v4"


# ---------------------------------------------------------------------------
# Thinking parameter resolution
# ---------------------------------------------------------------------------


class TestThinkingResolution:
    def test_default_thinking_enabled(self, provider):
        thinking, effort = provider._resolve_thinking_params({})
        assert thinking == {"type": "enabled"}
        assert effort == "high"

    def test_default_thinking_disabled(self, disabled_thinking_provider):
        thinking, effort = disabled_thinking_provider._resolve_thinking_params({})
        assert thinking == {"type": "disabled"}
        assert effort is None

    def test_override_thinking_string(self, provider):
        thinking, effort = provider._resolve_thinking_params({"thinking": "disabled"})
        assert thinking == {"type": "disabled"}
        assert effort is None

    def test_override_thinking_bool(self, provider):
        thinking, effort = provider._resolve_thinking_params({"thinking": False})
        assert thinking == {"type": "disabled"}
        thinking, effort = provider._resolve_thinking_params({"thinking": True})
        assert thinking == {"type": "enabled"}
        assert effort == "high"

    def test_override_effort(self, provider):
        kwargs: dict[str, Any] = {"reasoning_effort": "max"}
        _, effort = provider._resolve_thinking_params(kwargs)
        assert effort == "max"
        assert "reasoning_effort" not in kwargs

    def test_invalid_effort_falls_back_to_default(self, provider):
        _, effort = provider._resolve_thinking_params({"reasoning_effort": "ultra"})
        assert effort == "high"


# ---------------------------------------------------------------------------
# Open-interval clamping
# ---------------------------------------------------------------------------


class TestClamping:
    def test_clamp_upper_bound(self, provider):
        assert provider._clamp_open_interval(1.0) == 0.99
        assert provider._clamp_open_interval(2.5) == 0.99

    def test_clamp_lower_bound(self, provider):
        assert provider._clamp_open_interval(0) == 0.01
        assert provider._clamp_open_interval(-1) == 0.01

    def test_clamp_in_range(self, provider):
        assert provider._clamp_open_interval(0.7) == 0.7

    def test_clamp_non_numeric(self, provider):
        assert provider._clamp_open_interval("x") == "x"


# ---------------------------------------------------------------------------
# Message payload building
# ---------------------------------------------------------------------------


class TestMessagePayload:
    def test_basic_user_message(self, provider):
        payload = provider._build_message_payload(Message(role=Role.USER, content="Hello"))
        assert payload == {"role": "user", "content": "Hello"}

    def test_tool_message(self, provider):
        msg = Message(role=Role.TOOL, content='{"temp": 22}', tool_call_id="call_abc")
        payload = provider._build_message_payload(msg)
        assert payload["role"] == "tool"
        assert payload["tool_call_id"] == "call_abc"

    def test_assistant_with_reasoning_content(self, provider):
        msg = Message(
            role=Role.ASSISTANT,
            content="The answer is 42.",
            metadata={
                "reasoning_content": "Let me think...",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "calc", "arguments": "{}"}}
                ],
            },
        )
        payload = provider._build_message_payload(msg)
        assert payload["reasoning_content"] == "Let me think..."
        assert payload["tool_calls"] is not None

    def test_assistant_empty_content_with_tools(self, provider):
        msg = Message(
            role=Role.ASSISTANT,
            content="",
            metadata={
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ]
            },
        )
        payload = provider._build_message_payload(msg)
        assert payload["content"] is None
        assert "tool_calls" in payload


# ---------------------------------------------------------------------------
# Response extraction
# ---------------------------------------------------------------------------


class TestResponseExtraction:
    SAMPLE_RESPONSE: dict[str, Any] = {
        "id": "chatcmpl-test",
        "model": "glm-5.2",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "The capital of France is Paris.",
                    "reasoning_content": "The user is asking about geography...",
                },
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 30},
            "completion_tokens": 100,
            "total_tokens": 150,
            "completion_tokens_details": {"reasoning_tokens": 60},
        },
    }

    def test_extract_content(self, provider):
        assert (
            provider.extract_response_content(self.SAMPLE_RESPONSE)
            == "The capital of France is Paris."
        )

    def test_extract_reasoning_content(self, provider):
        assert (
            provider.extract_reasoning_content(self.SAMPLE_RESPONSE)
            == "The user is asking about geography..."
        )

    def test_extract_reasoning_content_absent(self, provider):
        assert provider.extract_reasoning_content({"choices": [{"message": {"content": "x"}}]}) is None

    def test_extract_usage_details(self, provider):
        usage = provider.extract_usage_details(self.SAMPLE_RESPONSE)
        assert usage["prompt_tokens"] == 50
        assert usage["cached_tokens"] == 30
        assert usage["completion_tokens"] == 100
        assert usage["reasoning_tokens"] == 60

    def test_extract_finish_reason(self, provider):
        assert provider.extract_finish_reason(self.SAMPLE_RESPONSE) == "stop"

    def test_extract_tool_calls(self, provider):
        resp = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Paris"}',
                                },
                            }
                        ]
                    }
                }
            ]
        }
        calls = provider.extract_tool_calls(resp)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {"location": "Paris"}

    def test_extract_tool_calls_invalid_json(self, provider):
        resp = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "c2", "type": "function", "function": {"name": "f", "arguments": "nope"}}
                        ]
                    }
                }
            ]
        }
        calls = provider.extract_tool_calls(resp)
        assert calls[0].arguments == {"_raw": "nope"}

    def test_extract_content_empty(self, provider):
        assert provider.extract_response_content({}) == ""
        assert provider.extract_response_content({"choices": []}) == ""


# ---------------------------------------------------------------------------
# Streaming delta extraction
# ---------------------------------------------------------------------------


class TestStreamingDelta:
    def test_extract_delta_content(self, provider):
        assert provider.extract_delta_content({"choices": [{"delta": {"content": "Hi"}}]}) == "Hi"

    def test_extract_delta_reasoning_content(self, provider):
        chunk = {"choices": [{"delta": {"reasoning_content": "thinking..."}}]}
        assert provider.extract_delta_reasoning_content(chunk) == "thinking..."

    def test_extract_delta_empty(self, provider):
        assert provider.extract_delta_content({}) == ""
        assert provider.extract_delta_content({"choices": []}) == ""


# ---------------------------------------------------------------------------
# Context length
# ---------------------------------------------------------------------------


class TestContextLength:
    def test_glm_5_2(self, provider):
        assert provider.get_max_context_length("glm-5.2") == 1_000_000

    def test_glm_4_6(self, provider):
        assert provider.get_max_context_length("glm-4.6") == 204_800

    def test_default_model(self, provider):
        assert provider.get_max_context_length() == 1_000_000

    def test_unknown_glm5_model(self, provider):
        assert provider.get_max_context_length("glm-5-future") == 1_000_000

    def test_completely_unknown_model(self, provider):
        assert provider.get_max_context_length("glm-future-xyz") == 131_072


# ---------------------------------------------------------------------------
# Supported parameters
# ---------------------------------------------------------------------------


class TestSupportedParameters:
    def test_includes_thinking_params(self, provider):
        params = provider.get_supported_parameters()
        assert "thinking" in params
        assert "reasoning_effort" in params

    def test_includes_platform_extras(self, provider):
        params = provider.get_supported_parameters()
        for key in ("do_sample", "request_id", "user_id", "watermark_enabled", "tool_stream"):
            assert key in params, f"Missing parameter: {key}"


# ---------------------------------------------------------------------------
# Chat completion (mocked)
# ---------------------------------------------------------------------------


class TestChatCompletion:
    @pytest.mark.asyncio
    async def test_basic_completion(self, provider):
        mock_resp = MagicMock()
        mock_resp.model_dump.return_value = {
            "choices": [{"message": {"content": "Paris"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        provider._client.chat.completions.create = AsyncMock(return_value=mock_resp)

        context = [Message(role=Role.USER, content="Capital of France?")]
        result = await provider.chat_completion(context)
        assert result["choices"][0]["message"]["content"] == "Paris"

        call_kwargs = provider._client.chat.completions.create.call_args
        extra = call_kwargs.kwargs.get("extra_body", {})
        assert extra.get("thinking") == {"type": "enabled"}
        assert extra.get("reasoning_effort") == "high"

    @pytest.mark.asyncio
    async def test_completion_thinking_disabled(self, disabled_thinking_provider):
        mock_resp = MagicMock()
        mock_resp.model_dump.return_value = {"choices": [{"message": {"content": "OK"}}], "usage": {}}
        disabled_thinking_provider._client.chat.completions.create = AsyncMock(return_value=mock_resp)

        await disabled_thinking_provider.chat_completion([Message(role=Role.USER, content="Hi")])
        call_kwargs = disabled_thinking_provider._client.chat.completions.create.call_args
        extra = call_kwargs.kwargs.get("extra_body", {})
        assert extra.get("thinking") == {"type": "disabled"}
        assert "reasoning_effort" not in extra

    @pytest.mark.asyncio
    async def test_per_request_thinking_override(self, provider):
        mock_resp = MagicMock()
        mock_resp.model_dump.return_value = {"choices": [{"message": {"content": "OK"}}], "usage": {}}
        provider._client.chat.completions.create = AsyncMock(return_value=mock_resp)

        await provider.chat_completion(
            [Message(role=Role.USER, content="Hi")], thinking=False, reasoning_effort="max"
        )
        extra = provider._client.chat.completions.create.call_args.kwargs.get("extra_body", {})
        assert extra.get("thinking") == {"type": "disabled"}
        assert "reasoning_effort" not in extra

    @pytest.mark.asyncio
    async def test_temperature_clamped(self, provider):
        mock_resp = MagicMock()
        mock_resp.model_dump.return_value = {"choices": [{"message": {"content": "OK"}}], "usage": {}}
        provider._client.chat.completions.create = AsyncMock(return_value=mock_resp)

        await provider.chat_completion(
            [Message(role=Role.USER, content="Hi")], temperature=1.0, top_p=0.0
        )
        call_kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("temperature") == 0.99
        assert call_kwargs.get("top_p") == 0.01

    @pytest.mark.asyncio
    async def test_platform_extras_routed_to_extra_body(self, provider):
        mock_resp = MagicMock()
        mock_resp.model_dump.return_value = {"choices": [{"message": {"content": "OK"}}], "usage": {}}
        provider._client.chat.completions.create = AsyncMock(return_value=mock_resp)

        await provider.chat_completion(
            [Message(role=Role.USER, content="Hi")],
            do_sample=False,
            request_id="req-1",
            watermark_enabled=True,
        )
        call_kwargs = provider._client.chat.completions.create.call_args.kwargs
        extra = call_kwargs.get("extra_body", {})
        assert extra.get("do_sample") is False
        assert extra.get("request_id") == "req-1"
        assert extra.get("watermark_enabled") is True
        # These must NOT leak as top-level OpenAI params
        assert "do_sample" not in call_kwargs
        assert "request_id" not in call_kwargs

    @pytest.mark.asyncio
    async def test_tools_payload(self, provider):
        mock_resp = MagicMock()
        mock_resp.model_dump.return_value = {"choices": [{"message": {"content": "OK"}}], "usage": {}}
        provider._client.chat.completions.create = AsyncMock(return_value=mock_resp)

        tool = Tool(
            name="get_weather",
            description="Get weather",
            parameters={"type": "object", "properties": {"location": {"type": "string"}}},
        )
        await provider.chat_completion(
            [Message(role=Role.USER, content="Weather?")], tools=[tool], tool_choice="auto"
        )
        call_kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("tool_choice") == "auto"
        assert call_kwargs["tools"][0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_unsupported_param_raises(self, provider):
        with pytest.raises(ValueError, match="Unsupported parameter"):
            await provider.chat_completion([Message(role=Role.USER, content="Hi")], bogus=True)

    @pytest.mark.asyncio
    async def test_streaming_includes_usage(self, provider):
        async def mock_stream():
            yield MagicMock(
                model_dump=lambda exclude_none=True: {"choices": [{"delta": {"content": "h"}}]}
            )

        provider._client.chat.completions.create = AsyncMock(return_value=mock_stream())
        await provider.chat_completion([Message(role=Role.USER, content="Hi")], stream=True)
        call_kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("stream_options") == {"include_usage": True}


# ---------------------------------------------------------------------------
# Embeddings (mocked)
# ---------------------------------------------------------------------------


class TestEmbeddings:
    @pytest.mark.asyncio
    async def test_create_embeddings(self, provider):
        mock_resp = MagicMock()
        mock_resp.model_dump.return_value = {
            "data": [{"embedding": [0.1, 0.2], "index": 0}],
            "model": "embedding-3",
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        }
        provider._client.embeddings.create = AsyncMock(return_value=mock_resp)

        result = await provider.create_embeddings("hello", dimensions=512)
        assert result["model"] == "embedding-3"
        call_kwargs = provider._client.embeddings.create.call_args.kwargs
        assert call_kwargs["model"] == "embedding-3"
        assert call_kwargs["input"] == "hello"
        assert call_kwargs["dimensions"] == 512


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


class TestTokenCounting:
    @pytest.mark.asyncio
    async def test_count_tokens(self, provider):
        count = await provider.count_tokens("Hello, world!")
        assert isinstance(count, int)
        assert count > 0

    @pytest.mark.asyncio
    async def test_count_tokens_empty(self, provider):
        assert await provider.count_tokens("") == 0

    @pytest.mark.asyncio
    async def test_count_message_tokens(self, provider):
        msgs = [
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi there!"),
        ]
        count = await provider.count_message_tokens(msgs)
        assert isinstance(count, int)
        assert count > 0


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


class TestClose:
    @pytest.mark.asyncio
    async def test_close_cleans_up(self, provider):
        provider._client.close = AsyncMock()
        await provider.close()
        assert provider._client is None


# ---------------------------------------------------------------------------
# Backend resolution and SDK / httpx transports
# ---------------------------------------------------------------------------


def _make_sdk_provider():
    """Build a provider forced onto the (simulated) native SDK backend."""
    import llmcore.providers.zai_provider as zp

    cfg = MINIMAL_CONFIG.copy()
    cfg["backend"] = "sdk"
    cfg["_instance_name"] = "zai"
    sdk_client = MagicMock()
    with patch.object(zp, "zai_sdk_available", True), patch.object(
        zp, "ZaiClient", MagicMock(return_value=sdk_client)
    ), patch.object(zp, "ZhipuAiClient", MagicMock(return_value=sdk_client)):
        p = zp.ZaiProvider(cfg)
    return p, sdk_client


class TestBackendResolution:
    def test_auto_prefers_available(self):
        from llmcore.providers.zai_provider import ZaiProvider

        # In this environment zai-sdk is absent, openai present → "openai".
        assert ZaiProvider._resolve_backend("auto") == "openai"

    def test_explicit_httpx(self):
        from llmcore.providers.zai_provider import ZaiProvider

        assert ZaiProvider._resolve_backend("httpx") == "httpx"

    def test_unavailable_sdk_falls_back(self):
        from llmcore.providers.zai_provider import ZaiProvider

        # sdk unavailable here → falls through to openai
        assert ZaiProvider._resolve_backend("sdk") == "openai"

    def test_unknown_backend_uses_auto(self):
        from llmcore.providers.zai_provider import ZaiProvider

        assert ZaiProvider._resolve_backend("bogus") == "openai"


class TestSDKBackend:
    def test_init_uses_sdk_client(self):
        p, sdk_client = _make_sdk_provider()
        assert p._backend == "sdk"
        assert p._sdk_client is sdk_client
        assert p._client is None

    @pytest.mark.asyncio
    async def test_chat_passes_native_thinking_kwargs(self):
        p, sdk_client = _make_sdk_provider()
        result_obj = MagicMock()
        result_obj.model_dump.return_value = {"choices": [{"message": {"content": "hi"}}]}
        sdk_client.chat.completions.create = MagicMock(return_value=result_obj)

        result = await p.chat_completion(
            [Message(role=Role.USER, content="Hi")], reasoning_effort="max", do_sample=False
        )
        assert result["choices"][0]["message"]["content"] == "hi"
        call = sdk_client.chat.completions.create.call_args.kwargs
        # SDK takes thinking / reasoning_effort as NATIVE kwargs (not extra_body)
        assert call["thinking"] == {"type": "enabled"}
        assert call["reasoning_effort"] == "max"
        assert call["do_sample"] is False
        assert "extra_body" not in call

    @pytest.mark.asyncio
    async def test_sdk_streaming_bridges_sync_iterator(self):
        p, sdk_client = _make_sdk_provider()
        chunks = [
            MagicMock(model_dump=lambda exclude_none=True: {"choices": [{"delta": {"content": "a"}}]}),
            MagicMock(model_dump=lambda exclude_none=True: {"choices": [{"delta": {"content": "b"}}]}),
        ]
        sdk_client.chat.completions.create = MagicMock(return_value=iter(chunks))

        gen = await p.chat_completion([Message(role=Role.USER, content="Hi")], stream=True)
        out = [c async for c in gen]
        assert [p.extract_delta_content(c) for c in out] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_sdk_embeddings(self):
        p, sdk_client = _make_sdk_provider()
        emb = MagicMock()
        emb.model_dump.return_value = {"data": [{"embedding": [0.1]}], "model": "embedding-3"}
        sdk_client.embeddings.create = MagicMock(return_value=emb)
        result = await p.create_embeddings("hi")
        assert result["model"] == "embedding-3"
        assert sdk_client.embeddings.create.call_args.kwargs["input"] == "hi"

    @pytest.mark.asyncio
    async def test_sdk_web_search(self):
        p, sdk_client = _make_sdk_provider()
        ws = MagicMock()
        ws.model_dump.return_value = {"search_result": [{"title": "T"}]}
        sdk_client.web_search.web_search = MagicMock(return_value=ws)
        result = await p.web_search("q")
        assert result["search_result"][0]["title"] == "T"


class TestHttpxBackend:
    @pytest.fixture
    def httpx_provider(self):
        cfg = MINIMAL_CONFIG.copy()
        cfg["backend"] = "httpx"
        with patch("llmcore.providers.zai_provider.AsyncOpenAI"):
            from llmcore.providers.zai_provider import ZaiProvider

            p = ZaiProvider(cfg)
        http = MagicMock()
        http.post = AsyncMock()
        p._get_http = lambda: http  # type: ignore[assignment]
        p._http = http
        return p, http

    @pytest.mark.asyncio
    async def test_chat_posts_inline_body(self, httpx_provider):
        p, http = httpx_provider
        http.post.return_value = _mock_http_response(
            json_body={"choices": [{"message": {"content": "ok"}}]}
        )
        result = await p.chat_completion(
            [Message(role=Role.USER, content="Hi")], reasoning_effort="high"
        )
        assert result["choices"][0]["message"]["content"] == "ok"
        body = http.post.call_args.kwargs["json"]
        assert http.post.call_args[0][0] == "/chat/completions"
        # httpx backend inlines thinking/effort directly in the body
        assert body["thinking"] == {"type": "enabled"}
        assert body["reasoning_effort"] == "high"

    @pytest.mark.asyncio
    async def test_embeddings_posts_to_endpoint(self, httpx_provider):
        p, http = httpx_provider
        http.post.return_value = _mock_http_response(json_body={"model": "embedding-3", "data": []})
        result = await p.create_embeddings(["a", "b"])
        assert result["model"] == "embedding-3"
        assert http.post.call_args[0][0] == "/embeddings"
        assert http.post.call_args.kwargs["json"]["input"] == ["a", "b"]


# ---------------------------------------------------------------------------
# Multimodal media APIs (image / TTS / STT / OCR / video / web search)
# ---------------------------------------------------------------------------


def _mock_http_response(*, json_body: Any = None, content: bytes | None = None) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    if json_body is not None:
        resp.json = MagicMock(return_value=json_body)
    if content is not None:
        resp.content = content
    return resp


@pytest.fixture
def media_provider():
    """Provider with a mocked raw httpx media client."""
    with patch("llmcore.providers.zai_provider.AsyncOpenAI"):
        from llmcore.providers.zai_provider import ZaiProvider

        p = ZaiProvider(MINIMAL_CONFIG.copy())
        http = MagicMock()
        http.post = AsyncMock()
        http.get = AsyncMock()
        p._http = http
        # Bypass the lazy creator so the mock is always returned.
        p._get_http = lambda: http  # type: ignore[assignment]
        return p


class TestMediaAPIs:
    @pytest.mark.asyncio
    async def test_generate_image(self, media_provider):
        media_provider._http.post.return_value = _mock_http_response(
            json_body={"created": 123, "data": [{"url": "https://img/1.png"}]}
        )
        result = await media_provider.generate_image("a red panda", size="1024x1024")
        assert result.model == "cogview-4"
        assert result.images[0].url == "https://img/1.png"
        path, kwargs = media_provider._http.post.call_args[0], media_provider._http.post.call_args.kwargs
        assert path[0] == "/images/generations"
        assert kwargs["json"]["prompt"] == "a red panda"
        assert kwargs["json"]["size"] == "1024x1024"

    @pytest.mark.asyncio
    async def test_generate_speech(self, media_provider):
        media_provider._http.post.return_value = _mock_http_response(content=b"RIFFfake")
        result = await media_provider.generate_speech("hello", voice="tongtong")
        assert result.audio_data == b"RIFFfake"
        assert result.format == "wav"
        assert result.voice == "tongtong"
        assert result.model == "glm-tts"
        assert media_provider._http.post.call_args[0][0] == "/audio/speech"

    @pytest.mark.asyncio
    async def test_transcribe_audio_bytes(self, media_provider):
        media_provider._http.post.return_value = _mock_http_response(
            json_body={"text": "hello world", "language": "en"}
        )
        result = await media_provider.transcribe_audio(b"\x00\x01audio")
        assert result.text == "hello world"
        assert result.model == "glm-asr-2512"
        kwargs = media_provider._http.post.call_args.kwargs
        assert "files" in kwargs and "file" in kwargs["files"]
        assert kwargs["data"]["model"] == "glm-asr-2512"

    @pytest.mark.asyncio
    async def test_ocr_url(self, media_provider):
        media_provider._http.post.return_value = _mock_http_response(
            json_body={"pages": [{"index": 0, "markdown": "# Title"}]}
        )
        result = await media_provider.ocr("https://example.com/doc.pdf")
        assert result.model == "glm-ocr"
        assert result.pages_processed == 1
        assert result.pages[0]["markdown"] == "# Title"
        kwargs = media_provider._http.post.call_args.kwargs
        assert media_provider._http.post.call_args[0][0] == "/layout_parsing"
        assert kwargs["json"]["file"] == "https://example.com/doc.pdf"

    @pytest.mark.asyncio
    async def test_generate_video_no_wait(self, media_provider):
        media_provider._http.post.return_value = _mock_http_response(
            json_body={"id": "task-1", "task_status": "PROCESSING"}
        )
        result = await media_provider.generate_video("a cat surfing")
        assert result["id"] == "task-1"
        assert media_provider._http.post.call_args[0][0] == "/videos/generations"
        assert media_provider._http.post.call_args.kwargs["json"]["prompt"] == "a cat surfing"

    @pytest.mark.asyncio
    async def test_generate_video_wait_polls(self, media_provider):
        media_provider._http.post.return_value = _mock_http_response(
            json_body={"id": "task-2", "task_status": "PROCESSING"}
        )
        media_provider._http.get.return_value = _mock_http_response(
            json_body={"id": "task-2", "task_status": "SUCCESS", "video_result": [{"url": "v.mp4"}]}
        )
        result = await media_provider.generate_video(
            "a dog", wait=True, poll_interval=0, max_wait_seconds=10
        )
        assert result["task_status"] == "SUCCESS"
        assert media_provider._http.get.call_args[0][0] == "/async-result/task-2"

    @pytest.mark.asyncio
    async def test_web_search(self, media_provider):
        media_provider._http.post.return_value = _mock_http_response(
            json_body={"search_result": [{"title": "T", "link": "https://x"}]}
        )
        result = await media_provider.web_search("llmcore", count=5)
        assert result["search_result"][0]["title"] == "T"
        kwargs = media_provider._http.post.call_args.kwargs
        assert media_provider._http.post.call_args[0][0] == "/web_search"
        assert kwargs["json"]["search_query"] == "llmcore"
        assert kwargs["json"]["count"] == 5


# ---------------------------------------------------------------------------
# Provider registration / aliases
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_in_provider_map(self):
        from llmcore.providers.manager import PROVIDER_MAP
        from llmcore.providers.zai_provider import ZaiProvider

        assert PROVIDER_MAP["zai"] is ZaiProvider
        for alias in ("glm", "zhipu", "zhipuai", "bigmodel"):
            assert PROVIDER_MAP[alias] is ZaiProvider

    def test_instance_aliases(self):
        from llmcore.providers.manager import _PROVIDER_INSTANCE_ALIASES

        for alias in ("glm", "zhipu", "zhipuai", "bigmodel"):
            assert _PROVIDER_INSTANCE_ALIASES[alias] == "zai"
