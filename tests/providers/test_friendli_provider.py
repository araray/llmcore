# tests/providers/test_friendli_provider.py
"""
Tests for the FriendliAI provider implementation.

Covers:
- Initialization: API key / team ID resolution, endpoint types, base URLs
- Backend resolution (openai -> httpx -> sdk) and explicit selection
- Request parameter splitting (native vs Friendli extras, chat_template_kwargs)
- Mutually-exclusive body fields (tools vs min_tokens / response_format)
- Message payload building (multimodal, tool_calls, reasoning_content)
- Chat completion on the ``openai`` backend (mocked AsyncOpenAI)
- Chat completion + SSE streaming on the ``httpx`` backend (respx)
- Response extraction (content, reasoning, tool calls, usage, finish reason)
- Model discovery from the rich Friendli catalog
- Context length resolution and token counting (local + native)
- Error mapping (401/403/404/429/context overflow)
- Endpoint-type gating for embeddings and image generation
- Provider registration and aliases

Backends are always selected explicitly so the suite is deterministic whether
or not the optional ``friendli`` SDK is installed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from llmcore.exceptions import ConfigError, ContextLengthError, ProviderError
from llmcore.models import Message, Role, Tool
from llmcore.providers.friendli_provider import (
    FriendliProvider,
    friendli_sdk_available,
)

SERVERLESS_URL = "https://api.friendli.ai/serverless/v1"
DEDICATED_URL = "https://api.friendli.ai/dedicated/v1"

BASE_CONFIG: dict[str, Any] = {
    "api_key": "flp_test_key",
    "default_model": "zai-org/GLM-5.3",
    "timeout": 30,
}

CATALOG_ENTRY: dict[str, Any] = {
    "id": "zai-org/GLM-5.3-Flash",
    "name": "zai-org/GLM-5.3-Flash",
    "created": 1787929200,
    "context_length": 1048576,
    "max_completion_tokens": 1048576,
    "pricing": {
        "input": "0.00000015",
        "output": "0.0000005",
        "prompt": "0.00000015",
        "completion": "0.0000005",
        "input_cache_read": "0.00000003",
    },
    "functionality": {
        "tool_call": True,
        "parallel_tool_call": True,
        "structured_output": True,
        "tool_choice": True,
        "system_messages": True,
    },
    "description": "Fast GLM model",
    "deprecation_date": None,
    "reasoning": True,
    "reasoning_options": [{"type": "effort", "values": ["low", "high", "max"]}],
    "input_modalities": ["text", "image", "video"],
    "output_modalities": ["text"],
    "interleaved": "reasoning_content",
    "base_model": "zhipuai/glm-5.3-flash",
    "mode": "chat",
    "default_params": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0},
}


def _clean_env(monkeypatch) -> None:
    """Remove every Friendli environment variable the provider consults."""
    for name in (
        "FRIENDLI_TOKEN",
        "FRIENDLIAI_API_KEY",
        "FRIENDLI_API_KEY",
        "FRIENDLI_TEAM_ID",
        "FRIENDLIAI_TEAM_ID",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    """Keep ambient Friendli credentials out of every test."""
    _clean_env(monkeypatch)
    yield


@pytest.fixture
def provider():
    """A FriendliProvider on the mocked ``openai`` backend."""
    with patch("llmcore.providers.friendli_provider.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        p = FriendliProvider({**BASE_CONFIG, "backend": "openai"}, log_raw_payloads=False)
    return p


@pytest.fixture
def httpx_provider():
    """A FriendliProvider on the ``httpx`` backend (exercised with respx)."""
    return FriendliProvider({**BASE_CONFIG, "backend": "httpx"})


def _mock_openai_response(payload: dict[str, Any]) -> MagicMock:
    resp = MagicMock()
    resp.model_dump.return_value = payload
    return resp


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    def test_basic_init(self, provider):
        assert provider.get_name() == "friendli"
        assert provider.default_model == "zai-org/GLM-5.3"
        assert provider._endpoint_type == "serverless"
        assert provider._base_url == SERVERLESS_URL
        assert provider._backend == "openai"

    def test_instance_name_override(self):
        with patch("llmcore.providers.friendli_provider.AsyncOpenAI"):
            p = FriendliProvider({**BASE_CONFIG, "backend": "openai", "_instance_name": "my-friendli"})
        assert p.get_name() == "my-friendli"

    def test_dedicated_base_url(self):
        p = FriendliProvider({**BASE_CONFIG, "backend": "httpx", "endpoint_type": "dedicated"})
        assert p._base_url == DEDICATED_URL

    def test_container_requires_base_url(self):
        with pytest.raises(ConfigError, match="requires an explicit base_url"):
            FriendliProvider({**BASE_CONFIG, "backend": "httpx", "endpoint_type": "container"})

    def test_container_without_key_uses_placeholder(self):
        p = FriendliProvider(
            {"backend": "httpx", "endpoint_type": "container", "base_url": "http://localhost:8000/v1"}
        )
        assert p._api_key == "EMPTY"
        assert p._base_url == "http://localhost:8000/v1"

    def test_missing_key_raises_for_hosted(self):
        with pytest.raises(ConfigError, match="Friendli API key not found"):
            FriendliProvider({"backend": "httpx", "default_model": "m"})

    def test_invalid_endpoint_type_falls_back(self):
        p = FriendliProvider({**BASE_CONFIG, "backend": "httpx", "endpoint_type": "bogus"})
        assert p._endpoint_type == "serverless"

    def test_base_url_trailing_slash_stripped(self):
        p = FriendliProvider({**BASE_CONFIG, "backend": "httpx", "base_url": f"{SERVERLESS_URL}/"})
        assert p._base_url == SERVERLESS_URL

    def test_invalid_reasoning_effort_ignored(self):
        p = FriendliProvider({**BASE_CONFIG, "backend": "httpx", "reasoning_effort": "turbo"})
        assert p._default_reasoning_effort is None

    def test_valid_reasoning_effort_kept(self):
        p = FriendliProvider({**BASE_CONFIG, "backend": "httpx", "reasoning_effort": "MAX"})
        assert p._default_reasoning_effort == "max"


class TestCredentialResolution:
    """API key and team ID resolve from config then the documented env vars."""

    @pytest.mark.parametrize(
        "env_var", ["FRIENDLI_TOKEN", "FRIENDLIAI_API_KEY", "FRIENDLI_API_KEY"]
    )
    def test_api_key_env_vars(self, monkeypatch, env_var):
        monkeypatch.setenv(env_var, f"flp_from_{env_var}")
        p = FriendliProvider({"backend": "httpx"})
        assert p._api_key == f"flp_from_{env_var}"

    def test_api_key_env_var_indirection(self, monkeypatch):
        monkeypatch.setenv("MY_CUSTOM_KEY", "flp_custom")
        p = FriendliProvider({"backend": "httpx", "api_key_env_var": "MY_CUSTOM_KEY"})
        assert p._api_key == "flp_custom"

    def test_explicit_key_wins(self, monkeypatch):
        monkeypatch.setenv("FRIENDLI_TOKEN", "flp_env")
        p = FriendliProvider({"backend": "httpx", "api_key": "flp_explicit"})
        assert p._api_key == "flp_explicit"

    def test_token_preferred_over_api_key_env(self, monkeypatch):
        monkeypatch.setenv("FRIENDLI_TOKEN", "flp_token")
        monkeypatch.setenv("FRIENDLIAI_API_KEY", "flp_aikey")
        p = FriendliProvider({"backend": "httpx"})
        assert p._api_key == "flp_token"

    @pytest.mark.parametrize("env_var", ["FRIENDLI_TEAM_ID", "FRIENDLIAI_TEAM_ID"])
    def test_team_id_env_vars(self, monkeypatch, env_var):
        monkeypatch.setenv(env_var, "team-123")
        p = FriendliProvider({**BASE_CONFIG, "backend": "httpx"})
        assert p._team_id == "team-123"
        assert p._team_headers()["X-Friendli-Team"] == "team-123"

    def test_team_id_absent(self, provider):
        assert provider._team_id is None
        assert "X-Friendli-Team" not in provider._team_headers()

    def test_explicit_team_id_wins(self, monkeypatch):
        monkeypatch.setenv("FRIENDLI_TEAM_ID", "env-team")
        p = FriendliProvider({**BASE_CONFIG, "backend": "httpx", "team_id": "cfg-team"})
        assert p._team_id == "cfg-team"


class TestBackendResolution:
    def test_auto_prefers_openai(self):
        with patch.multiple(
            "llmcore.providers.friendli_provider",
            openai_available=True,
            httpx_available=True,
            friendli_sdk_available=True,
        ):
            assert FriendliProvider._resolve_backend(None) == "openai"
            assert FriendliProvider._resolve_backend("auto") == "openai"

    def test_auto_falls_back_to_httpx(self):
        with patch.multiple(
            "llmcore.providers.friendli_provider",
            openai_available=False,
            httpx_available=True,
            friendli_sdk_available=True,
        ):
            assert FriendliProvider._resolve_backend(None) == "httpx"

    def test_auto_falls_back_to_sdk(self):
        with patch.multiple(
            "llmcore.providers.friendli_provider",
            openai_available=False,
            httpx_available=False,
            friendli_sdk_available=True,
        ):
            assert FriendliProvider._resolve_backend(None) == "sdk"

    def test_explicit_backend_honored(self):
        with patch.multiple(
            "llmcore.providers.friendli_provider",
            openai_available=True,
            httpx_available=True,
            friendli_sdk_available=True,
        ):
            assert FriendliProvider._resolve_backend("httpx") == "httpx"
            assert FriendliProvider._resolve_backend("sdk") == "sdk"

    def test_unavailable_backend_falls_back(self):
        with patch.multiple(
            "llmcore.providers.friendli_provider",
            openai_available=True,
            httpx_available=True,
            friendli_sdk_available=False,
        ):
            assert FriendliProvider._resolve_backend("sdk") == "openai"

    def test_unknown_backend_name_auto_detects(self):
        with patch.multiple(
            "llmcore.providers.friendli_provider",
            openai_available=True,
            httpx_available=True,
            friendli_sdk_available=False,
        ):
            assert FriendliProvider._resolve_backend("grpc") == "openai"

    def test_no_transport_raises(self):
        with patch.multiple(
            "llmcore.providers.friendli_provider",
            openai_available=False,
            httpx_available=False,
            friendli_sdk_available=False,
        ):
            with pytest.raises(ConfigError, match="requires one of"):
                FriendliProvider(dict(BASE_CONFIG))


# ---------------------------------------------------------------------------
# Request parameter handling
# ---------------------------------------------------------------------------


class TestRequestParams:
    def test_native_vs_extras_split(self, provider):
        native, extras = provider._resolve_request_params(
            {"temperature": 0.5, "max_tokens": 64, "top_k": 40, "repetition_penalty": 1.1}
        )
        assert native == {"temperature": 0.5, "max_tokens": 64}
        assert extras["top_k"] == 40
        assert extras["repetition_penalty"] == 1.1

    def test_parse_reasoning_default_on(self, provider):
        _, extras = provider._resolve_request_params({})
        assert extras["parse_reasoning"] is True

    def test_parse_reasoning_can_be_disabled_per_request(self, provider):
        _, extras = provider._resolve_request_params({"parse_reasoning": False})
        assert extras["parse_reasoning"] is False

    def test_reasoning_effort_default_applied(self):
        p = FriendliProvider({**BASE_CONFIG, "backend": "httpx", "reasoning_effort": "high"})
        _, extras = p._resolve_request_params({})
        assert extras["reasoning_effort"] == "high"

    def test_reasoning_effort_per_request_override(self):
        p = FriendliProvider({**BASE_CONFIG, "backend": "httpx", "reasoning_effort": "high"})
        _, extras = p._resolve_request_params({"reasoning_effort": "max"})
        assert extras["reasoning_effort"] == "max"

    def test_invalid_effort_falls_back_to_default(self):
        p = FriendliProvider({**BASE_CONFIG, "backend": "httpx", "reasoning_effort": "low"})
        _, extras = p._resolve_request_params({"reasoning_effort": "nope"})
        assert extras["reasoning_effort"] == "low"

    def test_ultracode_effort_accepted(self, provider):
        _, extras = provider._resolve_request_params({"reasoning_effort": "ultracode"})
        assert extras["reasoning_effort"] == "ultracode"

    def test_enable_thinking_folded_into_template_kwargs(self, provider):
        _, extras = provider._resolve_request_params({"enable_thinking": True})
        assert extras["chat_template_kwargs"] == {"enable_thinking": True}

    def test_clear_thinking_folded_into_template_kwargs(self, provider):
        _, extras = provider._resolve_request_params({"clear_thinking": True})
        assert extras["chat_template_kwargs"]["clear_thinking"] is True

    def test_explicit_template_kwargs_merge(self, provider):
        _, extras = provider._resolve_request_params(
            {"chat_template_kwargs": {"custom": 1}, "enable_thinking": False}
        )
        assert extras["chat_template_kwargs"] == {"custom": 1, "enable_thinking": False}

    def test_config_enable_thinking_default(self):
        p = FriendliProvider({**BASE_CONFIG, "backend": "httpx", "enable_thinking": True})
        _, extras = p._resolve_request_params({})
        assert extras["chat_template_kwargs"]["enable_thinking"] is True

    def test_list_seed_routed_to_extras(self, provider):
        native, extras = provider._resolve_request_params({"seed": [1, 2, 3]})
        assert "seed" not in native
        assert extras["seed"] == [1, 2, 3]

    def test_int_seed_stays_native(self, provider):
        native, extras = provider._resolve_request_params({"seed": 7})
        assert native["seed"] == 7
        assert "seed" not in extras

    def test_reasoning_budget(self, provider):
        _, extras = provider._resolve_request_params({"reasoning_budget": 1024})
        assert extras["reasoning_budget"] == 1024

    def test_tools_drop_min_tokens_and_response_format(self, provider):
        native = {"response_format": {"type": "json_object"}}
        extras = {"min_tokens": 5}
        provider._apply_mutual_exclusions(native, extras, has_tools=True)
        assert "response_format" not in native
        assert "min_tokens" not in extras

    def test_response_format_drops_min_tokens(self, provider):
        native = {"response_format": {"type": "json_object"}}
        extras = {"min_tokens": 5}
        provider._apply_mutual_exclusions(native, extras, has_tools=False)
        assert native["response_format"] == {"type": "json_object"}
        assert "min_tokens" not in extras

    def test_no_exclusions_when_unrelated(self, provider):
        native: dict[str, Any] = {"temperature": 0.2}
        extras: dict[str, Any] = {"min_tokens": 5}
        provider._apply_mutual_exclusions(native, extras, has_tools=False)
        assert extras["min_tokens"] == 5


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

    def test_assistant_first_class_tool_calls(self, provider):
        calls = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
        msg = Message(role=Role.ASSISTANT, content="", tool_calls=calls)
        payload = provider._build_message_payload(msg)
        assert payload["tool_calls"] == calls
        assert payload["content"] is None

    def test_first_class_tool_calls_beat_metadata(self, provider):
        first_class = [{"id": "new", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
        legacy = [{"id": "old", "type": "function", "function": {"name": "g", "arguments": "{}"}}]
        msg = Message(
            role=Role.ASSISTANT, content="", tool_calls=first_class, metadata={"tool_calls": legacy}
        )
        assert provider._build_message_payload(msg)["tool_calls"] == first_class

    def test_assistant_reasoning_content_preserved(self, provider):
        msg = Message(
            role=Role.ASSISTANT,
            content="42",
            metadata={"reasoning_content": "thinking..."},
        )
        assert provider._build_message_payload(msg)["reasoning_content"] == "thinking..."

    def test_inline_images(self, provider):
        msg = Message(
            role=Role.USER,
            content="What is this?",
            metadata={"inline_images": ["https://example.com/a.png"]},
        )
        parts = provider._build_message_payload(msg)["content"]
        assert parts[0] == {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
        assert parts[-1] == {"type": "text", "text": "What is this?"}

    def test_inline_audio_and_video(self, provider):
        msg = Message(
            role=Role.USER,
            content="Describe",
            metadata={
                "inline_audio": ["data:audio/wav;base64,AAA"],
                "inline_videos": [{"url": "https://example.com/v.mp4"}],
            },
        )
        kinds = [p["type"] for p in provider._build_message_payload(msg)["content"]]
        assert kinds == ["audio_url", "video_url", "text"]

    def test_content_parts_passthrough(self, provider):
        parts = [{"type": "text", "text": "hi"}]
        msg = Message(role=Role.USER, content="ignored", metadata={"content_parts": parts})
        assert provider._build_message_payload(msg)["content"] == parts

    def test_unrecognized_media_entry_skipped(self, provider):
        msg = Message(role=Role.USER, content="x", metadata={"inline_images": [42]})
        assert provider._build_message_payload(msg)["content"] == [{"type": "text", "text": "x"}]

    def test_name_field(self, provider):
        msg = Message(role=Role.USER, content="hi", metadata={"name": "alice"})
        assert provider._build_message_payload(msg)["name"] == "alice"


# ---------------------------------------------------------------------------
# Chat completion — openai backend
# ---------------------------------------------------------------------------


class TestChatCompletionOpenAIBackend:
    async def test_basic_completion(self, provider):
        provider._client.chat.completions.create = AsyncMock(
            return_value=_mock_openai_response(
                {"choices": [{"message": {"content": "Paris"}}], "usage": {}}
            )
        )
        result = await provider.chat_completion([Message(role=Role.USER, content="Capital?")])
        assert result["choices"][0]["message"]["content"] == "Paris"

        kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "zai-org/GLM-5.3"
        assert kwargs["extra_body"]["parse_reasoning"] is True

    async def test_friendli_params_routed_to_extra_body(self, provider):
        provider._client.chat.completions.create = AsyncMock(
            return_value=_mock_openai_response({"choices": [], "usage": {}})
        )
        await provider.chat_completion(
            [Message(role=Role.USER, content="Hi")],
            temperature=0.3,
            top_k=20,
            min_p=0.05,
            repetition_penalty=1.05,
            reasoning_effort="max",
        )
        kwargs = provider._client.chat.completions.create.call_args.kwargs
        extra = kwargs["extra_body"]
        assert kwargs["temperature"] == 0.3
        assert extra["top_k"] == 20
        assert extra["min_p"] == 0.05
        assert extra["repetition_penalty"] == 1.05
        assert extra["reasoning_effort"] == "max"
        for key in ("top_k", "min_p", "repetition_penalty", "reasoning_effort"):
            assert key not in kwargs

    async def test_tools_payload(self, provider):
        provider._client.chat.completions.create = AsyncMock(
            return_value=_mock_openai_response({"choices": [], "usage": {}})
        )
        tool = Tool(
            name="get_weather",
            description="Get weather",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        await provider.chat_completion(
            [Message(role=Role.USER, content="Weather?")], tools=[tool], tool_choice="required"
        )
        kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert kwargs["tools"][0]["function"]["name"] == "get_weather"
        assert kwargs["tool_choice"] == "required"

    async def test_stream_sets_include_usage(self, provider):
        async def _agen():
            yield _mock_openai_response({"choices": [{"delta": {"content": "Hi"}}]})

        provider._client.chat.completions.create = AsyncMock(return_value=_agen())
        gen = await provider.chat_completion([Message(role=Role.USER, content="Hi")], stream=True)
        chunks = [c async for c in gen]
        assert chunks[0]["choices"][0]["delta"]["content"] == "Hi"
        kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert kwargs["stream_options"] == {"include_usage": True}

    async def test_unsupported_param_raises(self, provider):
        with pytest.raises(ValueError, match="Unsupported parameter"):
            await provider.chat_completion([Message(role=Role.USER, content="Hi")], bogus=1)

    async def test_non_message_context_raises(self, provider):
        with pytest.raises(ProviderError, match="list\\[Message\\]"):
            await provider.chat_completion(["not a message"])  # type: ignore[list-item]

    async def test_empty_context_raises(self, provider):
        with pytest.raises(ProviderError, match="No valid messages"):
            await provider.chat_completion([])


# ---------------------------------------------------------------------------
# Chat completion — httpx backend
# ---------------------------------------------------------------------------


class TestChatCompletionHttpxBackend:
    @respx.mock
    async def test_non_streaming(self, httpx_provider):
        route = respx.post(f"{SERVERLESS_URL}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"index": 0, "message": {"content": "Hi"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                },
            )
        )
        result = await httpx_provider.chat_completion([Message(role=Role.USER, content="Hi")])
        assert result["choices"][0]["message"]["content"] == "Hi"

        body = route.calls[0].request.content.decode()
        assert '"parse_reasoning":true' in body.replace(" ", "")
        await httpx_provider.close()

    @respx.mock
    async def test_team_header_sent(self, monkeypatch):
        monkeypatch.setenv("FRIENDLI_TEAM_ID", "team-xyz")
        p = FriendliProvider({**BASE_CONFIG, "backend": "httpx"})
        route = respx.post(f"{SERVERLESS_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [], "usage": {}})
        )
        await p.chat_completion([Message(role=Role.USER, content="Hi")])
        assert route.calls[0].request.headers["X-Friendli-Team"] == "team-xyz"
        await p.close()

    @respx.mock
    async def test_streaming_sse(self, httpx_provider):
        sse = (
            'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"He"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"llo"}}]}\n\n'
            'data: {"choices":[],"usage":{"total_tokens":9}}\n\n'
            "data: [DONE]\n\n"
        )
        respx.post(f"{SERVERLESS_URL}/chat/completions").mock(
            return_value=httpx.Response(
                200, text=sse, headers={"Content-Type": "text/event-stream"}
            )
        )
        gen = await httpx_provider.chat_completion(
            [Message(role=Role.USER, content="Hi")], stream=True
        )
        text, reasoning, usage = "", "", None
        async for chunk in gen:
            text += httpx_provider.extract_delta_content(chunk)
            reasoning += httpx_provider.extract_delta_reasoning_content(chunk) or ""
            if chunk.get("usage"):
                usage = chunk["usage"]
        assert text == "Hello"
        assert reasoning == "think"
        assert usage == {"total_tokens": 9}
        await httpx_provider.close()

    @respx.mock
    async def test_streaming_error_maps_to_provider_error(self, httpx_provider):
        respx.post(f"{SERVERLESS_URL}/chat/completions").mock(
            return_value=httpx.Response(429, json={"message": "Rate limit exceeded"})
        )
        gen = await httpx_provider.chat_completion(
            [Message(role=Role.USER, content="Hi")], stream=True
        )
        with pytest.raises(ProviderError, match="rate limit"):
            async for _ in gen:
                pass
        await httpx_provider.close()


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


class TestErrorMapping:
    @respx.mock
    async def test_401_auth_message(self, httpx_provider):
        respx.post(f"{SERVERLESS_URL}/chat/completions").mock(
            return_value=httpx.Response(401, json={"detail": "Unauthorized"})
        )
        with pytest.raises(ProviderError, match="authentication failed"):
            await httpx_provider.chat_completion([Message(role=Role.USER, content="Hi")])
        await httpx_provider.close()

    @respx.mock
    async def test_403_mentions_team(self, monkeypatch):
        monkeypatch.setenv("FRIENDLI_TEAM_ID", "team-abc")
        p = FriendliProvider({**BASE_CONFIG, "backend": "httpx"})
        respx.post(f"{SERVERLESS_URL}/chat/completions").mock(
            return_value=httpx.Response(403, json={"detail": "Forbidden"})
        )
        with pytest.raises(ProviderError, match="team-abc"):
            await p.chat_completion([Message(role=Role.USER, content="Hi")])
        await p.close()

    @respx.mock
    async def test_404_message(self, httpx_provider):
        respx.post(f"{SERVERLESS_URL}/chat/completions").mock(
            return_value=httpx.Response(404, json={"detail": "Not Found"})
        )
        with pytest.raises(ProviderError, match="404"):
            await httpx_provider.chat_completion([Message(role=Role.USER, content="Hi")])
        await httpx_provider.close()

    @respx.mock
    async def test_429_retryable(self, httpx_provider):
        respx.post(f"{SERVERLESS_URL}/chat/completions").mock(
            return_value=httpx.Response(429, json={"message": "Rate limit exceeded"})
        )
        with pytest.raises(ProviderError) as exc:
            await httpx_provider.chat_completion([Message(role=Role.USER, content="Hi")])
        assert exc.value.status_code == 429
        assert exc.value.retryable is True
        await httpx_provider.close()

    @respx.mock
    async def test_context_overflow_maps_to_context_length_error(self, httpx_provider):
        respx.post(f"{SERVERLESS_URL}/chat/completions").mock(
            return_value=httpx.Response(
                400, json={"detail": "input is too long: prompt length exceeds context"}
            )
        )
        with pytest.raises(ContextLengthError):
            await httpx_provider.chat_completion([Message(role=Role.USER, content="Hi")])
        await httpx_provider.close()

    @respx.mock
    async def test_plain_400_is_provider_error(self, httpx_provider):
        respx.post(f"{SERVERLESS_URL}/chat/completions").mock(
            return_value=httpx.Response(400, json={"detail": "bad parameter"})
        )
        with pytest.raises(ProviderError) as exc:
            await httpx_provider.chat_completion([Message(role=Role.USER, content="Hi")])
        assert not isinstance(exc.value, ContextLengthError)
        await httpx_provider.close()


# ---------------------------------------------------------------------------
# Response extraction
# ---------------------------------------------------------------------------


class TestResponseExtraction:
    SAMPLE: dict[str, Any] = {
        "id": "chatcmpl-1",
        "model": "zai-org/GLM-5.3",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Paris.",
                    "reasoning_content": "The user asks about France.",
                },
            }
        ],
        "usage": {
            "prompt_tokens": 18,
            "completion_tokens": 4,
            "total_tokens": 22,
            "prompt_tokens_details": {"cached_tokens": 12},
        },
    }

    def test_extract_content(self, provider):
        assert provider.extract_response_content(self.SAMPLE) == "Paris."

    def test_extract_reasoning_content(self, provider):
        assert provider.extract_reasoning_content(self.SAMPLE) == "The user asks about France."

    def test_extract_reasoning_alias_field(self, provider):
        resp = {"choices": [{"message": {"content": "x", "reasoning": "alias"}}]}
        assert provider.extract_reasoning_content(resp) == "alias"

    def test_extract_reasoning_absent(self, provider):
        assert provider.extract_reasoning_content({"choices": [{"message": {"content": "x"}}]}) is None

    def test_extract_usage_with_cache(self, provider):
        usage = provider.extract_usage_details(self.SAMPLE)
        assert usage == {
            "prompt_tokens": 18,
            "completion_tokens": 4,
            "total_tokens": 22,
            "cached_tokens": 12,
        }

    def test_extract_usage_empty(self, provider):
        assert provider.extract_usage_details({}) == {}

    def test_extract_finish_reason(self, provider):
        assert provider.extract_finish_reason(self.SAMPLE) == "stop"

    def test_extract_tool_calls(self, provider):
        resp = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city":"Lisbon"}'},
                            }
                        ]
                    }
                }
            ]
        }
        calls = provider.extract_tool_calls(resp)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {"city": "Lisbon"}

    def test_extract_tool_calls_invalid_json(self, provider):
        resp = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{"}}
                        ]
                    }
                }
            ]
        }
        assert provider.extract_tool_calls(resp)[0].arguments == {"_raw": "{"}

    def test_extract_delta_content(self, provider):
        assert provider.extract_delta_content({"choices": [{"delta": {"content": "Hi"}}]}) == "Hi"

    def test_extract_delta_reasoning(self, provider):
        chunk = {"choices": [{"delta": {"reasoning_content": "hm"}}]}
        assert provider.extract_delta_reasoning_content(chunk) == "hm"

    def test_extraction_is_defensive(self, provider):
        assert provider.extract_response_content({}) == ""
        assert provider.extract_delta_content({"choices": []}) == ""
        assert provider.extract_finish_reason({}) is None
        assert provider.extract_tool_calls({}) == []


# ---------------------------------------------------------------------------
# Model discovery and context length
# ---------------------------------------------------------------------------


class TestModelDiscovery:
    @respx.mock
    async def test_get_models_details_from_catalog(self, httpx_provider):
        respx.get(f"{SERVERLESS_URL}/models").mock(
            return_value=httpx.Response(200, json={"data": [CATALOG_ENTRY]})
        )
        details = await httpx_provider.get_models_details()
        assert len(details) == 1
        d = details[0]
        assert d.id == "zai-org/GLM-5.3-Flash"
        assert d.context_length == 1_048_576
        assert d.max_output_tokens == 1_048_576
        assert d.supports_tools is True
        assert d.supports_vision is True
        assert d.supports_reasoning is True
        assert d.metadata["base_model"] == "zhipuai/glm-5.3-flash"
        assert d.metadata["endpoint_type"] == "serverless"
        await httpx_provider.close()

    @respx.mock
    async def test_catalog_is_cached(self, httpx_provider):
        route = respx.get(f"{SERVERLESS_URL}/models").mock(
            return_value=httpx.Response(200, json={"data": [CATALOG_ENTRY]})
        )
        await httpx_provider.get_models_details()
        await httpx_provider.get_models_details()
        assert route.call_count == 1
        await httpx_provider.close()

    @respx.mock
    async def test_catalog_failure_falls_back_to_static_table(self, httpx_provider):
        respx.get(f"{SERVERLESS_URL}/models").mock(return_value=httpx.Response(500, text="boom"))
        details = await httpx_provider.get_models_details()
        ids = {d.id for d in details}
        assert "zai-org/GLM-5.3" in ids
        await httpx_provider.close()

    @respx.mock
    async def test_warm_up_primes_context_lengths(self, httpx_provider):
        respx.get(f"{SERVERLESS_URL}/models").mock(
            return_value=httpx.Response(200, json={"data": [CATALOG_ENTRY]})
        )
        await httpx_provider.warm_up()
        assert httpx_provider.get_max_context_length("zai-org/GLM-5.3-Flash") == 1_048_576
        await httpx_provider.close()

    @respx.mock
    async def test_warm_up_survives_failure(self, httpx_provider):
        respx.get(f"{SERVERLESS_URL}/models").mock(side_effect=httpx.ConnectError("down"))
        await httpx_provider.warm_up()  # must not raise
        await httpx_provider.close()

    async def test_dedicated_has_no_catalog(self):
        p = FriendliProvider(
            {**BASE_CONFIG, "backend": "httpx", "endpoint_type": "dedicated", "default_model": "ep-1"}
        )
        details = await p.get_models_details()
        assert [d.id for d in details] == ["ep-1"]
        await p.close()


class TestContextLength:
    def test_known_model(self, provider):
        assert provider.get_max_context_length("zai-org/GLM-5.1") == 202_752

    def test_default_model(self, provider):
        assert provider.get_max_context_length() == 1_048_576

    def test_model_card_lookup(self, provider):
        # google/gemma-4-31B-it is absent from the static table but has a card.
        assert provider.get_max_context_length("google/gemma-4-31B-it") == 262_144

    def test_unknown_model_uses_fallback(self, provider):
        assert provider.get_max_context_length("nobody/nothing") == 131_072

    def test_configured_fallback_is_honored(self):
        p = FriendliProvider({**BASE_CONFIG, "backend": "httpx", "fallback_context_length": 8192})
        assert p.get_max_context_length("nobody/nothing") == 8192


class TestSupportedParameters:
    def test_includes_reasoning_controls(self, provider):
        params = provider.get_supported_parameters()
        for key in (
            "reasoning_effort",
            "reasoning_budget",
            "parse_reasoning",
            "include_reasoning",
        ):
            assert key in params

    def test_includes_friendli_engine_sampling(self, provider):
        params = provider.get_supported_parameters()
        for key in ("top_k", "min_p", "min_tokens", "repetition_penalty", "eos_token",
                    "xtc_threshold", "xtc_probability"):
            assert key in params

    def test_effort_enum_matches_api(self, provider):
        efforts = provider.get_supported_parameters()["reasoning_effort"]["enum"]
        assert set(efforts) == {
            "minimal", "low", "medium", "high", "xhigh", "max", "ultracode"
        }


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


class TestTokenization:
    @respx.mock
    async def test_tokenize(self, httpx_provider):
        route = respx.post(f"{SERVERLESS_URL}/tokenize").mock(
            return_value=httpx.Response(200, json={"tokens": [1, 2, 3]})
        )
        assert await httpx_provider.tokenize("hello") == [1, 2, 3]
        assert route.call_count == 1
        await httpx_provider.close()

    @respx.mock
    async def test_detokenize(self, httpx_provider):
        respx.post(f"{SERVERLESS_URL}/detokenize").mock(
            return_value=httpx.Response(200, json={"text": "hello"})
        )
        assert await httpx_provider.detokenize([1, 2, 3]) == "hello"
        await httpx_provider.close()

    @respx.mock
    async def test_render_chat(self, httpx_provider):
        respx.post(f"{SERVERLESS_URL}/chat/render").mock(
            return_value=httpx.Response(200, json={"text": "<|user|>Hi"})
        )
        rendered = await httpx_provider.render_chat([Message(role=Role.USER, content="Hi")])
        assert rendered == "<|user|>Hi"
        await httpx_provider.close()

    @respx.mock
    async def test_count_tokens_is_local_by_default(self, httpx_provider):
        route = respx.post(f"{SERVERLESS_URL}/tokenize")
        count = await httpx_provider.count_tokens("hello world")
        assert count > 0
        assert route.call_count == 0
        await httpx_provider.close()

    @respx.mock
    async def test_count_tokens_native_when_enabled(self):
        p = FriendliProvider({**BASE_CONFIG, "backend": "httpx", "native_token_count": True})
        respx.post(f"{SERVERLESS_URL}/tokenize").mock(
            return_value=httpx.Response(200, json={"tokens": [1, 2, 3, 4]})
        )
        assert await p.count_tokens("hello world") == 4
        await p.close()

    @respx.mock
    async def test_native_count_falls_back_on_error(self):
        p = FriendliProvider({**BASE_CONFIG, "backend": "httpx", "native_token_count": True})
        respx.post(f"{SERVERLESS_URL}/tokenize").mock(
            return_value=httpx.Response(404, json={"detail": "Not Found"})
        )
        assert await p.count_tokens("hello world") > 0
        await p.close()

    async def test_count_tokens_empty(self, httpx_provider):
        assert await httpx_provider.count_tokens("") == 0
        await httpx_provider.close()

    async def test_count_message_tokens_empty(self, httpx_provider):
        assert await httpx_provider.count_message_tokens([]) == 0
        await httpx_provider.close()

    @respx.mock
    async def test_count_message_tokens_local(self, httpx_provider):
        route = respx.post(f"{SERVERLESS_URL}/tokenize")
        messages = [
            Message(role=Role.SYSTEM, content="Be terse."),
            Message(role=Role.USER, content="Hi"),
        ]
        count = await httpx_provider.count_message_tokens(messages)
        assert count > 2 * 4
        assert route.call_count == 0
        await httpx_provider.close()


# ---------------------------------------------------------------------------
# Endpoint-type gating for the optional media/embedding surfaces
# ---------------------------------------------------------------------------


class TestEndpointTypeGating:
    async def test_embeddings_rejected_on_serverless(self, httpx_provider):
        with pytest.raises(ProviderError, match="do not expose an embeddings endpoint"):
            await httpx_provider.create_embeddings(["hello"])
        await httpx_provider.close()

    async def test_image_generation_rejected_on_serverless(self, httpx_provider):
        with pytest.raises(ProviderError, match="do not expose an image-generation endpoint"):
            await httpx_provider.generate_image("a cat")
        await httpx_provider.close()

    @respx.mock
    async def test_embeddings_on_dedicated(self):
        p = FriendliProvider(
            {**BASE_CONFIG, "backend": "httpx", "endpoint_type": "dedicated", "default_model": "ep-1"}
        )
        respx.post(f"{DEDICATED_URL}/embeddings").mock(
            return_value=httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]})
        )
        out = await p.create_embeddings(["hello"], encoding_format="float")
        assert out["data"][0]["embedding"] == [0.1, 0.2]
        await p.close()

    @respx.mock
    async def test_image_generation_on_dedicated(self):
        p = FriendliProvider(
            {**BASE_CONFIG, "backend": "httpx", "endpoint_type": "dedicated", "default_model": "ep-1"}
        )
        respx.post(f"{DEDICATED_URL}/images/generations").mock(
            return_value=httpx.Response(
                200, json={"data": [{"url": "https://cdn/img.png", "seed": 7}]}
            )
        )
        result = await p.generate_image("a cat", num_inference_steps=10)
        assert result.images[0].url == "https://cdn/img.png"
        assert result.model == "ep-1"
        await p.close()

    @respx.mock
    async def test_transcribe_audio(self, httpx_provider):
        respx.post(f"{SERVERLESS_URL}/audio/transcriptions").mock(
            return_value=httpx.Response(
                200, json={"text": "hello there", "usage": {"input_audio_length_ms": 2000}}
            )
        )
        result = await httpx_provider.transcribe_audio(
            b"RIFFfake", model="openai/whisper-large-v3", language="en"
        )
        assert result.text == "hello there"
        assert result.duration_seconds == 2.0
        await httpx_provider.close()


# ---------------------------------------------------------------------------
# Friendli Suite (team billing / usage)
# ---------------------------------------------------------------------------


class TestSuiteApis:
    @respx.mock
    async def test_get_team_cost(self, monkeypatch):
        monkeypatch.setenv("FRIENDLI_TEAM_ID", "team-1")
        p = FriendliProvider({**BASE_CONFIG, "backend": "httpx"})
        route = respx.get("https://api.friendli.ai/v1/team/cost").mock(
            return_value=httpx.Response(200, json={"data": [], "has_more": False})
        )
        out = await p.get_team_cost("2026-09-01T00:00:00Z", "2026-09-02T00:00:00Z", limit=3)
        assert out == {"data": [], "has_more": False}
        request = route.calls[0].request
        assert request.headers["X-Friendli-Team"] == "team-1"
        assert "limit=3" in str(request.url)
        await p.close()

    @respx.mock
    async def test_get_team_usage(self, httpx_provider):
        respx.get("https://api.friendli.ai/v1/team/usage").mock(
            return_value=httpx.Response(200, json={"data": [{"start_time": "x"}]})
        )
        out = await httpx_provider.get_team_usage("2026-09-01T00:00:00Z", "2026-09-02T00:00:00Z")
        assert out["data"][0]["start_time"] == "x"
        await httpx_provider.close()

    @respx.mock
    async def test_suite_error_maps(self, httpx_provider):
        respx.get("https://api.friendli.ai/v1/team/cost").mock(
            return_value=httpx.Response(403, json={"detail": "Forbidden"})
        )
        with pytest.raises(ProviderError):
            await httpx_provider.get_team_cost("2026-09-01T00:00:00Z", "2026-09-02T00:00:00Z")
        await httpx_provider.close()


# ---------------------------------------------------------------------------
# Official SDK backend (skipped when the vendor SDK is not installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not friendli_sdk_available, reason="the 'friendli' SDK is not installed")
class TestSdkBackend:
    """The vendor SDK is optional, so these only run when it is present."""

    @staticmethod
    def _sdk_provider(**overrides: Any) -> FriendliProvider:
        return FriendliProvider(
            {**BASE_CONFIG, "backend": "sdk", "parse_reasoning": False, **overrides}
        )

    def test_namespaces_resolve(self):
        p = self._sdk_provider()
        # The generated SDK spells this resource chat_render, not chatrender.
        for resource in ("chat", "token", "chat_render"):
            assert p._sdk_namespace(resource) is not None

    def test_dedicated_exposes_embeddings(self):
        p = self._sdk_provider(endpoint_type="dedicated", default_model="ep-1")
        assert p._sdk_namespace("embeddings") is not None

    def test_unknown_resource_raises_actionable_error(self):
        p = self._sdk_provider()
        with pytest.raises(ProviderError, match="do not expose 'nope'"):
            p._sdk_namespace("nope")

    def test_container_has_no_chat_render(self):
        p = self._sdk_provider(
            endpoint_type="container", base_url="http://localhost:8000/v1"
        )
        with pytest.raises(ProviderError, match="do not expose 'chat_render'"):
            p._sdk_namespace("chat_render")

    def test_undeclared_kwargs_are_dropped(self):
        p = self._sdk_provider()
        complete = p._sdk_namespace("chat").complete
        kept = p._filter_sdk_kwargs(
            complete, {"temperature": 0.5, "top_k": 10, "parse_reasoning": True, "bogus": 1}
        )
        assert kept == {"temperature": 0.5, "top_k": 10, "parse_reasoning": True}

    def test_filter_is_a_noop_for_unintrospectable_callables(self):
        p = self._sdk_provider()
        payload = {"a": 1}
        assert p._filter_sdk_kwargs(object(), payload) == payload

    def test_sdk_warns_when_reasoning_parsing_requested(self, caplog):
        import logging as _logging

        with caplog.at_level(_logging.WARNING, logger="llmcore.providers.friendli_provider"):
            FriendliProvider({**BASE_CONFIG, "backend": "sdk", "parse_reasoning": True})
        assert any("drop reasoning_content" in r.message for r in caplog.records)

    async def test_chat_via_sdk_filters_and_dispatches(self):
        p = self._sdk_provider()
        chat = MagicMock()
        chat.complete = AsyncMock(
            return_value=_mock_openai_response({"choices": [{"message": {"content": "ok"}}]})
        )
        chat.complete.__signature__ = __import__("inspect").Signature(
            [
                __import__("inspect").Parameter(name, kind=__import__("inspect").Parameter.KEYWORD_ONLY)
                for name in ("model", "messages", "temperature")
            ]
        )
        with patch.object(FriendliProvider, "_sdk_namespace", return_value=chat):
            result = await p.chat_completion(
                [Message(role=Role.USER, content="Hi")], temperature=0.4, top_k=5
            )
        assert result["choices"][0]["message"]["content"] == "ok"
        kwargs = chat.complete.call_args.kwargs
        assert kwargs["temperature"] == 0.4
        assert "top_k" not in kwargs  # dropped: not in the (stubbed) signature


# ---------------------------------------------------------------------------
# Provider registration
# ---------------------------------------------------------------------------


class TestProviderRegistration:
    def test_registered_in_provider_map(self):
        from llmcore.providers.manager import PROVIDER_MAP

        assert PROVIDER_MAP["friendli"] is FriendliProvider

    def test_aliases_registered(self):
        from llmcore.providers.manager import _PROVIDER_INSTANCE_ALIASES, PROVIDER_MAP

        for alias in ("friendliai", "friendli_ai"):
            assert PROVIDER_MAP[alias] is FriendliProvider
            assert _PROVIDER_INSTANCE_ALIASES[alias] == "friendli"

    def test_default_config_section_exists(self):
        import tomllib
        from pathlib import Path

        import llmcore

        path = Path(llmcore.__file__).parent / "config" / "default_config.toml"
        with open(path, "rb") as fh:
            cfg = tomllib.load(fh)
        friendli = cfg["providers"]["friendli"]
        assert friendli["default_model"]
        assert friendli["endpoint_type"] == "serverless"

    def test_model_cards_are_registered(self):
        from llmcore.model_cards.registry import get_model_card_registry

        registry = get_model_card_registry()
        card = registry.get("friendli", "zai-org/GLM-5.3")
        assert card is not None
        assert card.get_context_length() == 1_048_576


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


class TestClose:
    async def test_close_is_idempotent(self, provider):
        provider._client.close = AsyncMock()
        await provider.close()
        await provider.close()
        assert provider._client is None

    async def test_close_swallows_errors(self, provider):
        provider._client.close = AsyncMock(side_effect=RuntimeError("boom"))
        await provider.close()
        assert provider._client is None
