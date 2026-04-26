# tests/providers/test_deepseek_provider.py
"""
Tests for the DeepSeek provider implementation.

Covers:
- Initialization and configuration
- Thinking mode parameter resolution
- Message payload building (including reasoning_content preservation)
- Chat completion (mocked)
- Response extraction (content, reasoning, tools, usage/cache)
- Streaming delta extraction
- Token counting
- Error handling
- Context length lookup
- Beta endpoint routing
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmcore.models import Message, Role, Tool, ToolCall

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_CONFIG: dict[str, Any] = {
    "api_key": "sk-test-key-000",
    "default_model": "deepseek-v4-pro",
    "timeout": 30,
    "thinking": "enabled",
    "reasoning_effort": "high",
}


@pytest.fixture
def provider():
    """Create a DeepSeekProvider with mocked OpenAI client."""
    with patch("llmcore.providers.deepseek_provider.AsyncOpenAI") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance
        from llmcore.providers.deepseek_provider import DeepSeekProvider

        p = DeepSeekProvider(MINIMAL_CONFIG.copy(), log_raw_payloads=False)
        # Ensure we can inspect the mock
        p._mock_client_cls = MockClient
        return p


@pytest.fixture
def disabled_thinking_provider():
    """Provider with thinking disabled by default."""
    cfg = MINIMAL_CONFIG.copy()
    cfg["thinking"] = "disabled"
    with patch("llmcore.providers.deepseek_provider.AsyncOpenAI"):
        from llmcore.providers.deepseek_provider import DeepSeekProvider

        return DeepSeekProvider(cfg, log_raw_payloads=False)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    """Test provider initialization and configuration."""

    def test_basic_init(self, provider):
        assert provider.get_name() == "deepseek"
        assert provider.default_model == "deepseek-v4-pro"
        assert provider._default_thinking == "enabled"
        assert provider._default_reasoning_effort == "high"

    def test_init_with_instance_name(self):
        cfg = MINIMAL_CONFIG.copy()
        cfg["_instance_name"] = "my-deepseek"
        with patch("llmcore.providers.deepseek_provider.AsyncOpenAI"):
            from llmcore.providers.deepseek_provider import DeepSeekProvider

            p = DeepSeekProvider(cfg)
            assert p.get_name() == "my-deepseek"

    def test_init_disabled_thinking(self, disabled_thinking_provider):
        assert disabled_thinking_provider._default_thinking == "disabled"

    def test_init_max_effort(self):
        cfg = MINIMAL_CONFIG.copy()
        cfg["reasoning_effort"] = "max"
        with patch("llmcore.providers.deepseek_provider.AsyncOpenAI"):
            from llmcore.providers.deepseek_provider import DeepSeekProvider

            p = DeepSeekProvider(cfg)
            assert p._default_reasoning_effort == "max"

    def test_init_effort_normalization(self):
        """low/medium → high, xhigh → max."""
        for raw, expected in [("low", "high"), ("medium", "high"), ("xhigh", "max")]:
            cfg = MINIMAL_CONFIG.copy()
            cfg["reasoning_effort"] = raw
            with patch("llmcore.providers.deepseek_provider.AsyncOpenAI"):
                from llmcore.providers.deepseek_provider import DeepSeekProvider

                p = DeepSeekProvider(cfg)
                assert p._default_reasoning_effort == expected, f"{raw} → {expected}"

    def test_init_invalid_thinking_defaults_to_enabled(self):
        cfg = MINIMAL_CONFIG.copy()
        cfg["thinking"] = "bogus"
        with patch("llmcore.providers.deepseek_provider.AsyncOpenAI"):
            from llmcore.providers.deepseek_provider import DeepSeekProvider

            p = DeepSeekProvider(cfg)
            assert p._default_thinking == "enabled"

    def test_init_missing_api_key_raises(self):
        cfg = {"default_model": "deepseek-v4-pro"}
        with patch("llmcore.providers.deepseek_provider.AsyncOpenAI"):
            with patch.dict("os.environ", {}, clear=True):
                from llmcore.exceptions import ConfigError
                from llmcore.providers.deepseek_provider import DeepSeekProvider

                with pytest.raises(ConfigError, match="API key not found"):
                    DeepSeekProvider(cfg)

    def test_init_api_key_from_env(self):
        cfg = {"default_model": "deepseek-v4-pro"}
        with patch("llmcore.providers.deepseek_provider.AsyncOpenAI"):
            with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-env-key"}):
                from llmcore.providers.deepseek_provider import DeepSeekProvider

                p = DeepSeekProvider(cfg)
                assert p.default_model == "deepseek-v4-pro"

    def test_creates_both_stable_and_beta_clients(self, provider):
        """Should create separate clients for stable and beta endpoints."""
        assert provider._client is not None
        assert provider._beta_client is not None


# ---------------------------------------------------------------------------
# Thinking parameter resolution
# ---------------------------------------------------------------------------


class TestThinkingResolution:
    """Test _resolve_thinking_params logic."""

    def test_default_thinking_enabled(self, provider):
        kwargs: dict[str, Any] = {}
        thinking, effort = provider._resolve_thinking_params(kwargs)
        assert thinking == {"type": "enabled"}
        assert effort == "high"

    def test_default_thinking_disabled(self, disabled_thinking_provider):
        kwargs: dict[str, Any] = {}
        thinking, effort = disabled_thinking_provider._resolve_thinking_params(kwargs)
        assert thinking == {"type": "disabled"}
        assert effort is None  # No effort when disabled

    def test_override_thinking_dict(self, provider):
        kwargs: dict[str, Any] = {"thinking": {"type": "disabled"}}
        thinking, effort = provider._resolve_thinking_params(kwargs)
        assert thinking == {"type": "disabled"}
        assert effort is None
        assert "thinking" not in kwargs  # Consumed

    def test_override_thinking_string(self, provider):
        kwargs: dict[str, Any] = {"thinking": "disabled"}
        thinking, effort = provider._resolve_thinking_params(kwargs)
        assert thinking == {"type": "disabled"}

    def test_override_thinking_bool(self, provider):
        kwargs: dict[str, Any] = {"thinking": False}
        thinking, effort = provider._resolve_thinking_params(kwargs)
        assert thinking == {"type": "disabled"}

        kwargs = {"thinking": True}
        thinking, effort = provider._resolve_thinking_params(kwargs)
        assert thinking == {"type": "enabled"}
        assert effort == "high"

    def test_override_effort_max(self, provider):
        kwargs: dict[str, Any] = {"reasoning_effort": "max"}
        thinking, effort = provider._resolve_thinking_params(kwargs)
        assert effort == "max"
        assert "reasoning_effort" not in kwargs

    def test_effort_normalization_xhigh(self, provider):
        kwargs: dict[str, Any] = {"reasoning_effort": "xhigh"}
        _, effort = provider._resolve_thinking_params(kwargs)
        assert effort == "max"


# ---------------------------------------------------------------------------
# Message payload building
# ---------------------------------------------------------------------------


class TestMessagePayload:
    """Test _build_message_payload."""

    def test_basic_user_message(self, provider):
        msg = Message(role=Role.USER, content="Hello")
        payload = provider._build_message_payload(msg)
        assert payload == {"role": "user", "content": "Hello"}

    def test_system_message(self, provider):
        msg = Message(role=Role.SYSTEM, content="You are helpful.")
        payload = provider._build_message_payload(msg)
        assert payload == {"role": "system", "content": "You are helpful."}

    def test_tool_message(self, provider):
        msg = Message(
            role=Role.TOOL,
            content='{"temp": 22}',
            tool_call_id="call_abc",
        )
        payload = provider._build_message_payload(msg)
        assert payload["role"] == "tool"
        assert payload["tool_call_id"] == "call_abc"

    def test_assistant_with_reasoning_content(self, provider):
        """reasoning_content must be preserved for thinking-mode tool-call turns."""
        msg = Message(
            role=Role.ASSISTANT,
            content="The answer is 42.",
            metadata={
                "reasoning_content": "Let me think step by step...",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "calc", "arguments": "{}"},
                    }
                ],
            },
        )
        payload = provider._build_message_payload(msg)
        assert payload["reasoning_content"] == "Let me think step by step..."
        assert payload["tool_calls"] is not None
        assert payload["content"] == "The answer is 42."

    def test_assistant_prefix_completion(self, provider):
        msg = Message(
            role=Role.ASSISTANT,
            content="```python\n",
            metadata={"prefix": True},
        )
        payload = provider._build_message_payload(msg)
        assert payload["prefix"] is True

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
    """Test response content/reasoning/tool extraction."""

    SAMPLE_RESPONSE: dict[str, Any] = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "deepseek-v4-pro",
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
            "prompt_cache_hit_tokens": 30,
            "prompt_cache_miss_tokens": 20,
            "completion_tokens": 100,
            "total_tokens": 150,
            "completion_tokens_details": {"reasoning_tokens": 60},
        },
    }

    def test_extract_content(self, provider):
        content = provider.extract_response_content(self.SAMPLE_RESPONSE)
        assert content == "The capital of France is Paris."

    def test_extract_reasoning_content(self, provider):
        reasoning = provider.extract_reasoning_content(self.SAMPLE_RESPONSE)
        assert reasoning == "The user is asking about geography..."

    def test_extract_reasoning_content_absent(self, provider):
        resp = {"choices": [{"message": {"content": "Hello"}}]}
        assert provider.extract_reasoning_content(resp) is None

    def test_extract_usage_details(self, provider):
        usage = provider.extract_usage_details(self.SAMPLE_RESPONSE)
        assert usage["prompt_tokens"] == 50
        assert usage["prompt_cache_hit_tokens"] == 30
        assert usage["prompt_cache_miss_tokens"] == 20
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
        assert calls[0].id == "call_1"

    def test_extract_tool_calls_invalid_json(self, provider):
        resp = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "f",
                                    "arguments": "not json",
                                },
                            }
                        ]
                    }
                }
            ]
        }
        calls = provider.extract_tool_calls(resp)
        assert len(calls) == 1
        assert calls[0].arguments == {"_raw": "not json"}

    def test_extract_content_empty_response(self, provider):
        assert provider.extract_response_content({}) == ""
        assert provider.extract_response_content({"choices": []}) == ""


# ---------------------------------------------------------------------------
# Streaming delta extraction
# ---------------------------------------------------------------------------


class TestStreamingDelta:
    """Test streaming chunk extraction."""

    def test_extract_delta_content(self, provider):
        chunk = {"choices": [{"delta": {"content": "Hello"}}]}
        assert provider.extract_delta_content(chunk) == "Hello"

    def test_extract_delta_reasoning_content(self, provider):
        chunk = {"choices": [{"delta": {"reasoning_content": "thinking..."}}]}
        assert provider.extract_delta_reasoning_content(chunk) == "thinking..."

    def test_extract_delta_empty(self, provider):
        assert provider.extract_delta_content({}) == ""
        assert provider.extract_delta_content({"choices": []}) == ""

    def test_extract_delta_reasoning_absent(self, provider):
        chunk = {"choices": [{"delta": {"content": "x"}}]}
        assert provider.extract_delta_reasoning_content(chunk) is None


# ---------------------------------------------------------------------------
# Context length
# ---------------------------------------------------------------------------


class TestContextLength:
    """Test get_max_context_length."""

    def test_v4_flash(self, provider):
        assert provider.get_max_context_length("deepseek-v4-flash") == 1_000_000

    def test_v4_pro(self, provider):
        assert provider.get_max_context_length("deepseek-v4-pro") == 1_000_000

    def test_legacy_chat(self, provider):
        assert provider.get_max_context_length("deepseek-chat") == 131_072

    def test_legacy_reasoner(self, provider):
        assert provider.get_max_context_length("deepseek-reasoner") == 131_072

    def test_default_model(self, provider):
        # default_model = deepseek-v4-pro → 1M
        assert provider.get_max_context_length() == 1_000_000

    def test_unknown_v4_model(self, provider):
        # Any model with "v4" should get 1M
        assert provider.get_max_context_length("deepseek-v4-something") == 1_000_000

    def test_completely_unknown_model(self, provider):
        # Should fall back to 131072
        assert provider.get_max_context_length("deepseek-future-xyz") == 131_072


# ---------------------------------------------------------------------------
# Supported parameters
# ---------------------------------------------------------------------------


class TestSupportedParameters:
    """Test get_supported_parameters."""

    def test_includes_thinking_params(self, provider):
        params = provider.get_supported_parameters()
        assert "thinking" in params
        assert "reasoning_effort" in params

    def test_includes_standard_params(self, provider):
        params = provider.get_supported_parameters()
        for key in ("temperature", "top_p", "max_tokens", "response_format", "stop", "logprobs"):
            assert key in params, f"Missing parameter: {key}"


# ---------------------------------------------------------------------------
# Chat completion (mocked)
# ---------------------------------------------------------------------------


class TestChatCompletion:
    """Test chat_completion with mocked API calls."""

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

        # Verify thinking was injected
        call_kwargs = provider._client.chat.completions.create.call_args
        extra = call_kwargs.kwargs.get("extra_body", {})
        assert extra.get("thinking") == {"type": "enabled"}
        assert extra.get("reasoning_effort") == "high"

    @pytest.mark.asyncio
    async def test_completion_thinking_disabled(self, disabled_thinking_provider):
        mock_resp = MagicMock()
        mock_resp.model_dump.return_value = {
            "choices": [{"message": {"content": "OK"}}],
            "usage": {},
        }
        disabled_thinking_provider._client.chat.completions.create = AsyncMock(
            return_value=mock_resp
        )

        context = [Message(role=Role.USER, content="Hi")]
        await disabled_thinking_provider.chat_completion(context)

        call_kwargs = disabled_thinking_provider._client.chat.completions.create.call_args
        extra = call_kwargs.kwargs.get("extra_body", {})
        assert extra.get("thinking") == {"type": "disabled"}
        assert "reasoning_effort" not in extra

    @pytest.mark.asyncio
    async def test_per_request_thinking_override(self, provider):
        mock_resp = MagicMock()
        mock_resp.model_dump.return_value = {
            "choices": [{"message": {"content": "OK"}}],
            "usage": {},
        }
        provider._client.chat.completions.create = AsyncMock(return_value=mock_resp)

        context = [Message(role=Role.USER, content="Hi")]
        await provider.chat_completion(context, thinking=False, reasoning_effort="max")

        call_kwargs = provider._client.chat.completions.create.call_args
        extra = call_kwargs.kwargs.get("extra_body", {})
        # thinking=False → disabled, so no reasoning_effort
        assert extra.get("thinking") == {"type": "disabled"}
        assert "reasoning_effort" not in extra

    @pytest.mark.asyncio
    async def test_sampling_params_skipped_in_thinking_mode(self, provider):
        mock_resp = MagicMock()
        mock_resp.model_dump.return_value = {
            "choices": [{"message": {"content": "OK"}}],
            "usage": {},
        }
        provider._client.chat.completions.create = AsyncMock(return_value=mock_resp)

        context = [Message(role=Role.USER, content="Hi")]
        await provider.chat_completion(context, temperature=0.7, top_p=0.9)

        call_kwargs = provider._client.chat.completions.create.call_args
        # temperature/top_p should NOT be in the call when thinking is enabled
        assert "temperature" not in call_kwargs.kwargs
        assert "top_p" not in call_kwargs.kwargs

    @pytest.mark.asyncio
    async def test_sampling_params_kept_when_thinking_disabled(self, disabled_thinking_provider):
        mock_resp = MagicMock()
        mock_resp.model_dump.return_value = {
            "choices": [{"message": {"content": "OK"}}],
            "usage": {},
        }
        disabled_thinking_provider._client.chat.completions.create = AsyncMock(
            return_value=mock_resp
        )

        context = [Message(role=Role.USER, content="Hi")]
        await disabled_thinking_provider.chat_completion(context, temperature=0.7)

        call_kwargs = disabled_thinking_provider._client.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("temperature") == 0.7

    @pytest.mark.asyncio
    async def test_tools_payload(self, provider):
        mock_resp = MagicMock()
        mock_resp.model_dump.return_value = {
            "choices": [{"message": {"content": "OK"}}],
            "usage": {},
        }
        provider._client.chat.completions.create = AsyncMock(return_value=mock_resp)

        tool = Tool(
            name="get_weather",
            description="Get weather",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        )
        context = [Message(role=Role.USER, content="Weather in Paris?")]
        await provider.chat_completion(context, tools=[tool], tool_choice="auto")

        call_kwargs = provider._client.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("tool_choice") == "auto"
        tools_api = call_kwargs.kwargs.get("tools")
        assert len(tools_api) == 1
        assert tools_api[0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_beta_endpoint_for_prefix(self):
        """Prefix completion should route to beta client."""
        with patch("llmcore.providers.deepseek_provider.AsyncOpenAI") as MockCls:
            # Create distinct mocks for stable and beta
            stable_mock = MagicMock()
            beta_mock = MagicMock()
            MockCls.side_effect = [stable_mock, beta_mock]

            from llmcore.providers.deepseek_provider import DeepSeekProvider

            p = DeepSeekProvider(MINIMAL_CONFIG.copy())

            mock_resp = MagicMock()
            mock_resp.model_dump.return_value = {
                "choices": [{"message": {"content": "OK"}}],
                "usage": {},
            }
            stable_mock.chat.completions.create = AsyncMock(return_value=mock_resp)
            beta_mock.chat.completions.create = AsyncMock(return_value=mock_resp)

            context = [
                Message(role=Role.USER, content="Write fibonacci"),
                Message(role=Role.ASSISTANT, content="```python\n", metadata={"prefix": True}),
            ]
            await p.chat_completion(context)

            # Should have used beta client, not stable
            beta_mock.chat.completions.create.assert_called_once()
            stable_mock.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_unsupported_param_raises(self, provider):
        context = [Message(role=Role.USER, content="Hi")]
        with pytest.raises(ValueError, match="Unsupported parameter"):
            await provider.chat_completion(context, bogus_param=True)

    @pytest.mark.asyncio
    async def test_streaming_includes_usage(self, provider):
        """Stream requests should include stream_options.include_usage."""
        mock_resp = MagicMock()

        async def mock_stream():
            yield MagicMock(
                model_dump=lambda exclude_none=True: {"choices": [{"delta": {"content": "h"}}]}
            )

        provider._client.chat.completions.create = AsyncMock(return_value=mock_stream())

        context = [Message(role=Role.USER, content="Hi")]
        await provider.chat_completion(context, stream=True)

        call_kwargs = provider._client.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("stream_options") == {"include_usage": True}


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


class TestTokenCounting:
    """Test token counting methods."""

    @pytest.mark.asyncio
    async def test_count_tokens_with_encoding(self, provider):
        """If tiktoken is available, should use it."""
        if provider._encoding is None:
            pytest.skip("tiktoken not available")
        count = await provider.count_tokens("Hello, world!")
        assert isinstance(count, int)
        assert count > 0

    @pytest.mark.asyncio
    async def test_count_tokens_fallback(self):
        """Without tiktoken, falls back to heuristic."""
        with patch("llmcore.providers.deepseek_provider.AsyncOpenAI"):
            with patch("llmcore.providers.deepseek_provider.tiktoken_available", False):
                with patch("llmcore.providers.deepseek_provider.tiktoken", None):
                    from llmcore.providers.deepseek_provider import DeepSeekProvider

                    p = DeepSeekProvider(MINIMAL_CONFIG.copy())
                    count = await p.count_tokens("Hello, world!")
                    assert isinstance(count, int)
                    assert count > 0

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
    """Test resource cleanup."""

    @pytest.mark.asyncio
    async def test_close_cleans_up(self, provider):
        provider._client.close = AsyncMock()
        provider._beta_client.close = AsyncMock()
        await provider.close()
        provider._client is None
        provider._beta_client is None


# ---------------------------------------------------------------------------
# Provider in PROVIDER_MAP
# ---------------------------------------------------------------------------


class TestProviderRegistration:
    """Test that DeepSeekProvider is correctly registered."""

    def test_provider_map_has_deepseek(self):
        from llmcore.providers.deepseek_provider import DeepSeekProvider
        from llmcore.providers.manager import PROVIDER_MAP

        assert PROVIDER_MAP.get("deepseek") is DeepSeekProvider

    def test_deepseek_not_in_compat_defaults(self):
        from llmcore.providers.manager import _OPENAI_COMPATIBLE_DEFAULTS

        assert "deepseek" not in _OPENAI_COMPATIBLE_DEFAULTS
