# tests/providers/test_anthropic_provider.py
"""
Comprehensive tests for the Anthropic provider.

Covers:
- Initialization and configuration
- Message conversion (text, system, multimodal, tool results)
- Tool schema conversion (parameters → input_schema)
- Tool choice mapping
- Response normalization (non-streaming)
- Streaming normalization
- Tool call extraction
- Stop reason normalization
- Usage normalization
- Token counting
- Dynamic model discovery
- Error handling (rate limit, auth, bad request, overloaded)
- Vision/multimodal content (images, documents)
- Extended thinking support
- Prompt caching (cache_control)
"""

from __future__ import annotations

import json
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure anthropic SDK is mocked if not installed
try:
    import anthropic  # noqa: F401
except ImportError:
    pytest.skip("anthropic SDK not installed", allow_module_level=True)

from llmcore.exceptions import ConfigError, ContextLengthError, ProviderError
from llmcore.models import Message, Tool, ToolCall
from llmcore.models import Role as LLMCoreRole
from llmcore.providers.anthropic_provider import (
    DEFAULT_ANTHROPIC_TOKEN_LIMITS,
    DEFAULT_MODEL,
    AnthropicProvider,
)

# ====================================================================
# Fixtures
# ====================================================================


@pytest.fixture
def base_config() -> dict[str, Any]:
    """Minimal valid configuration."""
    return {"api_key": "test-key-123", "default_model": "claude-sonnet-4-5-20250929"}


@pytest.fixture
def provider(base_config: dict[str, Any]) -> AnthropicProvider:
    """Create an AnthropicProvider with mocked client."""
    with patch("llmcore.providers.anthropic_provider.AsyncAnthropic") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        p = AnthropicProvider(base_config)
        p._client = mock_client
        return p


@pytest.fixture
def simple_messages() -> list[Message]:
    """A simple user message list."""
    return [
        Message(role=LLMCoreRole.USER, content="Hello, Claude."),
    ]


@pytest.fixture
def messages_with_system() -> list[Message]:
    """Messages with a system prompt."""
    return [
        Message(role=LLMCoreRole.SYSTEM, content="You are helpful."),
        Message(role=LLMCoreRole.USER, content="What is 2+2?"),
    ]


@pytest.fixture
def multi_turn_messages() -> list[Message]:
    """Multi-turn conversation."""
    return [
        Message(role=LLMCoreRole.SYSTEM, content="Be concise."),
        Message(role=LLMCoreRole.USER, content="Hi"),
        Message(role=LLMCoreRole.ASSISTANT, content="Hello!"),
        Message(role=LLMCoreRole.USER, content="What is Python?"),
    ]


@pytest.fixture
def tool_call_messages() -> list[Message]:
    """Messages with tool call and result for multi-turn tool use."""
    return [
        Message(role=LLMCoreRole.USER, content="What's the weather?"),
        Message(
            role=LLMCoreRole.ASSISTANT,
            content="",
            metadata={
                "tool_calls": [
                    {
                        "type": "tool_use",
                        "id": "call_123",
                        "name": "get_weather",
                        "input": {"location": "London"},
                    }
                ]
            },
        ),
        Message(
            role=LLMCoreRole.TOOL,
            content='{"temp": 15, "condition": "cloudy"}',
            tool_call_id="call_123",
        ),
    ]


@pytest.fixture
def sample_tools() -> list[Tool]:
    """Sample tools for testing."""
    return [
        Tool(
            name="get_weather",
            description="Get the weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"],
            },
        ),
    ]


# ====================================================================
# Initialization Tests
# ====================================================================


class TestInitialization:
    """Tests for provider initialization and configuration."""

    def test_init_with_api_key(self, base_config: dict[str, Any]):
        """Provider initializes with explicit api_key."""
        with patch("llmcore.providers.anthropic_provider.AsyncAnthropic"):
            p = AnthropicProvider(base_config)
            assert p.api_key == "test-key-123"
            assert p.default_model == "claude-sonnet-4-5-20250929"

    def test_init_missing_api_key_raises(self):
        """Provider raises ConfigError when no API key is available."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ConfigError, match="API key not found"):
                AnthropicProvider({"default_model": "claude-sonnet-4-5-20250929"})

    def test_init_with_env_var(self):
        """Provider reads API key from environment variable."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key-456"}):
            with patch("llmcore.providers.anthropic_provider.AsyncAnthropic"):
                p = AnthropicProvider({})
                assert p.api_key == "env-key-456"

    def test_init_with_custom_env_var(self):
        """Provider reads API key from custom env var."""
        with patch.dict("os.environ", {"MY_CLAUDE_KEY": "custom-key-789"}):
            with patch("llmcore.providers.anthropic_provider.AsyncAnthropic"):
                p = AnthropicProvider({"api_key_env_var": "MY_CLAUDE_KEY"})
                assert p.api_key == "custom-key-789"

    def test_init_custom_timeout(self):
        """Provider accepts custom timeout."""
        with patch("llmcore.providers.anthropic_provider.AsyncAnthropic"):
            p = AnthropicProvider({"api_key": "k", "timeout": 300})
            assert p.timeout == 300.0

    def test_get_name_default(self, provider: AnthropicProvider):
        """Default name is 'anthropic'."""
        assert provider.get_name() == "anthropic"

    def test_get_name_custom(self, base_config: dict[str, Any]):
        """Custom instance name is returned."""
        base_config["_instance_name"] = "my_claude"
        with patch("llmcore.providers.anthropic_provider.AsyncAnthropic"):
            p = AnthropicProvider(base_config)
            assert p.get_name() == "my_claude"

    def test_default_model_is_current(self):
        """Default model should be a current model, not deprecated."""
        assert DEFAULT_MODEL == "claude-sonnet-4-5-20250929"
        assert "claude-3-haiku" not in DEFAULT_MODEL

    def test_default_token_limits_has_modern_models(self):
        """Token limits table includes Claude 4.x models."""
        assert "claude-sonnet-4-5-20250929" in DEFAULT_ANTHROPIC_TOKEN_LIMITS
        assert "claude-opus-4-6" in DEFAULT_ANTHROPIC_TOKEN_LIMITS
        assert "claude-haiku-4-5-20251001" in DEFAULT_ANTHROPIC_TOKEN_LIMITS


# ====================================================================
# Message Conversion Tests
# ====================================================================


class TestMessageConversion:
    """Tests for _convert_llmcore_msgs_to_anthropic."""

    def test_simple_user_message(self, provider: AnthropicProvider):
        """Single user message converts correctly."""
        msgs = [Message(role=LLMCoreRole.USER, content="Hello")]
        system, api_msgs = provider._convert_llmcore_msgs_to_anthropic(msgs)

        assert system is None
        assert len(api_msgs) == 1
        assert api_msgs[0]["role"] == "user"
        assert api_msgs[0]["content"] == [{"type": "text", "text": "Hello"}]

    def test_system_prompt_extraction(
        self, provider: AnthropicProvider, messages_with_system: list[Message]
    ):
        """System prompt is extracted as a separate string."""
        system, api_msgs = provider._convert_llmcore_msgs_to_anthropic(messages_with_system)

        assert system == "You are helpful."
        assert len(api_msgs) == 1
        assert api_msgs[0]["role"] == "user"

    def test_system_prompt_with_cache_control(self, provider: AnthropicProvider):
        """System prompt with cache_control becomes array of text blocks."""
        msgs = [
            Message(
                role=LLMCoreRole.SYSTEM,
                content="Cached system prompt",
                metadata={"cache_control": {"type": "ephemeral"}},
            ),
            Message(role=LLMCoreRole.USER, content="Hi"),
        ]
        system, api_msgs = provider._convert_llmcore_msgs_to_anthropic(msgs)

        assert isinstance(system, list)
        assert system[0]["type"] == "text"
        assert system[0]["text"] == "Cached system prompt"
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    def test_multi_turn_conversation(
        self, provider: AnthropicProvider, multi_turn_messages: list[Message]
    ):
        """Multi-turn conversation preserves alternating roles."""
        system, api_msgs = provider._convert_llmcore_msgs_to_anthropic(multi_turn_messages)

        assert system == "Be concise."
        assert len(api_msgs) == 3
        assert api_msgs[0]["role"] == "user"
        assert api_msgs[1]["role"] == "assistant"
        assert api_msgs[2]["role"] == "user"

    def test_consecutive_same_role_merged(self, provider: AnthropicProvider):
        """Consecutive same-role messages are merged."""
        msgs = [
            Message(role=LLMCoreRole.USER, content="First"),
            Message(role=LLMCoreRole.USER, content="Second"),
        ]
        system, api_msgs = provider._convert_llmcore_msgs_to_anthropic(msgs)

        assert len(api_msgs) == 1
        assert len(api_msgs[0]["content"]) == 2
        assert api_msgs[0]["content"][0]["text"] == "First"
        assert api_msgs[0]["content"][1]["text"] == "Second"

    def test_tool_result_becomes_user_message(
        self, provider: AnthropicProvider, tool_call_messages: list[Message]
    ):
        """Tool results are wrapped in user messages with tool_result blocks."""
        system, api_msgs = provider._convert_llmcore_msgs_to_anthropic(tool_call_messages)

        assert len(api_msgs) == 3
        # First: user
        assert api_msgs[0]["role"] == "user"
        # Second: assistant with tool_use
        assert api_msgs[1]["role"] == "assistant"
        assert api_msgs[1]["content"][0]["type"] == "tool_use"
        assert api_msgs[1]["content"][0]["name"] == "get_weather"
        # Third: user with tool_result
        assert api_msgs[2]["role"] == "user"
        assert api_msgs[2]["content"][0]["type"] == "tool_result"
        assert api_msgs[2]["content"][0]["tool_use_id"] == "call_123"

    def test_tool_result_error_flag(self, provider: AnthropicProvider):
        """Tool result with is_error metadata is flagged."""
        msgs = [
            Message(role=LLMCoreRole.USER, content="Do something"),
            Message(
                role=LLMCoreRole.ASSISTANT,
                content="",
                metadata={
                    "tool_calls": [
                        {
                            "type": "tool_use",
                            "id": "tc1",
                            "name": "do_thing",
                            "input": {},
                        }
                    ]
                },
            ),
            Message(
                role=LLMCoreRole.TOOL,
                content="Error: not found",
                tool_call_id="tc1",
                metadata={"is_error": True},
            ),
        ]
        _, api_msgs = provider._convert_llmcore_msgs_to_anthropic(msgs)

        tool_result = api_msgs[2]["content"][0]
        assert tool_result["is_error"] is True

    def test_assistant_tool_calls_openai_format(self, provider: AnthropicProvider):
        """Assistant messages with OpenAI-format tool_calls are converted."""
        msgs = [
            Message(role=LLMCoreRole.USER, content="Hi"),
            Message(
                role=LLMCoreRole.ASSISTANT,
                content="Let me check.",
                metadata={
                    "tool_calls": [
                        {
                            "id": "tc_1",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": '{"q": "test"}',
                            },
                        }
                    ]
                },
            ),
        ]
        _, api_msgs = provider._convert_llmcore_msgs_to_anthropic(msgs)

        asst = api_msgs[1]
        assert asst["role"] == "assistant"
        assert asst["content"][0]["type"] == "text"
        assert asst["content"][0]["text"] == "Let me check."
        assert asst["content"][1]["type"] == "tool_use"
        assert asst["content"][1]["name"] == "search"
        assert asst["content"][1]["input"] == {"q": "test"}


# ====================================================================
# Multimodal Content Tests
# ====================================================================


class TestMultimodalContent:
    """Tests for vision and document content handling."""

    def test_image_base64(self, provider: AnthropicProvider):
        """Base64 image in metadata produces ImageBlockParam."""
        msgs = [
            Message(
                role=LLMCoreRole.USER,
                content="What's in this image?",
                metadata={
                    "inline_images": [
                        {
                            "data": "iVBORw0KGgo=",
                            "media_type": "image/png",
                        }
                    ]
                },
            )
        ]
        _, api_msgs = provider._convert_llmcore_msgs_to_anthropic(msgs)

        blocks = api_msgs[0]["content"]
        assert len(blocks) == 2
        assert blocks[0]["type"] == "text"
        assert blocks[1]["type"] == "image"
        assert blocks[1]["source"]["type"] == "base64"
        assert blocks[1]["source"]["media_type"] == "image/png"

    def test_image_url(self, provider: AnthropicProvider):
        """URL image in metadata produces URL ImageBlockParam."""
        msgs = [
            Message(
                role=LLMCoreRole.USER,
                content="Describe this",
                metadata={"inline_images": [{"url": "https://example.com/photo.jpg"}]},
            )
        ]
        _, api_msgs = provider._convert_llmcore_msgs_to_anthropic(msgs)

        img_block = api_msgs[0]["content"][1]
        assert img_block["source"]["type"] == "url"
        assert img_block["source"]["url"] == "https://example.com/photo.jpg"

    def test_image_url_string(self, provider: AnthropicProvider):
        """Plain string URL image is handled."""
        msgs = [
            Message(
                role=LLMCoreRole.USER,
                content="Look",
                metadata={"inline_images": ["https://example.com/img.png"]},
            )
        ]
        _, api_msgs = provider._convert_llmcore_msgs_to_anthropic(msgs)

        img_block = api_msgs[0]["content"][1]
        assert img_block["source"]["type"] == "url"
        assert img_block["source"]["url"] == "https://example.com/img.png"

    def test_pdf_document(self, provider: AnthropicProvider):
        """PDF document in metadata produces DocumentBlockParam."""
        msgs = [
            Message(
                role=LLMCoreRole.USER,
                content="Summarize this PDF",
                metadata={
                    "inline_documents": [
                        {
                            "data": "JVBERi0xLjQ=",
                            "media_type": "application/pdf",
                            "title": "Report",
                        }
                    ]
                },
            )
        ]
        _, api_msgs = provider._convert_llmcore_msgs_to_anthropic(msgs)

        doc_block = api_msgs[0]["content"][1]
        assert doc_block["type"] == "document"
        assert doc_block["source"]["type"] == "base64"
        assert doc_block["source"]["media_type"] == "application/pdf"
        assert doc_block["title"] == "Report"

    def test_plain_text_document(self, provider: AnthropicProvider):
        """Plain text document in metadata."""
        msgs = [
            Message(
                role=LLMCoreRole.USER,
                content="Analyze",
                metadata={
                    "inline_documents": [
                        {
                            "data": "Some long text...",
                            "media_type": "text/plain",
                            "title": "Notes",
                        }
                    ]
                },
            )
        ]
        _, api_msgs = provider._convert_llmcore_msgs_to_anthropic(msgs)

        doc_block = api_msgs[0]["content"][1]
        assert doc_block["source"]["type"] == "text"
        assert doc_block["source"]["media_type"] == "text/plain"

    def test_content_parts_passthrough(self, provider: AnthropicProvider):
        """Pre-constructed content_parts are passed through directly."""
        custom_parts = [
            {"type": "text", "text": "Custom block"},
            {"type": "image", "source": {"type": "url", "url": "https://x.com/i.png"}},
        ]
        msgs = [
            Message(
                role=LLMCoreRole.USER,
                content="ignored",
                metadata={"content_parts": custom_parts},
            )
        ]
        _, api_msgs = provider._convert_llmcore_msgs_to_anthropic(msgs)

        assert api_msgs[0]["content"] == custom_parts

    def test_cache_control_on_last_block(self, provider: AnthropicProvider):
        """cache_control is applied to the last content block."""
        msgs = [
            Message(
                role=LLMCoreRole.USER,
                content="Hello",
                metadata={"cache_control": {"type": "ephemeral"}},
            )
        ]
        _, api_msgs = provider._convert_llmcore_msgs_to_anthropic(msgs)

        last_block = api_msgs[0]["content"][-1]
        assert last_block["cache_control"] == {"type": "ephemeral"}


# ====================================================================
# Tool Conversion Tests
# ====================================================================


class TestToolConversion:
    """Tests for tool schema and tool_choice conversion."""

    def test_tool_schema_uses_input_schema(self, sample_tools: list[Tool]):
        """Tool.parameters becomes input_schema (not parameters)."""
        result = AnthropicProvider._convert_tools_to_anthropic(sample_tools)

        assert len(result) == 1
        assert "input_schema" in result[0]
        assert "parameters" not in result[0]
        assert result[0]["name"] == "get_weather"
        assert result[0]["description"] == "Get the weather for a location"
        assert result[0]["input_schema"]["type"] == "object"

    def test_tool_choice_auto(self):
        """'auto' maps correctly."""
        assert AnthropicProvider._convert_tool_choice("auto") == {"type": "auto"}

    def test_tool_choice_any(self):
        """'any' maps correctly."""
        assert AnthropicProvider._convert_tool_choice("any") == {"type": "any"}

    def test_tool_choice_none(self):
        """'none' maps correctly."""
        assert AnthropicProvider._convert_tool_choice("none") == {"type": "none"}

    def test_tool_choice_specific_name(self):
        """Specific tool name maps correctly."""
        result = AnthropicProvider._convert_tool_choice("get_weather")
        assert result == {"type": "tool", "name": "get_weather"}


# ====================================================================
# Response Normalization Tests
# ====================================================================


class TestResponseNormalization:
    """Tests for response normalization."""

    def test_normalize_tool_calls_from_content(self):
        """tool_use blocks are normalized to OpenAI format."""
        blocks = [
            {"type": "text", "text": "Using tool..."},
            {
                "type": "tool_use",
                "id": "tc_1",
                "name": "search",
                "input": {"query": "test"},
            },
        ]
        result = AnthropicProvider._normalize_tool_calls_from_content(blocks)

        assert result is not None
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["id"] == "tc_1"
        assert result[0]["function"]["name"] == "search"
        assert json.loads(result[0]["function"]["arguments"]) == {"query": "test"}

    def test_normalize_tool_calls_no_tools(self):
        """No tool_use blocks returns None."""
        blocks = [{"type": "text", "text": "Just text"}]
        assert AnthropicProvider._normalize_tool_calls_from_content(blocks) is None

    def test_normalize_stop_reason_mapping(self):
        """Stop reasons are mapped correctly."""
        assert AnthropicProvider._normalize_stop_reason("end_turn") == "stop"
        assert AnthropicProvider._normalize_stop_reason("tool_use") == "tool_calls"
        assert AnthropicProvider._normalize_stop_reason("max_tokens") == "length"
        assert AnthropicProvider._normalize_stop_reason("stop_sequence") == "stop"
        assert AnthropicProvider._normalize_stop_reason("pause_turn") == "stop"
        assert AnthropicProvider._normalize_stop_reason("refusal") == "content_filter"
        assert AnthropicProvider._normalize_stop_reason(None) is None

    def test_normalize_usage(self):
        """Usage dict gets OpenAI-compatible aliases."""
        raw = {"input_tokens": 100, "output_tokens": 50}
        result = AnthropicProvider._normalize_usage(raw)

        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 50
        assert result["total_tokens"] == 150

    def test_normalize_usage_with_cache(self):
        """Usage with cache fields preserves them."""
        raw = {
            "input_tokens": 50,
            "output_tokens": 100,
            "cache_read_input_tokens": 1000,
            "cache_creation_input_tokens": 200,
        }
        result = AnthropicProvider._normalize_usage(raw)

        assert result["prompt_tokens"] == 50
        assert result["cache_read_input_tokens"] == 1000
        assert result["total_tokens"] == 150


# ====================================================================
# Chat Completion Tests (Non-Streaming)
# ====================================================================


class TestChatCompletion:
    """Tests for non-streaming chat_completion."""

    @pytest.mark.asyncio
    async def test_basic_completion(
        self,
        provider: AnthropicProvider,
        simple_messages: list[Message],
    ):
        """Basic non-streaming completion returns normalized response."""
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "id": "msg_123",
            "model": "claude-sonnet-4-5-20250929",
            "content": [{"type": "text", "text": "Hello! How can I help?"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 8},
        }
        provider._client.messages.create = AsyncMock(return_value=mock_response)

        result = await provider.chat_completion(simple_messages)

        assert result["id"] == "msg_123"
        assert result["choices"][0]["message"]["content"] == "Hello! How can I help?"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 8
        assert result["usage"]["total_tokens"] == 18

    @pytest.mark.asyncio
    async def test_tool_call_response(
        self,
        provider: AnthropicProvider,
        simple_messages: list[Message],
        sample_tools: list[Tool],
    ):
        """Response with tool_use blocks is normalized correctly."""
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "id": "msg_456",
            "model": "claude-sonnet-4-5-20250929",
            "content": [
                {"type": "text", "text": "Let me check the weather."},
                {
                    "type": "tool_use",
                    "id": "tc_abc",
                    "name": "get_weather",
                    "input": {"location": "London"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 50, "output_tokens": 30},
        }
        provider._client.messages.create = AsyncMock(return_value=mock_response)

        result = await provider.chat_completion(simple_messages, tools=sample_tools)

        assert result["choices"][0]["finish_reason"] == "tool_calls"
        tc = result["choices"][0]["message"]["tool_calls"]
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "get_weather"

        # Verify tools were converted with input_schema
        call_kwargs = provider._client.messages.create.call_args
        assert "tools" in call_kwargs.kwargs
        assert "input_schema" in call_kwargs.kwargs["tools"][0]

    @pytest.mark.asyncio
    async def test_thinking_response(
        self,
        provider: AnthropicProvider,
        simple_messages: list[Message],
    ):
        """Response with thinking blocks propagates thinking content."""
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "id": "msg_789",
            "model": "claude-sonnet-4-5-20250929",
            "content": [
                {"type": "thinking", "thinking": "Let me think about this..."},
                {"type": "text", "text": "The answer is 4."},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 20, "output_tokens": 100},
        }
        provider._client.messages.create = AsyncMock(return_value=mock_response)

        result = await provider.chat_completion(
            simple_messages,
            thinking={"type": "enabled", "budget_tokens": 5000},
        )

        msg = result["choices"][0]["message"]
        assert msg["content"] == "The answer is 4."
        assert msg["thinking"] == "Let me think about this..."

    @pytest.mark.asyncio
    async def test_max_tokens_default(
        self,
        provider: AnthropicProvider,
        simple_messages: list[Message],
    ):
        """max_tokens defaults to 4096 when not specified."""
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "id": "msg_def",
            "model": "claude-sonnet-4-5-20250929",
            "content": [{"type": "text", "text": "Ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }
        provider._client.messages.create = AsyncMock(return_value=mock_response)

        await provider.chat_completion(simple_messages)

        call_kwargs = provider._client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_metadata_passthrough(
        self,
        provider: AnthropicProvider,
        simple_messages: list[Message],
    ):
        """metadata kwarg is passed to API."""
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "id": "msg_meta",
            "model": "claude-sonnet-4-5-20250929",
            "content": [{"type": "text", "text": "Ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }
        provider._client.messages.create = AsyncMock(return_value=mock_response)

        await provider.chat_completion(simple_messages, metadata={"user_id": "usr_test123"})

        call_kwargs = provider._client.messages.create.call_args.kwargs
        assert call_kwargs["metadata"] == {"user_id": "usr_test123"}


# ====================================================================
# Streaming Tests
# ====================================================================


class TestStreaming:
    """Tests for streaming chat_completion."""

    @pytest.mark.asyncio
    async def test_streaming_text(
        self, provider: AnthropicProvider, simple_messages: list[Message]
    ):
        """Streaming yields text delta chunks."""

        async def mock_stream():
            events = [
                {"type": "message_start", "message": {"usage": {"input_tokens": 5}}},
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "Hello"},
                },
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": " world"},
                },
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 5},
                },
                {"type": "message_stop"},
            ]
            for e in events:
                mock_event = MagicMock()
                mock_event.model_dump.return_value = e
                yield mock_event

        provider._client.messages.create = AsyncMock(return_value=mock_stream())

        gen = await provider.chat_completion(simple_messages, stream=True)

        chunks = []
        async for chunk in gen:
            chunks.append(chunk)

        # Should have text deltas with choices
        text_chunks = [
            c for c in chunks if c.get("type") == "content_block_delta" and c.get("choices")
        ]
        assert len(text_chunks) == 2

        text = "".join(provider.extract_delta_content(c) for c in chunks)
        assert text == "Hello world"

    @pytest.mark.asyncio
    async def test_streaming_tool_calls(
        self, provider: AnthropicProvider, simple_messages: list[Message]
    ):
        """Streaming handles tool call events."""

        async def mock_stream():
            events = [
                {"type": "message_start", "message": {"usage": {"input_tokens": 10}}},
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "tc_s1",
                        "name": "search",
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": '{"q":',
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": '"test"}',
                    },
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use"},
                    "usage": {"output_tokens": 15},
                },
                {"type": "message_stop"},
            ]
            for e in events:
                mock_event = MagicMock()
                mock_event.model_dump.return_value = e
                yield mock_event

        provider._client.messages.create = AsyncMock(return_value=mock_stream())

        gen = await provider.chat_completion(simple_messages, stream=True)

        chunks = []
        async for chunk in gen:
            chunks.append(chunk)

        # Should have tool call delta chunks
        tool_chunks = [
            c
            for c in chunks
            if c.get("choices") and c["choices"][0].get("delta", {}).get("tool_calls")
        ]
        assert len(tool_chunks) >= 2  # start + json deltas


# ====================================================================
# Tool Call Extraction Tests
# ====================================================================


class TestExtractToolCalls:
    """Tests for extract_tool_calls method."""

    def test_extract_from_normalized_response(self, provider: AnthropicProvider):
        """Extracts ToolCall objects from normalized response."""
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "tc_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "London"}',
                                },
                            }
                        ]
                    }
                }
            ]
        }
        calls = provider.extract_tool_calls(response)

        assert len(calls) == 1
        assert isinstance(calls[0], ToolCall)
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {"location": "London"}
        assert calls[0].id == "tc_1"

    def test_extract_no_tool_calls(self, provider: AnthropicProvider):
        """Returns empty list when no tool calls."""
        response = {"choices": [{"message": {"content": "Just text"}}]}
        assert provider.extract_tool_calls(response) == []

    def test_extract_handles_malformed_json(self, provider: AnthropicProvider):
        """Handles malformed JSON arguments gracefully."""
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "tc_2",
                                "type": "function",
                                "function": {
                                    "name": "broken",
                                    "arguments": "{invalid json",
                                },
                            }
                        ]
                    }
                }
            ]
        }
        calls = provider.extract_tool_calls(response)
        assert len(calls) == 1
        assert calls[0].arguments == {"raw": "{invalid json"}


# ====================================================================
# Token Counting Tests
# ====================================================================


class TestTokenCounting:
    """Tests for token counting methods."""

    @pytest.mark.asyncio
    async def test_count_tokens_api(self, provider: AnthropicProvider):
        """count_tokens uses the messages.count_tokens API."""
        mock_result = MagicMock()
        mock_result.input_tokens = 42
        provider._client.messages.count_tokens = AsyncMock(return_value=mock_result)

        count = await provider.count_tokens("Hello world")

        assert count == 42
        provider._client.messages.count_tokens.assert_called_once()

    @pytest.mark.asyncio
    async def test_count_tokens_fallback(self, provider: AnthropicProvider):
        """count_tokens falls back to approximation on API failure."""
        provider._client.messages.count_tokens = AsyncMock(side_effect=Exception("API error"))

        count = await provider.count_tokens("Hello world test")

        assert count == (len("Hello world test") + 3) // 4

    @pytest.mark.asyncio
    async def test_count_tokens_empty(self, provider: AnthropicProvider):
        """count_tokens returns 0 for empty string."""
        assert await provider.count_tokens("") == 0

    @pytest.mark.asyncio
    async def test_count_message_tokens(self, provider: AnthropicProvider):
        """count_message_tokens uses messages.count_tokens API."""
        mock_result = MagicMock()
        mock_result.input_tokens = 120
        provider._client.messages.count_tokens = AsyncMock(return_value=mock_result)

        msgs = [
            Message(role=LLMCoreRole.SYSTEM, content="Be helpful."),
            Message(role=LLMCoreRole.USER, content="Tell me about Python"),
        ]
        count = await provider.count_message_tokens(msgs)

        assert count == 120
        call_kwargs = provider._client.messages.count_tokens.call_args.kwargs
        assert "system" in call_kwargs
        assert call_kwargs["system"] == "Be helpful."


# ====================================================================
# Model Discovery Tests
# ====================================================================


class TestModelDiscovery:
    """Tests for dynamic model discovery."""

    @pytest.mark.asyncio
    async def test_dynamic_discovery(self, provider: AnthropicProvider):
        """get_models_details queries the API."""
        mock_model = MagicMock()
        mock_model.id = "claude-sonnet-4-5-20250929"
        mock_model.display_name = "Claude Sonnet 4.5"
        mock_model.max_input_tokens = 200000
        mock_model.max_tokens = 64000
        mock_model.capabilities = None

        mock_page = MagicMock()
        mock_page.data = [mock_model]
        provider._client.models.list = AsyncMock(return_value=mock_page)

        details = await provider.get_models_details()

        assert len(details) == 1
        assert details[0].id == "claude-sonnet-4-5-20250929"
        assert details[0].context_length == 200000
        assert provider._discovered_context_lengths is not None
        assert provider._discovered_context_lengths["claude-sonnet-4-5-20250929"] == 200000

    @pytest.mark.asyncio
    async def test_discovery_fallback_on_error(self, provider: AnthropicProvider):
        """Falls back to static table on API error."""
        provider._client.models.list = AsyncMock(side_effect=Exception("Unauthorized"))

        details = await provider.get_models_details()

        assert len(details) == len(DEFAULT_ANTHROPIC_TOKEN_LIMITS)
        model_ids = {d.id for d in details}
        assert "claude-sonnet-4-5-20250929" in model_ids

    def test_context_length_uses_discovery(self, provider: AnthropicProvider):
        """get_max_context_length uses discovered data first."""
        provider._discovered_context_lengths = {"claude-test-model": 300000}
        assert provider.get_max_context_length("claude-test-model") == 300000

    def test_context_length_static_fallback(self, provider: AnthropicProvider):
        """get_max_context_length falls back to static table."""
        assert provider.get_max_context_length("claude-sonnet-4-5-20250929") == 200000

    def test_context_length_unknown_model(self, provider: AnthropicProvider):
        """Unknown model with no registry hit gets 200000 fallback."""
        with patch(
            "llmcore.providers.anthropic_provider.AnthropicProvider.get_max_context_length",
            wraps=provider.get_max_context_length,
        ):
            # Mock registry to return None for unknown model
            with patch("llmcore.model_cards.get_model_card_registry") as mock_reg:
                mock_reg.return_value.get_context_length.return_value = None
                assert provider.get_max_context_length("claude-future-model") == 200000


# ====================================================================
# Error Handling Tests
# ====================================================================


class TestErrorHandling:
    """Tests for error mapping and handling."""

    @pytest.mark.asyncio
    async def test_client_not_initialized(
        self, provider: AnthropicProvider, simple_messages: list[Message]
    ):
        """Raises ProviderError if client is None."""
        provider._client = None
        with pytest.raises(ProviderError, match="not initialized"):
            await provider.chat_completion(simple_messages)

    @pytest.mark.asyncio
    async def test_invalid_context_type(self, provider: AnthropicProvider):
        """Raises ProviderError for non-Message context."""
        with pytest.raises(ProviderError, match="Unsupported context"):
            await provider.chat_completion("not a list")  # type: ignore

    @pytest.mark.asyncio
    async def test_empty_messages(self, provider: AnthropicProvider):
        """Raises ProviderError for empty messages after processing."""
        # Only a system message, no user messages
        msgs = [Message(role=LLMCoreRole.SYSTEM, content="System only")]
        with pytest.raises(ProviderError, match="No valid messages"):
            await provider.chat_completion(msgs)

    @pytest.mark.asyncio
    async def test_context_length_error_detection(
        self, provider: AnthropicProvider, simple_messages: list[Message]
    ):
        """BadRequestError about tokens raises ContextLengthError."""
        from anthropic._exceptions import BadRequestError as SdkBadRequestError

        # Construct a proper BadRequestError mock
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.request = MagicMock()
        mock_response.headers = {}
        mock_exc = SdkBadRequestError(
            message="prompt is too long: 250000 tokens > 200000 maximum",
            response=mock_response,
            body={"error": {"type": "invalid_request_error", "message": "prompt is too long"}},
        )
        provider._client.messages.create = AsyncMock(side_effect=mock_exc)

        with pytest.raises(ContextLengthError):
            await provider.chat_completion(simple_messages)


# ====================================================================
# Supported Parameters Tests
# ====================================================================


class TestSupportedParameters:
    """Tests for get_supported_parameters."""

    def test_includes_thinking(self, provider: AnthropicProvider):
        """Supported params include thinking config."""
        params = provider.get_supported_parameters()
        assert "thinking" in params

    def test_includes_output_config(self, provider: AnthropicProvider):
        """Supported params include output_config."""
        params = provider.get_supported_parameters()
        assert "output_config" in params

    def test_includes_metadata(self, provider: AnthropicProvider):
        """Supported params include metadata."""
        params = provider.get_supported_parameters()
        assert "metadata" in params

    def test_includes_service_tier(self, provider: AnthropicProvider):
        """Supported params include service_tier."""
        params = provider.get_supported_parameters()
        assert "service_tier" in params


# ====================================================================
# Cleanup Tests
# ====================================================================


class TestCleanup:
    """Tests for provider cleanup."""

    @pytest.mark.asyncio
    async def test_close(self, provider: AnthropicProvider):
        """close() shuts down the client."""
        provider._client.close = AsyncMock()
        await provider.close()

        provider._client is None or provider._client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_handles_error(self, provider: AnthropicProvider):
        """close() doesn't raise on error."""
        provider._client.close = AsyncMock(side_effect=Exception("cleanup error"))
        await provider.close()  # Should not raise
        assert provider._client is None
