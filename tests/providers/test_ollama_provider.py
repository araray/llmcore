# tests/providers/test_ollama_provider.py
"""
Comprehensive unit tests for OllamaProvider and OllamaEmbedding.

Covers:
- Model discovery (get_models_details) — correct field access and capabilities
- Message building — text, tool results, multimodal images
- Chat completion — param routing (top-level vs options), tool serialization
- Tool call extraction (extract_tool_calls)
- Thinking content extraction (extract_thinking_content)
- Response content extraction (extract_response_content, extract_delta_content)
- Usage extraction (extract_usage)
- Supported parameters schema completeness
- Context length resolution chain
- Client lifecycle (close)
- Streaming normalization (yields dicts, not Pydantic models)
- Embedding: modern embed() API, batch support, dimensions, close

All tests use mocking — no live Ollama server required.
"""

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Module-level mocking: the ollama library may not be installed in CI.
# We mock the availability flag and the AsyncClient.
# ---------------------------------------------------------------------------


@pytest.fixture
def ollama_provider():
    """Create an OllamaProvider with mocked ollama library."""
    with (
        patch("llmcore.providers.ollama_provider.ollama_available", True),
        patch("llmcore.providers.ollama_provider.tiktoken_available", False),
        patch("llmcore.providers.ollama_provider.AsyncClient") as mock_cls,
    ):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client

        from llmcore.providers.ollama_provider import OllamaProvider

        config = {
            "default_model": "gemma3:4b",
            "host": "http://localhost:11434",
        }
        provider = OllamaProvider(config)
        provider._client = mock_client
        return provider


@pytest.fixture
def ollama_provider_with_keep_alive():
    """Create an OllamaProvider with default_keep_alive set."""
    with (
        patch("llmcore.providers.ollama_provider.ollama_available", True),
        patch("llmcore.providers.ollama_provider.tiktoken_available", False),
        patch("llmcore.providers.ollama_provider.AsyncClient") as mock_cls,
    ):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client

        from llmcore.providers.ollama_provider import OllamaProvider

        config = {
            "default_model": "gemma3:4b",
            "host": "http://localhost:11434",
            "keep_alive": "10m",
        }
        provider = OllamaProvider(config)
        provider._client = mock_client
        return provider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_message(role: str, content: str, **metadata_kw) -> "Message":
    """Create an llmcore Message for testing."""
    from llmcore.models import Message, Role

    return Message(role=Role(role), content=content, metadata=metadata_kw)


def _make_list_response_model(name: str, family: str = "llama", size: int = 4_000_000_000):
    """Simulate an Ollama ListResponse.Model with attribute access."""
    details = SimpleNamespace(
        family=family,
        families=["llama"],
        parameter_size="4B",
        quantization_level="Q4_K_M",
        parent_model=None,
        format="gguf",
    )
    return SimpleNamespace(
        model=name,
        size=size,
        digest="sha256:abc123",
        modified_at=None,
        details=details,
    )


def _make_show_response(capabilities: list[str] | None = None):
    """Simulate a ShowResponse."""
    return SimpleNamespace(
        capabilities=capabilities,
        template=None,
        modelfile=None,
        license=None,
        details=None,
        modelinfo=None,
        parameters=None,
        modified_at=None,
    )


def _make_chat_response(
    content: str = "Hello!",
    tool_calls: list[dict] | None = None,
    thinking: str | None = None,
    model: str = "gemma3:4b",
) -> dict[str, Any]:
    """Build a non-streaming chat response dict (post-model_dump)."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if thinking is not None:
        message["thinking"] = thinking
    return {
        "model": model,
        "message": message,
        "done": True,
        "done_reason": "stop",
        "total_duration": 5_000_000_000,
        "load_duration": 1_000_000_000,
        "prompt_eval_count": 42,
        "prompt_eval_duration": 2_000_000_000,
        "eval_count": 15,
        "eval_duration": 2_000_000_000,
    }


# ===========================================================================
# Tests: Model Discovery (get_models_details)
# ===========================================================================


class TestGetModelsDetails:
    """Validate that model discovery uses correct SDK field access."""

    @pytest.mark.asyncio
    async def test_discovers_models_with_correct_field_access(self, ollama_provider):
        """Models are read from ListResponse.Model.model (not .name)."""
        model_entries = [
            _make_list_response_model("gemma3:4b", family="gemma"),
            _make_list_response_model("llama3.2:latest", family="llama"),
        ]
        list_resp = SimpleNamespace(models=model_entries)
        ollama_provider._client.list = AsyncMock(return_value=list_resp)
        ollama_provider._client.show = AsyncMock(
            return_value=_make_show_response(["tools", "vision"])
        )

        details = await ollama_provider.get_models_details()

        assert len(details) == 2
        assert details[0].id == "gemma3:4b"
        assert details[1].id == "llama3.2:latest"

    @pytest.mark.asyncio
    async def test_populates_rich_fields(self, ollama_provider):
        """Family, parameter_count, quantization, file_size populated."""
        model_entries = [_make_list_response_model("gemma3:4b", family="gemma", size=3_800_000_000)]
        list_resp = SimpleNamespace(models=model_entries)
        ollama_provider._client.list = AsyncMock(return_value=list_resp)
        ollama_provider._client.show = AsyncMock(return_value=_make_show_response())

        details = await ollama_provider.get_models_details()

        assert details[0].family == "gemma"
        assert details[0].parameter_count == "4B"
        assert details[0].quantization_level == "Q4_K_M"
        assert details[0].file_size_bytes == 3_800_000_000

    @pytest.mark.asyncio
    async def test_capabilities_from_show(self, ollama_provider):
        """Model capabilities are read from show() response."""
        model_entries = [_make_list_response_model("llava:latest")]
        list_resp = SimpleNamespace(models=model_entries)
        ollama_provider._client.list = AsyncMock(return_value=list_resp)
        ollama_provider._client.show = AsyncMock(
            return_value=_make_show_response(["tools", "vision", "thinking"])
        )

        details = await ollama_provider.get_models_details()

        assert details[0].supports_tools is True
        assert details[0].supports_vision is True
        assert details[0].supports_reasoning is True

    @pytest.mark.asyncio
    async def test_show_failure_defaults_to_false(self, ollama_provider):
        """If show() fails, capabilities default to False."""
        model_entries = [_make_list_response_model("gemma3:4b")]
        list_resp = SimpleNamespace(models=model_entries)
        ollama_provider._client.list = AsyncMock(return_value=list_resp)
        ollama_provider._client.show = AsyncMock(side_effect=Exception("connection timeout"))

        details = await ollama_provider.get_models_details()

        assert details[0].supports_tools is False
        assert details[0].supports_vision is False
        assert details[0].supports_reasoning is False

    @pytest.mark.asyncio
    async def test_skips_models_without_model_name(self, ollama_provider):
        """Models where .model is None/empty are skipped."""
        model_entries = [
            SimpleNamespace(model=None, size=0, digest=None, modified_at=None, details=None),
            _make_list_response_model("gemma3:4b"),
        ]
        list_resp = SimpleNamespace(models=model_entries)
        ollama_provider._client.list = AsyncMock(return_value=list_resp)
        ollama_provider._client.show = AsyncMock(return_value=_make_show_response())

        details = await ollama_provider.get_models_details()
        assert len(details) == 1
        assert details[0].id == "gemma3:4b"


# ===========================================================================
# Tests: Message Building
# ===========================================================================


class TestBuildMessagePayload:
    """Validate _build_message_payload for various message types."""

    def test_basic_text_message(self, ollama_provider):
        """Standard text message produces role + content."""
        msg = _make_message("user", "Hello world")
        payload = ollama_provider._build_message_payload(msg)

        assert payload == {"role": "user", "content": "Hello world"}

    def test_system_message(self, ollama_provider):
        """System messages are preserved."""
        msg = _make_message("system", "You are helpful")
        payload = ollama_provider._build_message_payload(msg)

        assert payload["role"] == "system"
        assert payload["content"] == "You are helpful"

    def test_tool_result_message_includes_tool_name(self, ollama_provider):
        """Tool-result messages include tool_name from metadata."""
        msg = _make_message("tool", '{"temp": 22}', tool_name="get_weather")
        payload = ollama_provider._build_message_payload(msg)

        assert payload["role"] == "tool"
        assert payload["content"] == '{"temp": 22}'
        assert payload["tool_name"] == "get_weather"

    def test_tool_result_message_name_from_metadata_name(self, ollama_provider):
        """Tool name can come from metadata['name'] as fallback."""
        msg = _make_message("tool", "result", name="my_tool")
        payload = ollama_provider._build_message_payload(msg)

        assert payload["tool_name"] == "my_tool"

    def test_tool_result_without_name_omits_field(self, ollama_provider):
        """If no tool_name in metadata, the field is omitted."""
        msg = _make_message("tool", "result")
        payload = ollama_provider._build_message_payload(msg)

        assert "tool_name" not in payload

    def test_message_with_inline_images(self, ollama_provider):
        """Messages with inline_images include images in payload."""
        msg = _make_message(
            "user",
            "What is in this image?",
            inline_images=["base64encodeddata", "/path/to/img.png"],
        )
        payload = ollama_provider._build_message_payload(msg)

        assert "images" in payload
        assert payload["images"] == ["base64encodeddata", "/path/to/img.png"]

    def test_message_with_dict_images(self, ollama_provider):
        """Dict-format images extract the data field."""
        msg = _make_message(
            "user",
            "Describe this",
            inline_images=[{"data": "b64data", "mime_type": "image/png"}],
        )
        payload = ollama_provider._build_message_payload(msg)

        assert payload["images"] == ["b64data"]


# ===========================================================================
# Tests: Chat Completion — Parameter Routing
# ===========================================================================


class TestChatCompletionParamRouting:
    """Validate that kwargs are routed to correct locations."""

    @pytest.mark.asyncio
    async def test_top_level_params_not_in_options(self, ollama_provider):
        """format, think, logprobs go as top-level args, not inside options."""
        mock_response = SimpleNamespace(model_dump=lambda: _make_chat_response())
        ollama_provider._client.chat = AsyncMock(return_value=mock_response)

        msgs = [_make_message("user", "Hello")]
        await ollama_provider.chat_completion(
            context=msgs,
            stream=False,
            temperature=0.7,
            format="json",
            think=True,
            logprobs=True,
            top_logprobs=5,
        )

        call_kwargs = ollama_provider._client.chat.call_args[1]
        # Top-level params should be at the top level
        assert call_kwargs["format"] == "json"
        assert call_kwargs["think"] is True
        assert call_kwargs["logprobs"] is True
        assert call_kwargs["top_logprobs"] == 5
        # Sampling params should be in options
        assert call_kwargs["options"]["temperature"] == 0.7
        # Top-level params should NOT be in options
        assert "format" not in call_kwargs.get("options", {})
        assert "think" not in call_kwargs.get("options", {})

    @pytest.mark.asyncio
    async def test_max_tokens_aliased_to_num_predict(self, ollama_provider):
        """max_tokens kwarg becomes num_predict in options."""
        mock_response = SimpleNamespace(model_dump=lambda: _make_chat_response())
        ollama_provider._client.chat = AsyncMock(return_value=mock_response)

        msgs = [_make_message("user", "Hello")]
        await ollama_provider.chat_completion(context=msgs, stream=False, max_tokens=100)

        call_kwargs = ollama_provider._client.chat.call_args[1]
        assert call_kwargs["options"]["num_predict"] == 100
        assert "max_tokens" not in call_kwargs.get("options", {})

    @pytest.mark.asyncio
    async def test_default_keep_alive_applied(self, ollama_provider_with_keep_alive):
        """Config-level keep_alive is applied when not explicitly set."""
        provider = ollama_provider_with_keep_alive
        mock_response = SimpleNamespace(model_dump=lambda: _make_chat_response())
        provider._client.chat = AsyncMock(return_value=mock_response)

        msgs = [_make_message("user", "Hello")]
        await provider.chat_completion(context=msgs, stream=False)

        call_kwargs = provider._client.chat.call_args[1]
        assert call_kwargs["keep_alive"] == "10m"

    @pytest.mark.asyncio
    async def test_explicit_keep_alive_overrides_default(self, ollama_provider_with_keep_alive):
        """Explicit keep_alive kwarg overrides config default."""
        provider = ollama_provider_with_keep_alive
        mock_response = SimpleNamespace(model_dump=lambda: _make_chat_response())
        provider._client.chat = AsyncMock(return_value=mock_response)

        msgs = [_make_message("user", "Hello")]
        await provider.chat_completion(context=msgs, stream=False, keep_alive="5m")

        call_kwargs = provider._client.chat.call_args[1]
        assert call_kwargs["keep_alive"] == "5m"

    @pytest.mark.asyncio
    async def test_tools_passed_correctly(self, ollama_provider):
        """Tool objects are serialized to Ollama's expected format."""
        from llmcore.models import Tool

        mock_response = SimpleNamespace(model_dump=lambda: _make_chat_response())
        ollama_provider._client.chat = AsyncMock(return_value=mock_response)

        tool = Tool(
            name="get_weather",
            description="Get the weather",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
        msgs = [_make_message("user", "What's the weather?")]
        await ollama_provider.chat_completion(context=msgs, stream=False, tools=[tool])

        call_kwargs = ollama_provider._client.chat.call_args[1]
        tools_payload = call_kwargs["tools"]
        assert len(tools_payload) == 1
        assert tools_payload[0]["type"] == "function"
        assert tools_payload[0]["function"]["name"] == "get_weather"


# ===========================================================================
# Tests: Tool Call Extraction
# ===========================================================================


class TestExtractToolCalls:
    """Validate extract_tool_calls on Ollama response format."""

    def test_extracts_single_tool_call(self, ollama_provider):
        """Single tool call extracted with generated UUID."""
        response = _make_chat_response(
            content="",
            tool_calls=[
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "London"},
                    }
                }
            ],
        )

        calls = ollama_provider.extract_tool_calls(response)

        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {"city": "London"}
        assert calls[0].id  # UUID generated

    def test_extracts_multiple_tool_calls(self, ollama_provider):
        """Multiple tool calls extracted."""
        response = _make_chat_response(
            content="",
            tool_calls=[
                {"function": {"name": "func_a", "arguments": {"x": 1}}},
                {"function": {"name": "func_b", "arguments": {"y": 2}}},
            ],
        )

        calls = ollama_provider.extract_tool_calls(response)

        assert len(calls) == 2
        assert calls[0].name == "func_a"
        assert calls[1].name == "func_b"

    def test_no_tool_calls_returns_empty(self, ollama_provider):
        """Response without tool_calls returns empty list."""
        response = _make_chat_response(content="Hello!")

        calls = ollama_provider.extract_tool_calls(response)

        assert calls == []

    def test_string_arguments_parsed(self, ollama_provider):
        """JSON string arguments are parsed to dict."""
        response = _make_chat_response(
            content="",
            tool_calls=[
                {
                    "function": {
                        "name": "func",
                        "arguments": '{"a": 1}',
                    }
                }
            ],
        )

        calls = ollama_provider.extract_tool_calls(response)

        assert calls[0].arguments == {"a": 1}

    def test_invalid_json_arguments_wrapped(self, ollama_provider):
        """Invalid JSON string arguments wrapped in {raw: ...}."""
        response = _make_chat_response(
            content="",
            tool_calls=[
                {
                    "function": {
                        "name": "func",
                        "arguments": "not json",
                    }
                }
            ],
        )

        calls = ollama_provider.extract_tool_calls(response)

        assert calls[0].arguments == {"raw": "not json"}


# ===========================================================================
# Tests: Thinking Content Extraction
# ===========================================================================


class TestExtractThinking:
    """Validate extract_thinking_content."""

    def test_extracts_thinking(self, ollama_provider):
        """Thinking content extracted from message.thinking."""
        response = _make_chat_response(content="42", thinking="Let me think... 40 + 2 = 42")

        thinking = ollama_provider.extract_thinking_content(response)

        assert thinking == "Let me think... 40 + 2 = 42"

    def test_no_thinking_returns_none(self, ollama_provider):
        """No thinking field returns None."""
        response = _make_chat_response(content="Hello")

        thinking = ollama_provider.extract_thinking_content(response)

        assert thinking is None

    def test_empty_thinking_returns_none(self, ollama_provider):
        """Empty thinking string returns None."""
        response = _make_chat_response(content="Hello", thinking="")

        thinking = ollama_provider.extract_thinking_content(response)

        assert thinking is None


# ===========================================================================
# Tests: Response Content Extraction
# ===========================================================================


class TestExtractResponseContent:
    """Validate extract_response_content for Ollama format."""

    def test_extracts_content(self, ollama_provider):
        """Content extracted from message.content."""
        response = _make_chat_response(content="Hello world!")

        result = ollama_provider.extract_response_content(response)

        assert result == "Hello world!"

    def test_empty_content_returns_empty_string(self, ollama_provider):
        """Empty content returns empty string."""
        response = _make_chat_response(content="")

        result = ollama_provider.extract_response_content(response)

        assert result == ""

    def test_none_content_returns_empty_string(self, ollama_provider):
        """None content returns empty string."""
        response = {"message": {"role": "assistant", "content": None}}

        result = ollama_provider.extract_response_content(response)

        assert result == ""

    def test_missing_message_returns_empty_string(self, ollama_provider):
        """Missing message key returns empty string."""
        response = {"model": "test", "done": True}

        result = ollama_provider.extract_response_content(response)

        assert result == ""


# ===========================================================================
# Tests: Delta Content Extraction
# ===========================================================================


class TestExtractDeltaContent:
    """Validate extract_delta_content for Ollama streaming format."""

    def test_extracts_delta(self, ollama_provider):
        """Text delta extracted from streaming chunk dict."""
        chunk = {"message": {"role": "assistant", "content": "The"}, "done": False}

        result = ollama_provider.extract_delta_content(chunk)

        assert result == "The"

    def test_final_chunk_empty(self, ollama_provider):
        """Final done=true chunk with empty content returns empty string."""
        chunk = {"message": {"role": "assistant", "content": ""}, "done": True}

        result = ollama_provider.extract_delta_content(chunk)

        assert result == ""

    def test_missing_message_returns_empty(self, ollama_provider):
        """Chunk without message returns empty string."""
        chunk = {"done": True}

        result = ollama_provider.extract_delta_content(chunk)

        assert result == ""


# ===========================================================================
# Tests: Usage Extraction
# ===========================================================================


class TestExtractUsage:
    """Validate extract_usage mapping."""

    def test_maps_ollama_fields_to_openai_names(self, ollama_provider):
        """Ollama's prompt_eval_count/eval_count mapped correctly."""
        response = _make_chat_response()

        usage = ollama_provider.extract_usage(response)

        assert usage["prompt_tokens"] == 42
        assert usage["completion_tokens"] == 15
        assert usage["total_tokens"] == 57
        assert usage["total_duration"] == 5_000_000_000

    def test_missing_counts_default_to_zero(self, ollama_provider):
        """Missing counts default to 0."""
        response = {"message": {"content": "Hi"}, "done": True}

        usage = ollama_provider.extract_usage(response)

        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0
        assert usage["total_tokens"] == 0


# ===========================================================================
# Tests: Supported Parameters
# ===========================================================================


class TestSupportedParameters:
    """Validate expanded parameter schema."""

    def test_includes_all_sampling_params(self, ollama_provider):
        params = ollama_provider.get_supported_parameters()

        for key in [
            "temperature",
            "top_p",
            "top_k",
            "seed",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "repeat_penalty",
            "repeat_last_n",
            "tfs_z",
            "typical_p",
            "penalize_newline",
        ]:
            assert key in params, f"Missing sampling param: {key}"

    def test_includes_top_level_params(self, ollama_provider):
        params = ollama_provider.get_supported_parameters()

        for key in ["format", "keep_alive", "think", "logprobs", "top_logprobs"]:
            assert key in params, f"Missing top-level param: {key}"

    def test_includes_runtime_params(self, ollama_provider):
        params = ollama_provider.get_supported_parameters()

        for key in [
            "num_ctx",
            "num_predict",
            "num_batch",
            "num_gpu",
            "num_thread",
            "num_keep",
            "low_vram",
            "use_mmap",
            "use_mlock",
        ]:
            assert key in params, f"Missing runtime param: {key}"


# ===========================================================================
# Tests: Context Length
# ===========================================================================


class TestContextLength:
    """Context length resolution chain tests (extends existing test file)."""

    def test_hardcoded_model(self, ollama_provider):
        assert ollama_provider.get_max_context_length("llama3") == 8000

    def test_hardcoded_model_with_tag(self, ollama_provider):
        assert ollama_provider.get_max_context_length("qwen3:4b") == 262144

    def test_base_name_fallback(self, ollama_provider):
        assert ollama_provider.get_max_context_length("gemma3:4b") == 128000

    def test_new_models_in_dict(self, ollama_provider):
        """deepseek-r1 and phi4 added to token limits."""
        assert ollama_provider.get_max_context_length("deepseek-r1") == 131072
        assert ollama_provider.get_max_context_length("phi4") == 16384


# ===========================================================================
# Tests: Client Lifecycle
# ===========================================================================


class TestClose:
    """Validate close() calls the correct SDK method."""

    @pytest.mark.asyncio
    async def test_close_calls_client_close(self, ollama_provider):
        """close() should call client.close() (not aclose)."""
        mock_close = AsyncMock()
        ollama_provider._client.close = mock_close

        await ollama_provider.close()

        mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_sets_client_to_none(self, ollama_provider):
        """After close(), _client is None."""
        ollama_provider._client.close = AsyncMock()

        await ollama_provider.close()

        assert ollama_provider._client is None

    @pytest.mark.asyncio
    async def test_close_handles_error_gracefully(self, ollama_provider):
        """close() doesn't raise even if client.close() errors."""
        ollama_provider._client.close = AsyncMock(side_effect=RuntimeError("socket error"))

        await ollama_provider.close()

        assert ollama_provider._client is None

    @pytest.mark.asyncio
    async def test_close_when_no_client(self, ollama_provider):
        """close() is safe when _client is already None."""
        ollama_provider._client = None

        await ollama_provider.close()  # Should not raise


# ===========================================================================
# Tests: Streaming Normalization
# ===========================================================================


class TestStreamingNormalization:
    """Validate _wrap_stream yields dicts, not Pydantic models."""

    @pytest.mark.asyncio
    async def test_wrap_stream_yields_dicts(self, ollama_provider):
        """Streaming chunks are model_dump()'d to dicts."""
        chunk1 = SimpleNamespace(
            model_dump=lambda: {
                "message": {"role": "assistant", "content": "Hello"},
                "done": False,
            }
        )
        chunk2 = SimpleNamespace(
            model_dump=lambda: {
                "message": {"role": "assistant", "content": " world"},
                "done": True,
            }
        )

        async def fake_stream():
            for c in [chunk1, chunk2]:
                yield c

        wrapper = await ollama_provider._wrap_stream(fake_stream())

        chunks = []
        async for ch in wrapper:
            chunks.append(ch)

        assert len(chunks) == 2
        assert all(isinstance(ch, dict) for ch in chunks)
        assert chunks[0]["message"]["content"] == "Hello"
        assert chunks[1]["message"]["content"] == " world"

    @pytest.mark.asyncio
    async def test_wrap_stream_handles_plain_dicts(self, ollama_provider):
        """If stream yields plain dicts, they pass through."""

        async def fake_stream():
            yield {"message": {"content": "Hi"}, "done": True}

        wrapper = await ollama_provider._wrap_stream(fake_stream())

        chunks = []
        async for ch in wrapper:
            chunks.append(ch)

        assert chunks[0]["message"]["content"] == "Hi"


# ===========================================================================
# Tests: Embedding Provider
# ===========================================================================


@pytest.fixture
def ollama_embedding():
    """Create an OllamaEmbedding with mocked ollama library."""
    with (
        patch("llmcore.embedding.ollama.ollama_available", True),
        patch("llmcore.embedding.ollama.AsyncClient") as mock_cls,
    ):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client

        from llmcore.embedding.ollama import OllamaEmbedding

        config = {
            "default_model": "mxbai-embed-large",
            "host": "http://localhost:11434",
            "dimensions": 384,
        }
        emb = OllamaEmbedding(config)
        emb._client = mock_client
        return emb


class TestOllamaEmbedding:
    """Validate OllamaEmbedding uses modern embed() API."""

    @pytest.mark.asyncio
    async def test_generate_embedding_uses_embed_endpoint(self, ollama_embedding):
        """Single embedding uses embed() not deprecated embeddings()."""
        mock_response = SimpleNamespace(embeddings=[[0.1, 0.2, 0.3]])
        ollama_embedding._client.embed = AsyncMock(return_value=mock_response)

        result = await ollama_embedding.generate_embedding("hello world")

        ollama_embedding._client.embed.assert_called_once()
        call_kwargs = ollama_embedding._client.embed.call_args[1]
        assert call_kwargs["model"] == "mxbai-embed-large"
        assert call_kwargs["input"] == "hello world"
        assert call_kwargs["dimensions"] == 384
        assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch(self, ollama_embedding):
        """Batch embedding sends all texts in one request."""
        mock_response = SimpleNamespace(embeddings=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        ollama_embedding._client.embed = AsyncMock(return_value=mock_response)

        result = await ollama_embedding.generate_embeddings(["text1", "text2", "text3"])

        # Only one HTTP call
        ollama_embedding._client.embed.assert_called_once()
        call_kwargs = ollama_embedding._client.embed.call_args[1]
        assert call_kwargs["input"] == ["text1", "text2", "text3"]
        assert len(result) == 3
        assert result[0] == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_generate_embeddings_empty_list(self, ollama_embedding):
        """Empty input list returns empty result without API call."""
        result = await ollama_embedding.generate_embeddings([])

        assert result == []
        ollama_embedding._client.embed.assert_not_called()

    @pytest.mark.asyncio
    async def test_dimensions_passed_through(self, ollama_embedding):
        """Configured dimensions param is passed to embed()."""
        mock_response = SimpleNamespace(embeddings=[[0.1, 0.2]])
        ollama_embedding._client.embed = AsyncMock(return_value=mock_response)

        await ollama_embedding.generate_embedding("test")

        call_kwargs = ollama_embedding._client.embed.call_args[1]
        assert call_kwargs["dimensions"] == 384

    @pytest.mark.asyncio
    async def test_close_calls_client_close(self, ollama_embedding):
        """close() calls client.close() (not aclose)."""
        mock_close = AsyncMock()
        ollama_embedding._client.close = mock_close

        await ollama_embedding.close()

        mock_close.assert_called_once()
        assert ollama_embedding._client is None

    @pytest.mark.asyncio
    async def test_close_handles_error_gracefully(self, ollama_embedding):
        """close() doesn't raise on error."""
        ollama_embedding._client.close = AsyncMock(side_effect=RuntimeError("err"))

        await ollama_embedding.close()

        assert ollama_embedding._client is None

    @pytest.mark.asyncio
    async def test_empty_text_raises_error(self, ollama_embedding):
        """Empty text raises EmbeddingError."""
        from llmcore.exceptions import EmbeddingError

        with pytest.raises(EmbeddingError, match="cannot be empty"):
            await ollama_embedding.generate_embedding("")

    @pytest.mark.asyncio
    async def test_no_client_raises_error(self, ollama_embedding):
        """Calling without initialize raises EmbeddingError."""
        from llmcore.exceptions import EmbeddingError

        ollama_embedding._client = None

        with pytest.raises(EmbeddingError, match="not initialized"):
            await ollama_embedding.generate_embedding("test")


# ===========================================================================
# Tests: Provider name
# ===========================================================================


class TestProviderName:
    def test_default_name(self, ollama_provider):
        assert ollama_provider.get_name() == "ollama"

    def test_custom_instance_name(self, ollama_provider):
        ollama_provider._provider_instance_name = "ollama_gpu"
        assert ollama_provider.get_name() == "ollama_gpu"
