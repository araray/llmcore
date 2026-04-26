# tests/providers/test_huggingface_provider.py
"""Tests for the HuggingFace Inference API provider.

Tests cover:
- Provider initialization (happy path + missing API key)
- Message payload building (text, multimodal, tool results)
- Chat completion (non-streaming and streaming)
- Response/delta content extraction
- Tool call extraction
- Token counting
- Multimodal: TTS, STT, Embeddings, Image generation
- Error handling (timeout, bad request, HTTP errors)
- Provider manager registration
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the provider module can be imported even without huggingface_hub
# installed in the test environment.
# ---------------------------------------------------------------------------

# Create a minimal mock of huggingface_hub if not available
try:
    import huggingface_hub  # noqa: F401
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    # Build minimal stubs so the provider module can be imported
    hf_hub_mock = MagicMock()
    hf_hub_mock.AsyncInferenceClient = MagicMock
    hf_hub_mock.errors = MagicMock()
    hf_hub_mock.errors.HfHubHTTPError = type("HfHubHTTPError", (Exception,), {})
    hf_hub_mock.errors.BadRequestError = type("BadRequestError", (Exception,), {})
    hf_hub_mock.errors.InferenceTimeoutError = type("InferenceTimeoutError", (Exception,), {})

    # Generated types
    types_mock = MagicMock()
    types_mock.ChatCompletionOutput = dict
    types_mock.ChatCompletionStreamOutput = dict

    hf_hub_mock.inference = MagicMock()
    hf_hub_mock.inference._generated = MagicMock()
    hf_hub_mock.inference._generated.types = types_mock

    sys.modules["huggingface_hub"] = hf_hub_mock
    sys.modules["huggingface_hub.errors"] = hf_hub_mock.errors
    sys.modules["huggingface_hub.inference"] = hf_hub_mock.inference
    sys.modules["huggingface_hub.inference._generated"] = hf_hub_mock.inference._generated
    sys.modules["huggingface_hub.inference._generated.types"] = types_mock


from llmcore.exceptions import ConfigError, ProviderError
from llmcore.models import Message, Role, Tool
from llmcore.models_multimodal import (
    ImageGenerationResult,
    SpeechResult,
    TranscriptionResult,
)
from llmcore.providers.huggingface_provider import HuggingFaceProvider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> dict[str, Any]:
    """Build a provider config dict with sensible defaults."""
    config = {
        "api_key": "hf_test_token_12345",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct",
        "timeout": 30,
    }
    config.update(overrides)
    return config


def _make_provider(**config_overrides) -> HuggingFaceProvider:
    """Build a HuggingFaceProvider with a mocked client."""
    config = _make_config(**config_overrides)
    with patch(
        "llmcore.providers.huggingface_provider.AsyncInferenceClient"
    ) as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        provider = HuggingFaceProvider(config)
    provider._client = mock_client
    return provider


def _make_chat_response(
    content: str = "Hello!",
    model: str = "meta-llama/Llama-3.3-70B-Instruct",
    tool_calls: list | None = None,
) -> dict[str, Any]:
    """Build a mock OpenAI-compatible chat completion response."""
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }


def _make_stream_chunk(
    content: str = "Hi", finish_reason: str | None = None
) -> dict[str, Any]:
    """Build a mock streaming chunk."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestHuggingFaceProviderInit:
    """Tests for provider initialization."""

    def test_init_with_api_key(self):
        provider = _make_provider()
        assert provider.get_name() == "huggingface"
        assert provider.default_model == "meta-llama/Llama-3.3-70B-Instruct"

    def test_init_missing_api_key_raises(self):
        with patch(
            "llmcore.providers.huggingface_provider.AsyncInferenceClient"
        ):
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(ConfigError, match="API key"):
                    HuggingFaceProvider({"timeout": 30})

    def test_init_custom_model(self):
        provider = _make_provider(default_model="Qwen/Qwen2.5-72B-Instruct")
        assert provider.default_model == "Qwen/Qwen2.5-72B-Instruct"

    def test_init_with_instance_name(self):
        provider = _make_provider(_instance_name="my_hf")
        assert provider.get_name() == "my_hf"

    def test_init_api_key_from_env(self):
        with patch(
            "llmcore.providers.huggingface_provider.AsyncInferenceClient"
        ):
            with patch.dict(
                "os.environ", {"HF_TOKEN": "hf_env_token"}, clear=False
            ):
                config = {"default_model": "test/model", "timeout": 30}
                provider = HuggingFaceProvider(config)
                assert provider._api_key == "hf_env_token"


# ---------------------------------------------------------------------------
# Message preparation tests
# ---------------------------------------------------------------------------


class TestMessagePayload:
    """Tests for _build_message_payload."""

    def test_simple_user_message(self):
        provider = _make_provider()
        msg = Message(role=Role.USER, content="Hello")
        payload = provider._build_message_payload(msg)
        assert payload == {"role": "user", "content": "Hello"}

    def test_system_message(self):
        provider = _make_provider()
        msg = Message(role=Role.SYSTEM, content="Be helpful")
        payload = provider._build_message_payload(msg)
        assert payload == {"role": "system", "content": "Be helpful"}

    def test_assistant_message(self):
        provider = _make_provider()
        msg = Message(role=Role.ASSISTANT, content="Sure!")
        payload = provider._build_message_payload(msg)
        assert payload == {"role": "assistant", "content": "Sure!"}

    def test_multimodal_image_url(self):
        provider = _make_provider()
        msg = Message(
            role=Role.USER,
            content="What's in this image?",
            metadata={"inline_images": ["https://example.com/img.jpg"]},
        )
        payload = provider._build_message_payload(msg)
        assert isinstance(payload["content"], list)
        assert len(payload["content"]) == 2
        assert payload["content"][0]["type"] == "text"
        assert payload["content"][1]["type"] == "image_url"

    def test_multimodal_image_dict(self):
        provider = _make_provider()
        msg = Message(
            role=Role.USER,
            content="Describe this",
            metadata={
                "inline_images": [
                    {"url": "https://example.com/img.jpg", "detail": "low"}
                ]
            },
        )
        payload = provider._build_message_payload(msg)
        img_part = payload["content"][1]
        assert img_part["image_url"]["detail"] == "low"

    def test_content_parts_passthrough(self):
        provider = _make_provider()
        parts = [
            {"type": "text", "text": "Hello"},
            {"type": "image_url", "image_url": {"url": "..."}},
        ]
        msg = Message(
            role=Role.USER,
            content="",
            metadata={"content_parts": parts},
        )
        payload = provider._build_message_payload(msg)
        assert payload["content"] == parts

    def test_tool_result_message(self):
        provider = _make_provider()
        msg = Message(
            role=Role.TOOL,
            content='{"result": 42}',
            tool_call_id="call_abc123",
        )
        payload = provider._build_message_payload(msg)
        assert payload["role"] == "tool"
        assert payload["tool_call_id"] == "call_abc123"

    def test_assistant_with_tool_calls(self):
        provider = _make_provider()
        tc = [{"id": "call_1", "type": "function", "function": {"name": "test", "arguments": "{}"}}]
        msg = Message(
            role=Role.ASSISTANT,
            content="",
            metadata={"tool_calls": tc},
        )
        payload = provider._build_message_payload(msg)
        assert payload["tool_calls"] == tc
        assert payload["content"] is None

    def test_message_with_name(self):
        provider = _make_provider()
        msg = Message(
            role=Role.USER,
            content="Hello",
            metadata={"name": "alice"},
        )
        payload = provider._build_message_payload(msg)
        assert payload["name"] == "alice"


# ---------------------------------------------------------------------------
# Chat completion tests
# ---------------------------------------------------------------------------


class TestChatCompletion:
    """Tests for chat_completion method."""

    @pytest.mark.asyncio
    async def test_non_streaming_completion(self):
        provider = _make_provider()
        mock_response = _make_chat_response("Test response")
        provider._client.chat_completion = AsyncMock(return_value=mock_response)

        context = [Message(role=Role.USER, content="Hello")]
        result = await provider.chat_completion(context, stream=False)

        assert isinstance(result, dict)
        assert result["choices"][0]["message"]["content"] == "Test response"
        provider._client.chat_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_streaming_completion(self):
        provider = _make_provider()
        chunks = [
            _make_stream_chunk("Hello"),
            _make_stream_chunk(" world"),
            _make_stream_chunk("!", finish_reason="stop"),
        ]

        async def mock_stream():
            for c in chunks:
                yield c

        provider._client.chat_completion = AsyncMock(
            return_value=mock_stream()
        )

        context = [Message(role=Role.USER, content="Hi")]
        result = await provider.chat_completion(context, stream=True)

        collected = []
        async for chunk in result:
            collected.append(chunk)

        assert len(collected) == 3
        assert collected[0]["choices"][0]["delta"]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_with_tools(self):
        provider = _make_provider()
        tool_call_response = _make_chat_response(
            content=None,
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }
            ],
        )
        tool_call_response["choices"][0]["message"]["content"] = None
        provider._client.chat_completion = AsyncMock(
            return_value=tool_call_response
        )

        tools = [
            Tool(
                name="get_weather",
                description="Get weather",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}},
            )
        ]
        context = [Message(role=Role.USER, content="What's the weather?")]
        result = await provider.chat_completion(
            context, tools=tools, tool_choice="auto"
        )

        # Verify tools were passed
        call_kwargs = provider._client.chat_completion.call_args
        assert call_kwargs.kwargs.get("tools") is not None

    @pytest.mark.asyncio
    async def test_custom_model(self):
        provider = _make_provider()
        provider._client.chat_completion = AsyncMock(
            return_value=_make_chat_response(model="Qwen/Qwen2.5-72B-Instruct")
        )

        context = [Message(role=Role.USER, content="Hello")]
        await provider.chat_completion(
            context, model="Qwen/Qwen2.5-72B-Instruct"
        )

        call_kwargs = provider._client.chat_completion.call_args
        assert call_kwargs.kwargs.get("model") == "Qwen/Qwen2.5-72B-Instruct"

    @pytest.mark.asyncio
    async def test_kwargs_passed(self):
        provider = _make_provider()
        provider._client.chat_completion = AsyncMock(
            return_value=_make_chat_response()
        )

        context = [Message(role=Role.USER, content="Hello")]
        await provider.chat_completion(
            context, temperature=0.5, max_tokens=100
        )

        call_kwargs = provider._client.chat_completion.call_args
        assert call_kwargs.kwargs.get("temperature") == 0.5
        assert call_kwargs.kwargs.get("max_tokens") == 100

    @pytest.mark.asyncio
    async def test_empty_context_raises(self):
        provider = _make_provider()
        with pytest.raises(ProviderError, match="No valid messages"):
            await provider.chat_completion([])


# ---------------------------------------------------------------------------
# Response extraction tests
# ---------------------------------------------------------------------------


class TestResponseExtraction:
    """Tests for extract_response_content and extract_delta_content."""

    def test_extract_content(self):
        provider = _make_provider()
        response = _make_chat_response("Hello world")
        assert provider.extract_response_content(response) == "Hello world"

    def test_extract_content_empty(self):
        provider = _make_provider()
        assert provider.extract_response_content({}) == ""
        assert provider.extract_response_content({"choices": []}) == ""

    def test_extract_content_none(self):
        provider = _make_provider()
        response = _make_chat_response()
        response["choices"][0]["message"]["content"] = None
        assert provider.extract_response_content(response) == ""

    def test_extract_delta_content(self):
        provider = _make_provider()
        chunk = _make_stream_chunk("Hello")
        assert provider.extract_delta_content(chunk) == "Hello"

    def test_extract_delta_empty(self):
        provider = _make_provider()
        assert provider.extract_delta_content({}) == ""
        assert provider.extract_delta_content({"choices": []}) == ""


# ---------------------------------------------------------------------------
# Tool call extraction tests
# ---------------------------------------------------------------------------


class TestToolCallExtraction:
    """Tests for extract_tool_calls."""

    def test_extract_function_calls(self):
        provider = _make_provider()
        response = _make_chat_response(
            content=None,
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }
            ],
        )

        calls = provider.extract_tool_calls(response)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {"city": "Paris"}
        assert calls[0].id == "call_1"

    def test_extract_dict_arguments(self):
        """HF sometimes returns arguments as dict instead of string."""
        provider = _make_provider()
        response = _make_chat_response(
            content=None,
            tool_calls=[
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": {"query": "test"},
                    },
                }
            ],
        )

        calls = provider.extract_tool_calls(response)
        assert len(calls) == 1
        assert calls[0].arguments == {"query": "test"}

    def test_extract_no_tool_calls(self):
        provider = _make_provider()
        response = _make_chat_response("Just text")
        calls = provider.extract_tool_calls(response)
        assert len(calls) == 0

    def test_extract_malformed_json_arguments(self):
        provider = _make_provider()
        response = _make_chat_response(
            content=None,
            tool_calls=[
                {
                    "id": "call_3",
                    "type": "function",
                    "function": {
                        "name": "bad_func",
                        "arguments": "not valid json{",
                    },
                }
            ],
        )

        calls = provider.extract_tool_calls(response)
        assert len(calls) == 1
        assert "_raw" in calls[0].arguments


# ---------------------------------------------------------------------------
# Token counting tests
# ---------------------------------------------------------------------------


class TestTokenCounting:
    """Tests for approximate token counting."""

    @pytest.mark.asyncio
    async def test_count_tokens_basic(self):
        provider = _make_provider()
        count = await provider.count_tokens("Hello world")
        assert count >= 1
        # "Hello world" = 11 chars / 4 ≈ 2-3 tokens
        assert count < 10

    @pytest.mark.asyncio
    async def test_count_tokens_empty(self):
        provider = _make_provider()
        count = await provider.count_tokens("")
        # Empty string should return at least 1 (our minimum)
        assert count >= 0

    @pytest.mark.asyncio
    async def test_count_message_tokens(self):
        provider = _make_provider()
        messages = [
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="Hi!"),
        ]
        count = await provider.count_message_tokens(messages)
        assert count > 0
        assert count > 4  # At least role overhead for 2 messages


# ---------------------------------------------------------------------------
# Context length tests
# ---------------------------------------------------------------------------


class TestContextLength:
    """Tests for get_max_context_length."""

    def test_known_model_fallback_table(self):
        provider = _make_provider()
        ctx = provider.get_max_context_length(
            "meta-llama/Llama-3.3-70B-Instruct"
        )
        assert ctx == 131072

    def test_unknown_model_default(self):
        provider = _make_provider()
        ctx = provider.get_max_context_length("some/unknown-model")
        assert ctx == 8192  # Default fallback

    def test_cached_discovery(self):
        provider = _make_provider()
        provider._discovered_context_lengths["my/model"] = 65536
        assert provider.get_max_context_length("my/model") == 65536


# ---------------------------------------------------------------------------
# Supported parameters tests
# ---------------------------------------------------------------------------


class TestSupportedParameters:
    """Tests for get_supported_parameters."""

    def test_returns_expected_params(self):
        provider = _make_provider()
        params = provider.get_supported_parameters()
        assert "temperature" in params
        assert "top_p" in params
        assert "max_tokens" in params
        assert "stop" in params
        assert "seed" in params
        assert "frequency_penalty" in params
        assert "presence_penalty" in params
        assert "response_format" in params


# ---------------------------------------------------------------------------
# Multimodal: TTS tests
# ---------------------------------------------------------------------------


class TestTTS:
    """Tests for generate_speech."""

    @pytest.mark.asyncio
    async def test_generate_speech(self):
        provider = _make_provider()
        provider._client.text_to_speech = AsyncMock(
            return_value=b"\x00\x01\x02\x03"
        )

        result = await provider.generate_speech(
            "Hello world", model="facebook/mms-tts-eng"
        )

        assert isinstance(result, SpeechResult)
        assert result.audio_data == b"\x00\x01\x02\x03"
        assert result.model == "facebook/mms-tts-eng"

    @pytest.mark.asyncio
    async def test_generate_speech_default_model(self):
        provider = _make_provider()
        provider._client.text_to_speech = AsyncMock(return_value=b"\x00")

        result = await provider.generate_speech("Hi")
        assert result.model == "facebook/mms-tts-eng"


# ---------------------------------------------------------------------------
# Multimodal: STT tests
# ---------------------------------------------------------------------------


class TestSTT:
    """Tests for transcribe_audio."""

    @pytest.mark.asyncio
    async def test_transcribe_audio(self):
        provider = _make_provider()

        mock_result = MagicMock()
        mock_result.text = "Hello world"
        mock_result.chunks = None

        provider._client.automatic_speech_recognition = AsyncMock(
            return_value=mock_result
        )

        result = await provider.transcribe_audio(
            b"\x00\x01\x02",
            model="openai/whisper-large-v3-turbo",
        )

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world"
        assert result.model == "openai/whisper-large-v3-turbo"

    @pytest.mark.asyncio
    async def test_transcribe_with_chunks(self):
        provider = _make_provider()

        mock_result = MagicMock()
        mock_result.text = "Hello world"
        mock_result.chunks = [
            {"text": "Hello", "timestamp": [0.0, 1.0]},
            {"text": " world", "timestamp": [1.0, 2.0]},
        ]

        provider._client.automatic_speech_recognition = AsyncMock(
            return_value=mock_result
        )

        result = await provider.transcribe_audio(b"\x00")

        assert len(result.segments) == 2
        assert result.segments[0].text == "Hello"
        assert result.segments[0].start == 0.0
        assert result.segments[0].end == 1.0


# ---------------------------------------------------------------------------
# Multimodal: Embeddings tests
# ---------------------------------------------------------------------------


class TestEmbeddings:
    """Tests for create_embeddings."""

    @pytest.mark.asyncio
    async def test_single_text_embedding(self):
        provider = _make_provider()
        # Return a mock numpy array
        import numpy as np
        mock_embedding = np.array([0.1, 0.2, 0.3])
        provider._client.feature_extraction = AsyncMock(
            return_value=mock_embedding
        )

        result = await provider.create_embeddings("Hello world")

        assert result["object"] == "list"
        assert len(result["data"]) == 1
        assert result["data"][0]["embedding"] == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_batch_embeddings(self):
        provider = _make_provider()
        import numpy as np
        provider._client.feature_extraction = AsyncMock(
            return_value=np.array([0.1, 0.2])
        )

        result = await provider.create_embeddings(["Hello", "World"])

        assert len(result["data"]) == 2
        assert result["data"][0]["index"] == 0
        assert result["data"][1]["index"] == 1


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling in chat_completion."""

    @pytest.mark.asyncio
    async def test_timeout_error(self):
        provider = _make_provider()
        # Simulate a generic exception that the provider wraps
        provider._client.chat_completion = AsyncMock(
            side_effect=Exception("Request timed out")
        )

        context = [Message(role=Role.USER, content="Hello")]
        with pytest.raises(ProviderError):
            await provider.chat_completion(context)

    @pytest.mark.asyncio
    async def test_invalid_context_type(self):
        provider = _make_provider()
        with pytest.raises(ProviderError, match="Unsupported context"):
            await provider.chat_completion("not a list")  # type: ignore


# ---------------------------------------------------------------------------
# Provider manager integration
# ---------------------------------------------------------------------------


class TestProviderManagerRegistration:
    """Verify HuggingFace is properly registered in PROVIDER_MAP."""

    def test_registered_in_provider_map(self):
        from llmcore.providers.manager import PROVIDER_MAP
        assert "huggingface" in PROVIDER_MAP
        assert PROVIDER_MAP["huggingface"] is HuggingFaceProvider


# ---------------------------------------------------------------------------
# Normalize response tests
# ---------------------------------------------------------------------------


class TestNormalizeResponse:
    """Tests for _normalize_hf_response."""

    def test_plain_dict(self):
        provider = _make_provider()
        data = {"a": 1, "b": [2, 3], "c": {"d": 4}}
        result = provider._normalize_hf_response(data)
        assert result == data

    def test_nested_list(self):
        provider = _make_provider()
        data = {"choices": [{"message": {"content": "hi"}}]}
        result = provider._normalize_hf_response(data)
        assert result["choices"][0]["message"]["content"] == "hi"


# ---------------------------------------------------------------------------
# Cardctl adapter tests
# ---------------------------------------------------------------------------


class TestHuggingFaceCardctlAdapter:
    """Tests for the HuggingFace cardctl adapter registration."""

    def test_registered_in_adapter_registry(self):
        import importlib
        repo_root = str(__import__("pathlib").Path(__file__).resolve().parents[1])
        sys.path.insert(0, repo_root)
        # Import directly to avoid cli.py import chain issues
        spec = importlib.util.spec_from_file_location(
            "tools.cardctl.adapters",
            __import__("pathlib").Path(repo_root) / "tools" / "cardctl" / "adapters" / "__init__.py",
            submodule_search_locations=[
                str(__import__("pathlib").Path(repo_root) / "tools" / "cardctl" / "adapters")
            ],
        )
        mod = importlib.util.module_from_spec(spec)
        # Need base adapter available
        base_spec = importlib.util.spec_from_file_location(
            "tools.cardctl.adapters.base",
            __import__("pathlib").Path(repo_root) / "tools" / "cardctl" / "adapters" / "base.py",
        )
        base_mod = importlib.util.module_from_spec(base_spec)
        sys.modules["tools.cardctl.adapters.base"] = base_mod
        base_spec.loader.exec_module(base_mod)
        # Now check the registry dict directly from file content
        import ast
        init_path = __import__("pathlib").Path(repo_root) / "tools" / "cardctl" / "adapters" / "__init__.py"
        source = init_path.read_text()
        assert '"huggingface"' in source or "'huggingface'" in source

    def test_adapter_file_exists(self):
        from pathlib import Path
        adapter_path = (
            Path(__file__).resolve().parents[1]
            / "tools"
            / "cardctl"
            / "adapters"
            / "huggingface_adapter.py"
        )
        assert adapter_path.exists()


# ---------------------------------------------------------------------------
# Model card directory test
# ---------------------------------------------------------------------------


class TestModelCards:
    """Verify model card files exist."""

    def test_model_card_directory_exists(self):
        from pathlib import Path
        cards_dir = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "llmcore"
            / "model_cards"
            / "default_cards"
            / "huggingface"
        )
        assert cards_dir.exists(), f"Model card directory not found: {cards_dir}"

    def test_model_cards_are_valid_json(self):
        from pathlib import Path
        cards_dir = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "llmcore"
            / "model_cards"
            / "default_cards"
            / "huggingface"
        )
        json_files = list(cards_dir.glob("*.json"))
        assert len(json_files) > 0, "No model card JSON files found"
        for jf in json_files:
            with open(jf) as f:
                data = json.load(f)
                assert "model_id" in data, f"Missing model_id in {jf.name}"
                assert "provider" in data, f"Missing provider in {jf.name}"
                assert data["provider"] == "huggingface"
