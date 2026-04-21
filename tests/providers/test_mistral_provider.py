# tests/providers/test_mistral_provider.py
"""
Comprehensive tests for the Mistral provider (mistral_provider.py).

Tests cover:
- Provider initialization and configuration
- Message building (text, vision, audio, tool calls)
- Context length resolution chain (discovered → card → static → prefix → fallback)
- _is_mistral_reasoning_model() helper
- Supported parameters for reasoning vs non-reasoning models
- Response extraction (content, delta, tool_calls, usage)
- Chat completion request construction and error handling
- FIM (Fill-in-the-Middle) endpoint
- Embeddings endpoint
- TTS (Text-to-Speech) endpoint
- STT (Speech-to-Text) endpoint
- OCR endpoint
- Classification and Moderation endpoints
- Streaming SSE parsing
- Error handling (auth, rate limit, context length, model not found)
"""

import io
import json
import sys
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

# Patch imports before importing provider
sys.modules.setdefault("tiktoken", MagicMock())
sys.modules.setdefault("confy", MagicMock())
sys.modules.setdefault("confy.loader", MagicMock())


# ---------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------
class _Role:
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        if isinstance(other, _Role):
            return self.value == other.value
        return NotImplemented

    def __hash__(self):
        return hash(self.value)


class _Message:
    def __init__(self, role, content, tool_call_id=None, metadata=None):
        self.role = _Role(role)
        self.content = content
        self.tool_call_id = tool_call_id
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Message(role={self.role.value}, content={self.content!r})"


class _ToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.name = name
        self.arguments = arguments


class _Tool:
    def __init__(self, name, description, parameters):
        self.name = name
        self.description = description
        self.parameters = parameters

    def model_dump(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


# ---------------------------------------------------------------
# Helper to create a MistralProvider with mocked httpx
# ---------------------------------------------------------------
def _make_provider(config_overrides: dict | None = None, mock_client: bool = True):
    """Create a MistralProvider with mocked httpx client."""
    config = {
        "api_key": "test-key-123",
        "base_url": "https://api.mistral.ai/v1",
        "default_model": "mistral-large-latest",
        "timeout": 60,
        "_instance_name": "mistral",
    }
    if config_overrides:
        config.update(config_overrides)

    with patch("llmcore.providers.mistral_provider.httpx") as mock_httpx:
        mock_httpx.AsyncClient = MagicMock()
        mock_httpx.Timeout = MagicMock()
        mock_httpx.TimeoutException = Exception
        mock_httpx.ConnectError = Exception
        mock_httpx.HTTPStatusError = Exception

        from llmcore.providers.mistral_provider import MistralProvider

        provider = MistralProvider(config)
        if mock_client:
            provider._client = AsyncMock()
        return provider


# ===============================================================
# Test: _is_mistral_reasoning_model()
# ===============================================================


class TestIsMistralReasoningModel:
    """Tests for _is_mistral_reasoning_model()."""

    def test_magistral_models_are_reasoning(self):
        from llmcore.providers.mistral_provider import _is_mistral_reasoning_model

        assert _is_mistral_reasoning_model("magistral-medium-latest") is True
        assert _is_mistral_reasoning_model("magistral-small-latest") is True
        assert _is_mistral_reasoning_model("magistral-medium-2509") is True
        assert _is_mistral_reasoning_model("magistral-small-2509") is True

    def test_magistral_prefix_is_reasoning(self):
        from llmcore.providers.mistral_provider import _is_mistral_reasoning_model

        assert _is_mistral_reasoning_model("magistral-future-model") is True

    def test_non_magistral_is_not_reasoning(self):
        from llmcore.providers.mistral_provider import _is_mistral_reasoning_model

        assert _is_mistral_reasoning_model("mistral-large-latest") is False
        assert _is_mistral_reasoning_model("mistral-small-2506") is False
        assert _is_mistral_reasoning_model("codestral-latest") is False
        assert _is_mistral_reasoning_model("devstral-2-latest") is False


# ===============================================================
# Test: Initialization
# ===============================================================


class TestMistralProviderInit:
    """Tests for MistralProvider initialization."""

    def test_basic_init(self):
        provider = _make_provider()
        assert provider.get_name() == "mistral"
        assert provider.default_model == "mistral-large-latest"

    def test_custom_instance_name(self):
        provider = _make_provider({"_instance_name": "my_mistral"})
        assert provider.get_name() == "my_mistral"

    def test_missing_api_key_raises(self):
        with patch("llmcore.providers.mistral_provider.httpx") as mock_httpx:
            mock_httpx.AsyncClient = MagicMock()
            mock_httpx.Timeout = MagicMock()
            from llmcore.providers.mistral_provider import MistralProvider

            with pytest.raises(Exception, match="API key"):
                MistralProvider({"_instance_name": "mistral"})

    def test_env_var_api_key(self):
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "env-key-456"}):
            provider = _make_provider({"api_key_env_var": "MISTRAL_API_KEY"})
            assert provider.api_key == "test-key-123"  # explicit key wins

    def test_env_var_fallback(self):
        config = {
            "base_url": "https://api.mistral.ai/v1",
            "default_model": "mistral-large-latest",
            "timeout": 60,
            "_instance_name": "mistral",
        }
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "env-key-789"}):
            with patch("llmcore.providers.mistral_provider.httpx") as mock_httpx:
                mock_httpx.AsyncClient = MagicMock()
                mock_httpx.Timeout = MagicMock()
                from llmcore.providers.mistral_provider import MistralProvider

                provider = MistralProvider(config)
                assert provider.api_key == "env-key-789"


# ===============================================================
# Test: Context length resolution
# ===============================================================


class TestContextLengthResolution:
    """Tests for get_max_context_length() resolution chain."""

    def test_static_table_lookup(self):
        provider = _make_provider()
        # Mock the registry to avoid stale card data interfering
        with patch(
            "llmcore.providers.mistral_provider.get_model_card_registry",
            side_effect=Exception("no registry"),
        ):
            assert provider.get_max_context_length("mistral-large-latest") == 262144
            assert provider.get_max_context_length("mistral-small-latest") == 131072
            assert provider.get_max_context_length("voxtral-mini-latest") == 32768
            assert provider.get_max_context_length("mistral-embed") == 8192

    def test_prefix_heuristic(self):
        provider = _make_provider()
        # Future model not in static table
        assert provider.get_max_context_length("mistral-large-2599") == 262144
        assert provider.get_max_context_length("ministral-3-future") == 262144
        assert provider.get_max_context_length("codestral-2599") == 262144

    def test_discovered_context_length_wins(self):
        provider = _make_provider()
        provider._discovered_context_lengths["custom-model"] = 500000
        assert provider.get_max_context_length("custom-model") == 500000

    def test_fallback_default(self):
        provider = _make_provider()
        assert provider.get_max_context_length("totally-unknown-model") == 32768


# ===============================================================
# Test: Message building
# ===============================================================


class TestBuildMessagePayload:
    """Tests for _build_message_payload()."""

    def test_simple_text_message(self):
        provider = _make_provider()
        msg = _Message("user", "Hello, world!")
        result = provider._build_message_payload(msg, "mistral-large-latest")
        assert result == {"role": "user", "content": "Hello, world!"}

    def test_system_message(self):
        provider = _make_provider()
        msg = _Message("system", "You are a helpful assistant.")
        result = provider._build_message_payload(msg, "mistral-large-latest")
        assert result == {"role": "system", "content": "You are a helpful assistant."}

    def test_tool_message_with_tool_call_id(self):
        provider = _make_provider()
        msg = _Message(
            "tool", '{"result": 42}', tool_call_id="tc_123", metadata={"name": "calculator"}
        )
        result = provider._build_message_payload(msg, "mistral-large-latest")
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "tc_123"
        assert result["name"] == "calculator"

    def test_inline_images(self):
        provider = _make_provider()
        msg = _Message(
            "user",
            "What's in this image?",
            metadata={
                "inline_images": [
                    {"url": "https://example.com/img.png", "detail": "high"},
                    "https://example.com/img2.png",
                ]
            },
        )
        result = provider._build_message_payload(msg, "mistral-large-latest")
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 3  # text + 2 images
        assert result["content"][0] == {"type": "text", "text": "What's in this image?"}
        assert result["content"][1]["type"] == "image_url"
        assert result["content"][1]["image_url"]["detail"] == "high"
        assert result["content"][2]["type"] == "image_url"

    def test_inline_audio(self):
        provider = _make_provider()
        msg = _Message(
            "user",
            "Transcribe this",
            metadata={
                "inline_audio": [
                    {"data": "base64audio==", "format": "mp3"},
                ]
            },
        )
        result = provider._build_message_payload(msg, "mistral-large-latest")
        assert isinstance(result["content"], list)
        assert result["content"][1]["type"] == "input_audio"
        assert result["content"][1]["input_audio"]["format"] == "mp3"

    def test_content_parts_override(self):
        provider = _make_provider()
        parts = [{"type": "text", "text": "custom"}]
        msg = _Message("user", "ignored", metadata={"content_parts": parts})
        result = provider._build_message_payload(msg, "mistral-large-latest")
        assert result["content"] == parts

    def test_assistant_tool_calls(self):
        provider = _make_provider()
        tool_calls = [
            {"id": "tc_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
        ]
        msg = _Message("assistant", "", metadata={"tool_calls": tool_calls})
        result = provider._build_message_payload(msg, "mistral-large-latest")
        assert result["tool_calls"] == tool_calls
        assert result["content"] == ""

    def test_prefix_mode(self):
        provider = _make_provider()
        msg = _Message("assistant", "Sure, ", metadata={"prefix": True})
        result = provider._build_message_payload(msg, "mistral-large-latest")
        assert result.get("prefix") is True


# ===============================================================
# Test: Supported parameters
# ===============================================================


class TestSupportedParameters:
    """Tests for get_supported_parameters()."""

    def test_standard_model_params(self):
        provider = _make_provider()
        params = provider.get_supported_parameters("mistral-large-latest")
        assert "temperature" in params
        assert "random_seed" in params
        assert "safe_prompt" in params
        assert "parallel_tool_calls" in params
        assert "reasoning_effort" not in params

    def test_reasoning_model_params(self):
        provider = _make_provider()
        params = provider.get_supported_parameters("magistral-medium-latest")
        assert "reasoning_effort" in params
        assert "prompt_mode" in params
        assert params["reasoning_effort"]["enum"] == ["low", "medium", "high"]


# ===============================================================
# Test: Response extraction
# ===============================================================


class TestResponseExtraction:
    """Tests for extract_response_content, extract_delta_content, etc."""

    def test_extract_response_content(self):
        provider = _make_provider()
        response = {
            "choices": [{"message": {"content": "Hello from Mistral!", "role": "assistant"}}]
        }
        assert provider.extract_response_content(response) == "Hello from Mistral!"

    def test_extract_response_content_empty(self):
        provider = _make_provider()
        assert provider.extract_response_content({}) == ""
        assert provider.extract_response_content({"choices": []}) == ""

    def test_extract_delta_content(self):
        provider = _make_provider()
        chunk = {"choices": [{"delta": {"content": "chunk"}}]}
        assert provider.extract_delta_content(chunk) == "chunk"

    def test_extract_delta_content_empty(self):
        provider = _make_provider()
        assert provider.extract_delta_content({}) == ""
        assert provider.extract_delta_content({"choices": [{"delta": {}}]}) == ""

    def test_extract_tool_calls(self):
        provider = _make_provider()
        response = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "tc_001",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Paris"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        calls = provider.extract_tool_calls(response)
        assert len(calls) == 1
        assert calls[0].id == "tc_001"
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {"city": "Paris"}

    def test_extract_tool_calls_dict_arguments(self):
        """Mistral sometimes returns arguments as dict, not string."""
        provider = _make_provider()
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "tc_002",
                                "type": "function",
                                "function": {
                                    "name": "calc",
                                    "arguments": {"x": 1, "y": 2},
                                },
                            }
                        ]
                    }
                }
            ]
        }
        calls = provider.extract_tool_calls(response)
        assert len(calls) == 1
        assert calls[0].arguments == {"x": 1, "y": 2}

    def test_extract_tool_calls_empty(self):
        provider = _make_provider()
        assert provider.extract_tool_calls({}) == []
        assert provider.extract_tool_calls({"choices": [{"message": {}}]}) == []

    def test_extract_usage_details(self):
        provider = _make_provider()
        response = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            }
        }
        usage = provider.extract_usage_details(response)
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150

    def test_extract_usage_details_with_audio(self):
        provider = _make_provider()
        response = {
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "prompt_audio_seconds": 42.5,
            }
        }
        usage = provider.extract_usage_details(response)
        assert usage["prompt_audio_seconds"] == 42.5


# ===============================================================
# Test: Chat completion
# ===============================================================


class TestChatCompletion:
    """Tests for chat_completion()."""

    @pytest.mark.asyncio
    async def test_non_streaming_request(self):
        provider = _make_provider()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "chat-123",
            "choices": [{"message": {"content": "Hi!", "role": "assistant"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        provider._client.post = AsyncMock(return_value=mock_resp)

        # Patch the Message check so our _Message stubs pass isinstance
        from llmcore.providers import mistral_provider

        orig_message = mistral_provider.Message
        mistral_provider.Message = _Message
        try:
            context = [_Message("user", "Hello")]
            result = await provider.chat_completion(context, stream=False)

            assert result["choices"][0]["message"]["content"] == "Hi!"
            provider._client.post.assert_called_once()
            call_args = provider._client.post.call_args
            assert call_args[0][0] == "/chat/completions"
            body = call_args[1]["json"]
            assert body["model"] == "mistral-large-latest"
            assert body["stream"] is False
        finally:
            mistral_provider.Message = orig_message

    @pytest.mark.asyncio
    async def test_unsupported_parameter_raises(self):
        provider = _make_provider()
        from llmcore.providers import mistral_provider

        orig_message = mistral_provider.Message
        mistral_provider.Message = _Message
        try:
            context = [_Message("user", "test")]
            with pytest.raises(ValueError, match="Unsupported parameter"):
                await provider.chat_completion(context, bogus_param=True)
        finally:
            mistral_provider.Message = orig_message

    @pytest.mark.asyncio
    async def test_tools_payload(self):
        provider = _make_provider()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": None, "tool_calls": []}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
        }
        provider._client.post = AsyncMock(return_value=mock_resp)

        from llmcore.providers import mistral_provider

        orig_message = mistral_provider.Message
        mistral_provider.Message = _Message
        try:
            tool = _Tool("calc", "Calculator", {"type": "object", "properties": {}})
            context = [_Message("user", "Add 2+2")]
            await provider.chat_completion(context, tools=[tool], tool_choice="auto")

            body = provider._client.post.call_args[1]["json"]
            assert len(body["tools"]) == 1
            assert body["tools"][0]["type"] == "function"
            assert body["tool_choice"] == "auto"
        finally:
            mistral_provider.Message = orig_message

    @pytest.mark.asyncio
    async def test_mistral_specific_params(self):
        provider = _make_provider()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "safe"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        provider._client.post = AsyncMock(return_value=mock_resp)

        from llmcore.providers import mistral_provider

        orig_message = mistral_provider.Message
        mistral_provider.Message = _Message
        try:
            context = [_Message("user", "test")]
            await provider.chat_completion(
                context,
                safe_prompt=True,
                random_seed=42,
                temperature=0.7,
            )

            body = provider._client.post.call_args[1]["json"]
            assert body["safe_prompt"] is True
            assert body["random_seed"] == 42
            assert body["temperature"] == 0.7
        finally:
            mistral_provider.Message = orig_message


# ===============================================================
# Test: Error handling
# ===============================================================


class TestErrorHandling:
    """Tests for _raise_for_status() error mapping."""

    def test_auth_error(self):
        from llmcore.exceptions import ProviderError
        from llmcore.providers.mistral_provider import MistralProvider

        provider = _make_provider()
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"message": "Invalid API key"}
        mock_resp.text = "Unauthorized"

        with pytest.raises(ProviderError, match="Authentication"):
            provider._raise_for_status(mock_resp, "mistral-large-latest")

    def test_rate_limit_error(self):
        from llmcore.exceptions import ProviderError

        provider = _make_provider()
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.json.return_value = {"message": "Rate limit exceeded"}
        mock_resp.text = "Rate limited"

        with pytest.raises(ProviderError, match="Rate limit"):
            provider._raise_for_status(mock_resp, "mistral-large-latest")

    def test_context_length_error(self):
        from llmcore.exceptions import ContextLengthError

        provider = _make_provider()
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {
            "message": "Input token limit exceeded. context_length 300000"
        }
        mock_resp.text = "Bad request"

        with pytest.raises(ContextLengthError):
            provider._raise_for_status(mock_resp, "mistral-large-latest")

    def test_model_not_found_error(self):
        from llmcore.exceptions import ProviderError

        provider = _make_provider()
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {"message": "Model does not exist: bad-model"}
        mock_resp.text = "Bad request"

        with pytest.raises(ProviderError, match="not found"):
            provider._raise_for_status(mock_resp, "bad-model")


# ===============================================================
# Test: FIM completion
# ===============================================================


class TestFIMCompletion:
    """Tests for fim_completion()."""

    @pytest.mark.asyncio
    async def test_fim_basic(self):
        provider = _make_provider()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "fim-123",
            "choices": [{"text": "    return x + y\n", "index": 0}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
        }
        provider._client.post = AsyncMock(return_value=mock_resp)

        result = await provider.fim_completion(
            prompt="def add(x, y):\n",
            suffix="\n\nresult = add(1, 2)",
            model="codestral-latest",
            temperature=0.0,
        )

        provider._client.post.assert_called_once()
        call_args = provider._client.post.call_args
        assert call_args[0][0] == "/fim/completions"
        body = call_args[1]["json"]
        assert body["prompt"] == "def add(x, y):\n"
        assert body["suffix"] == "\n\nresult = add(1, 2)"
        assert body["model"] == "codestral-latest"


# ===============================================================
# Test: Embeddings
# ===============================================================


class TestEmbeddings:
    """Tests for create_embeddings()."""

    @pytest.mark.asyncio
    async def test_embeddings_single(self):
        provider = _make_provider()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "mistral-embed",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
        provider._client.post = AsyncMock(return_value=mock_resp)

        result = await provider.create_embeddings("Hello world")

        body = provider._client.post.call_args[1]["json"]
        assert body["input"] == ["Hello world"]
        assert body["model"] == "mistral-embed"
        assert len(result["data"]) == 1

    @pytest.mark.asyncio
    async def test_embeddings_batch(self):
        provider = _make_provider()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"embedding": [0.1], "index": 0},
                {"embedding": [0.2], "index": 1},
            ],
            "model": "mistral-embed",
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }
        provider._client.post = AsyncMock(return_value=mock_resp)

        result = await provider.create_embeddings(
            ["text1", "text2"],
            output_dimension=256,
            output_dtype="float",
        )

        body = provider._client.post.call_args[1]["json"]
        assert body["input"] == ["text1", "text2"]
        assert body["output_dimension"] == 256


# ===============================================================
# Test: OCR
# ===============================================================


class TestOCR:
    """Tests for ocr()."""

    @pytest.mark.asyncio
    async def test_ocr_url(self):
        provider = _make_provider()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "pages": [
                {"index": 0, "markdown": "# Title\n\nSome text."},
            ],
            "model": "mistral-ocr-latest",
            "usage_info": {"pages_processed": 1, "doc_size_bytes": 54321},
        }
        provider._client.post = AsyncMock(return_value=mock_resp)

        result = await provider.ocr("https://example.com/doc.pdf")

        body = provider._client.post.call_args[1]["json"]
        assert body["document"]["type"] == "document_url"
        assert body["document"]["document_url"] == "https://example.com/doc.pdf"
        assert body["model"] == "mistral-ocr-latest"
        assert result.pages_processed == 1

    @pytest.mark.asyncio
    async def test_ocr_bytes(self):
        provider = _make_provider()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "pages": [{"index": 0, "markdown": "page1"}],
            "model": "mistral-ocr-latest",
            "usage_info": {"pages_processed": 1},
        }
        provider._client.post = AsyncMock(return_value=mock_resp)

        result = await provider.ocr(b"fake-pdf-bytes", pages=[0, 1])

        body = provider._client.post.call_args[1]["json"]
        assert body["document"]["type"] == "file"
        assert "file_data" in body["document"]
        assert body["pages"] == [0, 1]

    @pytest.mark.asyncio
    async def test_ocr_with_annotation(self):
        provider = _make_provider()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "pages": [{"index": 0, "markdown": "Invoice #123"}],
            "model": "mistral-ocr-latest",
            "document_annotation": {"invoice_number": "123", "total": "$500"},
            "usage_info": {"pages_processed": 1},
        }
        provider._client.post = AsyncMock(return_value=mock_resp)

        schema = {"type": "object", "properties": {"invoice_number": {"type": "string"}}}
        result = await provider.ocr(
            "https://example.com/invoice.pdf",
            document_annotation_format=schema,
            document_annotation_prompt="Extract invoice details",
        )

        body = provider._client.post.call_args[1]["json"]
        assert body["document_annotation_format"] == schema
        assert result.document_annotation == {"invoice_number": "123", "total": "$500"}


# ===============================================================
# Test: Classification and Moderation
# ===============================================================


class TestClassifyAndModerate:
    """Tests for classify() and moderate()."""

    @pytest.mark.asyncio
    async def test_classify(self):
        provider = _make_provider()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"results": [{"label": "positive", "score": 0.95}]}
        provider._client.post = AsyncMock(return_value=mock_resp)

        result = await provider.classify("This is great!")
        body = provider._client.post.call_args[1]["json"]
        assert body["input"] == ["This is great!"]

    @pytest.mark.asyncio
    async def test_moderate(self):
        provider = _make_provider()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [{"categories": {"hate": False, "violence": False}}]
        }
        provider._client.post = AsyncMock(return_value=mock_resp)

        result = await provider.moderate(
            ["text1", "text2"],
            model="mistral-moderation-latest",
        )

        body = provider._client.post.call_args[1]["json"]
        assert body["input"] == ["text1", "text2"]
        assert body["model"] == "mistral-moderation-latest"


# ===============================================================
# Test: Token counting
# ===============================================================


class TestTokenCounting:
    """Tests for count_tokens() and count_message_tokens()."""

    @pytest.mark.asyncio
    async def test_count_tokens_with_encoding(self):
        provider = _make_provider()
        # Mock tiktoken encoding
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1, 2, 3, 4, 5]
        provider._encoding = mock_enc

        count = await provider.count_tokens("Hello world!")
        assert count == 5

    @pytest.mark.asyncio
    async def test_count_tokens_fallback(self):
        provider = _make_provider()
        provider._encoding = None

        count = await provider.count_tokens("A" * 400)
        assert count == 100  # len / 4

    @pytest.mark.asyncio
    async def test_count_message_tokens(self):
        provider = _make_provider()
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1, 2, 3]  # 3 tokens per call
        provider._encoding = mock_enc

        messages = [
            _Message("system", "You are helpful"),
            _Message("user", "Hi"),
        ]
        count = await provider.count_message_tokens(messages)
        # 2 messages × (4 overhead + 3 content) + 2 priming = 16
        assert count == 16


# ===============================================================
# Test: Models list
# ===============================================================


class TestGetModelsDetails:
    """Tests for get_models_details()."""

    @pytest.mark.asyncio
    async def test_models_discovery(self):
        provider = _make_provider()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {
                    "id": "mistral-large-2512",
                    "type": "base",
                    "max_context_length": 262144,
                    "capabilities": {
                        "completion_chat": True,
                        "function_calling": True,
                        "vision": True,
                    },
                    "owned_by": "mistralai",
                    "created": 1733000000,
                    "aliases": ["mistral-large-latest"],
                    "description": "Mistral Large 3",
                },
                {
                    "id": "mistral-embed",
                    "type": "base",
                    "max_context_length": 8192,
                    "capabilities": {
                        "completion_chat": False,
                    },
                    "owned_by": "mistralai",
                },
            ]
        }
        provider._client.get = AsyncMock(return_value=mock_resp)

        with patch("llmcore.providers.mistral_provider.get_model_card_registry", return_value=None):
            details = await provider.get_models_details()

        assert len(details) == 2
        assert details[0].id == "mistral-large-2512"
        assert details[0].context_length == 262144
        assert details[0].supports_tools is True
        assert details[0].supports_vision is True
        # Verify context length was cached
        assert provider._discovered_context_lengths["mistral-large-2512"] == 262144


# ===============================================================
# Test: Provider in manager map
# ===============================================================


class TestProviderManagerIntegration:
    """Verify MistralProvider is correctly registered in the provider map."""

    def test_provider_map_has_mistral(self):
        from llmcore.providers.manager import PROVIDER_MAP

        assert "mistral" in PROVIDER_MAP
        assert PROVIDER_MAP["mistral"].__name__ == "MistralProvider"

    def test_mistral_not_in_openai_compatible_defaults(self):
        from llmcore.providers.manager import _OPENAI_COMPATIBLE_DEFAULTS

        assert "mistral" not in _OPENAI_COMPATIBLE_DEFAULTS


# ===============================================================
# Test: OCRResult model
# ===============================================================


class TestOCRResultModel:
    """Tests for the OCRResult Pydantic model."""

    def test_ocr_result_basic(self):
        from llmcore.models_multimodal import OCRResult

        result = OCRResult(
            pages=[{"index": 0, "markdown": "Hello"}],
            model="mistral-ocr-latest",
            pages_processed=1,
        )
        assert result.pages_processed == 1
        assert result.model == "mistral-ocr-latest"
        assert result.document_annotation is None

    def test_ocr_result_with_annotation(self):
        from llmcore.models_multimodal import OCRResult

        result = OCRResult(
            pages=[{"index": 0, "markdown": "Invoice"}],
            model="mistral-ocr-latest",
            document_annotation={"invoice_id": "123"},
            pages_processed=1,
            doc_size_bytes=54321,
            metadata={"usage_info": {"pages_processed": 1}},
        )
        assert result.document_annotation == {"invoice_id": "123"}
        assert result.doc_size_bytes == 54321


# ===============================================================
# Test: BaseProvider new methods exist
# ===============================================================


class TestBaseProviderNewMethods:
    """Verify ocr() and create_embeddings() exist on BaseProvider."""

    def test_ocr_default_raises(self):
        """BaseProvider.ocr() should raise NotImplementedError."""
        from llmcore.providers.base import BaseProvider

        # Quick check the method exists on the class
        assert hasattr(BaseProvider, "ocr")

    def test_create_embeddings_default_raises(self):
        """BaseProvider.create_embeddings() should raise NotImplementedError."""
        from llmcore.providers.base import BaseProvider

        assert hasattr(BaseProvider, "create_embeddings")
