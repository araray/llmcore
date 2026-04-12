# tests/providers/test_gemini_provider.py
"""
Unit tests for the GeminiProvider covering:
- Multimodal content building (_build_multimodal_parts)
- Message-to-contents conversion (_convert_llmcore_msgs_to_genai_contents)
- Tool call normalization (_normalize_tool_calls_from_response)
- Tool call extraction (extract_tool_calls)
- Response content extraction (extract_response_content, extract_delta_content)
- Supported parameters schema
- Context length resolution chain
- Default model and token limits updates

All tests use mocking — no live API calls.
"""

import json

# We need to mock the google imports before importing the provider
import sys
import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Create mock modules for google.genai
mock_genai = MagicMock()
mock_types = MagicMock()
mock_api_core_exceptions = MagicMock()
mock_errors = MagicMock()

# Patch sys.modules before importing the provider module
with patch.dict(
    sys.modules,
    {
        "google": MagicMock(),
        "google.genai": mock_genai,
        "google.genai.types": mock_types,
        "google.genai.errors": mock_errors,
        "google.api_core": MagicMock(),
        "google.api_core.exceptions": mock_api_core_exceptions,
    },
):
    # Now set the module-level flags
    import importlib

    # We need a fresh import
    from llmcore.providers import gemini_provider as gp

    gp.google_genai_available = True
    gp.genai = mock_genai
    gp.types = mock_types
    gp.APIError = Exception
    gp.PermissionDenied = PermissionError
    gp.InvalidArgument = ValueError

from llmcore.models import Message, ModelDetails, Tool, ToolCall
from llmcore.models import Role as LLMCoreRole

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def provider_config():
    """Minimal provider config with fake API key."""
    return {
        "api_key": "fake-api-key-for-testing",
        "default_model": "gemini-3.1-flash-lite-preview",
    }


@pytest.fixture
def provider(provider_config):
    """Create a GeminiProvider with mocked client."""
    with patch.object(gp, "genai") as mock_g:
        mock_client = MagicMock()
        mock_g.Client.return_value = mock_client
        p = gp.GeminiProvider(provider_config)
        p._client = mock_client
        return p


# ---------------------------------------------------------------------------
# Test: Default model and token limits updated
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_default_model_is_current_gen(self):
        """DEFAULT_MODEL should be gemini-3.1-flash-lite-preview."""
        assert gp.DEFAULT_MODEL == "gemini-3.1-flash-lite-preview"

    def test_token_limits_include_current_models(self):
        """Token limits table should include Gemini 3.x and 2.5 models."""
        assert "gemini-3.1-pro-preview" in gp.DEFAULT_GEMINI_TOKEN_LIMITS
        assert "gemini-3-flash-preview" in gp.DEFAULT_GEMINI_TOKEN_LIMITS
        assert "gemini-3.1-flash-lite-preview" in gp.DEFAULT_GEMINI_TOKEN_LIMITS
        assert "gemini-2.5-flash" in gp.DEFAULT_GEMINI_TOKEN_LIMITS
        assert "gemini-2.5-pro" in gp.DEFAULT_GEMINI_TOKEN_LIMITS

    def test_legacy_models_removed(self):
        """Shutdown 1.x models should not be in the fallback table."""
        assert "gemini-1.0-pro" not in gp.DEFAULT_GEMINI_TOKEN_LIMITS
        assert "gemini-1.5-flash-latest" not in gp.DEFAULT_GEMINI_TOKEN_LIMITS


# ---------------------------------------------------------------------------
# Test: Multimodal content building
# ---------------------------------------------------------------------------


class TestBuildMultimodalParts:
    def test_text_only(self, provider):
        """Plain text message produces single text Part."""
        msg = Message(role=LLMCoreRole.USER, content="Hello world")
        parts = provider._build_multimodal_parts(msg)
        assert parts == [{"text": "Hello world"}]

    def test_empty_content_no_metadata(self, provider):
        """Empty content with no metadata produces empty text Part."""
        msg = Message(role=LLMCoreRole.USER, content="")
        parts = provider._build_multimodal_parts(msg)
        assert parts == [{"text": ""}]

    def test_inline_images(self, provider):
        """metadata["inline_images"] produces inline_data Parts."""
        msg = Message(
            role=LLMCoreRole.USER,
            content="What is this?",
            metadata={"inline_images": [{"mime_type": "image/png", "data": "base64encodeddata"}]},
        )
        parts = provider._build_multimodal_parts(msg)
        assert len(parts) == 2
        assert parts[0] == {"text": "What is this?"}
        assert parts[1] == {"inline_data": {"mime_type": "image/png", "data": "base64encodeddata"}}

    def test_file_uris(self, provider):
        """metadata["file_uris"] produces file_data Parts."""
        msg = Message(
            role=LLMCoreRole.USER,
            content="Analyze this",
            metadata={
                "file_uris": [{"file_uri": "gs://bucket/image.jpg", "mime_type": "image/jpeg"}]
            },
        )
        parts = provider._build_multimodal_parts(msg)
        assert len(parts) == 2
        assert parts[1]["file_data"]["file_uri"] == "gs://bucket/image.jpg"

    def test_raw_parts_passthrough(self, provider):
        """metadata["parts"] are passed through directly."""
        custom_part = {"inline_data": {"mime_type": "audio/wav", "data": "audiodata"}}
        msg = Message(
            role=LLMCoreRole.USER,
            content="",
            metadata={"parts": [custom_part]},
        )
        parts = provider._build_multimodal_parts(msg)
        # Empty content (falsy) does NOT add a text part; only the raw part
        assert len(parts) == 1
        assert parts[0] == custom_part

    def test_combined_multimodal(self, provider):
        """Text + inline images + file URIs all combine."""
        msg = Message(
            role=LLMCoreRole.USER,
            content="Describe all",
            metadata={
                "inline_images": [{"mime_type": "image/png", "data": "img1"}],
                "file_uris": [{"file_uri": "gs://b/f.pdf", "mime_type": "application/pdf"}],
            },
        )
        parts = provider._build_multimodal_parts(msg)
        assert len(parts) == 3


# ---------------------------------------------------------------------------
# Test: Message conversion
# ---------------------------------------------------------------------------


class TestConvertMessages:
    def test_system_extraction(self, provider):
        """System messages are extracted as system_instruction text."""
        msgs = [
            Message(role=LLMCoreRole.SYSTEM, content="Be helpful"),
            Message(role=LLMCoreRole.USER, content="Hi"),
        ]
        contents, sys_text = provider._convert_llmcore_msgs_to_genai_contents(msgs)
        assert sys_text == "Be helpful"
        assert len(contents) == 1
        assert contents[0]["role"] == "user"

    def test_multiple_system_messages(self, provider):
        """Multiple leading system messages are concatenated."""
        msgs = [
            Message(role=LLMCoreRole.SYSTEM, content="Rule 1"),
            Message(role=LLMCoreRole.SYSTEM, content="Rule 2"),
            Message(role=LLMCoreRole.USER, content="Hi"),
        ]
        contents, sys_text = provider._convert_llmcore_msgs_to_genai_contents(msgs)
        assert sys_text == "Rule 1\nRule 2"

    def test_role_mapping(self, provider):
        """User -> 'user', Assistant -> 'model'."""
        msgs = [
            Message(role=LLMCoreRole.USER, content="question"),
            Message(role=LLMCoreRole.ASSISTANT, content="answer"),
        ]
        contents, _ = provider._convert_llmcore_msgs_to_genai_contents(msgs)
        assert contents[0]["role"] == "user"
        assert contents[1]["role"] == "model"

    def test_consecutive_same_role_merged(self, provider):
        """Consecutive messages with same role are merged (Gemini requirement)."""
        msgs = [
            Message(role=LLMCoreRole.USER, content="part 1"),
            Message(role=LLMCoreRole.USER, content="part 2"),
        ]
        contents, _ = provider._convert_llmcore_msgs_to_genai_contents(msgs)
        assert len(contents) == 1
        assert len(contents[0]["parts"]) == 2

    def test_tool_message_converted_to_function_response(self, provider):
        """Tool messages become functionResponse parts with user role."""
        msgs = [
            Message(role=LLMCoreRole.USER, content="call tool"),
            Message(
                role=LLMCoreRole.TOOL,
                content='{"result": 42}',
                tool_call_id="call_123",
                metadata={"tool_name": "get_answer"},
            ),
        ]
        contents, _ = provider._convert_llmcore_msgs_to_genai_contents(msgs)
        # Tool message should be merged into a user-role content
        assert len(contents) == 2 or (len(contents) == 1 and contents[0]["role"] == "user")
        # Find the function_response part
        found = False
        for c in contents:
            for p in c["parts"]:
                if "function_response" in p:
                    assert p["function_response"]["name"] == "get_answer"
                    found = True
        assert found


# ---------------------------------------------------------------------------
# Test: Tool call normalization
# ---------------------------------------------------------------------------


class TestToolCallNormalization:
    def test_no_function_calls(self, provider):
        """Response without function_calls returns None."""
        response = SimpleNamespace(function_calls=None)
        assert provider._normalize_tool_calls_from_response(response) is None

    def test_single_function_call(self, provider):
        """Single function call is normalized to OpenAI format."""
        fc = SimpleNamespace(
            id="call_abc",
            name="get_weather",
            args={"location": "Tokyo"},
        )
        response = SimpleNamespace(function_calls=[fc])
        result = provider._normalize_tool_calls_from_response(response)
        assert len(result) == 1
        assert result[0]["id"] == "call_abc"
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert json.loads(result[0]["function"]["arguments"]) == {"location": "Tokyo"}

    def test_missing_id_generates_uuid(self, provider):
        """Function call without id gets a generated UUID."""
        fc = SimpleNamespace(id=None, name="do_thing", args={})
        response = SimpleNamespace(function_calls=[fc])
        result = provider._normalize_tool_calls_from_response(response)
        assert len(result[0]["id"]) > 0  # UUID was generated


# ---------------------------------------------------------------------------
# Test: extract_tool_calls (public API)
# ---------------------------------------------------------------------------


class TestExtractToolCalls:
    def test_extract_from_normalized_response(self, provider):
        """extract_tool_calls parses OpenAI-format tool_calls correctly."""
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "tc_1",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"query": "hello"}',
                                },
                            }
                        ],
                    },
                }
            ],
        }
        calls = provider.extract_tool_calls(response)
        assert len(calls) == 1
        assert isinstance(calls[0], ToolCall)
        assert calls[0].name == "search"
        assert calls[0].arguments == {"query": "hello"}

    def test_extract_empty_response(self, provider):
        """No tool_calls in response returns empty list."""
        response = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
        assert provider.extract_tool_calls(response) == []

    def test_extract_malformed_json_arguments(self, provider):
        """Malformed JSON in arguments is captured as raw string."""
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
                                    "arguments": "not valid json {{{",
                                },
                            }
                        ],
                    },
                }
            ],
        }
        calls = provider.extract_tool_calls(response)
        assert len(calls) == 1
        assert "raw" in calls[0].arguments


# ---------------------------------------------------------------------------
# Test: Response content extraction
# ---------------------------------------------------------------------------


class TestExtractResponseContent:
    def test_from_openai_normalized(self, provider):
        """Extract content from OpenAI-normalized dict."""
        response = {"choices": [{"message": {"role": "assistant", "content": "hello"}}]}
        assert provider.extract_response_content(response) == "hello"

    def test_from_native_gemini_dict(self, provider):
        """Extract content from native Gemini dict format."""
        response = {"candidates": [{"content": {"parts": [{"text": "world"}]}}]}
        assert provider.extract_response_content(response) == "world"

    def test_from_object_with_text(self, provider):
        """Extract content from response object with .text property."""
        response = SimpleNamespace(text="from object")
        assert provider.extract_response_content(response) == "from object"

    def test_empty_response(self, provider):
        """Empty/missing content returns empty string."""
        assert provider.extract_response_content({}) == ""
        assert provider.extract_response_content({"choices": []}) == ""


# ---------------------------------------------------------------------------
# Test: Streaming delta extraction
# ---------------------------------------------------------------------------


class TestExtractDeltaContent:
    def test_from_openai_normalized_chunk(self, provider):
        """Extract delta from OpenAI-format streaming chunk."""
        chunk = {"choices": [{"delta": {"content": "delta text"}}]}
        assert provider.extract_delta_content(chunk) == "delta text"

    def test_from_object_with_text(self, provider):
        """Extract delta from response object with .text property."""
        chunk = SimpleNamespace(text="streamed")
        assert provider.extract_delta_content(chunk) == "streamed"

    def test_empty_chunk(self, provider):
        """Empty chunk returns empty string."""
        assert provider.extract_delta_content({}) == ""


# ---------------------------------------------------------------------------
# Test: Supported parameters
# ---------------------------------------------------------------------------


class TestSupportedParameters:
    def test_includes_new_params(self, provider):
        """Supported params include v1.72.0 additions."""
        params = provider.get_supported_parameters()
        assert "thinking_config" in params
        assert "speech_config" in params
        assert "response_schema" in params
        assert "response_json_schema" in params
        assert "seed" in params
        assert "presence_penalty" in params
        assert "frequency_penalty" in params
        assert "response_modalities" in params
        assert "cached_content" in params
        assert "response_logprobs" in params


# ---------------------------------------------------------------------------
# Test: Context length resolution
# ---------------------------------------------------------------------------


class TestContextLength:
    def test_dynamic_cache_used(self, provider):
        """Dynamic discovery cache is consulted when no model card exists."""
        gp._discovered_context_lengths["test-dynamic-model"] = 999999
        try:
            assert provider.get_max_context_length("test-dynamic-model") == 999999
        finally:
            del gp._discovered_context_lengths["test-dynamic-model"]

    def test_static_table_fallback(self, provider):
        """Static table is used when both card and dynamic cache miss."""
        assert provider.get_max_context_length("gemini-2.0-flash") == 1048576

    def test_configured_fallback(self, provider):
        """Configured fallback used when all sources miss."""
        length = provider.get_max_context_length("totally-unknown-model")
        assert length == provider.fallback_context_length

    def test_default_model_in_static_table(self, provider):
        """The default model should be in the static fallback table."""
        assert provider.default_model in gp.DEFAULT_GEMINI_TOKEN_LIMITS


# ---------------------------------------------------------------------------
# Test: Provider name
# ---------------------------------------------------------------------------


class TestProviderName:
    def test_default_name(self, provider):
        assert provider.get_name() == "gemini"

    def test_instance_name_override(self, provider):
        provider._provider_instance_name = "google-vertex"
        assert provider.get_name() == "google-vertex"
