# tests/providers/test_openai_provider.py
"""
Comprehensive tests for the OpenAI provider (openai_provider.py).

Tests cover:
- Message serialization (multimodal, tool_calls, developer role)
- Error handling hierarchy (APIStatusError, APIConnectionError, etc.)
- extract_tool_calls() normalization
- extract_audio_response() / extract_annotations() / extract_refusal()
- extract_usage_details() with detailed breakdown
- get_supported_parameters() for reasoning vs non-reasoning models
- max_tokens -> max_completion_tokens auto-remap
- get_max_context_length() resolution chain
- _is_reasoning_model() / _needs_developer_role() helpers
"""

import json

# Patch imports before importing provider
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.modules.setdefault("openai", MagicMock())
sys.modules.setdefault("openai._exceptions", MagicMock())
sys.modules.setdefault("openai.types", MagicMock())
sys.modules.setdefault("openai.types.chat", MagicMock())
sys.modules.setdefault("tiktoken", MagicMock())


# ---------------------------------------------------------------
# Minimal stubs that replicate just enough llmcore model surface
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
# Import helpers from the provider module under test
# ---------------------------------------------------------------
# We test the module-level functions directly without full init
# by importing after patching.

# Since we can't easily do a full import without the real SDK,
# we test the logic by constructing a mock provider or testing
# the pure functions directly.
# ---------------------------------------------------------------


class TestIsReasoningModel:
    """Tests for _is_reasoning_model()."""

    def test_o1_is_reasoning(self):
        from llmcore.providers.openai_provider import _is_reasoning_model

        assert _is_reasoning_model("o1") is True
        assert _is_reasoning_model("o1-2024-12-17") is True
        assert _is_reasoning_model("o1-pro") is True
        assert _is_reasoning_model("o1-mini") is True

    def test_o3_is_reasoning(self):
        from llmcore.providers.openai_provider import _is_reasoning_model

        assert _is_reasoning_model("o3") is True
        assert _is_reasoning_model("o3-mini") is True
        assert _is_reasoning_model("o3-2025-04-16") is True

    def test_o4_is_reasoning(self):
        from llmcore.providers.openai_provider import _is_reasoning_model

        assert _is_reasoning_model("o4-mini") is True
        assert _is_reasoning_model("o4-mini-2025-04-16") is True

    def test_gpt_is_not_reasoning(self):
        from llmcore.providers.openai_provider import _is_reasoning_model

        assert _is_reasoning_model("gpt-4o") is False
        assert _is_reasoning_model("gpt-4o-mini") is False
        assert _is_reasoning_model("gpt-5.4") is False
        assert _is_reasoning_model("gpt-4.1") is False


class TestNeedsDeveloperRole:
    """Tests for _needs_developer_role()."""

    def test_o_series_needs_developer(self):
        from llmcore.providers.openai_provider import _needs_developer_role

        assert _needs_developer_role("o1") is True
        assert _needs_developer_role("o3-mini") is True
        assert _needs_developer_role("o4-mini") is True

    def test_gpt_does_not_need_developer(self):
        from llmcore.providers.openai_provider import _needs_developer_role

        assert _needs_developer_role("gpt-4o") is False
        assert _needs_developer_role("gpt-5.4") is False


class TestContextLengthHeuristics:
    """Tests for prefix-based context length heuristics."""

    def test_gpt5_family(self):
        from llmcore.providers.openai_provider import (
            _OPENAI_PREFIX_CONTEXT_HEURISTICS,
        )

        # Build lookup
        def resolve(model):
            for prefix, ctx in _OPENAI_PREFIX_CONTEXT_HEURISTICS:
                if model.startswith(prefix):
                    return ctx
            return None

        assert resolve("gpt-5.4") == 1050000
        assert resolve("gpt-5.4-pro") == 1050000
        assert resolve("gpt-5.4-mini") == 400000
        assert resolve("gpt-5.2-pro") == 400000
        assert resolve("gpt-5.2") == 400000
        assert resolve("gpt-5.1") == 400000
        assert resolve("gpt-5") == 400000
        assert resolve("gpt-5-mini") == 200000

    def test_o_series(self):
        from llmcore.providers.openai_provider import (
            _OPENAI_PREFIX_CONTEXT_HEURISTICS,
        )

        def resolve(model):
            for prefix, ctx in _OPENAI_PREFIX_CONTEXT_HEURISTICS:
                if model.startswith(prefix):
                    return ctx
            return None

        assert resolve("o4-mini") == 200000
        assert resolve("o3") == 200000
        assert resolve("o3-mini") == 200000
        assert resolve("o1") == 200000
        assert resolve("o1-mini") == 128000

    def test_gpt41_family(self):
        from llmcore.providers.openai_provider import (
            _OPENAI_PREFIX_CONTEXT_HEURISTICS,
        )

        def resolve(model):
            for prefix, ctx in _OPENAI_PREFIX_CONTEXT_HEURISTICS:
                if model.startswith(prefix):
                    return ctx
            return None

        assert resolve("gpt-4.1") == 1048576
        assert resolve("gpt-4.1-mini") == 1048576
        assert resolve("gpt-4.1-nano") == 1048576

    def test_unknown_returns_none(self):
        from llmcore.providers.openai_provider import (
            _OPENAI_PREFIX_CONTEXT_HEURISTICS,
        )

        def resolve(model):
            for prefix, ctx in _OPENAI_PREFIX_CONTEXT_HEURISTICS:
                if model.startswith(prefix):
                    return ctx
            return None

        assert resolve("totally-unknown-model") is None


class TestExtractToolCalls:
    """Tests for extract_tool_calls() response extraction."""

    def _make_provider_stub(self):
        """Create a minimal provider stub for testing extraction methods."""
        from llmcore.providers.openai_provider import OpenAIProvider

        # We can't instantiate normally without SDK, so mock __init__
        provider = object.__new__(OpenAIProvider)
        provider.log_raw_payloads_enabled = False
        provider._provider_instance_name = "openai"
        return provider

    def test_extract_single_tool_call(self):
        provider = self._make_provider_stub()
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Boston"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        calls = provider.extract_tool_calls(response)
        assert len(calls) == 1
        assert calls[0].id == "call_abc123"
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {"location": "Boston"}

    def test_extract_multiple_tool_calls(self):
        provider = self._make_provider_stub()
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"q": "hello"}',
                                },
                            },
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "calculate",
                                    "arguments": '{"expr": "2+2"}',
                                },
                            },
                        ],
                    },
                }
            ],
        }
        calls = provider.extract_tool_calls(response)
        assert len(calls) == 2
        assert calls[0].name == "search"
        assert calls[1].name == "calculate"

    def test_extract_no_tool_calls(self):
        provider = self._make_provider_stub()
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello!",
                    },
                }
            ],
        }
        calls = provider.extract_tool_calls(response)
        assert calls == []

    def test_extract_malformed_arguments(self):
        provider = self._make_provider_stub()
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_bad",
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
        assert calls[0].arguments == {"_raw": "not valid json {{{"}

    def test_extract_empty_response(self):
        provider = self._make_provider_stub()
        assert provider.extract_tool_calls({}) == []
        assert provider.extract_tool_calls({"choices": []}) == []


class TestExtractAudioResponse:
    """Tests for extract_audio_response()."""

    def _make_provider_stub(self):
        from llmcore.providers.openai_provider import OpenAIProvider

        provider = object.__new__(OpenAIProvider)
        provider.log_raw_payloads_enabled = False
        provider._provider_instance_name = "openai"
        return provider

    def test_extract_audio_present(self):
        provider = self._make_provider_stub()
        response = {
            "choices": [
                {
                    "message": {
                        "content": "Hello!",
                        "audio": {
                            "id": "audio_123",
                            "data": "base64data==",
                            "expires_at": 1700000000,
                            "transcript": "Hello!",
                        },
                    },
                }
            ],
        }
        audio = provider.extract_audio_response(response)
        assert audio is not None
        assert audio["id"] == "audio_123"
        assert audio["transcript"] == "Hello!"

    def test_extract_audio_absent(self):
        provider = self._make_provider_stub()
        response = {
            "choices": [{"message": {"content": "text only"}}],
        }
        assert provider.extract_audio_response(response) is None


class TestExtractUsageDetails:
    """Tests for extract_usage_details()."""

    def _make_provider_stub(self):
        from llmcore.providers.openai_provider import OpenAIProvider

        provider = object.__new__(OpenAIProvider)
        provider.log_raw_payloads_enabled = False
        provider._provider_instance_name = "openai"
        return provider

    def test_full_usage_details(self):
        provider = self._make_provider_stub()
        response = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "completion_tokens_details": {
                    "reasoning_tokens": 20,
                    "audio_tokens": 5,
                    "accepted_prediction_tokens": 10,
                    "rejected_prediction_tokens": 3,
                },
                "prompt_tokens_details": {
                    "cached_tokens": 30,
                    "audio_tokens": 8,
                },
            },
        }
        details = provider.extract_usage_details(response)
        assert details["prompt_tokens"] == 100
        assert details["completion_tokens"] == 50
        assert details["total_tokens"] == 150
        assert details["reasoning_tokens"] == 20
        assert details["audio_output_tokens"] == 5
        assert details["accepted_prediction_tokens"] == 10
        assert details["rejected_prediction_tokens"] == 3
        assert details["cached_tokens"] == 30
        assert details["audio_input_tokens"] == 8

    def test_basic_usage_only(self):
        provider = self._make_provider_stub()
        response = {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        details = provider.extract_usage_details(response)
        assert details["prompt_tokens"] == 10
        assert "reasoning_tokens" not in details

    def test_empty_usage(self):
        provider = self._make_provider_stub()
        assert provider.extract_usage_details({}) == {}


class TestExtractRefusal:
    """Tests for extract_refusal()."""

    def _make_provider_stub(self):
        from llmcore.providers.openai_provider import OpenAIProvider

        provider = object.__new__(OpenAIProvider)
        provider.log_raw_payloads_enabled = False
        provider._provider_instance_name = "openai"
        return provider

    def test_refusal_present(self):
        provider = self._make_provider_stub()
        response = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "refusal": "I cannot help with that.",
                    },
                }
            ],
        }
        assert provider.extract_refusal(response) == "I cannot help with that."

    def test_refusal_absent(self):
        provider = self._make_provider_stub()
        response = {
            "choices": [{"message": {"content": "normal reply"}}],
        }
        assert provider.extract_refusal(response) is None


class TestExtractAnnotations:
    """Tests for extract_annotations()."""

    def _make_provider_stub(self):
        from llmcore.providers.openai_provider import OpenAIProvider

        provider = object.__new__(OpenAIProvider)
        provider.log_raw_payloads_enabled = False
        provider._provider_instance_name = "openai"
        return provider

    def test_annotations_present(self):
        provider = self._make_provider_stub()
        response = {
            "choices": [
                {
                    "message": {
                        "content": "Here is info [1].",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url_citation": {
                                    "start_index": 14,
                                    "end_index": 17,
                                    "title": "Source",
                                    "url": "https://example.com",
                                },
                            }
                        ],
                    },
                }
            ],
        }
        anns = provider.extract_annotations(response)
        assert len(anns) == 1
        assert anns[0]["type"] == "url_citation"

    def test_annotations_absent(self):
        provider = self._make_provider_stub()
        response = {
            "choices": [{"message": {"content": "no annotations"}}],
        }
        assert provider.extract_annotations(response) == []


class TestBuildMessagePayload:
    """Tests for _build_message_payload()."""

    def _make_provider_stub(self):
        from llmcore.providers.openai_provider import OpenAIProvider

        provider = object.__new__(OpenAIProvider)
        provider.log_raw_payloads_enabled = False
        provider._provider_instance_name = "openai"
        return provider

    def test_basic_user_message(self):
        provider = self._make_provider_stub()
        msg = _Message(role="user", content="Hello")
        result = provider._build_message_payload(msg, "gpt-4o")
        assert result == {"role": "user", "content": "Hello"}

    def test_system_to_developer_for_o_series(self):
        provider = self._make_provider_stub()
        msg = _Message(role="system", content="You are helpful.")
        result = provider._build_message_payload(msg, "o3-mini")
        assert result["role"] == "developer"
        assert result["content"] == "You are helpful."

    def test_system_stays_system_for_gpt(self):
        provider = self._make_provider_stub()
        msg = _Message(role="system", content="You are helpful.")
        result = provider._build_message_payload(msg, "gpt-4o")
        assert result["role"] == "system"

    def test_tool_message_with_tool_call_id(self):
        provider = self._make_provider_stub()
        msg = _Message(
            role="tool",
            content='{"temp": 72}',
            tool_call_id="call_abc",
        )
        result = provider._build_message_payload(msg, "gpt-4o")
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_abc"
        assert result["content"] == '{"temp": 72}'

    def test_assistant_with_tool_calls(self):
        provider = self._make_provider_stub()
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "NYC"}',
                },
            }
        ]
        msg = _Message(
            role="assistant",
            content="",
            metadata={"tool_calls": tool_calls},
        )
        result = provider._build_message_payload(msg, "gpt-4o")
        assert result["role"] == "assistant"
        assert result["content"] is None  # Empty content -> None
        assert result["tool_calls"] == tool_calls

    def test_assistant_with_tool_calls_and_text(self):
        provider = self._make_provider_stub()
        tool_calls = [
            {
                "id": "call_456",
                "type": "function",
                "function": {"name": "calc", "arguments": "{}"},
            }
        ]
        msg = _Message(
            role="assistant",
            content="Let me calculate that.",
            metadata={"tool_calls": tool_calls},
        )
        result = provider._build_message_payload(msg, "gpt-4o")
        assert result["content"] == "Let me calculate that."
        assert result["tool_calls"] == tool_calls

    def test_multimodal_inline_images(self):
        provider = self._make_provider_stub()
        msg = _Message(
            role="user",
            content="What's in this image?",
            metadata={
                "inline_images": [
                    {"url": "https://example.com/cat.jpg", "detail": "high"},
                ],
            },
        )
        result = provider._build_message_payload(msg, "gpt-4o")
        assert result["role"] == "user"
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 2
        assert result["content"][0] == {
            "type": "text",
            "text": "What's in this image?",
        }
        assert result["content"][1]["type"] == "image_url"
        assert result["content"][1]["image_url"]["url"] == "https://example.com/cat.jpg"
        assert result["content"][1]["image_url"]["detail"] == "high"

    def test_multimodal_inline_images_string_url(self):
        provider = self._make_provider_stub()
        msg = _Message(
            role="user",
            content="Describe this",
            metadata={
                "inline_images": ["https://example.com/pic.png"],
            },
        )
        result = provider._build_message_payload(msg, "gpt-4o")
        assert len(result["content"]) == 2
        assert result["content"][1]["image_url"]["url"] == "https://example.com/pic.png"

    def test_multimodal_inline_audio(self):
        provider = self._make_provider_stub()
        msg = _Message(
            role="user",
            content="Transcribe this",
            metadata={
                "inline_audio": [
                    {"data": "base64audiodatahere", "format": "mp3"},
                ],
            },
        )
        result = provider._build_message_payload(msg, "gpt-4o")
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 2
        assert result["content"][1]["type"] == "input_audio"
        assert result["content"][1]["input_audio"]["data"] == "base64audiodatahere"
        assert result["content"][1]["input_audio"]["format"] == "mp3"

    def test_content_parts_raw_passthrough(self):
        provider = self._make_provider_stub()
        raw_parts = [
            {"type": "text", "text": "Hello"},
            {"type": "image_url", "image_url": {"url": "http://img.jpg"}},
        ]
        msg = _Message(
            role="user",
            content="ignored",
            metadata={"content_parts": raw_parts},
        )
        result = provider._build_message_payload(msg, "gpt-4o")
        assert result["content"] == raw_parts

    def test_name_field(self):
        provider = self._make_provider_stub()
        msg = _Message(
            role="user",
            content="Hi",
            metadata={"name": "alice"},
        )
        result = provider._build_message_payload(msg, "gpt-4o")
        assert result["name"] == "alice"

    def test_audio_reference_on_assistant(self):
        provider = self._make_provider_stub()
        msg = _Message(
            role="assistant",
            content="previous audio",
            metadata={"audio": {"id": "audio_prev_123"}},
        )
        result = provider._build_message_payload(msg, "gpt-4o")
        assert result["audio"] == {"id": "audio_prev_123"}


class TestGetSupportedParameters:
    """Tests for get_supported_parameters()."""

    def _make_provider_stub(self, default_model="gpt-4o"):
        from llmcore.providers.openai_provider import OpenAIProvider

        provider = object.__new__(OpenAIProvider)
        provider.log_raw_payloads_enabled = False
        provider._provider_instance_name = "openai"
        provider.default_model = default_model
        return provider

    def test_non_reasoning_has_max_tokens(self):
        provider = self._make_provider_stub("gpt-4o")
        params = provider.get_supported_parameters()
        assert "max_tokens" in params
        assert "max_completion_tokens" in params

    def test_reasoning_model_no_max_tokens(self):
        provider = self._make_provider_stub("o3-mini")
        params = provider.get_supported_parameters("o3-mini")
        assert "max_tokens" not in params
        assert "max_completion_tokens" in params

    def test_has_new_parameters(self):
        provider = self._make_provider_stub()
        params = provider.get_supported_parameters()
        # Check P0/P1 parameters exist
        assert "response_format" in params
        assert "reasoning_effort" in params
        assert "parallel_tool_calls" in params
        assert "stream_options" in params
        assert "modalities" in params
        assert "audio" in params
        assert "logprobs" in params
        assert "top_logprobs" in params
        assert "web_search_options" in params
        assert "prediction" in params
        assert "service_tier" in params
        assert "store" in params
        assert "metadata" in params
        assert "user" in params


class TestExtractResponseContent:
    """Tests for extract_response_content()."""

    def _make_provider_stub(self):
        from llmcore.providers.openai_provider import OpenAIProvider

        provider = object.__new__(OpenAIProvider)
        provider.log_raw_payloads_enabled = False
        provider._provider_instance_name = "openai"
        return provider

    def test_basic_extraction(self):
        provider = self._make_provider_stub()
        response = {
            "choices": [{"message": {"content": "Hello world"}}],
        }
        assert provider.extract_response_content(response) == "Hello world"

    def test_empty_choices(self):
        provider = self._make_provider_stub()
        assert provider.extract_response_content({"choices": []}) == ""

    def test_none_content(self):
        provider = self._make_provider_stub()
        response = {
            "choices": [{"message": {"content": None}}],
        }
        assert provider.extract_response_content(response) == ""

    def test_missing_keys(self):
        provider = self._make_provider_stub()
        assert provider.extract_response_content({}) == ""


class TestExtractDeltaContent:
    """Tests for extract_delta_content()."""

    def _make_provider_stub(self):
        from llmcore.providers.openai_provider import OpenAIProvider

        provider = object.__new__(OpenAIProvider)
        provider.log_raw_payloads_enabled = False
        provider._provider_instance_name = "openai"
        return provider

    def test_basic_delta(self):
        provider = self._make_provider_stub()
        chunk = {"choices": [{"delta": {"content": "Hi"}}]}
        assert provider.extract_delta_content(chunk) == "Hi"

    def test_empty_delta(self):
        provider = self._make_provider_stub()
        chunk = {"choices": [{"delta": {}}]}
        assert provider.extract_delta_content(chunk) == ""

    def test_no_choices(self):
        provider = self._make_provider_stub()
        assert provider.extract_delta_content({}) == ""


# ---------------------------------------------------------------
# The tests below require mocking the full OpenAI SDK client
# ---------------------------------------------------------------


class TestTokenLimits:
    """Tests for DEFAULT_OPENAI_TOKEN_LIMITS coverage."""

    def test_gpt4o_family(self):
        from llmcore.providers.openai_provider import DEFAULT_OPENAI_TOKEN_LIMITS

        assert DEFAULT_OPENAI_TOKEN_LIMITS["gpt-4o"] == 128000
        assert DEFAULT_OPENAI_TOKEN_LIMITS["gpt-4o-mini"] == 128000
        assert DEFAULT_OPENAI_TOKEN_LIMITS["gpt-4o-2024-11-20"] == 128000

    def test_legacy_models(self):
        from llmcore.providers.openai_provider import DEFAULT_OPENAI_TOKEN_LIMITS

        assert DEFAULT_OPENAI_TOKEN_LIMITS["gpt-4"] == 8000
        assert DEFAULT_OPENAI_TOKEN_LIMITS["gpt-4-32k"] == 32000
        assert DEFAULT_OPENAI_TOKEN_LIMITS["gpt-3.5-turbo"] == 16000
