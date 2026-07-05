# tests/providers/test_instrumented_chat_completion.py
"""Tests for BaseProvider._instrumented_chat_completion.

Regression coverage for the metrics path that previously called the
excised ``api_server.metrics.record_llm_request`` helper (F821) and
logged a swallowed NameError on every instrumented completion.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest

from llmcore.models import Message, ModelDetails, Role, Tool
from llmcore.providers.base import BaseProvider


class StubProvider(BaseProvider):
    """Minimal concrete provider for exercising base-class helpers."""

    def __init__(self, config: dict[str, Any] | None = None, log_raw_payloads: bool = False):
        super().__init__(config or {}, log_raw_payloads=log_raw_payloads)
        self.fail_with: Exception | None = None

    def get_name(self) -> str:
        return "stub"

    async def get_models_details(self) -> list[ModelDetails]:
        return []

    def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
        return {}

    def get_max_context_length(self, model: str | None = None) -> int:
        return 4096

    async def chat_completion(
        self,
        context: list[Message],
        model: str | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        if self.fail_with is not None:
            raise self.fail_with
        return {
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5},
        }

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        return len(text.split())

    async def count_message_tokens(self, messages: list[Message], model: str | None = None) -> int:
        return len(messages)

    def extract_response_content(self, response: dict[str, Any]) -> str:
        return ""

    def extract_delta_content(self, chunk: dict[str, Any]) -> str:
        return ""


@pytest.fixture
def context() -> list[Message]:
    return [Message(role=Role.USER, content="hi", session_id="session-1")]


class TestInstrumentedChatCompletion:
    async def test_success_returns_underlying_result(
        self, context: list[Message], caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = StubProvider()

        with caplog.at_level("DEBUG", logger="llmcore.providers.base"):
            result = await provider._instrumented_chat_completion(context=context)

        assert isinstance(result, dict)
        assert result["usage"]["completion_tokens"] == 5
        # The old dead metrics block logged this on every instrumented call.
        assert not any(
            "Failed to record LLM metrics" in rec.message for rec in caplog.records
        )

    async def test_error_propagates(self, context: list[Message]) -> None:
        provider = StubProvider()
        provider.fail_with = RuntimeError("api down")

        with pytest.raises(RuntimeError, match="api down"):
            await provider._instrumented_chat_completion(context=context)
