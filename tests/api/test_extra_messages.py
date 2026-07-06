# tests/api/test_extra_messages.py
"""Tests for the additive ``extra_messages`` parameter on ``LLMCore.chat`` (R-2).

``extra_messages`` lets a tool-calling caller (e.g. wairu) submit an
assistant message carrying ``tool_calls`` plus the matching ``role="tool"``
result messages for the feedback turn, instead of flattening results into
user text. This suite runs fully offline against a fake provider that
records the context payload it receives.
"""

from __future__ import annotations

from typing import Any

import pytest

from llmcore import LLMCore
from llmcore.models import Message, Role
from llmcore.providers.base import BaseProvider

_FAKE_REPLY = "It is 72F in NYC."

_TOOL_CALLS = [
    {
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
    }
]


class RecordingProvider(BaseProvider):
    """Offline provider that records every context payload it receives."""

    def __init__(self, config: dict[str, Any] | None = None, log_raw_payloads: bool = False):
        super().__init__(config or {}, log_raw_payloads)
        self.default_model = "fake-model-1"
        self.received_contexts: list[list[Message]] = []

    def get_name(self) -> str:
        return "fake"

    async def get_models_details(self) -> list[Any]:
        return []

    def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
        return {}

    def get_max_context_length(self, model: str | None = None) -> int:
        return 8192

    async def chat_completion(
        self,
        context: Any,
        model: str | None = None,
        stream: bool = False,
        tools: Any = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.received_contexts.append(list(context))
        return {"content": _FAKE_REPLY}

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        return max(1, len((text or "").split()))

    async def count_message_tokens(
        self, messages: list[Message], model: str | None = None
    ) -> int:
        total = 0
        for m in messages:
            total += len((getattr(m, "content", "") or "").split())
        return max(1, total)

    def extract_response_content(self, response: dict[str, Any]) -> str:
        return response["content"]

    def extract_delta_content(self, chunk: dict[str, Any]) -> str:
        return chunk.get("content", "")


@pytest.fixture
async def fake_llm(tmp_path):
    """An :class:`LLMCore` wired to an injected :class:`RecordingProvider`."""
    llm = await LLMCore.create(
        config_overrides={
            "storage": {
                "vector": {"type": ""},
                "session": {"type": "json", "path": str(tmp_path / "sessions")},
            }
        }
    )
    fake = RecordingProvider()
    llm._provider_manager._providers["fake"] = fake
    llm._provider_manager._default_provider_name = "fake"
    try:
        yield llm, fake
    finally:
        await llm.close()


def _extra_feedback_messages() -> list[Message]:
    return [
        Message(role=Role.ASSISTANT, content="", tool_calls=_TOOL_CALLS),
        Message(role=Role.TOOL, content='{"temp": 72}', tool_call_id="call_1"),
    ]


class TestExtraMessagesPlumbing:
    async def test_extra_messages_reach_provider_in_order(self, fake_llm):
        llm, fake = fake_llm
        reply = await llm.chat(
            message="Answer using the tool results above.",
            session_id="s-tools",
            extra_messages=_extra_feedback_messages(),
            enable_rag=False,
        )
        assert reply == _FAKE_REPLY

        payload = fake.received_contexts[-1]
        roles = [str(getattr(m.role, "value", m.role)) for m in payload]
        # assistant(tool_calls) then tool result, user query re-anchored last
        assert roles[-3:] == ["assistant", "tool", "user"]
        assert payload[-3].tool_calls == _TOOL_CALLS
        assert payload[-2].tool_call_id == "call_1"
        assert payload[-2].content == '{"temp": 72}'
        assert payload[-1].content == "Answer using the tool results above."

    async def test_extra_messages_get_session_id_stamped(self, fake_llm):
        llm, _ = fake_llm
        await llm.chat(
            message="continue",
            session_id="s-stamp",
            extra_messages=_extra_feedback_messages(),
            enable_rag=False,
        )
        session = await llm.get_session("s-stamp")
        tool_msgs = [m for m in session.messages if m.role == "tool"]
        assert tool_msgs and all(m.session_id == "s-stamp" for m in tool_msgs)

    async def test_extra_messages_persist_across_turns(self, fake_llm):
        """The stored extra messages survive a save/load cycle (JSON storage)."""
        llm, fake = fake_llm
        await llm.chat(
            message="continue",
            session_id="s-persist",
            extra_messages=_extra_feedback_messages(),
            save_session=True,
            enable_rag=False,
        )
        # Second turn reloads the session from storage; the tool-protocol
        # messages must still be in the replayed history.
        await llm.chat(
            message="and in SF?",
            session_id="s-persist",
            save_session=True,
            enable_rag=False,
        )
        payload = fake.received_contexts[-1]
        tool_msgs = [m for m in payload if str(getattr(m.role, "value", m.role)) == "tool"]
        assert tool_msgs and tool_msgs[0].tool_call_id == "call_1"
        assistants = [m for m in payload if m.tool_calls]
        assert assistants and assistants[0].tool_calls == _TOOL_CALLS

    async def test_chat_without_extra_messages_unchanged(self, fake_llm):
        """Omitting the parameter preserves the existing behavior."""
        llm, fake = fake_llm
        reply = await llm.chat(message="Hello", session_id="s-plain", enable_rag=False)
        assert reply == _FAKE_REPLY
        payload = fake.received_contexts[-1]
        assert all(str(getattr(m.role, "value", m.role)) != "tool" for m in payload)

    async def test_chat_with_usage_forwards_extra_messages(self, fake_llm):
        llm, fake = fake_llm
        text, usage = await llm.chat_with_usage(
            message="continue",
            session_id="s-usage",
            extra_messages=_extra_feedback_messages(),
            save_session=False,
            enable_rag=False,
        )
        assert text == _FAKE_REPLY
        payload = fake.received_contexts[-1]
        assert any(str(getattr(m.role, "value", m.role)) == "tool" for m in payload)
