# tests/providers/test_providers.py
"""Tests for provider-agnostic helpers in llmcore.providers.base."""

from __future__ import annotations

from llmcore.models import Message, Role
from llmcore.providers.base import flatten_tool_messages_for_text_protocol


class TestFlattenToolMessagesForTextProtocol:
    """R-2 fallback: tool-role messages render as plain user-role text."""

    def test_tool_message_becomes_user_text(self):
        msgs = [Message(role=Role.TOOL, content='{"temp": 72}', tool_call_id="call_1")]
        flattened = flatten_tool_messages_for_text_protocol(msgs)

        assert len(flattened) == 1
        assert flattened[0].role == "user"
        assert flattened[0].tool_call_id is None
        assert "call_1" in flattened[0].content
        assert '{"temp": 72}' in flattened[0].content

    def test_tool_message_without_call_id(self):
        msgs = [Message(role=Role.TOOL, content="result")]
        flattened = flatten_tool_messages_for_text_protocol(msgs)

        assert flattened[0].role == "user"
        assert "unknown" in flattened[0].content

    def test_non_tool_messages_pass_through_unchanged(self):
        user = Message(role=Role.USER, content="Hi")
        assistant = Message(
            role=Role.ASSISTANT,
            content="Checking.",
            tool_calls=[{"id": "c1", "type": "function", "function": {"name": "f"}}],
        )
        flattened = flatten_tool_messages_for_text_protocol([user, assistant])

        assert flattened[0] is user
        assert flattened[1] is assistant
        assert flattened[1].tool_calls is not None

    def test_mixed_sequence_preserves_order(self):
        msgs = [
            Message(role=Role.USER, content="Do it"),
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[{"id": "c1", "type": "function", "function": {"name": "f"}}],
            ),
            Message(role=Role.TOOL, content="ok", tool_call_id="c1"),
        ]
        flattened = flatten_tool_messages_for_text_protocol(msgs)

        assert [m.role for m in flattened] == ["user", "assistant", "user"]
        assert flattened[2].content.startswith("[Tool result for call c1]")

    def test_original_messages_not_mutated(self):
        tool_msg = Message(role=Role.TOOL, content="ok", tool_call_id="c1")
        flatten_tool_messages_for_text_protocol([tool_msg])

        assert tool_msg.role == "tool"
        assert tool_msg.tool_call_id == "c1"
