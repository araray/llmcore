"""Tests for provider-safe tool-call history sanitization."""

from __future__ import annotations

from llmcore.context import sanitize_tool_message_pairs
from llmcore.models import Message, Role


def test_sanitizer_drops_orphan_tool_result_dicts() -> None:
    messages = [
        {
            "role": "assistant",
            "metadata": {
                "tool_calls": [{"id": "call-ok", "function": {"name": "read_file"}}]
            },
        },
        {"role": "tool", "tool_call_id": "call-ok", "content": "ok"},
        {"role": "tool", "tool_call_id": "call-missing", "content": "orphan"},
    ]

    sanitized = sanitize_tool_message_pairs(messages)

    assert len(sanitized) == 2
    assert sanitized[-1]["tool_call_id"] == "call-ok"


def test_sanitizer_strips_dangling_assistant_tool_calls() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "I will inspect that.",
            "tool_calls": [{"id": "call-missing", "function": {"name": "inspect"}}],
        },
        {"role": "user", "content": "continue"},
    ]

    sanitized = sanitize_tool_message_pairs(messages)

    assert sanitized[0]["content"] == "I will inspect that."
    assert "tool_calls" not in sanitized[0]
    assert len(sanitized) == 2


def test_sanitizer_filters_unmatched_calls_from_multi_call_message() -> None:
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call-kept", "function": {"name": "search"}},
                {"id": "call-dropped", "function": {"name": "read_file"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call-kept", "content": "result"},
    ]

    sanitized = sanitize_tool_message_pairs(messages)

    assert sanitized[0]["tool_calls"] == [
        {"id": "call-kept", "function": {"name": "search"}}
    ]
    assert sanitized[1]["tool_call_id"] == "call-kept"


def test_sanitizer_handles_llmcore_message_models() -> None:
    assistant = Message(
        role=Role.ASSISTANT,
        content="",
        metadata={"tool_calls": [{"id": "call-1", "function": {"name": "lookup"}}]},
    )
    tool_result = Message(role=Role.TOOL, content="ok", tool_call_id="call-1")
    orphan = Message(role=Role.TOOL, content="orphan", tool_call_id="call-missing")

    sanitized = sanitize_tool_message_pairs([assistant, tool_result, orphan])

    assert len(sanitized) == 2
    assert isinstance(sanitized[0], Message)
    assert sanitized[0].metadata["tool_calls"][0]["id"] == "call-1"
    assert sanitized[1].tool_call_id == "call-1"
