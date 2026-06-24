"""Tests for memory context payload budgeting and pruning."""

from __future__ import annotations

from typing import Any

import pytest

from llmcore.memory.context_builder import build_context_payload
from llmcore.models import ChatSession, Message, Role


class FixedTokenProvider:
    async def count_message_tokens(
        self,
        messages: list[Message],
        model: str | None = None,
    ) -> int:
        return sum(message.tokens or 10 for message in messages)


@pytest.mark.asyncio
async def test_tool_schema_tokens_reduce_history_budget() -> None:
    session = ChatSession(
        id="session-budget",
        messages=[
            Message(role=Role.USER, content="old user", tokens=20),
            Message(role=Role.ASSISTANT, content="old assistant", tokens=20),
            Message(role=Role.USER, content="final query", tokens=10),
        ],
    )
    config: dict[str, Any] = {
        "reserved_response_tokens": 10,
        "tool_schema_tokens": 40,
        "inclusion_priority_order": ["history_chat", "final_user_query"],
        "truncation_priority_order": ["history_chat"],
    }

    details = await build_context_payload(
        session=session,
        provider=FixedTokenProvider(),  # type: ignore[arg-type]
        target_model="fake",
        max_model_tokens=80,
        config=config,
        final_user_query_content="final query",
    )

    assert details.available_context_tokens == 30
    assert details.tool_schema_tokens == 40
    assert details.final_token_count <= details.available_context_tokens
    assert [message.content for message in details.prepared_messages] == [
        "old assistant",
        "final query",
    ]


@pytest.mark.asyncio
async def test_context_builder_removes_orphan_tool_results_after_pruning() -> None:
    session = ChatSession(
        id="session-tools",
        messages=[
            Message(
                role=Role.ASSISTANT,
                content="",
                tokens=10,
                metadata={"tool_calls": [{"id": "call-1", "function": {"name": "lookup"}}]},
            ),
            Message(role=Role.TOOL, content="ok", tool_call_id="call-1", tokens=10),
            Message(role=Role.TOOL, content="orphan", tool_call_id="call-missing", tokens=10),
            Message(role=Role.USER, content="final query", tokens=10),
        ],
    )
    config: dict[str, Any] = {
        "reserved_response_tokens": 10,
        "inclusion_priority_order": ["history_chat", "final_user_query"],
        "truncation_priority_order": ["history_chat"],
    }

    details = await build_context_payload(
        session=session,
        provider=FixedTokenProvider(),  # type: ignore[arg-type]
        target_model="fake",
        max_model_tokens=100,
        config=config,
        final_user_query_content="final query",
    )

    tool_result_ids = [
        message.tool_call_id
        for message in details.prepared_messages
        if message.role == "tool"
    ]

    assert tool_result_ids == ["call-1"]
    assert any("orphan tool result" in item for item in details.truncation_actions_taken["details"])
