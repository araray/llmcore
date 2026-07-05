# tests/agents/test_cognitive_cycle_act.py
"""Tests for the ACT step of the cognitive cycle.

Regression coverage for the success path, which previously called the
excised ``api_server.metrics.record_tool_execution`` helper (F821) and
silently swallowed the resulting NameError on every tool execution.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from llmcore.agents.cognitive_cycle import act_step
from llmcore.models import ToolCall, ToolResult


@pytest.fixture
def tool_call() -> ToolCall:
    return ToolCall(id="call-1", name="calculator", arguments={"expression": "6*7"})


class TestActStep:
    async def test_success_path_returns_tool_result(
        self, tool_call: ToolCall, caplog: pytest.LogCaptureFixture
    ) -> None:
        tool_manager = MagicMock()
        tool_manager.execute_tool = AsyncMock(
            return_value=ToolResult(tool_call_id="call-1", content="42")
        )

        with caplog.at_level("DEBUG", logger="llmcore.agents.cognitive_cycle"):
            result = await act_step(
                tool_call=tool_call,
                session_id="session-1",
                task=MagicMock(),
                tool_manager=tool_manager,
                tracer=None,
                db_session=None,
            )

        assert result.content == "42"
        assert result.tool_call_id == "call-1"
        tool_manager.execute_tool.assert_awaited_once_with(tool_call, "session-1")
        # The old dead metrics block logged this on every successful call.
        assert not any(
            "Failed to record tool execution metrics" in rec.message for rec in caplog.records
        )

    async def test_tool_error_is_wrapped_in_error_result(self, tool_call: ToolCall) -> None:
        tool_manager = MagicMock()
        tool_manager.execute_tool = AsyncMock(side_effect=RuntimeError("tool exploded"))

        result = await act_step(
            tool_call=tool_call,
            session_id="session-1",
            task=MagicMock(),
            tool_manager=tool_manager,
            tracer=None,
            db_session=None,
        )

        assert result.tool_call_id == "call-1"
        assert result.content.startswith("ERROR:")
        assert "tool exploded" in result.content
