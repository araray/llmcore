"""Tests for context failure diagnostics."""

from __future__ import annotations

from llmcore.agents.observability import (
    ContextFailureDetectorConfig,
    ContextFailureType,
    ErrorEventType,
    EventSeverity,
    detect_context_failures,
)


def _types(diagnostics):
    return {item.failure_type for item in diagnostics}


def test_detects_context_overflow_and_converts_to_event() -> None:
    diagnostics = detect_context_failures(
        estimated_tokens=1250,
        max_context_tokens=1000,
        current_iteration=4,
        phase="think",
    )

    assert _types(diagnostics) == {ContextFailureType.CONTEXT_OVERFLOW}
    diagnostic = diagnostics[0]
    assert diagnostic.severity == EventSeverity.ERROR
    assert diagnostic.details["overflow_tokens"] == 250

    event = diagnostic.to_event("session-1", execution_id="exec-1", correlation_id="corr-1")
    assert event.event_type == ErrorEventType.CONTEXT_FAILURE
    assert event.error_type == "context_overflow"
    assert event.iteration == 4
    assert event.phase == "think"
    assert event.correlation_id == "corr-1"
    assert "context_failure" in event.tags


def test_detects_orphan_tool_result_from_message_dicts() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "I will call a tool",
            "metadata": {
                "tool_calls": [{"id": "call-ok", "function": {"name": "read_file"}}]
            },
        },
        {"role": "tool", "content": "ok", "tool_call_id": "call-ok"},
        {"role": "tool", "content": "orphan", "tool_call_id": "call-missing"},
    ]

    diagnostics = detect_context_failures(messages=messages, current_iteration=2)

    assert _types(diagnostics) == {ContextFailureType.ORPHAN_TOOL_RESULT}
    orphan = diagnostics[0]
    assert orphan.details["orphan_count"] == 1
    assert orphan.details["orphan_results"][0]["tool_call_id"] == "call-missing"
    assert orphan.details["known_tool_call_ids"] == ["call-ok"]


def test_valid_tool_call_result_pair_does_not_report_orphan() -> None:
    diagnostics = detect_context_failures(
        messages=[
            {
                "role": "assistant",
                "tool_calls": [{"id": "call-1", "function": {"name": "search"}}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "result"},
        ]
    )

    assert diagnostics == []


def test_detects_summary_thrash_in_recent_window() -> None:
    diagnostics = detect_context_failures(
        compression_events=[
            {"iteration": 6, "summary": "old"},
            {"iteration": 10, "summary": "summary a"},
            {"iteration": 11, "summary": "summary b"},
            {"iteration": 12, "summary": "summary c"},
        ],
        current_iteration=12,
        config=ContextFailureDetectorConfig(
            summary_thrash_threshold=3,
            summary_thrash_window=3,
        ),
    )

    assert _types(diagnostics) == {ContextFailureType.SUMMARY_THRASH}
    assert diagnostics[0].details["recent_compression_count"] == 3
    assert diagnostics[0].details["unique_summary_count"] == 3


def test_detects_stale_context() -> None:
    diagnostics = detect_context_failures(
        current_iteration=9,
        context_updated_iteration=3,
        config=ContextFailureDetectorConfig(stale_context_iterations=5),
    )

    assert _types(diagnostics) == {ContextFailureType.STALE_CONTEXT}
    assert diagnostics[0].details["age_iterations"] == 6


def test_detects_repeated_tool_failure_streak() -> None:
    diagnostics = detect_context_failures(
        tool_failures=[
            {"tool_name": "read_file", "success": True},
            {"tool_name": "run_command", "success": False, "error": "exit 1"},
            {"tool_name": "run_command", "success": False, "error": "exit 1"},
            {"tool_name": "run_command", "success": False, "error": "exit 1"},
        ],
        config=ContextFailureDetectorConfig(repeated_tool_failure_threshold=3),
    )

    assert _types(diagnostics) == {ContextFailureType.REPEATED_TOOL_FAILURE}
    assert diagnostics[0].details["tool_name"] == "run_command"
    assert diagnostics[0].details["consecutive_failures"] == 3


def test_success_breaks_repeated_tool_failure_streak() -> None:
    diagnostics = detect_context_failures(
        tool_failures=[
            {"tool_name": "run_command", "success": False},
            {"tool_name": "run_command", "success": False},
            {"tool_name": "run_command", "success": True},
        ],
        config=ContextFailureDetectorConfig(repeated_tool_failure_threshold=2),
    )

    assert diagnostics == []
