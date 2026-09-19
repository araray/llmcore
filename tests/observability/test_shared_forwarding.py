# tests/observability/test_shared_forwarding.py
"""
Smoke tests for UnifiedEvent forwarding into the shared event spine (S-3).

Covers both emission points:
- llmcore.observability.events.ObservabilityLogger (sync, buffered logger)
- llmcore.agents.observability.logger.EventLogger (async agent logger)

Forwarding must be additive: with no shared_events sinks registered the
existing behavior is unchanged, and a faulty sink never breaks logging.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llmcore import shared_events
from llmcore.agents.observability.logger import EventLogger, InMemorySink
from llmcore.observability.events import (
    EventBufferConfig,
    EventCategory,
    ObservabilityConfig,
    ObservabilityLogger,
)
from llmcore.shared_events import (
    UnifiedEvent,
    correlation_context,
    register_sink,
    unregister_sink,
)


@pytest.fixture(autouse=True)
def _clean_spine():
    """Run each test with an empty sink registry, restoring it afterwards."""
    saved = list(shared_events._sinks)
    for sink in saved:
        unregister_sink(sink)
    yield
    for sink in list(shared_events._sinks):
        unregister_sink(sink)
    for sink in saved:
        register_sink(sink)


@pytest.fixture
def obs_logger(tmp_path: Path) -> ObservabilityLogger:
    """Unbuffered ObservabilityLogger writing under tmp_path."""
    config = ObservabilityConfig(
        log_path=str(tmp_path / "events.jsonl"),
        buffer=EventBufferConfig(enabled=False),
    )
    return ObservabilityLogger(config)


# =============================================================================
# ObservabilityLogger (llmcore.observability.events)
# =============================================================================


class TestObservabilityLoggerForwarding:
    def test_no_sinks_behavior_unchanged(self, obs_logger: ObservabilityLogger) -> None:
        event = obs_logger.log_event(
            category=EventCategory.COGNITIVE,
            event_type="phase_completed",
            data={"phase": "THINK"},
        )
        assert event is not None
        assert obs_logger.stats["total_events"] == 1

    def test_forwards_unified_event(self, obs_logger: ObservabilityLogger) -> None:
        received: list[UnifiedEvent] = []
        register_sink(received.append)

        with correlation_context("corr-fwd"):
            event = obs_logger.log_event(
                category=EventCategory.COGNITIVE,
                event_type="phase_completed",
                data={"phase": "THINK", "duration_ms": 12},
            )

        assert event is not None
        assert len(received) == 1
        unified = received[0]
        assert unified.source == "llmcore"
        assert unified.type == "cognitive.phase_completed"
        assert unified.correlation_id == "corr-fwd"
        assert unified.payload["id"] == event.id
        assert unified.payload["data"] == {"phase": "THINK", "duration_ms": 12}

    def test_filtered_events_not_forwarded(self, obs_logger: ObservabilityLogger) -> None:
        received: list[UnifiedEvent] = []
        register_sink(received.append)

        # min_severity defaults to "info"; debug events are filtered out.
        result = obs_logger.log_event(
            category=EventCategory.LLM,
            event_type="request",
            severity="debug",
        )
        assert result is None
        assert received == []

    def test_faulty_sink_does_not_break_logging(self, obs_logger: ObservabilityLogger) -> None:
        def bad_sink(event: UnifiedEvent) -> None:
            raise RuntimeError("sink exploded")

        register_sink(bad_sink)
        event = obs_logger.log_event(
            category=EventCategory.ACTIVITY,
            event_type="tool_executed",
            data={"tool": "shell"},
        )
        assert event is not None
        assert obs_logger.stats["errors"] == 0


# =============================================================================
# EventLogger (llmcore.agents.observability.logger)
# =============================================================================


class TestAgentEventLoggerForwarding:
    async def test_no_sinks_behavior_unchanged(self) -> None:
        memory = InMemorySink()
        agent_logger = EventLogger(session_id="sess-1", sinks=[memory])
        event = await agent_logger.log_lifecycle_start(goal="test goal")
        assert event.event_id
        assert len(memory.get_events()) == 1

    async def test_forwards_unified_event_with_event_correlation(self) -> None:
        received: list[UnifiedEvent] = []
        register_sink(received.append)

        agent_logger = EventLogger(session_id="sess-1")
        agent_logger.set_correlation_id("corr-agent")
        event = await agent_logger.log_lifecycle_start(goal="test goal")

        assert len(received) == 1
        unified = received[0]
        assert unified.source == "llmcore"
        assert unified.type == "lifecycle.agent_started"
        assert unified.correlation_id == "corr-agent"
        assert unified.payload["event_id"] == event.event_id
        assert unified.payload["session_id"] == "sess-1"

    async def test_falls_back_to_ambient_correlation(self) -> None:
        received: list[UnifiedEvent] = []
        register_sink(received.append)

        agent_logger = EventLogger(session_id="sess-2")
        with correlation_context("corr-ambient"):
            await agent_logger.log_activity("shell", {"cmd": "ls"})

        assert len(received) == 1
        assert received[0].correlation_id == "corr-ambient"
        assert received[0].type.startswith("activity.")

    async def test_faulty_sink_does_not_break_logging(self) -> None:
        def bad_sink(event: UnifiedEvent) -> None:
            raise RuntimeError("sink exploded")

        register_sink(bad_sink)
        memory = InMemorySink()
        agent_logger = EventLogger(session_id="sess-3", sinks=[memory])
        await agent_logger.log_lifecycle_start(goal="still works")
        assert len(memory.get_events()) == 1
