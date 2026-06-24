from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from llmcore.agents.observability.events import LifecycleEvent as AgentLifecycleEvent
from llmcore.observability.events import Event, EventCategory, Severity
from llmcore.observability.federation import (
    FederatedEventSink,
    SourceSystem,
    federate_llmcore_event,
    federate_metastore_event,
    federate_semantiscan_event,
    federate_wairu_ipc_message,
    load_federated_agent_events_from_file,
    load_federated_llmcore_events_from_file,
    trace_ids_from_context,
)

TRACE_CONTEXT = {
    "traceparent": "00-0123456789abcdef0123456789abcdef-fedcba9876543210-01",
    "tracestate": "vendor=value",
}


def test_trace_ids_from_context_extracts_w3c_traceparent() -> None:
    assert trace_ids_from_context(TRACE_CONTEXT) == {
        "trace_id": "0123456789abcdef0123456789abcdef",
        "span_id": "fedcba9876543210",
    }


def test_federate_llmcore_event_normalizes_classic_observability_event() -> None:
    event = Event(
        category=EventCategory.COGNITIVE,
        event_type="phase_completed",
        severity=Severity.WARNING,
        execution_id="exec-1",
        session_id="sess-1",
        source="agents.cognitive",
        tags=["darwin"],
        data={"phase": "think", "duration_ms": 12.5},
    )

    federated = federate_llmcore_event(
        event,
        correlation_id="corr-1",
        trace_context=TRACE_CONTEXT,
    )

    assert federated.event_id == event.id
    assert federated.source_system == SourceSystem.LLMCORE
    assert federated.source_component == "agents.cognitive"
    assert federated.category == "cognitive"
    assert federated.event_type == "phase_completed"
    assert federated.severity == "warning"
    assert federated.execution_id == "exec-1"
    assert federated.session_id == "sess-1"
    assert federated.correlation_id == "corr-1"
    assert federated.trace_id == "0123456789abcdef0123456789abcdef"
    assert federated.span_id == "fedcba9876543210"
    assert federated.tags == ["darwin"]
    assert federated.phase == "think"
    assert federated.duration_ms == 12.5
    assert federated.payload == {"phase": "think", "duration_ms": 12.5}
    assert federated.to_dict()["source_system"] == "llmcore"


def test_federate_llmcore_event_normalizes_agent_event_shape() -> None:
    event = {
        "event_id": "evt-agent-1",
        "timestamp": "2026-06-23T12:00:00+00:00",
        "session_id": "sess-2",
        "execution_id": "exec-2",
        "category": "activity",
        "event_type": "activity_completed",
        "severity": "info",
        "phase": "act",
        "iteration": 3,
        "parent_event_id": "evt-parent",
        "correlation_id": "corr-2",
        "duration_ms": 44,
        "tags": ["tool"],
        "data": {"tool": "read_file"},
    }

    federated = federate_llmcore_event(event)

    assert federated.event_id == "evt-agent-1"
    assert federated.timestamp == datetime(2026, 6, 23, 12, 0, tzinfo=UTC)
    assert federated.category == "activity"
    assert federated.phase == "act"
    assert federated.iteration == 3
    assert federated.parent_event_id == "evt-parent"
    assert federated.correlation_id == "corr-2"
    assert federated.duration_ms == 44
    assert federated.payload == {"tool": "read_file"}


def test_federate_llmcore_event_preserves_agent_specific_payload_fields() -> None:
    event = AgentLifecycleEvent(
        session_id="sess-agent",
        execution_id="exec-agent",
        event_type="agent_started",
        goal="Scan the repository",
        goal_complexity="complex",
        recommended_strategy="plan_execute",
    )

    federated = federate_llmcore_event(event, source_component="agents.observability")

    assert federated.source_system == SourceSystem.LLMCORE
    assert federated.source_component == "agents.observability"
    assert federated.category == "lifecycle"
    assert federated.execution_id == "exec-agent"
    assert federated.payload["goal"] == "Scan the repository"
    assert federated.payload["goal_complexity"] == "complex"
    assert federated.payload["recommended_strategy"] == "plan_execute"


def test_federated_event_sink_forwards_normalized_agent_events() -> None:
    captured = []
    sink = FederatedEventSink(captured.append, correlation_id="corr-sink")
    event = AgentLifecycleEvent(
        session_id="sess-sink",
        execution_id="exec-sink",
        event_type="agent_started",
        goal="Forward this event",
    )

    async def write_event() -> None:
        await sink.write(event)
        await sink.flush()
        await sink.close()

    asyncio.run(write_event())

    assert len(captured) == 1
    assert captured[0].correlation_id == "corr-sink"
    assert captured[0].source_component == "agents.observability"
    assert captured[0].payload["goal"] == "Forward this event"


def test_load_federated_llmcore_events_from_file(tmp_path) -> None:
    log_path = tmp_path / "classic-events.jsonl"
    event = Event(
        category=EventCategory.LIFECYCLE,
        event_type="started",
        execution_id="exec-file",
        source="classic.logger",
        data={"step": "boot"},
    )
    log_path.write_text(event.to_jsonl() + "\n", encoding="utf-8")

    federated = load_federated_llmcore_events_from_file(
        log_path,
        execution_id="exec-file",
        correlation_id="corr-file",
    )

    assert len(federated) == 1
    assert federated[0].event_id == event.id
    assert federated[0].source_component == "classic.logger"
    assert federated[0].correlation_id == "corr-file"
    assert federated[0].payload == {"step": "boot"}


def test_load_federated_agent_events_from_file(tmp_path) -> None:
    log_path = tmp_path / "agent-events.jsonl"
    event = AgentLifecycleEvent(
        session_id="sess-file",
        execution_id="exec-file",
        event_type="agent_started",
        goal="Load from replay",
    )
    log_path.write_text(event.model_dump_json() + "\n", encoding="utf-8")

    federated = load_federated_agent_events_from_file(
        log_path,
        session_id="sess-file",
    )

    assert len(federated) == 1
    assert federated[0].event_id == event.event_id
    assert federated[0].source_component == "agents.observability"
    assert federated[0].payload["goal"] == "Load from replay"


def test_federate_semantiscan_event_normalizes_event_bus_shape() -> None:
    event = {
        "event_type": "retrieval_complete",
        "timestamp": "2026-06-23T12:01:00+00:00",
        "data": {
            "collection_id": "docs",
            "result_count": 4,
            "trace_context": TRACE_CONTEXT,
        },
        "source": "api.retrieve",
        "duration_ms": 8.25,
    }

    federated = federate_semantiscan_event(event, correlation_id="corr-3")

    assert federated.source_system == SourceSystem.SEMANTISCAN
    assert federated.source_component == "api.retrieve"
    assert federated.event_type == "retrieval_complete"
    assert federated.session_id == "docs"
    assert federated.duration_ms == 8.25
    assert federated.correlation_id == "corr-3"
    assert federated.trace_id == "0123456789abcdef0123456789abcdef"
    assert federated.payload["result_count"] == 4


def test_federate_metastore_event_uses_run_id_and_payload() -> None:
    event = {
        "event_id": 42,
        "run_id": "run-1",
        "event_type": "ENTITY_CREATED",
        "payload": {"entity_id": "ent-1"},
        "created_at": "2026-06-23T12:02:00+00:00",
    }

    federated = federate_metastore_event(event)

    assert federated.event_id == "42"
    assert federated.source_system == SourceSystem.SEMANTISCAN
    assert federated.source_component == "metastore"
    assert federated.category == "metastore"
    assert federated.execution_id == "run-1"
    assert federated.payload == {"entity_id": "ent-1"}


def test_federate_wairu_ipc_message_preserves_trace_context_and_payload() -> None:
    message = {
        "type": "request",
        "id": "ipc-1",
        "command": "status",
        "params": {"verbose": True},
        "trace_context": TRACE_CONTEXT,
    }

    federated = federate_wairu_ipc_message(message)

    assert federated.event_id == "ipc-1"
    assert federated.source_system == SourceSystem.WAIRU
    assert federated.source_component == "daemon.ipc"
    assert federated.category == "ipc"
    assert federated.event_type == "ipc.request.status"
    assert federated.trace_id == "0123456789abcdef0123456789abcdef"
    assert federated.span_id == "fedcba9876543210"
    assert federated.payload == {
        "command": "status",
        "params": {"verbose": True},
    }
