"""Federated event envelope and adapters for the Wairu ecosystem.

This module is the first P-01 federation slice. It does not replace the
existing local event systems in llmcore, semantiscan, wairu, or grimoire.
Instead, it provides a small stable envelope and adapters that can map those
local event shapes into one correlation-friendly representation.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SourceSystem(str, Enum):
    """Known source systems for federated ecosystem events."""

    LLMCORE = "llmcore"
    SEMANTISCAN = "semantiscan"
    WAIRU = "wairu"
    GRIMOIRE = "grimoire"
    UNKNOWN = "unknown"


class EcosystemEvent(BaseModel):
    """A normalized event envelope shared across the Wairu ecosystem."""

    model_config = ConfigDict(use_enum_values=True)

    event_id: str = Field(..., description="Stable event identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_system: SourceSystem | str = Field(..., description="Emitting system")
    source_component: str | None = Field(default=None, description="Emitting component")
    category: str = Field(default="custom", description="Coarse event category")
    event_type: str = Field(..., description="Specific event type")
    severity: str = Field(default="info", description="Event severity")

    session_id: str | None = None
    execution_id: str | None = None
    correlation_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    parent_event_id: str | None = None

    iteration: int | None = None
    phase: str | None = None
    duration_ms: float | None = None
    tags: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary, omitting null fields."""

        return self.model_dump(mode="json", exclude_none=True)


def federate_llmcore_event(
    event: Any,
    *,
    correlation_id: str | None = None,
    trace_context: dict[str, str] | None = None,
    source_component: str | None = None,
) -> EcosystemEvent:
    """Normalize an llmcore observability or agent event."""

    data = _event_mapping(event)
    payload = _payload_from(data)
    trace_ids = trace_ids_from_context(trace_context)

    return EcosystemEvent(
        event_id=_string_value(_first_present(data, "event_id", "id")) or _generated_event_id(),
        timestamp=_coerce_timestamp(_first_present(data, "timestamp")),
        source_system=SourceSystem.LLMCORE,
        source_component=source_component
        or _string_value(_first_present(data, "source", "source_component")),
        category=_string_value(_first_present(data, "category")) or "custom",
        event_type=_string_value(_first_present(data, "event_type")) or "event",
        severity=_string_value(_first_present(data, "severity")) or "info",
        session_id=_string_value(_first_present(data, "session_id")),
        execution_id=_string_value(_first_present(data, "execution_id")),
        correlation_id=correlation_id or _string_value(_first_present(data, "correlation_id")),
        trace_id=trace_ids.get("trace_id") or _string_value(_first_present(data, "trace_id")),
        span_id=trace_ids.get("span_id") or _string_value(_first_present(data, "span_id")),
        parent_event_id=_string_value(_first_present(data, "parent_event_id")),
        iteration=_coerce_int(_first_present_from(data, payload, "iteration")),
        phase=_string_value(_first_present_from(data, payload, "phase")),
        duration_ms=_coerce_float(_first_present_from(data, payload, "duration_ms")),
        tags=_string_list(_first_present(data, "tags")),
        payload=payload,
    )


def federate_llmcore_events(
    events: Iterable[Any],
    *,
    correlation_id: str | None = None,
    trace_context: dict[str, str] | None = None,
    source_component: str | None = None,
) -> list[EcosystemEvent]:
    """Normalize a collection of llmcore or agent events."""

    return [
        federate_llmcore_event(
            event,
            correlation_id=correlation_id,
            trace_context=trace_context,
            source_component=source_component,
        )
        for event in events
    ]


def load_federated_llmcore_events_from_file(
    log_path: str | Path,
    *,
    execution_id: str | None = None,
    category: Any | None = None,
    min_severity: Any | None = None,
    limit: int | None = None,
    correlation_id: str | None = None,
    trace_context: dict[str, str] | None = None,
    source_component: str | None = None,
) -> list[EcosystemEvent]:
    """Load a classic llmcore event JSONL file as federated events."""

    from llmcore.observability.events import load_events_from_file

    events = load_events_from_file(
        log_path,
        execution_id=execution_id,
        category=category,
        min_severity=min_severity,
        limit=limit,
    )
    return federate_llmcore_events(
        events,
        correlation_id=correlation_id,
        trace_context=trace_context,
        source_component=source_component,
    )


def load_federated_agent_events_from_file(
    log_path: str | Path,
    *,
    execution_id: str | None = None,
    session_id: str | None = None,
    limit: int | None = None,
    correlation_id: str | None = None,
    trace_context: dict[str, str] | None = None,
    source_component: str | None = "agents.observability",
) -> list[EcosystemEvent]:
    """Load an agent observability JSONL file as federated events."""

    from llmcore.agents.observability.replay import ExecutionReplay

    replay = ExecutionReplay.from_file(log_path, max_events=limit)
    events = replay.iter_events(execution_id=execution_id)
    if session_id is not None:
        events = (event for event in events if event.session_id == session_id)

    return federate_llmcore_events(
        events,
        correlation_id=correlation_id,
        trace_context=trace_context,
        source_component=source_component,
    )


class FederatedEventSink:
    """Agent EventLogger sink that forwards normalized ecosystem events."""

    def __init__(
        self,
        callback: Callable[[EcosystemEvent], None] | None = None,
        *,
        async_callback: Callable[[EcosystemEvent], Awaitable[Any]] | None = None,
        correlation_id: str | None = None,
        trace_context: dict[str, str] | None = None,
        source_component: str | None = "agents.observability",
    ) -> None:
        if callback is None and async_callback is None:
            raise ValueError("callback or async_callback is required")

        self._callback = callback
        self._async_callback = async_callback
        self._correlation_id = correlation_id
        self._trace_context = trace_context
        self._source_component = source_component

    async def write(self, event: Any) -> None:
        """Normalize an event and forward it to the configured callback."""

        federated = federate_llmcore_event(
            event,
            correlation_id=self._correlation_id,
            trace_context=self._trace_context,
            source_component=self._source_component,
        )
        if self._async_callback is not None:
            await self._async_callback(federated)
        elif self._callback is not None:
            self._callback(federated)

    async def flush(self) -> None:
        """No-op for callback-backed federation."""

    async def close(self) -> None:
        """No-op for callback-backed federation."""

    @property
    def name(self) -> str:
        """Return the sink name for EventLogger diagnostics."""

        return self.__class__.__name__


def federate_semantiscan_event(
    event: Any,
    *,
    correlation_id: str | None = None,
    trace_context: dict[str, str] | None = None,
    source_component: str | None = None,
) -> EcosystemEvent:
    """Normalize a semantiscan event-bus event."""

    data = _event_mapping(event)
    trace_ids = trace_ids_from_context(trace_context)
    payload = _payload_from(data)
    event_trace_ids = trace_ids_from_context(_coerce_trace_context(payload.get("trace_context")))

    return EcosystemEvent(
        event_id=_string_value(_first_present(data, "event_id", "id")) or _generated_event_id(),
        timestamp=_coerce_timestamp(_first_present(data, "timestamp")),
        source_system=SourceSystem.SEMANTISCAN,
        source_component=source_component or _string_value(_first_present(data, "source")),
        category=_string_value(_first_present(data, "category")) or "custom",
        event_type=_string_value(_first_present(data, "event_type")) or "event",
        severity=_string_value(_first_present(data, "severity")) or "info",
        session_id=_string_value(_first_present_from(data, payload, "session_id", "collection_id")),
        execution_id=_string_value(_first_present_from(data, payload, "execution_id", "run_id")),
        correlation_id=correlation_id
        or _string_value(_first_present_from(data, payload, "correlation_id")),
        trace_id=trace_ids.get("trace_id")
        or event_trace_ids.get("trace_id")
        or _string_value(_first_present(data, "trace_id")),
        span_id=trace_ids.get("span_id")
        or event_trace_ids.get("span_id")
        or _string_value(_first_present(data, "span_id")),
        duration_ms=_coerce_float(_first_present_from(data, payload, "duration_ms")),
        tags=_string_list(_first_present(data, "tags")),
        payload=payload,
    )


def federate_metastore_event(
    event: Any,
    *,
    correlation_id: str | None = None,
    trace_context: dict[str, str] | None = None,
    source_component: str = "metastore",
) -> EcosystemEvent:
    """Normalize a semantiscan MetaStore event log record."""

    data = _event_mapping(event)
    trace_ids = trace_ids_from_context(trace_context)
    payload = _payload_from(data)
    event_trace_ids = trace_ids_from_context(_coerce_trace_context(payload.get("trace_context")))

    return EcosystemEvent(
        event_id=_string_value(_first_present(data, "event_id", "id")) or _generated_event_id(),
        timestamp=_coerce_timestamp(_first_present(data, "created_at", "timestamp")),
        source_system=SourceSystem.SEMANTISCAN,
        source_component=source_component,
        category="metastore",
        event_type=_string_value(_first_present(data, "event_type")) or "event",
        severity=_string_value(_first_present(data, "severity")) or "info",
        execution_id=_string_value(_first_present_from(data, payload, "run_id", "execution_id")),
        correlation_id=correlation_id
        or _string_value(_first_present_from(data, payload, "correlation_id")),
        trace_id=trace_ids.get("trace_id")
        or event_trace_ids.get("trace_id")
        or _string_value(_first_present(data, "trace_id")),
        span_id=trace_ids.get("span_id")
        or event_trace_ids.get("span_id")
        or _string_value(_first_present(data, "span_id")),
        payload=payload,
    )


def federate_wairu_ipc_message(
    message: Any,
    *,
    correlation_id: str | None = None,
    source_component: str = "daemon.ipc",
) -> EcosystemEvent:
    """Normalize a Wairu IPC message or message-shaped mapping."""

    data = _event_mapping(message)
    payload = _wairu_ipc_payload(data)
    trace_context = _coerce_trace_context(_first_present(data, "trace_context"))
    trace_ids = trace_ids_from_context(trace_context)
    message_type = _string_value(_first_present(data, "type")) or "message"
    event_name = (
        _string_value(_first_present(data, "name"))
        or _string_value(_first_present(data, "command"))
        or message_type
    )

    return EcosystemEvent(
        event_id=_string_value(_first_present(data, "event_id", "id")) or _generated_event_id(),
        source_system=SourceSystem.WAIRU,
        source_component=source_component,
        category="ipc",
        event_type=f"ipc.{message_type}.{event_name}",
        severity="error" if _first_present(data, "error") else "info",
        correlation_id=correlation_id or _string_value(_first_present(data, "correlation_id")),
        trace_id=trace_ids.get("trace_id") or _string_value(_first_present(data, "trace_id")),
        span_id=trace_ids.get("span_id") or _string_value(_first_present(data, "span_id")),
        payload=payload,
    )


def trace_ids_from_context(trace_context: dict[str, str] | None) -> dict[str, str]:
    """Extract W3C trace and span IDs from a trace-context carrier."""

    if not trace_context:
        return {}

    traceparent = trace_context.get("traceparent") or trace_context.get("Traceparent")
    if not isinstance(traceparent, str):
        return {}

    parts = traceparent.split("-")
    if len(parts) < 4:
        return {}

    trace_id = parts[1].strip()
    span_id = parts[2].strip()
    if len(trace_id) != 32 or len(span_id) != 16:
        return {}
    return {"trace_id": trace_id, "span_id": span_id}


def _event_mapping(event: Any) -> dict[str, Any]:
    if isinstance(event, EcosystemEvent):
        return event.to_dict()
    if isinstance(event, dict):
        return dict(event)

    to_dict = getattr(event, "to_dict", None)
    if callable(to_dict):
        mapped = to_dict()
        if isinstance(mapped, dict):
            return dict(mapped)

    model_dump = getattr(event, "model_dump", None)
    if callable(model_dump):
        mapped = model_dump(mode="json")
        if isinstance(mapped, dict):
            return dict(mapped)

    if hasattr(event, "__dict__"):
        return dict(vars(event))

    return {}


def _first_present(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _first_present_from(
    primary: dict[str, Any],
    fallback: dict[str, Any],
    *keys: str,
) -> Any:
    value = _first_present(primary, *keys)
    if value is not None:
        return value
    return _first_present(fallback, *keys)


def _payload_from(data: dict[str, Any]) -> dict[str, Any]:
    payload = data.get("data")
    if isinstance(payload, dict):
        normalized = dict(payload)
    else:
        payload = data.get("payload")
        normalized = dict(payload) if isinstance(payload, dict) else {}

    for key, value in data.items():
        if key in _FEDERATED_ENVELOPE_KEYS or value is None:
            continue
        normalized.setdefault(key, value)
    return normalized


def _wairu_ipc_payload(data: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in ("command", "params", "data", "error", "name"):
        value = data.get(key)
        if value is not None:
            payload[key] = value
    return payload


def _string_value(value: Any) -> str | None:
    if value is None:
        return None
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return str(enum_value)
    return str(value)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            normalized = value.replace("Z", "+00:00")
            timestamp = datetime.fromisoformat(normalized)
            if timestamp.tzinfo is None:
                return timestamp.replace(tzinfo=UTC)
            return timestamp
        except ValueError:
            pass
    return datetime.now(UTC)


def _coerce_trace_context(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    carrier = {str(key): str(val) for key, val in value.items() if val is not None}
    return carrier or None


def _generated_event_id() -> str:
    return f"evt-{datetime.now(UTC).timestamp():.6f}"


_FEDERATED_ENVELOPE_KEYS = {
    "category",
    "collection_id",
    "correlation_id",
    "created_at",
    "data",
    "duration_ms",
    "event_id",
    "event_type",
    "execution_id",
    "id",
    "iteration",
    "parent_event_id",
    "payload",
    "phase",
    "run_id",
    "session_id",
    "severity",
    "source",
    "source_component",
    "span_id",
    "tags",
    "timestamp",
    "trace_context",
    "trace_id",
}


__all__ = [
    "EcosystemEvent",
    "FederatedEventSink",
    "SourceSystem",
    "federate_llmcore_event",
    "federate_llmcore_events",
    "federate_metastore_event",
    "federate_semantiscan_event",
    "federate_wairu_ipc_message",
    "load_federated_agent_events_from_file",
    "load_federated_llmcore_events_from_file",
    "trace_ids_from_context",
]
