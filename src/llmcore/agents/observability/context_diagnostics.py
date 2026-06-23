"""Context failure detectors for agent runtime observability.

The detectors in this module are deliberately data-shape tolerant: callers can
pass llmcore model objects, dictionaries, or lightweight test doubles. This
keeps diagnostics usable from Darwin phases, Wairu bridges, and replay tooling
without adding cross-package imports.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .events import ErrorEvent, ErrorEventType, EventSeverity, create_error_event


class ContextFailureType(str, Enum):
    """Named context failure classes used by diagnostics and regression tests."""

    CONTEXT_OVERFLOW = "context_overflow"
    ORPHAN_TOOL_RESULT = "orphan_tool_result"
    SUMMARY_THRASH = "summary_thrash"
    STALE_CONTEXT = "stale_context"
    REPEATED_TOOL_FAILURE = "repeated_tool_failure"


class ContextFailureDetectorConfig(BaseModel):
    """Thresholds used by :func:`detect_context_failures`."""

    summary_thrash_threshold: int = Field(default=3, ge=2)
    summary_thrash_window: int = Field(default=5, ge=1)
    stale_context_iterations: int = Field(default=5, ge=1)
    repeated_tool_failure_threshold: int = Field(default=3, ge=2)


class ContextFailureDiagnostic(BaseModel):
    """A detected context failure that can be converted into an ``ErrorEvent``."""

    failure_type: ContextFailureType
    message: str
    severity: EventSeverity = EventSeverity.WARNING
    details: dict[str, Any] = Field(default_factory=dict)
    iteration: int | None = None
    phase: str | None = None
    recoverable: bool = True
    recommendation: str | None = None
    tags: list[str] = Field(default_factory=list)

    def to_event(
        self,
        session_id: str,
        *,
        execution_id: str | None = None,
        correlation_id: str | None = None,
    ) -> ErrorEvent:
        """Convert this diagnostic to a structured observability event."""
        tags = ["context_failure", self.failure_type.value, *self.tags]
        data = {
            "failure_type": self.failure_type.value,
            "details": self.details,
        }
        if self.recommendation:
            data["recommendation"] = self.recommendation

        return create_error_event(
            session_id=session_id,
            execution_id=execution_id,
            event_type=ErrorEventType.CONTEXT_FAILURE,
            error_type=self.failure_type.value,
            error_message=self.message,
            severity=self.severity,
            recoverable=self.recoverable,
            iteration=self.iteration,
            phase=self.phase,
            correlation_id=correlation_id,
            tags=tags,
            data=data,
        )


def detect_context_failures(
    *,
    messages: Sequence[Any] | None = None,
    estimated_tokens: int | None = None,
    max_context_tokens: int | None = None,
    compression_events: Sequence[Any] | None = None,
    current_iteration: int | None = None,
    context_updated_iteration: int | None = None,
    tool_failures: Sequence[Any] | None = None,
    phase: str | None = None,
    config: ContextFailureDetectorConfig | None = None,
) -> list[ContextFailureDiagnostic]:
    """Run all context failure detectors and return named diagnostics."""
    cfg = config or ContextFailureDetectorConfig()
    diagnostics: list[ContextFailureDiagnostic] = []

    overflow = _detect_context_overflow(
        estimated_tokens=estimated_tokens,
        max_context_tokens=max_context_tokens,
        iteration=current_iteration,
        phase=phase,
    )
    if overflow is not None:
        diagnostics.append(overflow)

    orphan = _detect_orphan_tool_results(
        messages=messages or (),
        iteration=current_iteration,
        phase=phase,
    )
    if orphan is not None:
        diagnostics.append(orphan)

    thrash = _detect_summary_thrash(
        compression_events=compression_events or (),
        current_iteration=current_iteration,
        config=cfg,
        phase=phase,
    )
    if thrash is not None:
        diagnostics.append(thrash)

    stale = _detect_stale_context(
        current_iteration=current_iteration,
        context_updated_iteration=context_updated_iteration,
        config=cfg,
        phase=phase,
    )
    if stale is not None:
        diagnostics.append(stale)

    repeated_failure = _detect_repeated_tool_failure(
        tool_failures=tool_failures or (),
        config=cfg,
        iteration=current_iteration,
        phase=phase,
    )
    if repeated_failure is not None:
        diagnostics.append(repeated_failure)

    return diagnostics


def _detect_context_overflow(
    *,
    estimated_tokens: int | None,
    max_context_tokens: int | None,
    iteration: int | None,
    phase: str | None,
) -> ContextFailureDiagnostic | None:
    if estimated_tokens is None or max_context_tokens is None or max_context_tokens <= 0:
        return None
    if estimated_tokens <= max_context_tokens:
        return None

    overflow_tokens = estimated_tokens - max_context_tokens
    return ContextFailureDiagnostic(
        failure_type=ContextFailureType.CONTEXT_OVERFLOW,
        message=(
            "Prepared context exceeds the configured token budget "
            f"({estimated_tokens} > {max_context_tokens})."
        ),
        severity=EventSeverity.ERROR,
        iteration=iteration,
        phase=phase,
        details={
            "estimated_tokens": estimated_tokens,
            "max_context_tokens": max_context_tokens,
            "overflow_tokens": overflow_tokens,
            "overflow_ratio": estimated_tokens / max_context_tokens,
        },
        recommendation="Compress, summarize, or drop low-priority context before the next LLM call.",
    )


def _detect_orphan_tool_results(
    *,
    messages: Sequence[Any],
    iteration: int | None,
    phase: str | None,
) -> ContextFailureDiagnostic | None:
    tool_call_ids: set[str] = set()
    orphan_results: list[dict[str, Any]] = []

    for index, message in enumerate(messages):
        for tool_call_id in _tool_call_ids_from_message(message):
            tool_call_ids.add(tool_call_id)

        if not _is_tool_result_message(message):
            continue

        result_id = _string_field(message, "tool_call_id", "call_id", "id")
        if not result_id or result_id not in tool_call_ids:
            orphan_results.append(
                {
                    "index": index,
                    "tool_call_id": result_id,
                    "role": _normalized_role(message),
                }
            )

    if not orphan_results:
        return None

    return ContextFailureDiagnostic(
        failure_type=ContextFailureType.ORPHAN_TOOL_RESULT,
        message=f"Found {len(orphan_results)} tool result(s) without matching tool calls.",
        severity=EventSeverity.ERROR,
        iteration=iteration,
        phase=phase,
        details={
            "orphan_count": len(orphan_results),
            "known_tool_call_ids": sorted(tool_call_ids),
            "orphan_results": orphan_results[:10],
        },
        recommendation="Keep tool-call/result pairs together when pruning or replaying context.",
    )


def _detect_summary_thrash(
    *,
    compression_events: Sequence[Any],
    current_iteration: int | None,
    config: ContextFailureDetectorConfig,
    phase: str | None,
) -> ContextFailureDiagnostic | None:
    if not compression_events:
        return None

    recent = list(compression_events)
    if current_iteration is not None:
        recent = [
            event
            for event in recent
            if _event_iteration(event) is None
            or current_iteration - int(_event_iteration(event) or current_iteration)
            <= config.summary_thrash_window
        ]
    else:
        recent = recent[-config.summary_thrash_window :]

    if len(recent) < config.summary_thrash_threshold:
        return None

    digests = {_summary_digest(event) for event in recent}
    return ContextFailureDiagnostic(
        failure_type=ContextFailureType.SUMMARY_THRASH,
        message=(
            f"Context was compressed {len(recent)} time(s) within the recent "
            "diagnostic window."
        ),
        severity=EventSeverity.WARNING,
        iteration=current_iteration,
        phase=phase,
        details={
            "recent_compression_count": len(recent),
            "unique_summary_count": len(digests),
            "summary_thrash_threshold": config.summary_thrash_threshold,
            "summary_thrash_window": config.summary_thrash_window,
        },
        recommendation="Increase compression cooldown or preserve a stable objective summary.",
    )


def _detect_stale_context(
    *,
    current_iteration: int | None,
    context_updated_iteration: int | None,
    config: ContextFailureDetectorConfig,
    phase: str | None,
) -> ContextFailureDiagnostic | None:
    if current_iteration is None or context_updated_iteration is None:
        return None
    age = current_iteration - context_updated_iteration
    if age < config.stale_context_iterations:
        return None

    return ContextFailureDiagnostic(
        failure_type=ContextFailureType.STALE_CONTEXT,
        message=f"Context has not been refreshed for {age} iteration(s).",
        severity=EventSeverity.WARNING,
        iteration=current_iteration,
        phase=phase,
        details={
            "current_iteration": current_iteration,
            "context_updated_iteration": context_updated_iteration,
            "age_iterations": age,
            "stale_context_iterations": config.stale_context_iterations,
        },
        recommendation="Refresh retrieval/context sources before making another grounded decision.",
    )


def _detect_repeated_tool_failure(
    *,
    tool_failures: Sequence[Any],
    config: ContextFailureDetectorConfig,
    iteration: int | None,
    phase: str | None,
) -> ContextFailureDiagnostic | None:
    streak_tool: str | None = None
    streak = 0
    errors: list[str] = []

    for event in reversed(list(tool_failures)):
        tool_name = _string_field(event, "tool_name", "activity_name", "name") or "unknown"
        success = bool(_field(event, "success", default=False))
        if success:
            if streak_tool is None:
                return None
            if tool_name == streak_tool:
                break
        if streak_tool is None:
            streak_tool = tool_name
        if tool_name != streak_tool:
            break
        if success:
            break
        streak += 1
        error = _string_field(event, "error", "error_message", "message")
        if error:
            errors.append(error)

    if streak < config.repeated_tool_failure_threshold or streak_tool is None:
        return None

    return ContextFailureDiagnostic(
        failure_type=ContextFailureType.REPEATED_TOOL_FAILURE,
        message=f"Tool '{streak_tool}' failed {streak} consecutive time(s).",
        severity=EventSeverity.WARNING,
        iteration=iteration,
        phase=phase,
        details={
            "tool_name": streak_tool,
            "consecutive_failures": streak,
            "repeated_tool_failure_threshold": config.repeated_tool_failure_threshold,
            "recent_errors": list(reversed(errors[:5])),
        },
        recommendation="Change strategy, request human input, or suppress the repeated tool path.",
    )


def _field(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    metadata = None
    if isinstance(value, dict):
        metadata = value.get("metadata")
    elif hasattr(value, "metadata"):
        metadata = value.metadata
    if isinstance(metadata, dict):
        for name in names:
            if name in metadata:
                return metadata[name]
    return default


def _string_field(value: Any, *names: str) -> str | None:
    result = _field(value, *names)
    if result is None:
        return None
    text = str(result).strip()
    return text or None


def _normalized_role(message: Any) -> str | None:
    role = _field(message, "role")
    if role is None:
        return None
    value = getattr(role, "value", role)
    return str(value).lower()


def _is_tool_result_message(message: Any) -> bool:
    role = _normalized_role(message)
    if role == "tool":
        return True
    item_type = _string_field(message, "type", "kind")
    return item_type in {"tool_result", "activity_result"}


def _tool_call_ids_from_message(message: Any) -> list[str]:
    tool_call_ids: list[str] = []
    tool_calls = _field(message, "tool_calls")
    if tool_calls is None:
        metadata = _field(message, "metadata", default={})
        if isinstance(metadata, dict):
            tool_calls = metadata.get("tool_calls")
    if not isinstance(tool_calls, list | tuple):
        return tool_call_ids

    for call in tool_calls:
        call_id = _string_field(call, "id", "tool_call_id", "call_id")
        if call_id:
            tool_call_ids.append(call_id)
    return tool_call_ids


def _event_iteration(event: Any) -> int | None:
    value = _field(event, "iteration")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _summary_digest(event: Any) -> str:
    summary = _field(event, "summary", "summary_text", "objective_summary", "content")
    if summary is None:
        summary = repr(event)
    return hashlib.sha256(str(summary).encode("utf-8")).hexdigest()


__all__ = [
    "ContextFailureDetectorConfig",
    "ContextFailureDiagnostic",
    "ContextFailureType",
    "detect_context_failures",
]
