# src/llmcore/agents/observability/replay.py
"""
Execution Replay for Agent Observability.

Provides capabilities to load, replay, and analyze agent executions
from event logs. Useful for debugging, analysis, and audit.

Architecture:
    ExecutionReplay loads events from JSONL files and provides
    methods to reconstruct execution timelines, filter events,
    and generate analysis reports.

Usage:
    >>> from llmcore.agents.observability.replay import ExecutionReplay
    >>>
    >>> # Load from event log
    >>> replay = ExecutionReplay.from_file("events.jsonl")
    >>>
    >>> # List available executions
    >>> executions = replay.list_executions()
    >>>
    >>> # Replay specific execution
    >>> result = replay.replay("exec-123")
    >>> for step in result.timeline:
    ...     print(f"{step.timestamp}: {step.summary}")
    >>>
    >>> # Get events with filtering
    >>> errors = replay.get_events(
    ...     session_id="sess-123",
    ...     category=EventCategory.ERROR,
    ... )

References:
    - Master Plan: Section 28 (Debugging & Replay)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Type, Union

from pydantic import BaseModel

from .events import (
    ActivityEvent,
    AgentEvent,
    CognitiveEvent,
    ErrorEvent,
    EventCategory,
    EventSeverity,
    HITLEvent,
    LifecycleEvent,
    MemoryEvent,
    MetricEvent,
    RAGEvent,
    SandboxEvent,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ReplayStep:
    """
    A single step in an execution replay timeline.

    Attributes:
        timestamp: When the event occurred
        event_id: Event identifier
        category: Event category
        event_type: Specific event type
        phase: Cognitive phase (if applicable)
        iteration: Iteration number (if applicable)
        summary: Human-readable summary
        duration_ms: Duration (if available)
        event: Full event data
    """

    timestamp: datetime
    event_id: str
    category: EventCategory
    event_type: str
    phase: Optional[str]
    iteration: Optional[int]
    summary: str
    duration_ms: Optional[float]
    event: AgentEvent


@dataclass
class ExecutionInfo:
    """
    Information about an execution found in the log.

    Attributes:
        execution_id: Execution identifier
        session_id: Session identifier
        start_time: When execution started
        end_time: When execution ended (if known)
        goal: Goal being executed (if available)
        status: Execution status (if known)
        event_count: Number of events
        error_count: Number of error events
    """

    execution_id: str
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    goal: Optional[str] = None
    status: Optional[str] = None
    event_count: int = 0
    error_count: int = 0


@dataclass
class ReplayResult:
    """
    Result of replaying an execution.

    Attributes:
        execution_id: Execution identifier
        session_id: Session identifier
        start_time: When execution started
        end_time: When execution ended
        goal: Goal being executed
        status: Final status
        total_events: Total event count
        timeline: Ordered list of replay steps
        phases: Phases encountered
        activities: Activities executed
        errors: Errors encountered
        summary: Execution summary
    """

    execution_id: str
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    goal: Optional[str]
    status: Optional[str]
    total_events: int
    timeline: List[ReplayStep]
    phases: List[str] = field(default_factory=list)
    activities: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# EVENT TYPE MAPPING
# =============================================================================

# Map category to event class for deserialization
EVENT_CLASS_MAP: Dict[str, Type[AgentEvent]] = {
    "lifecycle": LifecycleEvent,
    "cognitive": CognitiveEvent,
    "activity": ActivityEvent,
    "memory": MemoryEvent,
    "hitl": HITLEvent,
    "error": ErrorEvent,
    "metric": MetricEvent,
    "sandbox": SandboxEvent,
    "rag": RAGEvent,
}


def parse_event(data: Dict[str, Any]) -> AgentEvent:
    """
    Parse event data into appropriate event class.

    Args:
        data: Event data dictionary

    Returns:
        Parsed event instance
    """
    category = data.get("category", "")
    event_class = EVENT_CLASS_MAP.get(category, AgentEvent)

    try:
        return event_class.model_validate(data)
    except Exception as e:
        logger.warning(
            f"Failed to parse as {event_class.__name__}, falling back to AgentEvent: {e}"
        )
        return AgentEvent.model_validate(data)


# =============================================================================
# EXECUTION REPLAY
# =============================================================================


class ExecutionReplay:
    """
    Load and replay agent executions from event logs.

    Provides capabilities for:
    - Loading events from JSONL files
    - Listing available executions
    - Replaying specific executions
    - Filtering events by various criteria
    - Generating execution summaries

    Usage:
        # Load from file
        replay = ExecutionReplay.from_file("events.jsonl")

        # List executions
        for info in replay.list_executions():
            print(f"{info.execution_id}: {info.goal}")

        # Replay execution
        result = replay.replay("exec-123")

        # Filter events
        errors = replay.get_events(category=EventCategory.ERROR)
    """

    def __init__(self, events: Optional[List[AgentEvent]] = None) -> None:
        """
        Initialize replay with events.

        Args:
            events: List of events to replay
        """
        self._events: List[AgentEvent] = events or []
        self._by_session: Dict[str, List[AgentEvent]] = {}
        self._by_execution: Dict[str, List[AgentEvent]] = {}
        self._execution_info: Dict[str, ExecutionInfo] = {}

        if self._events:
            self._index_events()

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        *,
        max_events: Optional[int] = None,
    ) -> "ExecutionReplay":
        """
        Load replay from JSONL file.

        Args:
            path: Path to JSONL file
            max_events: Maximum events to load

        Returns:
            ExecutionReplay instance
        """
        path = Path(path)
        events: List[AgentEvent] = []

        if not path.exists():
            logger.warning(f"Event log not found: {path}")
            return cls(events)

        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_events and i >= max_events:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    event = parse_event(data)
                    events.append(event)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {i + 1}: {e}")
                except Exception as e:
                    logger.warning(f"Failed to parse event on line {i + 1}: {e}")

        logger.info(f"Loaded {len(events)} events from {path}")
        return cls(events)

    @classmethod
    def from_events(cls, events: List[AgentEvent]) -> "ExecutionReplay":
        """
        Create replay from list of events.

        Args:
            events: List of events

        Returns:
            ExecutionReplay instance
        """
        return cls(list(events))

    def _index_events(self) -> None:
        """Index events by session and execution."""
        self._by_session.clear()
        self._by_execution.clear()
        self._execution_info.clear()

        for event in self._events:
            # Index by session
            session_id = event.session_id
            if session_id not in self._by_session:
                self._by_session[session_id] = []
            self._by_session[session_id].append(event)

            # Index by execution
            exec_id = event.execution_id
            if exec_id:
                if exec_id not in self._by_execution:
                    self._by_execution[exec_id] = []
                self._by_execution[exec_id].append(event)

                # Track execution info
                self._update_execution_info(exec_id, event)

    def _update_execution_info(
        self,
        execution_id: str,
        event: AgentEvent,
    ) -> None:
        """Update execution info from event."""
        if execution_id not in self._execution_info:
            self._execution_info[execution_id] = ExecutionInfo(
                execution_id=execution_id,
                session_id=event.session_id,
                start_time=event.timestamp,
            )

        info = self._execution_info[execution_id]
        info.event_count += 1

        # Update end time
        if info.end_time is None or event.timestamp > info.end_time:
            info.end_time = event.timestamp

        # Update from lifecycle events
        if isinstance(event, LifecycleEvent):
            if event.goal:
                info.goal = event.goal
            if event.final_status:
                info.status = event.final_status

        # Count errors
        if event.category == EventCategory.ERROR:
            info.error_count += 1

    # =========================================================================
    # LISTING METHODS
    # =========================================================================

    def list_executions(
        self,
        *,
        session_id: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[ExecutionInfo]:
        """
        List available executions.

        Args:
            session_id: Filter by session
            since: Only after this time
            until: Only before this time
            limit: Maximum results

        Returns:
            List of execution info
        """
        results: List[ExecutionInfo] = []

        for info in sorted(
            self._execution_info.values(),
            key=lambda x: x.start_time,
            reverse=True,
        ):
            if len(results) >= limit:
                break

            if session_id and info.session_id != session_id:
                continue

            if since and info.start_time < since:
                continue

            if until and info.start_time > until:
                continue

            results.append(info)

        return results

    def list_sessions(self) -> List[str]:
        """
        List all session IDs.

        Returns:
            List of session IDs
        """
        return sorted(self._by_session.keys())

    # =========================================================================
    # REPLAY METHODS
    # =========================================================================

    def replay(
        self,
        execution_id: str,
        *,
        include_metrics: bool = False,
    ) -> ReplayResult:
        """
        Replay a specific execution.

        Args:
            execution_id: Execution to replay
            include_metrics: Include metric events in timeline

        Returns:
            ReplayResult with timeline and summary
        """
        events = self._by_execution.get(execution_id, [])
        if not events:
            raise ValueError(f"Execution not found: {execution_id}")

        # Sort by timestamp
        events = sorted(events, key=lambda e: e.timestamp)

        # Build timeline
        timeline: List[ReplayStep] = []
        phases: Set[str] = set()
        activities: Set[str] = set()
        errors: List[str] = []

        goal: Optional[str] = None
        status: Optional[str] = None
        session_id = events[0].session_id
        start_time = events[0].timestamp
        end_time = events[-1].timestamp

        for event in events:
            # Skip metrics unless requested
            if not include_metrics and event.category == EventCategory.METRIC:
                continue

            # Build step
            step = ReplayStep(
                timestamp=event.timestamp,
                event_id=event.event_id,
                category=event.category,
                event_type=event.event_type,
                phase=event.phase,
                iteration=event.iteration,
                summary=self._summarize_event(event),
                duration_ms=event.duration_ms,
                event=event,
            )
            timeline.append(step)

            # Collect metadata
            if event.phase:
                phases.add(event.phase)

            if isinstance(event, ActivityEvent):
                activities.add(event.activity_name)

            if isinstance(event, ErrorEvent):
                errors.append(event.error_message)

            if isinstance(event, LifecycleEvent):
                if event.goal:
                    goal = event.goal
                if event.final_status:
                    status = event.final_status

        # Build summary
        summary = self._build_execution_summary(events)

        return ReplayResult(
            execution_id=execution_id,
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            goal=goal,
            status=status,
            total_events=len(events),
            timeline=timeline,
            phases=sorted(phases),
            activities=sorted(activities),
            errors=errors,
            summary=summary,
        )

    def _summarize_event(self, event: AgentEvent) -> str:
        """Generate human-readable summary for event."""
        if isinstance(event, LifecycleEvent):
            if event.event_type == "agent_started":
                return f"Started: {event.goal or 'Unknown goal'}"
            elif event.event_type == "agent_completed":
                return f"Completed: {event.final_status or 'success'}"
            elif event.event_type == "agent_failed":
                return f"Failed: {event.exit_reason or 'Unknown reason'}"
            elif event.event_type == "iteration_started":
                return f"Iteration {event.iteration} started"
            elif event.event_type == "iteration_completed":
                return f"Iteration {event.iteration} completed"
            else:
                return f"Lifecycle: {event.event_type}"

        elif isinstance(event, CognitiveEvent):
            phase = event.phase_name or event.phase or "unknown"
            if event.event_type == "phase_started":
                return f"Phase '{phase}' started"
            elif event.event_type == "phase_completed":
                output = event.output_summary or ""
                if len(output) > 50:
                    output = output[:50] + "..."
                return f"Phase '{phase}' completed: {output}"
            else:
                return f"Cognitive: {phase} - {event.event_type}"

        elif isinstance(event, ActivityEvent):
            status = "✓" if event.success else "✗"
            return f"{status} Activity '{event.activity_name}'"

        elif isinstance(event, HITLEvent):
            return f"HITL: {event.action_type} - {event.approval_status}"

        elif isinstance(event, ErrorEvent):
            return f"Error: {event.error_type}: {event.error_message[:50]}"

        elif isinstance(event, SandboxEvent):
            return f"Sandbox: {event.sandbox_type} - {event.operation}"

        elif isinstance(event, RAGEvent):
            return f"RAG: {event.num_results} results for '{event.query[:30]}'"

        elif isinstance(event, MemoryEvent):
            return f"Memory: {event.operation} on {event.memory_type}"

        elif isinstance(event, MetricEvent):
            return f"Metric: {event.metric_name} = {event.metric_value}"

        else:
            return f"{event.category.value}: {event.event_type}"

    def _build_execution_summary(
        self,
        events: List[AgentEvent],
    ) -> Dict[str, Any]:
        """Build execution summary from events."""
        # Count by category
        category_counts: Dict[str, int] = {}
        for event in events:
            # Handle both enum and string categories
            cat = event.category.value if hasattr(event.category, "value") else str(event.category)
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Count iterations
        iterations = set()
        for event in events:
            if event.iteration is not None:
                iterations.add(event.iteration)

        # Collect durations
        durations: List[float] = []
        for event in events:
            if event.duration_ms is not None:
                durations.append(event.duration_ms)

        # Collect activities
        activity_results: Dict[str, Dict[str, int]] = {}
        for event in events:
            if isinstance(event, ActivityEvent):
                name = event.activity_name
                if name not in activity_results:
                    activity_results[name] = {"success": 0, "failure": 0}
                if event.success:
                    activity_results[name]["success"] += 1
                else:
                    activity_results[name]["failure"] += 1

        return {
            "event_counts": category_counts,
            "total_iterations": len(iterations),
            "duration_sum_ms": sum(durations) if durations else 0,
            "activity_results": activity_results,
        }

    # =========================================================================
    # FILTERING METHODS
    # =========================================================================

    def get_events(
        self,
        *,
        session_id: Optional[str] = None,
        execution_id: Optional[str] = None,
        category: Optional[EventCategory] = None,
        event_type: Optional[str] = None,
        severity: Optional[EventSeverity] = None,
        phase: Optional[str] = None,
        iteration: Optional[int] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        has_duration: Optional[bool] = None,
        limit: Optional[int] = None,
    ) -> List[AgentEvent]:
        """
        Get filtered events.

        Args:
            session_id: Filter by session
            execution_id: Filter by execution
            category: Filter by category
            event_type: Filter by event type
            severity: Filter by severity
            phase: Filter by phase
            iteration: Filter by iteration
            since: Events after this time
            until: Events before this time
            has_duration: Filter by presence of duration
            limit: Maximum results

        Returns:
            Filtered list of events
        """
        # Start with appropriate base set
        if execution_id:
            events = self._by_execution.get(execution_id, [])
        elif session_id:
            events = self._by_session.get(session_id, [])
        else:
            events = self._events

        # Apply filters
        result: List[AgentEvent] = []

        for event in events:
            if limit and len(result) >= limit:
                break

            if category is not None and event.category != category:
                continue

            if event_type is not None and event.event_type != event_type:
                continue

            if severity is not None and event.severity != severity:
                continue

            if phase is not None and event.phase != phase:
                continue

            if iteration is not None and event.iteration != iteration:
                continue

            if since is not None and event.timestamp < since:
                continue

            if until is not None and event.timestamp > until:
                continue

            if has_duration is not None:
                has_dur = event.duration_ms is not None
                if has_dur != has_duration:
                    continue

            result.append(event)

        return result

    def get_errors(
        self,
        *,
        execution_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[ErrorEvent]:
        """
        Get all error events.

        Args:
            execution_id: Filter by execution
            session_id: Filter by session

        Returns:
            List of error events
        """
        events = self.get_events(
            execution_id=execution_id,
            session_id=session_id,
            category=EventCategory.ERROR,
        )
        return [e for e in events if isinstance(e, ErrorEvent)]

    def get_activities(
        self,
        *,
        execution_id: Optional[str] = None,
        activity_name: Optional[str] = None,
        success_only: bool = False,
        failed_only: bool = False,
    ) -> List[ActivityEvent]:
        """
        Get activity events.

        Args:
            execution_id: Filter by execution
            activity_name: Filter by activity name
            success_only: Only successful activities
            failed_only: Only failed activities

        Returns:
            List of activity events
        """
        events = self.get_events(
            execution_id=execution_id,
            category=EventCategory.ACTIVITY,
        )

        result: List[ActivityEvent] = []
        for event in events:
            if not isinstance(event, ActivityEvent):
                continue

            if activity_name and event.activity_name != activity_name:
                continue

            if success_only and not event.success:
                continue

            if failed_only and event.success:
                continue

            result.append(event)

        return result

    # =========================================================================
    # ITERATION METHODS
    # =========================================================================

    def iter_events(
        self,
        *,
        execution_id: Optional[str] = None,
    ) -> Iterator[AgentEvent]:
        """
        Iterate over events in chronological order.

        Args:
            execution_id: Filter by execution

        Yields:
            Events in order
        """
        if execution_id:
            events = self._by_execution.get(execution_id, [])
        else:
            events = self._events

        for event in sorted(events, key=lambda e: e.timestamp):
            yield event

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def event_count(self) -> int:
        """Get total event count."""
        return len(self._events)

    @property
    def execution_count(self) -> int:
        """Get number of executions."""
        return len(self._execution_info)

    @property
    def session_count(self) -> int:
        """Get number of sessions."""
        return len(self._by_session)


# =============================================================================
# PYDANTIC MODELS FOR API
# =============================================================================


class ReplayStepModel(BaseModel):
    """Pydantic model for replay step."""

    timestamp: datetime
    event_id: str
    category: str
    event_type: str
    phase: Optional[str]
    iteration: Optional[int]
    summary: str
    duration_ms: Optional[float]


class ReplayResultModel(BaseModel):
    """Pydantic model for replay result."""

    execution_id: str
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    goal: Optional[str]
    status: Optional[str]
    total_events: int
    timeline: List[ReplayStepModel]
    phases: List[str]
    activities: List[str]
    errors: List[str]
    summary: Dict[str, Any]


class ExecutionInfoModel(BaseModel):
    """Pydantic model for execution info."""

    execution_id: str
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    goal: Optional[str]
    status: Optional[str]
    event_count: int
    error_count: int


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data classes
    "ReplayStep",
    "ExecutionInfo",
    "ReplayResult",
    # Main class
    "ExecutionReplay",
    # Helpers
    "parse_event",
    "EVENT_CLASS_MAP",
    # Pydantic models
    "ReplayStepModel",
    "ReplayResultModel",
    "ExecutionInfoModel",
]
