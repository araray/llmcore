# src/llmcore/agents/observability/logger.py
"""
Event Logger for Agent Observability.

Provides centralized event logging with support for multiple sinks
(file, memory, custom handlers). Supports structured JSON output
and async operations.

Architecture:
    The EventLogger uses a sink-based architecture where events can
    be written to multiple destinations simultaneously. Built-in sinks
    include JSONL file output and in-memory storage for testing.

Usage:
    >>> from llmcore.agents.observability.logger import (
    ...     EventLogger,
    ...     JSONLFileSink,
    ...     InMemorySink,
    ... )
    >>>
    >>> # Create logger with file sink
    >>> logger = EventLogger(session_id="sess-123")
    >>> logger.add_sink(JSONLFileSink(Path("events.jsonl")))
    >>>
    >>> # Log events
    >>> await logger.log_lifecycle_start(goal="Analyze data")
    >>> await logger.log_cognitive_phase("think", input_summary="...")
    >>> await logger.log_activity("execute_python", {"code": "..."})
    >>> await logger.log_lifecycle_end(status="success")
    >>>
    >>> # Close logger
    >>> await logger.close()

References:
    - Master Plan: Section 26 (Structured Event Logging)
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union
from uuid import uuid4

from .events import (
    AgentEvent,
    ActivityEvent,
    ActivityEventType,
    CognitiveEvent,
    CognitiveEventType,
    ErrorEvent,
    ErrorEventType,
    EventCategory,
    EventSeverity,
    HITLEvent,
    HITLEventType,
    LifecycleEvent,
    LifecycleEventType,
    MetricEvent,
    MetricEventType,
    MemoryEvent,
    MemoryEventType,
    RAGEvent,
    RAGEventType,
    SandboxEvent,
    SandboxEventType,
    create_lifecycle_event,
    create_cognitive_event,
    create_activity_event,
    create_error_event,
    create_metric_event,
    create_hitl_event,
    create_sandbox_event,
)


logger = logging.getLogger(__name__)


# =============================================================================
# EVENT SINK PROTOCOL
# =============================================================================


class EventSink(ABC):
    """
    Abstract base class for event sinks.
    
    Event sinks receive events from the EventLogger and write them
    to their destination (file, database, network, etc.).
    """
    
    @abstractmethod
    async def write(self, event: AgentEvent) -> None:
        """
        Write an event to the sink.
        
        Args:
            event: Event to write
        """
        ...
    
    @abstractmethod
    async def flush(self) -> None:
        """Flush any buffered events."""
        ...
    
    @abstractmethod
    async def close(self) -> None:
        """Close the sink and release resources."""
        ...
    
    @property
    def name(self) -> str:
        """Return the sink name for identification."""
        return self.__class__.__name__


# =============================================================================
# BUILT-IN SINKS
# =============================================================================


class JSONLFileSink(EventSink):
    """
    Writes events to a JSONL (JSON Lines) file.
    
    Each event is written as a single JSON line. Supports automatic
    file creation and append mode.
    
    Attributes:
        path: Path to the output file
        buffer_size: Number of events to buffer before flushing
        create_dirs: Whether to create parent directories
    """
    
    def __init__(
        self,
        path: Union[str, Path],
        *,
        buffer_size: int = 1,
        create_dirs: bool = True,
    ) -> None:
        """
        Initialize JSONL file sink.
        
        Args:
            path: Path to output file
            buffer_size: Events to buffer before flush (1 = immediate)
            create_dirs: Create parent directories if needed
        """
        self.path = Path(path)
        self.buffer_size = buffer_size
        self.create_dirs = create_dirs
        self._buffer: List[str] = []
        self._file = None
        self._lock = asyncio.Lock()
    
    async def _ensure_file(self) -> None:
        """Ensure the output file is open."""
        if self._file is None:
            if self.create_dirs:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.path, "a", encoding="utf-8")
    
    async def write(self, event: AgentEvent) -> None:
        """
        Write event to file.
        
        Args:
            event: Event to write
        """
        async with self._lock:
            line = event.model_dump_json() + "\n"
            self._buffer.append(line)
            
            if len(self._buffer) >= self.buffer_size:
                await self._flush_buffer()
    
    async def _flush_buffer(self) -> None:
        """Flush buffered events to file."""
        if not self._buffer:
            return
        
        await self._ensure_file()
        for line in self._buffer:
            self._file.write(line)
        self._file.flush()
        self._buffer.clear()
    
    async def flush(self) -> None:
        """Flush all buffered events."""
        async with self._lock:
            await self._flush_buffer()
    
    async def close(self) -> None:
        """Close the file."""
        async with self._lock:
            await self._flush_buffer()
            if self._file is not None:
                self._file.close()
                self._file = None


class InMemorySink(EventSink):
    """
    Stores events in memory for testing and debugging.
    
    Events are stored in a list and can be retrieved for inspection.
    Supports filtering by category, session, etc.
    
    Attributes:
        max_events: Maximum events to store (oldest discarded)
    """
    
    def __init__(self, max_events: int = 10000) -> None:
        """
        Initialize in-memory sink.
        
        Args:
            max_events: Maximum events to store
        """
        self.max_events = max_events
        self._events: List[AgentEvent] = []
        self._lock = asyncio.Lock()
    
    async def write(self, event: AgentEvent) -> None:
        """
        Store event in memory.
        
        Args:
            event: Event to store
        """
        async with self._lock:
            self._events.append(event)
            # Trim if over limit
            if len(self._events) > self.max_events:
                self._events = self._events[-self.max_events:]
    
    async def flush(self) -> None:
        """No-op for memory sink."""
        pass
    
    async def close(self) -> None:
        """Clear stored events."""
        async with self._lock:
            self._events.clear()
    
    @property
    def events(self) -> List[AgentEvent]:
        """Get all stored events."""
        return list(self._events)
    
    def get_events(
        self,
        *,
        category: Optional[EventCategory] = None,
        session_id: Optional[str] = None,
        execution_id: Optional[str] = None,
        event_type: Optional[str] = None,
        severity: Optional[EventSeverity] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> List[AgentEvent]:
        """
        Get filtered events.
        
        Args:
            category: Filter by category
            session_id: Filter by session
            execution_id: Filter by execution
            event_type: Filter by event type
            severity: Filter by severity
            since: Events after this time
            until: Events before this time
            
        Returns:
            Filtered list of events
        """
        result = list(self._events)
        
        if category is not None:
            result = [e for e in result if e.category == category]
        if session_id is not None:
            result = [e for e in result if e.session_id == session_id]
        if execution_id is not None:
            result = [e for e in result if e.execution_id == execution_id]
        if event_type is not None:
            result = [e for e in result if e.event_type == event_type]
        if severity is not None:
            result = [e for e in result if e.severity == severity]
        if since is not None:
            result = [e for e in result if e.timestamp >= since]
        if until is not None:
            result = [e for e in result if e.timestamp <= until]
        
        return result
    
    def clear(self) -> None:
        """Clear all stored events."""
        self._events.clear()


class CallbackSink(EventSink):
    """
    Forwards events to a callback function.
    
    Useful for integration with custom logging systems or
    real-time event processing.
    """
    
    def __init__(
        self,
        callback: Callable[[AgentEvent], None],
        *,
        async_callback: Optional[Callable[[AgentEvent], Any]] = None,
    ) -> None:
        """
        Initialize callback sink.
        
        Args:
            callback: Synchronous callback function
            async_callback: Async callback (used if provided)
        """
        self._callback = callback
        self._async_callback = async_callback
    
    async def write(self, event: AgentEvent) -> None:
        """
        Forward event to callback.
        
        Args:
            event: Event to forward
        """
        if self._async_callback is not None:
            await self._async_callback(event)
        else:
            self._callback(event)
    
    async def flush(self) -> None:
        """No-op for callback sink."""
        pass
    
    async def close(self) -> None:
        """No-op for callback sink."""
        pass


class FilteredSink(EventSink):
    """
    Wraps another sink with filtering capabilities.
    
    Only forwards events that match the filter criteria.
    """
    
    def __init__(
        self,
        inner_sink: EventSink,
        *,
        categories: Optional[List[EventCategory]] = None,
        min_severity: Optional[EventSeverity] = None,
        include_types: Optional[List[str]] = None,
        exclude_types: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize filtered sink.
        
        Args:
            inner_sink: Sink to forward matching events to
            categories: Categories to include (None = all)
            min_severity: Minimum severity to include
            include_types: Event types to include
            exclude_types: Event types to exclude
        """
        self._inner = inner_sink
        self._categories = set(categories) if categories else None
        self._min_severity = min_severity
        self._include_types = set(include_types) if include_types else None
        self._exclude_types = set(exclude_types) if exclude_types else None
        
        # Severity ordering for comparison
        self._severity_order = {
            EventSeverity.DEBUG: 0,
            EventSeverity.INFO: 1,
            EventSeverity.WARNING: 2,
            EventSeverity.ERROR: 3,
            EventSeverity.CRITICAL: 4,
        }
    
    def _matches(self, event: AgentEvent) -> bool:
        """Check if event matches filter criteria."""
        # Category filter
        if self._categories is not None:
            if event.category not in self._categories:
                return False
        
        # Severity filter
        if self._min_severity is not None:
            event_level = self._severity_order.get(event.severity, 0)
            min_level = self._severity_order.get(self._min_severity, 0)
            if event_level < min_level:
                return False
        
        # Type filters
        if self._include_types is not None:
            if event.event_type not in self._include_types:
                return False
        
        if self._exclude_types is not None:
            if event.event_type in self._exclude_types:
                return False
        
        return True
    
    async def write(self, event: AgentEvent) -> None:
        """Write event if it matches filter."""
        if self._matches(event):
            await self._inner.write(event)
    
    async def flush(self) -> None:
        """Flush inner sink."""
        await self._inner.flush()
    
    async def close(self) -> None:
        """Close inner sink."""
        await self._inner.close()
    
    @property
    def name(self) -> str:
        return f"Filtered({self._inner.name})"


# =============================================================================
# EVENT LOGGER
# =============================================================================


class EventLogger:
    """
    Central event logger for agent observability.
    
    Provides a high-level API for logging agent events with support
    for multiple sinks and automatic context management.
    
    Usage:
        logger = EventLogger(session_id="sess-123")
        logger.add_sink(JSONLFileSink(Path("events.jsonl")))
        
        await logger.log_lifecycle_start(goal="Analyze data")
        await logger.log_cognitive_phase("think", ...)
        await logger.log_lifecycle_end(status="success")
        
        await logger.close()
    
    Context Manager:
        async with EventLogger(session_id="sess-123") as logger:
            logger.add_sink(JSONLFileSink(Path("events.jsonl")))
            await logger.log_lifecycle_start(goal="Analyze data")
            # Events automatically flushed on exit
    
    Attributes:
        session_id: Current session identifier
        execution_id: Current execution identifier
        sinks: List of event sinks
    """
    
    def __init__(
        self,
        session_id: str,
        *,
        execution_id: Optional[str] = None,
        sinks: Optional[List[EventSink]] = None,
        default_tags: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize event logger.
        
        Args:
            session_id: Session identifier
            execution_id: Execution identifier (auto-generated if None)
            sinks: Initial list of sinks
            default_tags: Tags to add to all events
        """
        self.session_id = session_id
        self.execution_id = execution_id or f"exec-{uuid4().hex[:12]}"
        self.sinks: List[EventSink] = sinks or []
        self.default_tags = default_tags or []
        
        # State tracking
        self._iteration: int = 0
        self._phase: Optional[str] = None
        self._correlation_id: Optional[str] = None
        self._parent_event_id: Optional[str] = None
        self._event_stack: List[str] = []  # For nested events
        
        # Statistics
        self._event_count: int = 0
        self._error_count: int = 0
        
        self._logger = logging.getLogger(__name__)
    
    async def __aenter__(self) -> "EventLogger":
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - flush and close."""
        await self.close()
    
    def add_sink(self, sink: EventSink) -> None:
        """
        Add an event sink.
        
        Args:
            sink: Sink to add
        """
        self.sinks.append(sink)
    
    def remove_sink(self, sink: EventSink) -> bool:
        """
        Remove an event sink.
        
        Args:
            sink: Sink to remove
            
        Returns:
            True if sink was removed
        """
        try:
            self.sinks.remove(sink)
            return True
        except ValueError:
            return False
    
    def set_iteration(self, iteration: int) -> None:
        """
        Set current iteration number.
        
        Args:
            iteration: Iteration number
        """
        self._iteration = iteration
    
    def set_phase(self, phase: Optional[str]) -> None:
        """
        Set current cognitive phase.
        
        Args:
            phase: Phase name
        """
        self._phase = phase
    
    def set_correlation_id(self, correlation_id: Optional[str]) -> None:
        """
        Set correlation ID for related events.
        
        Args:
            correlation_id: Correlation identifier
        """
        self._correlation_id = correlation_id
    
    @asynccontextmanager
    async def event_scope(
        self,
        parent_event: Optional[AgentEvent] = None,
    ) -> AsyncIterator[None]:
        """
        Context manager for nested events.
        
        Events logged within this scope will have their parent_event_id
        set to the provided parent event.
        
        Args:
            parent_event: Parent event for scope
            
        Yields:
            None
        """
        old_parent = self._parent_event_id
        if parent_event:
            self._parent_event_id = parent_event.event_id
            self._event_stack.append(parent_event.event_id)
        try:
            yield
        finally:
            self._parent_event_id = old_parent
            if parent_event:
                self._event_stack.pop()
    
    async def log(self, event: AgentEvent) -> AgentEvent:
        """
        Log an event to all sinks.
        
        Args:
            event: Event to log
            
        Returns:
            The logged event (with any modifications)
        """
        # Apply defaults
        if event.execution_id is None:
            event.execution_id = self.execution_id
        if event.iteration is None and self._iteration > 0:
            event.iteration = self._iteration
        if event.phase is None and self._phase:
            event.phase = self._phase
        if event.correlation_id is None and self._correlation_id:
            event.correlation_id = self._correlation_id
        if event.parent_event_id is None and self._parent_event_id:
            event.parent_event_id = self._parent_event_id
        
        # Add default tags
        for tag in self.default_tags:
            event.add_tag(tag)
        
        # Update statistics
        self._event_count += 1
        if event.category == EventCategory.ERROR:
            self._error_count += 1
        
        # Write to all sinks
        for sink in self.sinks:
            try:
                await sink.write(event)
            except Exception as e:
                self._logger.error(
                    f"Failed to write to sink {sink.name}: {e}"
                )
        
        return event
    
    async def flush(self) -> None:
        """Flush all sinks."""
        for sink in self.sinks:
            try:
                await sink.flush()
            except Exception as e:
                self._logger.error(
                    f"Failed to flush sink {sink.name}: {e}"
                )
    
    async def close(self) -> None:
        """Close all sinks."""
        await self.flush()
        for sink in self.sinks:
            try:
                await sink.close()
            except Exception as e:
                self._logger.error(
                    f"Failed to close sink {sink.name}: {e}"
                )
    
    # =========================================================================
    # CONVENIENCE METHODS - LIFECYCLE
    # =========================================================================
    
    async def log_lifecycle_start(
        self,
        goal: str,
        *,
        goal_complexity: Optional[str] = None,
        recommended_strategy: Optional[str] = None,
        **kwargs: Any,
    ) -> LifecycleEvent:
        """
        Log agent start event.
        
        Args:
            goal: Goal being processed
            goal_complexity: Classified complexity
            recommended_strategy: Recommended strategy
            **kwargs: Additional fields
            
        Returns:
            Created event
        """
        event = create_lifecycle_event(
            session_id=self.session_id,
            event_type=LifecycleEventType.AGENT_STARTED,
            goal=goal,
            goal_complexity=goal_complexity,
            recommended_strategy=recommended_strategy,
            **kwargs,
        )
        return await self.log(event)
    
    async def log_lifecycle_end(
        self,
        status: str,
        *,
        exit_reason: Optional[str] = None,
        total_iterations: Optional[int] = None,
        total_tokens: Optional[int] = None,
        duration_ms: Optional[float] = None,
        **kwargs: Any,
    ) -> LifecycleEvent:
        """
        Log agent completion event.
        
        Args:
            status: Final status (success, failure, cancelled)
            exit_reason: Reason for exit
            total_iterations: Total iterations
            total_tokens: Total tokens used
            duration_ms: Total duration
            **kwargs: Additional fields
            
        Returns:
            Created event
        """
        event_type = (
            LifecycleEventType.AGENT_COMPLETED
            if status == "success"
            else LifecycleEventType.AGENT_FAILED
        )
        
        event = create_lifecycle_event(
            session_id=self.session_id,
            event_type=event_type,
            final_status=status,
            exit_reason=exit_reason,
            total_iterations=total_iterations,
            total_tokens=total_tokens,
            **kwargs,
        )
        if duration_ms:
            event.duration_ms = duration_ms
        
        return await self.log(event)
    
    async def log_iteration_start(
        self,
        iteration: int,
        **kwargs: Any,
    ) -> LifecycleEvent:
        """
        Log iteration start.
        
        Args:
            iteration: Iteration number
            **kwargs: Additional fields
            
        Returns:
            Created event
        """
        self.set_iteration(iteration)
        event = create_lifecycle_event(
            session_id=self.session_id,
            event_type=LifecycleEventType.ITERATION_STARTED,
            **kwargs,
        )
        return await self.log(event)
    
    async def log_iteration_end(
        self,
        iteration: int,
        duration_ms: Optional[float] = None,
        **kwargs: Any,
    ) -> LifecycleEvent:
        """
        Log iteration completion.
        
        Args:
            iteration: Iteration number
            duration_ms: Iteration duration
            **kwargs: Additional fields
            
        Returns:
            Created event
        """
        event = create_lifecycle_event(
            session_id=self.session_id,
            event_type=LifecycleEventType.ITERATION_COMPLETED,
            **kwargs,
        )
        if duration_ms:
            event.duration_ms = duration_ms
        
        return await self.log(event)
    
    # =========================================================================
    # CONVENIENCE METHODS - COGNITIVE
    # =========================================================================
    
    async def log_cognitive_phase(
        self,
        phase: str,
        event_type: Union[CognitiveEventType, str] = CognitiveEventType.PHASE_COMPLETED,
        *,
        input_summary: Optional[str] = None,
        output_summary: Optional[str] = None,
        tokens_used: Optional[int] = None,
        reasoning: Optional[str] = None,
        confidence: Optional[float] = None,
        duration_ms: Optional[float] = None,
        **kwargs: Any,
    ) -> CognitiveEvent:
        """
        Log cognitive phase event.
        
        Args:
            phase: Phase name
            event_type: Type of cognitive event
            input_summary: Summary of input
            output_summary: Summary of output
            tokens_used: Tokens consumed
            reasoning: Reasoning process
            confidence: Confidence score
            duration_ms: Phase duration
            **kwargs: Additional fields
            
        Returns:
            Created event
        """
        self.set_phase(phase)
        event = create_cognitive_event(
            session_id=self.session_id,
            event_type=event_type,
            phase=phase,
            input_summary=input_summary,
            output_summary=output_summary,
            **kwargs,
        )
        if tokens_used:
            event.tokens_used = tokens_used
        if reasoning:
            event.reasoning = reasoning
        if confidence:
            event.confidence = confidence
        if duration_ms:
            event.duration_ms = duration_ms
        
        return await self.log(event)
    
    # =========================================================================
    # CONVENIENCE METHODS - ACTIVITY
    # =========================================================================
    
    async def log_activity(
        self,
        activity_name: str,
        activity_input: Optional[Dict[str, Any]] = None,
        activity_output: Optional[Any] = None,
        *,
        success: bool = True,
        error_message: Optional[str] = None,
        duration_ms: Optional[float] = None,
        event_type: Union[ActivityEventType, str] = ActivityEventType.ACTIVITY_COMPLETED,
        **kwargs: Any,
    ) -> ActivityEvent:
        """
        Log activity execution.
        
        Args:
            activity_name: Name of activity
            activity_input: Input parameters
            activity_output: Output result
            success: Whether succeeded
            error_message: Error if failed
            duration_ms: Execution duration
            event_type: Event type
            **kwargs: Additional fields
            
        Returns:
            Created event
        """
        event = create_activity_event(
            session_id=self.session_id,
            event_type=event_type,
            activity_name=activity_name,
            activity_input=activity_input,
            success=success,
            **kwargs,
        )
        if activity_output:
            event.activity_output = activity_output
        if error_message:
            event.error_message = error_message
        if duration_ms:
            event.duration_ms = duration_ms
        
        return await self.log(event)
    
    # =========================================================================
    # CONVENIENCE METHODS - ERROR
    # =========================================================================
    
    async def log_error(
        self,
        error_type: str,
        error_message: str,
        *,
        stack_trace: Optional[str] = None,
        recoverable: bool = True,
        severity: EventSeverity = EventSeverity.ERROR,
        source_component: Optional[str] = None,
        **kwargs: Any,
    ) -> ErrorEvent:
        """
        Log an error event.
        
        Args:
            error_type: Type of error
            error_message: Error message
            stack_trace: Stack trace
            recoverable: Is recoverable
            severity: Error severity
            source_component: Source component
            **kwargs: Additional fields
            
        Returns:
            Created event
        """
        event = create_error_event(
            session_id=self.session_id,
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack_trace,
            recoverable=recoverable,
            severity=severity,
            **kwargs,
        )
        if source_component:
            event.source_component = source_component
        
        return await self.log(event)
    
    # =========================================================================
    # CONVENIENCE METHODS - METRICS
    # =========================================================================
    
    async def log_metric(
        self,
        metric_name: str,
        metric_value: Union[int, float],
        *,
        metric_unit: Optional[str] = None,
        metric_type: Optional[str] = None,
        **kwargs: Any,
    ) -> MetricEvent:
        """
        Log a metric event.
        
        Args:
            metric_name: Name of metric
            metric_value: Metric value
            metric_unit: Unit of measurement
            metric_type: Type of metric
            **kwargs: Additional fields
            
        Returns:
            Created event
        """
        event = create_metric_event(
            session_id=self.session_id,
            metric_name=metric_name,
            metric_value=metric_value,
            metric_unit=metric_unit,
            **kwargs,
        )
        if metric_type:
            event.metric_type = metric_type
        
        return await self.log(event)
    
    # =========================================================================
    # CONVENIENCE METHODS - HITL
    # =========================================================================
    
    async def log_hitl(
        self,
        event_type: Union[HITLEventType, str],
        request_id: str,
        action_type: str,
        risk_level: str,
        approval_status: str,
        *,
        timeout_occurred: bool = False,
        responder_id: Optional[str] = None,
        feedback: Optional[str] = None,
        **kwargs: Any,
    ) -> HITLEvent:
        """
        Log HITL event.
        
        Args:
            event_type: HITL event type
            request_id: Request ID
            action_type: Action type
            risk_level: Risk level
            approval_status: Approval status
            timeout_occurred: If timed out
            responder_id: Who responded
            feedback: Response feedback
            **kwargs: Additional fields
            
        Returns:
            Created event
        """
        event = create_hitl_event(
            session_id=self.session_id,
            event_type=event_type,
            request_id=request_id,
            action_type=action_type,
            risk_level=risk_level,
            approval_status=approval_status,
            **kwargs,
        )
        event.timeout_occurred = timeout_occurred
        if responder_id:
            event.responder_id = responder_id
        if feedback:
            event.feedback = feedback
        
        return await self.log(event)
    
    # =========================================================================
    # CONVENIENCE METHODS - SANDBOX
    # =========================================================================
    
    async def log_sandbox(
        self,
        event_type: Union[SandboxEventType, str],
        sandbox_type: str,
        operation: str,
        *,
        sandbox_id: Optional[str] = None,
        image: Optional[str] = None,
        exit_code: Optional[int] = None,
        duration_ms: Optional[float] = None,
        **kwargs: Any,
    ) -> SandboxEvent:
        """
        Log sandbox event.
        
        Args:
            event_type: Sandbox event type
            sandbox_type: Type of sandbox
            operation: Operation performed
            sandbox_id: Sandbox ID
            image: Image used
            exit_code: Exit code
            duration_ms: Duration
            **kwargs: Additional fields
            
        Returns:
            Created event
        """
        event = create_sandbox_event(
            session_id=self.session_id,
            event_type=event_type,
            sandbox_type=sandbox_type,
            operation=operation,
            sandbox_id=sandbox_id,
            image=image,
            **kwargs,
        )
        if exit_code is not None:
            event.exit_code = exit_code
        if duration_ms:
            event.duration_ms = duration_ms
        
        return await self.log(event)
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    @property
    def event_count(self) -> int:
        """Get total event count."""
        return self._event_count
    
    @property
    def error_count(self) -> int:
        """Get error event count."""
        return self._error_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get logger statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "session_id": self.session_id,
            "execution_id": self.execution_id,
            "event_count": self._event_count,
            "error_count": self._error_count,
            "current_iteration": self._iteration,
            "current_phase": self._phase,
            "sink_count": len(self.sinks),
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_event_logger(
    session_id: str,
    *,
    log_path: Optional[Union[str, Path]] = None,
    in_memory: bool = False,
    execution_id: Optional[str] = None,
    default_tags: Optional[List[str]] = None,
) -> EventLogger:
    """
    Factory function to create an EventLogger with common configurations.
    
    Args:
        session_id: Session identifier
        log_path: Path to log file (creates JSONLFileSink)
        in_memory: Add InMemorySink for testing
        execution_id: Execution identifier
        default_tags: Default tags for all events
        
    Returns:
        Configured EventLogger
    """
    sinks: List[EventSink] = []
    
    if log_path:
        sinks.append(JSONLFileSink(Path(log_path)))
    
    if in_memory:
        sinks.append(InMemorySink())
    
    return EventLogger(
        session_id=session_id,
        execution_id=execution_id,
        sinks=sinks,
        default_tags=default_tags,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Sinks
    "EventSink",
    "JSONLFileSink",
    "InMemorySink",
    "CallbackSink",
    "FilteredSink",
    # Logger
    "EventLogger",
    "create_event_logger",
]
