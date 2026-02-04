# src/llmcore/agents/observability/events.py
"""
Structured Event Schema for Agent Observability.

This module defines the core event types used throughout the LLMCore agentic
system for comprehensive logging, debugging, and replay capabilities.

Event Categories:
    - LIFECYCLE: Agent start/stop, initialization, teardown
    - COGNITIVE: Thinking phases (perceive, think, plan, act, reflect, etc.)
    - ACTIVITY: Tool/activity invocations and results
    - MEMORY: Memory operations (read, write, update)
    - HITL: Human-in-the-loop approval events
    - ERROR: Errors, failures, and exceptions
    - METRIC: Performance and timing metrics
    - SANDBOX: Container/VM sandbox events
    - RAG: Retrieval-augmented generation events

Architecture:
    All events inherit from AgentEvent, which provides common fields:
    - Unique event_id for tracing
    - Timestamps with microsecond precision
    - Session and execution context
    - Parent/child relationships for nesting
    - Correlation IDs for cross-event tracking

Usage:
    >>> from llmcore.agents.observability.events import (
    ...     LifecycleEvent,
    ...     CognitiveEvent,
    ...     ActivityEvent,
    ...     EventCategory,
    ...     EventSeverity,
    ... )
    >>>
    >>> # Create a lifecycle event
    >>> event = LifecycleEvent(
    ...     session_id="sess-123",
    ...     event_type="agent_started",
    ...     goal="Analyze data",
    ... )
    >>>
    >>> # Create a cognitive phase event
    >>> event = CognitiveEvent(
    ...     session_id="sess-123",
    ...     event_type="phase_completed",
    ...     phase="think",
    ...     input_summary="User query about data",
    ...     output_summary="Determined to use pandas",
    ... )

References:
    - Master Plan: Section 26 (Structured Event Logging)
    - LLMCORE_AGENTIC_SYSTEM_MASTER_PLAN_G3.md: Section 26
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, UTC
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class EventCategory(str, Enum):
    """
    Categories of observable events.

    Each category groups related events for filtering and analysis:
    - LIFECYCLE: Agent initialization, execution, termination
    - COGNITIVE: All cognitive processing phases
    - ACTIVITY: Tool and activity executions
    - MEMORY: Memory system operations
    - HITL: Human approval workflow events
    - ERROR: All error and exception events
    - METRIC: Performance measurements
    - SANDBOX: Container/VM sandbox events
    - RAG: Retrieval-augmented generation events
    """

    LIFECYCLE = "lifecycle"
    COGNITIVE = "cognitive"
    ACTIVITY = "activity"
    MEMORY = "memory"
    HITL = "hitl"
    ERROR = "error"
    METRIC = "metric"
    SANDBOX = "sandbox"
    RAG = "rag"


class EventSeverity(str, Enum):
    """
    Event severity levels following standard logging conventions.

    DEBUG: Detailed diagnostic information
    INFO: Normal operational events
    WARNING: Unexpected but recoverable situations
    ERROR: Errors that prevent specific operations
    CRITICAL: Severe errors that may halt execution
    """

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LifecycleEventType(str, Enum):
    """Specific lifecycle event types."""

    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    AGENT_CANCELLED = "agent_cancelled"
    ITERATION_STARTED = "iteration_started"
    ITERATION_COMPLETED = "iteration_completed"
    GOAL_CLASSIFIED = "goal_classified"
    FAST_PATH_TRIGGERED = "fast_path_triggered"


class CognitiveEventType(str, Enum):
    """Specific cognitive phase event types."""

    PHASE_STARTED = "phase_started"
    PHASE_COMPLETED = "phase_completed"
    PHASE_FAILED = "phase_failed"
    PHASE_SKIPPED = "phase_skipped"
    DECISION_MADE = "decision_made"
    REASONING_STEP = "reasoning_step"


class ActivityEventType(str, Enum):
    """Specific activity/tool event types."""

    ACTIVITY_STARTED = "activity_started"
    ACTIVITY_COMPLETED = "activity_completed"
    ACTIVITY_FAILED = "activity_failed"
    ACTIVITY_TIMEOUT = "activity_timeout"
    ACTIVITY_RETRIED = "activity_retried"
    ACTIVITY_CANCELLED = "activity_cancelled"


class HITLEventType(str, Enum):
    """Specific HITL event types."""

    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    APPROVAL_TIMEOUT = "approval_timeout"
    APPROVAL_MODIFIED = "approval_modified"
    SCOPE_GRANTED = "scope_granted"
    SCOPE_REVOKED = "scope_revoked"


class ErrorEventType(str, Enum):
    """Specific error event types."""

    EXCEPTION = "exception"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT_ERROR = "timeout_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    API_ERROR = "api_error"
    SANDBOX_ERROR = "sandbox_error"
    RECOVERY_ATTEMPTED = "recovery_attempted"
    RECOVERY_FAILED = "recovery_failed"


class MetricEventType(str, Enum):
    """Specific metric event types."""

    LATENCY = "latency"
    TOKEN_USAGE = "token_usage"
    COST = "cost"
    THROUGHPUT = "throughput"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"


class MemoryEventType(str, Enum):
    """Specific memory event types."""

    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    MEMORY_UPDATE = "memory_update"
    MEMORY_DELETE = "memory_delete"
    MEMORY_SEARCH = "memory_search"


class SandboxEventType(str, Enum):
    """Specific sandbox event types."""

    SANDBOX_CREATED = "sandbox_created"
    SANDBOX_STARTED = "sandbox_started"
    SANDBOX_STOPPED = "sandbox_stopped"
    SANDBOX_DESTROYED = "sandbox_destroyed"
    CODE_EXECUTED = "code_executed"
    FILE_OPERATION = "file_operation"


class RAGEventType(str, Enum):
    """Specific RAG event types."""

    QUERY_STARTED = "query_started"
    QUERY_COMPLETED = "query_completed"
    DOCUMENTS_RETRIEVED = "documents_retrieved"
    CONTEXT_ASSEMBLED = "context_assembled"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _generate_event_id() -> str:
    """Generate a unique event identifier."""
    return f"evt-{uuid4().hex[:16]}"


def _utc_now() -> datetime:
    """Get current UTC timestamp with timezone awareness."""
    return datetime.now(UTC)


# =============================================================================
# BASE EVENT MODEL
# =============================================================================


class AgentEvent(BaseModel):
    """
    Base class for all agent events.

    Provides common fields for event tracking, correlation, and analysis.
    All specialized event types inherit from this base class.

    Attributes:
        event_id: Unique identifier for this event (auto-generated)
        timestamp: When the event occurred (auto-generated, UTC)
        session_id: Agent session identifier
        execution_id: Specific execution/run identifier
        category: Event category for filtering
        event_type: Specific event type within category
        severity: Event severity level
        phase: Current cognitive phase (if applicable)
        iteration: Current iteration number (if applicable)
        data: Additional event-specific payload
        duration_ms: Duration in milliseconds (for timed events)
        parent_event_id: Parent event for nested events
        correlation_id: ID for correlating related events
        tags: Additional tags for filtering/categorization
    """

    model_config = ConfigDict(
        extra="allow",
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "event_id": "evt-a1b2c3d4e5f67890",
                "timestamp": "2026-01-23T12:00:00Z",
                "session_id": "sess-123456",
                "category": "lifecycle",
                "event_type": "agent_started",
                "severity": "info",
            }
        },
    )

    # Identity
    event_id: str = Field(default_factory=_generate_event_id, description="Unique event identifier")
    timestamp: datetime = Field(default_factory=_utc_now, description="Event timestamp (UTC)")

    # Context
    session_id: str = Field(..., description="Agent session ID")
    execution_id: str | None = Field(None, description="Specific execution/run ID")

    # Classification
    category: EventCategory = Field(..., description="Event category")
    event_type: str = Field(..., description="Specific event type")
    severity: EventSeverity = Field(default=EventSeverity.INFO, description="Event severity")

    # State
    phase: str | None = Field(None, description="Current cognitive phase")
    iteration: int | None = Field(None, ge=0, description="Current iteration number")

    # Payload
    data: dict[str, Any] = Field(default_factory=dict, description="Event-specific data payload")

    # Timing
    duration_ms: float | None = Field(None, ge=0, description="Duration in milliseconds")

    # Relationships
    parent_event_id: str | None = Field(None, description="Parent event ID for nesting")
    correlation_id: str | None = Field(None, description="Correlation ID for related events")

    # Metadata
    tags: list[str] = Field(default_factory=list, description="Tags for filtering")

    def with_duration(self, start_time: datetime) -> AgentEvent:
        """
        Set duration based on start time.

        Args:
            start_time: When the operation started

        Returns:
            Self with duration_ms set
        """
        delta = _utc_now() - start_time
        self.duration_ms = delta.total_seconds() * 1000
        return self

    def with_parent(self, parent_id: str) -> AgentEvent:
        """
        Set parent event ID.

        Args:
            parent_id: Parent event identifier

        Returns:
            Self with parent_event_id set
        """
        self.parent_event_id = parent_id
        return self

    def with_correlation(self, correlation_id: str) -> AgentEvent:
        """
        Set correlation ID.

        Args:
            correlation_id: Correlation identifier

        Returns:
            Self with correlation_id set
        """
        self.correlation_id = correlation_id
        return self

    def add_tag(self, tag: str) -> AgentEvent:
        """
        Add a tag to the event.

        Args:
            tag: Tag to add

        Returns:
            Self with tag added
        """
        if tag not in self.tags:
            self.tags.append(tag)
        return self

    def to_dict(self) -> dict[str, Any]:
        """
        Convert event to dictionary for serialization.

        Returns:
            Dictionary representation of the event
        """
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        """
        Convert event to JSON string.

        Returns:
            JSON string representation
        """
        return self.model_dump_json()


# =============================================================================
# SPECIALIZED EVENT TYPES
# =============================================================================


class LifecycleEvent(AgentEvent):
    """
    Agent lifecycle events (start, stop, complete, fail).

    Tracks the overall lifecycle of an agent execution including
    initialization, goal processing, and termination.

    Attributes:
        goal: The goal being processed
        final_status: Final execution status (for completion events)
        total_iterations: Number of iterations performed
        total_tokens: Total tokens consumed
        exit_reason: Why the agent exited
        goal_complexity: Classified goal complexity
        recommended_strategy: Recommended execution strategy
    """

    category: EventCategory = Field(default=EventCategory.LIFECYCLE, frozen=True)

    # Lifecycle-specific fields
    goal: str | None = Field(None, description="Goal being processed")
    final_status: str | None = Field(None, description="Final status")
    total_iterations: int | None = Field(None, ge=0, description="Total iterations")
    total_tokens: int | None = Field(None, ge=0, description="Total tokens used")
    exit_reason: str | None = Field(None, description="Exit reason")

    # Complexity classification
    goal_complexity: str | None = Field(
        None, description="Classified goal complexity (trivial, simple, complex)"
    )
    recommended_strategy: str | None = Field(None, description="Recommended execution strategy")


class CognitiveEvent(AgentEvent):
    """
    Cognitive phase events (perceive, think, plan, act, reflect, etc.).

    Tracks each cognitive processing phase in the agent's reasoning cycle.
    Captures inputs, outputs, and reasoning for each phase.

    Attributes:
        input_summary: Summary of phase input
        output_summary: Summary of phase output
        tokens_used: Tokens consumed in this phase
        reasoning: Explicit reasoning steps taken
        confidence: Confidence score for decisions made
        phase_name: Specific phase name
        phase_order: Order in cognitive cycle
        decisions: Decisions made in this phase
    """

    category: EventCategory = Field(default=EventCategory.COGNITIVE, frozen=True)

    # Cognitive-specific fields
    input_summary: str | None = Field(None, description="Summary of input to phase")
    output_summary: str | None = Field(None, description="Summary of phase output")
    tokens_used: int | None = Field(None, ge=0, description="Tokens used in phase")
    reasoning: str | None = Field(None, description="Reasoning or thought process")
    confidence: float | None = Field(None, ge=0, le=1, description="Confidence score (0-1)")

    # Phase metadata
    phase_name: str | None = Field(None, description="Specific phase name")
    phase_order: int | None = Field(None, ge=0, description="Order in cognitive cycle")

    # Decisions made
    decisions: list[str] = Field(default_factory=list, description="Decisions made in this phase")


class ActivityEvent(AgentEvent):
    """
    Activity/tool execution events.

    Tracks tool and activity invocations including their inputs,
    outputs, and execution status.

    Attributes:
        activity_name: Name of the activity/tool
        activity_type: Type/category of activity
        activity_input: Input parameters
        activity_output: Output/result
        success: Whether execution succeeded
        error_message: Error message if failed
        retry_count: Number of retries attempted
        sandbox_type: Sandbox type used
        container_id: Container ID if applicable
    """

    category: EventCategory = Field(default=EventCategory.ACTIVITY, frozen=True)

    # Activity identification
    activity_name: str = Field(..., description="Activity/tool name")
    activity_type: str | None = Field(None, description="Activity type/category")

    # Execution details
    activity_input: dict[str, Any] = Field(default_factory=dict, description="Input parameters")
    activity_output: Any | None = Field(None, description="Output result")
    activity_output_truncated: bool = Field(
        default=False, description="Whether output was truncated"
    )

    # Status
    success: bool = Field(default=True, description="Execution success")
    error_message: str | None = Field(None, description="Error if failed")
    error_type: str | None = Field(None, description="Error type/class")

    # Retry information
    retry_count: int = Field(default=0, ge=0, description="Retry attempts")
    max_retries: int | None = Field(None, ge=0, description="Max retries")

    # Sandbox context
    sandbox_type: str | None = Field(None, description="Sandbox type (docker, vm, etc.)")
    container_id: str | None = Field(None, description="Container ID")


class MemoryEvent(AgentEvent):
    """
    Memory operation events.

    Tracks reads, writes, and updates to the agent's memory systems.

    Attributes:
        operation: Memory operation type
        memory_type: Type of memory
        key: Memory key or identifier
        value_summary: Summary of stored/retrieved value
        items_affected: Number of items affected
    """

    category: EventCategory = Field(default=EventCategory.MEMORY, frozen=True)

    # Operation details
    operation: str = Field(..., description="Operation type (read, write, update, delete)")
    memory_type: str = Field(..., description="Memory type (working, episodic, semantic, etc.)")

    # Content
    key: str | None = Field(None, description="Memory key")
    value_summary: str | None = Field(None, description="Summary of value")

    # Metrics
    items_affected: int = Field(default=1, ge=0, description="Number of items affected")


class HITLEvent(AgentEvent):
    """
    Human-in-the-loop events.

    Tracks all HITL approval workflow events including requests,
    responses, and scope management.

    Attributes:
        request_id: HITL request identifier
        action_type: Type of action requiring approval
        risk_level: Assessed risk level
        approval_status: Current approval status
        timeout_occurred: Whether approval timed out
        responder_id: ID of the human who responded
        feedback: Human feedback/comments
        scope_granted: Approval scope if granted
    """

    category: EventCategory = Field(default=EventCategory.HITL, frozen=True)

    # Request identification
    request_id: str = Field(..., description="HITL request ID")
    action_type: str = Field(..., description="Action requiring approval")

    # Risk assessment
    risk_level: str = Field(..., description="Risk level (safe, low, medium, high, critical)")
    risk_factors: list[str] = Field(default_factory=list, description="Contributing risk factors")

    # Response
    approval_status: str = Field(..., description="Approval status")
    timeout_occurred: bool = Field(default=False, description="Whether approval timed out")
    timeout_seconds: float | None = Field(None, ge=0, description="Timeout duration")

    # Human response
    responder_id: str | None = Field(None, description="Responder ID")
    feedback: str | None = Field(None, description="Human feedback")

    # Scope
    scope_granted: str | None = Field(None, description="Approval scope granted")
    scope_expiration: datetime | None = Field(None, description="When scope expires")


class ErrorEvent(AgentEvent):
    """
    Error and exception events.

    Tracks all errors, exceptions, and failure conditions.

    Attributes:
        error_type: Type/class of error
        error_message: Error message
        error_code: Error code
        stack_trace: Stack trace if available
        recoverable: Whether the error is recoverable
        recovery_action: Recovery action taken
        source_component: Component where error occurred
    """

    category: EventCategory = Field(default=EventCategory.ERROR, frozen=True)
    severity: EventSeverity = Field(default=EventSeverity.ERROR, description="Error severity")

    # Error details
    error_type: str = Field(..., description="Error type/class")
    error_message: str = Field(..., description="Error message")
    error_code: str | None = Field(None, description="Error code")

    # Debug info
    stack_trace: str | None = Field(None, description="Stack trace")
    source_component: str | None = Field(None, description="Source component")
    source_file: str | None = Field(None, description="Source file")
    source_line: int | None = Field(None, ge=0, description="Source line")

    # Recovery
    recoverable: bool = Field(default=True, description="Is recoverable")
    recovery_action: str | None = Field(None, description="Recovery action taken")
    recovery_successful: bool | None = Field(None, description="Whether recovery succeeded")

    # Context
    context_snapshot: dict[str, Any] = Field(
        default_factory=dict, description="Context at time of error"
    )


class MetricEvent(AgentEvent):
    """
    Performance and metric events.

    Tracks performance measurements, resource usage, and operational metrics.

    Attributes:
        metric_name: Name of the metric
        metric_value: Metric value (numeric)
        metric_unit: Unit of measurement
        metric_type: Type (counter, gauge, histogram)
        aggregation: Aggregation type
    """

    category: EventCategory = Field(default=EventCategory.METRIC, frozen=True)

    # Metric identification
    metric_name: str = Field(..., description="Metric name")
    metric_type: str | None = Field(
        None, description="Metric type (counter, gauge, histogram, etc.)"
    )

    # Value
    metric_value: int | float = Field(..., description="Metric value")
    metric_unit: str | None = Field(None, description="Unit (ms, bytes, etc.)")

    # Aggregation
    aggregation: str | None = Field(
        None, description="Aggregation type (sum, avg, max, min, etc.)"
    )

    # Additional values for distributions
    min_value: float | None = Field(None, description="Minimum value")
    max_value: float | None = Field(None, description="Maximum value")
    avg_value: float | None = Field(None, description="Average value")
    p50_value: float | None = Field(None, description="50th percentile")
    p95_value: float | None = Field(None, description="95th percentile")
    p99_value: float | None = Field(None, description="99th percentile")
    sample_count: int | None = Field(None, ge=0, description="Number of samples")


class SandboxEvent(AgentEvent):
    """
    Sandbox (Docker/VM) events.

    Tracks sandbox lifecycle and operations for code execution environments.

    Attributes:
        sandbox_type: Type of sandbox (docker, vm, etc.)
        sandbox_id: Sandbox identifier
        image: Container/VM image used
        operation: Sandbox operation
        exit_code: Process exit code
    """

    category: EventCategory = Field(default=EventCategory.SANDBOX, frozen=True)

    # Sandbox identification
    sandbox_type: str = Field(..., description="Sandbox type (docker, vm, ephemeral)")
    sandbox_id: str | None = Field(None, description="Sandbox ID")

    # Configuration
    image: str | None = Field(None, description="Image used")
    access_mode: str | None = Field(None, description="Access mode (restricted, full)")

    # Operation
    operation: str = Field(..., description="Operation (create, start, execute, stop, destroy)")

    # Execution results
    exit_code: int | None = Field(None, description="Exit code")
    stdout_length: int | None = Field(None, ge=0, description="Stdout bytes")
    stderr_length: int | None = Field(None, ge=0, description="Stderr bytes")

    # Resource usage
    memory_used_mb: float | None = Field(None, ge=0, description="Memory used (MB)")
    cpu_time_ms: float | None = Field(None, ge=0, description="CPU time (ms)")


class RAGEvent(AgentEvent):
    """
    RAG (Retrieval-Augmented Generation) events.

    Tracks retrieval operations and their results.

    Attributes:
        query: Search query
        query_type: Query type
        source: RAG source
        num_results: Number of results
        top_score: Highest similarity score
        documents_used: IDs of documents used
    """

    category: EventCategory = Field(default=EventCategory.RAG, frozen=True)

    # Query
    query: str = Field(..., description="Search query")
    query_type: str | None = Field(None, description="Query type (semantic, keyword, hybrid)")

    # Source
    source: str = Field(..., description="RAG source (vector_store, documents, etc.)")
    source_id: str | None = Field(None, description="Specific source ID")

    # Results
    num_results: int = Field(default=0, ge=0, description="Results count")
    results_truncated: bool = Field(default=False, description="Whether results were truncated")

    # Scores
    top_score: float | None = Field(None, description="Highest similarity score")
    avg_score: float | None = Field(None, description="Average similarity score")
    threshold_score: float | None = Field(None, description="Score threshold used")

    # Documents
    documents_used: list[str] = Field(default_factory=list, description="Document IDs used")


# =============================================================================
# EVENT FACTORY FUNCTIONS
# =============================================================================


def create_lifecycle_event(
    session_id: str,
    event_type: LifecycleEventType | str,
    *,
    execution_id: str | None = None,
    goal: str | None = None,
    severity: EventSeverity = EventSeverity.INFO,
    **kwargs: Any,
) -> LifecycleEvent:
    """
    Factory function to create lifecycle events.

    Args:
        session_id: Agent session ID
        event_type: Specific lifecycle event type
        execution_id: Execution ID
        goal: Goal being processed
        severity: Event severity
        **kwargs: Additional event fields

    Returns:
        LifecycleEvent instance
    """
    if isinstance(event_type, LifecycleEventType):
        event_type = event_type.value

    return LifecycleEvent(
        session_id=session_id,
        execution_id=execution_id,
        event_type=event_type,
        goal=goal,
        severity=severity,
        **kwargs,
    )


def create_cognitive_event(
    session_id: str,
    event_type: CognitiveEventType | str,
    phase: str,
    *,
    execution_id: str | None = None,
    iteration: int | None = None,
    input_summary: str | None = None,
    output_summary: str | None = None,
    **kwargs: Any,
) -> CognitiveEvent:
    """
    Factory function to create cognitive phase events.

    Args:
        session_id: Agent session ID
        event_type: Specific cognitive event type
        phase: Cognitive phase name
        execution_id: Execution ID
        iteration: Current iteration
        input_summary: Input summary
        output_summary: Output summary
        **kwargs: Additional event fields

    Returns:
        CognitiveEvent instance
    """
    if isinstance(event_type, CognitiveEventType):
        event_type = event_type.value

    return CognitiveEvent(
        session_id=session_id,
        execution_id=execution_id,
        event_type=event_type,
        phase=phase,
        phase_name=phase,
        iteration=iteration,
        input_summary=input_summary,
        output_summary=output_summary,
        **kwargs,
    )


def create_activity_event(
    session_id: str,
    event_type: ActivityEventType | str,
    activity_name: str,
    *,
    execution_id: str | None = None,
    activity_input: dict[str, Any] | None = None,
    success: bool = True,
    **kwargs: Any,
) -> ActivityEvent:
    """
    Factory function to create activity events.

    Args:
        session_id: Agent session ID
        event_type: Specific activity event type
        activity_name: Name of the activity
        execution_id: Execution ID
        activity_input: Activity input parameters
        success: Whether activity succeeded
        **kwargs: Additional event fields

    Returns:
        ActivityEvent instance
    """
    if isinstance(event_type, ActivityEventType):
        event_type = event_type.value

    return ActivityEvent(
        session_id=session_id,
        execution_id=execution_id,
        event_type=event_type,
        activity_name=activity_name,
        activity_input=activity_input or {},
        success=success,
        **kwargs,
    )


def create_error_event(
    session_id: str,
    error_type: str,
    error_message: str,
    *,
    execution_id: str | None = None,
    event_type: ErrorEventType | str = ErrorEventType.EXCEPTION,
    severity: EventSeverity = EventSeverity.ERROR,
    stack_trace: str | None = None,
    recoverable: bool = True,
    **kwargs: Any,
) -> ErrorEvent:
    """
    Factory function to create error events.

    Args:
        session_id: Agent session ID
        error_type: Type/class of error
        error_message: Error message
        execution_id: Execution ID
        event_type: Specific error event type
        severity: Error severity
        stack_trace: Stack trace
        recoverable: Whether error is recoverable
        **kwargs: Additional event fields

    Returns:
        ErrorEvent instance
    """
    if isinstance(event_type, ErrorEventType):
        event_type = event_type.value

    return ErrorEvent(
        session_id=session_id,
        execution_id=execution_id,
        event_type=event_type,
        error_type=error_type,
        error_message=error_message,
        stack_trace=stack_trace,
        severity=severity,
        recoverable=recoverable,
        **kwargs,
    )


def create_metric_event(
    session_id: str,
    metric_name: str,
    metric_value: float,
    *,
    execution_id: str | None = None,
    event_type: MetricEventType | str = "measurement",
    metric_unit: str | None = None,
    **kwargs: Any,
) -> MetricEvent:
    """
    Factory function to create metric events.

    Args:
        session_id: Agent session ID
        metric_name: Name of the metric
        metric_value: Metric value
        execution_id: Execution ID
        event_type: Metric event type
        metric_unit: Unit of measurement
        **kwargs: Additional event fields

    Returns:
        MetricEvent instance
    """
    if isinstance(event_type, MetricEventType):
        event_type = event_type.value

    return MetricEvent(
        session_id=session_id,
        execution_id=execution_id,
        event_type=event_type,
        metric_name=metric_name,
        metric_value=metric_value,
        metric_unit=metric_unit,
        **kwargs,
    )


def create_hitl_event(
    session_id: str,
    event_type: HITLEventType | str,
    request_id: str,
    action_type: str,
    risk_level: str,
    approval_status: str,
    *,
    execution_id: str | None = None,
    **kwargs: Any,
) -> HITLEvent:
    """
    Factory function to create HITL events.

    Args:
        session_id: Agent session ID
        event_type: Specific HITL event type
        request_id: HITL request ID
        action_type: Action requiring approval
        risk_level: Risk level
        approval_status: Approval status
        execution_id: Execution ID
        **kwargs: Additional event fields

    Returns:
        HITLEvent instance
    """
    if isinstance(event_type, HITLEventType):
        event_type = event_type.value

    return HITLEvent(
        session_id=session_id,
        execution_id=execution_id,
        event_type=event_type,
        request_id=request_id,
        action_type=action_type,
        risk_level=risk_level,
        approval_status=approval_status,
        **kwargs,
    )


def create_sandbox_event(
    session_id: str,
    event_type: SandboxEventType | str,
    sandbox_type: str,
    operation: str,
    *,
    execution_id: str | None = None,
    sandbox_id: str | None = None,
    image: str | None = None,
    **kwargs: Any,
) -> SandboxEvent:
    """
    Factory function to create sandbox events.

    Args:
        session_id: Agent session ID
        event_type: Specific sandbox event type
        sandbox_type: Type of sandbox
        operation: Sandbox operation
        execution_id: Execution ID
        sandbox_id: Sandbox ID
        image: Image used
        **kwargs: Additional event fields

    Returns:
        SandboxEvent instance
    """
    if isinstance(event_type, SandboxEventType):
        event_type = event_type.value

    return SandboxEvent(
        session_id=session_id,
        execution_id=execution_id,
        event_type=event_type,
        sandbox_type=sandbox_type,
        operation=operation,
        sandbox_id=sandbox_id,
        image=image,
        **kwargs,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "EventCategory",
    "EventSeverity",
    "LifecycleEventType",
    "CognitiveEventType",
    "ActivityEventType",
    "HITLEventType",
    "ErrorEventType",
    "MetricEventType",
    "MemoryEventType",
    "SandboxEventType",
    "RAGEventType",
    # Base event
    "AgentEvent",
    # Specialized events
    "LifecycleEvent",
    "CognitiveEvent",
    "ActivityEvent",
    "MemoryEvent",
    "HITLEvent",
    "ErrorEvent",
    "MetricEvent",
    "SandboxEvent",
    "RAGEvent",
    # Factory functions
    "create_lifecycle_event",
    "create_cognitive_event",
    "create_activity_event",
    "create_error_event",
    "create_metric_event",
    "create_hitl_event",
    "create_sandbox_event",
]
