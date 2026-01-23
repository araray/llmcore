# src/llmcore/agents/observability/__init__.py
"""
Observability System for LLMCore Agents.

Provides comprehensive observability capabilities including:
- Structured event logging with multiple event types
- Flexible sink-based logging architecture
- Metrics collection and aggregation
- Execution replay for debugging and analysis

Components:
    - Events: Structured event types for all agent activities
    - Logger: Central event logger with multiple sink support
    - Metrics: Execution metrics collection and aggregation
    - Replay: Load and replay executions from event logs

Usage:
    >>> from llmcore.agents.observability import (
    ...     EventLogger,
    ...     MetricsCollector,
    ...     ExecutionReplay,
    ...     JSONLFileSink,
    ... )
    >>>
    >>> # Create logger
    >>> async with EventLogger(session_id="sess-123") as logger:
    ...     logger.add_sink(JSONLFileSink(Path("events.jsonl")))
    ...     
    ...     # Log lifecycle events
    ...     await logger.log_lifecycle_start(goal="Analyze data")
    ...     
    ...     # Log cognitive phases
    ...     await logger.log_cognitive_phase(
    ...         "think",
    ...         input_summary="User query",
    ...         output_summary="Determined approach",
    ...     )
    ...     
    ...     # Log activities
    ...     await logger.log_activity(
    ...         "execute_python",
    ...         {"code": "print('hello')"},
    ...         "hello",
    ...     )
    ...     
    ...     # Log completion
    ...     await logger.log_lifecycle_end(status="success")
    >>>
    >>> # Collect metrics
    >>> collector = MetricsCollector()
    >>> metrics = collector.start_execution("exec-123", goal="Analyze data")
    >>> metrics.record_iteration(duration_ms=150)
    >>> metrics.complete(success=True)
    >>> summary = collector.get_summary()
    >>>
    >>> # Replay execution
    >>> replay = ExecutionReplay.from_file("events.jsonl")
    >>> result = replay.replay("exec-123")
    >>> for step in result.timeline:
    ...     print(f"{step.timestamp}: {step.summary}")

References:
    - Master Plan: Sections 26-28 (Observability)
    - LLMCORE_AGENTIC_SYSTEM_MASTER_PLAN_G3.md: Part VIII
"""

from __future__ import annotations

# =============================================================================
# EVENTS
# =============================================================================

from .events import (
    # Enums
    EventCategory,
    EventSeverity,
    LifecycleEventType,
    CognitiveEventType,
    ActivityEventType,
    HITLEventType,
    ErrorEventType,
    MetricEventType,
    MemoryEventType,
    SandboxEventType,
    RAGEventType,
    # Base event
    AgentEvent,
    # Specialized events
    LifecycleEvent,
    CognitiveEvent,
    ActivityEvent,
    MemoryEvent,
    HITLEvent,
    ErrorEvent,
    MetricEvent,
    SandboxEvent,
    RAGEvent,
    # Factory functions
    create_lifecycle_event,
    create_cognitive_event,
    create_activity_event,
    create_error_event,
    create_metric_event,
    create_hitl_event,
    create_sandbox_event,
)

# =============================================================================
# LOGGER
# =============================================================================

from .logger import (
    # Sinks
    EventSink,
    JSONLFileSink,
    InMemorySink,
    CallbackSink,
    FilteredSink,
    # Logger
    EventLogger,
    create_event_logger,
)

# =============================================================================
# METRICS
# =============================================================================

from .metrics import (
    # Enums
    MetricType,
    ExecutionStatus,
    # Data classes
    IterationMetrics,
    LLMCallMetrics,
    ActivityMetrics,
    HITLMetrics,
    # Main classes
    ExecutionMetrics,
    MetricsCollector,
    # Pydantic models
    MetricsSummary,
    ExecutionSummary,
)

# =============================================================================
# REPLAY
# =============================================================================

from .replay import (
    # Data classes
    ReplayStep,
    ExecutionInfo,
    ReplayResult,
    # Main class
    ExecutionReplay,
    # Helpers
    parse_event,
    EVENT_CLASS_MAP,
    # Pydantic models
    ReplayStepModel,
    ReplayResultModel,
    ExecutionInfoModel,
)

# =============================================================================
# VERSION
# =============================================================================

__version__ = "1.0.0"

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Version
    "__version__",
    
    # === EVENTS ===
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
    
    # === LOGGER ===
    # Sinks
    "EventSink",
    "JSONLFileSink",
    "InMemorySink",
    "CallbackSink",
    "FilteredSink",
    # Logger
    "EventLogger",
    "create_event_logger",
    
    # === METRICS ===
    # Enums
    "MetricType",
    "ExecutionStatus",
    # Data classes
    "IterationMetrics",
    "LLMCallMetrics",
    "ActivityMetrics",
    "HITLMetrics",
    # Main classes
    "ExecutionMetrics",
    "MetricsCollector",
    # Pydantic models
    "MetricsSummary",
    "ExecutionSummary",
    
    # === REPLAY ===
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
