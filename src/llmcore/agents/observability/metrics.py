# src/llmcore/agents/observability/metrics.py
"""
Metrics Collector for Agent Observability.

Provides comprehensive metrics collection for agent executions including
latency, token usage, cost tracking, and success rates.

Architecture:
    MetricsCollector aggregates metrics from multiple executions,
    providing both real-time tracking and historical summaries.
    ExecutionMetrics captures metrics for a single execution.

Usage:
    >>> from llmcore.agents.observability.metrics import (
    ...     MetricsCollector,
    ...     ExecutionMetrics,
    ... )
    >>>
    >>> # Create collector
    >>> collector = MetricsCollector()
    >>>
    >>> # Start execution tracking
    >>> metrics = collector.start_execution("exec-123", goal="Analyze data")
    >>>
    >>> # Record metrics
    >>> metrics.record_iteration(duration_ms=150)
    >>> metrics.record_tokens(input=100, output=50)
    >>> metrics.record_llm_call(model="gpt-4", tokens=150, cost=0.005)
    >>> metrics.record_activity("execute_python", success=True, duration_ms=200)
    >>>
    >>> # End execution
    >>> metrics.complete(success=True)
    >>>
    >>> # Get summary
    >>> summary = collector.get_summary()

References:
    - Master Plan: Section 27 (Metrics & Monitoring)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from statistics import mean, stdev
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class MetricType(str, Enum):
    """Types of metrics."""

    COUNTER = "counter"  # Monotonically increasing
    GAUGE = "gauge"  # Point-in-time value
    HISTOGRAM = "histogram"  # Distribution of values
    SUMMARY = "summary"  # Statistical summary


class ExecutionStatus(str, Enum):
    """Execution completion status."""

    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class IterationMetrics:
    """Metrics for a single iteration."""

    iteration: int
    duration_ms: float
    phase_durations: dict[str, float] = field(default_factory=dict)
    tokens_used: int = 0
    activities_executed: int = 0
    errors_occurred: int = 0


@dataclass
class LLMCallMetrics:
    """Metrics for a single LLM call."""

    model: str
    tokens_input: int
    tokens_output: int
    duration_ms: float
    cost: float
    cache_hit: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ActivityMetrics:
    """Metrics for a single activity execution."""

    activity_name: str
    success: bool
    duration_ms: float
    retry_count: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class HITLMetrics:
    """Metrics for HITL interactions."""

    request_id: str
    action_type: str
    risk_level: str
    approved: bool
    wait_time_ms: float
    timeout_occurred: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


# =============================================================================
# EXECUTION METRICS
# =============================================================================


class ExecutionMetrics:
    """
    Metrics for a single agent execution.

    Tracks all metrics during an execution lifecycle including
    iterations, LLM calls, activities, and HITL interactions.

    Attributes:
        execution_id: Unique execution identifier
        session_id: Session identifier
        goal: The goal being executed
        start_time: When execution started
        end_time: When execution ended
        status: Execution status
    """

    def __init__(
        self,
        execution_id: str,
        session_id: str | None = None,
        goal: str | None = None,
    ) -> None:
        """
        Initialize execution metrics.

        Args:
            execution_id: Execution identifier
            session_id: Session identifier
            goal: Goal being executed
        """
        self.execution_id = execution_id
        self.session_id = session_id or f"sess-{uuid4().hex[:12]}"
        self.goal = goal

        # Timing
        self.start_time = datetime.now(UTC)
        self.end_time: datetime | None = None
        self._start_monotonic = time.monotonic()

        # Status
        self.status = ExecutionStatus.RUNNING
        self.exit_reason: str | None = None

        # Iterations
        self._iterations: list[IterationMetrics] = []
        self._current_iteration: int = 0
        self._iteration_start: float | None = None

        # LLM calls
        self._llm_calls: list[LLMCallMetrics] = []

        # Activities
        self._activities: list[ActivityMetrics] = []

        # HITL
        self._hitl_interactions: list[HITLMetrics] = []

        # Aggregates
        self._total_tokens_input: int = 0
        self._total_tokens_output: int = 0
        self._total_cost: float = 0.0
        self._error_count: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    # =========================================================================
    # RECORDING METHODS
    # =========================================================================

    def start_iteration(self) -> int:
        """
        Start a new iteration.

        Returns:
            Iteration number
        """
        self._current_iteration += 1
        self._iteration_start = time.monotonic()
        return self._current_iteration

    def end_iteration(
        self,
        phase_durations: dict[str, float] | None = None,
        tokens_used: int = 0,
        activities_executed: int = 0,
        errors_occurred: int = 0,
    ) -> IterationMetrics:
        """
        End current iteration and record metrics.

        Args:
            phase_durations: Duration of each phase
            tokens_used: Tokens used in iteration
            activities_executed: Activities executed
            errors_occurred: Errors that occurred

        Returns:
            Iteration metrics
        """
        if self._iteration_start is None:
            duration_ms = 0.0
        else:
            duration_ms = (time.monotonic() - self._iteration_start) * 1000

        metrics = IterationMetrics(
            iteration=self._current_iteration,
            duration_ms=duration_ms,
            phase_durations=phase_durations or {},
            tokens_used=tokens_used,
            activities_executed=activities_executed,
            errors_occurred=errors_occurred,
        )
        self._iterations.append(metrics)
        self._error_count += errors_occurred
        self._iteration_start = None

        return metrics

    def record_iteration(
        self,
        duration_ms: float,
        *,
        phase_durations: dict[str, float] | None = None,
        tokens_used: int = 0,
        activities_executed: int = 0,
        errors_occurred: int = 0,
    ) -> IterationMetrics:
        """
        Record a completed iteration.

        Args:
            duration_ms: Iteration duration
            phase_durations: Duration of each phase
            tokens_used: Tokens used
            activities_executed: Activities executed
            errors_occurred: Errors occurred

        Returns:
            Iteration metrics
        """
        self._current_iteration += 1
        metrics = IterationMetrics(
            iteration=self._current_iteration,
            duration_ms=duration_ms,
            phase_durations=phase_durations or {},
            tokens_used=tokens_used,
            activities_executed=activities_executed,
            errors_occurred=errors_occurred,
        )
        self._iterations.append(metrics)
        self._error_count += errors_occurred

        return metrics

    def record_llm_call(
        self,
        model: str,
        tokens_input: int,
        tokens_output: int,
        duration_ms: float,
        cost: float,
        *,
        cache_hit: bool = False,
    ) -> LLMCallMetrics:
        """
        Record an LLM call.

        Args:
            model: Model used
            tokens_input: Input tokens
            tokens_output: Output tokens
            duration_ms: Call duration
            cost: Estimated cost
            cache_hit: Whether cached

        Returns:
            LLM call metrics
        """
        metrics = LLMCallMetrics(
            model=model,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            duration_ms=duration_ms,
            cost=cost,
            cache_hit=cache_hit,
        )
        self._llm_calls.append(metrics)

        # Update aggregates
        self._total_tokens_input += tokens_input
        self._total_tokens_output += tokens_output
        self._total_cost += cost
        if cache_hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1

        return metrics

    def record_tokens(
        self,
        input: int,
        output: int,
        cost: float = 0.0,
    ) -> None:
        """
        Record token usage without full LLM call details.

        Args:
            input: Input tokens
            output: Output tokens
            cost: Estimated cost
        """
        self._total_tokens_input += input
        self._total_tokens_output += output
        self._total_cost += cost

    def record_activity(
        self,
        activity_name: str,
        success: bool,
        duration_ms: float,
        *,
        retry_count: int = 0,
    ) -> ActivityMetrics:
        """
        Record an activity execution.

        Args:
            activity_name: Name of activity
            success: Whether succeeded
            duration_ms: Execution duration
            retry_count: Number of retries

        Returns:
            Activity metrics
        """
        metrics = ActivityMetrics(
            activity_name=activity_name,
            success=success,
            duration_ms=duration_ms,
            retry_count=retry_count,
        )
        self._activities.append(metrics)

        if not success:
            self._error_count += 1

        return metrics

    def record_hitl(
        self,
        request_id: str,
        action_type: str,
        risk_level: str,
        approved: bool,
        wait_time_ms: float,
        *,
        timeout_occurred: bool = False,
    ) -> HITLMetrics:
        """
        Record HITL interaction.

        Args:
            request_id: Request ID
            action_type: Action type
            risk_level: Risk level
            approved: Whether approved
            wait_time_ms: Wait time
            timeout_occurred: If timed out

        Returns:
            HITL metrics
        """
        metrics = HITLMetrics(
            request_id=request_id,
            action_type=action_type,
            risk_level=risk_level,
            approved=approved,
            wait_time_ms=wait_time_ms,
            timeout_occurred=timeout_occurred,
        )
        self._hitl_interactions.append(metrics)

        return metrics

    def record_error(self) -> None:
        """Record an error occurrence."""
        self._error_count += 1

    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        self._cache_hits += 1

    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        self._cache_misses += 1

    # =========================================================================
    # COMPLETION
    # =========================================================================

    def complete(
        self,
        success: bool,
        exit_reason: str | None = None,
    ) -> None:
        """
        Mark execution as complete.

        Args:
            success: Whether execution succeeded
            exit_reason: Reason for completion
        """
        self.end_time = datetime.now(UTC)
        self.status = ExecutionStatus.SUCCESS if success else ExecutionStatus.FAILURE
        self.exit_reason = exit_reason

    def timeout(self, reason: str = "Timeout exceeded") -> None:
        """
        Mark execution as timed out.

        Args:
            reason: Timeout reason
        """
        self.end_time = datetime.now(UTC)
        self.status = ExecutionStatus.TIMEOUT
        self.exit_reason = reason

    def cancel(self, reason: str = "Cancelled by user") -> None:
        """
        Mark execution as cancelled.

        Args:
            reason: Cancellation reason
        """
        self.end_time = datetime.now(UTC)
        self.status = ExecutionStatus.CANCELLED
        self.exit_reason = reason

    # =========================================================================
    # COMPUTED PROPERTIES
    # =========================================================================

    @property
    def total_duration_ms(self) -> float:
        """Get total execution duration in milliseconds."""
        if self.end_time:
            delta = self.end_time - self.start_time
            return delta.total_seconds() * 1000
        return (time.monotonic() - self._start_monotonic) * 1000

    @property
    def total_iterations(self) -> int:
        """Get total iterations."""
        return len(self._iterations)

    @property
    def total_tokens(self) -> int:
        """Get total tokens (input + output)."""
        return self._total_tokens_input + self._total_tokens_output

    @property
    def total_cost(self) -> float:
        """Get total estimated cost."""
        return self._total_cost

    @property
    def total_activities(self) -> int:
        """Get total activities executed."""
        return len(self._activities)

    @property
    def activity_success_rate(self) -> float:
        """Get activity success rate (0-1)."""
        if not self._activities:
            return 1.0
        successes = sum(1 for a in self._activities if a.success)
        return successes / len(self._activities)

    @property
    def cache_hit_rate(self) -> float:
        """Get cache hit rate (0-1)."""
        total = self._cache_hits + self._cache_misses
        if total == 0:
            return 0.0
        return self._cache_hits / total

    @property
    def avg_iteration_duration_ms(self) -> float:
        """Get average iteration duration."""
        if not self._iterations:
            return 0.0
        return mean(i.duration_ms for i in self._iterations)

    @property
    def avg_llm_call_duration_ms(self) -> float:
        """Get average LLM call duration."""
        if not self._llm_calls:
            return 0.0
        return mean(c.duration_ms for c in self._llm_calls)

    # =========================================================================
    # SUMMARY
    # =========================================================================

    def to_summary(self) -> dict[str, Any]:
        """
        Get comprehensive execution summary.

        Returns:
            Summary dictionary
        """
        # Activity breakdown
        activity_counts: dict[str, int] = defaultdict(int)
        activity_successes: dict[str, int] = defaultdict(int)
        activity_durations: dict[str, list[float]] = defaultdict(list)

        for act in self._activities:
            activity_counts[act.activity_name] += 1
            if act.success:
                activity_successes[act.activity_name] += 1
            activity_durations[act.activity_name].append(act.duration_ms)

        activity_breakdown = {
            name: {
                "count": activity_counts[name],
                "success_rate": (
                    activity_successes[name] / activity_counts[name]
                    if activity_counts[name] > 0
                    else 0.0
                ),
                "avg_duration_ms": mean(activity_durations[name]),
            }
            for name in activity_counts
        }

        # Model breakdown
        model_calls: dict[str, int] = defaultdict(int)
        model_tokens: dict[str, int] = defaultdict(int)
        model_costs: dict[str, float] = defaultdict(float)

        for call in self._llm_calls:
            model_calls[call.model] += 1
            model_tokens[call.model] += call.tokens_input + call.tokens_output
            model_costs[call.model] += call.cost

        model_breakdown = {
            model: {
                "calls": model_calls[model],
                "tokens": model_tokens[model],
                "cost": model_costs[model],
            }
            for model in model_calls
        }

        return {
            "execution_id": self.execution_id,
            "session_id": self.session_id,
            "goal": self.goal,
            "status": self.status.value,
            "exit_reason": self.exit_reason,
            "timing": {
                "start_time": self.start_time.isoformat(),
                "end_time": (self.end_time.isoformat() if self.end_time else None),
                "total_duration_ms": self.total_duration_ms,
                "avg_iteration_duration_ms": self.avg_iteration_duration_ms,
                "avg_llm_call_duration_ms": self.avg_llm_call_duration_ms,
            },
            "iterations": {
                "total": self.total_iterations,
                "durations_ms": [i.duration_ms for i in self._iterations],
            },
            "tokens": {
                "input": self._total_tokens_input,
                "output": self._total_tokens_output,
                "total": self.total_tokens,
            },
            "cost": {
                "total": self._total_cost,
                "by_model": dict(model_costs),
            },
            "activities": {
                "total": self.total_activities,
                "success_rate": self.activity_success_rate,
                "breakdown": activity_breakdown,
            },
            "llm_calls": {
                "total": len(self._llm_calls),
                "by_model": dict(model_calls),
            },
            "cache": {
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "hit_rate": self.cache_hit_rate,
            },
            "hitl": {
                "total": len(self._hitl_interactions),
                "approved": sum(1 for h in self._hitl_interactions if h.approved),
                "denied": sum(1 for h in self._hitl_interactions if not h.approved),
                "timeouts": sum(1 for h in self._hitl_interactions if h.timeout_occurred),
            },
            "errors": {
                "total": self._error_count,
            },
        }


# =============================================================================
# METRICS COLLECTOR
# =============================================================================


class MetricsCollector:
    """
    Collects and aggregates metrics across multiple executions.

    Provides historical tracking and statistical summaries for
    monitoring and analysis.

    Usage:
        collector = MetricsCollector()

        # Track execution
        metrics = collector.start_execution("exec-1", goal="...")
        # ... record metrics ...
        metrics.complete(success=True)

        # Get summary
        summary = collector.get_summary()
    """

    def __init__(
        self,
        max_history: int = 1000,
    ) -> None:
        """
        Initialize metrics collector.

        Args:
            max_history: Maximum executions to track
        """
        self.max_history = max_history
        self._executions: dict[str, ExecutionMetrics] = {}
        self._execution_order: list[str] = []  # For FIFO eviction
        self._active_executions: dict[str, ExecutionMetrics] = {}

    def start_execution(
        self,
        execution_id: str,
        *,
        session_id: str | None = None,
        goal: str | None = None,
    ) -> ExecutionMetrics:
        """
        Start tracking a new execution.

        Args:
            execution_id: Unique execution ID
            session_id: Session ID
            goal: Goal being executed

        Returns:
            ExecutionMetrics instance for tracking
        """
        metrics = ExecutionMetrics(
            execution_id=execution_id,
            session_id=session_id,
            goal=goal,
        )

        self._active_executions[execution_id] = metrics

        return metrics

    def end_execution(
        self,
        execution_id: str,
        success: bool,
        exit_reason: str | None = None,
    ) -> ExecutionMetrics | None:
        """
        End execution tracking and store metrics.

        Args:
            execution_id: Execution ID
            success: Whether succeeded
            exit_reason: Exit reason

        Returns:
            Completed metrics or None if not found
        """
        metrics = self._active_executions.pop(execution_id, None)
        if metrics is None:
            return None

        metrics.complete(success=success, exit_reason=exit_reason)
        self._store_execution(execution_id, metrics)

        return metrics

    def _store_execution(
        self,
        execution_id: str,
        metrics: ExecutionMetrics,
    ) -> None:
        """Store completed execution metrics."""
        # Evict oldest if at capacity
        while len(self._execution_order) >= self.max_history:
            oldest = self._execution_order.pop(0)
            self._executions.pop(oldest, None)

        self._executions[execution_id] = metrics
        self._execution_order.append(execution_id)

    def get_execution(
        self,
        execution_id: str,
    ) -> ExecutionMetrics | None:
        """
        Get metrics for a specific execution.

        Args:
            execution_id: Execution ID

        Returns:
            Execution metrics or None
        """
        # Check active first
        if execution_id in self._active_executions:
            return self._active_executions[execution_id]
        return self._executions.get(execution_id)

    def list_executions(
        self,
        *,
        status: ExecutionStatus | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[str]:
        """
        List execution IDs.

        Args:
            status: Filter by status
            since: Filter by start time
            limit: Maximum results

        Returns:
            List of execution IDs
        """
        result = []

        for exec_id in reversed(self._execution_order):
            if len(result) >= limit:
                break

            metrics = self._executions.get(exec_id)
            if metrics is None:
                continue

            if status is not None and metrics.status != status:
                continue

            if since is not None and metrics.start_time < since:
                continue

            result.append(exec_id)

        return result

    def get_summary(
        self,
        *,
        since: datetime | None = None,
        include_active: bool = False,
    ) -> dict[str, Any]:
        """
        Get aggregated summary statistics.

        Args:
            since: Only include executions after this time
            include_active: Include active executions

        Returns:
            Summary statistics dictionary
        """
        # Collect relevant executions
        executions: list[ExecutionMetrics] = []

        for metrics in self._executions.values():
            if since is not None and metrics.start_time < since:
                continue
            executions.append(metrics)

        if include_active:
            executions.extend(self._active_executions.values())

        if not executions:
            return self._empty_summary()

        # Compute aggregates
        total = len(executions)
        completed = [e for e in executions if e.status != ExecutionStatus.RUNNING]
        successes = [e for e in completed if e.status == ExecutionStatus.SUCCESS]

        durations = [e.total_duration_ms for e in completed]
        iterations = [e.total_iterations for e in executions]
        tokens = [e.total_tokens for e in executions]
        costs = [e.total_cost for e in executions]

        return {
            "total_executions": total,
            "active_executions": len(self._active_executions),
            "completed_executions": len(completed),
            "success_rate": len(successes) / len(completed) if completed else 0.0,
            "status_breakdown": {
                status.value: sum(1 for e in executions if e.status == status)
                for status in ExecutionStatus
            },
            "latency": self._compute_percentiles(durations) if durations else {},
            "iterations": {
                "total": sum(iterations),
                "avg": mean(iterations) if iterations else 0,
                "max": max(iterations) if iterations else 0,
            },
            "tokens": {
                "total": sum(tokens),
                "avg": mean(tokens) if tokens else 0,
            },
            "cost": {
                "total": sum(costs),
                "avg": mean(costs) if costs else 0,
            },
            "activity_success_rate": (
                mean(e.activity_success_rate for e in executions) if executions else 0.0
            ),
            "cache_hit_rate": (mean(e.cache_hit_rate for e in executions) if executions else 0.0),
        }

    def _compute_percentiles(
        self,
        values: list[float],
    ) -> dict[str, float]:
        """Compute percentile statistics."""
        if not values:
            return {}

        sorted_values = sorted(values)
        n = len(sorted_values)

        def percentile(p: float) -> float:
            idx = int(p * n)
            return sorted_values[min(idx, n - 1)]

        result = {
            "min": min(values),
            "max": max(values),
            "avg": mean(values),
            "p50": percentile(0.5),
            "p90": percentile(0.9),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
        }

        if n >= 2:
            result["stddev"] = stdev(values)

        return result

    def _empty_summary(self) -> dict[str, Any]:
        """Return empty summary structure."""
        return {
            "total_executions": 0,
            "active_executions": len(self._active_executions),
            "completed_executions": 0,
            "success_rate": 0.0,
            "status_breakdown": {s.value: 0 for s in ExecutionStatus},
            "latency": {},
            "iterations": {"total": 0, "avg": 0, "max": 0},
            "tokens": {"total": 0, "avg": 0},
            "cost": {"total": 0, "avg": 0},
            "activity_success_rate": 0.0,
            "cache_hit_rate": 0.0,
        }

    def clear(self) -> None:
        """Clear all stored metrics."""
        self._executions.clear()
        self._execution_order.clear()
        # Note: Does not clear active executions

    def reset(self) -> None:
        """Reset collector completely."""
        self._executions.clear()
        self._execution_order.clear()
        self._active_executions.clear()


# =============================================================================
# PYDANTIC MODELS FOR API
# =============================================================================


class MetricsSummary(BaseModel):
    """Pydantic model for metrics summary response."""

    total_executions: int = Field(description="Total executions tracked")
    active_executions: int = Field(description="Currently active executions")
    completed_executions: int = Field(description="Completed executions")
    success_rate: float = Field(description="Success rate (0-1)")

    status_breakdown: dict[str, int] = Field(description="Count by status")

    latency: dict[str, float] = Field(default_factory=dict, description="Latency percentiles")

    iterations: dict[str, int | float] = Field(description="Iteration statistics")
    tokens: dict[str, int | float] = Field(description="Token statistics")
    cost: dict[str, float] = Field(description="Cost statistics")

    activity_success_rate: float = Field(description="Activity success rate")
    cache_hit_rate: float = Field(description="Cache hit rate")


class ExecutionSummary(BaseModel):
    """Pydantic model for single execution summary."""

    execution_id: str
    session_id: str
    goal: str | None
    status: str
    exit_reason: str | None

    timing: dict[str, Any]
    iterations: dict[str, Any]
    tokens: dict[str, Any]
    cost: dict[str, Any]
    activities: dict[str, Any]
    llm_calls: dict[str, Any]
    cache: dict[str, Any]
    hitl: dict[str, Any]
    errors: dict[str, Any]


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
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
]
