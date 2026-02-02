# src/llmcore/storage/instrumentation.py
"""
Storage Instrumentation for Phase 4 (PANOPTICON).

Provides timing, metrics, and logging for all storage operations.

This module implements the observability layer for LLMCore's storage system:
- Operation timing with context managers
- Slow query detection and alerting
- Metrics collection integration
- Structured logging with operation context
- Async operation support

Design Philosophy:
- Observable by default, configurable intensity
- Zero-overhead when disabled
- Non-blocking metrics emission
- Compatible with Prometheus, StatsD, and OpenTelemetry

Usage:
    instrumentation = StorageInstrumentation(config)

    # Sync context manager
    with instrumentation.instrument("save_session", "postgres", "sessions",
                                   session_id="abc123"):
        await self._do_save(session)

    # Async context manager
    async with instrumentation.instrument_async("search_vectors", "pgvector",
                                                "embeddings", query_hash="xyz"):
        results = await self._do_search(embedding)

STORAGE SYSTEM V2 (Phase 4 - PANOPTICON):
- Instrumentation hooks for all storage operations
- Integration with metrics collection
- Slow query alerting
- Operation tracing support
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    TypeVar,
)

logger = logging.getLogger(__name__)

# Type variables for decorator support
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# CONFIGURATION
# =============================================================================


class MetricsBackend(str, Enum):
    """Supported metrics backends."""
    PROMETHEUS = "prometheus"
    STATSD = "statsd"
    NONE = "none"


class TracingBackend(str, Enum):
    """Supported tracing backends."""
    OPENTELEMETRY = "opentelemetry"
    NONE = "none"


@dataclass
class InstrumentationConfig:
    """
    Configuration for storage instrumentation.

    Attributes:
        enabled: Master switch for instrumentation (default: True).
        log_queries: Log all queries at DEBUG level (default: False).
        log_slow_queries: Log queries exceeding threshold at WARNING level (default: True).
        slow_query_threshold_seconds: Threshold for slow query detection (default: 1.0).
        include_query_params: Include query parameters in logs (default: False, for security).
        metrics_enabled: Enable metrics collection (default: True).
        metrics_backend: Backend for metrics (prometheus, statsd, none).
        tracing_enabled: Enable distributed tracing (default: False).
        tracing_backend: Backend for tracing (opentelemetry, none).
        sample_rate: Sampling rate for high-volume operations (default: 1.0 = 100%).
        operation_timeout_seconds: Default timeout for operations (default: 30.0).
    """
    enabled: bool = True
    log_queries: bool = False
    log_slow_queries: bool = True
    slow_query_threshold_seconds: float = 1.0
    include_query_params: bool = False
    metrics_enabled: bool = True
    metrics_backend: MetricsBackend = MetricsBackend.PROMETHEUS
    tracing_enabled: bool = False
    tracing_backend: TracingBackend = TracingBackend.NONE
    sample_rate: float = 1.0
    operation_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.slow_query_threshold_seconds <= 0:
            raise ValueError("slow_query_threshold_seconds must be positive")
        if not 0.0 <= self.sample_rate <= 1.0:
            raise ValueError("sample_rate must be between 0.0 and 1.0")
        if self.operation_timeout_seconds <= 0:
            raise ValueError("operation_timeout_seconds must be positive")


DEFAULT_INSTRUMENTATION_CONFIG = InstrumentationConfig()


# =============================================================================
# INSTRUMENTATION CONTEXT
# =============================================================================


@dataclass
class InstrumentationContext:
    """
    Context for an instrumented operation.

    Tracks timing, metadata, and outcome of a single storage operation.
    This object is yielded by the instrument() context manager.

    Attributes:
        operation: Name of the operation (e.g., "save_session", "search_vectors").
        backend: Storage backend type (e.g., "postgres", "pgvector", "chromadb").
        table: Table or collection name.
        start_time: Monotonic time when operation started.
        metadata: Additional context (session_id, user_id, etc.).
        success: Whether the operation completed successfully.
        error: Exception if operation failed.
        rows_affected: Number of rows affected (for write operations).
        rows_returned: Number of rows returned (for read operations).
    """
    operation: str
    backend: str
    table: str
    start_time: float = field(default_factory=time.perf_counter)
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[Exception] = None
    rows_affected: Optional[int] = None
    rows_returned: Optional[int] = None
    _end_time: Optional[float] = field(default=None, repr=False)

    @property
    def duration_seconds(self) -> float:
        """Get operation duration in seconds."""
        end = self._end_time if self._end_time is not None else time.perf_counter()
        return end - self.start_time

    @property
    def duration_ms(self) -> float:
        """Get operation duration in milliseconds."""
        return self.duration_seconds * 1000

    def mark_complete(self) -> None:
        """Mark the operation as complete and record end time."""
        self._end_time = time.perf_counter()

    def mark_error(self, error: Exception) -> None:
        """Mark the operation as failed with an error."""
        self.success = False
        self.error = error
        self.mark_complete()

    def set_rows_affected(self, count: int) -> None:
        """Set the number of rows affected by the operation."""
        self.rows_affected = count

    def set_rows_returned(self, count: int) -> None:
        """Set the number of rows returned by the operation."""
        self.rows_returned = count

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging/metrics."""
        result = {
            "operation": self.operation,
            "backend": self.backend,
            "table": self.table,
            "duration_ms": round(self.duration_ms, 3),
            "success": self.success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if self.metadata:
            result["metadata"] = self.metadata
        if self.rows_affected is not None:
            result["rows_affected"] = self.rows_affected
        if self.rows_returned is not None:
            result["rows_returned"] = self.rows_returned
        if self.error is not None:
            result["error_type"] = type(self.error).__name__
            result["error_message"] = str(self.error)

        return result


# =============================================================================
# OPERATION RECORD
# =============================================================================


@dataclass
class OperationRecord:
    """
    Historical record of a storage operation.

    Used for tracking operation history, computing statistics,
    and detecting patterns (e.g., repeated slow queries).
    """
    operation: str
    backend: str
    table: str
    duration_ms: float
    success: bool
    timestamp: datetime
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_context(cls, ctx: InstrumentationContext) -> "OperationRecord":
        """Create a record from an instrumentation context."""
        return cls(
            operation=ctx.operation,
            backend=ctx.backend,
            table=ctx.table,
            duration_ms=ctx.duration_ms,
            success=ctx.success,
            timestamp=datetime.now(timezone.utc),
            error_type=type(ctx.error).__name__ if ctx.error else None,
            metadata=ctx.metadata.copy(),
        )


# =============================================================================
# STORAGE INSTRUMENTATION
# =============================================================================


class StorageInstrumentation:
    """
    Wraps storage operations with timing, metrics, and logging.

    This is the main entry point for instrumenting storage operations.
    It provides context managers for both sync and async operations,
    and integrates with metrics collectors and event loggers.

    Usage:
        instrumentation = StorageInstrumentation(config)

        with instrumentation.instrument("save_session", "postgres", "sessions"):
            await self._do_save(session)

    Features:
        - Automatic timing of all operations
        - Slow query detection and logging
        - Metrics emission (operation counts, latency histograms)
        - Error tracking and categorization
        - Sampling support for high-volume operations
        - History tracking for debugging
    """

    def __init__(
        self,
        config: Optional[InstrumentationConfig] = None,
        metrics_collector: Optional[Any] = None,  # MetricsCollector
        event_logger: Optional[Any] = None,  # EventLogger
        tracer: Optional[Any] = None,  # Tracer from OpenTelemetry
    ):
        """
        Initialize storage instrumentation.

        Args:
            config: Instrumentation configuration.
            metrics_collector: Optional MetricsCollector for metrics emission.
            event_logger: Optional EventLogger for database event logging.
            tracer: Optional OpenTelemetry tracer for distributed tracing.
        """
        self.config = config or DEFAULT_INSTRUMENTATION_CONFIG
        self._metrics = metrics_collector
        self._event_logger = event_logger
        self._tracer = tracer

        # Operation history for debugging/analysis (bounded buffer)
        self._history: List[OperationRecord] = []
        self._history_max_size = 1000
        self._history_lock = asyncio.Lock()

        # Statistics
        self._total_operations = 0
        self._total_errors = 0
        self._slow_query_count = 0

        logger.debug(
            "StorageInstrumentation initialized",
            extra={
                "config": {
                    "enabled": self.config.enabled,
                    "log_queries": self.config.log_queries,
                    "slow_query_threshold_seconds": self.config.slow_query_threshold_seconds,
                    "metrics_enabled": self.config.metrics_enabled,
                }
            }
        )

    @property
    def enabled(self) -> bool:
        """Check if instrumentation is enabled."""
        return self.config.enabled

    @property
    def total_operations(self) -> int:
        """Get total number of instrumented operations."""
        return self._total_operations

    @property
    def total_errors(self) -> int:
        """Get total number of failed operations."""
        return self._total_errors

    @property
    def slow_query_count(self) -> int:
        """Get count of slow queries detected."""
        return self._slow_query_count

    def get_recent_operations(self, count: int = 100) -> List[OperationRecord]:
        """
        Get recent operations from history.

        Args:
            count: Maximum number of records to return.

        Returns:
            List of recent operation records (newest first).
        """
        return list(reversed(self._history[-count:]))

    def get_slow_queries(
        self,
        threshold_ms: Optional[float] = None,
        count: int = 100
    ) -> List[OperationRecord]:
        """
        Get recent slow queries.

        Args:
            threshold_ms: Threshold in milliseconds (defaults to config value).
            count: Maximum number of records to return.

        Returns:
            List of slow query records (newest first).
        """
        threshold = threshold_ms or (self.config.slow_query_threshold_seconds * 1000)
        slow_queries = [
            r for r in self._history
            if r.duration_ms > threshold
        ]
        return list(reversed(slow_queries[-count:]))

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get instrumentation statistics.

        Returns:
            Dictionary with operation counts, error rates, etc.
        """
        error_rate = (
            self._total_errors / self._total_operations
            if self._total_operations > 0
            else 0.0
        )

        durations = [r.duration_ms for r in self._history]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        return {
            "total_operations": self._total_operations,
            "total_errors": self._total_errors,
            "error_rate": round(error_rate, 4),
            "slow_query_count": self._slow_query_count,
            "history_size": len(self._history),
            "average_duration_ms": round(avg_duration, 3),
            "config": {
                "enabled": self.config.enabled,
                "slow_query_threshold_ms": self.config.slow_query_threshold_seconds * 1000,
            }
        }

    def _should_sample(self) -> bool:
        """Check if this operation should be sampled (based on sample_rate)."""
        if self.config.sample_rate >= 1.0:
            return True
        import random
        return random.random() < self.config.sample_rate

    @contextmanager
    def instrument(
        self,
        operation: str,
        backend: str,
        table: str,
        **metadata: Any,
    ) -> Iterator[InstrumentationContext]:
        """
        Context manager for instrumenting a storage operation.

        Args:
            operation: Operation name (e.g., "save_session", "search_vectors").
            backend: Backend type (e.g., "postgres", "pgvector", "chromadb").
            table: Table or collection name.
            **metadata: Additional context (session_id, user_id, etc.).

        Yields:
            InstrumentationContext for the operation.

        Example:
            with instrumentation.instrument("save_session", "postgres", "sessions",
                                           session_id="abc123"):
                await self._do_save(session)
        """
        if not self.config.enabled:
            # Yield a dummy context when disabled
            yield InstrumentationContext(operation=operation, backend=backend, table=table)
            return

        ctx = InstrumentationContext(
            operation=operation,
            backend=backend,
            table=table,
            metadata=metadata,
        )

        # Log query start if enabled
        if self.config.log_queries:
            log_extra = {
                "event": "storage_operation_start",
                "operation": operation,
                "backend": backend,
                "table": table,
            }
            if self.config.include_query_params:
                log_extra["metadata"] = metadata
            logger.debug("Storage operation starting", extra=log_extra)

        try:
            yield ctx
            ctx.mark_complete()
        except Exception as e:
            ctx.mark_error(e)
            raise
        finally:
            self._record_completion(ctx)

    @asynccontextmanager
    async def instrument_async(
        self,
        operation: str,
        backend: str,
        table: str,
        **metadata: Any,
    ) -> AsyncIterator[InstrumentationContext]:
        """
        Async context manager for instrumenting a storage operation.

        Same as instrument() but for async operations.

        Args:
            operation: Operation name.
            backend: Backend type.
            table: Table or collection name.
            **metadata: Additional context.

        Yields:
            InstrumentationContext for the operation.
        """
        if not self.config.enabled:
            yield InstrumentationContext(operation=operation, backend=backend, table=table)
            return

        ctx = InstrumentationContext(
            operation=operation,
            backend=backend,
            table=table,
            metadata=metadata,
        )

        if self.config.log_queries:
            log_extra = {
                "event": "storage_operation_start",
                "operation": operation,
                "backend": backend,
                "table": table,
            }
            if self.config.include_query_params:
                log_extra["metadata"] = metadata
            logger.debug("Storage operation starting", extra=log_extra)

        try:
            yield ctx
            ctx.mark_complete()
        except Exception as e:
            ctx.mark_error(e)
            raise
        finally:
            await self._record_completion_async(ctx)

    def _record_completion(self, ctx: InstrumentationContext) -> None:
        """Record operation completion (sync version)."""
        self._total_operations += 1

        if not ctx.success:
            self._total_errors += 1
            self._record_error(ctx)

        # Check for slow query
        if ctx.duration_seconds > self.config.slow_query_threshold_seconds:
            self._slow_query_count += 1
            self._record_slow_query(ctx)
        elif self.config.log_queries:
            self._log_operation(ctx)

        # Record metrics
        if self._metrics and self.config.metrics_enabled:
            self._emit_metrics(ctx)

        # Add to history if sampling
        if self._should_sample():
            self._add_to_history(ctx)

        # Emit event
        if self._event_logger:
            self._emit_event(ctx)

    async def _record_completion_async(self, ctx: InstrumentationContext) -> None:
        """Record operation completion (async version)."""
        self._total_operations += 1

        if not ctx.success:
            self._total_errors += 1
            self._record_error(ctx)

        # Check for slow query
        if ctx.duration_seconds > self.config.slow_query_threshold_seconds:
            self._slow_query_count += 1
            self._record_slow_query(ctx)
        elif self.config.log_queries:
            self._log_operation(ctx)

        # Record metrics
        if self._metrics and self.config.metrics_enabled:
            self._emit_metrics(ctx)

        # Add to history if sampling
        if self._should_sample():
            async with self._history_lock:
                self._add_to_history_unlocked(ctx)

        # Emit event (async)
        if self._event_logger:
            await self._emit_event_async(ctx)

    def _record_slow_query(self, ctx: InstrumentationContext) -> None:
        """Log a slow query warning."""
        if not self.config.log_slow_queries:
            return

        log_extra = {
            "event": "slow_query",
            "operation": ctx.operation,
            "backend": ctx.backend,
            "table": ctx.table,
            "duration_ms": round(ctx.duration_ms, 3),
            "threshold_ms": self.config.slow_query_threshold_seconds * 1000,
        }
        if self.config.include_query_params:
            log_extra["metadata"] = ctx.metadata

        logger.warning(
            f"Slow storage operation detected: {ctx.operation} took {ctx.duration_ms:.1f}ms "
            f"(threshold: {self.config.slow_query_threshold_seconds * 1000:.0f}ms)",
            extra=log_extra
        )

    def _record_error(self, ctx: InstrumentationContext) -> None:
        """Log an operation error."""
        log_extra = {
            "event": "storage_error",
            "operation": ctx.operation,
            "backend": ctx.backend,
            "table": ctx.table,
            "duration_ms": round(ctx.duration_ms, 3),
            "error_type": type(ctx.error).__name__ if ctx.error else "Unknown",
            "error_message": str(ctx.error) if ctx.error else "Unknown error",
        }
        if self.config.include_query_params:
            log_extra["metadata"] = ctx.metadata

        logger.error(
            f"Storage operation failed: {ctx.operation} on {ctx.backend}.{ctx.table}",
            extra=log_extra
        )

    def _log_operation(self, ctx: InstrumentationContext) -> None:
        """Log a normal operation completion."""
        log_extra = {
            "event": "storage_operation_complete",
            "operation": ctx.operation,
            "backend": ctx.backend,
            "table": ctx.table,
            "duration_ms": round(ctx.duration_ms, 3),
            "success": ctx.success,
        }
        if ctx.rows_affected is not None:
            log_extra["rows_affected"] = ctx.rows_affected
        if ctx.rows_returned is not None:
            log_extra["rows_returned"] = ctx.rows_returned

        logger.debug("Storage operation complete", extra=log_extra)

    def _emit_metrics(self, ctx: InstrumentationContext) -> None:
        """Emit metrics for the operation."""
        if self._metrics is None:
            return

        labels = {
            "operation": ctx.operation,
            "backend": ctx.backend,
            "table": ctx.table,
        }

        # Duration histogram
        self._metrics.observe_duration(
            name="llmcore_storage_query_duration_seconds",
            value=ctx.duration_seconds,
            labels=labels,
        )

        # Operation counter
        self._metrics.increment(
            name="llmcore_storage_operations_total",
            labels=labels,
        )

        # Error counter
        if not ctx.success:
            error_labels = labels.copy()
            error_labels["error_type"] = type(ctx.error).__name__ if ctx.error else "Unknown"
            self._metrics.increment(
                name="llmcore_storage_query_errors_total",
                labels=error_labels,
            )

    def _emit_event(self, ctx: InstrumentationContext) -> None:
        """Emit event to database logger (sync)."""
        if self._event_logger is None:
            return

        try:
            self._event_logger.log_event(
                event_type=f"storage_{ctx.operation}",
                session_id=ctx.metadata.get("session_id"),
                user_id=ctx.metadata.get("user_id"),
                collection_name=ctx.table if ctx.backend in ("pgvector", "chromadb") else None,
                operation_duration_ms=ctx.duration_ms,
                metadata={
                    "backend": ctx.backend,
                    "success": ctx.success,
                    "rows_affected": ctx.rows_affected,
                    "rows_returned": ctx.rows_returned,
                },
                error_message=str(ctx.error) if ctx.error else None,
            )
        except Exception as e:
            logger.warning(f"Failed to emit storage event: {e}")

    async def _emit_event_async(self, ctx: InstrumentationContext) -> None:
        """Emit event to database logger (async)."""
        if self._event_logger is None:
            return

        try:
            await self._event_logger.log_event_async(
                event_type=f"storage_{ctx.operation}",
                session_id=ctx.metadata.get("session_id"),
                user_id=ctx.metadata.get("user_id"),
                collection_name=ctx.table if ctx.backend in ("pgvector", "chromadb") else None,
                operation_duration_ms=ctx.duration_ms,
                metadata={
                    "backend": ctx.backend,
                    "success": ctx.success,
                    "rows_affected": ctx.rows_affected,
                    "rows_returned": ctx.rows_returned,
                },
                error_message=str(ctx.error) if ctx.error else None,
            )
        except Exception as e:
            logger.warning(f"Failed to emit storage event: {e}")

    def _add_to_history(self, ctx: InstrumentationContext) -> None:
        """Add operation to history (sync, uses list append)."""
        self._add_to_history_unlocked(ctx)

    def _add_to_history_unlocked(self, ctx: InstrumentationContext) -> None:
        """Add operation to history without locking."""
        record = OperationRecord.from_context(ctx)
        self._history.append(record)

        # Trim history if needed
        if len(self._history) > self._history_max_size:
            self._history = self._history[-self._history_max_size:]

    def reset_statistics(self) -> None:
        """Reset instrumentation statistics and history."""
        self._total_operations = 0
        self._total_errors = 0
        self._slow_query_count = 0
        self._history.clear()
        logger.debug("Instrumentation statistics reset")


# =============================================================================
# DECORATOR SUPPORT
# =============================================================================


def instrumented(
    instrumentation: StorageInstrumentation,
    operation: str,
    backend: str,
    table: str,
) -> Callable[[F], F]:
    """
    Decorator for instrumenting storage methods.

    Usage:
        @instrumented(self._instrumentation, "save_session", "postgres", "sessions")
        async def save_session(self, session: ChatSession) -> None:
            ...

    Args:
        instrumentation: StorageInstrumentation instance.
        operation: Operation name.
        backend: Backend type.
        table: Table name.

    Returns:
        Decorated function.
    """
    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                async with instrumentation.instrument_async(operation, backend, table):
                    return await func(*args, **kwargs)
            return async_wrapper  # type: ignore
        else:
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with instrumentation.instrument(operation, backend, table):
                    return func(*args, **kwargs)
            return sync_wrapper  # type: ignore
    return decorator


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "DEFAULT_INSTRUMENTATION_CONFIG",
    "InstrumentationConfig",
    "InstrumentationContext",
    "MetricsBackend",
    "OperationRecord",
    "StorageInstrumentation",
    "TracingBackend",
    "instrumented",
]
