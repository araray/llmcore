# src/llmcore/observability/metrics.py
"""
Performance Metrics Collection for LLMCore.

This module provides centralized performance metrics collection for all LLM
operations, including:

- Latency tracking with histogram distributions
- Throughput measurement (requests per second)
- Error rate monitoring
- Token throughput (tokens per second)
- Percentile calculations (P50, P90, P95, P99)

The metrics here are system-wide, complementing the agent-specific metrics
in llmcore.agents.observability.metrics.

Architecture:
    - MetricsRegistry: Central registry for all metrics
    - Counter: Monotonically increasing metric
    - Gauge: Point-in-time value
    - Histogram: Distribution of values with percentile calculations
    - Timer: Convenience wrapper for timing operations

Thread Safety:
    All metric operations are thread-safe using locks where necessary.

Usage:
    >>> from llmcore.observability.metrics import (
    ...     MetricsRegistry,
    ...     Counter,
    ...     Histogram,
    ...     Timer,
    ... )
    >>>
    >>> # Get or create the default registry
    >>> registry = MetricsRegistry.get_default()
    >>>
    >>> # Create a counter
    >>> requests = registry.counter("llm_requests_total", "Total LLM requests")
    >>> requests.inc()
    >>>
    >>> # Create a histogram for latencies
    >>> latency = registry.histogram("llm_latency_ms", "Request latency in ms")
    >>> latency.observe(150.5)
    >>>
    >>> # Use timer for automatic timing
    >>> with Timer(latency):
    ...     result = some_operation()
    >>>
    >>> # Get percentiles
    >>> print(f"P95 latency: {latency.percentile(95):.2f}ms")

References:
    - UNIFIED_IMPLEMENTATION_PLAN.md Phase 9
    - llmcore_spec_v2.md Section 13 (Observability System)
"""

from __future__ import annotations

import logging
import threading
import time
from bisect import insort
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from statistics import mean, median, stdev
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import psutil
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Default percentiles to track
DEFAULT_PERCENTILES = [50, 90, 95, 99]

# Maximum samples to keep in histogram (for memory efficiency)
MAX_HISTOGRAM_SAMPLES = 10000

# Default time window for rate calculations (seconds)
DEFAULT_RATE_WINDOW = 60


# =============================================================================
# ENUMS
# =============================================================================


class MetricType(str, Enum):
    """Types of metrics."""

    COUNTER = "counter"  # Monotonically increasing
    GAUGE = "gauge"  # Point-in-time value
    HISTOGRAM = "histogram"  # Distribution of values
    RATE = "rate"  # Events per time unit


class MetricUnit(str, Enum):
    """Units for metrics."""

    COUNT = "count"
    MILLISECONDS = "ms"
    SECONDS = "s"
    BYTES = "bytes"
    TOKENS = "tokens"
    PERCENT = "percent"
    REQUESTS_PER_SECOND = "req/s"
    TOKENS_PER_SECOND = "tokens/s"


# =============================================================================
# DATA MODELS
# =============================================================================


class MetricLabels(BaseModel):
    """Labels for metric dimensions."""

    provider: Optional[str] = None
    model: Optional[str] = None
    operation: Optional[str] = None
    status: Optional[str] = None
    error_type: Optional[str] = None

    def to_key(self) -> str:
        """Convert to a hashable key string."""
        parts = []
        if self.provider:
            parts.append(f"provider={self.provider}")
        if self.model:
            parts.append(f"model={self.model}")
        if self.operation:
            parts.append(f"operation={self.operation}")
        if self.status:
            parts.append(f"status={self.status}")
        if self.error_type:
            parts.append(f"error_type={self.error_type}")
        return ",".join(parts) if parts else "__default__"


class MetricSnapshot(BaseModel):
    """Snapshot of metric values at a point in time."""

    name: str
    type: MetricType
    unit: MetricUnit = MetricUnit.COUNT
    description: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    value: float = 0.0
    labels: Dict[str, str] = Field(default_factory=dict)

    # For histograms
    count: int = 0
    sum: float = 0.0
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    percentiles: Dict[int, float] = Field(default_factory=dict)


class MetricsSummary(BaseModel):
    """Summary of all collected metrics."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metrics: List[MetricSnapshot] = Field(default_factory=list)

    # Convenience aggregates
    total_requests: int = 0
    total_tokens: int = 0
    total_errors: int = 0
    error_rate: float = 0.0
    avg_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    throughput_rps: Optional[float] = None
    token_throughput: Optional[float] = None


# =============================================================================
# METRIC CLASSES
# =============================================================================


class Counter:
    """
    A monotonically increasing counter.

    Counters only go up (and reset to zero when the process restarts).
    Use for counting requests, errors, tokens, etc.

    Args:
        name: Metric name.
        description: Human-readable description.
        unit: Unit of measurement.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        unit: MetricUnit = MetricUnit.COUNT,
    ):
        self.name = name
        self.description = description
        self.unit = unit
        self._values: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(
        self,
        value: float = 1.0,
        labels: Optional[MetricLabels] = None,
    ) -> None:
        """Increment the counter.

        Args:
            value: Amount to increment (default: 1).
            labels: Optional labels for this observation.
        """
        key = labels.to_key() if labels else "__default__"
        with self._lock:
            self._values[key] += value

    def get(self, labels: Optional[MetricLabels] = None) -> float:
        """Get current counter value.

        Args:
            labels: Optional labels to filter by.

        Returns:
            Current counter value.
        """
        key = labels.to_key() if labels else "__default__"
        return self._values.get(key, 0.0)

    def get_all(self) -> Dict[str, float]:
        """Get all counter values by label."""
        with self._lock:
            return dict(self._values)

    def total(self) -> float:
        """Get total across all labels."""
        with self._lock:
            return sum(self._values.values())

    def reset(self) -> None:
        """Reset counter to zero."""
        with self._lock:
            self._values.clear()

    def snapshot(self, labels: Optional[MetricLabels] = None) -> MetricSnapshot:
        """Get a snapshot of this metric."""
        return MetricSnapshot(
            name=self.name,
            type=MetricType.COUNTER,
            unit=self.unit,
            description=self.description,
            value=self.get(labels),
            labels=labels.model_dump(exclude_none=True) if labels else {},
        )


class Gauge:
    """
    A gauge that can go up or down.

    Use for current values like active connections, memory usage, etc.

    Args:
        name: Metric name.
        description: Human-readable description.
        unit: Unit of measurement.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        unit: MetricUnit = MetricUnit.COUNT,
    ):
        self.name = name
        self.description = description
        self.unit = unit
        self._values: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    def set(
        self,
        value: float,
        labels: Optional[MetricLabels] = None,
    ) -> None:
        """Set the gauge value.

        Args:
            value: New value.
            labels: Optional labels for this observation.
        """
        key = labels.to_key() if labels else "__default__"
        with self._lock:
            self._values[key] = value

    def inc(
        self,
        value: float = 1.0,
        labels: Optional[MetricLabels] = None,
    ) -> None:
        """Increment the gauge."""
        key = labels.to_key() if labels else "__default__"
        with self._lock:
            self._values[key] += value

    def dec(
        self,
        value: float = 1.0,
        labels: Optional[MetricLabels] = None,
    ) -> None:
        """Decrement the gauge."""
        key = labels.to_key() if labels else "__default__"
        with self._lock:
            self._values[key] -= value

    def get(self, labels: Optional[MetricLabels] = None) -> float:
        """Get current gauge value."""
        key = labels.to_key() if labels else "__default__"
        return self._values.get(key, 0.0)

    def get_all(self) -> Dict[str, float]:
        """Get all gauge values by label."""
        with self._lock:
            return dict(self._values)

    def snapshot(self, labels: Optional[MetricLabels] = None) -> MetricSnapshot:
        """Get a snapshot of this metric."""
        return MetricSnapshot(
            name=self.name,
            type=MetricType.GAUGE,
            unit=self.unit,
            description=self.description,
            value=self.get(labels),
            labels=labels.model_dump(exclude_none=True) if labels else {},
        )

    def value(self, labels: Optional[MetricLabels] = None) -> float:
        """Alias for get() - get current gauge value."""
        return self.get(labels)

    @contextmanager
    def track_inprogress(
        self,
        labels: Optional[MetricLabels] = None,
    ) -> Generator[None, None, None]:
        """
        Context manager to track in-progress operations.

        Increments gauge on entry, decrements on exit.

        Usage:
            >>> gauge = Gauge("active_requests", "Active requests")
            >>> with gauge.track_inprogress():
            ...     process_request()
        """
        self.inc(1.0, labels)
        try:
            yield
        finally:
            self.dec(1.0, labels)


@dataclass
class HistogramBucket:
    """Internal storage for histogram samples."""

    samples: List[float] = field(default_factory=list)
    count: int = 0
    sum: float = 0.0
    min_val: Optional[float] = None
    max_val: Optional[float] = None

    def add(self, value: float, max_samples: int = MAX_HISTOGRAM_SAMPLES) -> None:
        """Add a sample to this bucket."""
        self.count += 1
        self.sum += value

        if self.min_val is None or value < self.min_val:
            self.min_val = value
        if self.max_val is None or value > self.max_val:
            self.max_val = value

        # Keep sorted list for percentile calculations
        if len(self.samples) < max_samples:
            insort(self.samples, value)
        else:
            # Reservoir sampling for large datasets
            import random

            if random.random() < max_samples / self.count:
                idx = random.randint(0, max_samples - 1)
                self.samples[idx] = value
                self.samples.sort()


class Histogram:
    """
    A histogram for tracking distributions of values.

    Automatically computes percentiles, mean, min, max, etc.

    Args:
        name: Metric name.
        description: Human-readable description.
        unit: Unit of measurement.
        percentiles: Which percentiles to track.
        max_samples: Maximum samples to keep in memory.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        unit: MetricUnit = MetricUnit.MILLISECONDS,
        percentiles: Optional[List[int]] = None,
        max_samples: int = MAX_HISTOGRAM_SAMPLES,
    ):
        self.name = name
        self.description = description
        self.unit = unit
        self.percentiles_to_track = percentiles or DEFAULT_PERCENTILES
        self.max_samples = max_samples
        self._buckets: Dict[str, HistogramBucket] = defaultdict(HistogramBucket)
        self._lock = threading.Lock()

    def observe(
        self,
        value: float,
        labels: Optional[MetricLabels] = None,
    ) -> None:
        """Record an observation.

        Args:
            value: Observed value.
            labels: Optional labels for this observation.
        """
        key = labels.to_key() if labels else "__default__"
        with self._lock:
            self._buckets[key].add(value, self.max_samples)

    def count(self, labels: Optional[MetricLabels] = None) -> int:
        """Get observation count.

        Args:
            labels: Optional labels. If None, returns total count across all labels.
        """
        if labels is not None:
            key = labels.to_key()
            bucket = self._buckets.get(key)
            return bucket.count if bucket else 0

        # No labels specified - sum across all buckets
        with self._lock:
            return sum(bucket.count for bucket in self._buckets.values())

    def sum(self, labels: Optional[MetricLabels] = None) -> float:
        """Get sum of all observations."""
        key = labels.to_key() if labels else "__default__"
        bucket = self._buckets.get(key)
        return bucket.sum if bucket else 0.0

    def mean(self, labels: Optional[MetricLabels] = None) -> Optional[float]:
        """Get arithmetic mean."""
        key = labels.to_key() if labels else "__default__"
        bucket = self._buckets.get(key)
        if bucket and bucket.count > 0:
            return bucket.sum / bucket.count
        return None

    def min(self, labels: Optional[MetricLabels] = None) -> Optional[float]:
        """Get minimum value."""
        key = labels.to_key() if labels else "__default__"
        bucket = self._buckets.get(key)
        return bucket.min_val if bucket else None

    def max(self, labels: Optional[MetricLabels] = None) -> Optional[float]:
        """Get maximum value."""
        key = labels.to_key() if labels else "__default__"
        bucket = self._buckets.get(key)
        return bucket.max_val if bucket else None

    def percentile(
        self,
        p: int,
        labels: Optional[MetricLabels] = None,
    ) -> Optional[float]:
        """Get a specific percentile.

        Args:
            p: Percentile (0-100).
            labels: Optional labels to filter by.

        Returns:
            Percentile value or None if no samples.
        """
        key = labels.to_key() if labels else "__default__"
        bucket = self._buckets.get(key)
        if not bucket or not bucket.samples:
            return None

        samples = bucket.samples
        idx = int(len(samples) * p / 100)
        idx = min(idx, len(samples) - 1)
        return samples[idx]

    def percentiles(
        self,
        labels: Optional[MetricLabels] = None,
    ) -> Dict[int, float]:
        """Get all tracked percentiles.

        Args:
            labels: Optional labels to filter by.

        Returns:
            Dictionary mapping percentile to value.
        """
        result = {}
        for p in self.percentiles_to_track:
            val = self.percentile(p, labels)
            if val is not None:
                result[p] = val
        return result

    def reset(self) -> None:
        """Reset all histogram data."""
        with self._lock:
            self._buckets.clear()

    def snapshot(self, labels: Optional[MetricLabels] = None) -> MetricSnapshot:
        """Get a snapshot of this metric."""
        key = labels.to_key() if labels else "__default__"
        bucket = self._buckets.get(key)

        snap = MetricSnapshot(
            name=self.name,
            type=MetricType.HISTOGRAM,
            unit=self.unit,
            description=self.description,
            labels=labels.model_dump(exclude_none=True) if labels else {},
        )

        if bucket:
            snap.count = bucket.count
            snap.sum = bucket.sum
            snap.min = bucket.min_val
            snap.max = bucket.max_val
            snap.mean = bucket.sum / bucket.count if bucket.count > 0 else None
            snap.percentiles = self.percentiles(labels)
            snap.value = snap.mean or 0.0

        return snap


class RateCounter:
    """
    A counter that tracks rate over time.

    Useful for calculating requests per second, tokens per second, etc.

    Args:
        name: Metric name.
        description: Human-readable description.
        unit: Unit of measurement.
        window_seconds: Time window for rate calculation.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        unit: MetricUnit = MetricUnit.REQUESTS_PER_SECOND,
        window_seconds: float = DEFAULT_RATE_WINDOW,
    ):
        self.name = name
        self.description = description
        self.unit = unit
        self.window_seconds = window_seconds
        self._events: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self._lock = threading.Lock()

    def inc(
        self,
        value: float = 1.0,
        labels: Optional[MetricLabels] = None,
    ) -> None:
        """Record an event.

        Args:
            value: Event value (e.g., number of tokens).
            labels: Optional labels.
        """
        key = labels.to_key() if labels else "__default__"
        now = time.time()

        with self._lock:
            self._events[key].append((now, value))
            # Cleanup old events
            cutoff = now - self.window_seconds
            self._events[key] = [(t, v) for t, v in self._events[key] if t >= cutoff]

    def rate(self, labels: Optional[MetricLabels] = None) -> float:
        """Get current rate (events per second).

        Args:
            labels: Optional labels to filter by.

        Returns:
            Rate in events per second.
        """
        key = labels.to_key() if labels else "__default__"
        now = time.time()
        cutoff = now - self.window_seconds

        with self._lock:
            events = self._events.get(key, [])
            recent = [(t, v) for t, v in events if t >= cutoff]

            if not recent:
                return 0.0

            total_value = sum(v for _, v in recent)
            time_span = now - recent[0][0] if len(recent) > 1 else self.window_seconds

            if time_span <= 0:
                return 0.0

            return total_value / time_span

    def count(self, labels: Optional[MetricLabels] = None) -> float:
        """Get total count in current window."""
        key = labels.to_key() if labels else "__default__"
        now = time.time()
        cutoff = now - self.window_seconds

        with self._lock:
            events = self._events.get(key, [])
            return sum(v for t, v in events if t >= cutoff)

    def total(self, labels: Optional[MetricLabels] = None) -> float:
        """Get total value of all events (regardless of window).

        Args:
            labels: Optional labels to filter by.

        Returns:
            Total accumulated value.
        """
        key = labels.to_key() if labels else "__default__"

        with self._lock:
            events = self._events.get(key, [])
            return sum(v for _, v in events)

    def snapshot(self, labels: Optional[MetricLabels] = None) -> MetricSnapshot:
        """Get a snapshot of this metric."""
        return MetricSnapshot(
            name=self.name,
            type=MetricType.RATE,
            unit=self.unit,
            description=self.description,
            value=self.rate(labels),
            labels=labels.model_dump(exclude_none=True) if labels else {},
        )


# =============================================================================
# TIMER CONTEXT MANAGER
# =============================================================================


class Timer:
    """
    Context manager for timing operations.

    Usage:
        >>> histogram = Histogram("operation_duration_ms")
        >>> with Timer(histogram):
        ...     do_something()

        >>> # Or with labels
        >>> with Timer(histogram, labels=MetricLabels(operation="query")):
        ...     run_query()
    """

    def __init__(
        self,
        histogram: Histogram,
        labels: Optional[MetricLabels] = None,
        unit_multiplier: float = 1000.0,  # Convert seconds to milliseconds
    ):
        """
        Initialize timer.

        Args:
            histogram: Histogram to record timing to.
            labels: Optional labels for the observation.
            unit_multiplier: Multiplier for time conversion (default: 1000 for ms).
        """
        self.histogram = histogram
        self.labels = labels
        self.unit_multiplier = unit_multiplier
        self._start_time: Optional[float] = None
        self._elapsed: Optional[float] = None

    def start(self) -> "Timer":
        """Start the timer manually."""
        self._start_time = time.perf_counter()
        self._elapsed = None
        return self

    def stop(self) -> float:
        """Stop the timer and record the duration.

        Returns:
            Elapsed time in the configured units (default: milliseconds).
        """
        if self._start_time is None:
            raise RuntimeError("Timer was not started")

        self._elapsed = (time.perf_counter() - self._start_time) * self.unit_multiplier
        self.histogram.observe(self._elapsed, self.labels)
        return self._elapsed

    def elapsed(self) -> float:
        """Get the elapsed time without stopping.

        Returns:
            Elapsed time in the configured units (default: milliseconds).
            If timer was stopped, returns the final elapsed time.
        """
        if self._elapsed is not None:
            return self._elapsed

        if self._start_time is None:
            return 0.0

        return (time.perf_counter() - self._start_time) * self.unit_multiplier

    @property
    def elapsed_ms(self) -> float:
        """Get the elapsed time in milliseconds.

        Returns:
            Elapsed time in milliseconds.
        """
        return self.elapsed()

    @property
    def elapsed_seconds(self) -> float:
        """Get the elapsed time in seconds.

        Returns:
            Elapsed time in seconds.
        """
        return self.elapsed() / 1000.0

    def __enter__(self) -> "Timer":
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._start_time is not None:
            duration = (time.perf_counter() - self._start_time) * self.unit_multiplier
            self._elapsed = duration
            self.histogram.observe(duration, self.labels)


@contextmanager
def timer(
    histogram: Histogram,
    labels: Optional[MetricLabels] = None,
    unit_multiplier: float = 1000.0,
) -> Generator[None, None, None]:
    """
    Context manager function for timing operations.

    Usage:
        >>> with timer(latency_histogram):
        ...     do_something()
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = (time.perf_counter() - start) * unit_multiplier
        histogram.observe(duration, labels)


# =============================================================================
# METRICS REGISTRY
# =============================================================================


class MetricsRegistry:
    """
    Central registry for all metrics.

    Provides a single place to create and access metrics, ensuring
    consistency and enabling global operations like reset and export.

    Usage:
        >>> registry = MetricsRegistry.get_default()
        >>> requests = registry.counter("requests_total", "Total requests")
        >>> latency = registry.histogram("latency_ms", "Request latency")
    """

    _default_instance: Optional["MetricsRegistry"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._rate_counters: Dict[str, RateCounter] = {}
        self._registry_lock = threading.Lock()

    @classmethod
    def get_default(cls) -> "MetricsRegistry":
        """Get the default singleton registry."""
        if cls._default_instance is None:
            with cls._lock:
                if cls._default_instance is None:
                    cls._default_instance = cls()
        return cls._default_instance

    @classmethod
    def reset_default(cls) -> None:
        """Reset the default registry (for testing)."""
        with cls._lock:
            if cls._default_instance:
                cls._default_instance.reset_all()
            cls._default_instance = None

    def counter(
        self,
        name: str,
        description: str = "",
        unit: MetricUnit = MetricUnit.COUNT,
    ) -> Counter:
        """Get or create a counter.

        Args:
            name: Metric name.
            description: Human-readable description.
            unit: Unit of measurement.

        Returns:
            Counter instance.
        """
        with self._registry_lock:
            if name not in self._counters:
                self._counters[name] = Counter(name, description, unit)
            return self._counters[name]

    def gauge(
        self,
        name: str,
        description: str = "",
        unit: MetricUnit = MetricUnit.COUNT,
    ) -> Gauge:
        """Get or create a gauge."""
        with self._registry_lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(name, description, unit)
            return self._gauges[name]

    def histogram(
        self,
        name: str,
        description: str = "",
        unit: MetricUnit = MetricUnit.MILLISECONDS,
        percentiles: Optional[List[int]] = None,
    ) -> Histogram:
        """Get or create a histogram."""
        with self._registry_lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(name, description, unit, percentiles)
            return self._histograms[name]

    def rate_counter(
        self,
        name: str,
        description: str = "",
        unit: MetricUnit = MetricUnit.REQUESTS_PER_SECOND,
        window_seconds: float = DEFAULT_RATE_WINDOW,
    ) -> RateCounter:
        """Get or create a rate counter."""
        with self._registry_lock:
            if name not in self._rate_counters:
                self._rate_counters[name] = RateCounter(name, description, unit, window_seconds)
            return self._rate_counters[name]

    def get_all_snapshots(self) -> List[MetricSnapshot]:
        """Get snapshots of all metrics."""
        snapshots = []

        with self._registry_lock:
            for counter in self._counters.values():
                snapshots.append(counter.snapshot())
            for gauge in self._gauges.values():
                snapshots.append(gauge.snapshot())
            for histogram in self._histograms.values():
                snapshots.append(histogram.snapshot())
            for rate_counter in self._rate_counters.values():
                snapshots.append(rate_counter.snapshot())

        return snapshots

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all registered metrics as a dictionary.

        Returns:
            Dictionary with metric names as keys and metric objects as values.
        """
        with self._registry_lock:
            metrics: Dict[str, Any] = {}
            metrics.update(self._counters)
            metrics.update(self._gauges)
            metrics.update(self._histograms)
            metrics.update(self._rate_counters)
            return metrics

    def reset_all(self) -> None:
        """Reset all metrics."""
        with self._registry_lock:
            for counter in self._counters.values():
                counter.reset()
            for histogram in self._histograms.values():
                histogram.reset()
            # Note: Gauges and rate counters reset naturally


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_metrics_registry() -> MetricsRegistry:
    """Get the default metrics registry."""
    return MetricsRegistry.get_default()


def record_llm_call(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    success: bool = True,
    operation: str = "chat",
    error_type: Optional[str] = None,
) -> None:
    """
    Record metrics for an LLM API call.

    This is a convenience function that updates all relevant metrics
    for a single LLM call.

    Args:
        provider: Provider name (e.g., "openai").
        model: Model identifier (e.g., "gpt-4o").
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        latency_ms: Request latency in milliseconds.
        success: Whether the call succeeded.
        operation: Operation type (e.g., "chat", "embedding"). Defaults to "chat".
        error_type: Error type if failed.
    """
    registry = get_metrics_registry()

    labels = MetricLabels(
        provider=provider,
        model=model,
        operation=operation,
        status="success" if success else "error",
        error_type=error_type,
    )

    # Request count
    requests = registry.counter(
        "llm_requests_total",
        "Total LLM API requests",
    )
    requests.inc(labels=labels)

    # Latency histogram
    latency = registry.histogram(
        "llm_latency_ms",
        "LLM request latency in milliseconds",
        MetricUnit.MILLISECONDS,
    )
    latency.observe(latency_ms, labels)

    # Token counts
    input_counter = registry.counter(
        "llm_input_tokens_total",
        "Total input tokens",
        MetricUnit.TOKENS,
    )
    input_counter.inc(input_tokens, labels)

    output_counter = registry.counter(
        "llm_output_tokens_total",
        "Total output tokens",
        MetricUnit.TOKENS,
    )
    output_counter.inc(output_tokens, labels)

    # Error tracking
    if not success:
        errors = registry.counter(
            "llm_errors_total",
            "Total LLM errors",
        )
        errors.inc(labels=labels)

    # Throughput
    throughput = registry.rate_counter(
        "llm_requests_per_second",
        "LLM requests per second",
        MetricUnit.REQUESTS_PER_SECOND,
    )
    throughput.inc(labels=labels)

    token_throughput = registry.rate_counter(
        "llm_tokens_per_second",
        "Tokens processed per second",
        MetricUnit.TOKENS_PER_SECOND,
    )
    token_throughput.inc(input_tokens + output_tokens, labels)


def get_metrics_summary() -> MetricsSummary:
    """
    Get a summary of all collected metrics.

    Returns:
        MetricsSummary with aggregated data.
    """
    registry = get_metrics_registry()

    summary = MetricsSummary(
        metrics=registry.get_all_snapshots(),
    )

    # Extract key metrics
    if "llm_requests_total" in registry._counters:
        summary.total_requests = int(registry._counters["llm_requests_total"].total())

    if "llm_input_tokens_total" in registry._counters:
        input_tokens = registry._counters["llm_input_tokens_total"].total()
        output_tokens = registry._counters.get("llm_output_tokens_total", Counter("", "")).total()
        summary.total_tokens = int(input_tokens + output_tokens)

    if "llm_errors_total" in registry._counters:
        summary.total_errors = int(registry._counters["llm_errors_total"].total())
        if summary.total_requests > 0:
            summary.error_rate = summary.total_errors / summary.total_requests

    if "llm_latency_ms" in registry._histograms:
        hist = registry._histograms["llm_latency_ms"]
        summary.avg_latency_ms = hist.mean()
        summary.p95_latency_ms = hist.percentile(95)

    if "llm_requests_per_second" in registry._rate_counters:
        summary.throughput_rps = registry._rate_counters["llm_requests_per_second"].rate()

    if "llm_tokens_per_second" in registry._rate_counters:
        summary.token_throughput = registry._rate_counters["llm_tokens_per_second"].rate()

    return summary


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_metrics_registry(name: str = "default") -> MetricsRegistry:
    """
    Factory function to create a new MetricsRegistry.

    Args:
        name: Name for the registry (for identification).

    Returns:
        A new MetricsRegistry instance.
    """
    return MetricsRegistry()


# =============================================================================
# COLLECTOR CLASSES
# =============================================================================


class LLMMetricsCollector:
    """
    Pre-configured metrics collector for LLM operations.

    Provides convenience methods for tracking common LLM metrics like
    requests, tokens, latency, and errors.

    Usage:
        >>> registry = MetricsRegistry()
        >>> collector = LLMMetricsCollector(registry)
        >>> collector.record_request(
        ...     provider="openai",
        ...     model="gpt-4o",
        ...     input_tokens=1000,
        ...     output_tokens=500,
        ...     latency_ms=150.0,
        ... )
    """

    def __init__(self, registry: Optional[MetricsRegistry] = None):
        """
        Initialize the collector.

        Args:
            registry: MetricsRegistry to use. If None, uses the global default.
        """
        self._registry = registry or get_metrics_registry()

        # Pre-create metrics
        self._requests = self._registry.counter(
            "llm_collector_requests_total",
            "Total LLM requests via collector",
        )
        self._input_tokens = self._registry.counter(
            "llm_collector_input_tokens_total",
            "Total input tokens via collector",
            MetricUnit.TOKENS,
        )
        self._output_tokens = self._registry.counter(
            "llm_collector_output_tokens_total",
            "Total output tokens via collector",
            MetricUnit.TOKENS,
        )
        self._errors = self._registry.counter(
            "llm_collector_errors_total",
            "Total errors via collector",
        )
        self._latency = self._registry.histogram(
            "llm_collector_latency_ms",
            "Request latency via collector",
            MetricUnit.MILLISECONDS,
        )

    def record_request(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        success: bool = True,
    ) -> None:
        """
        Record an LLM request.

        Args:
            provider: Provider name (e.g., "openai", "anthropic").
            model: Model name (e.g., "gpt-4o", "claude-3").
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            latency_ms: Request latency in milliseconds.
            success: Whether the request was successful.
        """
        labels = MetricLabels(provider=provider, model=model)

        self._requests.inc(labels=labels)
        self._input_tokens.inc(input_tokens, labels)
        self._output_tokens.inc(output_tokens, labels)
        self._latency.observe(latency_ms, labels)

        if not success:
            self._errors.inc(labels=labels)

    def record_error(
        self,
        provider: str,
        model: str,
        error_type: str,
    ) -> None:
        """
        Record an LLM error.

        Args:
            provider: Provider name.
            model: Model name.
            error_type: Type of error (e.g., "rate_limit", "timeout").
        """
        labels = MetricLabels(provider=provider, model=model, operation=error_type)
        self._errors.inc(labels=labels)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics.

        Returns:
            Dictionary with aggregated stats.
        """
        return {
            "total_requests": int(self._requests.total()),
            "total_input_tokens": int(self._input_tokens.total()),
            "total_output_tokens": int(self._output_tokens.total()),
            "total_errors": int(self._errors.total()),
            "avg_latency_ms": self._latency.mean(),
            "p95_latency_ms": self._latency.percentile(95),
        }


class SystemMetricsCollector:
    """
    Collector for system-level metrics (CPU, memory, etc.).

    Note: This is a placeholder for future implementation.
    Actual system metrics collection requires psutil or similar.

    Usage:
        >>> collector = SystemMetricsCollector()
        >>> collector.collect()  # Captures current system state
        >>> stats = collector.get_stats()
    """

    def __init__(self, registry: Optional[MetricsRegistry] = None):
        """
        Initialize the system metrics collector.

        Args:
            registry: MetricsRegistry to use.
        """
        self._registry = registry or get_metrics_registry()

        # Pre-create system metrics
        self._cpu_usage = self._registry.gauge(
            "system_cpu_usage_percent",
            "CPU usage percentage",
            MetricUnit.PERCENT,
        )
        self._memory_usage = self._registry.gauge(
            "system_memory_usage_bytes",
            "Memory usage in bytes",
            MetricUnit.BYTES,
        )
        self._memory_percent = self._registry.gauge(
            "system_memory_usage_percent",
            "Memory usage percentage",
            MetricUnit.PERCENT,
        )

    def collect(self) -> None:
        """
        Collect current system metrics.

        Note: This is a stub implementation. For real metrics,
        install psutil and uncomment the actual collection code.
        """
        self._cpu_usage.set(psutil.cpu_percent())
        mem = psutil.virtual_memory()
        self._memory_usage.set(mem.used)
        self._memory_percent.set(mem.percent)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current system statistics.

        Returns:
            Dictionary with system stats.
        """
        return {
            "cpu_usage_percent": self._cpu_usage.value(),
            "memory_usage_bytes": self._memory_usage.value(),
            "memory_usage_percent": self._memory_percent.value(),
        }
