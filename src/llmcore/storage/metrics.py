# src/llmcore/storage/metrics.py
"""
Metrics Collection for Phase 4 (PANOPTICON).

Provides a backend-agnostic metrics collection interface with support for
Prometheus, StatsD, and in-memory metrics for testing.

This module implements:
- Counter metrics (monotonically increasing values)
- Histogram metrics (distribution tracking with buckets)
- Gauge metrics (point-in-time values)
- Label support for dimensional metrics
- Optional Prometheus endpoint exposure

Design Philosophy:
- Zero dependencies when not using external backends
- Thread-safe and async-compatible
- Lazy initialization of metrics
- Graceful degradation if backend unavailable

Usage:
    # Create collector
    collector = MetricsCollector(config)

    # Increment counter
    collector.increment("operations_total", labels={"operation": "save"})

    # Observe histogram value
    collector.observe_duration("query_duration_seconds", 0.123,
                               labels={"backend": "postgres"})

    # Set gauge value
    collector.set_gauge("connections_active", 5, labels={"pool": "main"})

    # Get all metrics (for export or debugging)
    metrics = collector.get_metrics()

STORAGE SYSTEM V2 (Phase 4 - PANOPTICON):
- Prometheus-compatible metric types
- StatsD backend support
- In-memory fallback for testing
- Thread-safe operations
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


class MetricsBackendType(str, Enum):
    """Supported metrics backends."""
    PROMETHEUS = "prometheus"
    STATSD = "statsd"
    MEMORY = "memory"  # In-memory for testing
    NONE = "none"


@dataclass
class MetricsConfig:
    """
    Configuration for metrics collection.

    Attributes:
        enabled: Master switch for metrics collection.
        backend: Backend type (prometheus, statsd, memory, none).
        prefix: Prefix for all metric names.
        default_labels: Labels added to all metrics.
        prometheus_port: Port for Prometheus HTTP endpoint (if backend=prometheus).
        statsd_host: StatsD server host.
        statsd_port: StatsD server port.
        histogram_buckets: Default bucket boundaries for histograms.
        flush_interval_seconds: Interval for flushing metrics to backend.
    """
    enabled: bool = True
    backend: MetricsBackendType = MetricsBackendType.MEMORY
    prefix: str = "llmcore_storage"
    default_labels: Dict[str, str] = field(default_factory=dict)
    prometheus_port: int = 9090
    statsd_host: str = "localhost"
    statsd_port: int = 8125
    histogram_buckets: Tuple[float, ...] = (
        0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5,
        0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float("inf")
    )
    flush_interval_seconds: float = 10.0


DEFAULT_METRICS_CONFIG = MetricsConfig()


# =============================================================================
# METRIC TYPES
# =============================================================================


@dataclass
class MetricValue:
    """Base class for metric values."""
    name: str
    labels: Dict[str, str]
    timestamp: float = field(default_factory=time.time)


@dataclass
class CounterValue(MetricValue):
    """Counter metric value (monotonically increasing)."""
    value: float = 0.0


@dataclass
class GaugeValue(MetricValue):
    """Gauge metric value (can go up and down)."""
    value: float = 0.0


@dataclass
class HistogramValue(MetricValue):
    """Histogram metric value with buckets."""
    buckets: Dict[float, int] = field(default_factory=dict)
    sum: float = 0.0
    count: int = 0


# =============================================================================
# METRICS BACKEND INTERFACE
# =============================================================================


class MetricsBackend(ABC):
    """Abstract base class for metrics backends."""

    @abstractmethod
    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter metric."""
        pass

    @abstractmethod
    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge metric value."""
        pass

    @abstractmethod
    def observe(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Observe a histogram value."""
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics for export."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset all metrics."""
        pass


# =============================================================================
# IN-MEMORY BACKEND
# =============================================================================


class InMemoryMetricsBackend(MetricsBackend):
    """
    In-memory metrics backend for testing and development.

    Stores all metrics in memory with thread-safe access.
    Useful for unit tests and local development.
    """

    def __init__(self, config: MetricsConfig):
        self.config = config
        self._lock = threading.Lock()
        self._counters: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._gauges: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._histograms: Dict[str, Dict[str, HistogramValue]] = defaultdict(dict)

    def _labels_key(self, labels: Optional[Dict[str, str]]) -> str:
        """Convert labels dict to a hashable key."""
        if not labels:
            return ""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))

    def _full_name(self, name: str) -> str:
        """Get full metric name with prefix."""
        if self.config.prefix:
            return f"{self.config.prefix}_{name}"
        return name

    def _merge_labels(self, labels: Optional[Dict[str, str]]) -> Dict[str, str]:
        """Merge provided labels with default labels."""
        merged = dict(self.config.default_labels)
        if labels:
            merged.update(labels)
        return merged

    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter metric."""
        full_name = self._full_name(name)
        merged_labels = self._merge_labels(labels)
        key = self._labels_key(merged_labels)

        with self._lock:
            self._counters[full_name][key] += value

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge metric value."""
        full_name = self._full_name(name)
        merged_labels = self._merge_labels(labels)
        key = self._labels_key(merged_labels)

        with self._lock:
            self._gauges[full_name][key] = value

    def observe(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Observe a histogram value."""
        full_name = self._full_name(name)
        merged_labels = self._merge_labels(labels)
        key = self._labels_key(merged_labels)

        with self._lock:
            if key not in self._histograms[full_name]:
                self._histograms[full_name][key] = HistogramValue(
                    name=full_name,
                    labels=merged_labels,
                    buckets={b: 0 for b in self.config.histogram_buckets},
                )

            hist = self._histograms[full_name][key]
            hist.sum += value
            hist.count += 1
            hist.timestamp = time.time()

            # Update buckets
            for bucket in self.config.histogram_buckets:
                if value <= bucket:
                    hist.buckets[bucket] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics for export."""
        with self._lock:
            return {
                "counters": {
                    name: dict(values)
                    for name, values in self._counters.items()
                },
                "gauges": {
                    name: dict(values)
                    for name, values in self._gauges.items()
                },
                "histograms": {
                    name: {
                        key: {
                            "sum": hist.sum,
                            "count": hist.count,
                            "buckets": dict(hist.buckets),
                        }
                        for key, hist in histograms.items()
                    }
                    for name, histograms in self._histograms.items()
                },
            }

    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current counter value."""
        full_name = self._full_name(name)
        merged_labels = self._merge_labels(labels)
        key = self._labels_key(merged_labels)

        with self._lock:
            return self._counters[full_name].get(key, 0.0)

    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current gauge value."""
        full_name = self._full_name(name)
        merged_labels = self._merge_labels(labels)
        key = self._labels_key(merged_labels)

        with self._lock:
            return self._gauges[full_name].get(key, 0.0)

    def get_histogram_stats(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Get histogram statistics."""
        full_name = self._full_name(name)
        merged_labels = self._merge_labels(labels)
        key = self._labels_key(merged_labels)

        with self._lock:
            if full_name in self._histograms and key in self._histograms[full_name]:
                hist = self._histograms[full_name][key]
                return {
                    "sum": hist.sum,
                    "count": hist.count,
                    "mean": hist.sum / hist.count if hist.count > 0 else 0.0,
                    "buckets": dict(hist.buckets),
                }
        return None

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


# =============================================================================
# PROMETHEUS BACKEND
# =============================================================================


class PrometheusMetricsBackend(MetricsBackend):
    """
    Prometheus metrics backend using prometheus_client library.

    Requires: pip install prometheus-client
    """

    def __init__(self, config: MetricsConfig):
        self.config = config
        self._lock = threading.Lock()
        self._counters: Dict[str, Any] = {}
        self._gauges: Dict[str, Any] = {}
        self._histograms: Dict[str, Any] = {}
        self._server = None

        # Try to import prometheus_client
        try:
            import prometheus_client
            self._prometheus = prometheus_client
            self._available = True
        except ImportError:
            logger.warning(
                "prometheus_client not installed. "
                "Install with: pip install prometheus-client"
            )
            self._prometheus = None
            self._available = False

    def _get_or_create_counter(self, name: str, labels: Dict[str, str]) -> Any:
        """Get or create a Prometheus counter."""
        if not self._available:
            return None

        full_name = f"{self.config.prefix}_{name}" if self.config.prefix else name
        label_names = tuple(sorted(labels.keys()))
        key = (full_name, label_names)

        with self._lock:
            if key not in self._counters:
                self._counters[key] = self._prometheus.Counter(
                    full_name,
                    f"Counter for {name}",
                    label_names,
                )
            return self._counters[key].labels(**labels)

    def _get_or_create_gauge(self, name: str, labels: Dict[str, str]) -> Any:
        """Get or create a Prometheus gauge."""
        if not self._available:
            return None

        full_name = f"{self.config.prefix}_{name}" if self.config.prefix else name
        label_names = tuple(sorted(labels.keys()))
        key = (full_name, label_names)

        with self._lock:
            if key not in self._gauges:
                self._gauges[key] = self._prometheus.Gauge(
                    full_name,
                    f"Gauge for {name}",
                    label_names,
                )
            return self._gauges[key].labels(**labels)

    def _get_or_create_histogram(self, name: str, labels: Dict[str, str]) -> Any:
        """Get or create a Prometheus histogram."""
        if not self._available:
            return None

        full_name = f"{self.config.prefix}_{name}" if self.config.prefix else name
        label_names = tuple(sorted(labels.keys()))
        key = (full_name, label_names)

        with self._lock:
            if key not in self._histograms:
                # Filter out inf from buckets for Prometheus (it adds +Inf automatically)
                buckets = tuple(b for b in self.config.histogram_buckets if b != float("inf"))
                self._histograms[key] = self._prometheus.Histogram(
                    full_name,
                    f"Histogram for {name}",
                    label_names,
                    buckets=buckets,
                )
            return self._histograms[key].labels(**labels)

    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter metric."""
        merged_labels = dict(self.config.default_labels)
        if labels:
            merged_labels.update(labels)

        counter = self._get_or_create_counter(name, merged_labels)
        if counter:
            counter.inc(value)

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge metric value."""
        merged_labels = dict(self.config.default_labels)
        if labels:
            merged_labels.update(labels)

        gauge = self._get_or_create_gauge(name, merged_labels)
        if gauge:
            gauge.set(value)

    def observe(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Observe a histogram value."""
        merged_labels = dict(self.config.default_labels)
        if labels:
            merged_labels.update(labels)

        histogram = self._get_or_create_histogram(name, merged_labels)
        if histogram:
            histogram.observe(value)

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics for export."""
        if not self._available:
            return {"error": "prometheus_client not available"}

        # Generate Prometheus text format
        return {
            "format": "prometheus",
            "text": self._prometheus.generate_latest().decode("utf-8"),
        }

    def start_http_server(self) -> None:
        """Start Prometheus HTTP server for scraping."""
        if not self._available:
            logger.warning("Cannot start Prometheus server: prometheus_client not available")
            return

        if self._server is None:
            self._prometheus.start_http_server(self.config.prometheus_port)
            logger.info(f"Prometheus metrics server started on port {self.config.prometheus_port}")

    def reset(self) -> None:
        """Reset all metrics (not typically done in production)."""
        logger.warning("Resetting Prometheus metrics (not recommended in production)")
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


# =============================================================================
# NULL BACKEND (DISABLED METRICS)
# =============================================================================


class NullMetricsBackend(MetricsBackend):
    """No-op metrics backend when metrics are disabled."""

    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        pass

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        pass

    def observe(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        pass

    def get_metrics(self) -> Dict[str, Any]:
        return {"enabled": False}

    def reset(self) -> None:
        pass


# =============================================================================
# METRICS COLLECTOR (MAIN INTERFACE)
# =============================================================================


class MetricsCollector:
    """
    Main interface for metrics collection.

    Provides a unified API for collecting metrics across different backends.
    Automatically selects the appropriate backend based on configuration.

    Usage:
        config = MetricsConfig(backend=MetricsBackendType.PROMETHEUS)
        collector = MetricsCollector(config)

        # Increment counter
        collector.increment("operations_total", labels={"operation": "save"})

        # Observe duration
        collector.observe_duration("query_duration_seconds", 0.123,
                                   labels={"backend": "postgres"})
    """

    def __init__(self, config: Optional[MetricsConfig] = None):
        """
        Initialize metrics collector.

        Args:
            config: Metrics configuration. If None, uses default config.
        """
        self.config = config or DEFAULT_METRICS_CONFIG
        self._backend = self._create_backend()

        logger.debug(
            "MetricsCollector initialized",
            extra={
                "backend": self.config.backend.value,
                "enabled": self.config.enabled,
                "prefix": self.config.prefix,
            }
        )

    def _create_backend(self) -> MetricsBackend:
        """Create the appropriate metrics backend."""
        if not self.config.enabled:
            return NullMetricsBackend()

        if self.config.backend == MetricsBackendType.PROMETHEUS:
            return PrometheusMetricsBackend(self.config)
        elif self.config.backend == MetricsBackendType.MEMORY:
            return InMemoryMetricsBackend(self.config)
        elif self.config.backend == MetricsBackendType.NONE:
            return NullMetricsBackend()
        else:
            logger.warning(
                f"Unknown metrics backend: {self.config.backend}, using memory"
            )
            return InMemoryMetricsBackend(self.config)

    @property
    def enabled(self) -> bool:
        """Check if metrics collection is enabled."""
        return self.config.enabled

    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Increment a counter metric.

        Args:
            name: Metric name (will be prefixed).
            value: Amount to increment (default: 1.0).
            labels: Dimensional labels.
        """
        self._backend.increment(name, value, labels)

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Set a gauge metric value.

        Args:
            name: Metric name (will be prefixed).
            value: Gauge value.
            labels: Dimensional labels.
        """
        self._backend.set_gauge(name, value, labels)

    def observe(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Observe a histogram value.

        Args:
            name: Metric name (will be prefixed).
            value: Observed value.
            labels: Dimensional labels.
        """
        self._backend.observe(name, value, labels)

    def observe_duration(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Observe a duration value (alias for observe).

        Args:
            name: Metric name (will be prefixed).
            value: Duration in seconds.
            labels: Dimensional labels.
        """
        self.observe(name, value, labels)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics for export or debugging.

        Returns:
            Dictionary with all metrics data.
        """
        return self._backend.get_metrics()

    def reset(self) -> None:
        """Reset all metrics (primarily for testing)."""
        self._backend.reset()

    # Convenience methods for common storage metrics

    def record_operation(
        self,
        operation: str,
        backend: str,
        table: str,
        duration_seconds: float,
        success: bool = True,
        error_type: Optional[str] = None,
    ) -> None:
        """
        Record a complete storage operation.

        Args:
            operation: Operation name.
            backend: Backend type.
            table: Table/collection name.
            duration_seconds: Operation duration.
            success: Whether operation succeeded.
            error_type: Error type if failed.
        """
        labels = {
            "operation": operation,
            "backend": backend,
            "table": table,
        }

        # Duration histogram
        self.observe_duration("query_duration_seconds", duration_seconds, labels)

        # Operation counter
        self.increment("operations_total", labels=labels)

        # Error counter
        if not success:
            error_labels = labels.copy()
            error_labels["error_type"] = error_type or "unknown"
            self.increment("query_errors_total", labels=error_labels)

    def record_pool_status(
        self,
        pool_name: str,
        idle: int,
        in_use: int,
        total: int,
    ) -> None:
        """
        Record connection pool status.

        Args:
            pool_name: Pool identifier.
            idle: Number of idle connections.
            in_use: Number of connections in use.
            total: Total pool size.
        """
        self.set_gauge("pool_connections_idle", idle, labels={"pool": pool_name})
        self.set_gauge("pool_connections_in_use", in_use, labels={"pool": pool_name})
        self.set_gauge("pool_connections_total", total, labels={"pool": pool_name})

    def record_table_rows(self, table: str, count: int) -> None:
        """
        Record approximate row count for a table.

        Args:
            table: Table name.
            count: Approximate row count.
        """
        self.set_gauge("table_rows", count, labels={"table": table})


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "MetricsConfig",
    "MetricsBackendType",
    "MetricsBackend",
    "MetricsCollector",
    "InMemoryMetricsBackend",
    "PrometheusMetricsBackend",
    "NullMetricsBackend",
    "DEFAULT_METRICS_CONFIG",
]
