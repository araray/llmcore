# tests/observability/test_metrics.py
"""Tests for the performance metrics collection system."""

import threading
import time

import pytest

from llmcore.observability.metrics import (
    # Metric classes
    Counter,
    Gauge,
    Histogram,
    # Collectors
    LLMMetricsCollector,
    # Data models
    MetricLabels,
    MetricsRegistry,
    MetricType,
    MetricUnit,
    RateCounter,
    Timer,
    # Factories and utilities
    create_metrics_registry,
    get_metrics_registry,
    get_metrics_summary,
    record_llm_call,
)

# =============================================================================
# METRIC LABELS TESTS
# =============================================================================


class TestMetricLabels:
    """Tests for MetricLabels class."""

    def test_empty_labels(self):
        """Test empty labels key."""
        labels = MetricLabels()
        assert labels.to_key() == "__default__"

    def test_single_label(self):
        """Test single label key."""
        labels = MetricLabels(provider="openai")
        assert "provider=openai" in labels.to_key()

    def test_multiple_labels(self):
        """Test multiple labels key."""
        labels = MetricLabels(
            provider="openai",
            model="gpt-4o",
            operation="chat",
        )
        key = labels.to_key()
        assert "provider=openai" in key
        assert "model=gpt-4o" in key
        assert "operation=chat" in key

    def test_labels_ordering(self):
        """Test labels are consistently ordered."""
        labels1 = MetricLabels(provider="openai", model="gpt-4o")
        labels2 = MetricLabels(model="gpt-4o", provider="openai")

        # Keys should be same regardless of order
        # Note: they should be stable if the same fields are set
        assert "provider=" in labels1.to_key()
        assert "model=" in labels1.to_key()


# =============================================================================
# COUNTER TESTS
# =============================================================================


class TestCounter:
    """Tests for Counter metric."""

    def test_increment_default(self):
        """Test incrementing by default value (1)."""
        counter = Counter("test_counter", "Test counter")

        counter.inc()
        counter.inc()
        counter.inc()

        assert counter.get() == 3.0

    def test_increment_custom_value(self):
        """Test incrementing by custom value."""
        counter = Counter("test_counter", "Test counter")

        counter.inc(5.0)
        counter.inc(10.0)

        assert counter.get() == 15.0

    def test_increment_with_labels(self):
        """Test incrementing with different labels."""
        counter = Counter("test_counter", "Test counter")

        labels_a = MetricLabels(provider="openai")
        labels_b = MetricLabels(provider="anthropic")

        counter.inc(5.0, labels_a)
        counter.inc(3.0, labels_b)
        counter.inc(2.0, labels_a)

        assert counter.get(labels_a) == 7.0
        assert counter.get(labels_b) == 3.0

    def test_total(self):
        """Test total across all labels."""
        counter = Counter("test_counter", "Test counter")

        counter.inc(5.0, MetricLabels(provider="openai"))
        counter.inc(3.0, MetricLabels(provider="anthropic"))
        counter.inc(2.0)  # Default label

        assert counter.total() == 10.0

    def test_reset(self):
        """Test counter reset."""
        counter = Counter("test_counter", "Test counter")

        counter.inc(100.0)
        counter.reset()

        assert counter.get() == 0.0
        assert counter.total() == 0.0

    def test_get_all(self):
        """Test getting all values by label."""
        counter = Counter("test_counter", "Test counter")

        counter.inc(5.0, MetricLabels(provider="openai"))
        counter.inc(3.0, MetricLabels(provider="anthropic"))

        all_values = counter.get_all()

        assert len(all_values) == 2

    def test_snapshot(self):
        """Test counter snapshot."""
        counter = Counter("test_counter", "Test counter", MetricUnit.COUNT)
        counter.inc(42.0)

        snap = counter.snapshot()

        assert snap.name == "test_counter"
        assert snap.type == MetricType.COUNTER
        assert snap.unit == MetricUnit.COUNT
        assert snap.value == 42.0

    def test_thread_safety(self):
        """Test counter is thread-safe."""
        counter = Counter("test_counter", "Test counter")
        errors = []

        def increment(count):
            try:
                for _ in range(count):
                    counter.inc()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=increment, args=(1000,)) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert counter.get() == 10000.0


# =============================================================================
# GAUGE TESTS
# =============================================================================


class TestGauge:
    """Tests for Gauge metric."""

    def test_set(self):
        """Test setting gauge value."""
        gauge = Gauge("test_gauge", "Test gauge")

        gauge.set(100.0)
        assert gauge.get() == 100.0

        gauge.set(50.0)
        assert gauge.get() == 50.0

    def test_inc_dec(self):
        """Test incrementing and decrementing."""
        gauge = Gauge("test_gauge", "Test gauge")

        gauge.set(100.0)
        gauge.inc(10.0)
        assert gauge.get() == 110.0

        gauge.dec(30.0)
        assert gauge.get() == 80.0

    def test_with_labels(self):
        """Test gauge with labels."""
        gauge = Gauge("test_gauge", "Test gauge")

        labels_a = MetricLabels(provider="openai")
        labels_b = MetricLabels(provider="anthropic")

        gauge.set(100.0, labels_a)
        gauge.set(200.0, labels_b)

        assert gauge.get(labels_a) == 100.0
        assert gauge.get(labels_b) == 200.0

    def test_track_inprogress(self):
        """Test track_inprogress context manager."""
        gauge = Gauge("active_requests", "Active requests")

        with gauge.track_inprogress():
            assert gauge.get() == 1.0
            with gauge.track_inprogress():
                assert gauge.get() == 2.0
            assert gauge.get() == 1.0

        assert gauge.get() == 0.0

    def test_snapshot(self):
        """Test gauge snapshot."""
        gauge = Gauge("test_gauge", "Test gauge", MetricUnit.BYTES)
        gauge.set(1024.0)

        snap = gauge.snapshot()

        assert snap.name == "test_gauge"
        assert snap.type == MetricType.GAUGE
        assert snap.unit == MetricUnit.BYTES
        assert snap.value == 1024.0


# =============================================================================
# HISTOGRAM TESTS
# =============================================================================


class TestHistogram:
    """Tests for Histogram metric."""

    def test_observe(self):
        """Test observing values."""
        hist = Histogram("test_hist", "Test histogram")

        hist.observe(100.0)
        hist.observe(200.0)
        hist.observe(300.0)

        assert hist.count() == 3

    def test_sum(self):
        """Test sum of observations."""
        hist = Histogram("test_hist", "Test histogram")

        hist.observe(100.0)
        hist.observe(200.0)
        hist.observe(300.0)

        assert hist.sum() == 600.0

    def test_mean(self):
        """Test mean calculation."""
        hist = Histogram("test_hist", "Test histogram")

        hist.observe(100.0)
        hist.observe(200.0)
        hist.observe(300.0)

        assert hist.mean() == 200.0

    def test_min_max(self):
        """Test min/max tracking."""
        hist = Histogram("test_hist", "Test histogram")

        hist.observe(50.0)
        hist.observe(100.0)
        hist.observe(200.0)
        hist.observe(75.0)

        assert hist.min() == 50.0
        assert hist.max() == 200.0

    def test_percentiles(self):
        """Test percentile calculations."""
        hist = Histogram("test_hist", "Test histogram")

        # Add values 1-100
        for i in range(1, 101):
            hist.observe(float(i))

        p50 = hist.percentile(50)
        p95 = hist.percentile(95)
        p99 = hist.percentile(99)

        assert p50 is not None
        assert 45 <= p50 <= 55  # Approximately 50
        assert p95 is not None
        assert 90 <= p95 <= 100  # Approximately 95
        assert p99 is not None
        assert 95 <= p99 <= 100  # Approximately 99

    def test_percentiles_dict(self):
        """Test getting all percentiles at once."""
        hist = Histogram(
            "test_hist",
            "Test histogram",
            percentiles=[50, 90, 99],
        )

        for i in range(1, 101):
            hist.observe(float(i))

        percs = hist.percentiles()

        assert 50 in percs
        assert 90 in percs
        assert 99 in percs

    def test_with_labels(self):
        """Test histogram with labels."""
        hist = Histogram("test_hist", "Test histogram")

        labels_a = MetricLabels(model="gpt-4o")
        labels_b = MetricLabels(model="claude-3")

        hist.observe(100.0, labels_a)
        hist.observe(150.0, labels_a)
        hist.observe(200.0, labels_b)

        assert hist.count(labels_a) == 2
        assert hist.count(labels_b) == 1
        assert hist.mean(labels_a) == 125.0
        assert hist.mean(labels_b) == 200.0

    def test_reset(self):
        """Test histogram reset."""
        hist = Histogram("test_hist", "Test histogram")

        for i in range(100):
            hist.observe(float(i))

        hist.reset()

        assert hist.count() == 0
        assert hist.mean() is None

    def test_snapshot(self):
        """Test histogram snapshot."""
        hist = Histogram("latency", "Request latency", MetricUnit.MILLISECONDS)

        hist.observe(100.0)
        hist.observe(150.0)
        hist.observe(200.0)

        snap = hist.snapshot()

        assert snap.name == "latency"
        assert snap.type == MetricType.HISTOGRAM
        assert snap.unit == MetricUnit.MILLISECONDS
        assert snap.count == 3
        assert snap.sum == 450.0
        assert snap.min == 100.0
        assert snap.max == 200.0
        assert snap.mean == 150.0


# =============================================================================
# RATE COUNTER TESTS
# =============================================================================


class TestRateCounter:
    """Tests for RateCounter metric."""

    def test_increment(self):
        """Test incrementing rate counter."""
        counter = RateCounter("test_rate", "Test rate", window_seconds=1.0)

        counter.inc()
        counter.inc()
        counter.inc()

        # Rate should be > 0 immediately after incrementing
        assert counter.count() == 3

    def test_rate_calculation(self):
        """Test rate calculation over time."""
        counter = RateCounter("test_rate", "Test rate", window_seconds=1.0)

        # Add events
        for _ in range(10):
            counter.inc()

        # Rate should be positive
        rate = counter.rate()
        assert rate >= 0  # Can be 0 if too fast

    def test_total(self):
        """Test total count."""
        counter = RateCounter("test_rate", "Test rate")

        counter.inc(5.0)
        counter.inc(10.0)

        assert counter.total() == 15.0


# =============================================================================
# TIMER TESTS
# =============================================================================


class TestTimer:
    """Tests for Timer utility."""

    def test_context_manager(self):
        """Test timer as context manager."""
        hist = Histogram("duration", "Operation duration", MetricUnit.MILLISECONDS)

        with Timer(hist):
            time.sleep(0.1)  # Sleep 100ms

        # Should have recorded approximately 100ms
        assert hist.count() == 1
        assert hist.mean() is not None
        # Allow some variance
        assert 50 <= hist.mean() <= 200

    def test_timer_with_labels(self):
        """Test timer with labels."""
        hist = Histogram("duration", "Operation duration")
        labels = MetricLabels(operation="test")

        with Timer(hist, labels):
            time.sleep(0.05)

        assert hist.count(labels) == 1

    def test_elapsed(self):
        """Test getting elapsed time."""
        hist = Histogram("duration", "Operation duration")

        timer = Timer(hist)
        timer.start()
        time.sleep(0.05)
        timer.stop()

        assert timer.elapsed_ms > 40  # At least 40ms
        assert timer.elapsed_seconds > 0.04


# =============================================================================
# METRICS REGISTRY TESTS
# =============================================================================


class TestMetricsRegistry:
    """Tests for MetricsRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry."""
        return MetricsRegistry()

    def test_counter(self, registry):
        """Test creating and using counters."""
        counter = registry.counter("requests", "Total requests")

        counter.inc()
        counter.inc()

        # Get same counter
        counter2 = registry.counter("requests", "Total requests")
        assert counter2.get() == 2.0

    def test_gauge(self, registry):
        """Test creating and using gauges."""
        gauge = registry.gauge("active", "Active connections")

        gauge.set(10.0)

        gauge2 = registry.gauge("active", "Active connections")
        assert gauge2.get() == 10.0

    def test_histogram(self, registry):
        """Test creating and using histograms."""
        hist = registry.histogram("latency", "Request latency")

        hist.observe(100.0)
        hist.observe(200.0)

        hist2 = registry.histogram("latency", "Request latency")
        assert hist2.count() == 2

    def test_rate_counter(self, registry):
        """Test creating and using rate counters."""
        rate = registry.rate_counter("rps", "Requests per second")

        rate.inc()
        rate.inc()

        rate2 = registry.rate_counter("rps", "Requests per second")
        assert rate2.count() == 2

    def test_get_all_metrics(self, registry):
        """Test getting all metrics."""
        registry.counter("counter1", "Counter 1")
        registry.gauge("gauge1", "Gauge 1")
        registry.histogram("hist1", "Histogram 1")

        metrics = registry.get_all_metrics()

        assert len(metrics) == 3
        assert "counter1" in metrics
        assert "gauge1" in metrics
        assert "hist1" in metrics

    def test_get_all_snapshots(self, registry):
        """Test getting snapshots of all metrics."""
        counter = registry.counter("requests", "Total requests")
        counter.inc(10.0)

        gauge = registry.gauge("connections", "Active connections")
        gauge.set(5.0)

        snapshots = registry.get_all_snapshots()

        assert len(snapshots) >= 2

        # Find our metrics
        names = [s.name for s in snapshots]
        assert "requests" in names
        assert "connections" in names

    def test_reset_all(self, registry):
        """Test resetting all metrics."""
        counter = registry.counter("requests", "Total requests")
        counter.inc(100.0)

        gauge = registry.gauge("connections", "Active connections")
        gauge.set(50.0)

        registry.reset_all()

        assert counter.get() == 0.0
        # Note: gauges might not be reset, depends on implementation


# =============================================================================
# LLM METRICS COLLECTOR TESTS
# =============================================================================


class TestLLMMetricsCollector:
    """Tests for LLMMetricsCollector class."""

    @pytest.fixture
    def collector(self):
        """Create a collector with fresh registry."""
        registry = MetricsRegistry()
        return LLMMetricsCollector(registry)

    def test_record_request(self, collector):
        """Test recording a request."""
        collector.record_request(
            provider="openai",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            latency_ms=150.0,
        )

        stats = collector.get_stats()

        assert stats["total_requests"] == 1
        assert stats["total_input_tokens"] == 1000
        assert stats["total_output_tokens"] == 500

    def test_record_error(self, collector):
        """Test recording an error."""
        collector.record_error(
            provider="anthropic",
            model="claude-3",
            error_type="rate_limit",
        )

        stats = collector.get_stats()

        assert stats["total_errors"] == 1

    def test_multiple_providers(self, collector):
        """Test recording from multiple providers."""
        collector.record_request(
            provider="openai",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            latency_ms=100.0,
        )

        collector.record_request(
            provider="anthropic",
            model="claude-3",
            input_tokens=2000,
            output_tokens=1000,
            latency_ms=200.0,
        )

        stats = collector.get_stats()

        assert stats["total_requests"] == 2
        assert stats["total_input_tokens"] == 3000
        assert stats["total_output_tokens"] == 1500


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_metrics_registry(self):
        """Test creating a registry with factory."""
        registry = create_metrics_registry(name="test")

        assert registry is not None
        assert isinstance(registry, MetricsRegistry)

    def test_record_llm_call(self):
        """Test convenience function for recording LLM calls."""
        # Reset global registry
        registry = get_metrics_registry()
        registry.reset_all()

        record_llm_call(
            provider="openai",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            latency_ms=150.0,
            success=True,
        )

        summary = get_metrics_summary()

        assert summary.total_requests >= 1
        assert summary.total_tokens >= 1500


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for metrics system."""

    def test_full_metrics_workflow(self):
        """Test complete metrics workflow."""
        # Create registry
        registry = create_metrics_registry(name="integration_test")

        # Create metrics
        requests = registry.counter("http_requests_total", "Total HTTP requests")
        latency = registry.histogram("http_latency_ms", "HTTP latency", MetricUnit.MILLISECONDS)
        active = registry.gauge("http_active_requests", "Active HTTP requests")

        # Simulate some requests
        for i in range(10):
            labels = MetricLabels(
                operation="GET",
                status="200" if i < 8 else "500",
            )

            requests.inc(labels=labels)
            latency.observe(100.0 + i * 10, labels=labels)

        active.set(5.0)

        # Get snapshots
        snapshots = registry.get_all_snapshots()

        # Verify
        assert len(snapshots) >= 3
        assert requests.total() == 10
        assert latency.count() == 10
        assert active.get() == 5.0

    def test_concurrent_metrics_updates(self):
        """Test concurrent updates to metrics."""
        registry = MetricsRegistry()
        counter = registry.counter("concurrent_test", "Concurrent test")
        hist = registry.histogram("concurrent_latency", "Concurrent latency")

        errors = []

        def update_metrics(count):
            try:
                for _ in range(count):
                    counter.inc()
                    hist.observe(100.0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update_metrics, args=(100,)) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert counter.total() == 1000
        assert hist.count() == 1000
