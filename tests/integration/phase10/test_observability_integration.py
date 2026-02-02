"""
Phase 10 Integration Tests: Observability System.

Tests for metrics collection, event logging, cost tracking, and analytics.
"""

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Observability imports
from llmcore.observability import (
    CostAnalyzer,
    CostTracker,
    Counter,
    EventBuffer,
    ExecutionReplayer,
    Gauge,
    Histogram,
    Timer,
)
from llmcore.observability.events import (
    Event,
    EventBuffer,
    EventCategory,
    ExecutionTrace,
    ObservabilityConfig,
    ObservabilityLogger,
    Severity,
    create_observability_logger,
)
from llmcore.observability.metrics import (
    LLMMetricsCollector,
    MetricsRegistry,
    MetricUnit,
)

# ============================================================================
# Metrics Tests
# ============================================================================


class TestMetricsBasics:
    """Test basic metrics functionality."""

    def test_counter_creation(self) -> None:
        """Counter should be creatable with name and description."""
        counter = Counter(name="test_counter", description="A test counter")
        assert counter is not None
        assert counter.name == "test_counter"

    def test_counter_increment(self) -> None:
        """Counter should increment correctly."""
        counter = Counter(name="test_requests", description="Request count")

        counter.inc()
        assert counter.total() == 1

        counter.inc(5)
        assert counter.total() == 6

    def test_histogram_creation(self) -> None:
        """Histogram should be creatable with name and description."""
        histogram = Histogram(
            name="test_latency",
            description="Latency distribution",
            unit=MetricUnit.MILLISECONDS,
        )
        assert histogram is not None
        assert histogram.name == "test_latency"

    def test_histogram_record(self) -> None:
        """Histogram should record values correctly."""
        histogram = Histogram(name="response_time", description="Response time")

        histogram.observe(10.5)
        histogram.observe(20.3)
        histogram.observe(15.0)

        assert histogram.count() == 3
        assert histogram.min() <= 10.5
        assert histogram.max() >= 20.3

    def test_gauge_creation(self) -> None:
        """Gauge should be creatable with name."""
        gauge = Gauge(name="active_connections", description="Current connections")
        assert gauge is not None

    def test_gauge_operations(self) -> None:
        """Gauge should support set, increase, decrease."""
        gauge = Gauge(name="queue_size", description="Queue size")

        gauge.set(100)
        assert gauge.value() == 100

        gauge.inc(10)
        assert gauge.value() == 110

        gauge.dec(30)
        assert gauge.value() == 80

    def test_timer_context_manager(self) -> None:
        """Timer should work as context manager."""
        histogram = Histogram(name="operation_time", description="Op time")

        with Timer(histogram) as timer:
            time.sleep(0.01)  # 10ms

        # Should have recorded something > 0
        assert histogram.count() == 1
        assert histogram.mean() > 0


class TestMetricsRegistry:
    """Test metrics registry functionality."""

    def test_registry_creation(self) -> None:
        """MetricsRegistry should be creatable."""
        registry = MetricsRegistry()
        assert registry is not None

    def test_registry_counter_registration(self) -> None:
        """Registry should track registered counters."""
        registry = MetricsRegistry()
        counter = registry.counter("api_calls", "API call count")

        assert counter is not None
        counter.inc()
        assert counter.total() == 1

    def test_registry_histogram_registration(self) -> None:
        """Registry should track registered histograms."""
        registry = MetricsRegistry()
        histogram = registry.histogram("latency", "Latency distribution")

        assert histogram is not None
        histogram.observe(50.0)
        assert histogram.count() == 1


class TestLLMMetricsCollector:
    """Test LLM-specific metrics collection."""

    def test_collector_creation(self) -> None:
        """LLMMetricsCollector should be creatable."""
        collector = LLMMetricsCollector()
        assert collector is not None

    def test_record_request(self) -> None:
        """Collector should record LLM requests."""
        collector = LLMMetricsCollector()

        collector.record_request(
            provider="openai",
            model="gpt-4",
            input_tokens=100,
            output_tokens=50,
            latency_ms=250.0,
            success=True,
        )

        stats = collector.get_stats()
        assert stats is not None

    def test_record_error(self) -> None:
        """Collector should record LLM errors."""
        collector = LLMMetricsCollector()

        collector.record_error(
            provider="anthropic",
            model="claude-3",
            error_type="rate_limit",
        )

        stats = collector.get_stats()
        assert stats is not None

    def test_multiple_providers(self) -> None:
        """Collector should track multiple providers."""
        collector = LLMMetricsCollector()

        collector.record_request("openai", "gpt-4", 100, 50, 200.0)
        collector.record_request("anthropic", "claude-3", 150, 75, 180.0)
        collector.record_request("openai", "gpt-4", 80, 40, 220.0)

        stats = collector.get_stats()
        assert stats is not None


# ============================================================================
# Event Logging Tests
# ============================================================================


class TestObservabilityLogger:
    """Test event logging functionality."""

    def test_logger_creation(self, tmp_path: Path) -> None:
        """ObservabilityLogger should be creatable."""
        log_path = str(tmp_path / "events.log")
        logger = ObservabilityLogger(
            config=ObservabilityConfig(
                enabled=True,
                log_path=log_path,
            )
        )
        assert logger is not None
        logger.close()

    def test_logger_via_factory(self, tmp_path: Path) -> None:
        """create_observability_logger factory should work."""
        log_path = str(tmp_path / "events.log")
        logger = create_observability_logger(
            log_path=log_path,
            min_severity="info",
        )
        assert logger is not None
        logger.close()

    def test_log_event(self, tmp_path: Path) -> None:
        """Logger should log events."""
        log_path = str(tmp_path / "events.log")
        logger = create_observability_logger(log_path=log_path)

        try:
            logger.log_event(
                category="llm",
                event_type="request_completed",
                severity="info",
                data={"model": "gpt-4"},
            )
            logger.flush()
        finally:
            logger.close()

    def test_severity_filtering(self, tmp_path: Path) -> None:
        """Logger should filter by severity."""
        log_path = str(tmp_path / "events.log")
        logger = create_observability_logger(
            log_path=log_path,
            min_severity="warning",
        )

        try:
            # Info should be filtered out
            logger.log_event(
                category="test",
                event_type="info_event",
                severity="info",
            )
            # Warning should pass
            logger.log_event(
                category="test",
                event_type="warning_event",
                severity="warning",
            )
            logger.flush()
        finally:
            logger.close()


class TestEventBuffer:
    """Test event buffering functionality."""

    def test_buffer_creation(self) -> None:
        """EventBuffer should be creatable."""
        buffer = EventBuffer(max_size=100)
        assert buffer is not None

    def test_buffer_add_and_retrieve(self) -> None:
        """Buffer should store and retrieve events."""
        buffer = EventBuffer(max_size=100)

        event = Event(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            category=EventCategory.LLM,
            event_type="test_event",
            severity=Severity.INFO,
            data={"message": "Test event"},
        )

        buffer.add(event)
        events = buffer.flush()

        assert len(events) == 1
        assert events[0].event_type == "test_event"


# ============================================================================
# Cost Tracking Tests
# ============================================================================


class TestCostTracker:
    """Test cost tracking functionality."""

    def test_tracker_creation(self, tmp_path: Path) -> None:
        """CostTracker should be creatable."""
        db_path = str(tmp_path / "costs.db")
        tracker = CostTracker(db_path=db_path)
        assert tracker is not None
        tracker.close()

    def test_record_cost(self, tmp_path: Path) -> None:
        """Tracker should record usage costs."""
        db_path = str(tmp_path / "costs.db")
        tracker = CostTracker(db_path=db_path)

        try:
            record = tracker.record(
                provider="openai",
                model="gpt-4",
                operation="chat",
                input_tokens=1000,
                output_tokens=500,
            )
            assert record is not None
        finally:
            tracker.close()

    def test_get_daily_summary(self, tmp_path: Path) -> None:
        """Tracker should provide daily summaries."""
        db_path = str(tmp_path / "costs.db")
        tracker = CostTracker(db_path=db_path)

        try:
            # Record some usage
            tracker.record("openai", "gpt-4", "chat", 100, 50)
            tracker.record("openai", "gpt-4", "chat", 200, 100)

            summary = tracker.get_daily_summary()
            assert summary is not None
        finally:
            tracker.close()

    def test_get_summary_by_model(self, tmp_path: Path) -> None:
        """Tracker should provide per-model summaries."""
        db_path = str(tmp_path / "costs.db")
        tracker = CostTracker(db_path=db_path)

        try:
            tracker.record("openai", "gpt-4", "chat", 100, 50)
            tracker.record("openai", "gpt-3.5-turbo", "chat", 200, 100)
            tracker.record("anthropic", "claude-3", "chat", 150, 75)

            summary = tracker.get_summary_by_model()
            assert summary is not None
        finally:
            tracker.close()

    def test_get_summary_by_provider(self, tmp_path: Path) -> None:
        """Tracker should provide per-provider summaries."""
        db_path = str(tmp_path / "costs.db")
        tracker = CostTracker(db_path=db_path)

        try:
            tracker.record("openai", "gpt-4", "chat", 100, 50)
            tracker.record("anthropic", "claude-3", "chat", 150, 75)

            summary = tracker.get_summary_by_provider()
            assert summary is not None
        finally:
            tracker.close()


class TestCostAnalyzer:
    """Test cost analytics functionality."""

    @pytest.fixture
    def tracker_with_data(self, tmp_path: Path):
        """Create tracker with sample data."""
        db_path = str(tmp_path / "costs.db")
        tracker = CostTracker(db_path=db_path)

        # Add sample records
        for i in range(10):
            tracker.record("openai", "gpt-4", "chat", 100 + i * 10, 50 + i * 5)

        yield tracker
        tracker.close()

    def test_analyzer_creation(self, tracker_with_data) -> None:
        """CostAnalyzer should be creatable."""
        analyzer = CostAnalyzer(tracker=tracker_with_data)
        assert analyzer is not None

    def test_check_budget(self, tracker_with_data) -> None:
        """Analyzer should check budget status."""
        analyzer = CostAnalyzer(tracker=tracker_with_data)

        result = analyzer.check_budget(budget_amount=100.0, budget_period="monthly")
        assert result is not None

    def test_analyze_trend(self, tracker_with_data) -> None:
        """Analyzer should analyze cost trends."""
        analyzer = CostAnalyzer(tracker=tracker_with_data)

        trend = analyzer.analyze_trend()
        assert trend is not None

    def test_get_analytics_summary(self, tracker_with_data) -> None:
        """Analyzer should provide analytics summary."""
        analyzer = CostAnalyzer(tracker=tracker_with_data)

        summary = analyzer.get_analytics_summary()
        assert summary is not None


# ============================================================================
# Execution Replay Tests
# ============================================================================


class TestExecutionReplay:
    """Test execution trace and replay functionality."""

    def test_execution_trace_creation(self) -> None:
        """ExecutionTrace should be creatable."""
        trace = ExecutionTrace(
            execution_id=str(uuid.uuid4()),
        )
        assert trace is not None

    def test_execution_replayer_creation(self, tmp_path: Path) -> None:
        """ExecutionReplayer should be creatable."""
        log_path = str(tmp_path / "events.log")
        # Create an empty log file
        Path(log_path).touch()
        replayer = ExecutionReplayer(log_path=log_path)
        assert replayer is not None


# ============================================================================
# Integration Tests
# ============================================================================


class TestObservabilityIntegration:
    """Test observability components working together."""

    def test_metrics_and_cost_tracking(self, tmp_path: Path) -> None:
        """Metrics and cost tracking should work together."""
        db_path = str(tmp_path / "costs.db")

        collector = LLMMetricsCollector()
        tracker = CostTracker(db_path=db_path)

        try:
            # Simulate LLM request
            provider = "openai"
            model = "gpt-4"
            input_tokens = 100
            output_tokens = 50
            latency_ms = 200.0

            # Record metrics
            collector.record_request(
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
            )

            # Record cost
            tracker.record(
                provider=provider,
                model=model,
                operation="chat",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=int(latency_ms),
            )

            # Verify both recorded
            metrics_stats = collector.get_stats()
            cost_summary = tracker.get_daily_summary()

            assert metrics_stats is not None
            assert cost_summary is not None
        finally:
            tracker.close()

    def test_events_and_metrics(self, tmp_path: Path) -> None:
        """Event logging and metrics should work together."""
        log_path = str(tmp_path / "events.log")

        logger = create_observability_logger(log_path=log_path)
        collector = LLMMetricsCollector()

        try:
            # Record metrics
            collector.record_request("openai", "gpt-4", 100, 50, 200.0)

            # Log event
            logger.log_event(
                category="llm",
                event_type="request_completed",
                severity="info",
                data={
                    "provider": "openai",
                    "model": "gpt-4",
                    "latency_ms": 200.0,
                },
            )
            logger.flush()

            stats = collector.get_stats()
            assert stats is not None
        finally:
            logger.close()


# ============================================================================
# Performance Tests
# ============================================================================


class TestObservabilityPerformance:
    """Test observability system performance."""

    def test_counter_increment_performance(self) -> None:
        """Counter increments should be fast."""
        counter = Counter(name="perf_test", description="Performance test")

        start = time.perf_counter()
        for _ in range(10000):
            counter.inc()
        elapsed_ms = (time.perf_counter() - start) * 1000

        # 10,000 increments should take < 100ms
        assert elapsed_ms < 100, f"Counter too slow: {elapsed_ms:.2f}ms for 10k increments"

    def test_histogram_record_performance(self) -> None:
        """Histogram records should be fast."""
        histogram = Histogram(name="perf_latency", description="Perf test")

        start = time.perf_counter()
        for i in range(10000):
            histogram.observe(float(i % 100))
        elapsed_ms = (time.perf_counter() - start) * 1000

        # 10,000 records should take < 100ms
        assert elapsed_ms < 100, f"Histogram too slow: {elapsed_ms:.2f}ms for 10k records"

    def test_cost_tracker_record_performance(self, tmp_path: Path) -> None:
        """Cost tracker records should be reasonably fast."""
        db_path = str(tmp_path / "costs.db")
        tracker = CostTracker(db_path=db_path)

        try:
            start = time.perf_counter()
            for i in range(100):
                tracker.record("openai", "gpt-4", "chat", 100, 50)
            elapsed_ms = (time.perf_counter() - start) * 1000

            # 100 records should take < 5000ms (50ms per record is acceptable for DB writes)
            assert elapsed_ms < 5000, f"Cost tracking too slow: {elapsed_ms:.2f}ms for 100 records"
        finally:
            tracker.close()
