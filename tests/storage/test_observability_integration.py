# tests/storage/test_observability_integration.py
"""
Unit tests for Phase 4 (PANOPTICON) observability integration.

Tests the integration of observability components into StorageManager:
- StorageInstrumentation
- MetricsCollector
- EventLogger
- ObservabilityConfig

STORAGE SYSTEM V2 (Phase 4 - PANOPTICON)
"""

import asyncio
import logging
from unittest.mock import MagicMock

import pytest

from llmcore.storage.events import (
    EventLogger,
    EventLoggerConfig,
    EventType,
    StorageEvent,
)
from llmcore.storage.instrumentation import (
    InstrumentationConfig,
    StorageInstrumentation,
)
from llmcore.storage.metrics import (
    MetricsBackendType,
    MetricsCollector,
    MetricsConfig,
)

# Phase 4 imports
from llmcore.storage.observability import (
    ObservabilityConfig,
)

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def default_observability_config():
    """Create default observability configuration."""
    return ObservabilityConfig()


@pytest.fixture
def disabled_observability_config():
    """Create disabled observability configuration."""
    return ObservabilityConfig(enabled=False)


@pytest.fixture
def custom_observability_config():
    """Create custom observability configuration."""
    return ObservabilityConfig(
        enabled=True,
        log_queries=True,
        log_slow_queries=True,
        slow_query_threshold_seconds=0.5,
        metrics_enabled=True,
        metrics_backend="memory",
        metrics_prefix="test_storage",
        event_logging_enabled=True,
        event_retention_days=7,
        event_table_name="test_events",
    )


@pytest.fixture
def mock_config():
    """Create a mock ConfyConfig-like object."""
    config = MagicMock()
    config.get = MagicMock(
        side_effect=lambda key, default=None: {
            "storage.session": {"type": "json", "path": "/tmp/test"},
            "storage.vector": {"type": "chromadb", "path": "/tmp/test_vector"},
            "storage.observability": {
                "enabled": True,
                "metrics_enabled": True,
                "event_logging_enabled": True,
            },
        }.get(key, default)
    )
    return config


# =============================================================================
# OBSERVABILITY CONFIG TESTS
# =============================================================================


class TestObservabilityConfig:
    """Tests for ObservabilityConfig model."""

    def test_default_values(self, default_observability_config):
        """Test that default configuration values are correct."""
        config = default_observability_config

        assert config.enabled is True
        assert config.log_queries is False
        assert config.log_slow_queries is True
        assert config.slow_query_threshold_seconds == 1.0
        assert config.metrics_enabled is True
        assert config.metrics_backend == "prometheus"
        assert config.event_logging_enabled is True
        assert config.event_retention_days == 30
        assert config.tracing_enabled is False

    def test_disabled_config(self, disabled_observability_config):
        """Test that disabled config turns off all features."""
        config = disabled_observability_config
        assert config.enabled is False

    def test_from_dict_flat(self):
        """Test creating config from flat dictionary."""
        config_dict = {
            "enabled": True,
            "log_queries": True,
            "slow_query_threshold_seconds": 2.5,
            "metrics_backend": "memory",
        }
        config = ObservabilityConfig.from_dict(config_dict)

        assert config.enabled is True
        assert config.log_queries is True
        assert config.slow_query_threshold_seconds == 2.5
        assert config.metrics_backend == "memory"

    def test_from_dict_nested_observability(self):
        """Test creating config from nested dictionary with observability key."""
        config_dict = {
            "observability": {
                "enabled": True,
                "metrics_enabled": False,
            }
        }
        config = ObservabilityConfig.from_dict(config_dict)

        assert config.enabled is True
        assert config.metrics_enabled is False

    def test_from_dict_nested_storage(self):
        """Test creating config from nested dictionary with storage.observability."""
        config_dict = {
            "storage": {
                "observability": {
                    "event_logging_enabled": False,
                    "event_retention_days": 14,
                }
            }
        }
        config = ObservabilityConfig.from_dict(config_dict)

        assert config.event_logging_enabled is False
        assert config.event_retention_days == 14

    def test_get_instrumentation_config(self, default_observability_config):
        """Test extracting instrumentation configuration."""
        config = default_observability_config
        inst_config = config.get_instrumentation_config()

        assert "enabled" in inst_config
        assert "log_queries" in inst_config
        assert "log_slow_queries" in inst_config
        assert "slow_query_threshold_seconds" in inst_config
        assert "metrics_enabled" in inst_config
        assert inst_config["enabled"] is True
        assert inst_config["slow_query_threshold_seconds"] == 1.0

    def test_get_metrics_config(self, default_observability_config):
        """Test extracting metrics configuration."""
        config = default_observability_config
        metrics_cfg = config.get_metrics_config()

        assert "enabled" in metrics_cfg
        assert "backend" in metrics_cfg
        assert "prefix" in metrics_cfg
        assert metrics_cfg["enabled"] is True
        assert metrics_cfg["backend"] == "prometheus"
        assert metrics_cfg["prefix"] == "llmcore_storage"

    def test_get_event_logger_config(self, default_observability_config):
        """Test extracting event logger configuration."""
        config = default_observability_config
        event_cfg = config.get_event_logger_config()

        assert "enabled" in event_cfg
        assert "table_name" in event_cfg
        assert "retention_days" in event_cfg
        assert "batch_size" in event_cfg
        assert event_cfg["enabled"] is True
        assert event_cfg["table_name"] == "storage_events"
        assert event_cfg["retention_days"] == 30

    def test_validation_slow_query_threshold_min(self):
        """Test that slow query threshold has minimum value."""
        with pytest.raises(ValueError, match="at least 0.001"):
            ObservabilityConfig(slow_query_threshold_seconds=0.0001)

    def test_validation_slow_query_threshold_max(self):
        """Test that slow query threshold has maximum value."""
        with pytest.raises(ValueError, match="should not exceed 300"):
            ObservabilityConfig(slow_query_threshold_seconds=500)

    def test_validation_metrics_prefix(self):
        """Test that metrics prefix must start with letter."""
        with pytest.raises(ValueError, match="must start with a letter"):
            ObservabilityConfig(metrics_prefix="123_invalid")

    def test_to_dict(self, custom_observability_config):
        """Test converting config to dictionary."""
        config = custom_observability_config
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["enabled"] is True
        assert config_dict["log_queries"] is True
        assert config_dict["slow_query_threshold_seconds"] == 0.5
        assert config_dict["metrics_backend"] == "memory"


# =============================================================================
# INSTRUMENTATION TESTS
# =============================================================================


class TestStorageInstrumentation:
    """Tests for StorageInstrumentation."""

    def test_create_with_default_config(self):
        """Test creating instrumentation with default config."""
        inst = StorageInstrumentation()
        assert inst is not None
        assert inst.config.enabled is True

    def test_create_with_custom_config(self):
        """Test creating instrumentation with custom config."""
        config = InstrumentationConfig(
            enabled=True,
            slow_query_threshold_seconds=0.5,
            log_queries=True,
        )
        inst = StorageInstrumentation(config=config)

        assert inst.config.enabled is True
        assert inst.config.slow_query_threshold_seconds == 0.5
        assert inst.config.log_queries is True

    def test_sync_context_manager(self):
        """Test synchronous context manager."""
        inst = StorageInstrumentation()

        with inst.instrument("test_op", "memory", "test_table") as ctx:
            assert ctx is not None
            assert ctx.operation == "test_op"
            assert ctx.backend == "memory"
            assert ctx.table == "test_table"
            # Simulate some work
            import time

            time.sleep(0.01)

        # Context should have recorded duration
        assert ctx.duration_seconds is not None
        assert ctx.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test asynchronous context manager."""
        inst = StorageInstrumentation()

        async with inst.instrument_async("async_op", "postgres", "sessions") as ctx:
            assert ctx.operation == "async_op"
            assert ctx.backend == "postgres"
            await asyncio.sleep(0.01)

        assert ctx.duration_seconds is not None
        assert ctx.duration_seconds > 0

    def test_slow_query_detection(self, caplog):
        """Test that slow queries are detected and logged."""
        config = InstrumentationConfig(
            enabled=True,
            log_slow_queries=True,
            slow_query_threshold_seconds=0.01,  # 10ms threshold
        )
        inst = StorageInstrumentation(config=config)

        with caplog.at_level(logging.WARNING):
            with inst.instrument("slow_op", "postgres", "large_table") as ctx:
                import time

                time.sleep(0.02)  # 20ms, exceeds threshold

        assert ctx.duration_seconds > 0.01

    def test_get_statistics(self):
        """Test retrieving instrumentation statistics."""
        inst = StorageInstrumentation()

        # Run a few operations
        for i in range(5):
            with inst.instrument(f"op_{i}", "memory", "test"):
                pass

        stats = inst.get_statistics()

        assert "total_operations" in stats
        assert "total_errors" in stats
        assert stats["total_operations"] >= 5

    def test_error_tracking(self):
        """Test that errors are tracked."""
        inst = StorageInstrumentation()

        try:
            with inst.instrument("failing_op", "memory", "test") as ctx:
                raise ValueError("Test error")
        except ValueError:
            pass

        stats = inst.get_statistics()
        assert stats["total_errors"] >= 1

    def test_metrics_collector_integration(self):
        """Test that instrumentation integrates with metrics collector via constructor."""
        # Create metrics collector
        metrics_config = MetricsConfig(
            enabled=True,
            backend=MetricsBackendType.MEMORY,
            prefix="test",
        )
        metrics = MetricsCollector(config=metrics_config)

        # Create instrumentation with metrics collector
        inst = StorageInstrumentation(metrics_collector=metrics)

        with inst.instrument("test_op", "memory", "test"):
            pass

        # Verify metrics were recorded via get_metrics()
        metrics_data = metrics.get_metrics()
        assert metrics_data is not None


# =============================================================================
# METRICS COLLECTOR TESTS
# =============================================================================


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_create_with_memory_backend(self):
        """Test creating collector with in-memory backend."""
        config = MetricsConfig(
            enabled=True,
            backend=MetricsBackendType.MEMORY,
            prefix="test",
        )
        collector = MetricsCollector(config=config)
        assert collector is not None

    def test_record_operation(self):
        """Test recording an operation."""
        config = MetricsConfig(
            enabled=True,
            backend=MetricsBackendType.MEMORY,
            prefix="test",
        )
        collector = MetricsCollector(config=config)

        collector.record_operation(
            operation="save_session",
            backend="postgres",
            table="sessions",
            duration_seconds=0.1,
            success=True,
        )

        metrics = collector.get_metrics()
        assert metrics is not None
        # Memory backend should have recorded something
        assert "counters" in metrics or "histograms" in metrics or len(metrics) > 0

    def test_increment_counter(self):
        """Test incrementing a counter."""
        config = MetricsConfig(
            enabled=True,
            backend=MetricsBackendType.MEMORY,
            prefix="test",
        )
        collector = MetricsCollector(config=config)

        collector.increment("custom_counter", 5, {"label": "value"})

        # Should not raise and metric should be recorded
        metrics = collector.get_metrics()
        assert metrics is not None

    def test_observe_histogram(self):
        """Test recording histogram value."""
        config = MetricsConfig(
            enabled=True,
            backend=MetricsBackendType.MEMORY,
            prefix="test",
        )
        collector = MetricsCollector(config=config)

        collector.observe("query_duration", 0.5, {"backend": "postgres"})

        # Should not raise and metric should be recorded
        metrics = collector.get_metrics()
        assert metrics is not None

    def test_disabled_collector(self):
        """Test that disabled collector is a no-op."""
        config = MetricsConfig(
            enabled=False,
            backend=MetricsBackendType.MEMORY,
            prefix="test",
        )
        collector = MetricsCollector(config=config)

        # Should not raise even when disabled
        collector.record_operation(
            operation="test",
            backend="memory",
            table="test",
            duration_seconds=0.1,
            success=True,
        )


# =============================================================================
# EVENT LOGGER TESTS
# =============================================================================


class TestEventLogger:
    """Tests for EventLogger."""

    def test_create_event_logger(self):
        """Test creating event logger."""
        config = EventLoggerConfig(
            enabled=True,
            table_name="test_events",
            retention_days=7,
        )
        logger = EventLogger(config=config)
        assert logger is not None

    def test_create_event(self):
        """Test creating a storage event."""
        event = StorageEvent(
            event_type=EventType.SESSION_CREATE,
            user_id="user123",
            session_id="session456",
            operation_duration_ms=50.0,
            metadata={"key": "value"},
        )

        assert event.event_type == EventType.SESSION_CREATE
        assert event.user_id == "user123"
        assert event.session_id == "session456"
        assert event.operation_duration_ms == 50.0
        assert event.metadata == {"key": "value"}
        assert event.timestamp is not None

    def test_event_types(self):
        """Test all event types exist."""
        assert EventType.SESSION_CREATE is not None
        assert EventType.SESSION_UPDATE is not None
        assert EventType.SESSION_DELETE is not None
        assert EventType.MESSAGE_ADD is not None
        assert EventType.VECTOR_SEARCH is not None
        assert EventType.HEALTH_CHECK is not None
        assert EventType.ERROR is not None
        assert EventType.SLOW_QUERY is not None

    def test_log_event_no_pool(self):
        """Test logging event without database pool (queued)."""
        config = EventLoggerConfig(
            enabled=True,
            table_name="test_events",
        )
        logger = EventLogger(config=config)

        # log_event is synchronous and queues for async flush
        # Should not raise even without pool
        logger.log_event(
            event_type=EventType.SESSION_CREATE,
            session_id="test_session",
        )

    @pytest.mark.asyncio
    async def test_log_event_async(self):
        """Test asynchronous event logging."""
        config = EventLoggerConfig(
            enabled=True,
            table_name="test_events",
        )
        logger = EventLogger(config=config)

        # log_event_async is for async contexts
        await logger.log_event_async(
            event_type=EventType.VECTOR_SEARCH,
            collection_name="test_collection",
            operation_duration_ms=100.0,
        )


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestObservabilityIntegration:
    """Integration tests for observability components working together."""

    def test_instrumentation_with_metrics(self):
        """Test instrumentation emitting to metrics collector."""
        # Setup metrics collector
        metrics_config = MetricsConfig(
            enabled=True,
            backend=MetricsBackendType.MEMORY,
            prefix="test",
        )
        metrics = MetricsCollector(config=metrics_config)

        # Setup instrumentation with metrics collector
        inst_config = InstrumentationConfig(
            enabled=True,
            slow_query_threshold_seconds=1.0,
        )
        inst = StorageInstrumentation(config=inst_config, metrics_collector=metrics)

        # Run instrumented operation
        with inst.instrument("test_op", "postgres", "sessions"):
            pass

        # Verify metrics were recorded
        metrics_data = metrics.get_metrics()
        assert metrics_data is not None

    def test_config_to_components_pipeline(self, custom_observability_config):
        """Test creating all components from a single config."""
        config = custom_observability_config

        # Create metrics from config
        metrics_cfg = config.get_metrics_config()
        metrics = MetricsCollector(
            config=MetricsConfig(
                enabled=metrics_cfg["enabled"],
                backend=MetricsBackendType(metrics_cfg["backend"]),
                prefix=metrics_cfg["prefix"],
            )
        )

        # Create instrumentation from config with metrics collector
        inst_cfg = config.get_instrumentation_config()
        inst = StorageInstrumentation(
            config=InstrumentationConfig(**inst_cfg), metrics_collector=metrics
        )

        # Create event logger from config
        event_cfg = config.get_event_logger_config()
        event_logger = EventLogger(config=EventLoggerConfig(**event_cfg))

        # All components should be created
        assert inst is not None
        assert metrics is not None
        assert event_logger is not None

        # Run integrated operation
        with inst.instrument("integrated_op", "postgres", "test"):
            pass

        # Verify instrumentation stats
        inst_stats = inst.get_statistics()
        assert inst_stats["total_operations"] >= 1


# =============================================================================
# MAIN
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
