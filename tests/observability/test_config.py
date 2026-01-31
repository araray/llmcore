# tests/observability/test_config.py
"""Tests for observability configuration module."""

from typing import Any, Dict

import pytest

from llmcore.observability.config import (
    BufferConfig,
    EventRotationConfig,
    # Sub-configs
    EventsConfig,
    MetricsConfig,
    # Main config
    ObservabilityConfig,
    PerformanceConfig,
    ReplayConfig,
    # Enums
    RotationStrategy,
    Severity,
    SinksConfig,
    get_default_config,
    # Loaders
    load_observability_config,
    to_legacy_config,
)

# =============================================================================
# ENUM TESTS
# =============================================================================


class TestRotationStrategy:
    """Tests for RotationStrategy enum."""

    def test_values(self):
        """Test enum values."""
        assert RotationStrategy.NONE.value == "none"
        assert RotationStrategy.DAILY.value == "daily"
        assert RotationStrategy.SIZE.value == "size"
        assert RotationStrategy.BOTH.value == "both"

    def test_from_string(self):
        """Test creating from string."""
        assert RotationStrategy("none") == RotationStrategy.NONE
        assert RotationStrategy("size") == RotationStrategy.SIZE


class TestSeverity:
    """Tests for Severity enum."""

    def test_values(self):
        """Test enum values."""
        assert Severity.DEBUG.value == "debug"
        assert Severity.INFO.value == "info"
        assert Severity.WARNING.value == "warning"
        assert Severity.ERROR.value == "error"
        assert Severity.CRITICAL.value == "critical"


# =============================================================================
# SUB-CONFIG TESTS
# =============================================================================


class TestEventRotationConfig:
    """Tests for EventRotationConfig."""

    def test_defaults(self):
        """Test default values."""
        config = EventRotationConfig()

        assert config.strategy == RotationStrategy.SIZE
        assert config.max_size_mb == 100
        assert config.max_files == 10
        assert config.compress is True

    def test_from_dict(self):
        """Test creating from dict."""
        config = EventRotationConfig(
            strategy="both",
            max_size_mb=50,
            max_files=5,
            compress=False,
        )

        assert config.strategy == RotationStrategy.BOTH
        assert config.max_size_mb == 50
        assert config.max_files == 5
        assert config.compress is False

    def test_strategy_validation(self):
        """Test strategy string validation."""
        config = EventRotationConfig(strategy="daily")
        assert config.strategy == RotationStrategy.DAILY

        config = EventRotationConfig(strategy="NONE")
        assert config.strategy == RotationStrategy.NONE


class TestEventsConfig:
    """Tests for EventsConfig."""

    def test_defaults(self):
        """Test default values."""
        config = EventsConfig()

        assert config.enabled is True
        assert config.log_path == "~/.llmcore/events.jsonl"
        assert config.min_severity == Severity.INFO
        assert config.categories == []
        assert isinstance(config.rotation, EventRotationConfig)

    def test_from_dict(self):
        """Test creating from dict."""
        config = EventsConfig(
            enabled=False,
            log_path="/var/log/events.jsonl",
            min_severity="warning",
            categories=["cognitive", "hitl"],
        )

        assert config.enabled is False
        assert config.log_path == "/var/log/events.jsonl"
        assert config.min_severity == Severity.WARNING
        assert config.categories == ["cognitive", "hitl"]

    def test_nested_rotation(self):
        """Test nested rotation config."""
        config = EventsConfig(rotation={"strategy": "both", "max_size_mb": 10})

        assert config.rotation.strategy == RotationStrategy.BOTH
        assert config.rotation.max_size_mb == 10

    def test_log_path_expanded(self):
        """Test log path expansion."""
        config = EventsConfig(log_path="~/.llmcore/test.jsonl")
        expanded = config.log_path_expanded

        assert "~" not in str(expanded)
        assert expanded.name == "test.jsonl"


class TestBufferConfig:
    """Tests for BufferConfig."""

    def test_defaults(self):
        """Test default values."""
        config = BufferConfig()

        assert config.enabled is True
        assert config.size == 100
        assert config.flush_interval_seconds == 5.0
        assert config.flush_on_shutdown is True

    def test_from_dict(self):
        """Test creating from dict."""
        config = BufferConfig(
            enabled=False,
            size=50,
            flush_interval_seconds=10.0,
            flush_on_shutdown=False,
        )

        assert config.enabled is False
        assert config.size == 50
        assert config.flush_interval_seconds == 10.0
        assert config.flush_on_shutdown is False


class TestMetricsConfig:
    """Tests for MetricsConfig."""

    def test_defaults(self):
        """Test default values."""
        config = MetricsConfig()

        assert config.enabled is True
        assert config.collect == []
        assert config.track_cost is True
        assert config.track_tokens is True
        assert config.latency_percentiles == [50, 90, 95, 99]

    def test_from_dict(self):
        """Test creating from dict."""
        config = MetricsConfig(
            enabled=False,
            track_cost=False,
            track_tokens=False,
            latency_percentiles=[50, 99],
        )

        assert config.enabled is False
        assert config.track_cost is False
        assert config.track_tokens is False
        assert config.latency_percentiles == [50, 99]

    def test_percentiles_validation(self):
        """Test percentile validation."""
        # Should sort and dedupe
        config = MetricsConfig(latency_percentiles=[99, 50, 50, 90])
        assert config.latency_percentiles == [50, 90, 99]

    def test_invalid_percentile(self):
        """Test invalid percentile raises error."""
        with pytest.raises(ValueError, match="between 0 and 100"):
            MetricsConfig(latency_percentiles=[50, 150])


class TestReplayConfig:
    """Tests for ReplayConfig."""

    def test_defaults(self):
        """Test default values."""
        config = ReplayConfig()

        assert config.enabled is True
        assert config.cache_enabled is True
        assert config.cache_max_executions == 50

    def test_from_dict(self):
        """Test creating from dict."""
        config = ReplayConfig(
            enabled=False,
            cache_enabled=False,
            cache_max_executions=100,
        )

        assert config.enabled is False
        assert config.cache_enabled is False
        assert config.cache_max_executions == 100


class TestSinksConfig:
    """Tests for SinksConfig."""

    def test_defaults(self):
        """Test default values."""
        config = SinksConfig()

        assert config.file_enabled is True
        assert config.memory_enabled is False
        assert config.memory_max_events == 1000
        assert config.callback_enabled is False

    def test_from_dict(self):
        """Test creating from dict."""
        config = SinksConfig(
            file_enabled=False,
            memory_enabled=True,
            memory_max_events=500,
            callback_enabled=True,
        )

        assert config.file_enabled is False
        assert config.memory_enabled is True
        assert config.memory_max_events == 500
        assert config.callback_enabled is True


class TestPerformanceConfig:
    """Tests for PerformanceConfig."""

    def test_defaults(self):
        """Test default values."""
        config = PerformanceConfig()

        assert config.async_logging is True
        assert config.sampling_rate == 1.0
        assert config.max_event_data_bytes == 10000
        assert config.overhead_warning_threshold_percent == 5.0

    def test_from_dict(self):
        """Test creating from dict."""
        config = PerformanceConfig(
            async_logging=False,
            sampling_rate=0.5,
            max_event_data_bytes=5000,
            overhead_warning_threshold_percent=10.0,
        )

        assert config.async_logging is False
        assert config.sampling_rate == 0.5
        assert config.max_event_data_bytes == 5000
        assert config.overhead_warning_threshold_percent == 10.0

    def test_sampling_rate_bounds(self):
        """Test sampling rate bounds."""
        # Valid
        config = PerformanceConfig(sampling_rate=0.0)
        assert config.sampling_rate == 0.0

        config = PerformanceConfig(sampling_rate=1.0)
        assert config.sampling_rate == 1.0

        # Invalid
        with pytest.raises(ValueError):
            PerformanceConfig(sampling_rate=-0.1)

        with pytest.raises(ValueError):
            PerformanceConfig(sampling_rate=1.1)


# =============================================================================
# MAIN CONFIG TESTS
# =============================================================================


class TestObservabilityConfig:
    """Tests for ObservabilityConfig."""

    def test_defaults(self):
        """Test default values."""
        config = ObservabilityConfig()

        assert config.enabled is True
        assert isinstance(config.events, EventsConfig)
        assert isinstance(config.buffer, BufferConfig)
        assert isinstance(config.metrics, MetricsConfig)
        assert isinstance(config.replay, ReplayConfig)
        assert isinstance(config.sinks, SinksConfig)
        assert isinstance(config.performance, PerformanceConfig)

    def test_from_dict(self):
        """Test creating from nested dict."""
        config = ObservabilityConfig(
            enabled=True,
            events={"enabled": False, "log_path": "/tmp/events.jsonl"},
            metrics={"track_cost": False},
            replay={"cache_max_executions": 100},
        )

        assert config.events.enabled is False
        assert config.events.log_path == "/tmp/events.jsonl"
        assert config.metrics.track_cost is False
        assert config.replay.cache_max_executions == 100

    def test_helper_methods(self):
        """Test helper methods."""
        # All enabled
        config = ObservabilityConfig()
        assert config.is_event_logging_enabled() is True
        assert config.is_metrics_enabled() is True
        assert config.is_replay_enabled() is True
        assert config.should_track_cost() is True
        assert config.should_track_tokens() is True
        assert config.get_latency_percentiles() == [50, 90, 95, 99]

        # Master disabled
        config = ObservabilityConfig(enabled=False)
        assert config.is_event_logging_enabled() is False
        assert config.is_metrics_enabled() is False
        assert config.should_track_cost() is False

        # Events disabled
        config = ObservabilityConfig(events={"enabled": False})
        assert config.is_event_logging_enabled() is False

        # Metrics disabled
        config = ObservabilityConfig(metrics={"enabled": False})
        assert config.is_metrics_enabled() is False
        assert config.should_track_cost() is False
        assert config.get_latency_percentiles() == []


# =============================================================================
# LOADER TESTS
# =============================================================================


class TestLoadObservabilityConfig:
    """Tests for load_observability_config function."""

    def test_with_none(self):
        """Test loading with None returns defaults."""
        config = load_observability_config(None)

        assert isinstance(config, ObservabilityConfig)
        assert config.enabled is True

    def test_with_empty_dict(self):
        """Test loading with empty dict."""
        config = load_observability_config({})

        assert isinstance(config, ObservabilityConfig)
        assert config.enabled is True

    def test_with_nested_dict(self):
        """Test loading from nested config dict."""
        config_dict = {
            "agents": {
                "observability": {
                    "enabled": True,
                    "events": {
                        "enabled": False,
                        "log_path": "/var/log/test.jsonl",
                    },
                    "metrics": {
                        "track_cost": False,
                    },
                }
            }
        }

        config = load_observability_config(config_dict)

        assert config.events.enabled is False
        assert config.events.log_path == "/var/log/test.jsonl"
        assert config.metrics.track_cost is False

    def test_custom_section_path(self):
        """Test loading with custom section path."""
        config_dict = {
            "custom": {
                "obs": {
                    "enabled": False,
                }
            }
        }

        config = load_observability_config(config_dict, section_path="custom.obs")

        assert config.enabled is False

    def test_invalid_path_returns_defaults(self):
        """Test invalid path returns defaults."""
        config_dict = {"other": {"section": {}}}

        config = load_observability_config(config_dict)

        assert isinstance(config, ObservabilityConfig)
        assert config.enabled is True  # Default


class TestGetDefaultConfig:
    """Tests for get_default_config function."""

    def test_returns_defaults(self):
        """Test returns default config."""
        config = get_default_config()

        assert isinstance(config, ObservabilityConfig)
        assert config.enabled is True
        assert config.metrics.track_cost is True


class TestToLegacyConfig:
    """Tests for to_legacy_config function."""

    def test_converts_to_flat_dict(self):
        """Test conversion to flat dict."""
        config = ObservabilityConfig(
            enabled=True,
            events={"log_path": "/tmp/test.jsonl"},
            metrics={"track_cost": False},
        )

        legacy = to_legacy_config(config)

        assert legacy["enabled"] is True
        assert legacy["events_enabled"] is True
        assert legacy["log_path"] == "/tmp/test.jsonl"
        assert legacy["track_cost"] is False
        assert legacy["replay_enabled"] is True
        assert legacy["file_enabled"] is True

    def test_all_keys_present(self):
        """Test all expected keys are present."""
        config = ObservabilityConfig()
        legacy = to_legacy_config(config)

        expected_keys = [
            "enabled",
            "events_enabled",
            "log_path",
            "min_severity",
            "categories",
            "rotation",
            "buffer",
            "async_logging",
            "sampling_rate",
            "max_event_data_bytes",
            "metrics_enabled",
            "track_cost",
            "track_tokens",
            "latency_percentiles",
            "replay_enabled",
            "cache_enabled",
            "cache_max_executions",
            "file_enabled",
            "memory_enabled",
            "memory_max_events",
            "callback_enabled",
        ]

        for key in expected_keys:
            assert key in legacy, f"Missing key: {key}"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestConfigIntegration:
    """Integration tests for config loading."""

    def test_full_toml_structure(self):
        """Test loading full TOML-like structure."""
        # Simulates what confy would produce from the TOML
        config_dict = {
            "agents": {
                "observability": {
                    "enabled": True,
                    "events": {
                        "enabled": True,
                        "log_path": "/av/logs/llmcore/events.jsonl",
                        "min_severity": "info",
                        "categories": [],
                        "rotation": {
                            "strategy": "both",
                            "max_size_mb": 10,
                            "max_files": 0,
                            "compress": True,
                        },
                    },
                    "buffer": {
                        "enabled": True,
                        "size": 100,
                        "flush_interval_seconds": 5,
                        "flush_on_shutdown": True,
                    },
                    "metrics": {
                        "enabled": True,
                        "collect": [],
                        "track_cost": True,
                        "track_tokens": True,
                        "latency_percentiles": [50, 90, 95, 99],
                    },
                    "replay": {
                        "enabled": True,
                        "cache_enabled": True,
                        "cache_max_executions": 50,
                    },
                    "sinks": {
                        "file_enabled": True,
                        "memory_enabled": False,
                        "memory_max_events": 1000,
                        "callback_enabled": False,
                    },
                    "performance": {
                        "async_logging": True,
                        "sampling_rate": 1.0,
                        "max_event_data_bytes": 10000,
                        "overhead_warning_threshold_percent": 5,
                    },
                }
            }
        }

        config = load_observability_config(config_dict)

        # Verify all sections loaded correctly
        assert config.enabled is True

        # Events
        assert config.events.enabled is True
        assert config.events.log_path == "/av/logs/llmcore/events.jsonl"
        assert config.events.rotation.strategy == RotationStrategy.BOTH
        assert config.events.rotation.max_size_mb == 10

        # Buffer
        assert config.buffer.enabled is True
        assert config.buffer.size == 100

        # Metrics
        assert config.metrics.track_cost is True
        assert config.metrics.latency_percentiles == [50, 90, 95, 99]

        # Replay
        assert config.replay.cache_max_executions == 50

        # Sinks
        assert config.sinks.file_enabled is True
        assert config.sinks.memory_enabled is False

        # Performance
        assert config.performance.sampling_rate == 1.0
