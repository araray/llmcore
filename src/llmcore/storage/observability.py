# src/llmcore/storage/observability.py
"""
Observability Configuration for Phase 4 (PANOPTICON).

Provides unified configuration models for the observability layer:
- Instrumentation settings
- Metrics collection
- Event logging
- Distributed tracing

This module centralizes all observability configuration to enable
consistent setup across the storage system.

Usage:
    from llmcore.storage.observability import ObservabilityConfig

    config = ObservabilityConfig(
        log_slow_queries=True,
        metrics_enabled=True,
        event_logging_enabled=True,
    )

    # Or from TOML config
    [storage.observability]
    log_queries = false
    log_slow_queries = true
    slow_query_threshold_seconds = 1.0
    metrics_enabled = true
    metrics_backend = "prometheus"
    event_logging_enabled = true
    event_retention_days = 30

STORAGE SYSTEM V2 (Phase 4 - PANOPTICON):
- Unified observability configuration
- Pydantic model for validation
- TOML configuration support
- Environment variable overrides
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# =============================================================================
# OBSERVABILITY CONFIGURATION
# =============================================================================


class ObservabilityConfig(BaseModel):
    """
    Unified observability configuration for the storage system.

    Controls instrumentation, metrics, event logging, and tracing
    for all storage operations.

    Attributes:
        enabled: Master switch for all observability features.

        Logging:
            log_queries: Log all queries at DEBUG level.
            log_slow_queries: Log queries exceeding threshold at WARNING level.
            slow_query_threshold_seconds: Threshold for slow query detection.
            include_query_params: Include query parameters in logs (security risk).

        Metrics:
            metrics_enabled: Enable metrics collection.
            metrics_backend: Backend type (prometheus, statsd, none).
            metrics_prefix: Prefix for all metric names.
            metrics_port: Port for Prometheus HTTP endpoint.

        Event Logging:
            event_logging_enabled: Enable persistent event logging to database.
            event_retention_days: Days to retain events (0 = forever).
            event_table_name: Name of the events table.

        Tracing:
            tracing_enabled: Enable distributed tracing.
            tracing_backend: Backend type (opentelemetry, none).
            tracing_endpoint: OTLP collector endpoint.
            tracing_service_name: Service name for traces.
    """

    # Master switch
    enabled: bool = Field(default=True, description="Master switch for all observability features")

    # Logging configuration
    log_queries: bool = Field(default=False, description="Log all queries at DEBUG level (verbose)")
    log_slow_queries: bool = Field(
        default=True, description="Log queries exceeding threshold at WARNING level"
    )
    slow_query_threshold_seconds: float = Field(
        default=1.0, gt=0, description="Threshold in seconds for slow query detection"
    )
    include_query_params: bool = Field(
        default=False, description="Include query parameters in logs (security risk)"
    )

    # Metrics configuration
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    metrics_backend: Literal["prometheus", "statsd", "memory", "none"] = Field(
        default="prometheus", description="Metrics backend type"
    )
    metrics_prefix: str = Field(
        default="llmcore_storage", description="Prefix for all metric names"
    )
    metrics_port: int = Field(
        default=9090, ge=1024, le=65535, description="Port for Prometheus HTTP endpoint"
    )
    metrics_default_labels: dict[str, str] = Field(
        default_factory=dict, description="Default labels added to all metrics"
    )

    # Event logging configuration
    event_logging_enabled: bool = Field(
        default=True, description="Enable persistent event logging to database"
    )
    event_retention_days: int = Field(
        default=30, ge=0, description="Days to retain events (0 = forever)"
    )
    event_table_name: str = Field(default="storage_events", description="Name of the events table")
    event_batch_size: int = Field(
        default=100, ge=1, le=10000, description="Number of events to batch before flushing"
    )
    event_flush_interval_seconds: float = Field(
        default=5.0, gt=0, description="Maximum interval between event flushes"
    )

    # Tracing configuration
    tracing_enabled: bool = Field(default=False, description="Enable distributed tracing")
    tracing_backend: Literal["opentelemetry", "none"] = Field(
        default="none", description="Tracing backend type"
    )
    tracing_endpoint: str = Field(
        default="http://localhost:4317", description="OTLP collector endpoint"
    )
    tracing_service_name: str = Field(
        default="llmcore-storage", description="Service name for traces"
    )
    tracing_sample_rate: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Sampling rate for traces (1.0 = 100%)"
    )

    # Histogram bucket configuration
    histogram_buckets: list[float] = Field(
        default=[
            0.001,
            0.005,
            0.01,
            0.025,
            0.05,
            0.075,
            0.1,
            0.25,
            0.5,
            0.75,
            1.0,
            2.5,
            5.0,
            7.5,
            10.0,
        ],
        description="Default histogram bucket boundaries in seconds",
    )

    @field_validator("slow_query_threshold_seconds")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Validate slow query threshold is reasonable."""
        if v < 0.001:
            raise ValueError("slow_query_threshold_seconds must be at least 0.001")
        if v > 300:
            raise ValueError("slow_query_threshold_seconds should not exceed 300 seconds")
        return v

    @field_validator("metrics_prefix")
    @classmethod
    def validate_metrics_prefix(cls, v: str) -> str:
        """Validate metrics prefix format."""
        if not v:
            return v
        if not v[0].isalpha():
            raise ValueError("metrics_prefix must start with a letter")
        if not all(c.isalnum() or c == "_" for c in v):
            raise ValueError(
                "metrics_prefix must contain only alphanumeric characters and underscores"
            )
        return v

    @field_validator("event_table_name")
    @classmethod
    def validate_table_name(cls, v: str) -> str:
        """Validate event table name format."""
        if not v:
            raise ValueError("event_table_name cannot be empty")
        if not v[0].isalpha():
            raise ValueError("event_table_name must start with a letter")
        if not all(c.isalnum() or c == "_" for c in v):
            raise ValueError(
                "event_table_name must contain only alphanumeric characters and underscores"
            )
        return v

    @model_validator(mode="after")
    def validate_configuration(self) -> ObservabilityConfig:
        """Validate overall configuration consistency."""
        # If master switch is off, disable all features
        if not self.enabled:
            return self

        # Warn about verbose logging in production
        if self.log_queries and self.include_query_params:
            import warnings

            warnings.warn(
                "log_queries=True with include_query_params=True may expose "
                "sensitive data in logs. Use with caution in production.",
                UserWarning,
            )

        return self

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> ObservabilityConfig:
        """
        Create configuration from dictionary.

        Supports both flat and nested dictionary formats.

        Args:
            config: Configuration dictionary.

        Returns:
            ObservabilityConfig instance.
        """
        # Handle nested observability section
        if "observability" in config:
            config = config["observability"]
        elif "storage" in config and "observability" in config["storage"]:
            config = config["storage"]["observability"]

        return cls(**config)

    @classmethod
    def from_environment(cls, prefix: str = "LLMCORE_STORAGE_") -> ObservabilityConfig:
        """
        Create configuration from environment variables.

        Environment variables are expected in format:
        LLMCORE_STORAGE_LOG_QUERIES=true
        LLMCORE_STORAGE_METRICS_ENABLED=false

        Args:
            prefix: Environment variable prefix.

        Returns:
            ObservabilityConfig instance with env overrides.
        """
        config = {}

        # Map environment variables to config keys
        env_map = {
            "ENABLED": "enabled",
            "LOG_QUERIES": "log_queries",
            "LOG_SLOW_QUERIES": "log_slow_queries",
            "SLOW_QUERY_THRESHOLD_SECONDS": "slow_query_threshold_seconds",
            "INCLUDE_QUERY_PARAMS": "include_query_params",
            "METRICS_ENABLED": "metrics_enabled",
            "METRICS_BACKEND": "metrics_backend",
            "METRICS_PREFIX": "metrics_prefix",
            "METRICS_PORT": "metrics_port",
            "EVENT_LOGGING_ENABLED": "event_logging_enabled",
            "EVENT_RETENTION_DAYS": "event_retention_days",
            "EVENT_TABLE_NAME": "event_table_name",
            "TRACING_ENABLED": "tracing_enabled",
            "TRACING_BACKEND": "tracing_backend",
            "TRACING_ENDPOINT": "tracing_endpoint",
            "TRACING_SERVICE_NAME": "tracing_service_name",
            "TRACING_SAMPLE_RATE": "tracing_sample_rate",
        }

        for env_suffix, config_key in env_map.items():
            env_var = f"{prefix}{env_suffix}"
            value = os.environ.get(env_var)

            if value is not None:
                # Parse boolean values
                if value.lower() in ("true", "1", "yes"):
                    config[config_key] = True
                elif value.lower() in ("false", "0", "no"):
                    config[config_key] = False
                # Parse numeric values
                elif config_key.endswith("_seconds") or config_key.endswith("_rate"):
                    config[config_key] = float(value)
                elif (
                    config_key.endswith("_days")
                    or config_key.endswith("_port")
                    or config_key.endswith("_size")
                ):
                    config[config_key] = int(value)
                else:
                    config[config_key] = value

        return cls(**config)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()

    def get_instrumentation_config(self) -> dict[str, Any]:
        """
        Get configuration for StorageInstrumentation.

        Returns:
            Dictionary compatible with InstrumentationConfig.
        """
        return {
            "enabled": self.enabled,
            "log_queries": self.log_queries,
            "log_slow_queries": self.log_slow_queries,
            "slow_query_threshold_seconds": self.slow_query_threshold_seconds,
            "include_query_params": self.include_query_params,
            "metrics_enabled": self.metrics_enabled,
            "metrics_backend": self.metrics_backend,
            "tracing_enabled": self.tracing_enabled,
            "sample_rate": self.tracing_sample_rate if self.tracing_enabled else 1.0,
        }

    def get_metrics_config(self) -> dict[str, Any]:
        """
        Get configuration for MetricsCollector.

        Returns:
            Dictionary compatible with MetricsConfig.
        """
        return {
            "enabled": self.enabled and self.metrics_enabled,
            "backend": self.metrics_backend,
            "prefix": self.metrics_prefix,
            "default_labels": self.metrics_default_labels,
            "prometheus_port": self.metrics_port,
            "histogram_buckets": tuple(self.histogram_buckets),
        }

    def get_event_logger_config(self) -> dict[str, Any]:
        """
        Get configuration for EventLogger.

        Returns:
            Dictionary compatible with EventLoggerConfig.
        """
        return {
            "enabled": self.enabled and self.event_logging_enabled,
            "table_name": self.event_table_name,
            "retention_days": self.event_retention_days,
            "batch_size": self.event_batch_size,
            "flush_interval_seconds": self.event_flush_interval_seconds,
            "log_slow_queries": self.log_slow_queries,
            "log_errors": True,
            "include_metadata": True,
        }


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================


DEFAULT_OBSERVABILITY_CONFIG = ObservabilityConfig()


# =============================================================================
# TOML CONFIGURATION EXAMPLE
# =============================================================================

OBSERVABILITY_TOML_EXAMPLE = """
# Storage Observability Configuration
# Add this section to your llmcore.toml file

[storage.observability]
# Master switch for all observability features
enabled = true

# Logging
log_queries = false              # Log all queries at DEBUG level
log_slow_queries = true          # Log queries exceeding threshold at WARNING
slow_query_threshold_seconds = 1.0
include_query_params = false     # Security risk if enabled

# Metrics
metrics_enabled = true
metrics_backend = "prometheus"   # prometheus, statsd, memory, none
metrics_prefix = "llmcore_storage"
metrics_port = 9090              # For Prometheus scrape endpoint

# Event Logging (persistent audit trail)
event_logging_enabled = true
event_retention_days = 30        # 0 = keep forever
event_table_name = "storage_events"
event_batch_size = 100
event_flush_interval_seconds = 5.0

# Distributed Tracing
tracing_enabled = false
tracing_backend = "opentelemetry"  # opentelemetry, none
tracing_endpoint = "http://localhost:4317"
tracing_service_name = "llmcore-storage"
tracing_sample_rate = 1.0        # 1.0 = 100%
"""


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "DEFAULT_OBSERVABILITY_CONFIG",
    "OBSERVABILITY_TOML_EXAMPLE",
    "ObservabilityConfig",
]
