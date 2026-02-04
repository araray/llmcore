# src/llmcore/observability/config.py
"""
Observability Configuration Models.

This module provides Pydantic models that map to the [agents.observability]
section in the configuration file (default_config.toml / llmcore.toml).

Configuration Structure:
    [agents.observability]
    enabled = true

    [agents.observability.events]
    enabled = true
    log_path = "~/.llmcore/events.jsonl"
    min_severity = "info"
    categories = []

    [agents.observability.events.rotation]
    strategy = "both"
    max_size_mb = 100
    max_files = 10
    compress = true

    [agents.observability.buffer]
    enabled = true
    size = 100
    flush_interval_seconds = 5
    flush_on_shutdown = true

    [agents.observability.metrics]
    enabled = true
    collect = []
    track_cost = true
    track_tokens = true
    latency_percentiles = [50, 90, 95, 99]

    [agents.observability.replay]
    enabled = true
    cache_enabled = true
    cache_max_executions = 50

    [agents.observability.sinks]
    file_enabled = true
    memory_enabled = false
    memory_max_events = 1000
    callback_enabled = false

    [agents.observability.performance]
    async_logging = true
    sampling_rate = 1.0
    max_event_data_bytes = 10000
    overhead_warning_threshold_percent = 5

Usage:
    >>> from llmcore.observability.config import (
    ...     ObservabilityConfig,
    ...     load_observability_config,
    ... )
    >>>
    >>> # Load from config dict (typically from confy)
    >>> config = load_observability_config(config_dict)
    >>>
    >>> # Or use defaults
    >>> config = ObservabilityConfig()
    >>> print(config.metrics.track_cost)  # True
    >>> print(config.events.rotation.strategy)  # "size"

References:
    - llmcore_spec_v2.md Section 13.6
    - UNIFIED_IMPLEMENTATION_PLAN.md Phase 9
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class RotationStrategy(str, Enum):
    """Log file rotation strategy."""

    NONE = "none"  # No rotation
    DAILY = "daily"  # Rotate daily
    SIZE = "size"  # Rotate by size
    BOTH = "both"  # Rotate by size AND daily


class Severity(str, Enum):
    """Event severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# =============================================================================
# NESTED CONFIG MODELS
# =============================================================================


class EventRotationConfig(BaseModel):
    """
    Configuration for log file rotation.

    Maps to: [agents.observability.events.rotation]
    """

    strategy: RotationStrategy = Field(
        default=RotationStrategy.SIZE, description="Rotation strategy: none, daily, size, or both"
    )
    max_size_mb: int = Field(
        default=100, ge=1, description="Maximum file size in MB before rotation"
    )
    max_files: int = Field(
        default=10, ge=0, description="Maximum rotated files to keep (0 = unlimited)"
    )
    compress: bool = Field(default=True, description="Compress rotated files with gzip")

    @field_validator("strategy", mode="before")
    @classmethod
    def validate_strategy(cls, v: Any) -> RotationStrategy:
        if isinstance(v, str):
            return RotationStrategy(v.lower())
        return v


class EventsConfig(BaseModel):
    """
    Configuration for event logging.

    Maps to: [agents.observability.events]
    """

    enabled: bool = Field(default=True, description="Enable event logging")
    log_path: str = Field(default="~/.llmcore/events.jsonl", description="Path to event log file")
    min_severity: Severity = Field(default=Severity.INFO, description="Minimum severity to log")
    categories: list[str] = Field(
        default_factory=list, description="Categories to log (empty = all)"
    )
    rotation: EventRotationConfig = Field(
        default_factory=EventRotationConfig, description="Log rotation configuration"
    )

    @field_validator("min_severity", mode="before")
    @classmethod
    def validate_severity(cls, v: Any) -> Severity:
        if isinstance(v, str):
            return Severity(v.lower())
        return v

    @property
    def log_path_expanded(self) -> Path:
        """Get expanded log path."""
        return Path(self.log_path).expanduser()


class BufferConfig(BaseModel):
    """
    Configuration for event buffering.

    Maps to: [agents.observability.buffer]
    """

    enabled: bool = Field(default=True, description="Enable write buffering")
    size: int = Field(default=100, ge=1, description="Maximum events to buffer before flush")
    flush_interval_seconds: float = Field(
        default=5.0, gt=0, description="Flush interval in seconds"
    )
    flush_on_shutdown: bool = Field(default=True, description="Flush buffer on shutdown")


class MetricsConfig(BaseModel):
    """
    Configuration for metrics collection.

    Maps to: [agents.observability.metrics]
    """

    enabled: bool = Field(default=True, description="Enable metrics collection")
    collect: list[str] = Field(
        default_factory=list, description="Specific metrics to collect (empty = all)"
    )
    track_cost: bool = Field(default=True, description="Track API costs")
    track_tokens: bool = Field(default=True, description="Track token usage")
    latency_percentiles: list[int] = Field(
        default_factory=lambda: [50, 90, 95, 99], description="Latency percentiles to track"
    )

    @field_validator("latency_percentiles")
    @classmethod
    def validate_percentiles(cls, v: list[int]) -> list[int]:
        for p in v:
            if not 0 <= p <= 100:
                raise ValueError(f"Percentile {p} must be between 0 and 100")
        return sorted(set(v))


class ReplayConfig(BaseModel):
    """
    Configuration for execution replay.

    Maps to: [agents.observability.replay]
    """

    enabled: bool = Field(default=True, description="Enable execution replay")
    cache_enabled: bool = Field(default=True, description="Cache loaded executions")
    cache_max_executions: int = Field(default=50, ge=1, description="Maximum cached executions")


class SinksConfig(BaseModel):
    """
    Configuration for event sinks (output destinations).

    Maps to: [agents.observability.sinks]
    """

    file_enabled: bool = Field(default=True, description="Enable file sink")
    memory_enabled: bool = Field(default=False, description="Enable in-memory sink")
    memory_max_events: int = Field(default=1000, ge=1, description="Maximum events in memory sink")
    callback_enabled: bool = Field(
        default=False, description="Enable callback sink for custom handlers"
    )


class PerformanceConfig(BaseModel):
    """
    Configuration for observability performance tuning.

    Maps to: [agents.observability.performance]
    """

    async_logging: bool = Field(default=True, description="Use async logging (non-blocking)")
    sampling_rate: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Event sampling rate (1.0 = all events)"
    )
    max_event_data_bytes: int = Field(
        default=10000, ge=100, description="Maximum bytes for event data field"
    )
    overhead_warning_threshold_percent: float = Field(
        default=5.0,
        ge=0.0,
        le=100.0,
        description="Warn if observability overhead exceeds this percentage",
    )


# =============================================================================
# MAIN CONFIG MODEL
# =============================================================================


class ObservabilityConfig(BaseModel):
    """
    Complete observability configuration.

    Maps to: [agents.observability]

    This is the main configuration model that aggregates all observability
    settings. It can be instantiated from a config dict (from confy) or
    with defaults.

    Example:
        >>> config = ObservabilityConfig()
        >>> config.metrics.track_cost
        True

        >>> config = ObservabilityConfig(
        ...     enabled=True,
        ...     metrics={"track_cost": False}
        ... )
        >>> config.metrics.track_cost
        False
    """

    enabled: bool = Field(default=True, description="Master switch for observability")

    events: EventsConfig = Field(
        default_factory=EventsConfig, description="Event logging configuration"
    )

    buffer: BufferConfig = Field(
        default_factory=BufferConfig, description="Event buffering configuration"
    )

    metrics: MetricsConfig = Field(
        default_factory=MetricsConfig, description="Metrics collection configuration"
    )

    replay: ReplayConfig = Field(
        default_factory=ReplayConfig, description="Execution replay configuration"
    )

    sinks: SinksConfig = Field(default_factory=SinksConfig, description="Event sinks configuration")

    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig, description="Performance tuning configuration"
    )

    def is_event_logging_enabled(self) -> bool:
        """Check if event logging is enabled."""
        return self.enabled and self.events.enabled

    def is_metrics_enabled(self) -> bool:
        """Check if metrics collection is enabled."""
        return self.enabled and self.metrics.enabled

    def is_replay_enabled(self) -> bool:
        """Check if replay is enabled."""
        return self.enabled and self.replay.enabled

    def should_track_cost(self) -> bool:
        """Check if cost tracking is enabled."""
        return self.is_metrics_enabled() and self.metrics.track_cost

    def should_track_tokens(self) -> bool:
        """Check if token tracking is enabled."""
        return self.is_metrics_enabled() and self.metrics.track_tokens

    def get_latency_percentiles(self) -> list[int]:
        """Get latency percentiles to track."""
        if not self.is_metrics_enabled():
            return []
        return self.metrics.latency_percentiles


# =============================================================================
# LOADING FUNCTIONS
# =============================================================================


def load_observability_config(
    config_dict: dict[str, Any] | None = None,
    section_path: str = "agents.observability",
) -> ObservabilityConfig:
    """
    Load observability configuration from a config dictionary.

    Args:
        config_dict: Configuration dictionary (typically from confy).
                    If None, returns default configuration.
        section_path: Dot-separated path to observability section.

    Returns:
        ObservabilityConfig instance.

    Example:
        >>> from confy import Config
        >>> cfg = Config(file_path="llmcore.toml")
        >>> obs_config = load_observability_config(cfg.as_dict())
    """
    if config_dict is None:
        return ObservabilityConfig()

    # Navigate to the observability section
    section = config_dict
    for part in section_path.split("."):
        if not isinstance(section, dict):
            logger.warning(f"Config path '{section_path}' not found, using defaults")
            return ObservabilityConfig()
        section = section.get(part, {})

    if not isinstance(section, dict):
        logger.warning(f"Config section '{section_path}' is not a dict, using defaults")
        return ObservabilityConfig()

    try:
        return ObservabilityConfig.model_validate(section)
    except Exception as e:
        logger.warning(f"Failed to parse observability config: {e}. Using defaults.")
        return ObservabilityConfig()


def get_default_config() -> ObservabilityConfig:
    """
    Get the default observability configuration.

    Returns:
        ObservabilityConfig with all defaults.
    """
    return ObservabilityConfig()


# =============================================================================
# COMPATIBILITY LAYER
# =============================================================================


def to_legacy_config(config: ObservabilityConfig) -> dict[str, Any]:
    """
    Convert ObservabilityConfig to legacy flat dict format.

    For backward compatibility with code expecting the old config format.

    Args:
        config: ObservabilityConfig instance.

    Returns:
        Flat dictionary with legacy key names.
    """
    return {
        "enabled": config.enabled,
        "events_enabled": config.events.enabled,
        "log_path": config.events.log_path,
        "min_severity": config.events.min_severity.value,
        "categories": config.events.categories,
        "rotation": config.events.rotation.model_dump(),
        "buffer": config.buffer.model_dump(),
        "async_logging": config.performance.async_logging,
        "sampling_rate": config.performance.sampling_rate,
        "max_event_data_bytes": config.performance.max_event_data_bytes,
        # Metrics
        "metrics_enabled": config.metrics.enabled,
        "track_cost": config.metrics.track_cost,
        "track_tokens": config.metrics.track_tokens,
        "latency_percentiles": config.metrics.latency_percentiles,
        # Replay
        "replay_enabled": config.replay.enabled,
        "cache_enabled": config.replay.cache_enabled,
        "cache_max_executions": config.replay.cache_max_executions,
        # Sinks
        "file_enabled": config.sinks.file_enabled,
        "memory_enabled": config.sinks.memory_enabled,
        "memory_max_events": config.sinks.memory_max_events,
        "callback_enabled": config.sinks.callback_enabled,
    }
