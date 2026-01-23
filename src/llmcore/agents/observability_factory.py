# src/llmcore/agents/observability_factory.py
"""
Observability Factory for LLMCore Agents.

Provides factory functions to create observability components (EventLogger,
MetricsCollector) from configuration. This module bridges the gap between
the config system and the observability module.

Usage:
    >>> from llmcore.agents.observability_factory import (
    ...     create_observability_from_config,
    ...     ObservabilityComponents,
    ... )
    >>> from llmcore.config import LLMCoreConfig
    >>>
    >>> config = LLMCoreConfig.load()
    >>> obs = create_observability_from_config(config, session_id="sess-123")
    >>>
    >>> if obs.enabled:
    ...     await obs.logger.log_lifecycle_start(goal="Analyze data")
    ...     # ... agent execution
    ...     await obs.logger.log_lifecycle_end(status="success")
    ...     await obs.close()

References:
    - Master Plan: Section 29.8 (Phase 7 Observability)
    - Configuration: default_config.toml [agents.observability]
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config.models import LLMCoreConfig

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class ObservabilityComponents:
    """
    Container for observability components created from configuration.

    This dataclass holds all observability-related components and provides
    a convenient way to check if observability is enabled and access
    individual components.

    Attributes:
        enabled: Whether observability is enabled
        logger: EventLogger instance (None if disabled)
        metrics: MetricsCollector instance (None if disabled)
        config: Observability configuration used
    """

    enabled: bool = False
    logger: Optional[Any] = None  # EventLogger
    metrics: Optional[Any] = None  # MetricsCollector
    config: Dict[str, Any] = field(default_factory=dict)

    async def close(self) -> None:
        """Close all observability components."""
        if self.logger is not None:
            try:
                await self.logger.close()
            except Exception as e:
                logger.warning(f"Error closing event logger: {e}")

    def __bool__(self) -> bool:
        """Return True if observability is enabled."""
        return self.enabled


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_observability_from_config(
    config: Optional["LLMCoreConfig"],
    session_id: str,
    *,
    execution_id: Optional[str] = None,
    default_tags: Optional[List[str]] = None,
    override_enabled: Optional[bool] = None,
) -> ObservabilityComponents:
    """
    Create observability components from LLMCore configuration.

    This factory reads the [agents.observability] configuration section
    and creates appropriately configured EventLogger and MetricsCollector
    instances.

    Args:
        config: LLMCore configuration (None = use defaults)
        session_id: Session identifier for events
        execution_id: Optional execution identifier
        default_tags: Default tags for all events
        override_enabled: Override the config enabled setting

    Returns:
        ObservabilityComponents with configured logger and metrics

    Example:
        >>> config = LLMCoreConfig.load()
        >>> obs = create_observability_from_config(config, "sess-123")
        >>> if obs.enabled:
        ...     async with obs.logger:
        ...         await obs.logger.log_lifecycle_start(goal="task")
    """
    # Extract observability config
    obs_config = _extract_observability_config(config)

    # Check if enabled
    enabled = obs_config.get("enabled", True)
    if override_enabled is not None:
        enabled = override_enabled

    if not enabled:
        logger.debug("Observability disabled by configuration")
        return ObservabilityComponents(
            enabled=False,
            config=obs_config,
        )

    # Import observability module
    try:
        from .observability import (
            EventLogger,
            MetricsCollector,
            JSONLFileSink,
            InMemorySink,
            FilteredSink,
            EventSeverity,
            EventCategory,
        )
    except ImportError as e:
        logger.warning(f"Observability module not available: {e}")
        return ObservabilityComponents(
            enabled=False,
            config=obs_config,
        )

    # Create sinks based on config
    sinks = []
    events_config = obs_config.get("events", {})
    sinks_config = obs_config.get("sinks", {})
    buffer_config = obs_config.get("buffer", {})

    # File sink
    if events_config.get("enabled", True) and sinks_config.get("file_enabled", True):
        log_path = events_config.get("log_path", "~/.llmcore/events.jsonl")
        log_path = os.path.expanduser(log_path)

        buffer_size = 1
        if buffer_config.get("enabled", True):
            buffer_size = buffer_config.get("size", 100)

        file_sink = JSONLFileSink(
            Path(log_path),
            buffer_size=buffer_size,
        )

        # Apply severity filter if configured
        min_severity = events_config.get("min_severity", "info")
        categories = events_config.get("categories", [])

        if min_severity != "debug" or categories:
            # Map severity string to enum
            severity_map = {
                "debug": EventSeverity.DEBUG,
                "info": EventSeverity.INFO,
                "warning": EventSeverity.WARNING,
                "error": EventSeverity.ERROR,
                "critical": EventSeverity.CRITICAL,
            }
            min_sev = severity_map.get(min_severity.lower(), EventSeverity.INFO)

            # Map category strings to enums
            category_filter = None
            if categories:
                cat_map = {
                    "lifecycle": EventCategory.LIFECYCLE,
                    "cognitive": EventCategory.COGNITIVE,
                    "activity": EventCategory.ACTIVITY,
                    "hitl": EventCategory.HITL,
                    "error": EventCategory.ERROR,
                    "metric": EventCategory.METRIC,
                    "memory": EventCategory.MEMORY,
                    "sandbox": EventCategory.SANDBOX,
                    "rag": EventCategory.RAG,
                }
                category_filter = [
                    cat_map[c.lower()]
                    for c in categories
                    if c.lower() in cat_map
                ]

            file_sink = FilteredSink(
                sink=file_sink,
                min_severity=min_sev,
                categories=category_filter or None,
            )

        sinks.append(file_sink)

    # Memory sink (for debugging)
    if sinks_config.get("memory_enabled", False):
        max_events = sinks_config.get("memory_max_events", 1000)
        sinks.append(InMemorySink(max_events=max_events))

    # Create EventLogger
    event_logger = EventLogger(
        session_id=session_id,
        execution_id=execution_id,
        sinks=sinks,
        default_tags=default_tags,
    )

    # Create MetricsCollector if enabled
    metrics_collector = None
    metrics_config = obs_config.get("metrics", {})
    if metrics_config.get("enabled", True):
        metrics_collector = MetricsCollector()

    logger.info(
        f"Observability initialized: session={session_id}, "
        f"sinks={len(sinks)}, metrics={metrics_collector is not None}"
    )

    return ObservabilityComponents(
        enabled=True,
        logger=event_logger,
        metrics=metrics_collector,
        config=obs_config,
    )


def _extract_observability_config(config: Optional["LLMCoreConfig"]) -> Dict[str, Any]:
    """
    Extract observability configuration from LLMCore config.

    Args:
        config: LLMCore configuration

    Returns:
        Observability configuration dictionary
    """
    if config is None:
        return {"enabled": True}

    # Try to get agents.observability section
    try:
        if hasattr(config, "agents") and config.agents:
            if hasattr(config.agents, "observability"):
                obs = config.agents.observability
                if isinstance(obs, dict):
                    return obs
                elif hasattr(obs, "model_dump"):
                    return obs.model_dump()
                elif hasattr(obs, "__dict__"):
                    return dict(obs.__dict__)

        # Try raw config access
        if hasattr(config, "raw_config"):
            raw = config.raw_config
            if isinstance(raw, dict):
                return raw.get("agents", {}).get("observability", {"enabled": True})

    except Exception as e:
        logger.debug(f"Could not extract observability config: {e}")

    return {"enabled": True}


def create_event_logger_simple(
    session_id: str,
    *,
    log_path: Optional[str] = None,
    execution_id: Optional[str] = None,
    enabled: bool = True,
) -> Optional[Any]:
    """
    Create a simple EventLogger without full configuration.

    This is a convenience function for quick setup without
    a full LLMCoreConfig object.

    Args:
        session_id: Session identifier
        log_path: Path to log file (default: ~/.llmcore/events.jsonl)
        execution_id: Optional execution identifier
        enabled: Whether to enable logging

    Returns:
        EventLogger or None if disabled

    Example:
        >>> logger = create_event_logger_simple("sess-123")
        >>> if logger:
        ...     await logger.log_lifecycle_start(goal="task")
    """
    if not enabled:
        return None

    try:
        from .observability import EventLogger, JSONLFileSink

        sinks = []
        if log_path is None:
            log_path = os.path.expanduser("~/.llmcore/events.jsonl")

        sinks.append(JSONLFileSink(Path(log_path)))

        return EventLogger(
            session_id=session_id,
            execution_id=execution_id,
            sinks=sinks,
        )

    except ImportError as e:
        logger.warning(f"Observability module not available: {e}")
        return None


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    "ObservabilityComponents",
    "create_observability_from_config",
    "create_event_logger_simple",
]
