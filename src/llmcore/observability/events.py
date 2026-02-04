# src/llmcore/observability/events.py
"""
Structured Event Logging System for LLMCore.

This module provides a comprehensive event logging system for all LLMCore
operations, enabling:

- Structured event capture with standardized categories
- JSONL file logging with rotation
- In-memory buffering for performance
- Event filtering by category and severity
- Execution replay support

Architecture:
    - EventCategory: Enum of event types (lifecycle, cognitive, activity, etc.)
    - Event: Base event dataclass with standardized fields
    - EventBuffer: In-memory circular buffer for efficient logging
    - ObservabilityLogger: Main logger class with file and callback support

Thread Safety:
    All operations are thread-safe. The EventBuffer uses locks for
    concurrent access, and file operations are serialized.

Usage:
    >>> from llmcore.observability.events import (
    ...     ObservabilityLogger,
    ...     EventCategory,
    ...     ObservabilityConfig,
    ... )
    >>>
    >>> config = ObservabilityConfig(
    ...     events_enabled=True,
    ...     log_path="~/.llmcore/events.jsonl",
    ...     min_severity="info",
    ... )
    >>> logger = ObservabilityLogger(config)
    >>>
    >>> # Log a cognitive phase event
    >>> logger.log_event(
    ...     category=EventCategory.COGNITIVE,
    ...     event_type="phase_completed",
    ...     data={"phase": "THINK", "duration_ms": 1500},
    ... )
    >>>
    >>> # Log with context
    >>> logger.log_event(
    ...     category=EventCategory.ACTIVITY,
    ...     event_type="tool_executed",
    ...     data={"tool": "execute_python", "success": True},
    ...     execution_id="exec_abc123",
    ...     iteration=3,
    ... )

References:
    - UNIFIED_IMPLEMENTATION_PLAN.md Phase 9
    - llmcore_spec_v2.md Section 13 (Observability System)
"""

from __future__ import annotations

import gzip
import json
import logging
import shutil
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, UTC
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Deque,
    Dict,
    List,
    Optional,
    Set,
    Union,
)
from collections.abc import Callable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Default buffer size for in-memory events
DEFAULT_BUFFER_SIZE = 100

# Default flush interval in seconds
DEFAULT_FLUSH_INTERVAL = 5.0

# Default maximum file size before rotation (bytes)
DEFAULT_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# Default maximum rotated files to keep
DEFAULT_MAX_FILES = 10

# Severity levels (ordered by priority)
SEVERITY_LEVELS = ["debug", "info", "warning", "error", "critical"]


# =============================================================================
# ENUMS
# =============================================================================


class EventCategory(str, Enum):
    """
    Categories for structured events.

    Categories help organize events for filtering, analysis, and debugging.
    Each category represents a distinct aspect of system operation.
    """

    LIFECYCLE = "lifecycle"  # Agent/session start/stop
    COGNITIVE = "cognitive"  # Cognitive phase execution
    ACTIVITY = "activity"  # Tool/activity execution
    HITL = "hitl"  # Human-in-the-loop approval events
    ERROR = "error"  # Errors and exceptions
    METRIC = "metric"  # Performance metrics snapshots
    MEMORY = "memory"  # Memory/context operations
    SANDBOX = "sandbox"  # Container/sandbox lifecycle
    RAG = "rag"  # Retrieval operations
    LLM = "llm"  # LLM API calls
    STORAGE = "storage"  # Storage backend operations
    CONFIG = "config"  # Configuration changes
    CUSTOM = "custom"  # User-defined events


class Severity(str, Enum):
    """Event severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @classmethod
    def from_string(cls, value: str) -> Severity:
        """Parse severity from string (case-insensitive)."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.INFO

    def __ge__(self, other: Severity) -> bool:
        """Compare severity levels."""
        levels = list(Severity)
        return levels.index(self) >= levels.index(other)

    def __gt__(self, other: Severity) -> bool:
        """Compare severity levels."""
        levels = list(Severity)
        return levels.index(self) > levels.index(other)

    def __le__(self, other: Severity) -> bool:
        """Compare severity levels."""
        return not self.__gt__(other)

    def __lt__(self, other: Severity) -> bool:
        """Compare severity levels."""
        return not self.__ge__(other)


class RotationStrategy(str, Enum):
    """Log rotation strategies."""

    NONE = "none"  # No rotation
    DAILY = "daily"  # Rotate daily
    SIZE = "size"  # Rotate by file size
    BOTH = "both"  # Rotate by both size and daily


# =============================================================================
# DATA MODELS
# =============================================================================


class Event(BaseModel):
    """
    Structured event record.

    Every event has a standardized format with timestamp, category, type,
    severity, and optional context fields. Additional data is stored in
    the 'data' dictionary.
    """

    # Core fields
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    category: EventCategory
    event_type: str = Field(..., description="Specific event type within category")
    severity: Severity = Field(default=Severity.INFO)

    # Context fields (optional)
    execution_id: str | None = Field(
        default=None, description="ID of the execution/session this event belongs to"
    )
    iteration: int | None = Field(
        default=None, description="Iteration number for iterative processes"
    )
    session_id: str | None = Field(default=None, description="User session ID")
    user_id: str | None = Field(default=None, description="User identifier")

    # Event data
    data: dict[str, Any] = Field(default_factory=dict, description="Event-specific data payload")

    # Metadata
    source: str | None = Field(
        default=None, description="Source module/component that emitted the event"
    )
    tags: list[str] = Field(default_factory=list, description="Tags for filtering and organization")

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

    def to_jsonl(self) -> str:
        """Convert to JSONL-compatible string."""
        return self.model_dump_json()

    @classmethod
    def from_jsonl(cls, line: str) -> Event:
        """Parse from JSONL line."""
        return cls.model_validate_json(line)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump(mode="json")


class EventRotationConfig(BaseModel):
    """Configuration for log file rotation."""

    strategy: RotationStrategy = Field(
        default=RotationStrategy.SIZE, description="Rotation strategy"
    )
    max_size_mb: int = Field(
        default=100, ge=1, description="Maximum file size in MB before rotation"
    )
    max_files: int = Field(
        default=10, ge=0, description="Maximum rotated files to keep (0 = unlimited)"
    )
    compress: bool = Field(default=True, description="Compress rotated files with gzip")


class EventBufferConfig(BaseModel):
    """Configuration for event buffering."""

    enabled: bool = Field(default=True, description="Enable write buffering")
    size: int = Field(default=100, ge=1, description="Maximum events to buffer")
    flush_interval_seconds: float = Field(
        default=5.0, gt=0, description="Flush interval in seconds"
    )
    flush_on_shutdown: bool = Field(default=True, description="Flush buffer on shutdown")


class ObservabilityConfig(BaseModel):
    """Configuration for the observability system."""

    enabled: bool = Field(default=True, description="Enable observability")

    # Events configuration
    events_enabled: bool = Field(default=True, description="Enable event logging")
    log_path: str = Field(default="~/.llmcore/events.jsonl", description="Path to event log file")
    min_severity: str = Field(default="info", description="Minimum severity to log")
    categories: list[str] = Field(
        default_factory=list, description="Categories to log (empty = all)"
    )

    # Rotation configuration
    rotation: EventRotationConfig = Field(default_factory=EventRotationConfig)

    # Buffer configuration
    buffer: EventBufferConfig = Field(default_factory=EventBufferConfig)

    # Performance options
    async_logging: bool = Field(default=True, description="Use async logging (non-blocking)")
    sampling_rate: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Event sampling rate (1.0 = all events)"
    )
    max_event_data_bytes: int = Field(
        default=10000, ge=100, description="Maximum bytes for event data field"
    )


# =============================================================================
# EVENT BUFFER
# =============================================================================


class EventBuffer:
    """
    Thread-safe in-memory event buffer.

    Buffers events before writing to disk for improved performance.
    Automatically flushes when buffer is full or on timer.
    """

    def __init__(
        self,
        max_size: int = DEFAULT_BUFFER_SIZE,
        flush_callback: Callable[[list[Event]], None] | None = None,
    ):
        """
        Initialize the event buffer.

        Args:
            max_size: Maximum events to buffer before auto-flush
            flush_callback: Called when buffer is flushed with events
        """
        self._buffer: deque[Event] = deque(maxlen=max_size)
        self._max_size = max_size
        self._flush_callback = flush_callback
        self._lock = threading.RLock()
        self._total_events = 0
        self._flush_count = 0

    def add(self, event: Event) -> bool:
        """
        Add an event to the buffer.

        Args:
            event: Event to add

        Returns:
            True if event was added, False if buffer is full and flush failed
        """
        with self._lock:
            should_flush = len(self._buffer) >= self._max_size - 1

            if should_flush and self._flush_callback:
                self._flush()

            self._buffer.append(event)
            self._total_events += 1
            return True

    def flush(self) -> list[Event]:
        """
        Flush all events from the buffer.

        Returns:
            List of flushed events
        """
        with self._lock:
            return self._flush()

    def _flush(self) -> list[Event]:
        """Internal flush (must hold lock)."""
        events = list(self._buffer)
        self._buffer.clear()
        self._flush_count += 1

        if events and self._flush_callback:
            try:
                self._flush_callback(events)
            except Exception as e:
                logger.error(f"Flush callback failed: {e}")

        return events

    def __len__(self) -> int:
        """Return number of buffered events."""
        with self._lock:
            return len(self._buffer)

    @property
    def stats(self) -> dict[str, Any]:
        """Get buffer statistics."""
        with self._lock:
            return {
                "buffer_size": len(self._buffer),
                "max_size": self._max_size,
                "total_events": self._total_events,
                "flush_count": self._flush_count,
            }

    def clear(self) -> None:
        """Clear the buffer without flushing."""
        with self._lock:
            self._buffer.clear()


# =============================================================================
# FILE WRITER
# =============================================================================


class EventFileWriter:
    """
    Writes events to JSONL files with rotation support.

    Handles file rotation by size and/or daily, with optional compression
    of rotated files.
    """

    def __init__(
        self,
        log_path: str | Path,
        rotation_config: EventRotationConfig | None = None,
    ):
        """
        Initialize the file writer.

        Args:
            log_path: Path to the main log file
            rotation_config: Rotation configuration
        """
        self._log_path = Path(log_path).expanduser().resolve()
        self._rotation = rotation_config or EventRotationConfig()
        self._lock = threading.Lock()
        self._current_date = datetime.now(tz=UTC).date()
        self._bytes_written = 0

        # Ensure directory exists
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        # Get current file size if exists
        if self._log_path.exists():
            self._bytes_written = self._log_path.stat().st_size

    def write(self, events: list[Event]) -> int:
        """
        Write events to the log file.

        Args:
            events: Events to write

        Returns:
            Number of events written
        """
        if not events:
            return 0

        with self._lock:
            # Check if rotation needed
            self._maybe_rotate()

            # Write events
            written = 0
            try:
                with open(self._log_path, "a", encoding="utf-8") as f:
                    for event in events:
                        line = event.to_jsonl() + "\n"
                        f.write(line)
                        self._bytes_written += len(line.encode("utf-8"))
                        written += 1
            except Exception as e:
                logger.error(f"Failed to write events: {e}")

            return written

    def _maybe_rotate(self) -> None:
        """Check and perform rotation if needed."""
        if self._rotation.strategy == RotationStrategy.NONE:
            return

        should_rotate = False

        # Check size-based rotation
        if self._rotation.strategy in (RotationStrategy.SIZE, RotationStrategy.BOTH):
            max_bytes = self._rotation.max_size_mb * 1024 * 1024
            if self._bytes_written >= max_bytes:
                should_rotate = True

        # Check daily rotation
        if self._rotation.strategy in (RotationStrategy.DAILY, RotationStrategy.BOTH):
            current_date = datetime.now(tz=UTC).date()
            if current_date != self._current_date:
                should_rotate = True
                self._current_date = current_date

        if should_rotate:
            self._rotate()

    def _rotate(self) -> None:
        """Perform log file rotation."""
        if not self._log_path.exists():
            return

        try:
            # Generate rotated filename with timestamp
            timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
            rotated_name = f"{self._log_path.stem}_{timestamp}{self._log_path.suffix}"
            rotated_path = self._log_path.parent / rotated_name

            # Move current file to rotated name
            shutil.move(str(self._log_path), str(rotated_path))

            # Compress if configured
            if self._rotation.compress:
                self._compress_file(rotated_path)

            # Cleanup old files
            self._cleanup_old_files()

            # Reset counter
            self._bytes_written = 0

            logger.debug(f"Rotated log file to {rotated_name}")

        except Exception as e:
            logger.error(f"Failed to rotate log file: {e}")

    def _compress_file(self, file_path: Path) -> None:
        """Compress a file with gzip."""
        try:
            compressed_path = file_path.with_suffix(file_path.suffix + ".gz")
            with open(file_path, "rb") as f_in:
                with gzip.open(compressed_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            file_path.unlink()  # Remove original
        except Exception as e:
            logger.error(f"Failed to compress {file_path}: {e}")

    def _cleanup_old_files(self) -> None:
        """Remove old rotated files if max_files exceeded."""
        if self._rotation.max_files == 0:
            return  # Unlimited

        try:
            # Find all rotated files
            pattern = f"{self._log_path.stem}_*"
            rotated_files = sorted(
                self._log_path.parent.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True
            )

            # Remove excess files
            for old_file in rotated_files[self._rotation.max_files :]:
                old_file.unlink()
                logger.debug(f"Removed old log file: {old_file.name}")

        except Exception as e:
            logger.error(f"Failed to cleanup old files: {e}")

    @property
    def current_size_bytes(self) -> int:
        """Get current log file size in bytes."""
        return self._bytes_written

    @property
    def log_path(self) -> Path:
        """Get the log file path."""
        return self._log_path


# =============================================================================
# OBSERVABILITY LOGGER
# =============================================================================


class ObservabilityLogger:
    """
    Main observability logger for structured event logging.

    Provides:
    - Structured event logging with categories and severity
    - File output with rotation
    - In-memory buffering
    - Callback support for integrations
    - Event filtering

    Thread-safe for concurrent logging from multiple threads.
    """

    def __init__(self, config: ObservabilityConfig | None = None):
        """
        Initialize the observability logger.

        Args:
            config: Configuration options
        """
        self._config = config or ObservabilityConfig()
        self._lock = threading.RLock()
        self._callbacks: list[Callable[[Event], None]] = []
        self._enabled = self._config.enabled and self._config.events_enabled

        # Initialize min severity
        self._min_severity = Severity.from_string(self._config.min_severity)

        # Initialize categories filter
        self._categories: set[EventCategory] | None = None
        if self._config.categories:
            self._categories = {
                EventCategory(c)
                for c in self._config.categories
                if c in [e.value for e in EventCategory]
            }

        # Initialize file writer
        self._writer: EventFileWriter | None = None
        if self._enabled:
            self._writer = EventFileWriter(
                log_path=self._config.log_path,
                rotation_config=self._config.rotation,
            )

        # Initialize buffer
        self._buffer: EventBuffer | None = None
        if self._enabled and self._config.buffer.enabled:
            self._buffer = EventBuffer(
                max_size=self._config.buffer.size,
                flush_callback=self._write_events,
            )

        # Stats
        self._total_events = 0
        self._filtered_events = 0
        self._errors = 0

    def log_event(
        self,
        category: EventCategory | str,
        event_type: str,
        data: dict[str, Any] | None = None,
        severity: Severity | str = Severity.INFO,
        execution_id: str | None = None,
        iteration: int | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        source: str | None = None,
        tags: list[str] | None = None,
    ) -> Event | None:
        """
        Log a structured event.

        Args:
            category: Event category
            event_type: Specific event type within category
            data: Event-specific data payload
            severity: Event severity level
            execution_id: ID of current execution/session
            iteration: Iteration number if applicable
            session_id: User session ID
            user_id: User identifier
            source: Source module/component
            tags: Tags for filtering

        Returns:
            The logged Event, or None if filtered/disabled
        """
        if not self._enabled:
            return None

        # Parse enums
        if isinstance(category, str):
            try:
                category = EventCategory(category)
            except ValueError:
                category = EventCategory.CUSTOM

        if isinstance(severity, str):
            severity = Severity.from_string(severity)

        # Check filters
        if not self._should_log(category, severity):
            self._filtered_events += 1
            return None

        # Apply sampling
        if self._config.sampling_rate < 1.0:
            import random

            if random.random() > self._config.sampling_rate:
                return None

        # Truncate data if too large
        if data and len(json.dumps(data)) > self._config.max_event_data_bytes:
            data = {"_truncated": True, "_original_size": len(json.dumps(data))}

        # Create event
        event = Event(
            category=category,
            event_type=event_type,
            severity=severity,
            execution_id=execution_id,
            iteration=iteration,
            session_id=session_id,
            user_id=user_id,
            data=data or {},
            source=source,
            tags=tags or [],
        )

        # Process event
        try:
            self._process_event(event)
            self._total_events += 1
        except Exception as e:
            self._errors += 1
            logger.error(f"Failed to log event: {e}")
            return None

        return event

    def _should_log(self, category: EventCategory, severity: Severity) -> bool:
        """Check if event should be logged based on filters."""
        # Check severity
        if severity < self._min_severity:
            return False

        # Check category
        if self._categories and category not in self._categories:
            return False

        return True

    def _process_event(self, event: Event) -> None:
        """Process an event (buffer or write directly)."""
        # Call callbacks
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback failed: {e}")

        # Buffer or write directly
        if self._buffer:
            self._buffer.add(event)
        elif self._writer:
            self._writer.write([event])

    def _write_events(self, events: list[Event]) -> None:
        """Write events to file (callback for buffer)."""
        if self._writer:
            self._writer.write(events)

    def add_callback(self, callback: Callable[[Event], None]) -> None:
        """
        Add a callback to be called for each event.

        Args:
            callback: Function that takes an Event
        """
        with self._lock:
            self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[Event], None]) -> None:
        """Remove a callback."""
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def flush(self) -> int:
        """
        Flush any buffered events.

        Returns:
            Number of events flushed
        """
        if not self._buffer:
            return 0

        events = self._buffer.flush()
        if events and self._writer:
            return self._writer.write(events)
        return len(events)

    def enable(self) -> None:
        """Enable logging."""
        self._enabled = True

    def disable(self) -> None:
        """Disable logging."""
        self._enabled = False

    def set_min_severity(self, severity: Severity | str) -> None:
        """Update minimum severity level."""
        if isinstance(severity, str):
            severity = Severity.from_string(severity)
        self._min_severity = severity

    def set_categories(self, categories: list[EventCategory] | None) -> None:
        """Update categories filter."""
        if categories:
            self._categories = set(categories)
        else:
            self._categories = None

    @property
    def stats(self) -> dict[str, Any]:
        """Get logger statistics."""
        buffer_stats = self._buffer.stats if self._buffer else {}
        return {
            "enabled": self._enabled,
            "total_events": self._total_events,
            "filtered_events": self._filtered_events,
            "errors": self._errors,
            "min_severity": self._min_severity.value,
            "categories": [c.value for c in self._categories] if self._categories else [],
            "buffer": buffer_stats,
            "log_path": str(self._writer.log_path) if self._writer else None,
            "current_log_size_bytes": self._writer.current_size_bytes if self._writer else 0,
        }

    def close(self) -> None:
        """Close the logger and flush any remaining events."""
        if self._config.buffer.flush_on_shutdown:
            self.flush()
        self._enabled = False


# =============================================================================
# EXECUTION TRACE
# =============================================================================


@dataclass
class ExecutionTrace:
    """
    Collection of events for a single execution.

    Used for replay and debugging of agent executions.
    """

    execution_id: str
    events: list[Event] = field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_event(self, event: Event) -> None:
        """Add an event to the trace."""
        self.events.append(event)

        # Update time bounds
        if self.start_time is None or event.timestamp < self.start_time:
            self.start_time = event.timestamp
        if self.end_time is None or event.timestamp > self.end_time:
            self.end_time = event.timestamp

    @property
    def duration_seconds(self) -> float | None:
        """Get execution duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def filter_by_category(self, category: EventCategory) -> list[Event]:
        """Get events of a specific category."""
        return [e for e in self.events if e.category == category]

    def filter_by_type(self, event_type: str) -> list[Event]:
        """Get events of a specific type."""
        return [e for e in self.events if e.event_type == event_type]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "execution_id": self.execution_id,
            "events": [e.to_dict() for e in self.events],
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }


# =============================================================================
# EXECUTION REPLAYER
# =============================================================================


class ExecutionReplayer:
    """
    Replay agent executions for debugging.

    Loads events from log files and provides step-by-step replay
    with callbacks for visualization.
    """

    def __init__(self, log_path: str | Path):
        """
        Initialize the replayer.

        Args:
            log_path: Path to the event log file
        """
        self._log_path = Path(log_path).expanduser().resolve()
        self._execution_cache: dict[str, ExecutionTrace] = {}
        self._cache_enabled = True
        self._max_cache_size = 50

    def load_execution(self, execution_id: str) -> ExecutionTrace | None:
        """
        Load all events for an execution.

        Args:
            execution_id: The execution ID to load

        Returns:
            ExecutionTrace with all events, or None if not found
        """
        # Check cache
        if self._cache_enabled and execution_id in self._execution_cache:
            return self._execution_cache[execution_id]

        # Scan log file
        trace = ExecutionTrace(execution_id=execution_id)

        try:
            # Handle compressed files
            if self._log_path.suffix == ".gz":
                with gzip.open(self._log_path, "rt", encoding="utf-8") as f:
                    self._scan_file(f, execution_id, trace)
            else:
                with open(self._log_path, encoding="utf-8") as f:
                    self._scan_file(f, execution_id, trace)
        except FileNotFoundError:
            logger.warning(f"Log file not found: {self._log_path}")
            return None
        except Exception as e:
            logger.error(f"Failed to load execution {execution_id}: {e}")
            return None

        if not trace.events:
            return None

        # Cache
        if self._cache_enabled:
            self._cache_execution(execution_id, trace)

        return trace

    def _scan_file(self, file, execution_id: str, trace: ExecutionTrace) -> None:
        """Scan file for events matching execution_id."""
        for line in file:
            line = line.strip()
            if not line:
                continue

            try:
                event = Event.from_jsonl(line)
                if event.execution_id == execution_id:
                    trace.add_event(event)
            except Exception:
                continue  # Skip malformed lines

    def _cache_execution(self, execution_id: str, trace: ExecutionTrace) -> None:
        """Add execution to cache, evicting oldest if needed."""
        if len(self._execution_cache) >= self._max_cache_size:
            # Remove oldest (first) entry
            oldest = next(iter(self._execution_cache))
            del self._execution_cache[oldest]
        self._execution_cache[execution_id] = trace

    def list_executions(
        self,
        limit: int = 100,
        category: EventCategory | None = None,
    ) -> list[str]:
        """
        List execution IDs from the log file.

        Args:
            limit: Maximum number of executions to return
            category: Optional filter by event category

        Returns:
            List of execution IDs (most recent first)
        """
        execution_ids = set()

        try:
            if self._log_path.suffix == ".gz":
                with gzip.open(self._log_path, "rt", encoding="utf-8") as f:
                    for line in f:
                        if len(execution_ids) >= limit:
                            break
                        self._extract_execution_id(line, execution_ids, category)
            else:
                with open(self._log_path, encoding="utf-8") as f:
                    for line in f:
                        if len(execution_ids) >= limit:
                            break
                        self._extract_execution_id(line, execution_ids, category)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.error(f"Failed to list executions: {e}")

        return list(execution_ids)

    def _extract_execution_id(
        self,
        line: str,
        execution_ids: set[str],
        category: EventCategory | None,
    ) -> None:
        """Extract execution ID from a log line."""
        line = line.strip()
        if not line:
            return

        try:
            event = Event.from_jsonl(line)
            if event.execution_id:
                if category is None or event.category == category:
                    execution_ids.add(event.execution_id)
        except Exception:
            pass

    def replay_step_by_step(
        self,
        trace: ExecutionTrace,
        on_phase_start: Callable[[Event], None] | None = None,
        on_phase_end: Callable[[Event], None] | None = None,
        on_tool_call: Callable[[Event], None] | None = None,
        on_error: Callable[[Event], None] | None = None,
        on_event: Callable[[Event], None] | None = None,
    ) -> None:
        """
        Replay execution with callbacks.

        Args:
            trace: ExecutionTrace to replay
            on_phase_start: Called when a cognitive phase starts
            on_phase_end: Called when a cognitive phase ends
            on_tool_call: Called for tool/activity execution
            on_error: Called for error events
            on_event: Called for all events
        """
        for event in sorted(trace.events, key=lambda e: e.timestamp):
            # Call general callback
            if on_event:
                on_event(event)

            # Call specific callbacks
            if event.category == EventCategory.COGNITIVE:
                if "start" in event.event_type.lower() and on_phase_start:
                    on_phase_start(event)
                elif "end" in event.event_type.lower() and on_phase_end:
                    on_phase_end(event)
            elif event.category == EventCategory.ACTIVITY and on_tool_call:
                on_tool_call(event)
            elif event.category == EventCategory.ERROR and on_error:
                on_error(event)

    def clear_cache(self) -> None:
        """Clear the execution cache."""
        self._execution_cache.clear()

    @property
    def cache_size(self) -> int:
        """Get current cache size."""
        return len(self._execution_cache)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_observability_logger(
    log_path: str | None = None,
    min_severity: str = "info",
    categories: list[str] | None = None,
    rotation_strategy: str = "size",
    max_size_mb: int = 100,
    buffer_enabled: bool = True,
    buffer_size: int = 100,
    **kwargs,
) -> ObservabilityLogger:
    """
    Create an ObservabilityLogger with common configuration.

    Args:
        log_path: Path to event log file
        min_severity: Minimum severity to log
        categories: Categories to log (empty = all)
        rotation_strategy: Log rotation strategy
        max_size_mb: Max file size for rotation
        buffer_enabled: Enable buffering
        buffer_size: Buffer size
        **kwargs: Additional ObservabilityConfig fields

    Returns:
        Configured ObservabilityLogger
    """
    config = ObservabilityConfig(
        log_path=log_path or "~/.llmcore/events.jsonl",
        min_severity=min_severity,
        categories=categories or [],
        rotation=EventRotationConfig(
            strategy=RotationStrategy(rotation_strategy),
            max_size_mb=max_size_mb,
        ),
        buffer=EventBufferConfig(
            enabled=buffer_enabled,
            size=buffer_size,
        ),
        **kwargs,
    )
    return ObservabilityLogger(config)


def load_events_from_file(
    log_path: str | Path,
    execution_id: str | None = None,
    category: EventCategory | None = None,
    min_severity: Severity | None = None,
    limit: int | None = None,
) -> list[Event]:
    """
    Load events from a log file with optional filtering.

    Args:
        log_path: Path to the event log file
        execution_id: Filter by execution ID
        category: Filter by category
        min_severity: Filter by minimum severity
        limit: Maximum events to return

    Returns:
        List of matching events
    """
    log_path = Path(log_path).expanduser().resolve()
    events = []

    try:
        if log_path.suffix == ".gz":
            opener = gzip.open(log_path, "rt", encoding="utf-8")
        else:
            opener = open(log_path, encoding="utf-8")

        with opener as f:
            for line in f:
                if limit and len(events) >= limit:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    event = Event.from_jsonl(line)

                    # Apply filters
                    if execution_id and event.execution_id != execution_id:
                        continue
                    if category and event.category != category:
                        continue
                    if min_severity and Severity(event.severity) < min_severity:
                        continue

                    events.append(event)
                except Exception:
                    continue

    except FileNotFoundError:
        logger.warning(f"Log file not found: {log_path}")
    except Exception as e:
        logger.error(f"Failed to load events: {e}")

    return events


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "EventCategory",
    "Severity",
    "RotationStrategy",
    # Data models
    "Event",
    "EventRotationConfig",
    "EventBufferConfig",
    "ObservabilityConfig",
    # Core classes
    "EventBuffer",
    "EventFileWriter",
    "ObservabilityLogger",
    # Replay
    "ExecutionTrace",
    "ExecutionReplayer",
    # Factories
    "create_observability_logger",
    "load_events_from_file",
]
