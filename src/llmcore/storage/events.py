# src/llmcore/storage/events.py
"""
Event Logging for Phase 4 (PANOPTICON).

Provides persistent database logging of storage events for auditing,
debugging, and observability.

This module implements:
- Event capture for all storage operations
- Persistent storage in a dedicated events table
- Token usage tracking per session
- Error trace persistence
- Retention policy support
- Query API for event retrieval

Design Philosophy:
- Append-only event log for auditability
- Async-first for non-blocking operation
- Configurable retention for storage management
- Structured events for queryability

Event Types:
- session_create, session_update, session_delete
- message_add, message_delete
- vector_add, vector_search, vector_delete
- health_check, error, slow_query

Usage:
    logger = EventLogger(config, pool)
    await logger.initialize()

    # Log an event
    await logger.log_event_async(
        event_type="session_create",
        session_id="abc123",
        user_id="user_456",
        metadata={"source": "api"},
    )

    # Query events
    events = await logger.query_events(
        session_id="abc123",
        since=datetime.now() - timedelta(days=7)
    )

STORAGE SYSTEM V2 (Phase 4 - PANOPTICON):
- storage_events table with comprehensive schema
- Async and sync logging interfaces
- Configurable retention and cleanup
- Integration with instrumentation layer
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, UTC
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


class EventType(str, Enum):
    """Standard event types for storage operations."""

    # Session events
    SESSION_CREATE = "session_create"
    SESSION_UPDATE = "session_update"
    SESSION_DELETE = "session_delete"
    SESSION_LOAD = "session_load"

    # Message events
    MESSAGE_ADD = "message_add"
    MESSAGE_DELETE = "message_delete"

    # Vector events
    VECTOR_ADD = "vector_add"
    VECTOR_SEARCH = "vector_search"
    VECTOR_DELETE = "vector_delete"
    VECTOR_UPDATE = "vector_update"

    # System events
    HEALTH_CHECK = "health_check"
    SCHEMA_MIGRATION = "schema_migration"
    CLEANUP = "cleanup"

    # Error events
    ERROR = "error"
    SLOW_QUERY = "slow_query"
    CIRCUIT_OPEN = "circuit_open"
    CIRCUIT_CLOSE = "circuit_close"


@dataclass
class EventLoggerConfig:
    """
    Configuration for event logging.

    Attributes:
        enabled: Master switch for event logging.
        table_name: Name of the events table.
        retention_days: Number of days to retain events (0 = forever).
        batch_size: Number of events to batch before flushing.
        flush_interval_seconds: Maximum interval between flushes.
        async_logging: Use async queue for non-blocking logging.
        log_slow_queries: Log slow query events.
        log_errors: Log error events.
        include_metadata: Include metadata in events.
        max_metadata_size: Maximum size of metadata JSON (bytes).
    """

    enabled: bool = True
    table_name: str = "storage_events"
    retention_days: int = 30
    batch_size: int = 100
    flush_interval_seconds: float = 5.0
    async_logging: bool = True
    log_slow_queries: bool = True
    log_errors: bool = True
    include_metadata: bool = True
    max_metadata_size: int = 10000  # 10KB


DEFAULT_EVENT_LOGGER_CONFIG = EventLoggerConfig()


# =============================================================================
# EVENT DATA MODEL
# =============================================================================


@dataclass
class StorageEvent:
    """
    Represents a single storage event.

    Attributes:
        event_type: Type of event (from EventType enum).
        timestamp: When the event occurred.
        user_id: Associated user ID (if applicable).
        session_id: Associated session ID (if applicable).
        collection_name: Vector collection name (if applicable).
        operation_duration_ms: Duration of the operation (if applicable).
        metadata: Additional event context as JSON.
        error_message: Error message (if event is an error).
    """

    event_type: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    user_id: str | None = None
    session_id: str | None = None
    collection_name: str | None = None
    operation_duration_ms: float | None = None
    metadata: dict[str, Any] | None = None
    error_message: str | None = None
    id: int | None = None  # Assigned by database

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "collection_name": self.collection_name,
            "operation_duration_ms": self.operation_duration_ms,
            "metadata": self.metadata,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StorageEvent:
        """Create event from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        elif timestamp is None:
            timestamp = datetime.now(UTC)

        return cls(
            id=data.get("id"),
            event_type=data["event_type"],
            timestamp=timestamp,
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            collection_name=data.get("collection_name"),
            operation_duration_ms=data.get("operation_duration_ms"),
            metadata=data.get("metadata"),
            error_message=data.get("error_message"),
        )


# =============================================================================
# SQL SCHEMA
# =============================================================================


POSTGRES_EVENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS {table_name} (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    collection_name VARCHAR(255),
    operation_duration_ms REAL,
    metadata JSONB,
    error_message TEXT
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_{table_name}_timestamp
    ON {table_name} (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_{table_name}_user
    ON {table_name} (user_id) WHERE user_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_{table_name}_session
    ON {table_name} (session_id) WHERE session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_{table_name}_type
    ON {table_name} (event_type);
CREATE INDEX IF NOT EXISTS idx_{table_name}_type_timestamp
    ON {table_name} (event_type, timestamp DESC);
"""

SQLITE_EVENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS {table_name} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    user_id TEXT,
    session_id TEXT,
    collection_name TEXT,
    operation_duration_ms REAL,
    metadata TEXT,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_{table_name}_timestamp
    ON {table_name} (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_{table_name}_user
    ON {table_name} (user_id);
CREATE INDEX IF NOT EXISTS idx_{table_name}_session
    ON {table_name} (session_id);
CREATE INDEX IF NOT EXISTS idx_{table_name}_type
    ON {table_name} (event_type);
"""


# =============================================================================
# EVENT LOGGER
# =============================================================================


class EventLogger:
    """
    Persistent event logger for storage operations.

    Logs events to a database table for auditing and observability.
    Supports both sync and async logging with optional batching.

    Usage:
        logger = EventLogger(config, pool)
        await logger.initialize()

        await logger.log_event_async(
            event_type="session_create",
            session_id="abc123",
        )
    """

    def __init__(
        self,
        config: EventLoggerConfig | None = None,
        pool: Any | None = None,  # asyncpg pool or aiosqlite connection
        backend: str = "postgres",  # "postgres" or "sqlite"
    ):
        """
        Initialize event logger.

        Args:
            config: Event logger configuration.
            pool: Database connection pool or connection.
            backend: Backend type ("postgres" or "sqlite").
        """
        self.config = config or DEFAULT_EVENT_LOGGER_CONFIG
        self._pool = pool
        self._backend = backend
        self._initialized = False

        # Async event queue for non-blocking logging
        self._event_queue: asyncio.Queue[StorageEvent] = asyncio.Queue()
        self._flush_task: asyncio.Task | None = None
        self._shutdown = False

        # Sync fallback queue for when async is not available
        self._sync_queue: queue.Queue[StorageEvent] = queue.Queue()
        self._sync_thread: threading.Thread | None = None

        # Statistics
        self._events_logged = 0
        self._events_dropped = 0
        self._last_flush_time = time.time()

    @property
    def enabled(self) -> bool:
        """Check if event logging is enabled."""
        return self.config.enabled and self._pool is not None

    @property
    def events_logged(self) -> int:
        """Get total number of events logged."""
        return self._events_logged

    @property
    def events_dropped(self) -> int:
        """Get number of events dropped due to errors."""
        return self._events_dropped

    async def initialize(self) -> None:
        """
        Initialize the event logger.

        Creates the events table if it doesn't exist and starts
        the background flush task.
        """
        if self._initialized:
            return

        if not self.enabled:
            logger.debug("Event logging disabled; skipping initialization")
            return

        try:
            await self._create_schema()

            if self.config.async_logging:
                self._flush_task = asyncio.create_task(self._flush_loop())

            self._initialized = True
            logger.info(
                f"EventLogger initialized (table: {self.config.table_name}, "
                f"backend: {self._backend})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize EventLogger: {e}")
            raise

    async def shutdown(self) -> None:
        """
        Shutdown the event logger.

        Flushes any pending events and stops background tasks.
        """
        self._shutdown = True

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Flush remaining events
        await self._flush_events()

        logger.info(f"EventLogger shutdown complete ({self._events_logged} events logged)")

    async def _create_schema(self) -> None:
        """Create the events table schema."""
        if self._backend == "postgres":
            schema = POSTGRES_EVENTS_SCHEMA.format(table_name=self.config.table_name)
            async with self._pool.connection() as conn:
                await conn.execute(schema)
        else:
            schema = SQLITE_EVENTS_SCHEMA.format(table_name=self.config.table_name)
            await self._pool.executescript(schema)

    async def _flush_loop(self) -> None:
        """Background task to flush events periodically."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.config.flush_interval_seconds)
                await self._flush_events()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in event flush loop: {e}")

    async def _flush_events(self) -> None:
        """Flush pending events to database."""
        events: list[StorageEvent] = []

        # Drain the queue
        while not self._event_queue.empty() and len(events) < self.config.batch_size:
            try:
                event = self._event_queue.get_nowait()
                events.append(event)
            except asyncio.QueueEmpty:
                break

        if not events:
            return

        try:
            await self._write_events_batch(events)
            self._events_logged += len(events)
            self._last_flush_time = time.time()
        except Exception as e:
            logger.error(f"Failed to flush {len(events)} events: {e}")
            self._events_dropped += len(events)

    async def _write_events_batch(self, events: list[StorageEvent]) -> None:
        """Write a batch of events to the database."""
        if self._backend == "postgres":
            await self._write_events_postgres(events)
        else:
            await self._write_events_sqlite(events)

    async def _write_events_postgres(self, events: list[StorageEvent]) -> None:
        """Write events to PostgreSQL."""
        sql = f"""
            INSERT INTO {self.config.table_name}
            (event_type, timestamp, user_id, session_id, collection_name,
             operation_duration_ms, metadata, error_message)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """

        async with self._pool.connection() as conn:
            for event in events:
                metadata_json = (
                    json.dumps(event.metadata)[: self.config.max_metadata_size]
                    if event.metadata and self.config.include_metadata
                    else None
                )
                await conn.execute(
                    sql,
                    event.event_type,
                    event.timestamp,
                    event.user_id,
                    event.session_id,
                    event.collection_name,
                    event.operation_duration_ms,
                    metadata_json,
                    event.error_message,
                )

    async def _write_events_sqlite(self, events: list[StorageEvent]) -> None:
        """Write events to SQLite."""
        sql = f"""
            INSERT INTO {self.config.table_name}
            (event_type, timestamp, user_id, session_id, collection_name,
             operation_duration_ms, metadata, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        for event in events:
            metadata_json = (
                json.dumps(event.metadata)[: self.config.max_metadata_size]
                if event.metadata and self.config.include_metadata
                else None
            )
            await self._pool.execute(
                sql,
                (
                    event.event_type,
                    event.timestamp.isoformat(),
                    event.user_id,
                    event.session_id,
                    event.collection_name,
                    event.operation_duration_ms,
                    metadata_json,
                    event.error_message,
                ),
            )
        await self._pool.commit()

    # =========================================================================
    # PUBLIC LOGGING INTERFACE
    # =========================================================================

    async def log_event_async(
        self,
        event_type: str | EventType,
        user_id: str | None = None,
        session_id: str | None = None,
        collection_name: str | None = None,
        operation_duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> None:
        """
        Log an event asynchronously.

        Events are queued and flushed in batches for performance.

        Args:
            event_type: Type of event.
            user_id: Associated user ID.
            session_id: Associated session ID.
            collection_name: Vector collection name.
            operation_duration_ms: Operation duration.
            metadata: Additional context.
            error_message: Error message if applicable.
        """
        if not self.enabled:
            return

        event = StorageEvent(
            event_type=event_type.value if isinstance(event_type, EventType) else event_type,
            user_id=user_id,
            session_id=session_id,
            collection_name=collection_name,
            operation_duration_ms=operation_duration_ms,
            metadata=metadata,
            error_message=error_message,
        )

        try:
            self._event_queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("Event queue full; dropping event")
            self._events_dropped += 1

    def log_event(
        self,
        event_type: str | EventType,
        user_id: str | None = None,
        session_id: str | None = None,
        collection_name: str | None = None,
        operation_duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> None:
        """
        Log an event synchronously (queues for async flush).

        This is a sync wrapper for use in non-async contexts.
        The event is still flushed asynchronously.

        Args:
            event_type: Type of event.
            user_id: Associated user ID.
            session_id: Associated session ID.
            collection_name: Vector collection name.
            operation_duration_ms: Operation duration.
            metadata: Additional context.
            error_message: Error message if applicable.
        """
        if not self.enabled:
            return

        event = StorageEvent(
            event_type=event_type.value if isinstance(event_type, EventType) else event_type,
            user_id=user_id,
            session_id=session_id,
            collection_name=collection_name,
            operation_duration_ms=operation_duration_ms,
            metadata=metadata,
            error_message=error_message,
        )

        # Try to put in async queue from sync context
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.call_soon_threadsafe(lambda: self._event_queue.put_nowait(event))
            else:
                # No running loop; queue directly (may block)
                self._sync_queue.put_nowait(event)
        except Exception as e:
            logger.warning(f"Failed to queue event: {e}")
            self._events_dropped += 1

    # =========================================================================
    # QUERY INTERFACE
    # =========================================================================

    async def query_events(
        self,
        event_type: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[StorageEvent]:
        """
        Query stored events.

        Args:
            event_type: Filter by event type.
            user_id: Filter by user ID.
            session_id: Filter by session ID.
            since: Only events after this timestamp.
            until: Only events before this timestamp.
            limit: Maximum number of events to return.
            offset: Offset for pagination.

        Returns:
            List of matching StorageEvent objects.
        """
        if not self.enabled:
            return []

        if self._backend == "postgres":
            return await self._query_events_postgres(
                event_type, user_id, session_id, since, until, limit, offset
            )
        else:
            return await self._query_events_sqlite(
                event_type, user_id, session_id, since, until, limit, offset
            )

    async def _query_events_postgres(
        self,
        event_type: str | None,
        user_id: str | None,
        session_id: str | None,
        since: datetime | None,
        until: datetime | None,
        limit: int,
        offset: int,
    ) -> list[StorageEvent]:
        """Query events from PostgreSQL."""
        conditions = []
        params = []
        param_idx = 1

        if event_type:
            conditions.append(f"event_type = ${param_idx}")
            params.append(event_type)
            param_idx += 1

        if user_id:
            conditions.append(f"user_id = ${param_idx}")
            params.append(user_id)
            param_idx += 1

        if session_id:
            conditions.append(f"session_id = ${param_idx}")
            params.append(session_id)
            param_idx += 1

        if since:
            conditions.append(f"timestamp >= ${param_idx}")
            params.append(since)
            param_idx += 1

        if until:
            conditions.append(f"timestamp <= ${param_idx}")
            params.append(until)
            param_idx += 1

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        sql = f"""
            SELECT id, event_type, timestamp, user_id, session_id, collection_name,
                   operation_duration_ms, metadata, error_message
            FROM {self.config.table_name}
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([limit, offset])

        events = []
        async with self._pool.connection() as conn:
            result = await conn.execute(sql, *params)
            rows = await result.fetchall()

            for row in rows:
                events.append(
                    StorageEvent(
                        id=row[0],
                        event_type=row[1],
                        timestamp=row[2],
                        user_id=row[3],
                        session_id=row[4],
                        collection_name=row[5],
                        operation_duration_ms=row[6],
                        metadata=json.loads(row[7]) if row[7] else None,
                        error_message=row[8],
                    )
                )

        return events

    async def _query_events_sqlite(
        self,
        event_type: str | None,
        user_id: str | None,
        session_id: str | None,
        since: datetime | None,
        until: datetime | None,
        limit: int,
        offset: int,
    ) -> list[StorageEvent]:
        """Query events from SQLite."""
        conditions = []
        params = []

        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)

        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)

        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)

        if since:
            conditions.append("timestamp >= ?")
            params.append(since.isoformat())

        if until:
            conditions.append("timestamp <= ?")
            params.append(until.isoformat())

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        sql = f"""
            SELECT id, event_type, timestamp, user_id, session_id, collection_name,
                   operation_duration_ms, metadata, error_message
            FROM {self.config.table_name}
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        events = []
        cursor = await self._pool.execute(sql, params)
        rows = await cursor.fetchall()

        for row in rows:
            timestamp = datetime.fromisoformat(row[2]) if row[2] else datetime.now(UTC)
            events.append(
                StorageEvent(
                    id=row[0],
                    event_type=row[1],
                    timestamp=timestamp,
                    user_id=row[3],
                    session_id=row[4],
                    collection_name=row[5],
                    operation_duration_ms=row[6],
                    metadata=json.loads(row[7]) if row[7] else None,
                    error_message=row[8],
                )
            )

        return events

    async def count_events(
        self,
        event_type: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        since: datetime | None = None,
    ) -> int:
        """
        Count events matching criteria.

        Args:
            event_type: Filter by event type.
            user_id: Filter by user ID.
            session_id: Filter by session ID.
            since: Only count events after this timestamp.

        Returns:
            Number of matching events.
        """
        if not self.enabled:
            return 0

        conditions = []
        params = []

        if event_type:
            conditions.append(
                "event_type = $1" if self._backend == "postgres" else "event_type = ?"
            )
            params.append(event_type)

        if user_id:
            idx = len(params) + 1
            conditions.append(f"user_id = ${idx}" if self._backend == "postgres" else "user_id = ?")
            params.append(user_id)

        if session_id:
            idx = len(params) + 1
            conditions.append(
                f"session_id = ${idx}" if self._backend == "postgres" else "session_id = ?"
            )
            params.append(session_id)

        if since:
            idx = len(params) + 1
            conditions.append(
                f"timestamp >= ${idx}" if self._backend == "postgres" else "timestamp >= ?"
            )
            params.append(since if self._backend == "postgres" else since.isoformat())

        where_clause = (
            " AND ".join(conditions)
            if conditions
            else "TRUE"
            if self._backend == "postgres"
            else "1=1"
        )

        sql = f"SELECT COUNT(*) FROM {self.config.table_name} WHERE {where_clause}"

        if self._backend == "postgres":
            async with self._pool.connection() as conn:
                result = await conn.execute(sql, *params)
                row = await result.fetchone()
                return row[0] if row else 0
        else:
            cursor = await self._pool.execute(sql, params)
            row = await cursor.fetchone()
            return row[0] if row else 0

    # =========================================================================
    # CLEANUP
    # =========================================================================

    async def cleanup_old_events(self, retention_days: int | None = None) -> int:
        """
        Delete events older than the retention period.

        Args:
            retention_days: Override configured retention (0 = no cleanup).

        Returns:
            Number of events deleted.
        """
        if not self.enabled:
            return 0

        days = retention_days if retention_days is not None else self.config.retention_days
        if days <= 0:
            logger.debug("Event retention disabled; skipping cleanup")
            return 0

        cutoff = datetime.now(UTC) - timedelta(days=days)

        if self._backend == "postgres":
            sql = f"DELETE FROM {self.config.table_name} WHERE timestamp < $1"
            async with self._pool.connection() as conn:
                result = await conn.execute(sql, cutoff)
                deleted = result.rowcount if hasattr(result, "rowcount") else 0
        else:
            sql = f"DELETE FROM {self.config.table_name} WHERE timestamp < ?"
            cursor = await self._pool.execute(sql, (cutoff.isoformat(),))
            deleted = cursor.rowcount
            await self._pool.commit()

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} events older than {days} days")

        return deleted

    async def get_statistics(self) -> dict[str, Any]:
        """
        Get event logger statistics.

        Returns:
            Dictionary with event counts, storage stats, etc.
        """
        stats = {
            "enabled": self.enabled,
            "events_logged": self._events_logged,
            "events_dropped": self._events_dropped,
            "queue_size": self._event_queue.qsize(),
            "last_flush_time": self._last_flush_time,
        }

        if self.enabled:
            stats["total_events"] = await self.count_events()

            # Count by type
            type_counts = {}
            for event_type in EventType:
                count = await self.count_events(event_type=event_type.value)
                if count > 0:
                    type_counts[event_type.value] = count
            stats["events_by_type"] = type_counts

        return stats


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "DEFAULT_EVENT_LOGGER_CONFIG",
    "POSTGRES_EVENTS_SCHEMA",
    "SQLITE_EVENTS_SCHEMA",
    "EventLogger",
    "EventLoggerConfig",
    "EventType",
    "StorageEvent",
]
