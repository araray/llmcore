# src/llmcore/storage/tiers/persistent.py
"""
Persistent Storage Tier — Database-backed cold storage.

This is the final tier in the storage hierarchy.  It provides durable,
query-able storage for data that has been evicted from the volatile and
cached tiers or that must survive indefinitely (sessions, episodes,
agent state).

The persistent tier is a *thin adapter* that delegates to the existing
storage backends (:class:`~llmcore.storage.sqlite_session.SQLiteSessionStorage`,
:class:`~llmcore.storage.postgres_session_storage.PostgresSessionStorage`).
It adds:

- A uniform key-value interface consistent with the volatile and cached tiers
- Optional compression for large values
- Lifecycle hooks (promote / demote between tiers)

Architecture:
    VolatileMemoryTier (hot) → CachedStorageTier (warm) → **PersistentStorageTier (cold)**

Example::

    from llmcore.storage.tiers.persistent import PersistentStorageTier, PersistentStorageConfig

    tier = PersistentStorageTier(PersistentStorageConfig(backend="sqlite"))
    await tier.initialize()
    await tier.set("episode:42", {"summary": "...", "outcome": "success"})
    data = await tier.get("episode:42")

References:
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §7.1 (Storage Tiers)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class PersistentStorageConfig(BaseModel):
    """Configuration for the persistent storage tier.

    Attributes:
        enabled: Whether this tier is active.
        backend: Storage backend type (``"sqlite"`` or ``"postgres"``).
        db_path: SQLite file path (when backend == ``"sqlite"``).
        connection_string: PostgreSQL connection URL (when backend == ``"postgres"``).
        table_name: Table used for key-value persistent storage.
        enable_compression: Compress values larger than ``compression_threshold``.
        compression_threshold: Minimum value size (bytes) to trigger compression.
        enable_stats: Track read/write statistics.
    """

    enabled: bool = Field(default=True, description="Enable persistent storage tier")
    backend: str = Field(default="sqlite", description="Backend: 'sqlite' or 'postgres'")
    db_path: str = Field(
        default="~/.local/share/llmcore/persistent_tier.db",
        description="SQLite database path",
    )
    connection_string: str = Field(default="", description="PostgreSQL connection string")
    table_name: str = Field(default="persistent_kv", description="Table for key-value data")
    enable_compression: bool = Field(default=False, description="Compress large values")
    compression_threshold: int = Field(
        default=4096, ge=0, description="Min bytes to trigger compression"
    )
    enable_stats: bool = Field(default=True)


# ---------------------------------------------------------------------------
# Persistent tier implementation
# ---------------------------------------------------------------------------


class PersistentStorageTier:
    """Database-backed persistent key-value store.

    Acts as the cold-storage layer, providing durable persistence via
    SQLite (default) or PostgreSQL.

    Args:
        config: Tier configuration.
    """

    def __init__(self, config: PersistentStorageConfig | None = None) -> None:
        self._config = config or PersistentStorageConfig()
        self._db: Any = None
        self._stats = {"reads": 0, "writes": 0, "deletes": 0}
        logger.debug("PersistentStorageTier created (backend=%s).", self._config.backend)

    async def initialize(self) -> None:
        """Open the database and ensure the table exists."""
        if self._config.backend == "sqlite":
            await self._init_sqlite()
        elif self._config.backend == "postgres":
            await self._init_postgres()
        else:
            raise ValueError(f"Unsupported backend: {self._config.backend}")

    async def get(self, key: str) -> Any | None:
        """Read a value by key.

        Returns:
            The deserialised value, or *None* if the key does not exist.
        """
        if self._db is None:
            return None

        if self._config.backend == "sqlite":
            return await self._sqlite_get(key)
        else:
            return await self._postgres_get(key)

    async def set(self, key: str, value: Any) -> None:
        """Write a key-value pair (upsert)."""
        if self._db is None:
            return

        value_json = json.dumps(value) if not isinstance(value, str) else value

        # Optional compression
        compressed = False
        if (
            self._config.enable_compression
            and len(value_json.encode("utf-8")) > self._config.compression_threshold
        ):
            value_json = self._compress(value_json)
            compressed = True

        if self._config.backend == "sqlite":
            await self._sqlite_set(key, value_json, compressed)
        else:
            await self._postgres_set(key, value_json, compressed)

        self._stats["writes"] += 1

    async def delete(self, key: str) -> bool:
        """Delete a key."""
        if self._db is None:
            return False

        if self._config.backend == "sqlite":
            removed = await self._sqlite_delete(key)
        else:
            removed = await self._postgres_delete(key)

        if removed:
            self._stats["deletes"] += 1
        return removed

    async def exists(self, key: str) -> bool:
        """Check whether a key exists."""
        if self._db is None:
            return False

        if self._config.backend == "sqlite":
            async with self._db.execute(
                f"SELECT 1 FROM {self._config.table_name} WHERE key = ?", (key,)
            ) as cur:
                return (await cur.fetchone()) is not None
        else:
            # PostgreSQL path
            async with self._db.acquire() as conn:
                row = await conn.fetchrow(
                    f"SELECT 1 FROM {self._config.table_name} WHERE key = $1", key
                )
                return row is not None

    async def keys(self, prefix: str = "") -> list[str]:
        """List keys, optionally filtered by prefix."""
        if self._db is None:
            return []

        if self._config.backend == "sqlite":
            query = f"SELECT key FROM {self._config.table_name}"
            params: tuple[Any, ...] = ()
            if prefix:
                query += " WHERE key LIKE ?"
                params = (f"{prefix}%",)
            async with self._db.execute(query, params) as cur:
                rows = await cur.fetchall()
                return [r[0] for r in rows]
        else:
            async with self._db.acquire() as conn:
                if prefix:
                    rows = await conn.fetch(
                        f"SELECT key FROM {self._config.table_name} WHERE key LIKE $1",
                        f"{prefix}%",
                    )
                else:
                    rows = await conn.fetch(f"SELECT key FROM {self._config.table_name}")
                return [r["key"] for r in rows]

    def stats(self) -> dict[str, Any]:
        """Return read/write/delete statistics."""
        return dict(self._stats)

    async def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            if self._config.backend == "sqlite":
                await self._db.close()
            else:
                await self._db.close()
            self._db = None
            logger.info("PersistentStorageTier closed.")

    # -- SQLite helpers ------------------------------------------------------

    async def _init_sqlite(self) -> None:
        """Initialize SQLite backend."""
        try:
            import aiosqlite
        except ImportError:
            logger.warning("aiosqlite not installed; PersistentStorageTier(sqlite) will be no-op.")
            return

        import os
        from pathlib import Path

        db_path = os.path.expanduser(self._config.db_path)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(db_path)
        await self._db.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._config.table_name} (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                compressed INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        await self._db.commit()
        logger.info("PersistentStorageTier(sqlite) initialized at %s.", db_path)

    async def _init_postgres(self) -> None:
        """Initialize PostgreSQL backend."""
        try:
            import asyncpg
        except ImportError:
            logger.warning("asyncpg not installed; PersistentStorageTier(postgres) will be no-op.")
            return

        self._db = await asyncpg.create_pool(self._config.connection_string)
        async with self._db.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._config.table_name} (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    compressed BOOLEAN DEFAULT FALSE,
                    created_at DOUBLE PRECISION NOT NULL,
                    updated_at DOUBLE PRECISION NOT NULL
                )
            """)
        logger.info("PersistentStorageTier(postgres) initialized.")

    async def _sqlite_get(self, key: str) -> Any | None:
        async with self._db.execute(
            f"SELECT value, compressed FROM {self._config.table_name} WHERE key = ?",
            (key,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            self._stats["reads"] += 1
            return None
        value_json, compressed = row
        if compressed:
            value_json = self._decompress(value_json)
        self._stats["reads"] += 1
        try:
            return json.loads(value_json)
        except json.JSONDecodeError:
            return value_json

    async def _sqlite_set(self, key: str, value: str, compressed: bool) -> None:
        now = time.time()
        await self._db.execute(
            f"""INSERT OR REPLACE INTO {self._config.table_name}
                (key, value, compressed, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)""",
            (key, value, int(compressed), now, now),
        )
        await self._db.commit()

    async def _sqlite_delete(self, key: str) -> bool:
        cur = await self._db.execute(f"DELETE FROM {self._config.table_name} WHERE key = ?", (key,))
        await self._db.commit()
        return cur.rowcount > 0

    async def _postgres_get(self, key: str) -> Any | None:
        async with self._db.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT value, compressed FROM {self._config.table_name} WHERE key = $1",
                key,
            )
        if row is None:
            self._stats["reads"] += 1
            return None
        value_json = row["value"]
        if row["compressed"]:
            value_json = self._decompress(value_json)
        self._stats["reads"] += 1
        try:
            return json.loads(value_json)
        except json.JSONDecodeError:
            return value_json

    async def _postgres_set(self, key: str, value: str, compressed: bool) -> None:
        now = time.time()
        async with self._db.acquire() as conn:
            await conn.execute(
                f"""INSERT INTO {self._config.table_name}
                    (key, value, compressed, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (key) DO UPDATE SET
                        value = EXCLUDED.value,
                        compressed = EXCLUDED.compressed,
                        updated_at = EXCLUDED.updated_at""",
                key,
                value,
                compressed,
                now,
                now,
            )

    async def _postgres_delete(self, key: str) -> bool:
        async with self._db.acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM {self._config.table_name} WHERE key = $1", key
            )
            return result.split()[-1] != "0"

    # -- Compression helpers -------------------------------------------------

    @staticmethod
    def _compress(data: str) -> str:
        """Compress a string using zlib + base64 encoding."""
        import base64
        import zlib

        compressed = zlib.compress(data.encode("utf-8"), level=6)
        return base64.b64encode(compressed).decode("ascii")

    @staticmethod
    def _decompress(data: str) -> str:
        """Decompress a base64+zlib string."""
        import base64
        import zlib

        raw = base64.b64decode(data.encode("ascii"))
        return zlib.decompress(raw).decode("utf-8")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_persistent_tier(
    config: PersistentStorageConfig | None = None,
) -> PersistentStorageTier:
    """Create a PersistentStorageTier instance from config."""
    return PersistentStorageTier(config)


__all__ = [
    "PersistentStorageConfig",
    "PersistentStorageTier",
    "create_persistent_tier",
]
