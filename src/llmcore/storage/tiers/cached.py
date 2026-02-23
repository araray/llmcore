# src/llmcore/storage/tiers/cached.py
"""
Cached Storage Tier — Disk-backed LRU cache for warm data.

This tier sits between the volatile (in-memory) tier and the persistent
(database) tier.  It provides:

- Disk-based storage using SQLite for durability across process restarts
- LRU eviction when the cache exceeds size limits
- Optional TTL per item
- Promotion/demotion between tiers

Use cases:
- Frequently accessed RAG chunks that are too large for memory
- Embedding cache overflow
- Session context windows that have been evicted from volatile tier
  but aren't yet cold enough for full DB persistence

Architecture:
    VolatileMemoryTier (hot) → **CachedStorageTier (warm)** → PersistentStorageTier (cold)

Example::

    from llmcore.storage.tiers.cached import CachedStorageTier, CachedStorageConfig

    tier = CachedStorageTier(CachedStorageConfig(
        db_path="/tmp/llmcore_cache.db",
        max_items=50_000,
        default_ttl_seconds=86400,
    ))
    await tier.initialize()
    await tier.set("doc:123", {"text": "...", "embedding": [...]})
    data = await tier.get("doc:123")

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


class CachedStorageConfig(BaseModel):
    """Configuration for the cached storage tier.

    Attributes:
        enabled: Whether this tier is active.
        db_path: Path to the SQLite database file.
        max_items: Maximum number of items (0 = unlimited).
        max_size_bytes: Maximum total serialised size (0 = unlimited).
        default_ttl_seconds: Default TTL for items (None = no expiration).
        cleanup_interval_seconds: How often to evict expired / over-limit items.
        enable_stats: Track hit/miss/eviction statistics.
    """

    enabled: bool = Field(default=True, description="Enable cached storage tier")
    db_path: str = Field(
        default="~/.local/share/llmcore/cache_tier.db",
        description="SQLite database path",
    )
    max_items: int = Field(default=50_000, ge=0, description="Max cached items (0=unlimited)")
    max_size_bytes: int = Field(
        default=0, ge=0, description="Max total size in bytes (0=unlimited)"
    )
    default_ttl_seconds: float | None = Field(
        default=86400.0, description="Default TTL in seconds (None=forever)"
    )
    cleanup_interval_seconds: float = Field(default=300.0, ge=1.0)
    enable_stats: bool = Field(default=True)


# ---------------------------------------------------------------------------
# Cached tier implementation
# ---------------------------------------------------------------------------


class CachedStorageTier:
    """Disk-backed LRU cache using SQLite.

    This tier is designed for warm data — items that are accessed frequently
    enough to keep close but not so frequently that they need to live in
    memory (volatile tier).

    Thread/async safety: All public methods are async.  The underlying
    SQLite operations use aiosqlite for non-blocking I/O.

    Args:
        config: Tier configuration.
    """

    def __init__(self, config: CachedStorageConfig | None = None) -> None:
        self._config = config or CachedStorageConfig()
        self._db: Any = None  # aiosqlite connection
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
            "expirations": 0,
        }
        logger.debug(
            "CachedStorageTier created (db=%s, max_items=%d).",
            self._config.db_path,
            self._config.max_items,
        )

    async def initialize(self) -> None:
        """Open the SQLite database and create the cache table."""
        try:
            import aiosqlite
        except ImportError:
            logger.warning(
                "aiosqlite not installed; CachedStorageTier will operate in no-op mode. "
                "Install with: pip install aiosqlite"
            )
            return

        import os
        from pathlib import Path

        db_path = os.path.expanduser(self._config.db_path)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._db = await aiosqlite.connect(db_path)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS cache_tier (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                size_bytes INTEGER DEFAULT 0,
                ttl_expires_at REAL,
                last_accessed_at REAL NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_lru ON cache_tier(last_accessed_at)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_ttl ON cache_tier(ttl_expires_at)"
        )
        await self._db.commit()
        logger.info("CachedStorageTier initialized at %s.", db_path)

    async def get(self, key: str) -> Any | None:
        """Retrieve an item, updating its LRU timestamp.

        Returns:
            The deserialised value, or *None* on miss / expiry.
        """
        if self._db is None:
            return None

        now = time.time()
        async with self._db.execute(
            "SELECT value, ttl_expires_at FROM cache_tier WHERE key = ?", (key,)
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            self._stats["misses"] += 1
            return None

        value_json, ttl_expires = row
        if ttl_expires is not None and ttl_expires < now:
            # Expired
            await self._db.execute("DELETE FROM cache_tier WHERE key = ?", (key,))
            await self._db.commit()
            self._stats["expirations"] += 1
            self._stats["misses"] += 1
            return None

        # Update LRU timestamp
        await self._db.execute(
            "UPDATE cache_tier SET last_accessed_at = ? WHERE key = ?", (now, key)
        )
        await self._db.commit()
        self._stats["hits"] += 1

        try:
            return json.loads(value_json)
        except json.JSONDecodeError:
            return value_json

    async def set(self, key: str, value: Any, ttl_seconds: float | None = None) -> None:
        """Store an item in the cache.

        Args:
            key: Cache key.
            value: Value to store (must be JSON-serialisable).
            ttl_seconds: Override TTL (None → use config default).
        """
        if self._db is None:
            return

        now = time.time()
        value_json = json.dumps(value) if not isinstance(value, str) else value
        size = len(value_json.encode("utf-8"))

        ttl = ttl_seconds if ttl_seconds is not None else self._config.default_ttl_seconds
        expires = (now + ttl) if ttl is not None else None

        await self._db.execute(
            """INSERT OR REPLACE INTO cache_tier
               (key, value, size_bytes, ttl_expires_at, last_accessed_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (key, value_json, size, expires, now, now),
        )
        await self._db.commit()
        self._stats["sets"] += 1

        # Enforce limits
        await self._enforce_limits()

    async def delete(self, key: str) -> bool:
        """Remove an item from the cache."""
        if self._db is None:
            return False
        cursor = await self._db.execute("DELETE FROM cache_tier WHERE key = ?", (key,))
        await self._db.commit()
        return cursor.rowcount > 0

    async def clear(self) -> int:
        """Remove all items. Returns the number removed."""
        if self._db is None:
            return 0
        cursor = await self._db.execute("DELETE FROM cache_tier")
        await self._db.commit()
        return cursor.rowcount

    async def count(self) -> int:
        """Return the number of items in the cache."""
        if self._db is None:
            return 0
        async with self._db.execute("SELECT COUNT(*) FROM cache_tier") as cur:
            row = await cur.fetchone()
            return row[0] if row else 0

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        return {
            **self._stats,
            "hit_rate": self._stats["hits"] / total if total > 0 else 0.0,
        }

    async def cleanup_expired(self) -> int:
        """Remove all expired items. Returns the number removed."""
        if self._db is None:
            return 0
        now = time.time()
        cursor = await self._db.execute(
            "DELETE FROM cache_tier WHERE ttl_expires_at IS NOT NULL AND ttl_expires_at < ?",
            (now,),
        )
        await self._db.commit()
        removed = cursor.rowcount
        self._stats["expirations"] += removed
        return removed

    async def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None
            logger.info("CachedStorageTier closed.")

    # -- Internal helpers ----------------------------------------------------

    async def _enforce_limits(self) -> None:
        """Evict LRU items if cache exceeds configured limits."""
        if self._config.max_items <= 0:
            return

        current_count = await self.count()
        if current_count <= self._config.max_items:
            return

        # Evict oldest-accessed items
        overage = current_count - self._config.max_items
        cursor = await self._db.execute(
            "DELETE FROM cache_tier WHERE key IN "
            "(SELECT key FROM cache_tier ORDER BY last_accessed_at ASC LIMIT ?)",
            (overage,),
        )
        await self._db.commit()
        evicted = cursor.rowcount
        self._stats["evictions"] += evicted
        if evicted > 0:
            logger.debug("CachedStorageTier evicted %d items (LRU).", evicted)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_cached_tier(config: CachedStorageConfig | None = None) -> CachedStorageTier:
    """Create a CachedStorageTier instance from config."""
    return CachedStorageTier(config)


__all__ = [
    "CachedStorageConfig",
    "CachedStorageTier",
    "create_cached_tier",
]
