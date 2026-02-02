# src/llmcore/embedding/cache.py
"""
Embedding Cache Implementation.

This module provides a two-tier caching system for embeddings:
1. LRU in-memory cache for hot data (fast access)
2. SQLite on-disk cache for persistence (survives restarts)

The cache significantly reduces API costs by avoiding redundant embedding calls
for identical text+model combinations.

Key Features:
- Thread-safe operations with proper locking
- Batch operations for efficiency
- Cache statistics and hit rate tracking
- Configurable memory and disk sizes
- Optional TTL (time-to-live) for cache entries

Cache Key: SHA256(text + model_name + provider)
Cache Value: List[float] (the embedding vector)

Usage:
    cache = EmbeddingCache(
        memory_size=10000,
        disk_path="~/.cache/llmcore/embeddings.db"
    )

    # Check cache first
    embedding = cache.get(text, model, provider)
    if embedding is None:
        # Generate embedding via API
        embedding = api.embed(text)
        # Store in cache
        cache.set(text, model, provider, embedding)

    # Batch operations
    results, missing_indices = cache.get_batch(texts, model, provider)
    if missing_indices:
        new_embeddings = api.embed_batch([texts[i] for i in missing_indices])
        cache.set_batch([texts[i] for i in missing_indices], model, provider, new_embeddings)

References:
- UNIFIED_IMPLEMENTATION_PLAN.md Phase 1, Task 1.2
- RAG_ECOSYSTEM_REDESIGN_SPEC.md Section 4.1
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================


class EmbeddingCacheConfig(BaseModel):
    """Configuration for the embedding cache.

    Attributes:
        enabled: Whether caching is enabled.
        memory_size: Maximum number of entries in the in-memory LRU cache.
        disk_enabled: Whether to persist cache to disk (SQLite).
        disk_path: Path to the SQLite database file.
        disk_max_entries: Maximum number of entries on disk (0 = unlimited).
        ttl_hours: Time-to-live for cache entries in hours (0 = no expiration).
        compression_enabled: Whether to compress embeddings on disk (future).
    """

    enabled: bool = Field(default=True, description="Enable embedding caching")
    memory_size: int = Field(
        default=10000, ge=0, le=1000000, description="Max LRU cache entries in memory"
    )
    disk_enabled: bool = Field(default=True, description="Enable disk persistence")
    disk_path: str = Field(
        default="~/.cache/llmcore/embeddings.db", description="Path to SQLite cache DB"
    )
    disk_max_entries: int = Field(
        default=0, ge=0, description="Max disk entries (0=unlimited)"
    )
    ttl_hours: int = Field(default=0, ge=0, description="Cache TTL in hours (0=no expiry)")
    compression_enabled: bool = Field(
        default=False, description="Compress embeddings on disk (future)"
    )


# =============================================================================
# LRU CACHE (IN-MEMORY)
# =============================================================================


class LRUCache:
    """Thread-safe LRU cache for in-memory embedding storage.

    This cache provides O(1) access and update operations using an OrderedDict.
    When the cache reaches capacity, the least recently used item is evicted.

    Attributes:
        maxsize: Maximum number of items to store.
        hits: Total cache hits.
        misses: Total cache misses.
    """

    def __init__(self, maxsize: int = 10000) -> None:
        """Initialize LRU cache.

        Args:
            maxsize: Maximum number of items to store. Set to 0 to disable.
        """
        self.maxsize = maxsize
        self._cache: OrderedDict[str, List[float]] = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[List[float]]:
        """Get item from cache, moving it to end (most recent).

        Args:
            key: Cache key.

        Returns:
            Cached embedding or None if not found.
        """
        if self.maxsize == 0:
            self.misses += 1
            return None

        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self.hits += 1
                return self._cache[key]
            self.misses += 1
            return None

    def set(self, key: str, value: List[float]) -> None:
        """Store item in cache, evicting oldest if necessary.

        Args:
            key: Cache key.
            value: Embedding vector to store.
        """
        if self.maxsize == 0:
            return

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = value
            else:
                if len(self._cache) >= self.maxsize:
                    # Remove oldest item (first item)
                    self._cache.popitem(last=False)
                self._cache[key] = value

    def clear(self) -> None:
        """Clear all items from cache and reset statistics."""
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0

    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache without updating LRU order."""
        with self._lock:
            return key in self._cache

    def __len__(self) -> int:
        """Return number of items in cache."""
        with self._lock:
            return len(self._cache)

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with size, maxsize, hits, misses, and hit_rate.
        """
        with self._lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "maxsize": self.maxsize,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(hit_rate, 4),
            }


# =============================================================================
# SQLITE DISK CACHE
# =============================================================================


class DiskCache:
    """SQLite-based disk cache for embedding persistence.

    This cache provides durable storage that survives process restarts.
    Embeddings are stored as JSON-serialized arrays in a SQLite database.

    Table Schema:
        cache_key (TEXT PRIMARY KEY): SHA256 hash of text+model+provider
        embedding (TEXT): JSON-serialized embedding vector
        model (TEXT): Model identifier
        provider (TEXT): Provider name
        text_hash (TEXT): SHA256 hash of original text (for debugging)
        created_at (INTEGER): Unix timestamp of creation
        accessed_at (INTEGER): Unix timestamp of last access

    Attributes:
        db_path: Path to the SQLite database file.
        max_entries: Maximum number of entries (0 = unlimited).
        ttl_seconds: Time-to-live for entries in seconds (0 = no expiry).
    """

    def __init__(
        self,
        db_path: str,
        max_entries: int = 0,
        ttl_seconds: int = 0,
    ) -> None:
        """Initialize disk cache.

        Args:
            db_path: Path to SQLite database file.
            max_entries: Maximum entries to store (0 = unlimited).
            ttl_seconds: TTL for entries in seconds (0 = no expiry).
        """
        self.db_path = Path(db_path).expanduser()
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self._local = threading.local()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT PRIMARY KEY,
                    embedding TEXT NOT NULL,
                    model TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    created_at INTEGER NOT NULL,
                    accessed_at INTEGER NOT NULL
                )
            """)

            # Create indices for efficient queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_accessed_at
                ON embeddings(accessed_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_model_provider
                ON embeddings(model, provider)
            """)

            conn.commit()
            logger.debug(f"Disk cache initialized at {self.db_path}")

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get thread-local database connection.

        SQLite connections are not thread-safe, so we maintain one per thread.

        Yields:
            sqlite3.Connection for the current thread.
        """
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0,
            )
            # Enable WAL mode for better concurrent access
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")

        try:
            yield self._local.conn
        except Exception:
            self._local.conn.rollback()
            raise

    def get(self, key: str) -> Optional[List[float]]:
        """Get embedding from disk cache.

        Args:
            key: Cache key (SHA256 hash).

        Returns:
            Embedding vector or None if not found or expired.
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT embedding, created_at FROM embeddings WHERE cache_key = ?",
                    (key,),
                )
                row = cursor.fetchone()

                if row is None:
                    return None

                embedding_json, created_at = row

                # Check TTL
                if self.ttl_seconds > 0:
                    if time.time() - created_at > self.ttl_seconds:
                        # Entry expired, delete it
                        conn.execute(
                            "DELETE FROM embeddings WHERE cache_key = ?",
                            (key,),
                        )
                        conn.commit()
                        return None

                # Update accessed_at
                conn.execute(
                    "UPDATE embeddings SET accessed_at = ? WHERE cache_key = ?",
                    (int(time.time()), key),
                )
                conn.commit()

                return json.loads(embedding_json)

    def set(
        self,
        key: str,
        embedding: List[float],
        model: str,
        provider: str,
        text_hash: str,
    ) -> None:
        """Store embedding in disk cache.

        Args:
            key: Cache key (SHA256 hash).
            embedding: Embedding vector.
            model: Model identifier.
            provider: Provider name.
            text_hash: SHA256 hash of original text.
        """
        with self._lock:
            with self._get_connection() as conn:
                now = int(time.time())

                conn.execute(
                    """
                    INSERT OR REPLACE INTO embeddings
                    (cache_key, embedding, model, provider, text_hash, dimension, created_at, accessed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key,
                        json.dumps(embedding),
                        model,
                        provider,
                        text_hash,
                        len(embedding),
                        now,
                        now,
                    ),
                )

                # Enforce max_entries limit if set
                if self.max_entries > 0:
                    cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
                    count = cursor.fetchone()[0]

                    if count > self.max_entries:
                        # Delete oldest entries (by accessed_at)
                        excess = count - self.max_entries
                        conn.execute(
                            """
                            DELETE FROM embeddings WHERE cache_key IN (
                                SELECT cache_key FROM embeddings
                                ORDER BY accessed_at ASC
                                LIMIT ?
                            )
                            """,
                            (excess,),
                        )

                conn.commit()

    def get_batch(self, keys: List[str]) -> Dict[str, List[float]]:
        """Get multiple embeddings from disk cache.

        Args:
            keys: List of cache keys.

        Returns:
            Dictionary mapping found keys to embeddings.
        """
        if not keys:
            return {}

        with self._lock:
            with self._get_connection() as conn:
                placeholders = ",".join("?" * len(keys))
                cursor = conn.execute(
                    f"""
                    SELECT cache_key, embedding, created_at
                    FROM embeddings
                    WHERE cache_key IN ({placeholders})
                    """,
                    keys,
                )

                results = {}
                now = time.time()
                expired_keys = []
                found_keys = []

                for cache_key, embedding_json, created_at in cursor:
                    # Check TTL
                    if self.ttl_seconds > 0 and now - created_at > self.ttl_seconds:
                        expired_keys.append(cache_key)
                        continue

                    results[cache_key] = json.loads(embedding_json)
                    found_keys.append(cache_key)

                # Delete expired entries
                if expired_keys:
                    placeholders = ",".join("?" * len(expired_keys))
                    conn.execute(
                        f"DELETE FROM embeddings WHERE cache_key IN ({placeholders})",
                        expired_keys,
                    )

                # Update accessed_at for found entries
                if found_keys:
                    placeholders = ",".join("?" * len(found_keys))
                    conn.execute(
                        f"""
                        UPDATE embeddings SET accessed_at = ?
                        WHERE cache_key IN ({placeholders})
                        """,
                        [int(now)] + found_keys,
                    )

                conn.commit()
                return results

    def clear(self) -> None:
        """Clear all entries from disk cache."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM embeddings")
                conn.execute("VACUUM")
                conn.commit()
        logger.info("Disk cache cleared")

    def cleanup_expired(self) -> int:
        """Remove expired entries from disk cache.

        Returns:
            Number of entries removed.
        """
        if self.ttl_seconds <= 0:
            return 0

        with self._lock:
            with self._get_connection() as conn:
                cutoff = int(time.time()) - self.ttl_seconds
                cursor = conn.execute(
                    "DELETE FROM embeddings WHERE created_at <= ?",
                    (cutoff,),
                )
                deleted = cursor.rowcount
                conn.commit()

                if deleted > 0:
                    logger.info(f"Cleaned up {deleted} expired cache entries")

                return deleted

    @property
    def stats(self) -> Dict[str, Any]:
        """Get disk cache statistics.

        Returns:
            Dictionary with count, size_bytes, oldest_entry, etc.
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
                count = cursor.fetchone()[0]

                cursor = conn.execute(
                    "SELECT MIN(created_at), MAX(accessed_at) FROM embeddings"
                )
                oldest, newest = cursor.fetchone()

                # Get file size
                try:
                    size_bytes = self.db_path.stat().st_size
                except OSError:
                    size_bytes = 0

                return {
                    "count": count,
                    "max_entries": self.max_entries,
                    "ttl_seconds": self.ttl_seconds,
                    "size_bytes": size_bytes,
                    "size_mb": round(size_bytes / (1024 * 1024), 2),
                    "oldest_entry": (
                        datetime.fromtimestamp(oldest).isoformat() if oldest else None
                    ),
                    "newest_access": (
                        datetime.fromtimestamp(newest).isoformat() if newest else None
                    ),
                    "db_path": str(self.db_path),
                }

    def close(self) -> None:
        """Close database connection for current thread."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None


# =============================================================================
# UNIFIED EMBEDDING CACHE
# =============================================================================


class EmbeddingCache:
    """Two-tier embedding cache combining LRU memory cache and SQLite disk cache.

    This class provides a unified interface for caching embeddings with:
    - Fast in-memory access via LRU cache
    - Persistent storage via SQLite
    - Automatic promotion from disk to memory on access
    - Thread-safe operations
    - Batch operations for efficiency
    - Comprehensive statistics

    The caching strategy is:
    1. Check memory cache first (fastest)
    2. If miss, check disk cache
    3. If found on disk, promote to memory cache
    4. If miss on both, return None (caller should generate embedding)
    5. On set, store in both memory and disk

    Example:
        cache = EmbeddingCache(
            config=EmbeddingCacheConfig(
                memory_size=10000,
                disk_path="~/.cache/llmcore/embeddings.db"
            )
        )

        # Check cache
        embedding = cache.get("Hello world", "text-embedding-3-small", "openai")
        if embedding is None:
            embedding = api.embed("Hello world")
            cache.set("Hello world", "text-embedding-3-small", "openai", embedding)
    """

    def __init__(
        self,
        config: Optional[EmbeddingCacheConfig] = None,
        memory_size: int = 10000,
        disk_path: str = "~/.cache/llmcore/embeddings.db",
        disk_enabled: bool = True,
        disk_max_entries: int = 0,
        ttl_hours: int = 0,
    ) -> None:
        """Initialize embedding cache.

        Args:
            config: EmbeddingCacheConfig instance. If provided, other args are ignored.
            memory_size: Maximum LRU cache entries (used if config is None).
            disk_path: Path to SQLite cache DB (used if config is None).
            disk_enabled: Enable disk persistence (used if config is None).
            disk_max_entries: Max disk entries (used if config is None).
            ttl_hours: Cache TTL in hours (used if config is None).
        """
        if config is not None:
            self._config = config
        else:
            self._config = EmbeddingCacheConfig(
                enabled=True,
                memory_size=memory_size,
                disk_enabled=disk_enabled,
                disk_path=disk_path,
                disk_max_entries=disk_max_entries,
                ttl_hours=ttl_hours,
            )

        self._enabled = self._config.enabled

        # Initialize memory cache
        self._memory_cache = LRUCache(
            maxsize=self._config.memory_size if self._enabled else 0
        )

        # Initialize disk cache if enabled
        self._disk_cache: Optional[DiskCache] = None
        if self._enabled and self._config.disk_enabled:
            self._disk_cache = DiskCache(
                db_path=self._config.disk_path,
                max_entries=self._config.disk_max_entries,
                ttl_seconds=self._config.ttl_hours * 3600,
            )

        logger.info(
            f"EmbeddingCache initialized: enabled={self._enabled}, "
            f"memory_size={self._config.memory_size}, "
            f"disk_enabled={self._config.disk_enabled}"
        )

    @staticmethod
    def _make_cache_key(text: str, model: str, provider: str) -> str:
        """Generate cache key from text, model, and provider.

        Args:
            text: Text to embed.
            model: Model identifier.
            provider: Provider name.

        Returns:
            SHA256 hash as hex string.
        """
        combined = f"{provider}:{model}:{text}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_text(text: str) -> str:
        """Generate hash of text for debugging/tracking.

        Args:
            text: Text to hash.

        Returns:
            SHA256 hash as hex string.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str, model: str, provider: str) -> Optional[List[float]]:
        """Get embedding from cache.

        First checks memory cache, then disk cache.
        If found on disk, promotes to memory cache.

        Args:
            text: Text that was embedded.
            model: Model used for embedding.
            provider: Provider name.

        Returns:
            Cached embedding vector or None if not found.
        """
        if not self._enabled:
            return None

        cache_key = self._make_cache_key(text, model, provider)

        # Check memory cache first
        embedding = self._memory_cache.get(cache_key)
        if embedding is not None:
            logger.debug(f"Memory cache hit for model={model}")
            return embedding

        # Check disk cache
        if self._disk_cache is not None:
            embedding = self._disk_cache.get(cache_key)
            if embedding is not None:
                # Promote to memory cache
                self._memory_cache.set(cache_key, embedding)
                logger.debug(f"Disk cache hit for model={model}, promoted to memory")
                return embedding

        logger.debug(f"Cache miss for model={model}")
        return None

    def set(
        self,
        text: str,
        model: str,
        provider: str,
        embedding: List[float],
    ) -> None:
        """Store embedding in cache.

        Stores in both memory and disk caches.

        Args:
            text: Text that was embedded.
            model: Model used for embedding.
            provider: Provider name.
            embedding: Embedding vector to cache.
        """
        if not self._enabled:
            return

        cache_key = self._make_cache_key(text, model, provider)
        text_hash = self._hash_text(text)

        # Store in memory cache
        self._memory_cache.set(cache_key, embedding)

        # Store in disk cache
        if self._disk_cache is not None:
            self._disk_cache.set(cache_key, embedding, model, provider, text_hash)

        logger.debug(f"Cached embedding for model={model}, dim={len(embedding)}")

    def get_batch(
        self,
        texts: List[str],
        model: str,
        provider: str,
    ) -> Tuple[List[Optional[List[float]]], List[int]]:
        """Get embeddings for multiple texts from cache.

        Args:
            texts: List of texts to look up.
            model: Model used for embeddings.
            provider: Provider name.

        Returns:
            Tuple of:
            - List of embeddings (None for cache misses)
            - List of indices that were cache misses
        """
        if not self._enabled:
            return [None] * len(texts), list(range(len(texts)))

        results: List[Optional[List[float]]] = [None] * len(texts)
        missing_indices: List[int] = []

        cache_keys = [
            self._make_cache_key(text, model, provider) for text in texts
        ]

        # Check memory cache
        disk_lookup_needed: List[Tuple[int, str]] = []

        for i, key in enumerate(cache_keys):
            embedding = self._memory_cache.get(key)
            if embedding is not None:
                results[i] = embedding
            else:
                disk_lookup_needed.append((i, key))

        # Batch lookup from disk
        if disk_lookup_needed and self._disk_cache is not None:
            keys_to_lookup = [key for _, key in disk_lookup_needed]
            disk_results = self._disk_cache.get_batch(keys_to_lookup)

            for i, key in disk_lookup_needed:
                if key in disk_results:
                    embedding = disk_results[key]
                    results[i] = embedding
                    # Promote to memory cache
                    self._memory_cache.set(key, embedding)
                else:
                    missing_indices.append(i)
        else:
            # No disk cache, all memory misses are misses
            missing_indices = [i for i, _ in disk_lookup_needed]

        hits = len(texts) - len(missing_indices)
        if texts:
            logger.info(
                f"Batch cache lookup: {hits}/{len(texts)} hits "
                f"({100 * hits / len(texts):.1f}% hit rate)"
            )

        return results, missing_indices

    def set_batch(
        self,
        texts: List[str],
        model: str,
        provider: str,
        embeddings: List[List[float]],
    ) -> None:
        """Store multiple embeddings in cache.

        Args:
            texts: List of texts that were embedded.
            model: Model used for embeddings.
            provider: Provider name.
            embeddings: List of embedding vectors.
        """
        if not self._enabled:
            return

        if len(texts) != len(embeddings):
            raise ValueError(
                f"Length mismatch: {len(texts)} texts vs {len(embeddings)} embeddings"
            )

        for text, embedding in zip(texts, embeddings):
            self.set(text, model, provider, embedding)

    def clear(self) -> None:
        """Clear all entries from both memory and disk caches."""
        self._memory_cache.clear()
        if self._disk_cache is not None:
            self._disk_cache.clear()
        logger.info("Embedding cache cleared")

    def cleanup_expired(self) -> int:
        """Remove expired entries from disk cache.

        Returns:
            Number of entries removed.
        """
        if self._disk_cache is not None:
            return self._disk_cache.cleanup_expired()
        return 0

    @property
    def enabled(self) -> bool:
        """Return whether caching is enabled."""
        return self._enabled

    @property
    def stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics.

        Returns:
            Dictionary with memory and disk cache statistics.
        """
        memory_stats = self._memory_cache.stats
        disk_stats = self._disk_cache.stats if self._disk_cache else None

        # Calculate combined hit rate
        total_hits = memory_stats["hits"]
        total_misses = memory_stats["misses"]

        if disk_stats:
            # Disk hits are memory misses that were found on disk
            # This is approximate since we don't track disk hits separately
            pass

        total = total_hits + total_misses
        combined_hit_rate = total_hits / total if total > 0 else 0.0

        return {
            "enabled": self._enabled,
            "memory": memory_stats,
            "disk": disk_stats,
            "combined_hit_rate": round(combined_hit_rate, 4),
        }

    def close(self) -> None:
        """Close cache resources."""
        if self._disk_cache is not None:
            self._disk_cache.close()
        logger.debug("EmbeddingCache closed")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_embedding_cache(
    config: Optional[Dict[str, Any]] = None,
) -> EmbeddingCache:
    """Create an EmbeddingCache from a configuration dictionary.

    Args:
        config: Configuration dictionary (typically from confy config).
                Expected keys match EmbeddingCacheConfig attributes.

    Returns:
        Configured EmbeddingCache instance.

    Example:
        config = llmcore_config.get("embedding.cache", {})
        cache = create_embedding_cache(config)
    """
    if config is None:
        config = {}

    cache_config = EmbeddingCacheConfig(**config)
    return EmbeddingCache(config=cache_config)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "DiskCache",
    "EmbeddingCache",
    "EmbeddingCacheConfig",
    "LRUCache",
    "create_embedding_cache",
]
