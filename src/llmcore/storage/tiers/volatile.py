# src/llmcore/storage/tiers/volatile.py
"""
Volatile Memory Tier - In-memory storage with TTL.

This module provides a thread-safe in-memory storage tier optimized for
session data and temporary caching. It supports:
- Automatic TTL (time-to-live) expiration
- Memory size limits with LRU eviction
- Fast read/write operations for hot data
- Comprehensive statistics tracking

The volatile tier is intended for:
- Session context that doesn't need persistence
- Temporary computation results
- Hot data caching before promotion to persistent storage

Key Features:
- Thread-safe with RLock protection
- O(1) get/set operations in common cases
- Configurable memory and item limits
- Automatic background cleanup of expired items
- Detailed statistics for monitoring

Usage:
    tier = VolatileMemoryTier(max_items=10000, default_ttl_seconds=3600)

    # Store session data
    tier.set("session:123", {"user": "alice", "context": [...]})

    # Store temporary data with custom TTL
    tier.set("temp:result", large_computation_result, ttl_seconds=300)

    # Retrieve data
    data = tier.get("session:123")
    if data is None:
        # Item expired or doesn't exist
        pass

    # Check statistics
    stats = tier.stats()
    print(f"Hit rate: {stats['hit_rate']:.2%}")

References:
- UNIFIED_IMPLEMENTATION_PLAN.md Phase 3, Task 3.1
- Storage_System_Spec_v2r0.md Section 3.1 (Volatile Tier)
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections.abc import Iterator

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


class VolatileMemoryConfig(BaseModel):
    """Configuration for volatile memory tier.

    Attributes:
        enabled: Whether the volatile tier is enabled.
        max_items: Maximum number of items to store (0 = unlimited).
        max_size_bytes: Maximum total size in bytes (0 = unlimited).
        default_ttl_seconds: Default TTL for items (None = no expiration).
        cleanup_interval_seconds: How often to run cleanup of expired items.
        enable_stats: Whether to track detailed statistics.
    """

    enabled: bool = Field(default=True, description="Enable volatile memory tier")
    max_items: int = Field(default=10000, ge=0, le=10000000, description="Maximum number of items")
    max_size_bytes: int = Field(
        default=100 * 1024 * 1024,  # 100 MB
        ge=0,
        description="Maximum total size in bytes (0=unlimited)",
    )
    default_ttl_seconds: int | None = Field(
        default=3600, ge=0, description="Default TTL in seconds (None=no expiry)"
    )
    cleanup_interval_seconds: int = Field(
        default=60, ge=1, le=3600, description="Cleanup interval in seconds"
    )
    enable_stats: bool = Field(default=True, description="Enable statistics tracking")


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class VolatileItem:
    """An item stored in volatile memory.

    Attributes:
        key: Unique identifier for the item.
        value: The stored value (any picklable object).
        created_at: Unix timestamp when item was created.
        expires_at: Unix timestamp when item expires (None = never).
        access_count: Number of times item has been accessed.
        last_accessed: Unix timestamp of last access.
        size_bytes: Estimated size of the value in bytes.
        metadata: Optional metadata for the item.
    """

    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if the item has expired.

        Returns:
            True if the item has a TTL and the current time is past expires_at.
        """
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def touch(self) -> None:
        """Update access timestamp and increment access count."""
        self.last_accessed = time.time()
        self.access_count += 1

    def time_until_expiry(self) -> float | None:
        """Get seconds until expiration.

        Returns:
            Seconds until expiry, 0 if already expired, None if no TTL.
        """
        if self.expires_at is None:
            return None
        remaining = self.expires_at - time.time()
        return max(0.0, remaining)

    def age_seconds(self) -> float:
        """Get the age of the item in seconds."""
        return time.time() - self.created_at


# =============================================================================
# VOLATILE MEMORY TIER
# =============================================================================


class VolatileMemoryTier:
    """In-memory storage with TTL support and LRU eviction.

    This class provides a thread-safe dictionary-based storage system with:
    - Automatic expiration of items based on TTL
    - LRU (Least Recently Used) eviction when limits are reached
    - Memory size tracking and limits
    - Comprehensive statistics

    The implementation uses a single RLock for thread safety, which provides
    good performance for mixed read/write workloads. For extremely high
    concurrency, consider using concurrent.futures or asyncio.

    Example:
        tier = VolatileMemoryTier(max_items=10000, default_ttl_seconds=3600)

        # Store data
        tier.set("user:123:session", session_data)
        tier.set("cache:result:abc", result, ttl_seconds=300)

        # Retrieve data
        session = tier.get("user:123:session")

        # Check stats
        stats = tier.stats()
        print(f"Items: {stats['item_count']}, Hit rate: {stats['hit_rate']:.2%}")

    Attributes:
        max_items: Maximum number of items (0 = unlimited).
        max_size_bytes: Maximum total size in bytes (0 = unlimited).
        default_ttl_seconds: Default TTL for new items (None = no expiry).
        cleanup_interval: Seconds between automatic cleanup runs.
    """

    def __init__(
        self,
        max_items: int = 10000,
        max_size_bytes: int = 100 * 1024 * 1024,
        default_ttl_seconds: int | None = 3600,
        cleanup_interval_seconds: int = 60,
        config: VolatileMemoryConfig | None = None,
    ) -> None:
        """Initialize the volatile memory tier.

        Args:
            max_items: Maximum number of items to store. Set to 0 for unlimited.
            max_size_bytes: Maximum total size in bytes. Set to 0 for unlimited.
            default_ttl_seconds: Default TTL for items. Set to None for no expiry.
            cleanup_interval_seconds: How often to check for expired items.
            config: Optional configuration object (overrides other params).
        """
        # Apply config if provided
        if config is not None:
            max_items = config.max_items
            max_size_bytes = config.max_size_bytes
            default_ttl_seconds = config.default_ttl_seconds
            cleanup_interval_seconds = config.cleanup_interval_seconds

        self.max_items = max_items
        self.max_size_bytes = max_size_bytes
        self.default_ttl_seconds = default_ttl_seconds
        self.cleanup_interval = cleanup_interval_seconds

        # Internal storage
        self._store: dict[str, VolatileItem] = {}
        self._lock = threading.RLock()
        self._current_size_bytes: int = 0
        self._last_cleanup: float = time.time()

        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0,
            "expirations": 0,
        }

        logger.debug(
            f"VolatileMemoryTier initialized: max_items={max_items}, "
            f"max_size_bytes={max_size_bytes}, default_ttl={default_ttl_seconds}s"
        )

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    def _estimate_size(self, value: Any) -> int:
        """Estimate the memory size of a value in bytes.

        This is a best-effort estimate. For complex nested structures,
        the actual memory usage may differ.

        Args:
            value: The value to estimate size for.

        Returns:
            Estimated size in bytes.
        """
        try:
            # sys.getsizeof works for most built-in types
            return sys.getsizeof(value)
        except TypeError:
            # Fallback: serialize to string and measure
            try:
                return len(str(value).encode("utf-8"))
            except Exception:
                # Ultimate fallback: assume 1KB
                return 1024

    def _maybe_cleanup(self) -> None:
        """Run cleanup if enough time has passed since last cleanup.

        This is called during get/set operations to ensure expired items
        are periodically removed without requiring a background thread.
        """
        if time.time() - self._last_cleanup > self.cleanup_interval:
            self._cleanup_expired()
            self._last_cleanup = time.time()

    def _cleanup_expired(self) -> int:
        """Remove all expired items from storage.

        Returns:
            Number of items removed.
        """
        expired_keys = [k for k, v in self._store.items() if v.is_expired()]
        for key in expired_keys:
            item = self._store.pop(key, None)
            if item:
                self._current_size_bytes -= item.size_bytes
                self._stats["expirations"] += 1

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired items")

        return len(expired_keys)

    def _evict_lru(self, needed_space: int = 0) -> int:
        """Evict least recently used items to make room.

        Evicts up to 10% of items or enough to free needed_space bytes,
        whichever is larger.

        Args:
            needed_space: Minimum bytes to free (0 = just evict for count limit).

        Returns:
            Number of items evicted.
        """
        if not self._store:
            return 0

        # Sort by last_accessed (oldest first)
        sorted_items = sorted(self._store.items(), key=lambda x: x[1].last_accessed)

        evicted = 0
        space_freed = 0
        # Evict at least 10% or until we have enough space
        max_evict = max(1, len(self._store) // 10)

        for key, item in sorted_items:
            if evicted >= max_evict and space_freed >= needed_space:
                break
            del self._store[key]
            self._current_size_bytes -= item.size_bytes
            space_freed += item.size_bytes
            evicted += 1
            self._stats["evictions"] += 1

        if evicted > 0:
            logger.debug(f"Evicted {evicted} LRU items, freed {space_freed} bytes")

        return evicted

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get(self, key: str) -> Any | None:
        """Get an item from storage.

        If the item exists and hasn't expired, its access timestamp and
        count are updated (LRU tracking).

        Args:
            key: The key to retrieve.

        Returns:
            The stored value, or None if not found or expired.
        """
        with self._lock:
            self._maybe_cleanup()
            item = self._store.get(key)

            if item is None:
                self._stats["misses"] += 1
                return None

            if item.is_expired():
                # Remove expired item
                del self._store[key]
                self._current_size_bytes -= item.size_bytes
                self._stats["misses"] += 1
                self._stats["expirations"] += 1
                return None

            # Update LRU tracking
            item.touch()
            self._stats["hits"] += 1
            return item.value

    def get_with_metadata(self, key: str) -> tuple[Any, dict[str, Any]] | None:
        """Get an item and its metadata.

        Args:
            key: The key to retrieve.

        Returns:
            Tuple of (value, metadata) or None if not found.
        """
        with self._lock:
            self._maybe_cleanup()
            item = self._store.get(key)

            if item is None or item.is_expired():
                if item and item.is_expired():
                    del self._store[key]
                    self._current_size_bytes -= item.size_bytes
                    self._stats["expirations"] += 1
                self._stats["misses"] += 1
                return None

            item.touch()
            self._stats["hits"] += 1

            return item.value, {
                "created_at": item.created_at,
                "expires_at": item.expires_at,
                "access_count": item.access_count,
                "last_accessed": item.last_accessed,
                "size_bytes": item.size_bytes,
                **item.metadata,
            }

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Store an item in memory.

        If the key already exists, the item is replaced. If limits are
        exceeded, LRU eviction is triggered.

        Args:
            key: The key to store under.
            value: The value to store.
            ttl_seconds: TTL override (None uses default, 0 means no expiry).
            metadata: Optional metadata to attach to the item.

        Returns:
            True if the item was stored successfully.
        """
        with self._lock:
            self._maybe_cleanup()

            size_bytes = self._estimate_size(value)

            # Check item count limit
            if self.max_items > 0 and len(self._store) >= self.max_items:
                if key not in self._store:  # Only evict if adding new item
                    self._evict_lru()

            # Check size limit
            if self.max_size_bytes > 0:
                if self._current_size_bytes + size_bytes > self.max_size_bytes:
                    self._evict_lru(needed_space=size_bytes)

            # Determine TTL
            if ttl_seconds is None:
                ttl = self.default_ttl_seconds
            elif ttl_seconds == 0:
                ttl = None  # 0 means no expiry
            else:
                ttl = ttl_seconds

            expires_at = time.time() + ttl if ttl else None

            # Remove old item if exists (to update size tracking)
            old_item = self._store.get(key)
            if old_item:
                self._current_size_bytes -= old_item.size_bytes

            # Store new item
            self._store[key] = VolatileItem(
                key=key,
                value=value,
                expires_at=expires_at,
                size_bytes=size_bytes,
                metadata=metadata or {},
            )
            self._current_size_bytes += size_bytes
            self._stats["sets"] += 1

            return True

    def delete(self, key: str) -> bool:
        """Remove an item from storage.

        Args:
            key: The key to delete.

        Returns:
            True if the item was deleted, False if it didn't exist.
        """
        with self._lock:
            item = self._store.pop(key, None)
            if item:
                self._current_size_bytes -= item.size_bytes
                self._stats["deletes"] += 1
                return True
            return False

    def exists(self, key: str) -> bool:
        """Check if a key exists and is not expired.

        Args:
            key: The key to check.

        Returns:
            True if the key exists and hasn't expired.
        """
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return False
            if item.is_expired():
                # Clean up expired item
                del self._store[key]
                self._current_size_bytes -= item.size_bytes
                self._stats["expirations"] += 1
                return False
            return True

    def keys(self, pattern: str | None = None) -> list[str]:
        """Get all keys, optionally filtered by prefix pattern.

        Args:
            pattern: Optional prefix to filter keys by.

        Returns:
            List of matching keys (excluding expired items).
        """
        with self._lock:
            self._maybe_cleanup()
            if pattern is None:
                return [k for k, v in self._store.items() if not v.is_expired()]
            return [
                k for k, v in self._store.items() if k.startswith(pattern) and not v.is_expired()
            ]

    def clear(self) -> int:
        """Remove all items from storage.

        Returns:
            Number of items removed.
        """
        with self._lock:
            count = len(self._store)
            self._store.clear()
            self._current_size_bytes = 0
            logger.debug(f"Cleared {count} items from volatile storage")
            return count

    def stats(self) -> dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dictionary containing:
            - item_count: Current number of items
            - size_bytes: Current total size in bytes
            - max_items: Maximum allowed items
            - max_size_bytes: Maximum allowed size
            - hit_rate: Cache hit rate (0.0 to 1.0)
            - hits: Total cache hits
            - misses: Total cache misses
            - sets: Total set operations
            - deletes: Total delete operations
            - evictions: Total LRU evictions
            - expirations: Total TTL expirations
        """
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total if total > 0 else 0.0

            return {
                "item_count": len(self._store),
                "size_bytes": self._current_size_bytes,
                "max_items": self.max_items,
                "max_size_bytes": self.max_size_bytes,
                "hit_rate": hit_rate,
                "default_ttl_seconds": self.default_ttl_seconds,
                **self._stats.copy(),
            }

    def get_item_info(self, key: str) -> dict[str, Any] | None:
        """Get detailed information about an item without touching it.

        Args:
            key: The key to get info for.

        Returns:
            Dictionary with item details, or None if not found.
        """
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None

            return {
                "key": item.key,
                "created_at": item.created_at,
                "expires_at": item.expires_at,
                "time_until_expiry": item.time_until_expiry(),
                "age_seconds": item.age_seconds(),
                "access_count": item.access_count,
                "last_accessed": item.last_accessed,
                "size_bytes": item.size_bytes,
                "is_expired": item.is_expired(),
                "metadata": item.metadata,
            }

    def __len__(self) -> int:
        """Get the number of items in storage."""
        with self._lock:
            return len(self._store)

    def __contains__(self, key: str) -> bool:
        """Check if a key exists."""
        return self.exists(key)

    def __iter__(self) -> Iterator[str]:
        """Iterate over keys."""
        return iter(self.keys())


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_volatile_tier(
    config: VolatileMemoryConfig | None = None,
    **kwargs: Any,
) -> VolatileMemoryTier:
    """Factory function to create a volatile memory tier.

    Args:
        config: Optional configuration object.
        **kwargs: Additional arguments passed to VolatileMemoryTier.

    Returns:
        Configured VolatileMemoryTier instance.
    """
    if config is not None:
        return VolatileMemoryTier(config=config)
    return VolatileMemoryTier(**kwargs)
