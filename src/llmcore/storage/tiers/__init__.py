# src/llmcore/storage/tiers/__init__.py
"""
Storage Tiers Package.

Provides tiered storage implementations for different access patterns
and performance requirements.

Tiers:
- **VolatileMemoryTier** (hot): In-memory storage with TTL for session data
- **CachedStorageTier** (warm): Disk-based LRU cache for frequently accessed data
- **PersistentStorageTier** (cold): Database-backed durable storage

Architecture::

    VolatileMemoryTier → CachedStorageTier → PersistentStorageTier
         (memory)           (SQLite LRU)        (SQLite/PostgreSQL)

References:
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §7.1 (Storage Tiers)
"""

from .cached import (
    CachedStorageConfig,
    CachedStorageTier,
    create_cached_tier,
)
from .persistent import (
    PersistentStorageConfig,
    PersistentStorageTier,
    create_persistent_tier,
)
from .volatile import (
    VolatileItem,
    VolatileMemoryConfig,
    VolatileMemoryTier,
    create_volatile_tier,
)

__all__ = [
    # Volatile (hot)
    "VolatileItem",
    "VolatileMemoryConfig",
    "VolatileMemoryTier",
    "create_volatile_tier",
    # Cached (warm)
    "CachedStorageConfig",
    "CachedStorageTier",
    "create_cached_tier",
    # Persistent (cold)
    "PersistentStorageConfig",
    "PersistentStorageTier",
    "create_persistent_tier",
]
