# src/llmcore/storage/tiers/__init__.py
"""
Storage Tiers Package.

This package provides tiered storage implementations for different
access patterns and performance requirements.

Tiers:
- VolatileMemoryTier: In-memory storage with TTL for session data
- (Future) CachedTier: Disk-based LRU cache for warm data
- (Future) PersistentTier: Database-backed storage for cold data

References:
- UNIFIED_IMPLEMENTATION_PLAN.md Phase 3, Task 3.1
- Storage_System_Spec_v2r0.md Section 3 (Tiered Architecture)
"""

from .volatile import (
    VolatileItem,
    VolatileMemoryConfig,
    VolatileMemoryTier,
    create_volatile_tier,
)

__all__ = [
    "VolatileItem",
    "VolatileMemoryConfig",
    "VolatileMemoryTier",
    "create_volatile_tier",
]
