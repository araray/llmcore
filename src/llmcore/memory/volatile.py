# src/llmcore/memory/volatile.py
"""
Volatile Memory — In-memory working context.

This module provides the spec-mandated ``memory/volatile.py`` entry-point
that bridges to the actual volatile memory implementations:

- :class:`~llmcore.storage.tiers.volatile.VolatileMemoryTier` for raw KV storage
- :class:`~llmcore.agents.memory.memory_store.WorkingMemoryStore` for agent
  working memory

Volatile memory holds the agent's immediate context: the current goal,
recent tool results, conversation snippet, and any in-progress reasoning
state.  It is the fastest tier (pure in-memory) but does not survive
process restarts.

References:
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §8 (Memory System)
"""

from __future__ import annotations

from ..storage.tiers.volatile import (
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
