# src/llmcore/memory/__init__.py
"""
Memory management module for the LLMCore library.

This package provides the hierarchical memory system with five tiers:

1. **Volatile** — In-memory working context (fast, ephemeral)
2. **Session** — Conversation-scoped persistent memory
3. **Semantic** — Long-term knowledge via embedding-based retrieval (RAG)
4. **Episodic** — Past experiences and outcomes for learning
5. **Hierarchical** — Coordinator across all tiers

The :class:`MemoryManager` serves as the primary interface for context
preparation and retrieval orchestration.

Sub-modules:
    - ``memory.volatile`` — Volatile memory tier
    - ``memory.session`` — Session memory tier
    - ``memory.semantic`` — Semantic memory (RAG) tier
    - ``memory.episodic`` — Episodic memory tier
    - ``memory.hierarchical`` — Cross-tier coordinator

References:
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §8 (Memory System)
"""

from .hierarchical import HierarchicalMemoryManager, MemoryItem, MemoryTier
from .manager import MemoryManager

__all__ = [
    "MemoryManager",
    "HierarchicalMemoryManager",
    "MemoryItem",
    "MemoryTier",
]
