# src/llmcore/agents/memory/__init__.py
"""
Memory Integration Package for Darwin Layer 2.

Provides two-tier memory model (working + long-term) inspired by
cognitive science, with support for semantic, episodic, and
procedural memory types.

Features:
    - Working memory with capacity limits (7±2 items, like humans)
    - Long-term memory stores (semantic, episodic, procedural)
    - Importance-based recall and retrieval
    - Persistent storage for cross-session learning
    - Pinning mechanism for critical items

Usage:
    from llmcore.agents.memory import MemoryManager, WorkingMemory

    # Unified memory management
    manager = MemoryManager()
    manager.remember(
        "User prefers Python",
        memory_type=MemoryType.SEMANTIC,
        importance=MemoryImportance.HIGH
    )

    # Recall relevant memories
    results = manager.recall("programming preference")

    # Working memory
    working = WorkingMemory(capacity=10)
    working.add(MemoryItem(...))
    context = working.get_context()  # Format for prompt
"""

from .integration import CognitiveMemoryIntegrator
from .memory_store import (
    InMemoryStore,
    # Long-term memory
    LongTermMemoryStore,
    MemoryImportance,
    # Data models
    MemoryItem,
    # Unified manager
    MemoryManager,
    # Enums
    MemoryType,
    PersistentMemoryStore,
    RecallResult,
    # Working memory
    WorkingMemory,
)

__all__ = [
    # Integration
    "CognitiveMemoryIntegrator",
    # Memory enums
    "MemoryType",
    "MemoryImportance",
    # Memory data models
    "MemoryItem",
    "RecallResult",
    # Working memory
    "WorkingMemory",
    # Long-term memory stores
    "LongTermMemoryStore",
    "InMemoryStore",
    "PersistentMemoryStore",
    # Unified manager
    "MemoryManager",
]
