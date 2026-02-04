# src/llmcore/agents/memory/memory_store.py
"""
Memory System for Agent Learning and Persistence.

Implements a two-tier memory model inspired by cognitive science:
- Working Memory: In-context, limited capacity, fast access
- Long-term Memory: Persistent, unlimited capacity, retrieval-based

Memory Types:
- Semantic: Facts and knowledge
- Episodic: Task experiences and outcomes
- Procedural: Learned skills and procedures

Research Foundation:
- CoALA: Cognitive Architectures for Language Agents (Sumers et al., 2023)
- Human memory systems (declarative vs. procedural)

Usage:
    from llmcore.agents.memory import MemoryManager, MemoryType

    memory = MemoryManager()

    # Store a fact
    await memory.remember(
        "User prefers Python over JavaScript",
        memory_type=MemoryType.SEMANTIC
    )

    # Store an experience
    await memory.remember(
        "Successfully used pandas for CSV analysis",
        memory_type=MemoryType.EPISODIC,
        metadata={"task": "data_analysis", "success": True}
    )

    # Recall relevant memories
    memories = await memory.recall(
        "How should I analyze this CSV file?",
        max_results=5
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
)

try:
    from pydantic import BaseModel, Field

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object

    def Field(*args, **kwargs):
        return kwargs.get("default")


logger = logging.getLogger(__name__)


# =============================================================================
# Memory Types
# =============================================================================


class MemoryType(str, Enum):
    """Types of memory following cognitive science models."""

    SEMANTIC = "semantic"  # Facts and knowledge
    EPISODIC = "episodic"  # Task experiences and outcomes
    PROCEDURAL = "procedural"  # Learned skills and procedures
    WORKING = "working"  # Short-term, in-context memory


class MemoryImportance(str, Enum):
    """Importance levels for memory items."""

    CRITICAL = "critical"  # Always retrieve
    HIGH = "high"  # Prioritize in retrieval
    MEDIUM = "medium"  # Normal retrieval
    LOW = "low"  # Retrieve if relevant


# =============================================================================
# Memory Items
# =============================================================================


@dataclass
class MemoryItem:
    """A single memory item."""

    id: str
    content: str
    memory_type: MemoryType
    importance: MemoryImportance = MemoryImportance.MEDIUM
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    relevance: float = 0.0  # Set during retrieval
    source: str = ""

    def touch(self) -> None:
        """Update access tracking."""
        self.accessed_at = time.time()
        self.access_count += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "importance": self.importance.value,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "access_count": self.access_count,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryItem:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            importance=MemoryImportance(data.get("importance", "medium")),
            embedding=data.get("embedding"),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", time.time()),
            accessed_at=data.get("accessed_at", time.time()),
            access_count=data.get("access_count", 0),
            source=data.get("source", ""),
        )


@dataclass
class RecallResult:
    """Result of memory recall."""

    memories: list[MemoryItem]
    query: str
    total_searched: int
    retrieval_time_ms: int


# =============================================================================
# Working Memory (In-Context)
# =============================================================================


class WorkingMemory:
    """
    In-context working memory with limited capacity.

    Mimics human working memory:
    - Limited capacity (typically 7±2 items)
    - Fast access
    - Automatically manages what to keep

    Usage:
        wm = WorkingMemory(capacity=10)
        wm.add("Current goal is to analyze data")
        wm.add("User prefers verbose output")

        context = wm.get_context()  # Get all items for prompt
    """

    def __init__(
        self,
        capacity: int = 10,
        importance_threshold: MemoryImportance = MemoryImportance.LOW,
    ):
        self.capacity = capacity
        self.importance_threshold = importance_threshold

        self._items: OrderedDict[str, MemoryItem] = OrderedDict()
        self._pinned: set[str] = set()  # Items that won't be evicted

    def add(
        self,
        content: str,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        metadata: dict[str, Any] | None = None,
        pinned: bool = False,
    ) -> str:
        """
        Add item to working memory.

        Args:
            content: Memory content
            importance: Importance level
            metadata: Optional metadata
            pinned: If True, won't be evicted

        Returns:
            Memory item ID
        """
        item_id = hashlib.md5(content.encode()).hexdigest()[:12]

        item = MemoryItem(
            id=item_id,
            content=content,
            memory_type=MemoryType.WORKING,
            importance=importance,
            metadata=metadata or {},
        )

        # Evict if at capacity
        while len(self._items) >= self.capacity:
            evicted = self._evict_one()
            if not evicted:
                break  # All items pinned

        self._items[item_id] = item

        if pinned:
            self._pinned.add(item_id)

        return item_id

    def get(self, item_id: str) -> MemoryItem | None:
        """Get item by ID."""
        item = self._items.get(item_id)
        if item:
            item.touch()
            # Move to end (most recent)
            self._items.move_to_end(item_id)
        return item

    def remove(self, item_id: str) -> bool:
        """Remove item from working memory."""
        if item_id in self._items:
            del self._items[item_id]
            self._pinned.discard(item_id)
            return True
        return False

    def pin(self, item_id: str) -> bool:
        """Pin an item to prevent eviction."""
        if item_id in self._items:
            self._pinned.add(item_id)
            return True
        return False

    def unpin(self, item_id: str) -> None:
        """Unpin an item."""
        self._pinned.discard(item_id)

    def get_all(self) -> list[MemoryItem]:
        """Get all items in working memory."""
        return list(self._items.values())

    def get_context(self, max_items: int = None) -> str:
        """
        Get formatted context string for prompt inclusion.

        Args:
            max_items: Maximum items to include

        Returns:
            Formatted context string
        """
        items = self.get_all()
        if max_items:
            items = items[-max_items:]  # Most recent

        if not items:
            return ""

        lines = ["## Working Memory"]
        for item in items:
            prefix = "📌 " if item.id in self._pinned else "• "
            lines.append(f"{prefix}{item.content}")

        return "\n".join(lines)

    def clear(self, keep_pinned: bool = True) -> int:
        """
        Clear working memory.

        Args:
            keep_pinned: Whether to keep pinned items

        Returns:
            Number of items cleared
        """
        if keep_pinned:
            to_remove = [item_id for item_id in self._items if item_id not in self._pinned]
            for item_id in to_remove:
                del self._items[item_id]
            return len(to_remove)
        else:
            count = len(self._items)
            self._items.clear()
            self._pinned.clear()
            return count

    def _evict_one(self) -> bool:
        """Evict one item (LRU among non-pinned)."""
        for item_id in self._items:
            if item_id not in self._pinned:
                del self._items[item_id]
                return True
        return False

    @property
    def count(self) -> int:
        """Number of items in working memory."""
        return len(self._items)

    @property
    def free_slots(self) -> int:
        """Number of free slots."""
        return max(0, self.capacity - len(self._items))


# =============================================================================
# Long-term Memory Store (Abstract)
# =============================================================================


class LongTermMemoryStore(ABC):
    """Abstract base for long-term memory stores."""

    @abstractmethod
    async def store(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        """Store a memory item."""
        ...

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        max_results: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Retrieve relevant memories."""
        ...

    @abstractmethod
    async def delete(self, item_id: str) -> bool:
        """Delete a memory item."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """Get total count of memories."""
        ...


# =============================================================================
# In-Memory Long-term Store
# =============================================================================


class InMemoryStore(LongTermMemoryStore):
    """
    Simple in-memory implementation of long-term memory.

    For production, replace with vector database (ChromaDB, Pinecone, etc.)
    """

    def __init__(
        self,
        memory_type: MemoryType,
        max_items: int = 10000,
    ):
        self.memory_type = memory_type
        self.max_items = max_items

        self._items: dict[str, MemoryItem] = {}
        self._lock = asyncio.Lock()

    async def store(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        """Store a memory item."""
        item_id = hashlib.sha256(f"{content}{time.time()}".encode()).hexdigest()[:16]

        item = MemoryItem(
            id=item_id,
            content=content,
            memory_type=self.memory_type,
            embedding=embedding,
            metadata=metadata or {},
        )

        async with self._lock:
            # Evict if at capacity
            if len(self._items) >= self.max_items:
                await self._evict()

            self._items[item_id] = item

        return item_id

    async def retrieve(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        max_results: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Retrieve relevant memories using text similarity."""
        async with self._lock:
            candidates = list(self._items.values())

        # Apply metadata filter
        if filter_metadata:
            candidates = [
                item for item in candidates if self._matches_filter(item.metadata, filter_metadata)
            ]

        # Score by text similarity
        query_words = set(query.lower().split())
        scored = []

        for item in candidates:
            content_words = set(item.content.lower().split())
            if not query_words or not content_words:
                score = 0.0
            else:
                overlap = len(query_words & content_words)
                score = overlap / len(query_words)

            # Importance bonus
            importance_bonus = {
                MemoryImportance.CRITICAL: 0.3,
                MemoryImportance.HIGH: 0.2,
                MemoryImportance.MEDIUM: 0.1,
                MemoryImportance.LOW: 0.0,
            }
            score += importance_bonus.get(item.importance, 0)

            item.relevance = score
            scored.append((score, item))

        # Sort and return top results
        scored.sort(key=lambda x: -x[0])

        results = []
        for score, item in scored[:max_results]:
            if score > 0.1:  # Minimum threshold
                item.touch()
                results.append(item)

        return results

    async def delete(self, item_id: str) -> bool:
        """Delete a memory item."""
        async with self._lock:
            if item_id in self._items:
                del self._items[item_id]
                return True
        return False

    async def count(self) -> int:
        """Get total count."""
        return len(self._items)

    async def _evict(self) -> None:
        """Evict oldest, least accessed item."""
        if not self._items:
            return

        # Find least valuable item
        def value_score(item: MemoryItem) -> float:
            recency = 1.0 / (time.time() - item.accessed_at + 1)
            importance_mult = {
                MemoryImportance.CRITICAL: 100,
                MemoryImportance.HIGH: 10,
                MemoryImportance.MEDIUM: 1,
                MemoryImportance.LOW: 0.1,
            }
            return recency * importance_mult.get(item.importance, 1)

        worst = min(self._items.items(), key=lambda x: value_score(x[1]))
        del self._items[worst[0]]

    def _matches_filter(
        self,
        metadata: dict[str, Any],
        filter_: dict[str, Any],
    ) -> bool:
        """Check if metadata matches filter."""
        for key, value in filter_.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True


# =============================================================================
# Persistent Store (File-based)
# =============================================================================


class PersistentMemoryStore(LongTermMemoryStore):
    """
    File-based persistent memory store.

    Stores memories in JSON files for persistence across sessions.
    For production, use a proper database.
    """

    def __init__(
        self,
        memory_type: MemoryType,
        storage_path: Path,
        max_items: int = 10000,
    ):
        self.memory_type = memory_type
        self.storage_path = storage_path
        self.max_items = max_items

        self._in_memory = InMemoryStore(memory_type, max_items)
        self._loaded = False
        self._lock = asyncio.Lock()

    async def _ensure_loaded(self) -> None:
        """Ensure memories are loaded from disk."""
        if self._loaded:
            return

        async with self._lock:
            if self._loaded:
                return

            if self.storage_path.exists():
                try:
                    data = json.loads(self.storage_path.read_text())
                    for item_data in data.get("items", []):
                        item = MemoryItem.from_dict(item_data)
                        self._in_memory._items[item.id] = item
                    logger.info(
                        f"Loaded {len(self._in_memory._items)} memories from {self.storage_path}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load memories: {e}")

            self._loaded = True

    async def _save(self) -> None:
        """Save memories to disk."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "memory_type": self.memory_type.value,
                "items": [item.to_dict() for item in self._in_memory._items.values()],
            }
            self.storage_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save memories: {e}")

    async def store(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        """Store a memory item."""
        await self._ensure_loaded()
        item_id = await self._in_memory.store(content, metadata, embedding)
        await self._save()
        return item_id

    async def retrieve(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        max_results: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Retrieve relevant memories."""
        await self._ensure_loaded()
        return await self._in_memory.retrieve(query, query_embedding, max_results, filter_metadata)

    async def delete(self, item_id: str) -> bool:
        """Delete a memory item."""
        await self._ensure_loaded()
        result = await self._in_memory.delete(item_id)
        if result:
            await self._save()
        return result

    async def count(self) -> int:
        """Get total count."""
        await self._ensure_loaded()
        return await self._in_memory.count()


# =============================================================================
# Memory Manager
# =============================================================================


class MemoryManager:
    """
    Unified memory manager for all memory types.

    Coordinates:
    - Working memory (in-context)
    - Semantic memory (facts and knowledge)
    - Episodic memory (task experiences)
    - Procedural memory (learned skills)

    Usage:
        manager = MemoryManager()

        # Remember something
        await manager.remember(
            "User prefers detailed explanations",
            memory_type=MemoryType.SEMANTIC
        )

        # Recall relevant memories
        memories = await manager.recall(
            "How should I explain this concept?"
        )

        # Get working memory context
        context = manager.get_working_context()
    """

    def __init__(
        self,
        working_memory_capacity: int = 10,
        persist_path: Path | None = None,
    ):
        # Working memory
        self.working = WorkingMemory(capacity=working_memory_capacity)

        # Long-term stores
        if persist_path:
            self.semantic = PersistentMemoryStore(
                MemoryType.SEMANTIC,
                persist_path / "semantic.json",
            )
            self.episodic = PersistentMemoryStore(
                MemoryType.EPISODIC,
                persist_path / "episodic.json",
            )
            self.procedural = PersistentMemoryStore(
                MemoryType.PROCEDURAL,
                persist_path / "procedural.json",
            )
        else:
            self.semantic = InMemoryStore(MemoryType.SEMANTIC)
            self.episodic = InMemoryStore(MemoryType.EPISODIC)
            self.procedural = InMemoryStore(MemoryType.PROCEDURAL)

        self._stores = {
            MemoryType.SEMANTIC: self.semantic,
            MemoryType.EPISODIC: self.episodic,
            MemoryType.PROCEDURAL: self.procedural,
        }

    async def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        metadata: dict[str, Any] | None = None,
        also_working: bool = False,
    ) -> str:
        """
        Store a memory.

        Args:
            content: Memory content
            memory_type: Type of memory
            importance: Importance level
            metadata: Optional metadata
            also_working: Also add to working memory

        Returns:
            Memory item ID
        """
        if memory_type == MemoryType.WORKING:
            return self.working.add(
                content,
                importance=importance,
                metadata=metadata,
            )

        store = self._stores.get(memory_type)
        if not store:
            raise ValueError(f"Unknown memory type: {memory_type}")

        full_metadata = metadata or {}
        full_metadata["importance"] = importance.value

        item_id = await store.store(content, full_metadata)

        if also_working:
            self.working.add(content, importance=importance, metadata=metadata)

        logger.debug(f"Stored {memory_type.value} memory: {content[:50]}")
        return item_id

    async def recall(
        self,
        query: str,
        memory_types: list[MemoryType] | None = None,
        max_results: int = 5,
        include_working: bool = True,
        filter_metadata: dict[str, Any] | None = None,
    ) -> RecallResult:
        """
        Recall relevant memories.

        Args:
            query: Query to search for
            memory_types: Types to search (None = semantic + episodic)
            max_results: Maximum results
            include_working: Include working memory
            filter_metadata: Filter by metadata

        Returns:
            RecallResult with memories
        """
        start_time = time.time()

        memory_types = memory_types or [MemoryType.SEMANTIC, MemoryType.EPISODIC]
        all_results: list[MemoryItem] = []
        total_searched = 0

        # Search long-term stores
        for memory_type in memory_types:
            if memory_type == MemoryType.WORKING:
                continue

            store = self._stores.get(memory_type)
            if store:
                results = await store.retrieve(
                    query,
                    max_results=max_results,
                    filter_metadata=filter_metadata,
                )
                all_results.extend(results)
                total_searched += await store.count()

        # Include working memory
        if include_working:
            for item in self.working.get_all():
                # Simple relevance scoring
                query_words = set(query.lower().split())
                content_words = set(item.content.lower().split())
                if query_words & content_words:
                    item.relevance = len(query_words & content_words) / len(query_words)
                    all_results.append(item)
            total_searched += self.working.count

        # Sort by relevance and deduplicate
        all_results.sort(key=lambda x: -x.relevance)

        seen = set()
        unique_results = []
        for item in all_results:
            if item.content not in seen:
                seen.add(item.content)
                unique_results.append(item)

        retrieval_time = int((time.time() - start_time) * 1000)

        return RecallResult(
            memories=unique_results[:max_results],
            query=query,
            total_searched=total_searched,
            retrieval_time_ms=retrieval_time,
        )

    def add_to_working(
        self,
        content: str,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        pinned: bool = False,
    ) -> str:
        """Convenience method to add to working memory."""
        return self.working.add(content, importance=importance, pinned=pinned)

    def get_working_context(self, max_items: int = None) -> str:
        """Get working memory context for prompts."""
        return self.working.get_context(max_items)

    def clear_working(self, keep_pinned: bool = True) -> int:
        """Clear working memory."""
        return self.working.clear(keep_pinned=keep_pinned)

    async def forget(
        self,
        item_id: str,
        memory_type: MemoryType,
    ) -> bool:
        """
        Delete a specific memory.

        Args:
            item_id: Memory item ID
            memory_type: Type of memory

        Returns:
            True if deleted
        """
        if memory_type == MemoryType.WORKING:
            return self.working.remove(item_id)

        store = self._stores.get(memory_type)
        if store:
            return await store.delete(item_id)
        return False

    async def get_statistics(self) -> dict[str, Any]:
        """Get memory statistics."""
        return {
            "working": {
                "count": self.working.count,
                "capacity": self.working.capacity,
                "free_slots": self.working.free_slots,
            },
            "semantic": {
                "count": await self.semantic.count(),
            },
            "episodic": {
                "count": await self.episodic.count(),
            },
            "procedural": {
                "count": await self.procedural.count(),
            },
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def create_memory_manager(
    working_capacity: int = 10,
    persist_path: str | None = None,
) -> MemoryManager:
    """Create a memory manager with default settings."""
    path = Path(persist_path) if persist_path else None
    return MemoryManager(
        working_memory_capacity=working_capacity,
        persist_path=path,
    )


__all__ = [
    # Enums
    "MemoryType",
    "MemoryImportance",
    # Data models
    "MemoryItem",
    "RecallResult",
    # Working memory
    "WorkingMemory",
    # Long-term stores
    "LongTermMemoryStore",
    "InMemoryStore",
    "PersistentMemoryStore",
    # Manager
    "MemoryManager",
    # Convenience
    "create_memory_manager",
]
