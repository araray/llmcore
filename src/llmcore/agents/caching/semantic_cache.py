# src/llmcore/agents/caching/semantic_cache.py
"""
Semantic Cache for LLM Response Caching.

Caches LLM responses using semantic similarity for retrieval.
Research shows 31% of queries are semantically similar, enabling
significant cost savings ($80K+ quarterly reported).

Key Features:
    - Embedding-based similarity matching
    - TTL-based expiration
    - LRU eviction policy
    - Multiple embedding provider support
    - Plan caching for agent tasks

Research:
    - APC (Adaptive Plan Cache): 50% cost reduction, 27% latency reduction
    - 96.6% optimal performance with cached plans

Usage:
    from llmcore.agents.caching import SemanticCache

    cache = SemanticCache(
        embedding_provider=embedding_fn,
        similarity_threshold=0.92
    )

    # Check cache
    cached = await cache.get(query)
    if cached:
        return cached.response

    # Cache response
    response = await llm.chat(query)
    await cache.set(query, response)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
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
# Type Definitions
# =============================================================================


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        ...


@dataclass
class CacheEntry:
    """A cached response entry."""

    key: str
    query: str
    response: str
    embedding: list[float]
    metadata: dict[str, Any]
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    ttl_seconds: float = 3600.0

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return (time.time() - self.created_at) > self.ttl_seconds

    def touch(self) -> None:
        """Update access time and count."""
        self.accessed_at = time.time()
        self.access_count += 1


@dataclass
class CacheHit:
    """Result of a successful cache lookup."""

    response: str
    similarity: float
    key: str
    metadata: dict[str, Any]
    age_seconds: float


@dataclass
class CacheStats:
    """Cache statistics."""

    total_entries: int
    total_size_bytes: int
    hit_count: int
    miss_count: int
    eviction_count: int
    avg_similarity_on_hit: float

    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0


# =============================================================================
# Embedding Providers
# =============================================================================


class SimpleEmbeddingProvider:
    """
    Simple embedding provider using word overlap.

    For production, use OpenAI, Cohere, or local embeddings.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    async def embed(self, text: str) -> list[float]:
        """Generate simple embedding based on character hashing."""
        # Normalize text
        text = text.lower().strip()

        # Create embedding via character n-gram hashing
        embedding = [0.0] * self.dimension
        words = text.split()

        for word in words:
            for i in range(len(word)):
                ngram = word[i : i + 3]
                h = hash(ngram) % self.dimension
                embedding[h] += 1.0

        # Normalize
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        return [await self.embed(t) for t in texts]


# =============================================================================
# Similarity Functions
# =============================================================================


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0

    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = sum(x * x for x in a) ** 0.5
    magnitude_b = sum(x * x for x in b) ** 0.5

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def euclidean_distance(a: list[float], b: list[float]) -> float:
    """Compute Euclidean distance between two vectors."""
    if len(a) != len(b):
        return float("inf")

    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


# =============================================================================
# Semantic Cache Implementation
# =============================================================================


class SemanticCacheConfig:
    """Configuration for semantic cache."""

    def __init__(
        self,
        max_entries: int = 1000,
        similarity_threshold: float = 0.92,
        default_ttl_seconds: float = 3600.0,
        eviction_policy: str = "lru",  # lru, lfu, ttl
        auto_cleanup_interval: float = 300.0,
        min_query_length: int = 10,
        max_query_length: int = 10000,
    ):
        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        self.default_ttl_seconds = default_ttl_seconds
        self.eviction_policy = eviction_policy
        self.auto_cleanup_interval = auto_cleanup_interval
        self.min_query_length = min_query_length
        self.max_query_length = max_query_length


class SemanticCache:
    """
    Semantic cache for LLM responses.

    Uses embedding similarity to find cached responses for
    semantically similar queries.

    Args:
        embedding_provider: Provider for generating embeddings
        config: Cache configuration
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        config: SemanticCacheConfig | None = None,
    ):
        self.embedding_provider = embedding_provider or SimpleEmbeddingProvider()
        self.config = config or SemanticCacheConfig()

        self._entries: dict[str, CacheEntry] = {}
        self._embedding_index: list[tuple[str, list[float]]] = []  # (key, embedding)

        self._stats = CacheStats(
            total_entries=0,
            total_size_bytes=0,
            hit_count=0,
            miss_count=0,
            eviction_count=0,
            avg_similarity_on_hit=0.0,
        )

        self._similarity_sum = 0.0
        self._last_cleanup = time.time()
        self._lock = asyncio.Lock()

    async def get(
        self,
        query: str,
        metadata_filter: dict[str, Any] | None = None,
    ) -> CacheHit | None:
        """
        Look up cached response for a query.

        Args:
            query: Query to look up
            metadata_filter: Optional metadata filter

        Returns:
            CacheHit if found, None otherwise
        """
        # Validate query
        if not self._is_valid_query(query):
            return None

        # Periodic cleanup
        await self._maybe_cleanup()

        # Generate embedding
        try:
            query_embedding = await self.embedding_provider.embed(query)
        except Exception as e:
            logger.warning(f"Failed to embed query: {e}")
            return None

        # Search for similar entries
        async with self._lock:
            best_match: tuple[float, CacheEntry] | None = None

            for key, stored_embedding in self._embedding_index:
                entry = self._entries.get(key)
                if not entry or entry.is_expired:
                    continue

                # Apply metadata filter
                if metadata_filter:
                    if not self._matches_filter(entry.metadata, metadata_filter):
                        continue

                # Compute similarity
                similarity = cosine_similarity(query_embedding, stored_embedding)

                if similarity >= self.config.similarity_threshold:
                    if best_match is None or similarity > best_match[0]:
                        best_match = (similarity, entry)

            if best_match:
                similarity, entry = best_match
                entry.touch()

                # Update stats
                self._stats.hit_count += 1
                self._similarity_sum += similarity
                self._stats.avg_similarity_on_hit = self._similarity_sum / self._stats.hit_count

                logger.debug(f"Cache hit (similarity={similarity:.3f}): {query[:50]}")

                return CacheHit(
                    response=entry.response,
                    similarity=similarity,
                    key=entry.key,
                    metadata=entry.metadata,
                    age_seconds=time.time() - entry.created_at,
                )

        self._stats.miss_count += 1
        return None

    async def set(
        self,
        query: str,
        response: str,
        metadata: dict[str, Any] | None = None,
        ttl_seconds: float | None = None,
    ) -> str:
        """
        Cache a response.

        Args:
            query: Query string
            response: Response to cache
            metadata: Optional metadata
            ttl_seconds: TTL override

        Returns:
            Cache key
        """
        # Validate query
        if not self._is_valid_query(query):
            logger.warning("Query too short/long for caching")
            return ""

        # Generate key and embedding
        key = self._generate_key(query)

        try:
            embedding = await self.embedding_provider.embed(query)
        except Exception as e:
            logger.warning(f"Failed to embed query for caching: {e}")
            return ""

        # Create entry
        entry = CacheEntry(
            key=key,
            query=query,
            response=response,
            embedding=embedding,
            metadata=metadata or {},
            ttl_seconds=ttl_seconds or self.config.default_ttl_seconds,
        )

        async with self._lock:
            # Check if we need to evict
            if len(self._entries) >= self.config.max_entries:
                await self._evict()

            # Store entry
            self._entries[key] = entry
            self._embedding_index.append((key, embedding))

            # Update stats
            self._stats.total_entries = len(self._entries)
            self._stats.total_size_bytes += len(query) + len(response)

        logger.debug(f"Cached response: {query[:50]}")
        return key

    async def delete(self, key: str) -> bool:
        """Delete a cached entry."""
        async with self._lock:
            if key in self._entries:
                entry = self._entries.pop(key)
                self._embedding_index = [(k, e) for k, e in self._embedding_index if k != key]
                self._stats.total_entries = len(self._entries)
                self._stats.total_size_bytes -= len(entry.query) + len(entry.response)
                return True
        return False

    async def clear(self) -> int:
        """Clear all cached entries."""
        async with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._embedding_index.clear()
            self._stats = CacheStats(
                total_entries=0,
                total_size_bytes=0,
                hit_count=self._stats.hit_count,
                miss_count=self._stats.miss_count,
                eviction_count=self._stats.eviction_count,
                avg_similarity_on_hit=self._stats.avg_similarity_on_hit,
            )
            return count

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _is_valid_query(self, query: str) -> bool:
        """Check if query is valid for caching."""
        length = len(query)
        return self.config.min_query_length <= length <= self.config.max_query_length

    def _generate_key(self, query: str) -> str:
        """Generate cache key for query."""
        return hashlib.sha256(query.encode()).hexdigest()[:16]

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

    async def _maybe_cleanup(self) -> None:
        """Periodic cleanup of expired entries."""
        if time.time() - self._last_cleanup < self.config.auto_cleanup_interval:
            return

        async with self._lock:
            expired_keys = [key for key, entry in self._entries.items() if entry.is_expired]

            for key in expired_keys:
                entry = self._entries.pop(key)
                self._stats.eviction_count += 1
                self._stats.total_size_bytes -= len(entry.query) + len(entry.response)

            self._embedding_index = [
                (k, e) for k, e in self._embedding_index if k not in expired_keys
            ]

            self._stats.total_entries = len(self._entries)
            self._last_cleanup = time.time()

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired entries")

    async def _evict(self) -> None:
        """Evict entries based on policy."""
        if not self._entries:
            return

        policy = self.config.eviction_policy

        if policy == "lru":
            # Least Recently Used
            oldest = min(self._entries.items(), key=lambda x: x[1].accessed_at)
            key = oldest[0]
        elif policy == "lfu":
            # Least Frequently Used
            least_used = min(self._entries.items(), key=lambda x: x[1].access_count)
            key = least_used[0]
        else:  # ttl - oldest by creation time
            oldest = min(self._entries.items(), key=lambda x: x[1].created_at)
            key = oldest[0]

        entry = self._entries.pop(key)
        self._embedding_index = [(k, e) for k, e in self._embedding_index if k != key]

        self._stats.eviction_count += 1
        self._stats.total_size_bytes -= len(entry.query) + len(entry.response)

        logger.debug(f"Evicted entry: {entry.query[:50]}")


# =============================================================================
# Plan Cache
# =============================================================================


@dataclass
class CachedPlan:
    """A cached execution plan."""

    plan_id: str
    goal: str
    plan_steps: list[dict[str, Any]]
    embedding: list[float]
    success_count: int = 0
    failure_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5


class PlanCache:
    """
    Cache for execution plans (Adaptive Plan Cache - APC).

    Research shows 50% cost reduction, 27% latency reduction
    with 96.6% optimal performance.

    Usage:
        cache = PlanCache(embedding_provider)

        # Check for applicable plan
        plan = await cache.find_similar_plan(goal)
        if plan and plan.success_rate > 0.8:
            execute(plan.plan_steps)
        else:
            new_plan = await agent.plan(goal)
            await cache.store_plan(goal, new_plan)
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        similarity_threshold: float = 0.85,
        min_success_rate: float = 0.7,
        max_plans: int = 500,
    ):
        self.embedding_provider = embedding_provider or SimpleEmbeddingProvider()
        self.similarity_threshold = similarity_threshold
        self.min_success_rate = min_success_rate
        self.max_plans = max_plans

        self._plans: dict[str, CachedPlan] = {}
        self._embedding_index: list[tuple[str, list[float]]] = []
        self._lock = asyncio.Lock()

    async def find_similar_plan(
        self,
        goal: str,
        context: dict[str, Any] | None = None,
    ) -> CachedPlan | None:
        """
        Find a similar successful plan for a goal.

        Args:
            goal: Goal to find plan for
            context: Optional context for filtering

        Returns:
            CachedPlan if found and applicable
        """
        try:
            goal_embedding = await self.embedding_provider.embed(goal)
        except Exception as e:
            logger.warning(f"Failed to embed goal: {e}")
            return None

        async with self._lock:
            best_match: tuple[float, CachedPlan] | None = None

            for plan_id, stored_embedding in self._embedding_index:
                plan = self._plans.get(plan_id)
                if not plan:
                    continue

                # Check success rate
                if plan.success_rate < self.min_success_rate:
                    continue

                # Compute similarity
                similarity = cosine_similarity(goal_embedding, stored_embedding)

                if similarity >= self.similarity_threshold:
                    if best_match is None or similarity > best_match[0]:
                        best_match = (similarity, plan)

            if best_match:
                _, plan = best_match
                plan.last_used = time.time()
                logger.debug(f"Found cached plan for: {goal[:50]}")
                return plan

        return None

    async def store_plan(
        self,
        goal: str,
        plan_steps: list[dict[str, Any]],
    ) -> str:
        """
        Store a new plan.

        Args:
            goal: Goal the plan solves
            plan_steps: List of plan steps

        Returns:
            Plan ID
        """
        try:
            embedding = await self.embedding_provider.embed(goal)
        except Exception as e:
            logger.warning(f"Failed to embed goal for plan storage: {e}")
            return ""

        plan_id = hashlib.sha256(goal.encode()).hexdigest()[:16]

        plan = CachedPlan(
            plan_id=plan_id,
            goal=goal,
            plan_steps=plan_steps,
            embedding=embedding,
        )

        async with self._lock:
            # Evict if necessary
            if len(self._plans) >= self.max_plans:
                await self._evict_plan()

            self._plans[plan_id] = plan
            self._embedding_index.append((plan_id, embedding))

        logger.debug(f"Stored plan: {goal[:50]}")
        return plan_id

    async def record_outcome(
        self,
        plan_id: str,
        success: bool,
    ) -> None:
        """Record the outcome of using a cached plan."""
        async with self._lock:
            plan = self._plans.get(plan_id)
            if plan:
                if success:
                    plan.success_count += 1
                else:
                    plan.failure_count += 1

    async def _evict_plan(self) -> None:
        """Evict least effective plan."""
        if not self._plans:
            return

        # Find plan with lowest success rate * recency score
        def score(plan: CachedPlan) -> float:
            recency = 1.0 / (time.time() - plan.last_used + 1)
            return plan.success_rate * recency

        worst = min(self._plans.items(), key=lambda x: score(x[1]))
        plan_id = worst[0]

        self._plans.pop(plan_id)
        self._embedding_index = [(k, e) for k, e in self._embedding_index if k != plan_id]

    def get_statistics(self) -> dict[str, Any]:
        """Get plan cache statistics."""
        if not self._plans:
            return {
                "total_plans": 0,
                "avg_success_rate": 0.0,
            }

        success_rates = [p.success_rate for p in self._plans.values()]

        return {
            "total_plans": len(self._plans),
            "avg_success_rate": sum(success_rates) / len(success_rates),
            "total_uses": sum(p.success_count + p.failure_count for p in self._plans.values()),
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def create_semantic_cache(
    similarity_threshold: float = 0.92,
    max_entries: int = 1000,
    ttl_seconds: float = 3600.0,
) -> SemanticCache:
    """Create a semantic cache with default settings."""
    config = SemanticCacheConfig(
        similarity_threshold=similarity_threshold,
        max_entries=max_entries,
        default_ttl_seconds=ttl_seconds,
    )
    return SemanticCache(config=config)


def create_plan_cache(
    similarity_threshold: float = 0.85,
    max_plans: int = 500,
) -> PlanCache:
    """Create a plan cache with default settings."""
    return PlanCache(
        similarity_threshold=similarity_threshold,
        max_plans=max_plans,
    )


__all__ = [
    # Data models
    "CacheEntry",
    "CacheHit",
    "CacheStats",
    "CachedPlan",
    # Config
    "SemanticCacheConfig",
    # Providers
    "EmbeddingProvider",
    "SimpleEmbeddingProvider",
    # Similarity
    "cosine_similarity",
    "euclidean_distance",
    # Main classes
    "SemanticCache",
    "PlanCache",
    # Convenience
    "create_semantic_cache",
    "create_plan_cache",
]
