# src/llmcore/agents/caching/__init__.py
"""
Semantic Caching Module for Agents.

Provides embedding-based response caching and adaptive plan caching
to reduce redundant LLM calls and improve response latency.

Research Foundation:
    - 31% of queries are semantically similar (can be served from cache)
    - Adaptive Plan Cache (APC): 50% cost reduction, 27% latency reduction
    - 96.6% optimal performance vs non-cached baseline

Usage:
    from llmcore.agents.caching import SemanticCache, PlanCache

    # Response caching
    cache = SemanticCache()
    hit = cache.get("What is the capital of France?")
    if not hit:
        response = llm.generate(query)
        cache.set(query, response)

    # Plan caching
    plan_cache = PlanCache()
    similar_plan = plan_cache.find_similar_plan(goal, embedding)
    if similar_plan and similar_plan.success_rate > 0.7:
        # Reuse cached plan
        ...
"""

from .semantic_cache import (
    CachedPlan,
    # Data models
    CacheEntry,
    CacheHit,
    CacheStats,
    # Protocols
    EmbeddingProvider,
    PlanCache,
    SemanticCache,
    SemanticCacheConfig,
    # Implementations
    SimpleEmbeddingProvider,
    # Utilities
    cosine_similarity,
    euclidean_distance,
)

__all__ = [
    # Data models
    "CacheEntry",
    "CacheHit",
    "CacheStats",
    "CachedPlan",
    # Protocols
    "EmbeddingProvider",
    # Implementations
    "SimpleEmbeddingProvider",
    "SemanticCacheConfig",
    "SemanticCache",
    "PlanCache",
    # Utilities
    "cosine_similarity",
    "euclidean_distance",
]
