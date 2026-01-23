# src/llmcore/agents/learning/fast_path.py
"""
Fast-Path Execution for Trivial Goals.

Bypasses the full cognitive cycle for trivial goals (greetings,
simple questions) to achieve <5 second response times.

Problem:
    Current: "hello" → 21-day project plan → 392 seconds
    Target: "hello" → Direct response → <5 seconds

Solution:
    Pre-classify goals and route trivial ones directly to LLM
    without planning, tool use, or multiple iterations.

Usage:
    from llmcore.agents.learning import FastPathExecutor
    from llmcore.agents.cognitive import GoalClassifier, GoalComplexity

    classifier = GoalClassifier()
    fast_path = FastPathExecutor(llm_provider)

    classification = classifier.classify(goal)
    if classification.complexity == GoalComplexity.TRIVIAL:
        result = await fast_path.execute(goal)
        # ~1-3 seconds
    else:
        result = await full_cognitive_cycle.run(goal)
        # Full processing
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

try:
    from pydantic import BaseModel, Field

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object

    def Field(*args, **kwargs):
        return kwargs.get("default")


if TYPE_CHECKING:
    from llmcore.agents.cognitive.goal_classifier import GoalClassification, GoalComplexity
    from llmcore.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class FastPathStrategy(str, Enum):
    """Strategies for fast-path execution."""

    DIRECT = "direct"  # Single LLM call, no tools
    CACHED = "cached"  # Use cached response
    TEMPLATED = "templated"  # Use response template
    SINGLE_TOOL = "single_tool"  # Single tool call, then respond


@dataclass
class FastPathResult:
    """Result of fast-path execution."""

    success: bool
    response: str
    strategy: FastPathStrategy
    duration_ms: int
    from_cache: bool = False
    iterations: int = 1
    error: Optional[str] = None

    @property
    def under_target(self) -> bool:
        """Check if execution was under 5 second target."""
        return self.duration_ms < 5000


# =============================================================================
# Response Templates
# =============================================================================


# Templates for common trivial responses
RESPONSE_TEMPLATES: Dict[str, str] = {
    "greeting": "Hello! How can I help you today?",
    "greeting_morning": "Good morning! How can I assist you?",
    "greeting_afternoon": "Good afternoon! What can I do for you?",
    "greeting_evening": "Good evening! How may I help you?",
    "thanks": "You're welcome! Is there anything else I can help you with?",
    "goodbye": "Goodbye! Feel free to return if you need any assistance.",
    "acknowledgment": "I understand. Please let me know how I can help.",
}


def get_template_response(
    intent: str,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Get a template response if available."""
    intent_lower = intent.lower()

    # Check for direct match
    if intent_lower in RESPONSE_TEMPLATES:
        return RESPONSE_TEMPLATES[intent_lower]

    # Check for partial match
    for key, template in RESPONSE_TEMPLATES.items():
        if key in intent_lower or intent_lower in key:
            return template

    return None


# =============================================================================
# Fast-Path Prompts
# =============================================================================


FAST_PATH_SYSTEM_PROMPT = """You are a helpful AI assistant. Provide a direct,
concise response to the user's message. Do not over-explain or add unnecessary
context. Keep your response natural and friendly."""

FAST_PATH_USER_PROMPT = """User message: {goal}

Respond directly and concisely."""


# =============================================================================
# Response Cache
# =============================================================================


class ResponseCache:
    """
    Simple in-memory cache for trivial responses.

    Uses semantic similarity (basic) for cache lookup.
    For production, consider using embedding-based similarity.
    """

    def __init__(
        self,
        max_entries: int = 100,
        similarity_threshold: float = 0.8,
        ttl_seconds: float = 3600.0,
    ):
        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds

        self._cache: Dict[str, Dict[str, Any]] = {}

    def get(self, query: str) -> Optional[str]:
        """
        Get cached response for query.

        Args:
            query: User query

        Returns:
            Cached response or None
        """
        query_normalized = self._normalize(query)

        # Exact match
        if query_normalized in self._cache:
            entry = self._cache[query_normalized]
            if time.time() - entry["timestamp"] < self.ttl_seconds:
                return entry["response"]
            else:
                # Expired
                del self._cache[query_normalized]

        # Similarity search
        for cached_query, entry in list(self._cache.items()):
            if time.time() - entry["timestamp"] >= self.ttl_seconds:
                del self._cache[cached_query]
                continue

            similarity = self._similarity(query_normalized, cached_query)
            if similarity >= self.similarity_threshold:
                return entry["response"]

        return None

    def set(self, query: str, response: str) -> None:
        """
        Cache a response.

        Args:
            query: User query
            response: Generated response
        """
        query_normalized = self._normalize(query)

        self._cache[query_normalized] = {
            "response": response,
            "timestamp": time.time(),
        }

        # Prune if needed
        if len(self._cache) > self.max_entries:
            # Remove oldest entries
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1]["timestamp"],
            )
            for key, _ in sorted_entries[: len(sorted_entries) - self.max_entries]:
                del self._cache[key]

    def clear(self) -> None:
        """Clear the cache."""
        self._cache = {}

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        return text.lower().strip()

    def _similarity(self, a: str, b: str) -> float:
        """Calculate simple text similarity."""
        # Jaccard similarity on words
        words_a = set(a.split())
        words_b = set(b.split())

        if not words_a or not words_b:
            return 0.0

        intersection = len(words_a & words_b)
        union = len(words_a | words_b)

        return intersection / union if union > 0 else 0.0


# =============================================================================
# Fast-Path Executor
# =============================================================================


class FastPathConfig:
    """Configuration for fast-path execution."""

    def __init__(
        self,
        max_response_time_ms: int = 5000,
        use_cache: bool = True,
        use_templates: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 500,
        fallback_on_timeout: bool = True,
    ):
        self.max_response_time_ms = max_response_time_ms
        self.use_cache = use_cache
        self.use_templates = use_templates
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fallback_on_timeout = fallback_on_timeout


class FastPathExecutor:
    """
    Fast-path executor for trivial goals.

    Bypasses the full cognitive cycle to achieve <5 second
    response times for simple interactions.

    Execution order:
    1. Check response cache
    2. Check response templates
    3. Direct LLM call (no tools)

    Args:
        llm_provider: LLM provider for direct calls
        config: Fast-path configuration
    """

    def __init__(
        self,
        llm_provider: Optional["BaseLLMProvider"] = None,
        config: Optional[FastPathConfig] = None,
    ):
        self.llm_provider = llm_provider
        self.config = config or FastPathConfig()

        self._cache = ResponseCache() if self.config.use_cache else None
        self._stats = {
            "total_executions": 0,
            "cache_hits": 0,
            "template_hits": 0,
            "llm_calls": 0,
            "under_target": 0,
            "total_duration_ms": 0,
        }

    async def execute(
        self,
        goal: str,
        classification: Optional["GoalClassification"] = None,
        context: Optional[str] = None,
    ) -> FastPathResult:
        """
        Execute fast-path for a goal.

        Args:
            goal: The user's goal/message
            classification: Pre-computed classification (optional)
            context: Additional context (optional)

        Returns:
            FastPathResult with response
        """
        start_time = time.time()
        self._stats["total_executions"] += 1

        try:
            # Strategy 1: Check cache
            if self._cache and self.config.use_cache:
                cached = self._cache.get(goal)
                if cached:
                    self._stats["cache_hits"] += 1
                    duration_ms = int((time.time() - start_time) * 1000)
                    return FastPathResult(
                        success=True,
                        response=cached,
                        strategy=FastPathStrategy.CACHED,
                        duration_ms=duration_ms,
                        from_cache=True,
                    )

            # Strategy 2: Check templates
            if self.config.use_templates:
                intent = classification.intent.value if classification else "unknown"
                template_response = get_template_response(intent, {"goal": goal})
                if template_response:
                    self._stats["template_hits"] += 1
                    duration_ms = int((time.time() - start_time) * 1000)

                    # Cache for future
                    if self._cache:
                        self._cache.set(goal, template_response)

                    return FastPathResult(
                        success=True,
                        response=template_response,
                        strategy=FastPathStrategy.TEMPLATED,
                        duration_ms=duration_ms,
                    )

            # Strategy 3: Direct LLM call
            if self.llm_provider:
                self._stats["llm_calls"] += 1
                response = await self._call_llm(goal, context)
                duration_ms = int((time.time() - start_time) * 1000)

                # Cache for future
                if self._cache:
                    self._cache.set(goal, response)

                result = FastPathResult(
                    success=True,
                    response=response,
                    strategy=FastPathStrategy.DIRECT,
                    duration_ms=duration_ms,
                )

                if result.under_target:
                    self._stats["under_target"] += 1

                self._stats["total_duration_ms"] += duration_ms
                return result

            # No LLM provider - use fallback
            duration_ms = int((time.time() - start_time) * 1000)
            return FastPathResult(
                success=False,
                response="I'm unable to process your request right now.",
                strategy=FastPathStrategy.DIRECT,
                duration_ms=duration_ms,
                error="No LLM provider available",
            )

        except asyncio.TimeoutError:
            duration_ms = int((time.time() - start_time) * 1000)

            if self.config.fallback_on_timeout:
                return FastPathResult(
                    success=True,
                    response="I apologize, but I'm taking longer than expected. Could you please try again?",
                    strategy=FastPathStrategy.TEMPLATED,
                    duration_ms=duration_ms,
                    error="Timeout",
                )

            return FastPathResult(
                success=False,
                response="",
                strategy=FastPathStrategy.DIRECT,
                duration_ms=duration_ms,
                error="Timeout",
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.exception(f"Fast-path execution error: {e}")

            return FastPathResult(
                success=False,
                response="",
                strategy=FastPathStrategy.DIRECT,
                duration_ms=duration_ms,
                error=str(e),
            )

    async def _call_llm(
        self,
        goal: str,
        context: Optional[str] = None,
    ) -> str:
        """Make direct LLM call."""
        if not self.llm_provider:
            raise ValueError("No LLM provider configured")

        # Build prompt
        user_content = FAST_PATH_USER_PROMPT.format(goal=goal)
        if context:
            user_content = f"{context}\n\n{user_content}"

        # Set timeout
        timeout = self.config.max_response_time_ms / 1000.0

        try:
            response = await asyncio.wait_for(
                self.llm_provider.chat_async(
                    messages=[
                        {"role": "system", "content": FAST_PATH_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                ),
                timeout=timeout,
            )

            # Extract content
            if hasattr(response, "content"):
                return response.content
            elif isinstance(response, dict):
                return response.get("content", str(response))
            else:
                return str(response)

        except asyncio.TimeoutError:
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        total = self._stats["total_executions"]

        return {
            "total_executions": total,
            "cache_hit_rate": self._stats["cache_hits"] / total if total > 0 else 0.0,
            "template_hit_rate": self._stats["template_hits"] / total if total > 0 else 0.0,
            "llm_call_rate": self._stats["llm_calls"] / total if total > 0 else 0.0,
            "target_success_rate": self._stats["under_target"] / total if total > 0 else 0.0,
            "avg_duration_ms": self._stats["total_duration_ms"] / total if total > 0 else 0,
        }

    def clear_cache(self) -> None:
        """Clear response cache."""
        if self._cache:
            self._cache.clear()


# =============================================================================
# Convenience Functions
# =============================================================================


def should_use_fast_path(
    classification: "GoalClassification",
    threshold_complexity: Optional["GoalComplexity"] = None,
) -> bool:
    """
    Determine if fast-path should be used.

    Args:
        classification: Goal classification
        threshold_complexity: Maximum complexity for fast-path
            (defaults to TRIVIAL)

    Returns:
        True if fast-path is appropriate
    """
    from llmcore.agents.cognitive.goal_classifier import GoalComplexity

    threshold = threshold_complexity or GoalComplexity.TRIVIAL

    complexity_order = {
        GoalComplexity.TRIVIAL: 0,
        GoalComplexity.SIMPLE: 1,
        GoalComplexity.MODERATE: 2,
        GoalComplexity.COMPLEX: 3,
        GoalComplexity.AMBIGUOUS: 4,
    }

    current_level = complexity_order.get(classification.complexity, 99)
    threshold_level = complexity_order.get(threshold, 0)

    return current_level <= threshold_level


async def execute_fast_path(
    goal: str,
    llm_provider: "BaseLLMProvider",
    timeout_ms: int = 5000,
) -> FastPathResult:
    """
    Convenience function for fast-path execution.

    Args:
        goal: User's goal
        llm_provider: LLM provider
        timeout_ms: Maximum execution time

    Returns:
        FastPathResult
    """
    config = FastPathConfig(max_response_time_ms=timeout_ms)
    executor = FastPathExecutor(llm_provider=llm_provider, config=config)
    return await executor.execute(goal)


__all__ = [
    # Enums
    "FastPathStrategy",
    # Data models
    "FastPathResult",
    # Cache
    "ResponseCache",
    # Config
    "FastPathConfig",
    # Executor
    "FastPathExecutor",
    # Convenience
    "should_use_fast_path",
    "execute_fast_path",
    "get_template_response",
]
