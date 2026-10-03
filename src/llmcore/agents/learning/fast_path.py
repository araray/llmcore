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
import difflib
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
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
    error: str | None = None

    @property
    def under_target(self) -> bool:
        """Check if execution was under 5 second target."""
        return self.duration_ms < 5000


# =============================================================================
# Response Templates (grimoire promptlets)
# =============================================================================

#: Promptlet id prefix for canned fast-path responses. The bundled pack ships
#: one promptlet per intent (greeting, greeting_morning, greeting_afternoon,
#: greeting_evening, thanks, goodbye, acknowledgment); user layers may add or
#: override intents without touching code.
FAST_PATH_PROMPTLET_PREFIX = "llmcore/fast_path/"


def _fast_path_promptlets(grimoire: Any) -> dict[str, str]:
    """Map intent key → canned response from ``llmcore/fast_path/*`` promptlets."""
    # The facade does not surface promptlet listing (grimoire 0.4.x); the
    # repo view (single-root or layered composite) does.
    source = grimoire if hasattr(grimoire, "list_promptlets") else grimoire._repo
    prefix = FAST_PATH_PROMPTLET_PREFIX
    return {
        p.id[len(prefix):]: str(p.content).strip()
        for p in source.list_promptlets()
        if p.id.startswith(prefix)
    }


def get_template_response(
    intent: str,
    context: dict[str, Any] | None = None,
    *,
    grimoire: Any | None = None,
) -> str | None:
    """Get a canned response for an intent from grimoire promptlets.

    Canned fast-path responses live as ``llmcore/fast_path/<intent>``
    promptlets (bundled pack; user layers can override). Matching keeps the
    historical semantics over the promptlet id tails: exact intent match
    first, then partial containment either way.

    Args:
        intent: Classified intent (e.g. ``"greeting"``).
        context: Unused; kept for call-site compatibility.
        grimoire: Grimoire facade to read promptlets from. When None, the
            bundled-only registry's instance is used.

    Returns:
        The canned response text, or None when no promptlet matches.
    """
    del context
    if grimoire is None:
        from llmcore.grimoire_runtime import bundled_prompt_registry

        grimoire = bundled_prompt_registry().grimoire

    try:
        templates = _fast_path_promptlets(grimoire)
    except Exception as exc:
        # Canned responses are an optimization: an unusable promptlet source
        # (e.g. a non-grimoire object) means "no template" — the caller then
        # takes the direct LLM path, which renders fail-loud.
        logger.debug("Fast-path promptlet lookup unavailable: %s", exc)
        return None
    intent_lower = intent.lower()

    # Check for direct match
    if intent_lower in templates:
        return templates[intent_lower]

    # Check for partial match (over promptlet id tails)
    for key, template in sorted(templates.items()):
        if key in intent_lower or intent_lower in key:
            return template

    return None


# =============================================================================
# Response Cache
# =============================================================================


#: Openings that make a prompt a *pointer into conversation state* rather
#: than a self-contained question. Measured on 6,694 real harness prompts:
#: the most repeated prompt was "continue... ensure you commit often" (78
#: occurrences), and of all repeated prompts 92% recurred inside a single
#: session while 63% produced outputs differing by more than 2x. The same
#: text, a different correct answer, because the state moved -- so these can
#: never be served from cache.
STATE_POINTER_OPENINGS = frozenset({
    "continue", "continues", "go", "proceed", "resume", "next", "again",
    "more", "keep", "carry",
})

#: Whole prompts that only acknowledge and carry no request of their own.
ACKNOWLEDGEMENTS = frozenset({
    "ok", "okay", "k", "yes", "y", "yep", "yeah", "sure", "no", "n", "nope",
    "thanks", "thank you", "ty", "great", "good", "nice", "perfect", "done",
    "fine", "cool", "right", "correct", "exactly", "got it", "understood",
})


def is_cacheable(query: str) -> bool:
    """False when a prompt's answer depends on conversation state.

    A response cache keyed on prompt text is only sound for self-contained
    questions. "continue" is not a question; it is a reference to whatever
    was happening, and the right answer changes every time. Caching it
    returns a stale answer with full confidence, which is worse than a
    cache miss by a wide margin.
    """
    text = re.sub(r"[^\w\s]", " ", (query or "").lower())
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return False
    if text in ACKNOWLEDGEMENTS:
        return False
    first = text.split(" ", 1)[0]
    # The opening word decides: "continue, then run the tests" is still a
    # continuation, however much follows it.
    return first not in STATE_POINTER_OPENINGS


class ResponseCache:
    """
    In-memory cache for trivial, self-contained responses.

    Two properties this needs that it did not have:

    **Scope.** The key was the prompt text alone, so one conversation's
    answer to "continue" was served to another's. Entries are now namespaced
    by a caller-supplied scope (a session id), and an entry stored under one
    scope is invisible to another.

    **Order.** Lookup matched on Jaccard similarity over *word sets*, which
    ignores word order entirely -- "delete the old file" and "the file
    delete old" scored 1.0 and returned each other's response. Matching is
    now exact by default; a threshold below 1.0 enables fuzzy matching that
    respects sequence.

    It also refuses to store prompts whose answer depends on conversation
    state; see :func:`is_cacheable`.
    """

    def __init__(
        self,
        max_entries: int = 100,
        similarity_threshold: float = 1.0,
        ttl_seconds: float = 3600.0,
    ):
        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds

        self._cache: dict[tuple[str, str], dict[str, Any]] = {}

    def get(self, query: str, *, scope: str | None = None) -> str | None:
        """
        Get cached response for query within ``scope``.

        Args:
            query: User query
            scope: Conversation the query belongs to. Entries never cross
                scopes; omitting it uses a shared namespace, which is only
                safe for genuinely global, stateless prompts.

        Returns:
            Cached response or None
        """
        if not is_cacheable(query):
            return None

        key = self._key(query, scope)
        entry = self._cache.get(key)
        if entry is not None:
            if time.time() - entry["timestamp"] < self.ttl_seconds:
                return entry["response"]
            del self._cache[key]

        if self.similarity_threshold >= 1.0:
            return None

        # Fuzzy lookup, within this scope only and sequence-aware.
        scope_key = self._scope(scope)
        normalized = self._normalize(query)
        for (entry_scope, cached_query), entry in list(self._cache.items()):
            if entry_scope != scope_key:
                continue
            if time.time() - entry["timestamp"] >= self.ttl_seconds:
                del self._cache[(entry_scope, cached_query)]
                continue
            if self._similarity(normalized, cached_query) >= self.similarity_threshold:
                return entry["response"]

        return None

    def set(self, query: str, response: str, *, scope: str | None = None) -> None:
        """
        Cache a response.

        Args:
            query: User query
            response: Generated response
            scope: Conversation this belongs to; the entry is invisible
                outside it.

        A prompt whose answer depends on conversation state is not stored at
        all, so the mistake cannot be made later at lookup time either.
        """
        if not is_cacheable(query):
            logger.debug(
                "not caching %r: its answer depends on conversation state",
                query[:60],
            )
            return

        self._cache[self._key(query, scope)] = {
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

    @staticmethod
    def _scope(scope: str | None) -> str:
        return scope or ""

    def _key(self, query: str, scope: str | None) -> tuple[str, str]:
        return (self._scope(scope), self._normalize(query))

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        return text.lower().strip()

    def _similarity(self, a: str, b: str) -> float:
        """Sequence-aware similarity in [0, 1].

        Jaccard over word *sets* was wrong for a response cache: it scores
        "delete the old file" and "the file delete old" as identical, so one
        could be served the other's answer. ``SequenceMatcher`` respects
        order, so a reordering is no longer a perfect match.
        """
        if not a or not b:
            return 0.0
        return difflib.SequenceMatcher(None, a.split(), b.split()).ratio()


# =============================================================================
# Fast-Path Executor
# =============================================================================


class FastPathConfig:
    """Runtime configuration for fast-path execution.

    .. warning::
        This is **not** the user-facing config section. A separate
        :class:`llmcore.config.agents_config.FastPathConfig` holds the
        ``[agents.fast_path]`` settings a user actually writes, and the two
        classes share a name while disagreeing about field names
        (``cache_enabled`` vs ``use_cache``, ``templates_enabled`` vs
        ``use_templates``) and about which fields exist at all.

        That divergence is why the user-facing section was inert: the
        executor was built without a config, so it silently used these
        defaults. Use :meth:`from_agents_config` to translate, rather than
        passing one where the other is expected — the names are close
        enough that a mistake would not raise.
    """

    def __init__(
        self,
        max_response_time_ms: int = 5000,
        use_cache: bool = True,
        use_templates: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 500,
        fallback_on_timeout: bool = True,
        cache_max_entries: int = 100,
        cache_ttl_seconds: float = 3600.0,
    ):
        self.max_response_time_ms = max_response_time_ms
        self.use_cache = use_cache
        self.use_templates = use_templates
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fallback_on_timeout = fallback_on_timeout
        self.cache_max_entries = cache_max_entries
        self.cache_ttl_seconds = cache_ttl_seconds

    @classmethod
    def from_agents_config(cls, section: Any) -> FastPathConfig:
        """Translate the user-facing ``[agents.fast_path]`` section.

        The mapping is written out field by field on purpose. The two
        classes name the same concepts differently, so anything automatic
        (``model_dump()`` into ``**kwargs``) would drop exactly the
        renamed fields and leave them at their defaults — which is the bug
        this method exists to fix.

        Missing attributes fall back to the defaults above, so duck-typed
        and legacy config objects keep working.
        """
        def pick(name: str, default: Any) -> Any:
            """Read one field, falling back to the default on anything odd.

            Callers build managers with mocks and partially-formed config
            objects, so an attribute can exist and still not be a usable
            value. Coercion alone is not enough to catch that: a MagicMock
            implements ``__float__`` and happily becomes ``1.0``, which
            would silently give the cache a one-second TTL. So the value
            has to *be* the right kind of thing, not merely convertible to
            it.
            """
            value = getattr(section, name, None)
            if value is None:
                return default
            if isinstance(default, bool):
                return value if isinstance(value, bool) else default
            if isinstance(default, (int, float)):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    return default
                return type(default)(value)
            return value

        return cls(
            max_response_time_ms=pick("max_response_time_ms", 5000),
            # cache_enabled -> use_cache
            use_cache=pick("cache_enabled", True),
            # templates_enabled -> use_templates
            use_templates=pick("templates_enabled", True),
            temperature=pick("temperature", 0.7),
            max_tokens=pick("max_tokens", 500),
            fallback_on_timeout=pick("fallback_on_timeout", True),
            cache_max_entries=pick("cache_max_entries", 100),
            cache_ttl_seconds=pick("cache_ttl_seconds", 3600.0),
        )


class FastPathExecutor:
    """
    Fast-path executor for trivial goals.

    Bypasses the full cognitive cycle to achieve <5 second
    response times for simple interactions.

    Execution order:
    1. Check response cache
    2. Check response templates (grimoire promptlets)
    3. Direct LLM call (no tools)

    Args:
        llm_provider: LLM provider for direct calls
        config: Fast-path configuration
        prompt_registry: Prompt registry rendering the ``fast_path`` template
            and (via its grimoire) the canned-response promptlets. When None,
            the bundled-only adapter is self-built lazily.
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider | None = None,
        config: FastPathConfig | None = None,
        prompt_registry: Any | None = None,
    ):
        self.llm_provider = llm_provider
        self.config = config or FastPathConfig()
        self._prompt_registry = prompt_registry

        self._cache = (
            ResponseCache(
                max_entries=self.config.cache_max_entries,
                ttl_seconds=self.config.cache_ttl_seconds,
            )
            if self.config.use_cache
            else None
        )
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
        classification: GoalClassification | None = None,
        context: str | None = None,
        scope: str | None = None,
    ) -> FastPathResult:
        """
        Execute fast-path for a goal.

        Args:
            goal: The user's goal/message
            classification: Pre-computed classification (optional)
            context: Additional context (optional)
            scope: Conversation id. Cache entries never cross scopes, so
                omitting it shares one namespace across conversations --
                pass the session id.

        Returns:
            FastPathResult with response
        """
        start_time = time.time()
        self._stats["total_executions"] += 1

        try:
            # Strategy 1: Check cache
            if self._cache and self.config.use_cache:
                cached = self._cache.get(goal, scope=scope)
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

            # Strategy 2: Check templates (grimoire promptlets)
            if self.config.use_templates:
                intent = classification.intent.value if classification else "unknown"
                template_response = get_template_response(
                    intent,
                    {"goal": goal},
                    grimoire=getattr(self._ensure_prompt_registry(), "grimoire", None),
                )
                if template_response:
                    self._stats["template_hits"] += 1
                    duration_ms = int((time.time() - start_time) * 1000)

                    # Cache for future
                    if self._cache:
                        self._cache.set(goal, template_response, scope=scope)

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
                    self._cache.set(goal, response, scope=scope)

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

        except TimeoutError:
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

    def _ensure_prompt_registry(self) -> Any:
        """Return the injected registry, self-building the bundled adapter once."""
        if self._prompt_registry is None:
            from llmcore.grimoire_runtime import bundled_prompt_registry

            self._prompt_registry = bundled_prompt_registry()
        return self._prompt_registry

    async def _call_llm(
        self,
        goal: str,
        context: str | None = None,
    ) -> str:
        """Make direct LLM call.

        The SYSTEM + USER contract renders atomically from the ``fast_path``
        template (grimoire spell ``llmcore/learning/fast_path``) — fail-loud,
        no inline fallback. Caller-supplied context is prepended to the USER
        message (dynamic block, stays code-composed).
        """
        if not self.llm_provider:
            raise ValueError("No LLM provider configured")

        messages = self._ensure_prompt_registry().render_messages(
            "fast_path", {"goal": goal}
        )
        if context:
            for message in messages:
                if message.get("role") == "user":
                    message["content"] = f"{context}\n\n{message['content']}"
                    break

        # Set timeout
        timeout = self.config.max_response_time_ms / 1000.0

        try:
            response = await asyncio.wait_for(
                self.llm_provider.chat_async(
                    messages=messages,
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

        except TimeoutError:
            raise

    def get_statistics(self) -> dict[str, Any]:
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
    classification: GoalClassification,
    threshold_complexity: GoalComplexity | None = None,
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
    llm_provider: BaseLLMProvider,
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
    "FAST_PATH_PROMPTLET_PREFIX",
]
