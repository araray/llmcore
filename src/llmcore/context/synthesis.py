# src/llmcore/context/synthesis.py
"""
Adaptive Context Synthesis for Autonomous Operation.

Reconstructs optimal context for each agent iteration by:
1. Gathering context from multiple sources in parallel
2. Scoring and ranking chunks by composite relevance
3. Fitting chunks into available token budget
4. Applying compression when threshold is exceeded
5. Returning coherent, prioritized context

The five context source tiers (highest → lowest priority):

    1. Core Context (100): system prompt, goals, escalations, constraints
    2. Recent Context (80): last N turns, tool executions, recent errors
    3. Semantic Context (60): RAG-retrieved chunks, relevant code
    4. Episodic Context (40): past completions, failure patterns, preferences
    5. Skill Context (varies): SKILL.md files, tool docs, API refs

Example::

    synthesizer = ContextSynthesizer(max_tokens=100_000)

    synthesizer.add_source("goals", goal_source, priority=100)
    synthesizer.add_source("recent", recent_source, priority=80)
    synthesizer.add_source("semantic", rag_source, priority=60)

    context = await synthesizer.synthesize(current_task=my_goal)
    response = await llm.complete(system=context.content, messages=[...])

References:
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §12 (Adaptive Context Synthesis)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols
# =============================================================================


class ContextSource(Protocol):
    """
    Protocol for context sources.

    Implementations retrieve domain-specific context (goals, history,
    RAG chunks, skills, etc.) and return it as a ``ContextChunk``.
    """

    async def get_context(
        self,
        task: Optional[Any] = None,
        max_tokens: int = 10_000,
    ) -> "ContextChunk":
        """
        Retrieve context relevant to the current task.

        Args:
            task: Current task or goal for relevance scoring.
                  Typically a ``Goal`` instance, but protocol accepts
                  ``Any`` to avoid circular imports.
            max_tokens: Maximum tokens to return.

        Returns:
            ContextChunk with content and metadata.
        """
        ...


class TokenCounter(Protocol):
    """Protocol for counting tokens in text."""

    def count(self, text: str) -> int:
        """
        Count tokens in the given text.

        Args:
            text: Input string.

        Returns:
            Number of tokens.
        """
        ...


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class ContextChunk:
    """
    A chunk of context from a single source.

    Attributes:
        source: Name identifying the source (e.g. ``"goals"``).
        content: The actual textual content.
        tokens: Token count of ``content``.
        priority: Base priority of this chunk (0–100, higher = more important).
        relevance: Relevance score to current task (0.0–1.0).
        recency: Recency score (0.0–1.0, 1.0 = most recent).
        metadata: Arbitrary metadata dict for debugging / tracing.
    """

    source: str
    content: str
    tokens: int
    priority: int = 50
    relevance: float = 1.0
    recency: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        """
        Composite score for prioritization.

        Formula::

            priority * (0.5 + 0.3 * relevance + 0.2 * recency)

        Higher score → more likely to be included in the final context.
        """
        return self.priority * (0.5 + 0.3 * self.relevance + 0.2 * self.recency)


@dataclass
class SynthesizedContext:
    """
    The final assembled context for an agent iteration.

    Attributes:
        content: The assembled context string.
        total_tokens: Total token count.
        max_tokens: Maximum allowed tokens (budget).
        sources_included: Names of sources that contributed.
        sources_truncated: Names of sources that were truncated.
        compression_applied: Whether compression was triggered.
        synthesis_time_ms: How long synthesis took (milliseconds).
    """

    content: str
    total_tokens: int
    max_tokens: int
    sources_included: List[str]
    sources_truncated: List[str] = field(default_factory=list)
    compression_applied: bool = False
    synthesis_time_ms: float = 0.0

    @property
    def utilization(self) -> float:
        """Context window utilization as a fraction (0.0–1.0)."""
        if self.max_tokens <= 0:
            return 0.0
        return self.total_tokens / self.max_tokens


# =============================================================================
# Token Counter Implementations
# =============================================================================


class TiktokenCounter:
    """Token counter using tiktoken (``cl100k_base`` encoding)."""

    def __init__(self) -> None:
        import tiktoken

        self._encoding = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        """Count tokens via tiktoken."""
        return len(self._encoding.encode(text))


class EstimateCounter:
    """
    Fallback token counter using character-based estimation.

    Uses ~4 characters per token as a rough heuristic.
    """

    def __init__(self, chars_per_token: int = 4) -> None:
        self._chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        """Estimate tokens from character count."""
        return max(1, len(text) // self._chars_per_token) if text else 0


def _make_default_counter() -> TokenCounter:
    """
    Create the best available token counter.

    Tries tiktoken first; falls back to estimation if unavailable.
    """
    try:
        return TiktokenCounter()
    except (ImportError, Exception):
        logger.warning("tiktoken not available — using character-based token estimation")
        return EstimateCounter()


# =============================================================================
# Context Synthesizer
# =============================================================================


class ContextSynthesizer:
    """
    Synthesizes optimal context from multiple registered sources.

    The synthesizer orchestrates context assembly by:

    1. Querying all registered sources in parallel.
    2. Scoring chunks via a composite formula (priority × relevance × recency).
    3. Fitting chunks into the token budget in score-descending order.
    4. Truncating oversized chunks when beneficial.
    5. Triggering compression when utilization exceeds the threshold.

    Example::

        synth = ContextSynthesizer(max_tokens=100_000)

        synth.add_source("goals", goal_source, priority=100)
        synth.add_source("recent", recent_source, priority=80)

        ctx = await synth.synthesize(current_task=my_goal)
        print(ctx.utilization, ctx.sources_included)
    """

    def __init__(
        self,
        max_tokens: int = 100_000,
        compression_threshold: float = 0.75,
        token_counter: Optional[TokenCounter] = None,
    ) -> None:
        """
        Initialize the synthesizer.

        Args:
            max_tokens: Maximum tokens for synthesized context.
            compression_threshold: Trigger compression when utilization
                exceeds this fraction of ``max_tokens``.
            token_counter: Token counting implementation.  Defaults to
                tiktoken (cl100k_base) with character-estimation fallback.
        """
        self.max_tokens = max_tokens
        self.compression_threshold = compression_threshold
        self.token_counter: TokenCounter = (
            token_counter if token_counter is not None else _make_default_counter()
        )
        # name → (source, priority)
        self._sources: Dict[str, Tuple[ContextSource, int]] = {}

    # ------------------------------------------------------------------
    # Source management
    # ------------------------------------------------------------------

    def add_source(
        self,
        name: str,
        source: ContextSource,
        priority: int = 50,
    ) -> None:
        """
        Register a context source.

        Args:
            name: Unique name for the source.
            source: Object implementing the ``ContextSource`` protocol.
            priority: Base priority (0–100, higher = more important).
        """
        self._sources[name] = (source, priority)
        logger.debug("Registered context source: %s (priority=%d)", name, priority)

    def remove_source(self, name: str) -> None:
        """
        Remove a previously registered context source.

        Args:
            name: Name of the source to remove.  No-op if not found.
        """
        self._sources.pop(name, None)

    def list_sources(self) -> Dict[str, int]:
        """
        Return registered source names and their priorities.

        Returns:
            Dict mapping source name → priority.
        """
        return {name: prio for name, (_, prio) in self._sources.items()}

    # ------------------------------------------------------------------
    # Core synthesis
    # ------------------------------------------------------------------

    async def synthesize(
        self,
        current_task: Optional[Any] = None,
        model: Optional[str] = None,
        include_sources: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
    ) -> SynthesizedContext:
        """
        Synthesize context from all registered sources.

        Args:
            current_task: Current task / goal for relevance scoring.
            model: Model name (reserved for future model-specific budgets).
            include_sources: Whitelist — only include these sources.
                ``None`` means include all.
            exclude_sources: Blacklist — exclude these sources.

        Returns:
            ``SynthesizedContext`` ready for use as a system prompt
            or context injection.
        """
        start = datetime.now(timezone.utc)

        # Determine which sources to query
        source_names = set(self._sources.keys())
        if include_sources is not None:
            source_names &= set(include_sources)
        if exclude_sources is not None:
            source_names -= set(exclude_sources)

        if not source_names:
            elapsed = _elapsed_ms(start)
            return SynthesizedContext(
                content="",
                total_tokens=0,
                max_tokens=self.max_tokens,
                sources_included=[],
                synthesis_time_ms=elapsed,
            )

        # Gather from all sources in parallel
        fetch_tasks = []
        for name in source_names:
            source, priority = self._sources[name]
            fetch_tasks.append(self._fetch_source(name, source, priority, current_task))

        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        # Filter out errors and empty chunks
        valid_chunks: List[ContextChunk] = []
        for result in results:
            if isinstance(result, BaseException):
                logger.warning("Context source failed: %s", result)
            elif result is not None and result.content.strip():
                valid_chunks.append(result)

        # Sort by composite score (highest first)
        valid_chunks.sort(key=lambda c: c.score, reverse=True)

        # Fit into budget
        selected, truncated_names = self._fit_to_budget(valid_chunks)

        # Assemble content
        content = self._assemble_content(selected)
        total_tokens = self.token_counter.count(content) if content else 0

        # Compress if over threshold
        compression_applied = False
        if (
            total_tokens > 0
            and self.max_tokens > 0
            and total_tokens > self.max_tokens * self.compression_threshold
        ):
            content = await self._compress(content, current_task)
            total_tokens = self.token_counter.count(content)
            compression_applied = True

        elapsed = _elapsed_ms(start)

        return SynthesizedContext(
            content=content,
            total_tokens=total_tokens,
            max_tokens=self.max_tokens,
            sources_included=[c.source for c in selected],
            sources_truncated=truncated_names,
            compression_applied=compression_applied,
            synthesis_time_ms=elapsed,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch_source(
        self,
        name: str,
        source: ContextSource,
        priority: int,
        task: Optional[Any],
    ) -> Optional[ContextChunk]:
        """Fetch context from a single source, applying base priority."""
        try:
            # Allocate per-source budget proportional to priority
            budget = max(
                1000,
                int(self.max_tokens * (priority / 100.0) * 0.5),
            )
            chunk = await source.get_context(task=task, max_tokens=budget)

            # Ensure the chunk carries the correct source name and priority
            if chunk is not None:
                chunk.source = name
                chunk.priority = priority
            return chunk
        except Exception as exc:
            logger.warning("Failed to fetch context from %s: %s", name, exc)
            return None

    def _fit_to_budget(
        self,
        chunks: List[ContextChunk],
    ) -> Tuple[List[ContextChunk], List[str]]:
        """
        Select and optionally truncate chunks to fit the token budget.

        Args:
            chunks: Pre-sorted list of ContextChunks (highest score first).

        Returns:
            Tuple of (selected chunks, names of truncated sources).
        """
        selected: List[ContextChunk] = []
        truncated: List[str] = []
        remaining = self.max_tokens

        for chunk in chunks:
            if remaining <= 0:
                break

            if chunk.tokens <= remaining:
                # Fits entirely
                selected.append(chunk)
                remaining -= chunk.tokens
            elif remaining >= 200:
                # Partially fits — truncate
                truncated_chunk = self._truncate_chunk(chunk, remaining)
                if truncated_chunk is not None:
                    selected.append(truncated_chunk)
                    remaining -= truncated_chunk.tokens
                    truncated.append(chunk.source)

        return selected, truncated

    def _truncate_chunk(
        self,
        chunk: ContextChunk,
        max_tokens: int,
    ) -> Optional[ContextChunk]:
        """
        Truncate a chunk to fit within ``max_tokens``.

        Uses binary search on character position to find the cut point.

        Returns:
            Truncated ContextChunk, or ``None`` if result is too short
            to be useful (< 100 tokens).
        """
        content = chunk.content
        tokens = self.token_counter.count(content)

        if tokens <= max_tokens:
            return chunk

        # Binary search for truncation point
        low, high = 0, len(content)
        while low < high:
            mid = (low + high + 1) // 2
            if self.token_counter.count(content[:mid]) <= max_tokens:
                low = mid
            else:
                high = mid - 1

        if self.token_counter.count(content[:low]) < 100:
            return None

        truncated_content = content[:low] + "\n[... truncated ...]"
        return ContextChunk(
            source=chunk.source,
            content=truncated_content,
            tokens=self.token_counter.count(truncated_content),
            priority=chunk.priority,
            relevance=chunk.relevance,
            recency=chunk.recency,
            metadata={**chunk.metadata, "truncated": True},
        )

    def _assemble_content(self, chunks: List[ContextChunk]) -> str:
        """Assemble selected chunks into the final context string."""
        sections: List[str] = []
        for chunk in chunks:
            text = chunk.content.strip()
            if text:
                sections.append(f"## {chunk.source.upper()}\n\n{text}")
        return "\n\n---\n\n".join(sections)

    async def _compress(
        self,
        content: str,
        task: Optional[Any],
    ) -> str:
        """
        Compress context to fit within the token budget.

        Current implementation: simple truncation.
        Future: LLM-based summarization of lower-priority sections.

        Args:
            content: Full context string.
            task: Current task (for relevance-aware compression).

        Returns:
            Compressed context string.
        """
        max_chars = self.max_tokens * 4  # conservative estimate
        if len(content) > max_chars:
            return content[:max_chars] + "\n\n[Context compressed to fit limits]"
        return content


# =============================================================================
# Utility
# =============================================================================


def _elapsed_ms(start: datetime) -> float:
    """Milliseconds elapsed since ``start``."""
    return (datetime.now(timezone.utc) - start).total_seconds() * 1000.0
