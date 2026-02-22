# src/llmcore/context/prioritization.py
"""
Content Prioritization for Adaptive Context Synthesis.

Provides configurable scoring and ranking of context chunks to determine
which content should be included in the limited token budget.  The
prioritizer computes a composite score from multiple signals:

- **Priority** (0–100): Static importance tier from the context source.
- **Relevance** (0.0–1.0): Semantic similarity to the current task.
- **Recency** (0.0–1.0): Temporal freshness of the content.
- **Utility** (0.0–1.0): Estimated usefulness based on past interactions.

The weights for each signal are configurable, allowing tuning for
different use cases (e.g., emphasize recency for fast-changing tasks,
or emphasize relevance for research queries).

Example::

    from llmcore.context.prioritization import ContentPrioritizer, PriorityWeights

    prioritizer = ContentPrioritizer(
        weights=PriorityWeights(
            priority=0.4,
            relevance=0.3,
            recency=0.2,
            utility=0.1,
        ),
    )

    ranked = prioritizer.rank(chunks)
    # ranked[0] is the highest-scoring chunk

References:
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §12 (Adaptive Context Synthesis)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Sequence

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class PriorityWeights:
    """
    Weights for the composite scoring formula.

    All weights should sum to ~1.0 for interpretability, but this
    is not enforced — the prioritizer normalizes internally.

    Attributes:
        priority: Weight for static priority tier (default 0.40).
        relevance: Weight for task relevance score (default 0.30).
        recency: Weight for temporal freshness (default 0.20).
        utility: Weight for historical usefulness (default 0.10).
    """

    priority: float = 0.40
    relevance: float = 0.30
    recency: float = 0.20
    utility: float = 0.10

    @property
    def total(self) -> float:
        """Sum of all weights."""
        return self.priority + self.relevance + self.recency + self.utility


@dataclass
class ScoredChunk:
    """
    A context chunk with its computed composite score.

    Attributes:
        source: Name of the context source.
        content: The text content.
        tokens: Token count.
        priority: Raw priority value (0–100).
        relevance: Relevance score (0.0–1.0).
        recency: Recency score (0.0–1.0).
        utility: Utility score (0.0–1.0).
        composite_score: Final computed score.
        metadata: Original metadata dict.
    """

    source: str
    content: str
    tokens: int
    priority: int = 50
    relevance: float = 1.0
    recency: float = 1.0
    utility: float = 1.0
    composite_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Prioritizer
# =============================================================================


class ContentPrioritizer:
    """
    Scores and ranks context chunks using a configurable weighted formula.

    The composite score for each chunk is::

        score = w_p * (priority / 100)
              + w_r * relevance
              + w_c * recency
              + w_u * utility

    Where ``w_p``, ``w_r``, ``w_c``, ``w_u`` are the normalized weights.

    Args:
        weights: Scoring weights (see ``PriorityWeights``).
        min_score_threshold: Chunks below this score are excluded.
        boost_rules: Optional dict mapping source names to multiplicative
            boost factors (e.g., ``{"goals": 1.5}`` gives goals a 50%
            score boost).

    Example::

        prioritizer = ContentPrioritizer()
        ranked = prioritizer.rank(chunks)
        # Select top-N or use with token budget fitting
    """

    def __init__(
        self,
        weights: PriorityWeights | None = None,
        min_score_threshold: float = 0.0,
        boost_rules: Dict[str, float] | None = None,
    ) -> None:
        self.weights = weights or PriorityWeights()
        self.min_score_threshold = min_score_threshold
        self.boost_rules = boost_rules or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        *,
        priority: int = 50,
        relevance: float = 1.0,
        recency: float = 1.0,
        utility: float = 1.0,
        source: str = "",
    ) -> float:
        """
        Compute composite score for a single chunk.

        Args:
            priority: Static priority (0–100).
            relevance: Task relevance (0.0–1.0).
            recency: Temporal freshness (0.0–1.0).
            utility: Historical usefulness (0.0–1.0).
            source: Source name (for boost rules).

        Returns:
            Composite score (higher = more important).
        """
        w = self.weights
        total_weight = w.total
        if total_weight <= 0:
            total_weight = 1.0

        # Normalize priority to 0–1 scale
        norm_priority = max(0.0, min(priority / 100.0, 1.0))

        composite = (
            (w.priority / total_weight) * norm_priority
            + (w.relevance / total_weight) * max(0.0, min(relevance, 1.0))
            + (w.recency / total_weight) * max(0.0, min(recency, 1.0))
            + (w.utility / total_weight) * max(0.0, min(utility, 1.0))
        )

        # Apply source-specific boost
        boost = self.boost_rules.get(source, 1.0)
        composite *= boost

        return composite

    def rank(
        self,
        chunks: Sequence[Any],
        *,
        task_description: str | None = None,
    ) -> list[ScoredChunk]:
        """
        Score and rank a list of context chunks.

        Each chunk must have the following attributes (matching
        ``ContextChunk``): ``source``, ``content``, ``tokens``,
        ``priority``, ``relevance``, ``recency``.  Optionally
        ``metadata`` and ``utility``.

        Args:
            chunks: Sequence of ``ContextChunk`` or compatible objects.
            task_description: Current task text (reserved for future
                relevance re-scoring).

        Returns:
            List of ``ScoredChunk`` sorted by composite score
            (descending).
        """
        scored: list[ScoredChunk] = []

        for chunk in chunks:
            priority = getattr(chunk, "priority", 50)
            relevance = getattr(chunk, "relevance", 1.0)
            recency = getattr(chunk, "recency", 1.0)
            utility = getattr(chunk, "utility", 1.0)
            source = getattr(chunk, "source", "")
            metadata = getattr(chunk, "metadata", {})

            composite = self.score(
                priority=priority,
                relevance=relevance,
                recency=recency,
                utility=utility,
                source=source,
            )

            if composite >= self.min_score_threshold:
                scored.append(
                    ScoredChunk(
                        source=source,
                        content=getattr(chunk, "content", ""),
                        tokens=getattr(chunk, "tokens", 0),
                        priority=priority,
                        relevance=relevance,
                        recency=recency,
                        utility=utility,
                        composite_score=composite,
                        metadata=metadata,
                    )
                )

        # Sort by composite score (descending)
        scored.sort(key=lambda s: s.composite_score, reverse=True)
        return scored

    def select_within_budget(
        self,
        ranked_chunks: list[ScoredChunk],
        token_budget: int,
    ) -> list[ScoredChunk]:
        """
        Select the highest-scoring chunks that fit within a token budget.

        Greedily selects chunks in score order until the budget is
        exhausted.

        Args:
            ranked_chunks: Pre-ranked chunks (highest score first).
            token_budget: Maximum tokens to select.

        Returns:
            Selected chunks (still in score order).
        """
        selected: list[ScoredChunk] = []
        remaining = token_budget

        for chunk in ranked_chunks:
            if remaining <= 0:
                break

            if chunk.tokens <= remaining:
                selected.append(chunk)
                remaining -= chunk.tokens

        return selected

    def adjust_weights(self, **kwargs: float) -> None:
        """
        Dynamically adjust weights at runtime.

        Args:
            **kwargs: Weight names and new values.
                E.g., ``adjust_weights(relevance=0.5, recency=0.1)``
        """
        for key, value in kwargs.items():
            if hasattr(self.weights, key):
                setattr(self.weights, key, value)
                logger.debug("Weight '%s' adjusted to %.3f", key, value)
            else:
                logger.warning("Unknown weight: %s", key)
