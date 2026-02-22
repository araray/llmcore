# tests/context/test_prioritization.py
"""
Tests for Content Prioritization Module.

Covers:
- PriorityWeights: defaults, custom values, total property
- ScoredChunk: creation, defaults
- ContentPrioritizer.score(): composite formula, normalization, boost rules
- ContentPrioritizer.rank(): ordering, threshold filtering, duck-typing
- ContentPrioritizer.select_within_budget(): greedy token fitting
- ContentPrioritizer.adjust_weights(): runtime weight modification
- Edge cases: zero weights, out-of-range inputs, empty chunks
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from llmcore.context.prioritization import (
    ContentPrioritizer,
    PriorityWeights,
    ScoredChunk,
)

# =============================================================================
# Fixtures
# =============================================================================


@dataclass
class FakeChunk:
    """Duck-typed context chunk for testing."""

    source: str = "test"
    content: str = "Sample content."
    tokens: int = 100
    priority: int = 50
    relevance: float = 0.8
    recency: float = 0.7
    utility: float = 0.5
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@pytest.fixture
def default_weights():
    return PriorityWeights()


@pytest.fixture
def default_prioritizer():
    return ContentPrioritizer()


@pytest.fixture
def sample_chunks():
    """Diverse chunks for ranking tests."""
    return [
        FakeChunk(
            source="goals",
            content="Primary goal.",
            tokens=50,
            priority=100,
            relevance=1.0,
            recency=1.0,
            utility=0.9,
        ),
        FakeChunk(
            source="recent",
            content="Recent history.",
            tokens=80,
            priority=80,
            relevance=0.9,
            recency=1.0,
            utility=0.8,
        ),
        FakeChunk(
            source="semantic",
            content="RAG result.",
            tokens=120,
            priority=60,
            relevance=0.7,
            recency=0.5,
            utility=0.6,
        ),
        FakeChunk(
            source="episodic",
            content="Past lesson.",
            tokens=60,
            priority=40,
            relevance=0.3,
            recency=0.2,
            utility=0.4,
        ),
    ]


# =============================================================================
# PriorityWeights Tests
# =============================================================================


class TestPriorityWeights:
    """Tests for PriorityWeights dataclass."""

    def test_defaults(self):
        w = PriorityWeights()
        assert w.priority == 0.40
        assert w.relevance == 0.30
        assert w.recency == 0.20
        assert w.utility == 0.10

    def test_total(self):
        w = PriorityWeights()
        assert w.total == pytest.approx(1.0)

    def test_custom(self):
        w = PriorityWeights(priority=0.5, relevance=0.3, recency=0.1, utility=0.1)
        assert w.total == pytest.approx(1.0)

    def test_non_normalized(self):
        """Weights don't need to sum to 1.0 — prioritizer normalizes."""
        w = PriorityWeights(priority=2.0, relevance=3.0, recency=1.0, utility=1.0)
        assert w.total == pytest.approx(7.0)


# =============================================================================
# ScoredChunk Tests
# =============================================================================


class TestScoredChunk:
    """Tests for ScoredChunk dataclass."""

    def test_default_creation(self):
        chunk = ScoredChunk(source="goals", content="Test.", tokens=100)
        assert chunk.priority == 50
        assert chunk.relevance == 1.0
        assert chunk.composite_score == 0.0
        assert chunk.metadata == {}

    def test_custom_creation(self):
        chunk = ScoredChunk(
            source="semantic",
            content="RAG result.",
            tokens=200,
            priority=70,
            relevance=0.9,
            recency=0.5,
            utility=0.3,
            composite_score=0.72,
            metadata={"score": 0.95},
        )
        assert chunk.composite_score == 0.72
        assert chunk.metadata["score"] == 0.95


# =============================================================================
# ContentPrioritizer.score() Tests
# =============================================================================


class TestPrioritizerScore:
    """Tests for the composite scoring formula."""

    def test_perfect_scores(self, default_prioritizer):
        """All signals at max → score ≈ 1.0."""
        score = default_prioritizer.score(
            priority=100,
            relevance=1.0,
            recency=1.0,
            utility=1.0,
        )
        assert score == pytest.approx(1.0, abs=0.01)

    def test_zero_scores(self, default_prioritizer):
        """All signals at zero → score ≈ 0.0."""
        score = default_prioritizer.score(
            priority=0,
            relevance=0.0,
            recency=0.0,
            utility=0.0,
        )
        assert score == pytest.approx(0.0, abs=0.01)

    def test_priority_weight_dominance(self):
        """When priority weight is high, high-priority chunks score higher."""
        p = ContentPrioritizer(
            weights=PriorityWeights(
                priority=0.9,
                relevance=0.03,
                recency=0.03,
                utility=0.04,
            )
        )
        high_prio = p.score(priority=100, relevance=0.1, recency=0.1, utility=0.1)
        low_prio = p.score(priority=10, relevance=0.9, recency=0.9, utility=0.9)
        assert high_prio > low_prio

    def test_relevance_weight_dominance(self):
        """When relevance weight is high, relevant chunks win."""
        p = ContentPrioritizer(
            weights=PriorityWeights(
                priority=0.03,
                relevance=0.9,
                recency=0.03,
                utility=0.04,
            )
        )
        high_rel = p.score(priority=10, relevance=1.0, recency=0.1, utility=0.1)
        low_rel = p.score(priority=90, relevance=0.1, recency=0.9, utility=0.9)
        assert high_rel > low_rel

    def test_boost_rules(self):
        """Source-specific boost multiplies the composite score."""
        p = ContentPrioritizer(boost_rules={"goals": 1.5})
        boosted = p.score(priority=50, relevance=0.5, recency=0.5, utility=0.5, source="goals")
        normal = p.score(priority=50, relevance=0.5, recency=0.5, utility=0.5, source="other")
        assert boosted == pytest.approx(normal * 1.5)

    def test_normalization_with_non_unit_weights(self):
        """Weights are normalized so total=1 internally."""
        p = ContentPrioritizer(
            weights=PriorityWeights(
                priority=2.0,
                relevance=2.0,
                recency=2.0,
                utility=2.0,
            )
        )
        score = p.score(priority=100, relevance=1.0, recency=1.0, utility=1.0)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_clamping(self, default_prioritizer):
        """Out-of-range values are clamped."""
        score = default_prioritizer.score(
            priority=200,  # > 100
            relevance=1.5,  # > 1.0
            recency=-0.5,  # < 0.0
            utility=0.5,
        )
        # priority clamped to 100/100 = 1.0, relevance to 1.0, recency to 0.0
        assert score > 0.0
        assert score <= 2.0  # can exceed 1.0 with clamped-to-max values


# =============================================================================
# ContentPrioritizer.rank() Tests
# =============================================================================


class TestPrioritizerRank:
    """Tests for chunk ranking."""

    def test_rank_order(self, default_prioritizer, sample_chunks):
        """Higher-scoring chunks come first."""
        ranked = default_prioritizer.rank(sample_chunks)
        assert len(ranked) == 4
        # Goals (highest priority + relevance) should be first
        assert ranked[0].source == "goals"
        # Episodic (lowest everything) should be last
        assert ranked[-1].source == "episodic"
        # Scores are descending
        scores = [c.composite_score for c in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_min_score_threshold(self, sample_chunks):
        """Chunks below threshold are excluded."""
        p = ContentPrioritizer(min_score_threshold=0.7)
        ranked = p.rank(sample_chunks)
        for chunk in ranked:
            assert chunk.composite_score >= 0.7

    def test_empty_chunks(self, default_prioritizer):
        """Empty input returns empty output."""
        assert default_prioritizer.rank([]) == []

    def test_duck_typing(self, default_prioritizer):
        """Works with any object that has the right attributes."""
        mock = MagicMock()
        mock.source = "test"
        mock.content = "Data."
        mock.tokens = 50
        mock.priority = 70
        mock.relevance = 0.8
        mock.recency = 0.6
        mock.utility = 0.5
        mock.metadata = {}

        ranked = default_prioritizer.rank([mock])
        assert len(ranked) == 1
        assert ranked[0].source == "test"
        assert ranked[0].composite_score > 0

    def test_missing_attributes_use_defaults(self, default_prioritizer):
        """Chunks with missing attributes use safe defaults."""

        class MinimalChunk:
            source = "bare"
            content = "Bare minimum."
            tokens = 10

        ranked = default_prioritizer.rank([MinimalChunk()])
        assert len(ranked) == 1
        assert ranked[0].priority == 50  # default
        assert ranked[0].relevance == 1.0  # default


# =============================================================================
# ContentPrioritizer.select_within_budget() Tests
# =============================================================================


class TestSelectWithinBudget:
    """Tests for greedy budget fitting."""

    def test_basic_selection(self, default_prioritizer, sample_chunks):
        """Selects highest-scoring chunks that fit."""
        ranked = default_prioritizer.rank(sample_chunks)
        # Budget for ~2 chunks (50 + 80 = 130)
        selected = default_prioritizer.select_within_budget(ranked, token_budget=140)
        total_tokens = sum(c.tokens for c in selected)
        assert total_tokens <= 140
        assert len(selected) >= 1

    def test_exact_fit(self, default_prioritizer, sample_chunks):
        """All chunks selected when budget is generous."""
        ranked = default_prioritizer.rank(sample_chunks)
        total_tokens = sum(c.tokens for c in ranked)
        selected = default_prioritizer.select_within_budget(
            ranked,
            token_budget=total_tokens + 100,
        )
        assert len(selected) == len(ranked)

    def test_zero_budget(self, default_prioritizer, sample_chunks):
        """Zero budget returns nothing."""
        ranked = default_prioritizer.rank(sample_chunks)
        selected = default_prioritizer.select_within_budget(ranked, token_budget=0)
        assert selected == []

    def test_skips_oversized_chunks(self, default_prioritizer):
        """Large chunks that don't fit are skipped."""
        ranked = [
            ScoredChunk(source="big", content="Big.", tokens=1000, composite_score=0.9),
            ScoredChunk(source="small", content="Sm.", tokens=50, composite_score=0.8),
            ScoredChunk(source="medium", content="Med.", tokens=200, composite_score=0.7),
        ]
        selected = default_prioritizer.select_within_budget(ranked, token_budget=100)
        # Only "small" fits
        assert len(selected) == 1
        assert selected[0].source == "small"

    def test_greedy_order_preserved(self, default_prioritizer):
        """Selected chunks maintain score ordering."""
        ranked = [
            ScoredChunk(source="a", content=".", tokens=30, composite_score=0.9),
            ScoredChunk(source="b", content=".", tokens=30, composite_score=0.8),
            ScoredChunk(source="c", content=".", tokens=30, composite_score=0.7),
        ]
        selected = default_prioritizer.select_within_budget(ranked, token_budget=65)
        assert [c.source for c in selected] == ["a", "b"]


# =============================================================================
# ContentPrioritizer.adjust_weights() Tests
# =============================================================================


class TestAdjustWeights:
    """Tests for runtime weight adjustment."""

    def test_adjust_single(self, default_prioritizer):
        default_prioritizer.adjust_weights(relevance=0.5)
        assert default_prioritizer.weights.relevance == 0.5

    def test_adjust_multiple(self, default_prioritizer):
        default_prioritizer.adjust_weights(relevance=0.5, recency=0.1)
        assert default_prioritizer.weights.relevance == 0.5
        assert default_prioritizer.weights.recency == 0.1

    def test_unknown_weight_ignored(self, default_prioritizer):
        """Unknown weight names are silently ignored."""
        default_prioritizer.adjust_weights(nonexistent=0.99)
        # No crash, no change
        assert default_prioritizer.weights.priority == 0.40

    def test_score_changes_after_adjustment(self, default_prioritizer):
        """Adjusting weights changes subsequent scores."""
        score_before = default_prioritizer.score(
            priority=50,
            relevance=0.9,
            recency=0.1,
        )
        default_prioritizer.adjust_weights(recency=0.8, relevance=0.05)
        score_after = default_prioritizer.score(
            priority=50,
            relevance=0.9,
            recency=0.1,
        )
        assert score_before != pytest.approx(score_after, abs=0.01)


# =============================================================================
# Integration: Score → Rank → Select
# =============================================================================


class TestFullPipeline:
    """End-to-end pipeline: score, rank, budget-select."""

    def test_pipeline(self, sample_chunks):
        """Full pipeline produces usable, budget-constrained output."""
        prioritizer = ContentPrioritizer(
            weights=PriorityWeights(
                priority=0.4,
                relevance=0.3,
                recency=0.2,
                utility=0.1,
            ),
            boost_rules={"goals": 1.3},
        )

        ranked = prioritizer.rank(sample_chunks)
        assert len(ranked) == 4
        assert ranked[0].source == "goals"  # Boosted + highest base score

        # Select within 200-token budget
        selected = prioritizer.select_within_budget(ranked, token_budget=200)
        total_tokens = sum(c.tokens for c in selected)
        assert total_tokens <= 200
        assert len(selected) >= 1

        # Goals chunk (50 tokens) + recent (80) + episodic (60) = 190 ≤ 200
        selected_sources = {c.source for c in selected}
        assert "goals" in selected_sources
