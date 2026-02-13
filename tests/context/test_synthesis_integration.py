# tests/context/test_synthesis_integration.py
"""
Integration tests for ContextSynthesizer ↔ ContextCompressor/ContentPrioritizer wiring.

Covers:
- G8 wiring: Compressor delegation when utilization exceeds threshold
- G8 wiring: Prioritizer-based chunk ranking replaces simple score ordering
- Fallback: Compressor failure falls back to built-in truncation
- Fallback: Prioritizer failure falls back to score-based ordering
- Backward compatibility: None compressor/prioritizer → original behaviour
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmcore.context.compression import (
    CompressionResult,
    CompressionStrategy,
    ContextCompressor,
)
from llmcore.context.prioritization import (
    ContentPrioritizer,
    PriorityWeights,
)
from llmcore.context.synthesis import (
    ContextChunk,
    ContextSynthesizer,
    EstimateCounter,
    SynthesizedContext,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def counter():
    """Deterministic token counter (4 chars per token)."""
    return EstimateCounter(chars_per_token=4)


class MockContextSource:
    """Configurable mock context source for tests."""

    def __init__(
        self,
        content: str = "mock content",
        tokens: int = 10,
        relevance: float = 0.8,
        recency: float = 0.7,
    ) -> None:
        self.content = content
        self.tokens = tokens
        self.relevance = relevance
        self.recency = recency

    async def get_context(self, task=None, max_tokens=10_000):
        from llmcore.context.synthesis import ContextChunk

        return ContextChunk(
            source="mock",
            content=self.content,
            tokens=self.tokens,
            relevance=self.relevance,
            recency=self.recency,
        )


# =============================================================================
# G8: Compressor Integration Tests
# =============================================================================


class TestCompressorWiring:
    """Tests that ContextSynthesizer delegates to ContextCompressor."""

    @pytest.mark.asyncio
    async def test_compressor_called_when_over_threshold(self, counter):
        """Compressor.compress() is called when utilization > threshold."""
        # max_tokens=500, threshold=0.75 → trigger at 375 tokens.
        # Source provides 400 tokens of content → exceeds threshold.
        compressor = ContextCompressor(
            strategy="truncation",
            token_counter=counter,
        )
        synth = ContextSynthesizer(
            max_tokens=500,
            compression_threshold=0.75,
            token_counter=counter,
            compressor=compressor,
        )

        big_content = "A" * 1600  # 1600/4 = 400 tokens
        source = MockContextSource(content=big_content, tokens=400)
        synth.add_source("big", source, priority=100)

        result = await synth.synthesize()

        assert result.compression_applied is True
        assert result.total_tokens <= 500

    @pytest.mark.asyncio
    async def test_compressor_not_called_below_threshold(self, counter):
        """Compressor is NOT called when utilization is below threshold."""
        compressor = MagicMock()
        compressor.compress = AsyncMock()

        synth = ContextSynthesizer(
            max_tokens=1000,
            compression_threshold=0.75,
            token_counter=counter,
            compressor=compressor,
        )

        source = MockContextSource(content="short", tokens=2)
        synth.add_source("small", source, priority=50)

        result = await synth.synthesize()

        assert result.compression_applied is False
        compressor.compress.assert_not_called()

    @pytest.mark.asyncio
    async def test_compressor_failure_falls_back_to_truncation(self, counter):
        """When compressor raises, fall back to built-in truncation."""
        compressor = MagicMock()
        compressor.compress = AsyncMock(side_effect=RuntimeError("Cross-encoder OOM"))

        synth = ContextSynthesizer(
            max_tokens=500,
            compression_threshold=0.75,
            token_counter=counter,
            compressor=compressor,
        )

        big_content = "B" * 1600  # 400 tokens
        source = MockContextSource(content=big_content, tokens=400)
        synth.add_source("big", source, priority=100)

        result = await synth.synthesize()

        # Compression still applied (fallback truncation)
        assert result.compression_applied is True
        compressor.compress.assert_called_once()

    @pytest.mark.asyncio
    async def test_compressor_receives_correct_arguments(self, counter):
        """Verify the arguments passed to compressor.compress()."""
        compressor = MagicMock()
        compressor.compress = AsyncMock(
            return_value=CompressionResult(
                content="compressed",
                original_tokens=400,
                compressed_tokens=100,
                strategy_used="truncation",
            )
        )

        synth = ContextSynthesizer(
            max_tokens=500,
            compression_threshold=0.75,
            token_counter=counter,
            compressor=compressor,
        )

        big_content = "C" * 1600  # 400 tokens
        source = MockContextSource(content=big_content, tokens=400)
        synth.add_source("big", source, priority=100)

        await synth.synthesize()

        compressor.compress.assert_called_once()
        call_kwargs = compressor.compress.call_args
        assert call_kwargs.kwargs["target_tokens"] == 375  # 500 * 0.75
        assert call_kwargs.kwargs["current_tokens"] > 0
        assert isinstance(call_kwargs.kwargs["content"], str)

    @pytest.mark.asyncio
    async def test_no_compressor_uses_builtin_truncation(self, counter):
        """Without compressor, the original truncation fallback is used."""
        synth = ContextSynthesizer(
            max_tokens=500,
            compression_threshold=0.75,
            token_counter=counter,
            # No compressor
        )

        big_content = "D" * 1600  # 400 tokens
        source = MockContextSource(content=big_content, tokens=400)
        synth.add_source("big", source, priority=100)

        result = await synth.synthesize()
        assert result.compression_applied is True


# =============================================================================
# G8: Prioritizer Integration Tests
# =============================================================================


class TestPrioritizerWiring:
    """Tests that ContextSynthesizer uses ContentPrioritizer for ranking."""

    @pytest.mark.asyncio
    async def test_prioritizer_changes_chunk_order(self, counter):
        """Prioritizer can re-rank chunks differently than the built-in score."""
        # Boost "episodic" so it appears first despite lower built-in priority.
        prioritizer = ContentPrioritizer(
            boost_rules={"episodic": 5.0},
        )

        synth = ContextSynthesizer(
            max_tokens=10_000,
            token_counter=counter,
            prioritizer=prioritizer,
        )

        # Without prioritizer, "goals" (priority=100) would be first.
        goals_source = MockContextSource(
            content="Goals content here",
            tokens=5,
            relevance=0.9,
            recency=0.5,
        )
        episodic_source = MockContextSource(
            content="Episodic memory content",
            tokens=6,
            relevance=0.3,
            recency=0.2,
        )

        synth.add_source("goals", goals_source, priority=100)
        synth.add_source("episodic", episodic_source, priority=30)

        result = await synth.synthesize()

        # Both should be included
        assert "goals" in result.sources_included
        assert "episodic" in result.sources_included
        # With the 5x boost on episodic, it should come first in content.
        # The assembled content has sections separated by "---".
        sections = result.content.split("---")
        first_section = sections[0].strip()
        assert first_section.startswith("## EPISODIC")

    @pytest.mark.asyncio
    async def test_prioritizer_failure_falls_back_to_score(self, counter):
        """When prioritizer raises, fall back to ContextChunk.score ordering."""
        prioritizer = MagicMock()
        prioritizer.rank = MagicMock(side_effect=RuntimeError("Rank failure"))

        synth = ContextSynthesizer(
            max_tokens=10_000,
            token_counter=counter,
            prioritizer=prioritizer,
        )

        source = MockContextSource(content="fallback content", tokens=5)
        synth.add_source("test", source, priority=50)

        result = await synth.synthesize()

        # Should still work despite prioritizer failure
        assert "test" in result.sources_included
        assert "fallback content" in result.content

    @pytest.mark.asyncio
    async def test_no_prioritizer_uses_builtin_score(self, counter):
        """Without prioritizer, chunks are sorted by ContextChunk.score."""
        synth = ContextSynthesizer(
            max_tokens=10_000,
            token_counter=counter,
            # No prioritizer
        )

        high = MockContextSource(content="high priority", tokens=4, relevance=1.0, recency=1.0)
        low = MockContextSource(content="low priority", tokens=4, relevance=0.1, recency=0.1)

        synth.add_source("high", high, priority=100)
        synth.add_source("low", low, priority=10)

        result = await synth.synthesize()

        sections = result.content.split("---")
        assert sections[0].strip().startswith("## HIGH")

    @pytest.mark.asyncio
    async def test_prioritizer_with_custom_weights(self, counter):
        """Custom PriorityWeights change the ranking."""
        # Heavily weight recency: source with high recency wins.
        prioritizer = ContentPrioritizer(
            weights=PriorityWeights(
                priority=0.1,
                relevance=0.1,
                recency=0.7,
                utility=0.1,
            ),
        )

        synth = ContextSynthesizer(
            max_tokens=10_000,
            token_counter=counter,
            prioritizer=prioritizer,
        )

        old_important = MockContextSource(
            content="Old but important",
            tokens=5,
            relevance=1.0,
            recency=0.1,
        )
        fresh_trivial = MockContextSource(
            content="Fresh but trivial",
            tokens=5,
            relevance=0.1,
            recency=1.0,
        )

        synth.add_source("old", old_important, priority=100)
        synth.add_source("fresh", fresh_trivial, priority=10)

        result = await synth.synthesize()

        sections = result.content.split("---")
        # With recency weight=0.7, "fresh" should rank first despite
        # lower priority.
        assert sections[0].strip().startswith("## FRESH")


# =============================================================================
# G8: Combined Compressor + Prioritizer Tests
# =============================================================================


class TestCompressorAndPrioritizerCombined:
    """Tests with both compressor and prioritizer active."""

    @pytest.mark.asyncio
    async def test_both_active_without_conflict(self, counter):
        """Prioritizer ranks, then compressor compresses — no conflict."""
        prioritizer = ContentPrioritizer()
        compressor = ContextCompressor(
            strategy="truncation",
            token_counter=counter,
        )

        synth = ContextSynthesizer(
            max_tokens=500,
            compression_threshold=0.75,
            token_counter=counter,
            compressor=compressor,
            prioritizer=prioritizer,
        )

        big = MockContextSource(content="X" * 1600, tokens=400)  # 400 tokens
        synth.add_source("big", big, priority=80)

        result = await synth.synthesize()

        # Both should work: prioritizer ranked (one source = trivial),
        # compressor triggered because 400 > 500 * 0.75 = 375
        assert result.compression_applied is True
        assert result.total_tokens <= 500

    @pytest.mark.asyncio
    async def test_backward_compat_none_both(self, counter):
        """Passing neither compressor nor prioritizer = original behaviour."""
        synth = ContextSynthesizer(
            max_tokens=10_000,
            token_counter=counter,
        )

        assert synth.compressor is None
        assert synth.prioritizer is None

        source = MockContextSource(content="hello", tokens=2)
        synth.add_source("test", source, priority=50)

        result = await synth.synthesize()
        assert result.sources_included == ["test"]
