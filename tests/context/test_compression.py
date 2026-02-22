# tests/context/test_compression.py
"""
Tests for Context Compression Module.

Covers:
- CompressionStrategy enum and selection
- CompressionResult dataclass and compression_ratio
- ContextCompressor with truncation, extractive, abstractive strategies
- _split_sections utility
- _extract_key_sentences heuristic scoring
- _split_into_sentences code-block handling
- Edge cases: empty content, already-fits, no LLM summarizer
- Fallback behaviors (abstractive → extractive when no LLM)
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from llmcore.context.compression import (
    CompressionResult,
    CompressionStrategy,
    ContextCompressor,
    _EstimateCounter,
    _extract_key_sentences,
    _split_into_sentences,
    _split_sections,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def counter():
    """Deterministic token counter (4 chars per token)."""
    return _EstimateCounter()


@pytest.fixture
def multi_section_content():
    """Multi-section context string with --- delimiters."""
    return (
        "## GOALS\n\nPrimary goal: Build the test suite.\n"
        "Sub-goal 1: Write unit tests.\n"
        "Sub-goal 2: Write integration tests."
        "\n\n---\n\n"
        "## RECENT CONTEXT\n\nUSER: Run pytest.\n"
        "ASSISTANT: All 849 tests passed.\n"
        "USER: Great, now continue.\n"
        "ASSISTANT: Implementing compression module."
        "\n\n---\n\n"
        "## SEMANTIC CONTEXT\n\nThe ContextSynthesizer orchestrates context assembly. "
        "It pulls from five tiers of context sources. "
        "Each source provides chunks ranked by priority, relevance, and recency. "
        "The synthesizer fits chunks within the token budget. "
        "Compression is triggered when utilization exceeds the configured threshold."
        "\n\n---\n\n"
        "## EPISODIC MEMORY\n\nPast lesson: Always run tests before committing. "
        "Past lesson: Code blocks have high information density. "
        "Past lesson: Preserve headers during compression."
    )


@pytest.fixture
def small_content():
    """Content that fits easily within any budget."""
    return "## GOALS\n\nBuild it."


# =============================================================================
# CompressionStrategy Enum
# =============================================================================


class TestCompressionStrategy:
    """Tests for CompressionStrategy enum."""

    def test_values(self):
        assert CompressionStrategy.TRUNCATION.value == "truncation"
        assert CompressionStrategy.EXTRACTIVE.value == "extractive"
        assert CompressionStrategy.ABSTRACTIVE.value == "abstractive"

    def test_from_string(self):
        assert CompressionStrategy("truncation") == CompressionStrategy.TRUNCATION
        assert CompressionStrategy("extractive") == CompressionStrategy.EXTRACTIVE


# =============================================================================
# CompressionResult
# =============================================================================


class TestCompressionResult:
    """Tests for CompressionResult dataclass."""

    def test_compression_ratio(self):
        result = CompressionResult(
            content="short",
            original_tokens=1000,
            compressed_tokens=250,
            strategy_used="truncation",
        )
        assert result.compression_ratio == pytest.approx(0.25)

    def test_compression_ratio_zero_original(self):
        result = CompressionResult(
            content="",
            original_tokens=0,
            compressed_tokens=0,
            strategy_used="none",
        )
        assert result.compression_ratio == 1.0

    def test_no_compression_ratio(self):
        result = CompressionResult(
            content="same",
            original_tokens=100,
            compressed_tokens=100,
            strategy_used="none",
        )
        assert result.compression_ratio == 1.0


# =============================================================================
# ContextCompressor — Already Fits
# =============================================================================


class TestCompressorNoOp:
    """When content already fits, no compression is performed."""

    @pytest.mark.asyncio
    async def test_content_within_budget(self, small_content, counter):
        compressor = ContextCompressor(
            strategy="truncation",
            token_counter=counter,
        )
        result = await compressor.compress(
            content=small_content,
            target_tokens=10_000,
        )
        assert result.strategy_used == "none"
        assert result.content == small_content
        assert result.compression_ratio == 1.0

    @pytest.mark.asyncio
    async def test_explicit_current_tokens(self, small_content, counter):
        compressor = ContextCompressor(token_counter=counter)
        result = await compressor.compress(
            content=small_content,
            target_tokens=10_000,
            current_tokens=5,  # explicitly provided
        )
        assert result.strategy_used == "none"
        assert result.original_tokens == 5


# =============================================================================
# Truncation Strategy
# =============================================================================


class TestTruncationStrategy:
    """Tests for the truncation compression strategy."""

    @pytest.mark.asyncio
    async def test_truncates_to_budget(self, multi_section_content, counter):
        """Sections beyond the budget are dropped."""
        compressor = ContextCompressor(
            strategy="truncation",
            token_counter=counter,
        )
        total = counter.count(multi_section_content)

        # Budget = half of total
        result = await compressor.compress(
            content=multi_section_content,
            target_tokens=total // 2,
            current_tokens=total,
        )
        assert result.strategy_used == "truncation"
        assert result.compressed_tokens <= total // 2 + 50  # small overhead tolerance
        assert result.compression_ratio < 1.0

    @pytest.mark.asyncio
    async def test_preserves_first_section(self, multi_section_content, counter):
        """The first section (goals) is always kept."""
        compressor = ContextCompressor(
            strategy="truncation",
            token_counter=counter,
        )
        total = counter.count(multi_section_content)

        result = await compressor.compress(
            content=multi_section_content,
            target_tokens=total // 4,
            current_tokens=total,
        )
        assert "## GOALS" in result.content

    @pytest.mark.asyncio
    async def test_adds_truncation_notice(self, multi_section_content, counter):
        """Truncated output includes a notice."""
        compressor = ContextCompressor(
            strategy="truncation",
            token_counter=counter,
        )
        total = counter.count(multi_section_content)

        result = await compressor.compress(
            content=multi_section_content,
            target_tokens=total // 4,
            current_tokens=total,
        )
        assert "truncated" in result.content.lower()


# =============================================================================
# Extractive Strategy
# =============================================================================


class TestExtractiveStrategy:
    """Tests for the extractive compression strategy."""

    @pytest.mark.asyncio
    async def test_preserves_top_n_sections(self, multi_section_content, counter):
        """Top-N sections are kept verbatim."""
        compressor = ContextCompressor(
            strategy="extractive",
            preserve_top_n=2,
            token_counter=counter,
        )
        total = counter.count(multi_section_content)

        result = await compressor.compress(
            content=multi_section_content,
            target_tokens=total // 2,
            current_tokens=total,
        )
        assert result.strategy_used == "extractive"
        assert "## GOALS" in result.content
        assert "## RECENT CONTEXT" in result.content
        assert result.sections_preserved == 2

    @pytest.mark.asyncio
    async def test_extracts_from_remaining(self, multi_section_content, counter):
        """Non-preserved sections get key-sentence extraction."""
        compressor = ContextCompressor(
            strategy="extractive",
            preserve_top_n=1,
            token_counter=counter,
        )
        total = counter.count(multi_section_content)

        result = await compressor.compress(
            content=multi_section_content,
            target_tokens=total // 2,
            current_tokens=total,
        )
        assert result.sections_summarized >= 2  # At least 2 sections compressed
        assert result.compressed_tokens < total

    @pytest.mark.asyncio
    async def test_preserve_zero(self, multi_section_content, counter):
        """preserve_top_n=0 compresses all sections."""
        compressor = ContextCompressor(
            strategy="extractive",
            preserve_top_n=0,
            token_counter=counter,
        )
        total = counter.count(multi_section_content)

        result = await compressor.compress(
            content=multi_section_content,
            target_tokens=total // 3,
            current_tokens=total,
        )
        assert result.sections_preserved == 0
        assert result.sections_summarized >= 1


# =============================================================================
# Abstractive Strategy
# =============================================================================


class TestAbstractiveStrategy:
    """Tests for the abstractive (LLM-based) compression strategy."""

    @pytest.mark.asyncio
    async def test_falls_back_without_llm(self, multi_section_content, counter):
        """Without an LLM summarizer, falls back to extractive."""
        compressor = ContextCompressor(
            strategy="abstractive",
            token_counter=counter,
            llm_summarizer=None,  # No LLM
        )
        total = counter.count(multi_section_content)

        result = await compressor.compress(
            content=multi_section_content,
            target_tokens=total // 2,
            current_tokens=total,
        )
        # Falls back to extractive
        assert result.strategy_used == "extractive"

    @pytest.mark.asyncio
    async def test_uses_llm_when_available(self, multi_section_content, counter):
        """With an LLM summarizer, uses abstractive strategy."""
        mock_llm = MagicMock()
        mock_llm.summarize = AsyncMock(return_value="Summary of context.")

        compressor = ContextCompressor(
            strategy="abstractive",
            preserve_top_n=1,
            token_counter=counter,
            llm_summarizer=mock_llm,
        )
        total = counter.count(multi_section_content)

        result = await compressor.compress(
            content=multi_section_content,
            target_tokens=total // 2,
            current_tokens=total,
        )
        assert result.strategy_used == "abstractive"
        assert "## GOALS" in result.content  # First section preserved
        assert "COMPRESSED CONTEXT" in result.content
        mock_llm.summarize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back(self, multi_section_content, counter):
        """When LLM summarization fails, falls back to extractive."""
        mock_llm = MagicMock()
        mock_llm.summarize = AsyncMock(side_effect=RuntimeError("API timeout"))

        compressor = ContextCompressor(
            strategy="abstractive",
            preserve_top_n=1,
            token_counter=counter,
            llm_summarizer=mock_llm,
        )
        total = counter.count(multi_section_content)

        result = await compressor.compress(
            content=multi_section_content,
            target_tokens=total // 2,
            current_tokens=total,
        )
        # Should not crash — falls back to extractive for remaining
        assert result.strategy_used == "abstractive"
        assert result.compressed_tokens > 0


# =============================================================================
# String Constructor
# =============================================================================


class TestStrategyFromString:
    """Test strategy construction from string."""

    def test_valid_strings(self):
        c1 = ContextCompressor(strategy="truncation")
        assert c1.strategy == CompressionStrategy.TRUNCATION

        c2 = ContextCompressor(strategy="extractive")
        assert c2.strategy == CompressionStrategy.EXTRACTIVE

    def test_invalid_string(self):
        with pytest.raises(ValueError):
            ContextCompressor(strategy="nonexistent")


# =============================================================================
# Utility Functions
# =============================================================================


class TestSplitSections:
    """Tests for _split_sections utility."""

    def test_basic_split(self):
        content = "Section A\n\n---\n\nSection B\n\n---\n\nSection C"
        sections = _split_sections(content)
        assert len(sections) == 3
        assert sections[0] == "Section A"
        assert sections[2] == "Section C"

    def test_no_delimiter(self):
        content = "Single section with no delimiters."
        sections = _split_sections(content)
        assert len(sections) == 1

    def test_empty_sections_filtered(self):
        content = "Section A\n\n---\n\n\n\n---\n\nSection C"
        sections = _split_sections(content)
        # Empty middle section filtered
        assert len(sections) == 2

    def test_empty_string(self):
        assert _split_sections("") == []


class TestSplitIntoSentences:
    """Tests for _split_into_sentences utility."""

    def test_basic_splitting(self):
        text = "First sentence. Second sentence! Third sentence?"
        sentences = _split_into_sentences(text)
        assert len(sentences) == 3

    def test_code_block_preserved(self):
        text = "Before.\n```python\ndef hello():\n    print('hi')\n```\nAfter."
        sentences = _split_into_sentences(text)
        # Code block should be one atomic unit
        code_blocks = [s for s in sentences if "def hello" in s]
        assert len(code_blocks) == 1
        assert "print" in code_blocks[0]  # Not split within block

    def test_unclosed_code_block(self):
        text = "Before.\n```python\ndef hello():\n    pass"
        sentences = _split_into_sentences(text)
        assert len(sentences) >= 1

    def test_empty_text(self):
        assert _split_into_sentences("") == []

    def test_blank_lines_ignored(self):
        text = "First.\n\n\nSecond."
        sentences = _split_into_sentences(text)
        assert len(sentences) == 2


class TestExtractKeySentences:
    """Tests for _extract_key_sentences heuristic."""

    def test_respects_budget(self, counter):
        """Extracted output should not exceed budget."""
        text = (
            "## Section\n\n"
            "This is the first important sentence with many useful words. "
            "This is the second sentence. "
            "This is the third sentence. "
            "This is the fourth and final sentence."
        )
        budget = counter.count(text) // 2

        result = _extract_key_sentences(text, budget, counter)
        assert counter.count(result) <= budget + 10  # small tolerance

    def test_preserves_headers(self, counter):
        """Headers are always included."""
        text = "## Important Header\n\nSome content here."
        result = _extract_key_sentences(text, 100, counter)
        assert "## Important Header" in result

    def test_code_blocks_score_higher(self, counter):
        """Code blocks get a scoring bonus."""
        text = (
            "This is regular text.\n"
            "```python\ndef important_function():\n    pass\n```\n"
            "This is more regular text."
        )
        # Budget enough for only ~half the content
        budget = counter.count(text) // 2
        result = _extract_key_sentences(text, budget, counter)
        # Code block should be preferentially included
        assert "def important_function" in result

    def test_position_bonus(self, counter):
        """Earlier sentences score higher than later ones."""
        text = (
            "First sentence is most important. "
            "Second sentence is also important. "
            "Third sentence has some info. "
            "Fourth sentence is least important."
        )
        # Very tight budget
        budget = counter.count("First sentence is most important.")
        result = _extract_key_sentences(text, budget + 5, counter)
        assert "First sentence" in result
