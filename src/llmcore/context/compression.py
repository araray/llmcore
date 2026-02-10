# src/llmcore/context/compression.py
"""
Context Compression for Adaptive Context Synthesis.

Provides strategies for reducing context size while preserving the
most important information.  Used by ``ContextSynthesizer`` when
utilization exceeds the configured threshold.

Three compression strategies are available:

1. **Truncation** (default, no LLM cost):
   Keeps the highest-priority content and truncates from the bottom.

2. **Extractive** (lightweight):
   Extracts key sentences from each section using heuristic scoring
   (position, keyword density, length).

3. **Abstractive** (LLM-based):
   Uses an LLM to summarize lower-priority sections while keeping
   high-priority sections verbatim.

Example::

    compressor = ContextCompressor(strategy="extractive")
    compressed = await compressor.compress(
        content=long_context,
        target_tokens=50_000,
        current_tokens=120_000,
    )

References:
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §12 (Adaptive Context Synthesis)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Protocols
# =============================================================================


class CompressionStrategy(Enum):
    """Available compression strategies."""

    TRUNCATION = "truncation"
    """Simple tail truncation — fastest, no quality loss for head content."""

    EXTRACTIVE = "extractive"
    """Heuristic sentence extraction — moderate quality, no LLM cost."""

    ABSTRACTIVE = "abstractive"
    """LLM-based summarization — best quality, costs tokens/money."""


class TokenCounter(Protocol):
    """Protocol for counting tokens in text."""

    def count(self, text: str) -> int: ...


class LLMSummarizer(Protocol):
    """Protocol for LLM-based summarization (used by abstractive strategy)."""

    async def summarize(self, text: str, max_tokens: int) -> str: ...


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class CompressionResult:
    """
    Result of a compression operation.

    Attributes:
        content: Compressed text.
        original_tokens: Token count before compression.
        compressed_tokens: Token count after compression.
        strategy_used: Which strategy was applied.
        sections_summarized: Number of sections that were summarized.
        sections_preserved: Number of sections kept verbatim.
    """

    content: str
    original_tokens: int
    compressed_tokens: int
    strategy_used: str
    sections_summarized: int = 0
    sections_preserved: int = 0

    @property
    def compression_ratio(self) -> float:
        """Ratio of compressed to original size (0.0–1.0)."""
        if self.original_tokens <= 0:
            return 1.0
        return self.compressed_tokens / self.original_tokens


# =============================================================================
# Compressor
# =============================================================================


class ContextCompressor:
    """
    Compresses context content to fit within a token budget.

    The compressor operates on section-delimited context strings
    (as produced by ``ContextSynthesizer._assemble_content``).  Sections
    are separated by ``---`` markers and start with ``## SOURCENAME``
    headers.

    The ``preserve_top_n`` parameter controls how many of the first
    (highest-priority) sections are kept verbatim regardless of
    compression strategy.

    Args:
        strategy: Compression strategy to use.
        preserve_top_n: Number of top-priority sections to keep
            verbatim (default 2: usually goals + recent context).
        token_counter: Optional token counter; falls back to
            character-based estimation.
        llm_summarizer: Required only for ``abstractive`` strategy.

    Example::

        compressor = ContextCompressor(
            strategy="extractive",
            preserve_top_n=2,
        )
        result = await compressor.compress(
            content=assembled_context,
            target_tokens=50_000,
            current_tokens=120_000,
        )
        print(result.compression_ratio)
    """

    def __init__(
        self,
        strategy: str | CompressionStrategy = CompressionStrategy.TRUNCATION,
        preserve_top_n: int = 2,
        token_counter: TokenCounter | None = None,
        llm_summarizer: LLMSummarizer | None = None,
    ) -> None:
        if isinstance(strategy, str):
            strategy = CompressionStrategy(strategy)
        self.strategy = strategy
        self.preserve_top_n = max(0, preserve_top_n)
        self.token_counter = token_counter or _EstimateCounter()
        self.llm_summarizer = llm_summarizer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def compress(
        self,
        content: str,
        target_tokens: int,
        current_tokens: int | None = None,
    ) -> CompressionResult:
        """
        Compress context to fit within ``target_tokens``.

        Args:
            content: The full context string (section-delimited).
            target_tokens: Desired token budget.
            current_tokens: Current token count (computed if ``None``).

        Returns:
            ``CompressionResult`` with compressed content and metadata.
        """
        if current_tokens is None:
            current_tokens = self.token_counter.count(content)

        if current_tokens <= target_tokens:
            # Already fits — no compression needed
            return CompressionResult(
                content=content,
                original_tokens=current_tokens,
                compressed_tokens=current_tokens,
                strategy_used="none",
            )

        if self.strategy == CompressionStrategy.TRUNCATION:
            return await self._truncate(content, target_tokens, current_tokens)
        elif self.strategy == CompressionStrategy.EXTRACTIVE:
            return await self._extract(content, target_tokens, current_tokens)
        elif self.strategy == CompressionStrategy.ABSTRACTIVE:
            return await self._abstractive(content, target_tokens, current_tokens)
        else:
            # Fallback
            return await self._truncate(content, target_tokens, current_tokens)

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    async def _truncate(
        self,
        content: str,
        target_tokens: int,
        current_tokens: int,
    ) -> CompressionResult:
        """Truncation strategy: keep beginning, drop tail."""
        sections = _split_sections(content)

        kept: list[str] = []
        used_tokens = 0

        for section in sections:
            section_tokens = self.token_counter.count(section)
            if used_tokens + section_tokens <= target_tokens:
                kept.append(section)
                used_tokens += section_tokens
            else:
                # Partial fit — truncate this section
                remaining = target_tokens - used_tokens
                if remaining > 200:
                    truncated = self._truncate_text(section, remaining)
                    kept.append(truncated)
                    used_tokens += self.token_counter.count(truncated)
                break

        compressed = "\n\n---\n\n".join(kept)
        if len(kept) < len(sections):
            compressed += "\n\n[... context truncated to fit token budget ...]"

        compressed_tokens = self.token_counter.count(compressed)
        return CompressionResult(
            content=compressed,
            original_tokens=current_tokens,
            compressed_tokens=compressed_tokens,
            strategy_used="truncation",
            sections_preserved=len(kept),
            sections_summarized=0,
        )

    async def _extract(
        self,
        content: str,
        target_tokens: int,
        current_tokens: int,
    ) -> CompressionResult:
        """
        Extractive strategy: keep top-N sections verbatim, extract
        key sentences from remaining sections.
        """
        sections = _split_sections(content)

        preserved: list[str] = []
        to_compress: list[str] = []
        preserved_tokens = 0

        for i, section in enumerate(sections):
            section_tokens = self.token_counter.count(section)
            if i < self.preserve_top_n:
                preserved.append(section)
                preserved_tokens += section_tokens
            else:
                to_compress.append(section)

        # Budget remaining for compressed sections
        remaining_budget = max(0, target_tokens - preserved_tokens)

        if to_compress and remaining_budget > 0:
            # Distribute budget across sections proportionally
            total_compress_tokens = sum(self.token_counter.count(s) for s in to_compress)
            compressed_sections = []

            for section in to_compress:
                section_tokens = self.token_counter.count(section)
                if total_compress_tokens > 0:
                    section_budget = int(
                        remaining_budget * (section_tokens / total_compress_tokens)
                    )
                else:
                    section_budget = remaining_budget // len(to_compress)

                compressed_section = _extract_key_sentences(
                    section, section_budget, self.token_counter
                )
                compressed_sections.append(compressed_section)

            all_sections = preserved + compressed_sections
        else:
            all_sections = preserved

        compressed = "\n\n---\n\n".join(all_sections)
        compressed_tokens = self.token_counter.count(compressed)

        return CompressionResult(
            content=compressed,
            original_tokens=current_tokens,
            compressed_tokens=compressed_tokens,
            strategy_used="extractive",
            sections_preserved=len(preserved),
            sections_summarized=len(to_compress),
        )

    async def _abstractive(
        self,
        content: str,
        target_tokens: int,
        current_tokens: int,
    ) -> CompressionResult:
        """
        Abstractive strategy: use LLM to summarize lower-priority
        sections.  Falls back to extractive if no LLM summarizer.
        """
        if self.llm_summarizer is None:
            logger.warning(
                "Abstractive compression requested but no LLM summarizer "
                "configured — falling back to extractive"
            )
            return await self._extract(content, target_tokens, current_tokens)

        sections = _split_sections(content)

        preserved: list[str] = []
        to_summarize: list[str] = []
        preserved_tokens = 0

        for i, section in enumerate(sections):
            section_tokens = self.token_counter.count(section)
            if i < self.preserve_top_n:
                preserved.append(section)
                preserved_tokens += section_tokens
            else:
                to_summarize.append(section)

        remaining_budget = max(0, target_tokens - preserved_tokens)

        summarized_sections: list[str] = []
        if to_summarize and remaining_budget > 0:
            # Summarize each section
            combined = "\n\n".join(to_summarize)
            try:
                summary = await self.llm_summarizer.summarize(combined, max_tokens=remaining_budget)
                summarized_sections.append(f"## COMPRESSED CONTEXT\n\n{summary}")
            except Exception as exc:
                logger.warning("LLM summarization failed: %s — using extractive", exc)
                # Fallback to extractive for the remaining sections
                for section in to_summarize:
                    section_budget = remaining_budget // max(1, len(to_summarize))
                    extracted = _extract_key_sentences(section, section_budget, self.token_counter)
                    summarized_sections.append(extracted)

        all_sections = preserved + summarized_sections
        compressed = "\n\n---\n\n".join(all_sections)
        compressed_tokens = self.token_counter.count(compressed)

        return CompressionResult(
            content=compressed,
            original_tokens=current_tokens,
            compressed_tokens=compressed_tokens,
            strategy_used="abstractive",
            sections_preserved=len(preserved),
            sections_summarized=len(to_summarize),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within max_tokens."""
        # Use binary search for efficient truncation
        low, high = 0, len(text)
        while low < high:
            mid = (low + high + 1) // 2
            if self.token_counter.count(text[:mid]) <= max_tokens:
                low = mid
            else:
                high = mid - 1
        return text[:low] + "\n[... truncated ...]" if low < len(text) else text


# =============================================================================
# Utilities
# =============================================================================


class _EstimateCounter:
    """Fallback token counter using character-based estimation."""

    def count(self, text: str) -> int:
        return max(1, len(text) // 4) if text else 0


def _split_sections(content: str) -> list[str]:
    """Split assembled context into sections by ``---`` delimiter."""
    raw_parts = re.split(r"\n\n---\n\n", content)
    return [p for p in raw_parts if p.strip()]


def _extract_key_sentences(
    text: str,
    budget_tokens: int,
    counter: TokenCounter,
) -> str:
    """
    Extract the most important sentences from text using heuristic
    scoring.

    Scoring heuristics:
    - Position bonus: earlier sentences score higher.
    - Length bonus: medium-length sentences (20–80 words) preferred.
    - Header proximity: sentences near headers score higher.
    - Code blocks: preserved verbatim (compact, high information).
    """
    # Preserve headers
    lines = text.split("\n")
    header_lines = [l for l in lines if l.startswith("#")]
    header = "\n".join(header_lines) + "\n\n" if header_lines else ""

    # Split into sentences (rough)
    # Keep code blocks as atomic units
    body = "\n".join(l for l in lines if not l.startswith("#"))
    sentences = _split_into_sentences(body)

    if not sentences:
        return header.rstrip()

    # Score sentences
    scored: list[tuple[float, str]] = []
    total = len(sentences)

    for i, sent in enumerate(sentences):
        score = 0.0
        words = sent.split()
        word_count = len(words)

        # Position bonus (first sentences more important)
        score += max(0.0, 1.0 - (i / max(1, total)))

        # Length bonus (prefer medium length)
        if 20 <= word_count <= 80:
            score += 0.3
        elif word_count > 5:
            score += 0.1

        # Code block bonus (high information density)
        if "```" in sent or sent.strip().startswith(("def ", "class ", "async ")):
            score += 0.5

        scored.append((score, sent))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Select sentences within budget
    selected: list[tuple[int, str]] = []  # (original_index, text)
    used_tokens = counter.count(header)

    # Map sentences back to original indices for ordering
    sentence_indices = {id(sent): i for i, sent in enumerate(sentences)}

    for _score, sent in scored:
        sent_tokens = counter.count(sent)
        if used_tokens + sent_tokens <= budget_tokens:
            orig_idx = sentence_indices.get(id(sent), 0)
            selected.append((orig_idx, sent))
            used_tokens += sent_tokens

    # Re-order by original position
    selected.sort(key=lambda x: x[0])

    result = header + "\n".join(s for _, s in selected)
    return result.rstrip()


def _split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences, preserving code blocks as atomic units.
    """
    sentences: list[str] = []
    in_code_block = False
    current_block: list[str] = []

    for line in text.split("\n"):
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code_block:
                # End of code block
                current_block.append(line)
                sentences.append("\n".join(current_block))
                current_block = []
                in_code_block = False
            else:
                # Start of code block
                in_code_block = True
                current_block.append(line)
            continue

        if in_code_block:
            current_block.append(line)
            continue

        # Regular text — split by sentence boundaries
        if stripped:
            # Simple sentence splitting on ". " boundary
            parts = re.split(r"(?<=[.!?])\s+", stripped)
            sentences.extend(parts)

    # Handle unclosed code block
    if current_block:
        sentences.append("\n".join(current_block))

    return [s for s in sentences if s.strip()]
