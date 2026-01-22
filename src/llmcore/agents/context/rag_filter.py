"""
RAG Context Quality Filter for LLMCore Agent System.

This module filters RAG retrieval results to ensure context quality:
- Removes low-similarity results
- Detects and filters garbage/placeholder content
- Deduplicates similar results
- Limits total results to prevent context bloat

Usage:
    from llmcore.agents.context.rag_filter import RAGContextFilter, RAGResult

    filter = RAGContextFilter(min_similarity=0.7, max_results=5)

    # Raw results from vector store
    raw_results = [RAGResult(...), ...]

    # Filtered, quality-assured results
    filtered = filter.filter(raw_results)

Integration Point:
    Call in PERCEIVE phase after RAG retrieval, before context building.

Author: llmcore team
Date: 2026-01-21
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class RAGResult:
    """
    A single RAG retrieval result.

    Attributes:
        content: The retrieved text content
        similarity: Similarity score (0.0 to 1.0)
        source: Source identifier (file path, URL, etc.)
        metadata: Additional metadata from the vector store
    """

    content: str
    similarity: float
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Normalize similarity to 0-1 range
        if self.similarity > 1.0:
            self.similarity = self.similarity / 100.0


@dataclass
class FilterStats:
    """Statistics from a filter operation."""

    input_count: int = 0
    output_count: int = 0
    removed_low_similarity: int = 0
    removed_garbage: int = 0
    removed_duplicate: int = 0
    removed_limit: int = 0

    @property
    def total_removed(self) -> int:
        return self.input_count - self.output_count

    @property
    def retention_rate(self) -> float:
        if self.input_count == 0:
            return 0.0
        return self.output_count / self.input_count


# =============================================================================
# Garbage Detection Patterns
# =============================================================================

# Patterns that indicate garbage/placeholder content
GARBAGE_PATTERNS = [
    # Placeholder patterns
    r"^Document\s+\d+\s*$",  # "Document 1"
    r"^(Section|Chapter|Part)\s+\d+\s*$",  # "Section 1"
    r"^(TODO|FIXME|XXX|HACK)\b",  # TODO markers
    r"^(Lorem ipsum|placeholder|sample\s+text)",  # Placeholder text
    r"^\[.*?\]\s*$",  # Just brackets [...]
    r"^(test|example|sample)\s+\d*\s*$",  # "test 1", "example"
    # Structural artifacts
    r"^={3,}\s*$",  # Separator lines ===
    r"^-{3,}\s*$",  # Separator lines ---
    r"^\*{3,}\s*$",  # Separator lines ***
    r"^#{1,6}\s*$",  # Empty headers
    # Empty/whitespace
    r"^\s*$",  # Empty/whitespace only
    r"^[\s\n\r]+$",  # Just newlines
]

# Patterns for potentially truncated/corrupted content
CORRUPTION_PATTERNS = [
    r"\x00",  # Null bytes
    r"[\ufffd\ufffe\uffff]",  # Unicode replacement chars
    r"^.{0,5}$",  # Extremely short (< 5 chars)
]


# =============================================================================
# RAG Context Filter Implementation
# =============================================================================


class RAGContextFilter:
    """
    Filter RAG retrieval results for quality.

    Removes low-quality results to prevent polluting the LLM context with:
    - Low similarity matches
    - Placeholder/garbage content
    - Duplicate or near-duplicate content
    - Excessive results

    Args:
        min_similarity: Minimum similarity threshold (0.0 to 1.0)
        max_results: Maximum number of results to return
        dedup_threshold: Similarity threshold for deduplication (0.0 to 1.0)
        min_content_length: Minimum content length in characters
        max_repetition_ratio: Maximum ratio of repeated lines allowed
    """

    def __init__(
        self,
        min_similarity: float = 0.7,
        max_results: int = 5,
        dedup_threshold: float = 0.85,
        min_content_length: int = 20,
        max_repetition_ratio: float = 0.5,
    ):
        self.min_similarity = min_similarity
        self.max_results = max_results
        self.dedup_threshold = dedup_threshold
        self.min_content_length = min_content_length
        self.max_repetition_ratio = max_repetition_ratio

        # Compile patterns
        self._garbage_patterns = [
            re.compile(p, re.IGNORECASE | re.MULTILINE) for p in GARBAGE_PATTERNS
        ]
        self._corruption_patterns = [re.compile(p) for p in CORRUPTION_PATTERNS]

    def filter(self, results: List[RAGResult]) -> List[RAGResult]:
        """
        Filter RAG results for quality.

        Processing order:
        1. Remove below-threshold similarity
        2. Remove garbage/placeholder content
        3. Remove corrupted content
        4. Deduplicate similar results
        5. Sort by relevance and limit

        Args:
            results: Raw RAG retrieval results

        Returns:
            Filtered, quality-assured results
        """
        if not results:
            return []

        stats = FilterStats(input_count=len(results))

        # Step 1: Similarity filter
        filtered = [r for r in results if r.similarity >= self.min_similarity]
        stats.removed_low_similarity = len(results) - len(filtered)
        logger.debug(
            f"After similarity filter ({self.min_similarity}): {len(filtered)}/{len(results)}"
        )

        # Step 2: Garbage filter
        pre_garbage = len(filtered)
        filtered = [r for r in filtered if not self._is_garbage(r)]
        stats.removed_garbage = pre_garbage - len(filtered)
        logger.debug(f"After garbage filter: {len(filtered)}/{pre_garbage}")

        # Step 3: Deduplication
        pre_dedup = len(filtered)
        filtered = self._deduplicate(filtered)
        stats.removed_duplicate = pre_dedup - len(filtered)
        logger.debug(f"After deduplication: {len(filtered)}/{pre_dedup}")

        # Step 4: Sort by relevance and limit
        filtered.sort(key=lambda r: r.similarity, reverse=True)
        pre_limit = len(filtered)
        filtered = filtered[: self.max_results]
        stats.removed_limit = pre_limit - len(filtered)

        stats.output_count = len(filtered)

        if stats.total_removed > 0:
            logger.info(
                f"RAG filter: {stats.input_count} → {stats.output_count} "
                f"(removed: {stats.removed_low_similarity} low-sim, "
                f"{stats.removed_garbage} garbage, "
                f"{stats.removed_duplicate} dups, "
                f"{stats.removed_limit} limit)"
            )

        return filtered

    def filter_with_stats(self, results: List[RAGResult]) -> tuple[List[RAGResult], FilterStats]:
        """
        Filter RAG results and return statistics.

        Same as filter(), but also returns FilterStats for monitoring/debugging.

        Args:
            results: Raw RAG retrieval results

        Returns:
            Tuple of (filtered_results, stats)
        """
        if not results:
            return [], FilterStats(input_count=0, output_count=0)

        stats = FilterStats(input_count=len(results))

        # Step 1: Similarity filter
        filtered = [r for r in results if r.similarity >= self.min_similarity]
        stats.removed_low_similarity = len(results) - len(filtered)

        # Step 2: Garbage filter
        pre_garbage = len(filtered)
        filtered = [r for r in filtered if not self._is_garbage(r)]
        stats.removed_garbage = pre_garbage - len(filtered)

        # Step 3: Deduplication
        pre_dedup = len(filtered)
        filtered = self._deduplicate(filtered)
        stats.removed_duplicate = pre_dedup - len(filtered)

        # Step 4: Sort by relevance and limit
        filtered.sort(key=lambda r: r.similarity, reverse=True)
        pre_limit = len(filtered)
        filtered = filtered[: self.max_results]
        stats.removed_limit = pre_limit - len(filtered)

        stats.output_count = len(filtered)

        return filtered, stats

    def _is_garbage(self, result: RAGResult) -> bool:
        """
        Detect garbage/placeholder RAG results.

        Returns True if the result should be filtered out.
        """
        content = result.content

        # Check 1: Content too short
        if len(content.strip()) < self.min_content_length:
            logger.debug(f"Filtered: too short ({len(content)} chars)")
            return True

        # Check 2: Matches garbage patterns
        for pattern in self._garbage_patterns:
            if pattern.match(content.strip()):
                logger.debug(f"Filtered: matches garbage pattern: {content[:50]}")
                return True

        # Check 3: Corruption patterns
        for pattern in self._corruption_patterns:
            if pattern.search(content):
                logger.debug(f"Filtered: contains corruption pattern")
                return True

        # Check 4: Excessive repetition
        if self._has_excessive_repetition(content):
            logger.debug(f"Filtered: excessive repetition: {content[:50]}")
            return True

        # Check 5: All same character
        unique_chars = set(content.replace(" ", "").replace("\n", ""))
        if len(unique_chars) < 3:
            logger.debug(f"Filtered: too few unique characters")
            return True

        return False

    def _has_excessive_repetition(self, text: str) -> bool:
        """
        Check if text has excessive line repetition.

        This catches cases like:
        "Document 1
         Document 1
         Document 1"
        """
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        if len(lines) < 3:
            return False

        # Count unique lines
        line_counts = Counter(lines)
        most_common_count = line_counts.most_common(1)[0][1]

        # If one line appears more than threshold ratio, it's repetitive
        repetition_ratio = most_common_count / len(lines)

        if repetition_ratio > self.max_repetition_ratio:
            return True

        # Also check for sequential repetition
        consecutive_same = 0
        max_consecutive = 0
        prev_line = None

        for line in lines:
            if line == prev_line:
                consecutive_same += 1
                max_consecutive = max(max_consecutive, consecutive_same)
            else:
                consecutive_same = 1
            prev_line = line

        # If 3+ identical consecutive lines, it's suspicious
        if max_consecutive >= 3:
            return True

        return False

    def _deduplicate(self, results: List[RAGResult]) -> List[RAGResult]:
        """
        Remove near-duplicate results.

        Keeps the highest-similarity version of similar content.
        """
        if len(results) <= 1:
            return results

        # Sort by similarity (highest first) so we keep the best match
        sorted_results = sorted(results, key=lambda r: r.similarity, reverse=True)

        unique: List[RAGResult] = []

        for result in sorted_results:
            is_duplicate = False

            for existing in unique:
                similarity = self._text_similarity(result.content, existing.content)
                if similarity >= self.dedup_threshold:
                    is_duplicate = True
                    logger.debug(
                        f"Deduplicated (sim={similarity:.2f}): "
                        f"'{result.content[:30]}...' vs '{existing.content[:30]}...'"
                    )
                    break

            if not is_duplicate:
                unique.append(result)

        return unique

    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using word overlap (Jaccard).

        This is a fast approximation suitable for deduplication.
        For more accurate similarity, use embeddings.
        """
        # Normalize
        text1 = text1.lower()
        text2 = text2.lower()

        # Strip punctuation for better matching
        # "dog." and "dog" should be considered the same word
        punctuation = ".,!?;:'\"()[]{}—–-"
        for p in punctuation:
            text1 = text1.replace(p, " ")
            text2 = text2.replace(p, " ")

        # Tokenize (whitespace split, filter empty)
        words1 = set(w for w in text1.split() if w)
        words2 = set(w for w in text2.split() if w)

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        if union == 0:
            return 0.0

        return intersection / union

    # =========================================================================
    # Batch Processing
    # =========================================================================

    def filter_batches(
        self, results_by_query: Dict[str, List[RAGResult]]
    ) -> Dict[str, List[RAGResult]]:
        """
        Filter multiple query results while deduplicating across queries.

        Args:
            results_by_query: Map of query string to RAG results

        Returns:
            Filtered results, with cross-query deduplication
        """
        # First, filter each query's results individually
        filtered_by_query = {
            query: self.filter(results) for query, results in results_by_query.items()
        }

        # Then deduplicate across queries
        seen_contents: Set[str] = set()
        final_results: Dict[str, List[RAGResult]] = {}

        for query, results in filtered_by_query.items():
            unique = []
            for result in results:
                # Use content hash for cross-query dedup
                content_key = result.content[:200].lower()  # First 200 chars
                if content_key not in seen_contents:
                    seen_contents.add(content_key)
                    unique.append(result)
            final_results[query] = unique

        return final_results


# =============================================================================
# Convenience Functions
# =============================================================================


def filter_rag_results(
    results: List[RAGResult],
    min_similarity: float = 0.7,
    max_results: int = 5,
) -> List[RAGResult]:
    """
    Convenience function for quick RAG filtering.

    Args:
        results: Raw RAG results
        min_similarity: Minimum similarity threshold
        max_results: Maximum results to return

    Returns:
        Filtered results
    """
    filter = RAGContextFilter(
        min_similarity=min_similarity,
        max_results=max_results,
    )
    return filter.filter(results)


def format_rag_context(
    results: List[RAGResult],
    max_chars: int = 4000,
    include_sources: bool = True,
) -> str:
    """
    Format filtered RAG results for LLM context.

    Args:
        results: Filtered RAG results
        max_chars: Maximum total characters
        include_sources: Whether to include source attribution

    Returns:
        Formatted context string
    """
    if not results:
        return ""

    sections = []
    total_chars = 0

    for i, result in enumerate(results, 1):
        # Build section
        header = f"[{i}]"
        if include_sources and result.source:
            header += f" Source: {result.source}"

        section = f"{header}\n{result.content.strip()}"

        # Check length limit
        if total_chars + len(section) > max_chars:
            # Truncate this section if needed
            remaining = max_chars - total_chars - 50  # Leave room for ellipsis
            if remaining > 100:
                section = section[:remaining] + "... [truncated]"
                sections.append(section)
            break

        sections.append(section)
        total_chars += len(section) + 2  # +2 for newlines

    return "\n\n".join(sections)


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    print("Running RAG Context Filter self-tests...\n")

    passed = 0
    failed = 0

    # Test 1: Similarity filtering
    print("Test 1: Similarity filtering")
    filter = RAGContextFilter(min_similarity=0.7)

    results = [
        RAGResult(content="High quality content here", similarity=0.9),
        RAGResult(content="Medium quality content", similarity=0.75),
        RAGResult(content="Low quality content", similarity=0.5),
    ]

    filtered = filter.filter(results)
    if len(filtered) == 2:
        print(f"  ✅ Filtered from 3 to 2 (removed low similarity)")
        passed += 1
    else:
        print(f"  ❌ Expected 2 results, got {len(filtered)}")
        failed += 1

    # Test 2: Garbage detection
    print("\nTest 2: Garbage detection")

    results = [
        RAGResult(content="Document 1", similarity=0.9),
        RAGResult(content="Document 1\nDocument 1\nDocument 1", similarity=0.9),
        RAGResult(content="Valid content with actual information", similarity=0.8),
        RAGResult(content="TODO: fill this in", similarity=0.85),
        RAGResult(content="", similarity=0.9),
    ]

    filtered = filter.filter(results)
    valid_contents = [r.content for r in filtered]

    if "Valid content with actual information" in valid_contents and len(filtered) == 1:
        print(f"  ✅ Kept only valid content, filtered {len(results) - 1} garbage items")
        passed += 1
    else:
        print(f"  ❌ Expected 1 result, got {len(filtered)}: {valid_contents}")
        failed += 1

    # Test 3: Deduplication
    print("\nTest 3: Deduplication")

    results = [
        RAGResult(content="The quick brown fox jumps over the lazy dog", similarity=0.9),
        RAGResult(
            content="The quick brown fox jumps over the lazy dog today", similarity=0.85
        ),  # Near dup
        RAGResult(content="Something completely different and unique", similarity=0.8),
    ]

    filtered = filter.filter(results)
    if len(filtered) == 2:
        print(f"  ✅ Deduplicated from 3 to 2")
        passed += 1
    else:
        print(f"  ❌ Expected 2 results, got {len(filtered)}")
        failed += 1

    # Test 4: Max results limit
    print("\nTest 4: Max results limit")
    filter_limited = RAGContextFilter(max_results=2)

    results = [
        RAGResult(content="Content A with information", similarity=0.9),
        RAGResult(content="Content B with information", similarity=0.85),
        RAGResult(content="Content C with information", similarity=0.8),
        RAGResult(content="Content D with information", similarity=0.75),
    ]

    filtered = filter_limited.filter(results)
    if len(filtered) == 2 and filtered[0].similarity >= filtered[1].similarity:
        print(f"  ✅ Limited to 2 results, highest similarity first")
        passed += 1
    else:
        print(f"  ❌ Expected 2 results sorted by similarity, got {len(filtered)}")
        failed += 1

    # Test 5: Repetition detection
    print("\nTest 5: Repetition detection")

    results = [
        RAGResult(
            content="Line 1\nLine 1\nLine 1\nLine 1\nLine 1",  # 5 identical lines
            similarity=0.9,
        ),
        RAGResult(
            content="Line A\nLine B\nLine C\nLine D\nLine E",  # 5 different lines
            similarity=0.85,
        ),
    ]

    filtered = filter.filter(results)
    if len(filtered) == 1 and "Line A" in filtered[0].content:
        print(f"  ✅ Filtered repetitive content")
        passed += 1
    else:
        print(f"  ❌ Expected to filter repetitive content")
        failed += 1

    # Test 6: Format output
    print("\nTest 6: Format output")

    results = [
        RAGResult(content="First document content", similarity=0.9, source="doc1.txt"),
        RAGResult(content="Second document content", similarity=0.85, source="doc2.txt"),
    ]

    formatted = format_rag_context(results, include_sources=True)

    if "[1]" in formatted and "Source: doc1.txt" in formatted:
        print(f"  ✅ Formatted context with sources")
        passed += 1
    else:
        print(f"  ❌ Formatting failed")
        failed += 1

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("All tests passed! ✅")
    else:
        print("Some tests failed ❌")
        exit(1)
