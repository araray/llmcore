# src/llmcore/ingestion/chunking.py
"""
Chunking Strategies for Document Ingestion.

Phase 3 (SYMBIOSIS): This module provides various text chunking strategies
for preparing documents for vector storage. These are designed to work
with LLMCore's unified storage layer.

Chunking Strategies:
- FixedSizeChunker: Simple character/token-based chunking
- RecursiveTextChunker: Recursive splitting on hierarchical separators
- SentenceChunker: Sentence-based chunking with grouping

The chunkers produce Chunk objects that are compatible with both
LLMCore's storage and SemantiScan's existing pipelines.

Usage:
    from llmcore.ingestion import RecursiveTextChunker, Chunk

    chunker = RecursiveTextChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.chunk(
        text="Long document text...",
        metadata={"source": "document.md", "language": "en"}
    )

    for chunk in chunks:
        print(f"Chunk {chunk.id}: {len(chunk.content)} chars")
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class Chunk:
    """
    A chunk of text with metadata for vector storage.

    This dataclass is designed to be compatible with SemantiScan's Chunk model
    while being usable independently within LLMCore.

    Attributes:
        id: Unique identifier for the chunk.
        content: The text content of the chunk.
        metadata: Dictionary of metadata (source, position, etc.).
        embedding: Optional pre-computed embedding vector.
    """

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    @classmethod
    def create(
        cls,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_id: Optional[str] = None,
    ) -> "Chunk":
        """
        Factory method to create a chunk with auto-generated ID.

        Args:
            content: The text content.
            metadata: Optional metadata dictionary.
            chunk_id: Optional specific ID (auto-generated if not provided).

        Returns:
            New Chunk instance.
        """
        if chunk_id is None:
            # Generate ID from content hash + random suffix for uniqueness
            content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
            unique_suffix = str(uuid.uuid4())[:8]
            chunk_id = f"chunk_{content_hash}_{unique_suffix}"

        return cls(
            id=chunk_id,
            content=content,
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary representation."""
        result = {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
        }
        if self.embedding:
            result["embedding"] = self.embedding
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """Create chunk from dictionary representation."""
        return cls(
            id=data["id"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
        )

    def __len__(self) -> int:
        """Return the length of the content."""
        return len(self.content)


@dataclass
class ChunkingConfig:
    """
    Configuration for chunking operations.

    Attributes:
        chunk_size: Target chunk size in characters or tokens.
        chunk_overlap: Overlap between consecutive chunks.
        min_chunk_size: Minimum chunk size (smaller chunks are merged).
        max_chunk_size: Maximum chunk size (larger chunks are split).
        length_function: Function to measure text length (default: len).
        preserve_sentences: Try to preserve sentence boundaries.
        separators: List of separators for recursive splitting.
    """

    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    length_function: Callable[[str], int] = len
    preserve_sentences: bool = True
    separators: List[str] = field(
        default_factory=lambda: [
            "\n\n",  # Paragraphs
            "\n",  # Lines
            ". ",  # Sentences
            "! ",  # Exclamations
            "? ",  # Questions
            "; ",  # Semicolons
            ", ",  # Commas
            " ",  # Words
            "",  # Characters (last resort)
        ]
    )


# =============================================================================
# BASE STRATEGY
# =============================================================================


class ChunkingStrategy(ABC):
    """
    Abstract base class for chunking strategies.

    All chunking strategies must implement the `chunk` method that takes
    text and produces a list of Chunk objects.
    """

    @abstractmethod
    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        Split text into chunks.

        Args:
            text: The text to chunk.
            metadata: Base metadata to include in all chunks.

        Returns:
            List of Chunk objects.
        """
        pass

    def chunk_documents(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[Chunk]:
        """
        Chunk multiple documents.

        Args:
            documents: List of dicts with 'content' and optional 'metadata'.

        Returns:
            List of all Chunk objects from all documents.
        """
        all_chunks = []
        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            chunks = self.chunk(content, metadata)
            all_chunks.extend(chunks)
        return all_chunks


# =============================================================================
# FIXED SIZE CHUNKER
# =============================================================================


class FixedSizeChunker(ChunkingStrategy):
    """
    Simple fixed-size character-based chunker.

    Splits text into chunks of approximately equal size with overlap.
    Does not attempt to preserve semantic boundaries.

    Best for:
    - Uniform processing requirements
    - When semantic boundaries aren't important
    - Fast, predictable chunking

    Example:
        chunker = FixedSizeChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk("Long text...")
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Target size for each chunk.
            chunk_overlap: Number of characters to overlap between chunks.
            length_function: Function to measure text length.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        Split text into fixed-size chunks.

        Args:
            text: The text to chunk.
            metadata: Base metadata for chunks.

        Returns:
            List of Chunk objects.
        """
        if not text or not text.strip():
            return []

        metadata = metadata or {}
        chunks = []

        step = self.chunk_size - self.chunk_overlap
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            if chunk_text.strip():  # Only add non-empty chunks
                chunk_metadata = {
                    **metadata,
                    "chunk_index": chunk_index,
                    "chunk_start": start,
                    "chunk_end": min(end, len(text)),
                    "chunk_method": "fixed_size",
                }

                chunks.append(
                    Chunk.create(
                        content=chunk_text.strip(),
                        metadata=chunk_metadata,
                    )
                )
                chunk_index += 1

            start += step

        logger.debug(f"FixedSizeChunker: {len(text)} chars -> {len(chunks)} chunks")
        return chunks


# =============================================================================
# RECURSIVE TEXT CHUNKER
# =============================================================================


class RecursiveTextChunker(ChunkingStrategy):
    """
    Recursive text splitter using hierarchical separators.

    Attempts to split text on semantic boundaries (paragraphs, sentences)
    while respecting the target chunk size. Falls back to smaller separators
    when needed.

    This is the recommended general-purpose chunker, similar to
    LangChain's RecursiveCharacterTextSplitter.

    Best for:
    - General text processing
    - When semantic boundaries should be preserved
    - Mixed content (prose, code, lists)

    Example:
        chunker = RecursiveTextChunker(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " "]
        )
        chunks = chunker.chunk("Long document...")
    """

    DEFAULT_SEPARATORS = [
        "\n\n",  # Paragraphs
        "\n",  # Lines
        ". ",  # Sentences
        "! ",
        "? ",
        "; ",
        ", ",
        " ",  # Words
        "",  # Characters (last resort)
    ]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        length_function: Callable[[str], int] = len,
        strip_whitespace: bool = True,
    ):
        """
        Initialize the recursive chunker.

        Args:
            chunk_size: Target size for each chunk.
            chunk_overlap: Overlap between chunks.
            separators: Ordered list of separators (most to least preferred).
            length_function: Function to measure text length.
            strip_whitespace: Whether to strip whitespace from chunks.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.length_function = length_function
        self.strip_whitespace = strip_whitespace

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        Recursively split text into chunks.

        Args:
            text: The text to chunk.
            metadata: Base metadata for chunks.

        Returns:
            List of Chunk objects.
        """
        if not text or not text.strip():
            return []

        metadata = metadata or {}

        # Get raw splits
        splits = self._split_text(text, self.separators)

        # Merge splits into chunks of appropriate size
        chunks = self._merge_splits(splits, metadata)

        logger.debug(f"RecursiveTextChunker: {len(text)} chars -> {len(chunks)} chunks")
        return chunks

    def _split_text(
        self,
        text: str,
        separators: List[str],
    ) -> List[str]:
        """
        Recursively split text using hierarchical separators.

        Args:
            text: Text to split.
            separators: Remaining separators to try.

        Returns:
            List of text fragments.
        """
        final_chunks = []

        # Find the appropriate separator
        separator = separators[-1]  # Default to last (smallest) separator
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1 :]
                break

        # Split on the chosen separator
        if separator:
            splits = text.split(separator)
        else:
            # Character-level splitting
            splits = list(text)

        # Process each split
        good_splits = []
        separator_to_add = separator if separator else ""

        for split in splits:
            if self.length_function(split) < self.chunk_size:
                good_splits.append(split)
            else:
                # Split is too large, recursively split it
                if good_splits:
                    merged = self._merge_small_splits(good_splits, separator_to_add)
                    final_chunks.extend(merged)
                    good_splits = []

                if new_separators:
                    # Recursively split with smaller separators
                    other_chunks = self._split_text(split, new_separators)
                    final_chunks.extend(other_chunks)
                else:
                    # No more separators, add as-is
                    final_chunks.append(split)

        # Add remaining good splits
        if good_splits:
            merged = self._merge_small_splits(good_splits, separator_to_add)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_small_splits(
        self,
        splits: List[str],
        separator: str,
    ) -> List[str]:
        """
        Merge small splits that fit within chunk_size.

        Args:
            splits: List of text fragments.
            separator: Separator to use when joining.

        Returns:
            List of merged text fragments.
        """
        if not splits:
            return []

        merged = []
        current = []
        current_length = 0

        for split in splits:
            split_length = self.length_function(split)

            # Calculate new length if we add this split
            new_length = current_length
            if current:
                new_length += self.length_function(separator)
            new_length += split_length

            if new_length <= self.chunk_size:
                current.append(split)
                current_length = new_length
            else:
                if current:
                    merged.append(separator.join(current))
                current = [split]
                current_length = split_length

        if current:
            merged.append(separator.join(current))

        return merged

    def _merge_splits(
        self,
        splits: List[str],
        metadata: Dict[str, Any],
    ) -> List[Chunk]:
        """
        Merge splits into final chunks with overlap.

        Args:
            splits: List of text fragments.
            metadata: Base metadata.

        Returns:
            List of Chunk objects.
        """
        if not splits:
            return []

        chunks = []
        current_texts = []
        current_length = 0
        chunk_index = 0
        char_position = 0

        for split in splits:
            split_text = split.strip() if self.strip_whitespace else split
            if not split_text:
                continue

            split_length = self.length_function(split_text)

            # Would adding this exceed chunk_size?
            if current_length + split_length > self.chunk_size and current_texts:
                # Create chunk from current accumulated texts
                chunk_content = " ".join(current_texts)
                if self.strip_whitespace:
                    chunk_content = chunk_content.strip()

                if chunk_content:
                    chunk_metadata = {
                        **metadata,
                        "chunk_index": chunk_index,
                        "chunk_start": char_position,
                        "chunk_method": "recursive",
                    }
                    chunks.append(
                        Chunk.create(
                            content=chunk_content,
                            metadata=chunk_metadata,
                        )
                    )
                    chunk_index += 1

                # Handle overlap: keep some texts for next chunk
                while current_texts and current_length > self.chunk_overlap:
                    removed = current_texts.pop(0)
                    current_length -= self.length_function(removed)
                    char_position += len(removed) + 1  # +1 for space

            # Add current split
            current_texts.append(split_text)
            current_length += split_length

        # Handle remaining texts
        if current_texts:
            chunk_content = " ".join(current_texts)
            if self.strip_whitespace:
                chunk_content = chunk_content.strip()

            if chunk_content:
                chunk_metadata = {
                    **metadata,
                    "chunk_index": chunk_index,
                    "chunk_start": char_position,
                    "chunk_method": "recursive",
                }
                chunks.append(
                    Chunk.create(
                        content=chunk_content,
                        metadata=chunk_metadata,
                    )
                )

        return chunks


# =============================================================================
# SENTENCE CHUNKER
# =============================================================================


class SentenceChunker(ChunkingStrategy):
    """
    Sentence-based chunker that groups sentences to meet target size.

    Splits text into sentences first, then groups sentences into chunks
    that respect the target chunk size while preserving sentence boundaries.

    Best for:
    - Prose text where sentence boundaries are important
    - Question-answering systems
    - Semantic search applications

    Example:
        chunker = SentenceChunker(
            target_chunk_size=1000,
            min_sentences=1,
            max_sentences=10
        )
        chunks = chunker.chunk("Document with many sentences...")
    """

    # Regex for sentence splitting
    SENTENCE_ENDINGS = re.compile(r"(?<=[.!?])\s+")

    def __init__(
        self,
        target_chunk_size: int = 1000,
        chunk_overlap_sentences: int = 1,
        min_sentences: int = 1,
        max_sentences: int = 20,
        length_function: Callable[[str], int] = len,
    ):
        """
        Initialize the sentence chunker.

        Args:
            target_chunk_size: Target size for chunks in characters.
            chunk_overlap_sentences: Number of sentences to overlap.
            min_sentences: Minimum sentences per chunk.
            max_sentences: Maximum sentences per chunk.
            length_function: Function to measure text length.
        """
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap_sentences = chunk_overlap_sentences
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.length_function = length_function

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Text to split.

        Returns:
            List of sentences.
        """
        # Split on sentence endings
        sentences = self.SENTENCE_ENDINGS.split(text)

        # Clean up sentences
        cleaned = []
        for sent in sentences:
            sent = sent.strip()
            if sent:
                cleaned.append(sent)

        return cleaned

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        Split text into sentence-based chunks.

        Args:
            text: The text to chunk.
            metadata: Base metadata for chunks.

        Returns:
            List of Chunk objects.
        """
        if not text or not text.strip():
            return []

        metadata = metadata or {}
        sentences = self._split_sentences(text)

        if not sentences:
            return []

        chunks = []
        current_sentences = []
        current_length = 0
        chunk_index = 0

        for sentence in sentences:
            sent_length = self.length_function(sentence)

            # Check if adding this sentence exceeds target size
            # or we've hit max sentences
            if (
                current_length + sent_length > self.target_chunk_size
                and len(current_sentences) >= self.min_sentences
            ) or len(current_sentences) >= self.max_sentences:
                # Create chunk
                chunk_content = " ".join(current_sentences)
                chunk_metadata = {
                    **metadata,
                    "chunk_index": chunk_index,
                    "sentence_count": len(current_sentences),
                    "chunk_method": "sentence",
                }
                chunks.append(
                    Chunk.create(
                        content=chunk_content,
                        metadata=chunk_metadata,
                    )
                )
                chunk_index += 1

                # Handle overlap
                overlap_start = max(0, len(current_sentences) - self.chunk_overlap_sentences)
                current_sentences = current_sentences[overlap_start:]
                current_length = sum(self.length_function(s) for s in current_sentences)

            current_sentences.append(sentence)
            current_length += sent_length

        # Handle remaining sentences
        if current_sentences:
            chunk_content = " ".join(current_sentences)
            chunk_metadata = {
                **metadata,
                "chunk_index": chunk_index,
                "sentence_count": len(current_sentences),
                "chunk_method": "sentence",
            }
            chunks.append(
                Chunk.create(
                    content=chunk_content,
                    metadata=chunk_metadata,
                )
            )

        logger.debug(f"SentenceChunker: {len(sentences)} sentences -> {len(chunks)} chunks")
        return chunks


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_chunker(
    strategy: str = "recursive",
    **kwargs,
) -> ChunkingStrategy:
    """
    Factory function to create a chunker by strategy name.

    Args:
        strategy: Strategy name ('fixed', 'recursive', 'sentence').
        **kwargs: Strategy-specific arguments.

    Returns:
        Configured ChunkingStrategy instance.

    Raises:
        ValueError: If strategy name is unknown.
    """
    strategy_map = {
        "fixed": FixedSizeChunker,
        "fixed_size": FixedSizeChunker,
        "recursive": RecursiveTextChunker,
        "recursive_text": RecursiveTextChunker,
        "sentence": SentenceChunker,
        "sentences": SentenceChunker,
    }

    strategy_lower = strategy.lower()
    if strategy_lower not in strategy_map:
        valid = list(strategy_map.keys())
        raise ValueError(f"Unknown chunking strategy: {strategy}. Valid options: {valid}")

    return strategy_map[strategy_lower](**kwargs)


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    strategy: str = "recursive",
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Chunk]:
    """
    Convenience function to chunk text with default settings.

    Args:
        text: Text to chunk.
        chunk_size: Target chunk size.
        chunk_overlap: Overlap between chunks.
        strategy: Chunking strategy name.
        metadata: Optional metadata for chunks.

    Returns:
        List of Chunk objects.
    """
    chunker = create_chunker(
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return chunker.chunk(text, metadata)
