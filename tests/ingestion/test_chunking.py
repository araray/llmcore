# tests/ingestion/test_chunking.py
"""
Tests for LLMCore Chunking Strategies.

Phase 3 (SYMBIOSIS): Tests for the chunking module that provides
text splitting strategies for document ingestion.

These tests verify:
- Chunk data model
- Fixed-size chunking
- Recursive text chunking
- Sentence-based chunking
- Factory functions
"""

import pytest

from llmcore.ingestion import (
    Chunk,
    ChunkingConfig,
    ChunkingStrategy,
    FixedSizeChunker,
    RecursiveTextChunker,
    SentenceChunker,
)
from llmcore.ingestion.chunking import create_chunker, chunk_text


# =============================================================================
# TESTS FOR CHUNK DATA MODEL
# =============================================================================


class TestChunk:
    """Tests for the Chunk data model."""

    def test_chunk_creation_basic(self):
        """Test basic chunk creation."""
        chunk = Chunk(
            id="chunk_001",
            content="Hello, world!",
            metadata={"source": "test.txt"},
        )

        assert chunk.id == "chunk_001"
        assert chunk.content == "Hello, world!"
        assert chunk.metadata["source"] == "test.txt"
        assert chunk.embedding is None

    def test_chunk_creation_with_embedding(self):
        """Test chunk creation with embedding."""
        chunk = Chunk(
            id="chunk_002",
            content="Test content",
            metadata={},
            embedding=[0.1, 0.2, 0.3],
        )

        assert chunk.embedding == [0.1, 0.2, 0.3]

    def test_chunk_create_factory(self):
        """Test Chunk.create() factory method."""
        chunk = Chunk.create(
            content="Factory created content",
            metadata={"key": "value"},
        )

        assert "chunk_" in chunk.id  # Auto-generated ID
        assert chunk.content == "Factory created content"
        assert chunk.metadata["key"] == "value"

    def test_chunk_create_with_custom_id(self):
        """Test Chunk.create() with custom ID."""
        chunk = Chunk.create(
            content="Content",
            chunk_id="custom_id_123",
        )

        assert chunk.id == "custom_id_123"

    def test_chunk_to_dict(self):
        """Test chunk serialization to dictionary."""
        chunk = Chunk(
            id="chunk_003",
            content="Serializable content",
            metadata={"file": "test.py"},
            embedding=[0.5],
        )

        d = chunk.to_dict()

        assert d["id"] == "chunk_003"
        assert d["content"] == "Serializable content"
        assert d["metadata"]["file"] == "test.py"
        assert d["embedding"] == [0.5]

    def test_chunk_from_dict(self):
        """Test chunk deserialization from dictionary."""
        data = {
            "id": "chunk_004",
            "content": "Deserialized content",
            "metadata": {"source": "api"},
            "embedding": [0.1, 0.2],
        }

        chunk = Chunk.from_dict(data)

        assert chunk.id == "chunk_004"
        assert chunk.content == "Deserialized content"
        assert chunk.metadata["source"] == "api"
        assert chunk.embedding == [0.1, 0.2]

    def test_chunk_len(self):
        """Test chunk length (content length)."""
        chunk = Chunk.create(content="12345")

        assert len(chunk) == 5


# =============================================================================
# TESTS FOR FIXED SIZE CHUNKER
# =============================================================================


class TestFixedSizeChunker:
    """Tests for FixedSizeChunker."""

    def test_basic_chunking(self):
        """Test basic fixed-size chunking."""
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=0)

        text = "0123456789" * 3  # 30 characters
        chunks = chunker.chunk(text)

        assert len(chunks) == 3
        assert chunks[0].content == "0123456789"
        assert chunks[1].content == "0123456789"
        assert chunks[2].content == "0123456789"

    def test_chunking_with_overlap(self):
        """Test chunking with overlap."""
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=3)

        text = "ABCDEFGHIJ" * 2  # 20 characters
        chunks = chunker.chunk(text)

        # With chunk_size=10, overlap=3, step=7
        # Chunk 1: 0-10 -> ABCDEFGHIJ
        # Chunk 2: 7-17 -> HIJABCDEFG
        # Chunk 3: 14-20 -> FGHIJ (remaining)
        assert len(chunks) == 3

    def test_empty_text(self):
        """Test chunking empty text."""
        chunker = FixedSizeChunker()

        chunks = chunker.chunk("")

        assert chunks == []

    def test_whitespace_only(self):
        """Test chunking whitespace-only text."""
        chunker = FixedSizeChunker()

        chunks = chunker.chunk("   \n\t  ")

        assert chunks == []

    def test_invalid_overlap(self):
        """Test that overlap >= chunk_size raises error."""
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=10, chunk_overlap=10)

        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=10, chunk_overlap=15)

    def test_metadata_preservation(self):
        """Test that metadata is preserved in chunks."""
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=0)

        chunks = chunker.chunk(
            "0123456789",
            metadata={"source": "test.txt", "page": 1},
        )

        assert len(chunks) == 1
        assert chunks[0].metadata["source"] == "test.txt"
        assert chunks[0].metadata["page"] == 1
        assert chunks[0].metadata["chunk_method"] == "fixed_size"

    def test_chunk_position_metadata(self):
        """Test that chunk position metadata is added."""
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=0)

        text = "A" * 25  # 25 characters -> 3 chunks
        chunks = chunker.chunk(text)

        assert chunks[0].metadata["chunk_index"] == 0
        assert chunks[0].metadata["chunk_start"] == 0
        assert chunks[1].metadata["chunk_index"] == 1
        assert chunks[2].metadata["chunk_index"] == 2


# =============================================================================
# TESTS FOR RECURSIVE TEXT CHUNKER
# =============================================================================


class TestRecursiveTextChunker:
    """Tests for RecursiveTextChunker."""

    def test_paragraph_splitting(self):
        """Test splitting on paragraph boundaries."""
        chunker = RecursiveTextChunker(chunk_size=50, chunk_overlap=10)

        text = """First paragraph here.

Second paragraph with more content.

Third paragraph to end."""

        chunks = chunker.chunk(text)

        # Should produce multiple chunks split on paragraphs
        assert len(chunks) >= 1
        assert all(c.metadata["chunk_method"] == "recursive" for c in chunks)

    def test_sentence_splitting(self):
        """Test splitting on sentence boundaries when paragraphs are too long."""
        chunker = RecursiveTextChunker(chunk_size=30, chunk_overlap=5)

        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.chunk(text)

        # Should produce chunks that respect sentence boundaries
        assert len(chunks) >= 1

    def test_custom_separators(self):
        """Test with custom separators."""
        chunker = RecursiveTextChunker(
            chunk_size=20,
            chunk_overlap=0,
            separators=["|", " "],
        )

        text = "Part A|Part B|Part C"
        chunks = chunker.chunk(text)

        assert len(chunks) >= 1

    def test_empty_text(self):
        """Test chunking empty text."""
        chunker = RecursiveTextChunker()

        chunks = chunker.chunk("")

        assert chunks == []

    def test_small_text(self):
        """Test text smaller than chunk size."""
        chunker = RecursiveTextChunker(chunk_size=1000, chunk_overlap=100)

        text = "Short text."
        chunks = chunker.chunk(text)

        assert len(chunks) == 1
        assert chunks[0].content == "Short text."

    def test_metadata_preservation(self):
        """Test that base metadata is preserved."""
        chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=0)

        chunks = chunker.chunk(
            "Test content here.",
            metadata={"file": "doc.md", "author": "Test"},
        )

        assert chunks[0].metadata["file"] == "doc.md"
        assert chunks[0].metadata["author"] == "Test"
        assert "chunk_index" in chunks[0].metadata

    def test_whitespace_handling(self):
        """Test handling of whitespace."""
        chunker = RecursiveTextChunker(
            chunk_size=20,
            chunk_overlap=0,
            strip_whitespace=True,
        )

        text = "  Word1   Word2   Word3  "
        chunks = chunker.chunk(text)

        # Should have stripped whitespace
        assert all(not c.content.startswith(" ") for c in chunks)


# =============================================================================
# TESTS FOR SENTENCE CHUNKER
# =============================================================================


class TestSentenceChunker:
    """Tests for SentenceChunker."""

    def test_basic_sentence_chunking(self):
        """Test basic sentence-based chunking."""
        chunker = SentenceChunker(
            target_chunk_size=100,
            min_sentences=1,
            max_sentences=3,
        )

        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunker.chunk(text)

        assert len(chunks) >= 1
        assert all(c.metadata["chunk_method"] == "sentence" for c in chunks)

    def test_sentence_count_metadata(self):
        """Test that sentence count is recorded in metadata."""
        chunker = SentenceChunker(
            target_chunk_size=200,
            min_sentences=1,
            max_sentences=10,
        )

        text = "Sentence one. Sentence two. Sentence three."
        chunks = chunker.chunk(text)

        assert len(chunks) == 1
        assert chunks[0].metadata["sentence_count"] == 3

    def test_max_sentences_limit(self):
        """Test that max_sentences limit is respected."""
        chunker = SentenceChunker(
            target_chunk_size=10000,  # Very large
            min_sentences=1,
            max_sentences=2,
        )

        text = "S1. S2. S3. S4. S5."
        chunks = chunker.chunk(text)

        # Should have multiple chunks due to max_sentences=2
        assert len(chunks) >= 2
        for chunk in chunks:
            assert chunk.metadata["sentence_count"] <= 2

    def test_overlap_sentences(self):
        """Test sentence overlap between chunks."""
        chunker = SentenceChunker(
            target_chunk_size=20,
            chunk_overlap_sentences=1,
            min_sentences=1,
            max_sentences=2,
        )

        text = "One. Two. Three. Four. Five."
        chunks = chunker.chunk(text)

        # Should have overlap
        assert len(chunks) >= 2

    def test_empty_text(self):
        """Test chunking empty text."""
        chunker = SentenceChunker()

        chunks = chunker.chunk("")

        assert chunks == []

    def test_single_sentence(self):
        """Test chunking single sentence."""
        chunker = SentenceChunker()

        chunks = chunker.chunk("Just one sentence here.")

        assert len(chunks) == 1

    def test_question_exclamation_splits(self):
        """Test splitting on questions and exclamations."""
        chunker = SentenceChunker(
            target_chunk_size=50,
            min_sentences=1,
            max_sentences=1,
        )

        text = "What is this? It is a test! Yes it is."
        chunks = chunker.chunk(text)

        # Should split into separate sentences
        assert len(chunks) == 3


# =============================================================================
# TESTS FOR FACTORY FUNCTIONS
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory and convenience functions."""

    def test_create_chunker_fixed(self):
        """Test creating fixed-size chunker via factory."""
        chunker = create_chunker("fixed", chunk_size=100, chunk_overlap=20)

        assert isinstance(chunker, FixedSizeChunker)

    def test_create_chunker_recursive(self):
        """Test creating recursive chunker via factory."""
        chunker = create_chunker("recursive", chunk_size=500)

        assert isinstance(chunker, RecursiveTextChunker)

    def test_create_chunker_sentence(self):
        """Test creating sentence chunker via factory."""
        chunker = create_chunker("sentence", target_chunk_size=200)

        assert isinstance(chunker, SentenceChunker)

    def test_create_chunker_unknown(self):
        """Test creating chunker with unknown strategy."""
        with pytest.raises(ValueError) as exc:
            create_chunker("unknown_strategy")

        assert "unknown_strategy" in str(exc.value).lower()

    def test_chunk_text_convenience(self):
        """Test chunk_text convenience function."""
        text = "This is a sample text for chunking. It has multiple sentences."

        chunks = chunk_text(
            text,
            chunk_size=50,
            chunk_overlap=10,
            strategy="recursive",
        )

        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_text_with_metadata(self):
        """Test chunk_text with metadata."""
        chunks = chunk_text(
            "Sample content",
            metadata={"source": "test"},
        )

        assert chunks[0].metadata["source"] == "test"


# =============================================================================
# TESTS FOR CHUNKING DOCUMENTS (BATCH)
# =============================================================================


class TestChunkDocuments:
    """Tests for batch document chunking."""

    def test_chunk_multiple_documents(self):
        """Test chunking multiple documents."""
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=0)

        documents = [
            {"content": "Document one content here.", "metadata": {"doc": 1}},
            {"content": "Document two content here.", "metadata": {"doc": 2}},
        ]

        chunks = chunker.chunk_documents(documents)

        # Should have chunks from both documents
        assert len(chunks) >= 2

    def test_chunk_documents_metadata_preserved(self):
        """Test that document metadata is preserved."""
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=0)

        documents = [
            {"content": "Content A", "metadata": {"file": "a.txt"}},
            {"content": "Content B", "metadata": {"file": "b.txt"}},
        ]

        chunks = chunker.chunk_documents(documents)

        # Find chunks from each document
        files = {c.metadata.get("file") for c in chunks}
        assert "a.txt" in files
        assert "b.txt" in files

    def test_chunk_documents_empty_list(self):
        """Test chunking empty document list."""
        chunker = RecursiveTextChunker()

        chunks = chunker.chunk_documents([])

        assert chunks == []


# =============================================================================
# TESTS FOR EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and special inputs."""

    def test_unicode_text(self):
        """Test chunking Unicode text."""
        chunker = FixedSizeChunker(chunk_size=20, chunk_overlap=0)

        text = "こんにちは世界" * 3  # Japanese "Hello World" repeated
        chunks = chunker.chunk(text)

        assert len(chunks) >= 1

    def test_mixed_newlines(self):
        """Test handling of mixed newline styles."""
        chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=0)

        text = "Line 1\nLine 2\r\nLine 3\rLine 4"
        chunks = chunker.chunk(text)

        # Should handle all newline styles
        assert len(chunks) >= 1

    def test_very_long_word(self):
        """Test handling of very long words."""
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=0)

        text = "A" * 50  # Single "word" longer than chunk size
        chunks = chunker.chunk(text)

        # Should split even the long word
        assert len(chunks) >= 5

    def test_code_like_content(self):
        """Test chunking code-like content."""
        chunker = RecursiveTextChunker(
            chunk_size=100,
            chunk_overlap=20,
            separators=["\n\n", "\n", "; ", " "],
        )

        code = """def hello():
    print("Hello")

def world():
    print("World")
"""
        chunks = chunker.chunk(code, metadata={"language": "python"})

        assert len(chunks) >= 1
        assert all(c.metadata["language"] == "python" for c in chunks)


# =============================================================================
# RUN TESTS
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
