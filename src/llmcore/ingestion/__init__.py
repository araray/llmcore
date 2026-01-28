# src/llmcore/ingestion/__init__.py
"""
LLMCore Ingestion Module.

Phase 3 (SYMBIOSIS): This module provides high-level document ingestion
capabilities for RAG pipelines, including chunking strategies and
document processing utilities.

The goal is to centralize ingestion logic in LLMCore so that external
engines (like SemantiScan) can delegate document processing here.

Usage:
    from llmcore.ingestion import (
        ChunkingStrategy,
        FixedSizeChunker,
        SemanticChunker,
        RecursiveTextChunker,
        DocumentIngestionPipeline,
    )

    # Create a chunker
    chunker = RecursiveTextChunker(chunk_size=1000, chunk_overlap=200)

    # Chunk text
    chunks = chunker.chunk(text, metadata={"source": "file.txt"})

    # Or use the full pipeline
    pipeline = DocumentIngestionPipeline(llmcore_instance, chunker)
    await pipeline.ingest_documents(documents)
"""

from .chunking import (
    Chunk,
    ChunkingStrategy,
    FixedSizeChunker,
    RecursiveTextChunker,
    SentenceChunker,
    ChunkingConfig,
)

__all__ = [
    # Data models
    "Chunk",
    # Strategies
    "ChunkingStrategy",
    "FixedSizeChunker",
    "RecursiveTextChunker",
    "SentenceChunker",
    # Configuration
    "ChunkingConfig",
]
