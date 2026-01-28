# src/llmcore/embedding/__init__.py
"""
Embedding model management for the LLMCore library.

This package provides a consistent interface for generating text embeddings,
which are crucial for Retrieval Augmented Generation (RAG) and other
similarity-based tasks. It supports various embedding models,
both local and service-based.

Key Components:
- EmbeddingManager: Central manager for embedding model instances
- EmbeddingCache: Two-tier caching (LRU + SQLite) for embeddings
- BaseEmbeddingModel: Protocol/base class for embedding providers

Usage:
    from llmcore.embedding import EmbeddingManager, EmbeddingCache

    # Create cache
    cache = EmbeddingCache(memory_size=10000)

    # Use with manager
    manager = EmbeddingManager(config)
    embedding = await manager.generate_embedding("text")
"""

# Import necessary classes to be available when importing the package
from .base import BaseEmbeddingModel
from .cache import (
    EmbeddingCache,
    EmbeddingCacheConfig,
    LRUCache,
    DiskCache,
    create_embedding_cache,
)
from .manager import EmbeddingManager
from .sentence_transformer import SentenceTransformerEmbedding

# Define what gets imported with 'from llmcore.embedding import *'
__all__ = [
    # Base classes
    "BaseEmbeddingModel",
    "SentenceTransformerEmbedding",
    # Manager
    "EmbeddingManager",
    # Cache
    "EmbeddingCache",
    "EmbeddingCacheConfig",
    "LRUCache",
    "DiskCache",
    "create_embedding_cache",
]
