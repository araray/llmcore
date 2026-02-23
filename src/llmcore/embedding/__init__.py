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
    DiskCache,
    EmbeddingCache,
    EmbeddingCacheConfig,
    LRUCache,
    create_embedding_cache,
)
from .manager import EmbeddingManager
from .sentence_transformer import SentenceTransformerEmbedding

# Optional providers — imported lazily to avoid hard dependency on their SDKs.
# Users should install the relevant package (cohere, voyageai) before use.
try:
    from .cohere import CohereEmbedding
except ImportError:
    CohereEmbedding = None  # type: ignore[assignment,misc]

try:
    from .voyageai import VoyageAIEmbedding
except ImportError:
    VoyageAIEmbedding = None  # type: ignore[assignment,misc]

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
    # Optional providers (may be None if SDK not installed)
    "CohereEmbedding",
    "VoyageAIEmbedding",
]
