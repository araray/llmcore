# src/llmcore/embedding/__init__.py
"""
Embedding model management for the LLMCore library.

This package provides a consistent interface for generating text embeddings,
which are crucial for Retrieval Augmented Generation (RAG) and other
similarity-based tasks. It supports various embedding models,
both local and service-based.
"""

# Import necessary classes to be available when importing the package
from .base import BaseEmbeddingModel
# Import other embedding models when they are created
# from .openai import OpenAIEmbedding
# from .google import GoogleAIEmbedding
from .manager import EmbeddingManager  # Import the manager
from .sentence_transformer import SentenceTransformerEmbedding

# Define what gets imported with 'from llmcore.embedding import *'
# Or just make them available for direct import: from llmcore.embedding import EmbeddingManager
__all__ = [
    "BaseEmbeddingModel",
    "SentenceTransformerEmbedding",
    # "OpenAIEmbedding",
    # "GoogleAIEmbedding",
    "EmbeddingManager", # Export the manager
]
