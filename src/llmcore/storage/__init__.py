# src/llmcore/storage/__init__.py
"""
Storage management module for the LLMCore library.

This package handles the persistence and retrieval of chat sessions,
context presets, episodic memory, and vector embeddings for RAG.
It provides both session storage (for conversations and episodes) and
vector storage (for semantic memory) backends.
"""

# Import key storage components for easier access
from .manager import StorageManager
from .base_session import BaseSessionStorage
from .base_vector import BaseVectorStorage

# Import concrete implementations
from .json_session import JsonSessionStorage
from .sqlite_session import SqliteSessionStorage
from .postgres_session_storage import PostgresSessionStorage
from .pgvector_storage import PgVectorStorage
from .chromadb_vector import ChromaVectorStorage

__all__ = [
    "StorageManager",
    "BaseSessionStorage",
    "BaseVectorStorage",
    "JsonSessionStorage",
    "SqliteSessionStorage",
    "PostgresSessionStorage",
    "PgVectorStorage",
    "ChromaVectorStorage",
]
