# src/llmcore/__init__.py
"""
LLMCore: A unified, flexible, and extensible interface for interacting
with various Large Language Models (LLMs).

This library provides a robust foundation for building applications that
require LLM-driven chat capabilities, session management, context handling,
and Retrieval Augmented Generation (RAG).
"""

__version__ = "0.5.0" # Version will be updated as features are added

# Import core models and exceptions for easier access by library users
# as per the API specification.
from .models import (
    Role,
    Message,
    ChatSession,
    ContextDocument
)
from .exceptions import (
    LLMCoreError,
    ConfigError,
    ProviderError,
    StorageError,
    SessionStorageError,
    VectorStorageError,
    ContextError,
    ContextLengthError,
    EmbeddingError,
    SessionNotFoundError,
    MCPError # Added MCPError
)

# Import the main LLMCore class from api.py
from .api import LLMCore

# Expose specific elements for the public API
__all__ = [
    "__version__",
    "LLMCore", # Expose the main class
    # Models
    "Role",
    "Message",
    "ChatSession",
    "ContextDocument",
    # Exceptions
    "LLMCoreError",
    "ConfigError",
    "ProviderError",
    "StorageError",
    "SessionStorageError",
    "VectorStorageError",
    "ContextError",
    "ContextLengthError",
    "EmbeddingError",
    "SessionNotFoundError",
    "MCPError", # Added MCPError
]

# Initialize logging for the library (optional, can be configured by the application)
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
