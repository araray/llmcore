# src/llmcore/__init__.py
"""
LLMCore: A unified, flexible, and extensible interface for interacting
with various Large Language Models (LLMs).

This library provides a robust foundation for building applications that
require LLM-driven chat capabilities, session management, context handling,
and Retrieval Augmented Generation (RAG).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("llmcore")
except PackageNotFoundError:
    from .get_version import _get_version_from_pyproject
    __version__ = _get_version_from_pyproject()

# Import the main LLMCore class from api.py
from .api import LLMCore
from .exceptions import (ConfigError, ContextError,  # MCPError removed
                         ContextLengthError, EmbeddingError, LLMCoreError,
                         ProviderError, SessionNotFoundError,
                         SessionStorageError, StorageError, VectorStorageError)
# Import core models and exceptions for easier access by library users
# as per the API specification.
from .models import (ChatSession, ContextDocument, ContextItem,
                     ContextItemType, Message, Role, ContextPreparationDetails, # Added ContextPreparationDetails
                     ContextPreset, ContextPresetItem) # Added ContextPreset and ContextPresetItem

# Expose specific elements for the public API
__all__ = [
    "__version__",
    "LLMCore", # Expose the main class
    # Models
    "Role",
    "Message",
    "ChatSession",
    "ContextDocument",
    "ContextItem",
    "ContextItemType",
    "ContextPreparationDetails", # Added ContextPreparationDetails to __all__
    "ContextPreset",             # Added ContextPreset to __all__
    "ContextPresetItem",         # Added ContextPresetItem to __all__
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
    # MCPError removed
]

# Initialize logging for the library (optional, can be configured by the application)
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
