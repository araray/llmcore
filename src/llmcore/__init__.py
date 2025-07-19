# src/llmcore/__init__.py
"""
LLMCore - A comprehensive library for LLM interaction, session management, and RAG.

This library provides a unified interface for working with multiple LLM providers,
managing conversation sessions, implementing Retrieval Augmented Generation (RAG),
and now supporting hierarchical memory including episodic memory for agent experiences.
"""

from .api import LLMCore
from .models import (
    ChatSession,
    Message,
    Role,
    ContextDocument,
    ContextItem,
    ContextItemType,
    ContextPreset,
    ContextPresetItem,
    Episode,           # Added Episode
    EpisodeType,       # Added EpisodeType
    AgentState,        # Added AgentState
    AgentTask,         # Added AgentTask
    ModelDetails,
    Tool,
    ToolCall,
    ToolResult,
    ContextPreparationDetails
)
from .exceptions import (
    LLMCoreError,
    ConfigError,
    ProviderError,
    StorageError,
    SessionStorageError,
    VectorStorageError,
    SessionNotFoundError,
    ContextError,
    ContextLengthError,
    EmbeddingError
)
from .storage import StorageManager
from .agents import AgentManager, ToolManager  # Added new agent components

# Version information
try:
    from .get_version import get_version
    __version__ = get_version()
except ImportError:
    __version__ = "unknown"

__all__ = [
    # Core API
    "LLMCore",

    # Data Models
    "ChatSession",
    "Message",
    "Role",
    "ContextDocument",
    "ContextItem",
    "ContextItemType",
    "ContextPreset",
    "ContextPresetItem",
    "Episode",           # Added Episode
    "EpisodeType",       # Added EpisodeType
    "AgentState",        # Added AgentState
    "AgentTask",         # Added AgentTask
    "ModelDetails",
    "Tool",
    "ToolCall",
    "ToolResult",
    "ContextPreparationDetails",

    # Exceptions
    "LLMCoreError",
    "ConfigError",
    "ProviderError",
    "StorageError",
    "SessionStorageError",
    "VectorStorageError",
    "SessionNotFoundError",
    "ContextError",
    "ContextLengthError",
    "EmbeddingError",

    # Storage
    "StorageManager",

    # Agents
    "AgentManager",      # Added AgentManager
    "ToolManager",       # Added ToolManager

    # Version
    "__version__",
]
