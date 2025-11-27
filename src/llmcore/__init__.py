# src/llmcore/__init__.py
"""
LLMCore - A comprehensive library for LLM interaction, session management, and RAG.

This library provides a unified interface for working with multiple LLM providers,
managing conversation sessions, implementing Retrieval Augmented Generation (RAG),
and supporting hierarchical memory including episodic memory for agent experiences.

UPDATED: Pure library mode - removed service-oriented component exports.
         AgentManager and ToolManager remain available for advanced users.
"""

from importlib.metadata import PackageNotFoundError, version

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
    Episode,
    EpisodeType,
    AgentState,
    AgentTask,
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
from .agents import AgentManager, ToolManager

from .logging_config import (
    configure_logging,
    get_log_file_path,
    set_console_level,
    set_file_level,
    set_component_level,
    disable_console_logging,
    enable_console_logging,
)

try:
    __version__ = version("llmcore")
except PackageNotFoundError:
    from .get_version import _get_version_from_pyproject
    __version__ = _get_version_from_pyproject()


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
    "Episode",
    "EpisodeType",
    "AgentState",
    "AgentTask",
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

    # Agents (available for advanced library users)
    "AgentManager",
    "ToolManager",

    # Version
    "__version__",
]
