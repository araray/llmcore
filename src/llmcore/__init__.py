# src/llmcore/__init__.py
"""
LLMCore - A comprehensive library for LLM interaction, session management, and RAG.

This library provides a unified interface for working with multiple LLM providers,
managing conversation sessions, implementing Retrieval Augmented Generation (RAG),
and supporting hierarchical memory including episodic memory for agent experiences.

UPDATED v0.26.0: Added sandbox system for isolated agent code execution.
                 AgentManager now supports optional sandbox integration.
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

# =============================================================================
# SANDBOX EXPORTS (NEW in v0.26.0)
# =============================================================================
# These exports provide access to the sandbox system for isolated agent execution.
# Import these directly from llmcore or from llmcore.agents.sandbox

from .agents import (
    # Sandbox core classes
    SandboxRegistry,
    SandboxRegistryConfig,
    SandboxConfig,
    SandboxProvider,
    SandboxMode,
    SandboxAccessLevel,
    SandboxStatus,
    ExecutionResult,
    FileInfo,

    # Sandbox providers
    DockerSandboxProvider,
    VMSandboxProvider,

    # Sandbox utilities
    EphemeralResourceManager,
    OutputTracker,

    # Sandbox configuration helpers
    load_sandbox_config,
    create_registry_config,

    # Sandbox exceptions
    SandboxError,
    SandboxInitializationError,
    SandboxExecutionError,
    SandboxTimeoutError,
    SandboxAccessDenied,
    SandboxResourceError,
    SandboxConnectionError,
    SandboxCleanupError,

    # Sandbox tool management
    set_active_sandbox,
    clear_active_sandbox,
    get_active_sandbox,
    SANDBOX_TOOL_IMPLEMENTATIONS,
    SANDBOX_TOOL_SCHEMAS,

    # Integration bridge
    SandboxIntegration,
    SandboxContext,
    SandboxAgentMixin,
    register_sandbox_tools,
    get_sandbox_tool_definitions,
)

try:
    __version__ = version("llmcore")
except PackageNotFoundError:
    from .get_version import _get_version_from_pyproject
    __version__ = _get_version_from_pyproject()


__all__ = [
    # ==========================================================================
    # Core API
    # ==========================================================================
    "LLMCore",

    # ==========================================================================
    # Data Models
    # ==========================================================================
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

    # ==========================================================================
    # Core Exceptions
    # ==========================================================================
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

    # ==========================================================================
    # Storage
    # ==========================================================================
    "StorageManager",

    # ==========================================================================
    # Agents (Core)
    # ==========================================================================
    "AgentManager",
    "ToolManager",

    # ==========================================================================
    # Sandbox System (NEW in v0.26.0)
    # ==========================================================================
    # Core classes
    "SandboxRegistry",
    "SandboxRegistryConfig",
    "SandboxConfig",
    "SandboxProvider",
    "SandboxMode",
    "SandboxAccessLevel",
    "SandboxStatus",
    "ExecutionResult",
    "FileInfo",

    # Providers
    "DockerSandboxProvider",
    "VMSandboxProvider",

    # Utilities
    "EphemeralResourceManager",
    "OutputTracker",

    # Configuration
    "load_sandbox_config",
    "create_registry_config",

    # Sandbox Exceptions
    "SandboxError",
    "SandboxInitializationError",
    "SandboxExecutionError",
    "SandboxTimeoutError",
    "SandboxAccessDenied",
    "SandboxResourceError",
    "SandboxConnectionError",
    "SandboxCleanupError",

    # Tool Management
    "set_active_sandbox",
    "clear_active_sandbox",
    "get_active_sandbox",
    "SANDBOX_TOOL_IMPLEMENTATIONS",
    "SANDBOX_TOOL_SCHEMAS",

    # Integration Bridge
    "SandboxIntegration",
    "SandboxContext",
    "SandboxAgentMixin",
    "register_sandbox_tools",
    "get_sandbox_tool_definitions",

    # ==========================================================================
    # Version
    # ==========================================================================
    "__version__",
]
