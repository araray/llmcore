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

# =============================================================================
# SANDBOX EXPORTS (NEW in v0.26.0)
# =============================================================================
# These exports provide access to the sandbox system for isolated agent execution.
# Import these directly from llmcore or from llmcore.agents.sandbox
from .agents import (
    SANDBOX_TOOL_IMPLEMENTATIONS,
    SANDBOX_TOOL_SCHEMAS,
    AgentManager,
    # Sandbox providers
    DockerSandboxProvider,
    # Sandbox utilities
    EphemeralResourceManager,
    ExecutionResult,
    FileInfo,
    OutputTracker,
    SandboxAccessDenied,
    SandboxAccessLevel,
    SandboxAgentMixin,
    SandboxCleanupError,
    SandboxConfig,
    SandboxConnectionError,
    SandboxContext,
    # Sandbox exceptions
    SandboxError,
    SandboxExecutionError,
    SandboxInitializationError,
    # Integration bridge
    SandboxIntegration,
    SandboxMode,
    SandboxProvider,
    # Sandbox core classes
    SandboxRegistry,
    SandboxRegistryConfig,
    SandboxResourceError,
    SandboxStatus,
    SandboxTimeoutError,
    ToolManager,
    VMSandboxProvider,
    clear_active_sandbox,
    create_registry_config,
    get_active_sandbox,
    get_sandbox_tool_definitions,
    # Sandbox configuration helpers
    load_sandbox_config,
    register_sandbox_tools,
    # Sandbox tool management
    set_active_sandbox,
)
from .api import LLMCore
from .exceptions import (
    ConfigError,
    ContextError,
    ContextLengthError,
    EmbeddingError,
    LLMCoreError,
    ProviderError,
    SessionNotFoundError,
    SessionStorageError,
    StorageError,
    VectorStorageError,
)

# =============================================================================
# MODEL CARD LIBRARY EXPORTS (Phase 4 Integration)
# =============================================================================
# These exports provide access to the Model Card Library for model metadata.
from .model_cards import (
    AnthropicExtension,
    ArchitectureType,
    DeepSeekExtension,
    EmbeddingConfig,
    GoogleExtension,
    MistralExtension,
    # Schema components
    ModelArchitecture,
    ModelCapabilities,
    # Core models
    ModelCard,
    # Registry
    ModelCardRegistry,
    ModelCardSummary,
    ModelContext,
    ModelLifecycle,
    ModelPricing,
    ModelStatus,
    ModelType,
    # Provider extensions
    OllamaExtension,
    OpenAIExtension,
    # Enums
    Provider,
    QwenExtension,
    TokenPricing,
    XAIExtension,
    clear_model_card_cache,
    get_model_card,
    get_model_card_registry,
)
from .models import (
    AgentState,
    AgentTask,
    ChatSession,
    ContextDocument,
    ContextItem,
    ContextItemType,
    ContextPreparationDetails,
    ContextPreset,
    ContextPresetItem,
    CostEstimate,
    Episode,
    EpisodeType,
    Message,
    ModelDetails,
    ModelValidationResult,
    PullProgress,
    PullResult,
    Role,
    SessionTokenStats,
    Tool,
    ToolCall,
    ToolResult,
)
from .storage import StorageManager

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
    "ModelValidationResult",
    "PullProgress",
    "PullResult",
    "Tool",
    "ToolCall",
    "ToolResult",
    "ContextPreparationDetails",
    # ==========================================================================
    # Model Card Library
    # ==========================================================================
    # Core models
    "ModelCard",
    "ModelCardSummary",
    # Schema components
    "ModelArchitecture",
    "ModelContext",
    "ModelCapabilities",
    "ModelPricing",
    "ModelLifecycle",
    "TokenPricing",
    # Enums
    "Provider",
    "ModelType",
    "ModelStatus",
    "ArchitectureType",
    # Provider extensions
    "OllamaExtension",
    "OpenAIExtension",
    "AnthropicExtension",
    "GoogleExtension",
    "DeepSeekExtension",
    "QwenExtension",
    "MistralExtension",
    "XAIExtension",
    "EmbeddingConfig",
    # Registry
    "ModelCardRegistry",
    "get_model_card_registry",
    "get_model_card",
    "clear_model_card_cache",
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
    # Statistics Models
    "SessionTokenStats",
    "CostEstimate",
]
