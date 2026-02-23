# src/llmcore/__init__.py
"""
LLMCore — Comprehensive AI Agent Infrastructure Library.

Provides a unified interface for working with multiple LLM providers,
managing conversation sessions, implementing Retrieval Augmented Generation (RAG),
supporting hierarchical memory (including episodic memory for agent experiences),
and orchestrating autonomous agents via the Darwin cognitive cycle.

Key subsystems:
    - **Providers**: OpenAI, Anthropic, Google/Gemini, Ollama + OpenAI-compatible
      (DeepSeek, Mistral, xAI, Groq, Together)
    - **Agents**: 8-phase Darwin cognitive cycle, personas, HITL, sandboxing
    - **Memory**: Volatile / Session / Semantic / Episodic tiers
    - **Storage**: SQLite, PostgreSQL, ChromaDB, pgvector backends
    - **Embedding**: OpenAI, Google, Ollama, SentenceTransformers, Cohere, VoyageAI
    - **Context**: Adaptive Context Synthesis with compression & prioritization
    - **Autonomous**: Goal management, heartbeat scheduling, resource monitoring
    - **Observability**: Cost tracking, metrics, structured events, tracing

Most classes can be imported directly from ``llmcore``::

    from llmcore import (
        LLMCore,
        AgentManager,
        ContextSynthesizer,
        GoalManager,
        MemoryManager,
        StorageManager,
    )

For advanced or less-common types, import from sub-packages::

    from llmcore.agents.hitl import HITLManager
    from llmcore.autonomous import AutonomousScheduler

Version: 0.41.1
"""

from importlib.metadata import PackageNotFoundError, version

# =============================================================================
# AGENTS (Darwin Layer 2)
# =============================================================================
from .agents import (
    # Sandbox: core classes
    SANDBOX_TOOL_IMPLEMENTATIONS,
    SANDBOX_TOOL_SCHEMAS,
    # Resilience
    AgentCircuitBreaker,
    # Core managers
    AgentManager,
    AgentMode,
    # Persona system
    AgentPersona,
    # Single agent
    AgentResult,
    # Routing
    CapabilityChecker,
    # Cognitive cycle
    CognitiveCycle,
    # Memory integration
    CognitiveMemoryIntegrator,
    CognitivePhase,
    DockerSandboxProvider,
    EnhancedAgentManager,
    EnhancedAgentState,
    EphemeralResourceManager,
    ExecutionResult,
    FileInfo,
    OutputTracker,
    PersonaManager,
    # Prompts
    PromptComposer,
    PromptRegistry,
    PromptTemplate,
    # Context
    RAGContextFilter,
    SandboxAccessDenied,
    SandboxAccessLevel,
    SandboxAgentMixin,
    SandboxCleanupError,
    SandboxConfig,
    SandboxConnectionError,
    SandboxContext,
    SandboxError,
    SandboxExecutionError,
    SandboxInitializationError,
    SandboxIntegration,
    SandboxMode,
    SandboxProvider,
    SandboxRegistry,
    SandboxRegistryConfig,
    SandboxResourceError,
    SandboxStatus,
    SandboxTimeoutError,
    SingleAgentMode,
    # Tools
    ToolManager,
    VMSandboxProvider,
    clear_active_sandbox,
    create_registry_config,
    get_active_sandbox,
    get_sandbox_tool_definitions,
    load_sandbox_config,
    register_sandbox_tools,
    set_active_sandbox,
)

# =============================================================================
# HITL (via agents — tightly coupled to agent execution)
# =============================================================================
from .agents.hitl import (
    HITLConfig,
    HITLManager,
    RiskAssessor,
    RiskLevel,
)

# =============================================================================
# CORE API
# =============================================================================
from .api import LLMCore

# =============================================================================
# AUTONOMOUS
# =============================================================================
from .autonomous import (
    AutonomousScheduler,
    AutonomousState,
    ConstraintViolation,
    Escalation,
    EscalationLevel,
    EscalationManager,
    EscalationReason,
    Goal,
    GoalManager,
    GoalPriority,
    GoalStatus,
    GoalStore,
    HeartbeatManager,
    HeartbeatTask,
    ResourceConstraints,
    ResourceMonitor,
    ResourceStatus,
    Skill,
    SkillLoader,
    SkillMetadata,
    StateManager,
    SuccessCriterion,
    TaskPriority,
    heartbeat_task,
)

# =============================================================================
# CONTEXT (Adaptive Context Synthesis)
# =============================================================================
from .context import (
    ContentPrioritizer,
    ContextChunk,
    ContextCompressor,
    ContextSource,
    ContextSynthesizer,
    EstimateCounter,
    SynthesizedContext,
    TiktokenCounter,
    TokenCounter,
)

# =============================================================================
# EMBEDDING
# =============================================================================
from .embedding import (
    BaseEmbeddingModel,
    EmbeddingCache,
    EmbeddingCacheConfig,
    EmbeddingManager,
    SentenceTransformerEmbedding,
    create_embedding_cache,
)

# =============================================================================
# EXCEPTIONS
# =============================================================================
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
# INGESTION
# =============================================================================
from .ingestion import (
    Chunk,
    ChunkingConfig,
    ChunkingStrategy,
    FixedSizeChunker,
    RecursiveTextChunker,
    SentenceChunker,
)

# =============================================================================
# INTEGRATION
# =============================================================================
from .integration import (
    LLMCoreVectorClient,
    LLMCoreVectorClientConfig,
    VectorClientProtocol,
)

# =============================================================================
# MEMORY
# =============================================================================
from .memory import MemoryManager

# =============================================================================
# MODEL CARD LIBRARY
# =============================================================================
from .model_cards import (
    AnthropicExtension,
    ArchitectureType,
    DeepSeekExtension,
    EmbeddingConfig,
    GoogleExtension,
    MistralExtension,
    ModelArchitecture,
    ModelCapabilities,
    ModelCard,
    ModelCardRegistry,
    ModelCardSummary,
    ModelContext,
    ModelLifecycle,
    ModelPricing,
    ModelStatus,
    ModelType,
    OllamaExtension,
    OpenAIExtension,
    Provider,
    QwenExtension,
    TokenPricing,
    XAIExtension,
    clear_model_card_cache,
    get_model_card,
    get_model_card_registry,
)

# =============================================================================
# DATA MODELS
# =============================================================================
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

# =============================================================================
# OBSERVABILITY
# =============================================================================
from .observability import (
    PRICING_DATA,
    CostAnalyzer,
    CostTracker,
    CostTrackingConfig,
    MetricsRegistry,
    ObservabilityLogger,
    UsageRecord,
    UsageSummary,
    create_cost_tracker,
    get_price_per_million_tokens,
)

# =============================================================================
# PROVIDERS
# =============================================================================
from .providers.base import BaseProvider
from .providers.manager import ProviderManager

# =============================================================================
# SESSIONS
# =============================================================================
from .sessions.manager import SessionManager

# =============================================================================
# STORAGE
# =============================================================================
from .storage import StorageManager

# =============================================================================
# VERSION
# =============================================================================
try:
    __version__ = version("llmcore")
except PackageNotFoundError:
    from .get_version import _get_version_from_pyproject

    __version__ = _get_version_from_pyproject()


# =============================================================================
# PUBLIC API (__all__)
# =============================================================================
__all__ = [
    # -- Core API --
    "LLMCore",
    # -- Providers --
    "BaseProvider",
    "ProviderManager",
    # -- Data Models --
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
    "SessionTokenStats",
    "CostEstimate",
    # -- Exceptions --
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
    # -- Storage --
    "StorageManager",
    # -- Embedding --
    "BaseEmbeddingModel",
    "EmbeddingManager",
    "EmbeddingCache",
    "EmbeddingCacheConfig",
    "SentenceTransformerEmbedding",
    "create_embedding_cache",
    # -- Agents --
    "AgentManager",
    "AgentMode",
    "EnhancedAgentManager",
    "SingleAgentMode",
    "AgentResult",
    "CognitiveCycle",
    "CognitivePhase",
    "EnhancedAgentState",
    "ToolManager",
    "AgentPersona",
    "PersonaManager",
    "PromptRegistry",
    "PromptTemplate",
    "PromptComposer",
    "CognitiveMemoryIntegrator",
    "RAGContextFilter",
    "AgentCircuitBreaker",
    "CapabilityChecker",
    # -- HITL --
    "HITLManager",
    "HITLConfig",
    "RiskAssessor",
    "RiskLevel",
    # -- Sandbox --
    "SandboxRegistry",
    "SandboxRegistryConfig",
    "SandboxConfig",
    "SandboxProvider",
    "SandboxMode",
    "SandboxAccessLevel",
    "SandboxStatus",
    "ExecutionResult",
    "FileInfo",
    "DockerSandboxProvider",
    "VMSandboxProvider",
    "EphemeralResourceManager",
    "OutputTracker",
    "load_sandbox_config",
    "create_registry_config",
    "SandboxError",
    "SandboxInitializationError",
    "SandboxExecutionError",
    "SandboxTimeoutError",
    "SandboxAccessDenied",
    "SandboxResourceError",
    "SandboxConnectionError",
    "SandboxCleanupError",
    "set_active_sandbox",
    "clear_active_sandbox",
    "get_active_sandbox",
    "SANDBOX_TOOL_IMPLEMENTATIONS",
    "SANDBOX_TOOL_SCHEMAS",
    "SandboxIntegration",
    "SandboxContext",
    "SandboxAgentMixin",
    "register_sandbox_tools",
    "get_sandbox_tool_definitions",
    # -- Memory --
    "MemoryManager",
    # -- Sessions --
    "SessionManager",
    # -- Autonomous --
    "Goal",
    "GoalManager",
    "GoalPriority",
    "GoalStatus",
    "GoalStore",
    "SuccessCriterion",
    "Escalation",
    "EscalationLevel",
    "EscalationManager",
    "EscalationReason",
    "HeartbeatManager",
    "HeartbeatTask",
    "heartbeat_task",
    "ResourceMonitor",
    "ResourceConstraints",
    "ResourceStatus",
    "ConstraintViolation",
    "SkillLoader",
    "Skill",
    "SkillMetadata",
    "AutonomousState",
    "StateManager",
    "AutonomousScheduler",
    "TaskPriority",
    # -- Context --
    "ContextSynthesizer",
    "ContextSource",
    "ContextChunk",
    "SynthesizedContext",
    "TokenCounter",
    "TiktokenCounter",
    "EstimateCounter",
    "ContextCompressor",
    "ContentPrioritizer",
    # -- Ingestion --
    "Chunk",
    "ChunkingStrategy",
    "ChunkingConfig",
    "FixedSizeChunker",
    "RecursiveTextChunker",
    "SentenceChunker",
    # -- Integration --
    "LLMCoreVectorClient",
    "LLMCoreVectorClientConfig",
    "VectorClientProtocol",
    # -- Model Card Library --
    "ModelCard",
    "ModelCardSummary",
    "ModelArchitecture",
    "ModelContext",
    "ModelCapabilities",
    "ModelPricing",
    "ModelLifecycle",
    "TokenPricing",
    "Provider",
    "ModelType",
    "ModelStatus",
    "ArchitectureType",
    "OllamaExtension",
    "OpenAIExtension",
    "AnthropicExtension",
    "GoogleExtension",
    "DeepSeekExtension",
    "QwenExtension",
    "MistralExtension",
    "XAIExtension",
    "EmbeddingConfig",
    "ModelCardRegistry",
    "get_model_card_registry",
    "get_model_card",
    "clear_model_card_cache",
    # -- Observability --
    "CostTracker",
    "CostTrackingConfig",
    "UsageRecord",
    "UsageSummary",
    "PRICING_DATA",
    "create_cost_tracker",
    "get_price_per_million_tokens",
    "MetricsRegistry",
    "ObservabilityLogger",
    "CostAnalyzer",
    # -- Version --
    "__version__",
]
