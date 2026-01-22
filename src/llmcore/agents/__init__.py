# src/llmcore/agents/__init__.py
"""
Darwin Layer 2 - Enhanced Agent System.

This package provides comprehensive agent management with the enhanced
cognitive cycle, persona system, and prompt library.

Key Components:
    - AgentManager: Original agent manager (preserved for backward compatibility)
    - EnhancedAgentManager: Darwin Layer 2 agent with 8-phase cognitive cycle
    - SingleAgentMode: Autonomous single-agent execution
    - CognitiveCycle: 8-phase reasoning system
    - PersonaManager: Agent personality customization
    - PromptRegistry: Centralized prompt management
    - CognitiveMemoryIntegrator: Enhanced memory integration

Example - Original (Still Works):
    >>> from llmcore.agents import AgentManager
    >>> manager = AgentManager(provider_manager, memory_manager, storage_manager)
    >>> result = await manager.run_agent_loop(task=task)

Example - Enhanced (New):
    >>> from llmcore.agents import EnhancedAgentManager, AgentMode
    >>> manager = EnhancedAgentManager(...)
    >>> result = await manager.run(
    ...     goal="Calculate factorial of 10",
    ...     mode=AgentMode.SINGLE
    ... )

References:
    - Technical Spec: Darwin Layer 2 (Sections 5.1-5.7)
    - Implementation Dossiers: Steps 2.1-2.11
"""

# =============================================================================
# CORE MANAGERS (Preserved for backward compatibility)
# =============================================================================
# =============================================================================
# COGNITIVE SYSTEM
# =============================================================================
from .cognitive import (
    ActInput,
    ActOutput,
    # Orchestrator
    CognitiveCycle,
    # Enums
    CognitivePhase,
    ConfidenceLevel,
    # Models
    CycleIteration,
    # State
    EnhancedAgentState,
    ExecutionStrategy,
    GoalClassification,
    # Goal Classification (NEW)
    GoalClassifier,
    GoalComplexity,
    GoalIntent,
    IterationStatus,
    ModelTier,
    ObserveInput,
    ObserveOutput,
    # Phase I/O Models (for advanced usage)
    PerceiveInput,
    PerceiveOutput,
    PlanInput,
    PlanOutput,
    ReflectInput,
    ReflectOutput,
    ThinkInput,
    ThinkOutput,
    UpdateInput,
    UpdateOutput,
    ValidateInput,
    ValidateOutput,
    ValidationResult,
    act_phase,
    classify_goal,
    is_trivial_goal,
    needs_clarification,
    observe_phase,
    # Individual phases (for advanced usage)
    perceive_phase,
    plan_phase,
    reflect_phase,
    think_phase,
    update_phase,
    validate_phase,
)

# =============================================================================
# CONTEXT MANAGEMENT (NEW - Phase 1)
# =============================================================================
from .context import (
    FilterStats,
    RAGContextFilter,
    RAGResult,
)
from .manager import AgentManager, AgentMode, EnhancedAgentManager

# =============================================================================
# MEMORY INTEGRATION
# =============================================================================
from .memory import CognitiveMemoryIntegrator

# =============================================================================
# PERSONA SYSTEM
# =============================================================================
from .persona import (
    AgentPersona,
    CommunicationPreferences,
    CommunicationStyle,
    DecisionMakingPreferences,
    PersonalityTrait,
    PersonaManager,
    PersonaTrait,
    PlanningDepth,
    PromptModifications,
    RiskTolerance,
)

# =============================================================================
# PROMPT LIBRARY
# =============================================================================
from .prompts import (
    PromptComposer,
    PromptRegistry,
    PromptTemplate,
    PromptVersion,
    TemplateLoader,
)

# =============================================================================
# RESILIENCE SYSTEM (NEW - Phase 1)
# =============================================================================
from .resilience import (
    AgentCircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerResult,
    CircuitState,
    TripReason,
    create_circuit_breaker,
)

# =============================================================================
# MODEL ROUTING (NEW - Phase 1)
# =============================================================================
from .routing import (
    Capability,
    CapabilityChecker,
    CapabilityIssue,
    CompatibilityResult,
    IssueSeverity,
)

# =============================================================================
# SANDBOX SYSTEM
# =============================================================================
from .sandbox import (
    SANDBOX_TOOL_IMPLEMENTATIONS,
    SANDBOX_TOOL_SCHEMAS,
    # Providers
    DockerSandboxProvider,
    # Utilities
    EphemeralResourceManager,
    ExecutionResult,
    FileInfo,
    OutputTracker,
    SandboxAccessDenied,
    SandboxAccessLevel,
    SandboxCleanupError,
    SandboxConfig,
    SandboxConnectionError,
    # Exceptions
    SandboxError,
    SandboxExecutionError,
    SandboxInitializationError,
    SandboxMode,
    SandboxProvider,
    # Core classes
    SandboxRegistry,
    SandboxRegistryConfig,
    SandboxResourceError,
    SandboxStatus,
    SandboxTimeoutError,
    VMSandboxProvider,
    clear_active_sandbox,
    create_registry_config,
    get_active_sandbox,
    # Configuration
    load_sandbox_config,
    # Tool management
    set_active_sandbox,
)

# =============================================================================
# SANDBOX INTEGRATION
# =============================================================================
from .sandbox_integration import (
    SandboxAgentMixin,
    SandboxContext,
    SandboxIntegration,
    get_sandbox_tool_definitions,
    register_sandbox_tools,
)

# =============================================================================
# SINGLE AGENT MODE
# =============================================================================
from .single_agent import AgentResult, IterationUpdate, SingleAgentMode
from .tools import ToolManager

# =============================================================================
# EXPORTS
# =============================================================================
__all__ = [
    # Core Managers
    "AgentManager",  # Original (preserved)
    "ToolManager",
    "EnhancedAgentManager",  # Darwin Layer 2
    "AgentMode",
    # Single Agent
    "SingleAgentMode",
    "AgentResult",
    "IterationUpdate",
    # Cognitive Cycle
    "CognitiveCycle",
    "EnhancedAgentState",
    "CognitivePhase",
    "IterationStatus",
    "ValidationResult",
    "ConfidenceLevel",
    "CycleIteration",
    # Cognitive Phases (advanced)
    "perceive_phase",
    "plan_phase",
    "think_phase",
    "validate_phase",
    "act_phase",
    "observe_phase",
    "reflect_phase",
    "update_phase",
    # Phase I/O (advanced)
    "PerceiveInput",
    "PerceiveOutput",
    "PlanInput",
    "PlanOutput",
    "ThinkInput",
    "ThinkOutput",
    "ValidateInput",
    "ValidateOutput",
    "ActInput",
    "ActOutput",
    "ObserveInput",
    "ObserveOutput",
    "ReflectInput",
    "ReflectOutput",
    "UpdateInput",
    "UpdateOutput",
    # Goal Classification (NEW)
    "GoalClassifier",
    "GoalClassification",
    "GoalComplexity",
    "GoalIntent",
    "ExecutionStrategy",
    "ModelTier",
    "classify_goal",
    "is_trivial_goal",
    "needs_clarification",
    # Resilience (NEW)
    "AgentCircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerResult",
    "CircuitState",
    "TripReason",
    "create_circuit_breaker",
    # Context Management (NEW)
    "RAGContextFilter",
    "RAGResult",
    "FilterStats",
    # Model Routing (NEW)
    "Capability",
    "CapabilityChecker",
    "CapabilityIssue",
    "CompatibilityResult",
    "IssueSeverity",
    # Personas
    "PersonaManager",
    "AgentPersona",
    "PersonalityTrait",
    "CommunicationStyle",
    "RiskTolerance",
    "PlanningDepth",
    "PersonaTrait",
    "CommunicationPreferences",
    "DecisionMakingPreferences",
    "PromptModifications",
    # Prompts
    "PromptRegistry",
    "PromptTemplate",
    "PromptVersion",
    "PromptComposer",
    "TemplateLoader",
    # Memory
    "CognitiveMemoryIntegrator",
    # Sandbox Core
    "SandboxRegistry",
    "SandboxRegistryConfig",
    "SandboxConfig",
    "SandboxProvider",
    "SandboxMode",
    "SandboxAccessLevel",
    "SandboxStatus",
    "ExecutionResult",
    "FileInfo",
    # Sandbox Providers
    "DockerSandboxProvider",
    "VMSandboxProvider",
    # Sandbox Utilities
    "EphemeralResourceManager",
    "OutputTracker",
    # Sandbox Configuration
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
    # Sandbox Tools
    "set_active_sandbox",
    "clear_active_sandbox",
    "get_active_sandbox",
    "SANDBOX_TOOL_IMPLEMENTATIONS",
    "SANDBOX_TOOL_SCHEMAS",
    # Sandbox Integration
    "SandboxIntegration",
    "SandboxContext",
    "SandboxAgentMixin",
    "register_sandbox_tools",
    "get_sandbox_tool_definitions",
]

__version__ = "0.27.0"  # Phase 1: Foundation (Bug Fixes + Infrastructure)
