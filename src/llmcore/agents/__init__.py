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
from .manager import AgentManager, AgentMode, EnhancedAgentManager
from .tools import ToolManager

# =============================================================================
# COGNITIVE SYSTEM
# =============================================================================
from .cognitive import (
    # Orchestrator
    CognitiveCycle,
    # Enums
    CognitivePhase,
    IterationStatus,
    ValidationResult,
    ConfidenceLevel,
    # Models
    CycleIteration,
    # State
    EnhancedAgentState,
    # Phase I/O Models (for advanced usage)
    PerceiveInput,
    PerceiveOutput,
    PlanInput,
    PlanOutput,
    ThinkInput,
    ThinkOutput,
    ValidateInput,
    ValidateOutput,
    ActInput,
    ActOutput,
    ObserveInput,
    ObserveOutput,
    ReflectInput,
    ReflectOutput,
    UpdateInput,
    UpdateOutput,
    # Individual phases (for advanced usage)
    perceive_phase,
    plan_phase,
    think_phase,
    validate_phase,
    act_phase,
    observe_phase,
    reflect_phase,
    update_phase,
)

# =============================================================================
# MEMORY INTEGRATION
# =============================================================================
from .memory import CognitiveMemoryIntegrator

# =============================================================================
# PERSONA SYSTEM
# =============================================================================
from .persona import (
    PersonaManager,
    AgentPersona,
    PersonalityTrait,
    CommunicationStyle,
    RiskTolerance,
    PlanningDepth,
    PersonaTrait,
    CommunicationPreferences,
    DecisionMakingPreferences,
    PromptModifications,
)

# =============================================================================
# PROMPT LIBRARY
# =============================================================================
from .prompts import (
    PromptRegistry,
    PromptTemplate,
    PromptVersion,
    PromptComposer,
    TemplateLoader,
)

# =============================================================================
# SINGLE AGENT MODE
# =============================================================================
from .single_agent import AgentResult, IterationUpdate, SingleAgentMode

# =============================================================================
# SANDBOX SYSTEM
# =============================================================================
from .sandbox import (
    # Core classes
    SandboxRegistry,
    SandboxRegistryConfig,
    SandboxConfig,
    SandboxProvider,
    SandboxMode,
    SandboxAccessLevel,
    SandboxStatus,
    ExecutionResult,
    FileInfo,
    # Providers
    DockerSandboxProvider,
    VMSandboxProvider,
    # Utilities
    EphemeralResourceManager,
    OutputTracker,
    # Configuration
    load_sandbox_config,
    create_registry_config,
    # Exceptions
    SandboxError,
    SandboxInitializationError,
    SandboxExecutionError,
    SandboxTimeoutError,
    SandboxAccessDenied,
    SandboxResourceError,
    SandboxConnectionError,
    SandboxCleanupError,
    # Tool management
    set_active_sandbox,
    clear_active_sandbox,
    get_active_sandbox,
    SANDBOX_TOOL_IMPLEMENTATIONS,
    SANDBOX_TOOL_SCHEMAS,
)

# =============================================================================
# SANDBOX INTEGRATION
# =============================================================================
from .sandbox_integration import (
    SandboxIntegration,
    SandboxContext,
    SandboxAgentMixin,
    register_sandbox_tools,
    get_sandbox_tool_definitions,
)

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

__version__ = "0.26.0"  # Darwin Layer 2 Complete
