# src/llmcore/agents/__init__.py
"""
Darwin Agent System - Layer 2 Complete.

The Darwin agent system provides a comprehensive framework for building
autonomous AI agents with:

- **8-Phase Cognitive Cycle**: PERCEIVE → PLAN → THINK → VALIDATE → ACT → OBSERVE → REFLECT → UPDATE
- **Persona System**: Customizable agent personalities and behaviors
- **Prompt Library**: Versioned, testable prompts with A/B testing
- **Memory Integration**: Episodic and semantic memory with learning
- **Single & Multi-Agent**: Support for both modes
- **Sandbox Execution**: Safe isolated execution environments

Main Components:
    - EnhancedAgentManager: Main entry point for agent orchestration
    - SingleAgentMode: High-level single-agent interface
    - CognitiveCycle: 8-phase cognitive loop orchestrator
    - PersonaManager: Persona creation and management
    - PromptRegistry: Prompt library and versioning

Quick Start - Single Agent:
    >>> from llmcore.agents import EnhancedAgentManager, AgentMode
    >>>
    >>> manager = EnhancedAgentManager(
    ...     provider_manager=provider_manager,
    ...     memory_manager=memory_manager,
    ...     storage_manager=storage_manager,
    ...     tool_manager=tool_manager
    ... )
    >>>
    >>> result = await manager.run(
    ...     goal="Analyze sales data and generate report",
    ...     mode=AgentMode.SINGLE,
    ...     persona="analyst",
    ...     max_iterations=10
    ... )
    >>>
    >>> print(result.final_answer)

Quick Start - Legacy Mode:
    >>> # Backward compatible with existing code
    >>> result = await manager.run(
    ...     goal="Calculate factorial of 10",
    ...     mode=AgentMode.LEGACY
    ... )

References:
    - Technical Spec: Darwin Layer 2 (Sections 5.1-5.7)
    - Implementation Dossiers: Steps 2.1-2.11
"""

# Core Manager
# Cognitive System
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
    IterationStatus,
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
    observe_phase,
    # Individual phases (for advanced usage)
    perceive_phase,
    plan_phase,
    reflect_phase,
    think_phase,
    update_phase,
    validate_phase,
)
from .manager import AgentMode, EnhancedAgentManager

# Memory Integration
from .memory import CognitiveMemoryIntegrator

# Persona System
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

# Prompt Library
from .prompts import (
    PromptComposer,
    PromptRegistry,
    PromptTemplate,
    PromptVersion,
    TemplateLoader,
)

# Single Agent Mode
from .single_agent import AgentResult, IterationUpdate, SingleAgentMode

__all__ = [
    # Core Manager
    "EnhancedAgentManager",
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
]

__version__ = "0.26.0"  # Darwin Layer 2 Complete
