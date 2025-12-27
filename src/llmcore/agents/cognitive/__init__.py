# src/llmcore/agents/cognitive/__init__.py
"""
Darwin Layer 2 - Enhanced Cognitive Cycle System.

This package provides the 8-phase enhanced cognitive cycle:
PERCEIVE → PLAN → THINK → VALIDATE → ACT → OBSERVE → REFLECT → UPDATE

The cognitive system enables sophisticated ReAct-style reasoning with:
- Structured phase execution
- Safety validation
- Learning and reflection
- Progress tracking
- Memory integration

Main Components:
    - CognitiveCycle: Orchestrator for the 8-phase cycle
    - EnhancedAgentState: Extended state with iteration tracking
    - Phase functions: Individual phase implementations
    - I/O models: Structured inputs/outputs for each phase

Usage:
    >>> from llmcore.agents.cognitive import (
    ...     CognitiveCycle,
    ...     EnhancedAgentState,
    ...     CognitivePhase,
    ... )
    >>>
    >>> state = EnhancedAgentState(goal="Analyze data")
    >>> cycle = CognitiveCycle(
    ...     provider_manager=provider_manager,
    ...     memory_manager=memory_manager,
    ...     storage_manager=storage_manager,
    ...     tool_manager=tool_manager,
    ... )
    >>>
    >>> iteration = await cycle.run_iteration(state, session_id="session-1")

References:
    - Technical Spec: Section 5.3 (Enhanced Cognitive Cycle)
    - Implementation Dossiers: Steps 2.4-2.7
"""

# =============================================================================
# MODELS - Core data models and enums
# =============================================================================

from .models import (
    # Enums
    CognitivePhase,
    ConfidenceLevel,
    IterationStatus,
    ValidationResult,
    # Phase I/O Models
    ActInput,
    ActOutput,
    ObserveInput,
    ObserveOutput,
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
    # Iteration Tracking
    CycleIteration,
    # Enhanced State
    EnhancedAgentState,
)

# =============================================================================
# PHASE FUNCTIONS - Individual cognitive phase implementations
# =============================================================================

from .phases import (
    act_phase,
    observe_phase,
    perceive_phase,
    plan_phase,
    reflect_phase,
    think_phase,
    update_phase,
    validate_phase,
)

# =============================================================================
# ORCHESTRATOR - Cognitive cycle coordinator
# =============================================================================

from .phases.cycle import CognitiveCycle

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "CognitivePhase",
    "IterationStatus",
    "ValidationResult",
    "ConfidenceLevel",
    # Phase I/O Models
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
    # Iteration Tracking
    "CycleIteration",
    # Enhanced State
    "EnhancedAgentState",
    # Phase Functions
    "perceive_phase",
    "plan_phase",
    "think_phase",
    "validate_phase",
    "act_phase",
    "observe_phase",
    "reflect_phase",
    "update_phase",
    # Orchestrator
    "CognitiveCycle",
]
