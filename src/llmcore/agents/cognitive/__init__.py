# src/llmcore/agents/cognitive/__init__.py
"""
Darwin Layer 2 - Enhanced Cognitive Cycle System.

This package provides the 8-phase enhanced cognitive cycle:
PERCEIVE → PLAN → THINK → VALIDATE → ACT → OBSERVE → REFLECT → UPDATE
"""

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

from .phases.cycle import CognitiveCycle

__all__ = [
    "CognitivePhase",
    "IterationStatus",
    "ValidationResult",
    "ConfidenceLevel",
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
    "CycleIteration",
    "EnhancedAgentState",
    "perceive_phase",
    "plan_phase",
    "think_phase",
    "validate_phase",
    "act_phase",
    "observe_phase",
    "reflect_phase",
    "update_phase",
    "CognitiveCycle",
]
