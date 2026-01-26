# src/llmcore/agents/cognitive/__init__.py
"""
Darwin Layer 2 - Enhanced Cognitive Cycle System.

This package provides the 8-phase enhanced cognitive cycle:
PERCEIVE → PLAN → THINK → VALIDATE → ACT → OBSERVE → REFLECT → UPDATE

Also provides goal classification for intelligent routing.
"""

from .goal_classifier import (
    ExecutionStrategy,
    GoalClassification,
    GoalClassifier,
    GoalComplexity,
    GoalIntent,
    ModelTier,
    classify_goal,
    is_trivial_goal,
    needs_clarification,
)
from .models import (
    # Phase I/O Models
    ActInput,
    ActOutput,
    # Enums
    CognitivePhase,
    ConfidenceLevel,
    # Iteration Tracking
    CycleIteration,
    # Enhanced State
    EnhancedAgentState,
    IterationStatus,
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
    ValidationResult,
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
from .phases.cycle import CognitiveCycle, StreamingIterationResult

__all__ = [
    # Enums
    "CognitivePhase",
    "IterationStatus",
    "ValidationResult",
    "ConfidenceLevel",
    # Phase I/O
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
    # Phase functions
    "perceive_phase",
    "plan_phase",
    "think_phase",
    "validate_phase",
    "act_phase",
    "observe_phase",
    "reflect_phase",
    "update_phase",
    # Main orchestrator
    "CognitiveCycle",
    "StreamingIterationResult",
    # Goal classification
    "GoalClassifier",
    "GoalClassification",
    "GoalComplexity",
    "GoalIntent",
    "ExecutionStrategy",
    "ModelTier",
    "classify_goal",
    "is_trivial_goal",
    "needs_clarification",
]
