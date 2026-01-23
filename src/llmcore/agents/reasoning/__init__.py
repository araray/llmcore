# src/llmcore/agents/reasoning/__init__.py
"""
Reasoning Frameworks for LLMCore Agent System.

This module provides implementations of research-backed reasoning frameworks:
- ReAct: Reasoning + Acting in interleaved steps (Yao et al., 2022)
- Reflexion: Learning from failures with verbal self-reflection (Shinn et al., 2023)

Usage:
    from llmcore.agents.reasoning import ReActReasoner, ReflexionReasoner

    # Simple ReAct reasoning
    reasoner = ReActReasoner(llm_provider, activity_loop)
    result = await reasoner.reason(goal="Find log files")

    # Reflexion with learning
    reflexion = ReflexionReasoner(llm_provider, activity_loop)
    result = await reflexion.reason_with_reflection(
        goal="Write a working function",
        max_trials=3
    )

Research References:
    1. Yao et al., "ReAct: Synergizing Reasoning and Acting" (2022)
    2. Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning" (2023)
"""

# ReAct exports
from .react import (
    ReActAction,
    # Config
    ReActConfig,
    ReActObservation,
    # Main class
    ReActReasoner,
    ReActResult,
    ReActStatus,
    # Enums
    ReActStep,
    # Data models
    ReActThought,
    ReActTrajectory,
    # Convenience
    reason_with_react,
)

# Reflexion exports
from .reflexion import (
    # Data models
    Reflection,
    # Config
    ReflexionConfig,
    # Main class
    ReflexionReasoner,
    ReflexionResult,
    # Enums
    TrialOutcome,
    TrialResult,
    # Convenience
    reason_with_reflexion,
)

__all__ = [
    # === ReAct ===
    # Enums
    "ReActStep",
    "ReActStatus",
    # Data models
    "ReActThought",
    "ReActAction",
    "ReActObservation",
    "ReActTrajectory",
    "ReActResult",
    # Config
    "ReActConfig",
    # Main class
    "ReActReasoner",
    # Convenience
    "reason_with_react",
    # === Reflexion ===
    # Enums
    "TrialOutcome",
    # Data models
    "Reflection",
    "TrialResult",
    "ReflexionResult",
    # Config
    "ReflexionConfig",
    # Main class
    "ReflexionReasoner",
    # Convenience
    "reason_with_reflexion",
]
