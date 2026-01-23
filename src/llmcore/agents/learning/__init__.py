# llmcore/agents/learning/__init__.py
"""
Learning module for LLMCore agentic system.

This module provides learning mechanisms that enable agents to improve
their behavior over time through reflection, failure analysis, and
pattern recognition.

Components:
    - ReflectionBridge: Converts REFLECT phase insights into actionable guidance
    - FailureMemory: Tracks and learns from failures across sessions
    - FastPathExecutor: Bypasses full cognitive cycle for trivial goals

Research Foundation:
    - Reflexion: Shinn et al., "Reflexion: Language Agents with Verbal
      Reinforcement Learning" (2023)
    - CoALA: Sumers et al., "Cognitive Architectures for Language Agents" (2023)
"""

from .failure_memory import (
    # Constants
    FAILURE_SUGGESTIONS,
    # Main class
    FailureMemory,
    FailurePattern,
    # Data models
    FailureRecord,
    FailureSeverity,
    # Enums
    FailureType,
)
from .fast_path import (
    # Constants
    RESPONSE_TEMPLATES,
    FastPathConfig,
    FastPathExecutor,
    # Data models
    FastPathResult,
    # Enums
    FastPathStrategy,
    # Classes
    ResponseCache,
    execute_fast_path,
    # Convenience functions
    should_use_fast_path,
)
from .reflection_bridge import (
    # Constants
    INSIGHT_PATTERNS,
    GuidanceSet,
    # Main classes
    InsightExtractor,
    InsightPriority,
    # Enums
    InsightType,
    ReflectionBridge,
    # Data models
    ReflectionInsight,
)

__all__ = [
    # Reflection Bridge
    "InsightType",
    "InsightPriority",
    "ReflectionInsight",
    "GuidanceSet",
    "InsightExtractor",
    "ReflectionBridge",
    "INSIGHT_PATTERNS",
    # Failure Memory
    "FailureType",
    "FailureSeverity",
    "FailureRecord",
    "FailurePattern",
    "FailureMemory",
    "FAILURE_SUGGESTIONS",
    # Fast Path
    "FastPathStrategy",
    "FastPathResult",
    "ResponseCache",
    "FastPathConfig",
    "FastPathExecutor",
    "should_use_fast_path",
    "execute_fast_path",
    "RESPONSE_TEMPLATES",
]
