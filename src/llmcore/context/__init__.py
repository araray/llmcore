# src/llmcore/context/__init__.py
"""
Adaptive Context Synthesis module for llmcore.

Provides the infrastructure for dynamically reconstructing optimal
context for each agent iteration.  The five context source tiers:

    1. **Core Context** (priority 100): Goals, escalations, constraints.
    2. **Recent Context** (priority 80): Conversation turns, tool results.
    3. **Semantic Context** (priority 60): RAG-retrieved knowledge.
    4. **Episodic Context** (priority 40): Past learnings, failure patterns.
    5. **Skill Context** (varies): On-demand SKILL.md files.

Example::

    from llmcore.context import ContextSynthesizer
    from llmcore.context.sources import (
        GoalContextSource,
        RecentContextSource,
        SemanticContextSource,
    )

    synthesizer = ContextSynthesizer(max_tokens=100_000)
    synthesizer.add_source("goals", GoalContextSource(goal_manager), priority=100)
    synthesizer.add_source("recent", RecentContextSource(max_turns=20), priority=80)

    context = await synthesizer.synthesize(current_task=my_goal)
    # context.content → assembled string
    # context.utilization → 0.0–1.0

References:
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §12 (Adaptive Context Synthesis)
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §13 (Skill Loading System)
"""

from .synthesis import (
    ContextChunk,
    ContextSource,
    ContextSynthesizer,
    EstimateCounter,
    SynthesizedContext,
    TiktokenCounter,
    TokenCounter,
)

__all__ = [
    # Core synthesizer
    "ContextSynthesizer",
    # Data models
    "ContextChunk",
    "SynthesizedContext",
    # Protocols
    "ContextSource",
    "TokenCounter",
    # Token counter implementations
    "TiktokenCounter",
    "EstimateCounter",
]
