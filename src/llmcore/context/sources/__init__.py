# src/llmcore/context/sources/__init__.py
"""
Built-in context source implementations for Adaptive Context Synthesis.

Sources:
    - ``GoalContextSource``: Current goals, progress, learned strategies.
    - ``RecentContextSource``: Recent conversation turns and tool executions.
    - ``SemanticContextSource``: RAG-retrieved chunks via retrieval function.
    - ``EpisodicContextSource``: Historical patterns and past learnings.
    - ``SkillContextSource``: On-demand SKILL.md files via SkillLoader.
"""

from .episodic import EpisodicContextSource
from .goals import GoalContextSource
from .recent import RecentContextSource
from .semantic import SemanticContextSource
from .skills import SkillContextSource

__all__ = [
    "EpisodicContextSource",
    "GoalContextSource",
    "RecentContextSource",
    "SemanticContextSource",
    "SkillContextSource",
]
