# src/llmcore/context/sources/goals.py
"""
Goal context source for Adaptive Context Synthesis.

Provides context about the current goal hierarchy including:
- Primary goal description and success criteria
- Active sub-goals and their progress
- Learned strategies from past attempts

This is a **Core Context** source (priority 100) that is always
included in the synthesized context.

References:
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §12.2 (Core Context tier)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..synthesis import ContextChunk

if TYPE_CHECKING:
    from ...autonomous.goals import GoalManager

logger = logging.getLogger(__name__)


class GoalContextSource:
    """
    Context source that reads from a ``GoalManager``.

    Formats the current goal tree into a structured markdown section
    suitable for injection into the LLM context window.

    Example::

        from llmcore.autonomous import GoalManager
        from llmcore.context.sources import GoalContextSource

        goals = GoalManager(...)
        source = GoalContextSource(goals)
        chunk = await source.get_context(max_tokens=2000)
    """

    def __init__(self, goal_manager: GoalManager) -> None:
        """
        Args:
            goal_manager: The GoalManager instance to read goals from.
        """
        self.goal_manager = goal_manager

    async def get_context(
        self,
        task: Any | None = None,
        max_tokens: int = 2000,
    ) -> ContextChunk:
        """
        Get goal context.

        Retrieves all active goals and formats them with status,
        progress, success criteria, and learned strategies.

        Args:
            task: Current task/goal (unused — goals are always relevant).
            max_tokens: Maximum tokens to return (for budget hints).

        Returns:
            ContextChunk with formatted goal information.
        """
        lines = ["# Current Goals\n"]

        try:
            active = await self.goal_manager.get_active_goals()
        except Exception as exc:
            logger.warning("Failed to get active goals: %s", exc)
            active = []

        if not active:
            # Also try all goals if nothing is active yet
            try:
                all_goals = await self.goal_manager.get_all_goals()
                # Include pending goals as well
                active = [g for g in all_goals if g.status.value in ("active", "pending")]
            except Exception as exc:
                logger.warning("Failed to get all goals: %s", exc)

        for goal in active:
            lines.append(f"## {goal.description}")
            lines.append(f"Status: {goal.status.value}")
            lines.append(f"Progress: {goal.progress:.0%}")

            if goal.success_criteria:
                lines.append("\nSuccess Criteria:")
                for criterion in goal.success_criteria:
                    met = criterion.is_met() if hasattr(criterion, "is_met") else False
                    marker = "✓" if met else "○"
                    lines.append(f"  {marker} {criterion.description}")

            if goal.learned_strategies:
                lines.append("\nLearned Strategies:")
                for strategy in goal.learned_strategies[-3:]:  # Most recent 3
                    lines.append(f"  - {strategy}")

            lines.append("")

        content = "\n".join(lines)
        tokens = len(content) // 4  # estimate

        return ContextChunk(
            source="goals",
            content=content,
            tokens=tokens,
            priority=100,
            relevance=1.0,
            recency=1.0,
        )
