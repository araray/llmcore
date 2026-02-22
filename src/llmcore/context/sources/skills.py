# src/llmcore/context/sources/skills.py
"""
Skill context source for Adaptive Context Synthesis.

Provides dynamically loaded SKILL.md content based on the current task.
Delegates to ``SkillLoader`` for file discovery, parsing, and caching.

This is a **Skill Context** source (priority varies per skill,
typically 30–70), loaded on demand.

References:
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §12.2 (Skill Context tier)
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §13 (Skill Loading System)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..synthesis import ContextChunk

if TYPE_CHECKING:
    from ...autonomous.skills import SkillLoader

logger = logging.getLogger(__name__)


class SkillContextSource:
    """
    Context source backed by the ``SkillLoader``.

    Loads relevant skills for the current task description and
    formats them into a context chunk.

    Example::

        from llmcore.autonomous.skills import SkillLoader
        from llmcore.context.sources import SkillContextSource

        loader = SkillLoader()
        loader.add_skill_directory("~/.local/share/llmcore/skills")

        source = SkillContextSource(loader)
        chunk = await source.get_context(task=my_goal, max_tokens=10000)
    """

    def __init__(
        self,
        skill_loader: SkillLoader,
        default_priority: int = 50,
    ) -> None:
        """
        Args:
            skill_loader: SkillLoader instance with registered directories.
            default_priority: Default priority for the skill context chunk.
        """
        self.skill_loader = skill_loader
        self.default_priority = default_priority

    async def get_context(
        self,
        task: Any | None = None,
        max_tokens: int = 10_000,
    ) -> ContextChunk:
        """
        Get skill context for the current task.

        If no task is provided or the task has no description,
        returns an empty chunk.

        Args:
            task: Current task/goal — must have a ``description``
                  attribute for keyword matching.
            max_tokens: Maximum tokens for loaded skills.

        Returns:
            ContextChunk with formatted skill content.
        """
        if task is None:
            return ContextChunk(
                source="skills",
                content="",
                tokens=0,
                priority=self.default_priority,
            )

        description = getattr(task, "description", str(task))
        if not description:
            return ContextChunk(
                source="skills",
                content="",
                tokens=0,
                priority=self.default_priority,
            )

        try:
            skills = await self.skill_loader.load_for_task(
                task_description=description,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            logger.warning("Skill loading failed: %s", exc)
            return ContextChunk(
                source="skills",
                content="",
                tokens=0,
                priority=self.default_priority,
            )

        if not skills:
            return ContextChunk(
                source="skills",
                content="",
                tokens=0,
                priority=self.default_priority,
            )

        content = self.skill_loader.format_skills(skills)
        tokens = len(content) // 4  # estimate

        return ContextChunk(
            source="skills",
            content=content,
            tokens=tokens,
            priority=self.default_priority,
            relevance=0.7,
            recency=0.5,
        )
