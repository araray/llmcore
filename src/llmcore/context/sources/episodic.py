# src/llmcore/context/sources/episodic.py
"""
Episodic context source for Adaptive Context Synthesis.

Provides historical context from past agent episodes including:
- Completed goal summaries and outcomes
- Failure patterns and root causes
- Successful strategies and approaches
- User preferences and interaction patterns

This is an **Episodic Context** source (priority 40), weighted by
significance of past experiences.

References:
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §12.2 (Episodic Context tier)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, UTC
from typing import Any, Dict, List, Optional

from ..synthesis import ContextChunk

logger = logging.getLogger(__name__)


class EpisodicContextSource:
    """
    Context source for long-term episodic memory.

    Stores discrete "episodes" — summaries of past interactions,
    goal completions, failure analyses, and learned patterns.
    Episodes are added explicitly (e.g. after REFLECT phase) and
    retrieved by recency and optional relevance matching.

    Example::

        episodic = EpisodicContextSource(max_episodes=100)

        episodic.add_episode(
            summary="Deployed v0.29.0 successfully after fixing import chain",
            tags=["deployment", "import", "fix"],
            significance=0.8,
        )

        chunk = await episodic.get_context(task=my_goal)
    """

    def __init__(self, max_episodes: int = 100) -> None:
        """
        Args:
            max_episodes: Maximum number of episodes to retain.
        """
        self.max_episodes = max_episodes
        self._episodes: list[dict[str, Any]] = []

    def add_episode(
        self,
        summary: str,
        tags: list[str] | None = None,
        significance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Record an episode.

        Args:
            summary: Natural language summary of the episode.
            tags: Tags for matching against future tasks.
            significance: Importance score (0.0–1.0).
            metadata: Arbitrary additional data.
        """
        self._episodes.append(
            {
                "summary": summary,
                "tags": tags or [],
                "significance": significance,
                "timestamp": datetime.now(UTC),
                "metadata": metadata or {},
            }
        )

        # Trim oldest
        if len(self._episodes) > self.max_episodes:
            self._episodes = self._episodes[-self.max_episodes :]

    def clear(self) -> None:
        """Clear all stored episodes."""
        self._episodes.clear()

    @property
    def episode_count(self) -> int:
        """Number of stored episodes."""
        return len(self._episodes)

    async def get_context(
        self,
        task: Any | None = None,
        max_tokens: int = 3000,
    ) -> ContextChunk:
        """
        Get episodic context.

        If a task is provided and has a ``description`` attribute,
        episodes are filtered and ranked by tag overlap.  Otherwise,
        the most recent episodes are returned.

        Args:
            task: Current task/goal for relevance matching.
            max_tokens: Maximum tokens to return.

        Returns:
            ContextChunk with formatted episode summaries.
        """
        if not self._episodes:
            return ContextChunk(
                source="episodic",
                content="",
                tokens=0,
                priority=40,
                relevance=0.5,
                recency=0.3,
            )

        # Score episodes by relevance to task
        scored = self._rank_episodes(task)

        lines = ["# Past Experiences\n"]
        char_budget = max_tokens * 4  # rough char budget
        used = 0

        for episode in scored:
            entry = f"- [{episode['significance']:.0%}] {episode['summary']}"
            if used + len(entry) > char_budget:
                break
            lines.append(entry)
            used += len(entry)

        content = "\n".join(lines)
        tokens = len(content) // 4

        return ContextChunk(
            source="episodic",
            content=content,
            tokens=tokens,
            priority=40,
            relevance=0.5,
            recency=0.3,
        )

    def _rank_episodes(
        self,
        task: Any | None,
    ) -> list[dict[str, Any]]:
        """
        Rank episodes by relevance to the current task.

        Scoring: significance + tag-overlap bonus + recency bonus.
        """
        task_words = set()
        if task is not None:
            desc = getattr(task, "description", str(task))
            task_words = set(desc.lower().split())

        scored: list[tuple] = []
        for episode in self._episodes:
            score = episode["significance"]

            # Tag overlap bonus
            if task_words:
                ep_tags = set(t.lower() for t in episode["tags"])
                overlap = len(task_words & ep_tags)
                score += overlap * 0.2

            scored.append((score, episode))

        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored]
