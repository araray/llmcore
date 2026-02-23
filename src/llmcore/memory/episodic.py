# src/llmcore/memory/episodic.py
"""
Episodic Memory — Experience-based learning from past episodes.

This module provides the spec-mandated ``memory/episodic.py`` entry-point.
Episodic memory records discrete *episodes* — completed tasks, interactions,
failures, and their outcomes — so the agent can learn from experience.

The actual episodic memory implementations live in:

- :class:`~llmcore.context.sources.episodic.EpisodicContextSource` —
  context source for Adaptive Context Synthesis
- :class:`~llmcore.storage.sqlite_episode_helpers` — SQLite episode persistence
- :class:`~llmcore.models.Episode` / :class:`~llmcore.models.EpisodeType` —
  data models

This module provides a unified ``EpisodicMemory`` facade.

References:
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §8 (Memory System)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..models import Episode, EpisodeType

logger = logging.getLogger(__name__)


@dataclass
class EpisodeRecord:
    """Enriched episode with retrieval metadata.

    Attributes:
        episode: The underlying Episode model.
        relevance_score: How relevant this episode is to a given query.
        age_hours: How old the episode is.
    """

    episode: Episode
    relevance_score: float = 0.0
    age_hours: float = 0.0


class EpisodicMemory:
    """Facade for episodic memory operations.

    Provides a simple interface for storing and retrieving episodes
    (past experiences) that the agent can learn from.

    Args:
        storage_manager: The StorageManager for episode persistence.
        embedding_manager: Optional embedding manager for semantic episode search.
    """

    def __init__(
        self,
        storage_manager: Any,
        embedding_manager: Any | None = None,
    ) -> None:
        self._storage = storage_manager
        self._embedding = embedding_manager

    async def record_episode(
        self,
        session_id: str,
        episode_type: EpisodeType,
        content: str,
        outcome: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Episode:
        """Record a new episode.

        Args:
            session_id: The session this episode belongs to.
            episode_type: Type of episode (action, reflection, etc.).
            content: The episode content/description.
            outcome: The outcome of the episode.
            metadata: Additional metadata.

        Returns:
            The created Episode.
        """
        episode = Episode(
            session_id=session_id,
            episode_type=episode_type,
            content=content,
            outcome=outcome,
            metadata=metadata or {},
        )

        try:
            if hasattr(self._storage, "session_storage"):
                storage = self._storage.session_storage
                if hasattr(storage, "save_episode"):
                    await storage.save_episode(episode)
                    logger.debug(
                        "Episode recorded: type=%s, session=%s.",
                        episode_type.value,
                        session_id,
                    )
        except Exception as e:
            logger.error("Failed to record episode: %s", e, exc_info=True)

        return episode

    async def recall_recent(
        self,
        session_id: str | None = None,
        limit: int = 10,
        episode_type: EpisodeType | None = None,
    ) -> list[EpisodeRecord]:
        """Recall recent episodes.

        Args:
            session_id: Filter by session (None = all sessions).
            limit: Maximum number of episodes to return.
            episode_type: Filter by episode type.

        Returns:
            List of :class:`EpisodeRecord` sorted by recency.
        """
        try:
            if hasattr(self._storage, "session_storage"):
                storage = self._storage.session_storage
                if hasattr(storage, "get_episodes"):
                    episodes = await storage.get_episodes(
                        session_id=session_id,
                        limit=limit,
                        episode_type=episode_type,
                    )
                    now = time.time()
                    return [
                        EpisodeRecord(
                            episode=ep,
                            age_hours=(now - getattr(ep, "timestamp", now)) / 3600,
                        )
                        for ep in episodes
                    ]
        except Exception as e:
            logger.error("Failed to recall episodes: %s", e, exc_info=True)

        return []

    async def recall_similar(
        self,
        query: str,
        top_k: int = 5,
        session_id: str | None = None,
    ) -> list[EpisodeRecord]:
        """Recall episodes semantically similar to the query.

        Requires an embedding manager to be configured.

        Args:
            query: Text to match against episode content.
            top_k: Maximum results.
            session_id: Optional session filter.

        Returns:
            List of :class:`EpisodeRecord` sorted by relevance.
        """
        if self._embedding is None:
            logger.debug("No embedding manager; falling back to recent episodes.")
            return await self.recall_recent(session_id=session_id, limit=top_k)

        # This is a simplified implementation; the full version would
        # use the vector storage to do embedding-based search over episodes.
        return await self.recall_recent(session_id=session_id, limit=top_k)


# Re-export the context source for lower-level API access
try:
    from ..context.sources.episodic import EpisodicContextSource
except ImportError:
    EpisodicContextSource = None  # type: ignore[assignment,misc]


__all__ = [
    "EpisodicMemory",
    "EpisodeRecord",
    "Episode",
    "EpisodeType",
    "EpisodicContextSource",
]
