# src/llmcore/context/sources/recent.py
"""
Recent context source for Adaptive Context Synthesis.

Maintains a sliding window of recent conversation turns and tool
executions, providing temporal continuity across agent iterations.

This is a **Recent Context** source (priority 80), weighted by
recency within the sliding window.

References:
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §12.2 (Recent Context tier)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from ..synthesis import ContextChunk

logger = logging.getLogger(__name__)


class RecentContextSource:
    """
    Context source providing recent conversation / action history.

    Maintains an internal sliding window of turns.  Each turn is a
    ``(role, content, timestamp, metadata)`` record.  The window is
    automatically trimmed to ``max_turns`` on every insertion.

    Example::

        recent = RecentContextSource(max_turns=20)
        recent.add_turn("user", "What's the build status?")
        recent.add_turn("assistant", "All 219 tests pass.")

        chunk = await recent.get_context()
    """

    def __init__(self, max_turns: int = 20) -> None:
        """
        Args:
            max_turns: Maximum number of turns to retain in the window.
        """
        self.max_turns = max_turns
        self._history: list[dict[str, Any]] = []

    def add_turn(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a conversational turn to the history.

        Args:
            role: Speaker role (e.g. ``"user"``, ``"assistant"``, ``"tool"``).
            content: Text content of the turn.
            metadata: Optional metadata (tool name, execution result, etc.).
        """
        self._history.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now(UTC),
                "metadata": metadata or {},
            }
        )

        # Trim to max
        if len(self._history) > self.max_turns:
            self._history = self._history[-self.max_turns :]

    def clear(self) -> None:
        """Clear all history."""
        self._history.clear()

    @property
    def turn_count(self) -> int:
        """Number of turns currently in the window."""
        return len(self._history)

    async def get_context(
        self,
        task: Any | None = None,
        max_tokens: int = 5000,
    ) -> ContextChunk:
        """
        Get recent conversation context.

        Formats the sliding window into a markdown section.
        Returns an empty chunk if no history is available.

        Args:
            task: Current task/goal (unused — recency is the primary signal).
            max_tokens: Maximum tokens to return.

        Returns:
            ContextChunk with formatted recent history.
        """
        if not self._history:
            return ContextChunk(
                source="recent",
                content="",
                tokens=0,
                priority=80,
                relevance=0.9,
                recency=1.0,
            )

        lines = ["# Recent History\n"]

        for turn in self._history:
            role = turn["role"].upper()
            content = turn["content"]
            lines.append(f"**{role}:** {content}\n")

        content = "\n".join(lines)
        tokens = len(content) // 4  # estimate

        return ContextChunk(
            source="recent",
            content=content,
            tokens=tokens,
            priority=80,
            relevance=0.9,
            recency=1.0,
        )
