# src/llmcore/context/sources/semantic.py
"""
Semantic context source for Adaptive Context Synthesis.

Provides RAG-retrieved context by delegating to a user-supplied
retrieval function.  This allows integration with semantiscan,
llmcore's built-in RAG, or any other retrieval engine.

This is a **Semantic Context** source (priority 60), scored by
relevance to the current task.

Important:
    When using this source together with an LLM call, always set
    ``enable_rag=False`` on the LLM call to avoid double-RAG.

References:
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §12.2 (Semantic Context tier)
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §21.1 (External RAG pattern)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from collections.abc import Awaitable, Callable

from ..synthesis import ContextChunk

logger = logging.getLogger(__name__)

# Type for the retrieval function: (query, top_k) → list of chunk dicts
RetrievalFn = Callable[..., Awaitable[list[dict[str, Any]]]]


class SemanticContextSource:
    """
    Context source that delegates to a RAG retrieval function.

    The retrieval function signature::

        async def retrieve(query: str, top_k: int = 10) -> List[Dict[str, Any]]

    Each returned dict should contain at least ``"content"`` and
    optionally ``"source"`` and ``"score"`` keys.

    Example::

        async def my_retriever(query: str, top_k: int = 10):
            return await semantiscan.retrieve(query, top_k=top_k)

        semantic = SemanticContextSource(retrieval_fn=my_retriever)
        chunk = await semantic.get_context(task=my_goal)
    """

    def __init__(self, retrieval_fn: RetrievalFn) -> None:
        """
        Args:
            retrieval_fn: Async callable ``(query, top_k) → list[dict]``.
        """
        self.retrieval_fn = retrieval_fn

    async def get_context(
        self,
        task: Any | None = None,
        max_tokens: int = 10_000,
    ) -> ContextChunk:
        """
        Get semantic context via the retrieval function.

        If ``task`` is ``None`` or has no ``description`` attribute,
        returns an empty chunk.

        Args:
            task: Current task/goal — must have a ``description`` attribute
                  to build the retrieval query.
            max_tokens: Maximum tokens for the result.

        Returns:
            ContextChunk with RAG-retrieved knowledge.
        """
        if task is None:
            return ContextChunk(
                source="semantic",
                content="",
                tokens=0,
                priority=60,
            )

        # Build query from task description
        query = getattr(task, "description", str(task))
        if not query:
            return ContextChunk(
                source="semantic",
                content="",
                tokens=0,
                priority=60,
            )

        try:
            chunks = await self.retrieval_fn(query, top_k=10)
        except Exception as exc:
            logger.warning("Semantic retrieval failed: %s", exc)
            return ContextChunk(
                source="semantic",
                content="",
                tokens=0,
                priority=60,
            )

        lines = ["# Relevant Knowledge\n"]
        for chunk in chunks:
            source_label = chunk.get("source", "unknown")
            content_text = chunk.get("content", "")
            if content_text:
                lines.append(f"### {source_label}")
                lines.append(content_text)
                lines.append("")

        content = "\n".join(lines)
        tokens = len(content) // 4  # estimate

        return ContextChunk(
            source="semantic",
            content=content,
            tokens=tokens,
            priority=60,
            relevance=0.8,
            recency=0.5,
        )
