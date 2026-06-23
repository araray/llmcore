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
from collections.abc import Awaitable, Callable
from typing import Any

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

    def __init__(self, retrieval_fn: RetrievalFn, observability: Any | None = None) -> None:
        """
        Args:
            retrieval_fn: Async callable ``(query, top_k) → list[dict]``.
            observability: Optional llmcore observability components used to
                emit bounded RAG retrieval events.
        """
        self.retrieval_fn = retrieval_fn
        self.observability = observability

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
            await self._log_rag_event(
                event_type="query_completed",
                query=query,
                chunks=[],
                severity="warning",
                data={"error": str(exc)[:1000]},
            )
            return ContextChunk(
                source="semantic",
                content="",
                tokens=0,
                priority=60,
            )

        await self._log_rag_event(
            event_type="documents_retrieved",
            query=query,
            chunks=chunks,
        )

        lines = ["# Relevant Knowledge\n"]
        for chunk in chunks:
            source_label = chunk.get("source", "unknown")
            content_text = chunk.get("content", "")
            if content_text:
                lines.append(f"### {source_label}")
                citation_label = self._citation_label(chunk)
                if citation_label and citation_label != source_label:
                    lines.append(f"Citation: {citation_label}")
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

    async def _log_rag_event(
        self,
        *,
        event_type: str,
        query: str,
        chunks: list[dict[str, Any]],
        severity: str = "info",
        data: dict[str, Any] | None = None,
    ) -> None:
        """Emit a best-effort RAG event without coupling retrieval to logging."""
        logger_obj = getattr(self.observability, "logger", None)
        if logger_obj is None:
            return

        try:
            from llmcore.agents.observability import (
                EventSeverity,
                RAGEvent,
                RAGEventType,
            )

            scores = [
                float(chunk.get("score"))
                for chunk in chunks
                if isinstance(chunk, dict) and chunk.get("score") is not None
            ]
            event_data = dict(data or {})
            citation_summaries = self._citation_summaries(chunks)
            if citation_summaries:
                event_data.setdefault("citations", citation_summaries)

            event = RAGEvent(
                session_id=getattr(logger_obj, "session_id", "semantic"),
                event_type=RAGEventType(event_type),
                severity=EventSeverity(severity),
                query=str(query)[:1000],
                query_type="semantic",
                source="semantic",
                num_results=len(chunks),
                top_score=max(scores) if scores else None,
                avg_score=(sum(scores) / len(scores)) if scores else None,
                documents_used=self._documents_used(chunks),
                data=event_data,
            )
            await logger_obj.log(event)
        except Exception as exc:
            logger.debug("Semantic RAG observability event skipped: %s", exc)

    @classmethod
    def _citation_label(cls, chunk: dict[str, Any]) -> str:
        """Return a compact source label from a retrieval chunk citation."""
        citation = cls._first_citation(chunk)
        if citation is None:
            citation = cls._metadata_citation(chunk)
        if citation is None:
            return ""

        source = cls._citation_source(citation)
        start_line = cls._int_or_none(citation.get("start_line"))
        end_line = cls._int_or_none(citation.get("end_line"))
        if source:
            if start_line is not None and end_line is not None and end_line != start_line:
                return f"{source}:{start_line}-{end_line}"
            if start_line is not None:
                return f"{source}:{start_line}"
            return source

        chunk_id = citation.get("chunk_id")
        return str(chunk_id)[:200] if chunk_id else ""

    @classmethod
    def _citation_summaries(cls, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return bounded citation metadata for observability event payloads."""
        summaries: list[dict[str, Any]] = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            citation = cls._first_citation(chunk) or cls._metadata_citation(chunk)
            if citation is None:
                continue
            summary = {
                "source": cls._citation_source(citation),
                "chunk_id": cls._string_or_none(citation.get("chunk_id")),
                "document_id": cls._string_or_none(citation.get("document_id")),
                "start_line": cls._int_or_none(citation.get("start_line")),
                "end_line": cls._int_or_none(citation.get("end_line")),
                "metastore_pid": cls._string_or_none(citation.get("metastore_pid")),
                "metastore_entity_id": cls._string_or_none(
                    citation.get("metastore_entity_id")
                ),
            }
            summary = {key: value for key, value in summary.items() if value not in (None, "")}
            if summary:
                summaries.append(summary)
            if len(summaries) >= 20:
                break
        return summaries

    @classmethod
    def _first_citation(cls, chunk: dict[str, Any]) -> dict[str, Any] | None:
        citation = chunk.get("citation")
        if isinstance(citation, dict):
            return citation

        citations = chunk.get("citations")
        if isinstance(citations, list):
            for citation in citations:
                if isinstance(citation, dict):
                    return citation
        return None

    @classmethod
    def _metadata_citation(cls, chunk: dict[str, Any]) -> dict[str, Any] | None:
        metadata = chunk.get("metadata")
        if not isinstance(metadata, dict):
            return None

        citation_keys = {
            "chunk_id",
            "document_id",
            "source_id",
            "source_file",
            "path",
            "file_path",
            "uri",
            "url",
            "start_line",
            "end_line",
            "metastore_pid",
            "metastore_entity_id",
        }
        if not any(key in metadata for key in citation_keys):
            return None
        return metadata

    @classmethod
    def _citation_source(cls, citation: dict[str, Any]) -> str:
        for key in ("path", "source_file", "file_path", "source_id", "document_id", "uri", "url"):
            value = citation.get(key)
            if value:
                return str(value)[:200]
        return ""

    @classmethod
    def _documents_used(cls, chunks: list[dict[str, Any]]) -> list[str]:
        """Return bounded document/source IDs from retrieval chunks."""
        documents: list[str] = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            document = cls._chunk_document_id(chunk)
            if document and document not in documents:
                documents.append(document[:200])
            if len(documents) >= 20:
                break
        return documents

    @classmethod
    def _chunk_document_id(cls, chunk: dict[str, Any]) -> str:
        metadata = chunk.get("metadata")
        if isinstance(metadata, dict):
            for key in ("document_id", "source_id", "chunk_id", "path", "uri"):
                value = metadata.get(key)
                if value:
                    return str(value)

        citations = chunk.get("citations")
        if isinstance(citations, list):
            for citation in citations:
                if not isinstance(citation, dict):
                    continue
                for key in ("document_id", "source_id", "chunk_id", "path", "uri"):
                    value = citation.get(key)
                    if value:
                        return str(value)

        for key in ("source", "chunk_id", "id"):
            value = chunk.get(key)
            if value:
                return str(value)
        return ""

    @staticmethod
    def _int_or_none(value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _string_or_none(value: Any) -> str | None:
        if value is None or value == "":
            return None
        return str(value)[:200]
