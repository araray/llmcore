# src/llmcore/memory/semantic.py
"""
Semantic Memory — Long-term knowledge retrieval.

This module provides the spec-mandated ``memory/semantic.py`` entry-point.
Semantic memory stores factual knowledge, documentation, and reference
material that can be retrieved via embedding-based similarity search (RAG).

The actual semantic memory implementations live in:

- :class:`~llmcore.context.sources.semantic.SemanticContextSource` —
  context source for the Adaptive Context Synthesis engine
- :class:`~llmcore.embedding.manager.EmbeddingManager` — embedding generation
- :class:`~llmcore.storage.chromadb_vector.ChromaDBVectorStorage` — vector store
- :class:`~llmcore.storage.pgvector_storage.PgvectorStorage` — alt vector store

This module provides a unified ``SemanticMemory`` facade that wraps the
retrieval pipeline for direct use.

References:
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §8 (Memory System)
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §12 (Adaptive Context Synthesis)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SemanticResult:
    """A single result from semantic memory retrieval.

    Attributes:
        content: The retrieved text.
        score: Similarity score (higher = more relevant).
        source: Provenance identifier (document ID, URL, etc.).
        metadata: Extra metadata from the vector store.
    """

    content: str = ""
    score: float = 0.0
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class SemanticMemory:
    """Facade for semantic (RAG-based) memory retrieval.

    Wraps the embedding manager and vector storage to provide a simple
    ``query()`` interface for retrieving relevant knowledge.

    Args:
        embedding_manager: For generating query embeddings.
        vector_storage: For similarity search.
    """

    def __init__(self, embedding_manager: Any, vector_storage: Any) -> None:
        self._embedding = embedding_manager
        self._vector = vector_storage

    async def query(
        self,
        text: str,
        top_k: int = 5,
        threshold: float = 0.0,
        collection: str | None = None,
    ) -> list[SemanticResult]:
        """Retrieve semantically similar content.

        Args:
            text: The query text.
            top_k: Maximum number of results.
            threshold: Minimum similarity score.
            collection: Optional collection/namespace filter.

        Returns:
            List of :class:`SemanticResult` sorted by descending score.
        """
        try:
            embedding = await self._embedding.generate_embedding(text)
            results = await self._vector.query(
                query_vector=embedding,
                top_k=top_k,
                collection=collection,
            )

            semantic_results: list[SemanticResult] = []
            for r in results:
                score = getattr(r, "score", getattr(r, "distance", 0.0))
                if score >= threshold:
                    semantic_results.append(
                        SemanticResult(
                            content=getattr(r, "content", getattr(r, "text", str(r))),
                            score=score,
                            source=getattr(r, "id", getattr(r, "source", "")),
                            metadata=getattr(r, "metadata", {}),
                        )
                    )
            return sorted(semantic_results, key=lambda x: x.score, reverse=True)
        except Exception as e:
            logger.error("Semantic memory query failed: %s", e, exc_info=True)
            return []

    async def store(
        self,
        text: str,
        source: str = "",
        metadata: dict[str, Any] | None = None,
        collection: str | None = None,
    ) -> None:
        """Store content in semantic memory.

        Args:
            text: The content to store.
            source: Provenance identifier.
            metadata: Additional metadata.
            collection: Target collection/namespace.
        """
        try:
            embedding = await self._embedding.generate_embedding(text)
            await self._vector.upsert(
                texts=[text],
                embeddings=[embedding],
                metadatas=[{"source": source, **(metadata or {})}],
                collection=collection,
            )
        except Exception as e:
            logger.error("Semantic memory store failed: %s", e, exc_info=True)


# Re-export the context source for users who want the lower-level API
try:
    from ..context.sources.semantic import SemanticContextSource
except ImportError:
    SemanticContextSource = None  # type: ignore[assignment,misc]


__all__ = [
    "SemanticMemory",
    "SemanticResult",
    "SemanticContextSource",
]
