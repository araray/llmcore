# src/llmcore/memory/backends.py
"""Backend-neutral memory retrieval contracts and adapters.

The memory package owns llmcore's typed view of external retrieval systems.
Adapters in this module must keep optional dependencies lazy so packages such
as semantiscan can be installed, upgraded, or absent without affecting normal
llmcore imports.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Citation:
    """Typed source attribution for retrieved memory.

    ``metastore_pid`` and ``metastore_entity_id`` intentionally use neutral
    field names so llmcore can retain semantiscan provenance without importing
    semantiscan's MetaStore models.
    """

    source_id: str = ""
    chunk_id: str = ""
    document_id: str | None = None
    uri: str | None = None
    path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    metastore_entity_id: str | None = None
    metastore_pid: str | None = None
    run_id: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the citation without dropping empty but meaningful fields."""
        return {
            "source_id": self.source_id,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "uri": self.uri,
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "metastore_entity_id": self.metastore_entity_id,
            "metastore_pid": self.metastore_pid,
            "run_id": self.run_id,
            "provenance": dict(self.provenance),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MemoryRecord:
    """A normalized retrieval result returned by a memory backend."""

    content: str
    score: float = 0.0
    source: str = ""
    citations: list[Citation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    strategy: str = "unknown"

    def to_context_dict(self) -> dict[str, Any]:
        """Return the dict shape consumed by ``SemanticContextSource``."""
        return {
            "content": self.content,
            "score": self.score,
            "source": self.source,
            "strategy": self.strategy,
            "metadata": dict(self.metadata),
            "citations": [citation.to_dict() for citation in self.citations],
        }


@runtime_checkable
class MemoryBackendProtocol(Protocol):
    """Protocol for external memory/RAG backends.

    Implementations retrieve context that has already been assembled by the
    backend. Callers should pass ``enable_rag=False`` to subsequent LLM calls
    when using this protocol to preserve the external-RAG invariant.
    """

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[MemoryRecord]:
        """Retrieve memory records relevant to *query*."""
        ...


RetrieveFn = Callable[..., Awaitable[Any] | Any]


class SemantiscanMemoryBackend:
    """Adapter from semantiscan retrieval batches to llmcore memory records.

    Semantiscan is an optional dependency. Supplying ``retrieve_fn`` is useful
    for tests or preconfigured services; otherwise ``semantiscan.api.retrieve``
    is imported lazily on first use.
    """

    def __init__(
        self,
        *,
        retrieve_fn: RetrieveFn | None = None,
        collection: str | None = None,
        storage: Any | None = None,
        embedder: Any | None = None,
        strategy: str = "vector",
        llm: Any | None = None,
        metadata_store: Any | None = None,
        graph_store: Any | None = None,
        default_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self._retrieve_fn = retrieve_fn
        self.collection = collection
        self.storage = storage
        self.embedder = embedder
        self.strategy = strategy
        self.llm = llm
        self.metadata_store = metadata_store
        self.graph_store = graph_store
        self.default_kwargs = dict(default_kwargs or {})

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[MemoryRecord]:
        """Retrieve and normalize semantiscan results."""
        retrieve_fn = self._resolve_retrieve_fn()
        call_kwargs = self._build_retrieve_kwargs(top_k=top_k, filters=filters, extra=kwargs)
        batch = retrieve_fn(query, **call_kwargs)
        if inspect.isawaitable(batch):
            batch = await batch
        return self._records_from_batch(batch)

    def as_retrieval_fn(self) -> RetrieveFn:
        """Expose this backend as a ``SemanticContextSource`` retrieval function."""

        async def retrieve_for_context(query: str, top_k: int = 10, **kwargs: Any):
            records = await self.retrieve(query, top_k=top_k, **kwargs)
            return [record.to_context_dict() for record in records]

        return retrieve_for_context

    def _resolve_retrieve_fn(self) -> RetrieveFn:
        if self._retrieve_fn is not None:
            return self._retrieve_fn

        try:
            from semantiscan.api import retrieve as semantiscan_retrieve
        except ImportError as exc:  # pragma: no cover - exercised by users without semantiscan
            raise RuntimeError(
                "semantiscan is required unless SemantiscanMemoryBackend receives retrieve_fn"
            ) from exc

        self._retrieve_fn = semantiscan_retrieve
        return semantiscan_retrieve

    def _build_retrieve_kwargs(
        self,
        *,
        top_k: int,
        filters: Mapping[str, Any] | None,
        extra: Mapping[str, Any],
    ) -> dict[str, Any]:
        call_kwargs = dict(self.default_kwargs)
        call_kwargs.update(extra)
        call_kwargs.setdefault("top_k", top_k)
        if filters is not None:
            call_kwargs.setdefault("filters", dict(filters))

        optional_defaults = {
            "collection": self.collection,
            "storage": self.storage,
            "embedder": self.embedder,
            "strategy": self.strategy,
            "llm": self.llm,
            "metadata_store": self.metadata_store,
            "graph_store": self.graph_store,
        }
        for key, value in optional_defaults.items():
            if value is not None:
                call_kwargs.setdefault(key, value)
        return call_kwargs

    def _records_from_batch(self, batch: Any) -> list[MemoryRecord]:
        results = self._extract_results(batch)
        batch_strategy = str(self._get(batch, "strategy", self.strategy) or self.strategy)
        batch_metadata = self._metadata_from(batch)
        records: list[MemoryRecord] = []
        for result in results:
            metadata = self._metadata_from(result)
            if batch_metadata:
                metadata.setdefault("retrieval_batch", batch_metadata)
            chunk_id = str(self._get(result, "chunk_id", metadata.get("chunk_id", "")) or "")
            source = self._source_from(result, metadata, chunk_id)
            records.append(
                MemoryRecord(
                    content=str(self._get(result, "content", metadata.get("content", "")) or ""),
                    score=float(self._get(result, "score", metadata.get("score", 0.0)) or 0.0),
                    source=source,
                    citations=[self._citation_from(result, metadata, chunk_id, source)],
                    metadata=metadata,
                    strategy=str(self._get(result, "strategy", batch_strategy) or batch_strategy),
                )
            )
        return records

    def _extract_results(self, batch: Any) -> list[Any]:
        if batch is None:
            return []
        results = self._get(batch, "results", None)
        if results is None and isinstance(batch, (list, tuple)):
            results = batch
        if results is None:
            return []
        return list(results)

    def _citation_from(
        self,
        result: Any,
        metadata: dict[str, Any],
        chunk_id: str,
        source: str,
    ) -> Citation:
        typed_citation = self._get(result, "citation", None)
        if typed_citation is not None:
            return self._citation_from_typed(typed_citation, chunk_id=chunk_id, source=source)

        provenance = metadata.get("provenance")
        if not isinstance(provenance, dict):
            provenance = {}

        return Citation(
            source_id=source,
            chunk_id=chunk_id,
            document_id=self._first(metadata, "document_id", "doc_id", "repo_name"),
            uri=self._first(metadata, "uri", "url"),
            path=self._first(metadata, "file_path", "source_file", "path"),
            start_line=self._line(metadata, "start_line"),
            end_line=self._line(metadata, "end_line"),
            metastore_entity_id=self._first(
                metadata,
                "chunk_entity_id",
                "entity_id",
                "metastore_entity_id",
            ),
            metastore_pid=self._first(metadata, "chunk_pid", "pid", "metastore_pid"),
            run_id=self._first(metadata, "run_id", "source_run"),
            provenance=provenance,
            metadata={
                "strategy": self._get(result, "strategy", self.strategy),
                "score": self._get(result, "score", 0.0),
            },
        )

    def _citation_from_typed(self, value: Any, *, chunk_id: str, source: str) -> Citation:
        provenance = self._get(value, "provenance", {})
        metadata = self._get(value, "metadata", {})
        return Citation(
            source_id=str(self._get(value, "source_file", source) or source),
            chunk_id=str(self._get(value, "chunk_id", chunk_id) or chunk_id),
            document_id=self._string_or_none(self._get(value, "document_id", None)),
            uri=self._string_or_none(self._get(value, "uri", None)),
            path=self._string_or_none(self._get(value, "source_file", None)),
            start_line=self._int_or_none(self._get(value, "start_line", None)),
            end_line=self._int_or_none(self._get(value, "end_line", None)),
            metastore_entity_id=self._string_or_none(
                self._get(value, "metastore_entity_id", None)
            ),
            metastore_pid=self._string_or_none(self._get(value, "metastore_pid", None)),
            run_id=self._string_or_none(self._get(value, "run_id", None)),
            provenance=dict(provenance) if isinstance(provenance, Mapping) else {},
            metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        )

    def _source_from(self, result: Any, metadata: dict[str, Any], chunk_id: str) -> str:
        source = self._first(metadata, "source", "file_path", "source_file", "path", "uri", "url")
        if source:
            return source
        return str(self._get(result, "source", chunk_id) or chunk_id)

    def _metadata_from(self, value: Any) -> dict[str, Any]:
        metadata = self._get(value, "metadata", {})
        if isinstance(metadata, Mapping):
            return dict(metadata)
        return {}

    def _first(self, metadata: Mapping[str, Any], *keys: str) -> str | None:
        for key in keys:
            value = metadata.get(key)
            if value is not None and value != "":
                return str(value)
        return None

    def _line(self, metadata: Mapping[str, Any], key: str) -> int | None:
        value = metadata.get(key)
        return self._int_or_none(value)

    def _int_or_none(self, value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _string_or_none(self, value: Any) -> str | None:
        if value is None or value == "":
            return None
        return str(value)

    def _get(self, value: Any, key: str, default: Any = None) -> Any:
        if isinstance(value, Mapping):
            return value.get(key, default)
        return getattr(value, key, default)


__all__ = [
    "Citation",
    "MemoryBackendProtocol",
    "MemoryRecord",
    "SemantiscanMemoryBackend",
]
