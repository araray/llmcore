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
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
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


@dataclass(frozen=True)
class ConsolidationReport:
    """Backend-neutral result for a long-term memory consolidation pass."""

    backend: str
    run_id: str | None = None
    started_at: str = ""
    finished_at: str | None = None
    duration_ms: float = 0.0
    items_scanned: int = 0
    items_decayed: int = 0
    items_expired: int = 0
    items_distilled: int = 0
    items_archived: int = 0
    summaries_created: int = 0
    clusters_formed: int = 0
    contradictions_found: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report using plain JSON-compatible values."""
        return {
            "backend": self.backend,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "items_scanned": self.items_scanned,
            "items_decayed": self.items_decayed,
            "items_expired": self.items_expired,
            "items_distilled": self.items_distilled,
            "items_archived": self.items_archived,
            "summaries_created": self.summaries_created,
            "clusters_formed": self.clusters_formed,
            "contradictions_found": self.contradictions_found,
            "diagnostics": dict(self.diagnostics),
            "warnings": list(self.warnings),
        }

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Pydantic-compatible serialization shim for downstream callers."""
        return self.to_dict()

    @classmethod
    def from_result(
        cls,
        result: Any,
        *,
        backend: str,
        started_at: datetime | str | None = None,
        duration_ms: float | None = None,
        warnings: list[str] | None = None,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> "ConsolidationReport":
        """Normalize a backend-specific consolidation result into a report."""
        if isinstance(result, cls):
            return result

        data = _plain_mapping(result)
        backend_warnings = list(warnings or [])
        errors = data.get("errors", [])
        if isinstance(errors, list):
            backend_warnings.extend(str(error) for error in errors)
        elif errors:
            backend_warnings.append(str(errors))

        report_started_at = data.get("started_at", started_at)
        report_finished_at = data.get("finished_at", None)
        observed_duration = _float_or_none(data.get("duration_ms"))
        if observed_duration is None:
            observed_duration = duration_ms if duration_ms is not None else 0.0

        summaries_created = _int_value(data.get("summaries_created"))
        return cls(
            backend=str(data.get("backend") or backend),
            run_id=_string_or_none(data.get("run_id")),
            started_at=_iso_datetime(report_started_at),
            finished_at=_iso_datetime(report_finished_at) if report_finished_at else None,
            duration_ms=max(0.0, observed_duration),
            items_scanned=_int_value(data.get("items_scanned")),
            items_decayed=_int_value(data.get("items_decayed", data.get("decayed"))),
            items_expired=_int_value(data.get("items_expired", data.get("expired"))),
            items_distilled=_int_value(data.get("items_distilled", summaries_created)),
            items_archived=_int_value(data.get("items_archived", data.get("removed"))),
            summaries_created=summaries_created,
            clusters_formed=_int_value(data.get("clusters_formed")),
            contradictions_found=_int_value(data.get("contradictions_found")),
            diagnostics=dict(diagnostics or {}),
            warnings=backend_warnings,
        )

    @classmethod
    def no_op(
        cls,
        *,
        backend: str,
        started_at: datetime | str | None = None,
        duration_ms: float = 0.0,
        items_scanned: int = 0,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> "ConsolidationReport":
        """Build a report for an intentionally no-op consolidation pass."""
        return cls(
            backend=backend,
            started_at=_iso_datetime(started_at),
            duration_ms=max(0.0, float(duration_ms)),
            items_scanned=max(0, int(items_scanned)),
            diagnostics=dict(diagnostics or {}),
        )

    @classmethod
    def warning(
        cls,
        *,
        backend: str,
        warning: str,
        started_at: datetime | str | None = None,
        duration_ms: float = 0.0,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> "ConsolidationReport":
        """Build a non-fatal warning report for unavailable or failed backends."""
        return cls(
            backend=backend,
            started_at=_iso_datetime(started_at),
            duration_ms=max(0.0, float(duration_ms)),
            diagnostics=dict(diagnostics or {}),
            warnings=[warning],
        )


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


@runtime_checkable
class MemoryConsolidationBackendProtocol(Protocol):
    """Optional protocol for memory backends that can consolidate long-term state."""

    async def consolidate(self) -> ConsolidationReport:
        """Run a backend-owned consolidation pass."""
        ...


RetrieveFn = Callable[..., Awaitable[Any] | Any]
ConsolidateFn = Callable[..., Awaitable[Any] | Any]


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
        consolidate_fn: ConsolidateFn | None = None,
        memory_service: Any | None = None,
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
        self._consolidate_fn = consolidate_fn
        self.memory_service = memory_service
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

    async def consolidate(self) -> ConsolidationReport:
        """Delegate memory consolidation to a configured semantiscan service."""
        started = datetime.now(UTC)
        consolidate_fn = self._resolve_consolidate_fn()
        if consolidate_fn is None:
            return ConsolidationReport.warning(
                backend="semantiscan",
                started_at=started,
                warning=(
                    "semantiscan consolidation unavailable; provide consolidate_fn "
                    "or memory_service= with consolidate()"
                ),
                diagnostics={"configured": False},
            )

        try:
            result = consolidate_fn()
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:
            duration_ms = (datetime.now(UTC) - started).total_seconds() * 1000
            return ConsolidationReport.warning(
                backend="semantiscan",
                started_at=started,
                duration_ms=duration_ms,
                warning=f"semantiscan consolidation failed: {exc}",
                diagnostics={"configured": True, "error_type": type(exc).__name__},
            )

        duration_ms = (datetime.now(UTC) - started).total_seconds() * 1000
        return ConsolidationReport.from_result(
            result,
            backend="semantiscan",
            started_at=started,
            duration_ms=duration_ms,
            diagnostics={"configured": True},
        )

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

    def _resolve_consolidate_fn(self) -> ConsolidateFn | None:
        if self._consolidate_fn is not None:
            return self._consolidate_fn
        if self.memory_service is not None:
            consolidate = getattr(self.memory_service, "consolidate", None)
            if callable(consolidate):
                return consolidate
        return None

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
        typed_citation = self._get(result, "citation", None)
        if typed_citation is not None:
            citation_source = self._source_from_typed_citation(typed_citation)
            if citation_source:
                return citation_source

        source = self._first(metadata, "source", "file_path", "source_file", "path", "uri", "url")
        if source:
            return source
        return str(self._get(result, "source", chunk_id) or chunk_id)

    def _source_from_typed_citation(self, value: Any) -> str | None:
        for key in ("source_file", "source", "path", "source_id", "document_id", "uri", "url"):
            source = self._get(value, key, None)
            if source:
                return str(source)
        return None

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


def _plain_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        for kwargs in ({"mode": "json"}, {}):
            try:
                dumped = model_dump(**kwargs)
            except TypeError:
                continue
            if isinstance(dumped, Mapping):
                return dict(dumped)
    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        dumped = dict_method()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    if is_dataclass(value):
        return asdict(value)

    fields = (
        "backend",
        "run_id",
        "started_at",
        "finished_at",
        "duration_ms",
        "items_scanned",
        "items_decayed",
        "items_expired",
        "items_distilled",
        "items_archived",
        "summaries_created",
        "clusters_formed",
        "contradictions_found",
        "errors",
        "decayed",
        "expired",
        "removed",
    )
    return {
        field_name: getattr(value, field_name)
        for field_name in fields
        if getattr(value, field_name, None) is not None
    }


def _iso_datetime(value: datetime | str | Any | None) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if value:
        return str(value)
    return datetime.now(UTC).isoformat()


def _string_or_none(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _int_value(value: Any) -> int:
    if value is None or value == "":
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "Citation",
    "ConsolidationReport",
    "MemoryBackendProtocol",
    "MemoryConsolidationBackendProtocol",
    "MemoryRecord",
    "SemantiscanMemoryBackend",
]
