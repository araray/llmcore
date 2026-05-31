# src/llmcore/search/models.py
"""Provider-agnostic result models for LLMCore web/data search providers.

This module defines the normalized data structures returned by every
:class:`~llmcore.search.base.BaseSearchProvider` implementation.  They are
intentionally decoupled from any vendor SDK so that consuming applications can
switch between search providers (e.g. Bright Data, a future provider) without
changing how they read results — exactly mirroring how LLM responses are
normalized across :class:`~llmcore.providers.base.BaseProvider` implementations.

Design notes
------------
* Every result inherits from :class:`SearchResultBase`, which carries the common
  ``success`` / ``error`` / ``provider`` / timing / ``cost`` fields and a
  ``to_dict()`` serializer (datetimes → ISO-8601 strings).
* Results never raise on partial failure; a provider returns
  ``success=False`` with a populated ``error`` string instead so that callers
  can branch without exception handling on the hot path.  Hard configuration or
  transport faults are still raised as
  :class:`~llmcore.exceptions.SearchProviderError`.
* The ``raw`` field always preserves the provider's untouched payload so power
  users can extract fields the normalizer does not surface.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

__all__ = [
    "SearchResultBase",
    "SearchItem",
    "WebSearchResult",
    "ScrapeResult",
    "DiscoverItem",
    "DiscoverResult",
    "DatasetInfo",
    "DatasetField",
    "DatasetMetadata",
    "DatasetSnapshot",
]


def _serialize(value: Any) -> Any:
    """Recursively convert datetimes to ISO strings for JSON compatibility.

    Args:
        value: Any value that may contain nested ``datetime`` objects inside
            lists or dictionaries.

    Returns:
        A structurally identical value with every ``datetime`` replaced by its
        ISO-8601 string representation.
    """
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    return value


@dataclass
class SearchResultBase:
    """Common fields shared by all search-provider result objects.

    Attributes:
        success: Whether the operation completed successfully.
        provider: Name of the search provider instance that produced the result
            (e.g. ``"brightdata"``).
        error: Human-readable error message when ``success`` is ``False``;
            otherwise ``None``.
        cost: Optional cost in USD for the operation, when the provider reports
            it.  ``None`` when unknown.
        trigger_sent_at: UTC timestamp when the request was dispatched.
        data_fetched_at: UTC timestamp when the response was received.
    """

    success: bool = True
    provider: str | None = None
    error: str | None = None
    cost: float | None = None
    trigger_sent_at: datetime | None = None
    data_fetched_at: datetime | None = None

    def elapsed_ms(self) -> float | None:
        """Return the wall-clock latency in milliseconds, if both timestamps exist.

        Returns:
            Elapsed milliseconds between ``trigger_sent_at`` and
            ``data_fetched_at``, or ``None`` if either is missing.
        """
        if self.trigger_sent_at and self.data_fetched_at:
            return (self.data_fetched_at - self.trigger_sent_at).total_seconds() * 1000
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result to a JSON-compatible dictionary.

        Returns:
            A dict with all dataclass fields; ``datetime`` values are converted
            to ISO-8601 strings.
        """
        return {k: _serialize(v) for k, v in asdict(self).items()}

    def to_json(self, indent: int | None = None) -> str:
        """Serialize the result to a JSON string.

        Args:
            indent: Optional indentation for pretty-printing.

        Returns:
            JSON string representation of :meth:`to_dict`.
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)


@dataclass
class SearchItem:
    """A single organic result from a web search (SERP).

    Attributes:
        position: 1-based rank within the result set.
        title: Result title.
        url: Destination URL.
        description: Snippet / description text.
        displayed_url: Human-displayed URL (breadcrumb), when available.
    """

    position: int | None = None
    title: str = ""
    url: str = ""
    description: str = ""
    displayed_url: str = ""


@dataclass
class WebSearchResult(SearchResultBase):
    """Normalized result of a web search query.

    Attributes:
        query: The query string that was searched.
        engine: Search engine used (``"google"``, ``"bing"``, ``"yandex"``).
        items: Ranked list of :class:`SearchItem` organic results.
        total_results: Total result count reported by the engine, if available.
        raw: The provider's untouched response payload (parsed JSON or
            ``{"raw_html": ...}`` when the engine returned HTML).
    """

    query: str = ""
    engine: str | None = None
    items: list[SearchItem] = field(default_factory=list)
    total_results: int | None = None
    raw: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize, expanding nested :class:`SearchItem` objects."""
        data = super().to_dict()
        data["items"] = [asdict(i) for i in self.items]
        return data


@dataclass
class ScrapeResult(SearchResultBase):
    """Result of scraping a single URL via an unlocking proxy.

    Attributes:
        url: The URL that was requested.
        content: The scraped payload — an HTML/text string for ``raw`` format
            or a parsed object for ``json`` format.
        response_format: ``"raw"`` or ``"json"`` — how ``content`` is encoded.
        status: Lifecycle status (``"ready"``, ``"error"``, ``"timeout"``).
        root_domain: Registered domain extracted from ``url``, when computed.
        content_char_size: Length of ``content`` when it is a string.
        raw: The provider's untouched response payload, when the provider
            returns structured data alongside the extracted ``content`` (e.g.
            Serper's scrape returns JSON with ``metadata``/``jsonld``). ``None``
            when the provider returns only the raw body.
    """

    url: str = ""
    content: Any = None
    response_format: str = "raw"
    status: str = "ready"
    root_domain: str | None = None
    content_char_size: int | None = None
    raw: Any = None


@dataclass
class DiscoverItem:
    """A single AI-ranked result from the Discover API.

    Attributes:
        title: Result title.
        url: Destination URL.
        description: Snippet / description text.
        relevance_score: AI-assigned relevance score (provider-defined scale),
            when present.
        content: Full-page content (markdown) when ``include_content`` was
            requested; otherwise ``None``.
    """

    title: str = ""
    url: str = ""
    description: str = ""
    relevance_score: float | None = None
    content: str | None = None


@dataclass
class DiscoverResult(SearchResultBase):
    """Normalized result of an AI-ranked Discover search.

    Attributes:
        query: The query string that was searched.
        intent: The intent string used to guide relevance ranking, if any.
        items: Relevance-ranked list of :class:`DiscoverItem`.
        total_results: Number of results returned.
        task_id: Provider-side asynchronous task identifier.
        duration_seconds: Server-side processing time, when reported.
        raw: The provider's untouched response payload.
    """

    query: str = ""
    intent: str | None = None
    items: list[DiscoverItem] = field(default_factory=list)
    total_results: int | None = None
    task_id: str | None = None
    duration_seconds: float | None = None
    raw: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize, expanding nested :class:`DiscoverItem` objects."""
        data = super().to_dict()
        data["items"] = [asdict(i) for i in self.items]
        return data


@dataclass
class DatasetInfo:
    """Summary entry returned when listing available datasets.

    Attributes:
        id: Provider dataset identifier (e.g. ``"gd_l1viktl72bvl7bjuj0"``).
        name: Human-readable dataset name.
        size: Approximate record count, when reported.
    """

    id: str = ""
    name: str = ""
    size: int = 0


@dataclass
class DatasetField:
    """Schema description of a single dataset field.

    Attributes:
        name: Field name.
        type: Field type as reported by the provider (e.g. ``"text"``,
            ``"number"``, ``"url"``, ``"array"``, ``"object"``, ``"boolean"``).
        active: Whether the field is currently active.
        required: Whether the field is required.
        description: Optional human-readable description.
    """

    name: str = ""
    type: str = "text"
    active: bool = True
    required: bool = False
    description: str | None = None


@dataclass
class DatasetMetadata:
    """Field schema for a dataset, used to discover filterable fields.

    Attributes:
        id: Dataset identifier.
        fields: List of :class:`DatasetField` describing each column.
    """

    id: str = ""
    fields: list[DatasetField] = field(default_factory=list)

    def field_names(self) -> list[str]:
        """Return the list of field names in declaration order."""
        return [f.name for f in self.fields]


@dataclass
class DatasetSnapshot(SearchResultBase):
    """Status (and optionally data) of a dataset filter/collection snapshot.

    A snapshot is created by filtering a dataset; the provider builds it
    asynchronously.  ``status`` advances ``scheduled`` → ``building`` →
    ``ready`` (or ``failed``).  When fetched via a download convenience method,
    ``records`` is populated.

    Attributes:
        dataset_id: The dataset this snapshot was created from.
        snapshot_id: Provider snapshot identifier.
        status: Lifecycle status (``"scheduled"``, ``"building"``, ``"ready"``,
            ``"failed"``).
        records: Downloaded records, when available; otherwise ``None``.
        dataset_size: Record count in the snapshot, when reported.
        file_size: Size of the snapshot file in bytes, when reported.
    """

    dataset_id: str | None = None
    snapshot_id: str | None = None
    status: str = "scheduled"
    records: list[dict[str, Any]] | None = None
    dataset_size: int | None = None
    file_size: int | None = None

    @property
    def record_count(self) -> int:
        """Return the number of downloaded records (0 when none fetched)."""
        return len(self.records) if self.records else 0
