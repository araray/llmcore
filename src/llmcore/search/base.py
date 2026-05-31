# src/llmcore/search/base.py
"""Abstract base class for LLMCore web/data **search** providers.

This module defines the contract that every search-provider implementation
(e.g. Bright Data) must satisfy.  It is the search-side analogue of
:class:`llmcore.providers.base.BaseProvider`: the same library exposes a second
family of pluggable backends so that *whoever consumes llmcore can use web /
data search the same way they use LLM providers* — config-driven, discovered
through a manager, and accessed via a uniform interface.

Why a separate hierarchy (not ``BaseProvider``)?
------------------------------------------------
A search provider has **no** concept of ``chat_completion``, token counting, or
context windows, so forcing it into :class:`BaseProvider` would require stubbing
out the entire LLM contract and would violate the Liskov substitution principle
for every consumer that iterates LLM providers.  Keeping search providers in
their own ABC preserves llmcore's "clean, general-purpose framework" invariant:
each interface stays cohesive and minimal.

Capability model
----------------
Concrete providers declare which operations they support via
:meth:`get_capabilities`.  Every operation method has a default implementation
that raises :class:`NotImplementedError`; providers override only what they
support.  This mirrors how :class:`BaseProvider` treats optional modalities
(TTS / STT / image / OCR / embeddings).
"""

from __future__ import annotations

import abc
import logging
from contextlib import contextmanager
from enum import Enum
from typing import Any

from .models import (
    DatasetInfo,
    DatasetMetadata,
    DatasetSnapshot,
    DiscoverResult,
    ScrapeResult,
    WebSearchResult,
)

logger = logging.getLogger(__name__)


class SearchCapability(str, Enum):
    """Operations a search provider may support.

    Returned (as a set) from :meth:`BaseSearchProvider.get_capabilities` so
    callers can introspect a provider before invoking an operation.
    """

    WEB_SEARCH = "web_search"
    """Search engine results (SERP) → :class:`~llmcore.search.models.WebSearchResult`."""

    SCRAPE = "scrape"
    """Fetch a single URL through an unlocking proxy → ``ScrapeResult``."""

    DISCOVER = "discover"
    """AI-relevance-ranked web search → ``DiscoverResult``."""

    DATASET_SEARCH = "dataset_search"
    """Filter/collect structured records from datasets → ``DatasetSnapshot``."""

    CRAWL = "crawl"
    """Crawl/discover an entire domain (optional, provider-specific)."""


class BaseSearchProvider(abc.ABC):
    """Abstract base class for web/data search provider integrations.

    Concrete subclasses implement HTTP access to a specific vendor and override
    only the capability methods they support.  Instances are created by
    :class:`llmcore.search.manager.SearchProviderManager` from a
    ``[search_providers.<name>]`` configuration section.

    Attributes:
        log_raw_payloads_enabled: Whether to log raw request/response payloads
            (parity with :class:`llmcore.providers.base.BaseProvider`).
    """

    log_raw_payloads_enabled: bool

    @abc.abstractmethod
    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False) -> None:
        """Initialize the provider from its configuration section.

        Args:
            config: Provider-specific settings loaded from
                ``[search_providers.<name>]`` (e.g. ``api_key``, ``base_url``,
                zone names, ``timeout``).  The manager injects an
                ``_instance_name`` key identifying the section name.
            log_raw_payloads: Whether raw payloads should be logged by this
                instance.

        Raises:
            ConfigError: If required configuration (such as an API key) is
                missing.  The manager catches this and skips the provider.
        """
        self.log_raw_payloads_enabled = log_raw_payloads
        self._provider_instance_name: str | None = config.get("_instance_name")

    # ------------------------------------------------------------------
    # Identity / capabilities (required)
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def get_name(self) -> str:
        """Return the unique identifier for this provider type (e.g. ``"brightdata"``)."""

    @abc.abstractmethod
    def get_capabilities(self) -> set[str]:
        """Return the set of supported :class:`SearchCapability` values (as strings)."""

    def supports(self, capability: str | SearchCapability) -> bool:
        """Return ``True`` if this provider supports *capability*.

        Args:
            capability: A :class:`SearchCapability` member or its string value.

        Returns:
            Whether the capability is advertised by :meth:`get_capabilities`.
        """
        value = capability.value if isinstance(capability, SearchCapability) else capability
        return value in self.get_capabilities()

    # ------------------------------------------------------------------
    # Optional capability operations (default: NotImplementedError)
    # ------------------------------------------------------------------
    async def web_search(
        self,
        query: str,
        *,
        count: int = 10,
        country: str | None = None,
        language: str = "en",
        device: str = "desktop",
        engine: str | None = None,
        mode: str = "sync",
        **kwargs: Any,
    ) -> WebSearchResult:
        """Run a web search (SERP) and return normalized organic results.

        Args:
            query: The search query string.
            count: Desired number of organic results.
            country: Two-letter ISO country code for geolocated results.
            language: Two-letter language code (e.g. ``"en"``).
            device: ``"desktop"`` or ``"mobile"``.
            engine: Search engine to use (provider-specific; e.g. ``"google"``,
                ``"bing"``, ``"yandex"``).  ``None`` uses the provider default.
            mode: ``"sync"`` (blocking) or ``"async"`` (trigger + poll).
            **kwargs: Provider-specific extras (e.g. ``safe_search``,
                ``time_range``).

        Returns:
            A :class:`~llmcore.search.models.WebSearchResult`.

        Raises:
            NotImplementedError: If the provider does not support web search.
            SearchProviderError: On configuration or transport faults.
        """
        raise NotImplementedError(
            f"Search provider '{self.get_name()}' does not support web search."
        )

    async def scrape(
        self,
        url: str,
        *,
        response_format: str = "raw",
        country: str | None = None,
        method: str = "GET",
        mode: str = "sync",
        **kwargs: Any,
    ) -> ScrapeResult:
        """Fetch a single URL through an unlocking proxy.

        Args:
            url: The target URL.
            response_format: ``"raw"`` for HTML/text or ``"json"`` for the
                provider's structured payload.
            country: Two-letter ISO country code for the exit node.
            method: HTTP method to use for the upstream request.
            mode: ``"sync"`` (blocking) or ``"async"`` (trigger + poll).
            **kwargs: Provider-specific extras.

        Returns:
            A :class:`~llmcore.search.models.ScrapeResult`.

        Raises:
            NotImplementedError: If the provider does not support scraping.
            SearchProviderError: On configuration or transport faults.
        """
        raise NotImplementedError(
            f"Search provider '{self.get_name()}' does not support URL scraping."
        )

    async def discover(
        self,
        query: str,
        *,
        intent: str | None = None,
        include_content: bool = False,
        country: str | None = None,
        city: str | None = None,
        language: str | None = None,
        filter_keywords: list[str] | None = None,
        count: int | None = None,
        timeout: int = 60,
        poll_interval: int = 2,
        **kwargs: Any,
    ) -> DiscoverResult:
        """Run an AI-relevance-ranked web search (Discover API).

        Args:
            query: The search query string.
            intent: Why you are searching — guides AI relevance ranking.
            include_content: If ``True``, include full-page content (markdown).
            country: Country code for localized results.
            city: City for localized results.
            language: Language code for localized results.
            filter_keywords: Keywords used to filter results.
            count: Number of results to return.
            timeout: Maximum seconds to wait for the async task.
            poll_interval: Seconds between status polls.
            **kwargs: Provider-specific extras.

        Returns:
            A :class:`~llmcore.search.models.DiscoverResult`.

        Raises:
            NotImplementedError: If the provider does not support discovery.
            SearchProviderError: On configuration or transport faults.
        """
        raise NotImplementedError(
            f"Search provider '{self.get_name()}' does not support discovery."
        )

    async def list_datasets(self) -> list[DatasetInfo]:
        """List datasets available to the account.

        Returns:
            A list of :class:`~llmcore.search.models.DatasetInfo`.

        Raises:
            NotImplementedError: If the provider does not support datasets.
        """
        raise NotImplementedError(f"Search provider '{self.get_name()}' does not support datasets.")

    async def dataset_metadata(self, dataset_id: str) -> DatasetMetadata:
        """Return the field schema for *dataset_id*.

        Args:
            dataset_id: The dataset identifier.

        Returns:
            A :class:`~llmcore.search.models.DatasetMetadata`.

        Raises:
            NotImplementedError: If the provider does not support datasets.
        """
        raise NotImplementedError(f"Search provider '{self.get_name()}' does not support datasets.")

    async def dataset_filter(
        self,
        dataset_id: str,
        filter: dict[str, Any],
        *,
        records_limit: int | None = None,
        **kwargs: Any,
    ) -> DatasetSnapshot:
        """Create a snapshot by filtering *dataset_id* (returns immediately).

        Args:
            dataset_id: The dataset to filter.
            filter: Filter criteria (provider-specific schema).
            records_limit: Maximum number of records to collect.
            **kwargs: Provider-specific extras.

        Returns:
            A :class:`~llmcore.search.models.DatasetSnapshot` containing the
            ``snapshot_id`` and initial ``status`` (records not yet fetched).

        Raises:
            NotImplementedError: If the provider does not support datasets.
            SearchProviderError: On configuration or transport faults.
        """
        raise NotImplementedError(f"Search provider '{self.get_name()}' does not support datasets.")

    async def dataset_status(self, snapshot_id: str) -> DatasetSnapshot:
        """Return the current status of a dataset snapshot.

        Args:
            snapshot_id: The snapshot identifier.

        Returns:
            A :class:`~llmcore.search.models.DatasetSnapshot` with the latest
            ``status`` (records not fetched).

        Raises:
            NotImplementedError: If the provider does not support datasets.
        """
        raise NotImplementedError(f"Search provider '{self.get_name()}' does not support datasets.")

    async def dataset_download(
        self,
        snapshot_id: str,
        *,
        format: str = "jsonl",
        timeout: int = 300,
        poll_interval: int = 5,
        **kwargs: Any,
    ) -> DatasetSnapshot:
        """Poll a snapshot until ready, then download and return its records.

        Args:
            snapshot_id: The snapshot identifier.
            format: Download format (``"json"``, ``"jsonl"``, or ``"csv"``).
            timeout: Maximum seconds to wait for the snapshot to be ready.
            poll_interval: Seconds between status polls.
            **kwargs: Provider-specific extras.

        Returns:
            A :class:`~llmcore.search.models.DatasetSnapshot` with ``records``
            populated.

        Raises:
            NotImplementedError: If the provider does not support datasets.
            SearchProviderError: On configuration or transport faults.
        """
        raise NotImplementedError(f"Search provider '{self.get_name()}' does not support datasets.")

    async def dataset_search(
        self,
        dataset_id: str,
        filter: dict[str, Any],
        *,
        records_limit: int | None = None,
        format: str = "jsonl",
        timeout: int = 300,
        poll_interval: int = 5,
        **kwargs: Any,
    ) -> DatasetSnapshot:
        """Convenience: filter a dataset, wait for the snapshot, download records.

        The default implementation composes :meth:`dataset_filter` and
        :meth:`dataset_download`.  Providers may override for efficiency.

        Args:
            dataset_id: The dataset to filter.
            filter: Filter criteria (provider-specific schema).
            records_limit: Maximum number of records to collect.
            format: Download format (``"json"``, ``"jsonl"``, or ``"csv"``).
            timeout: Maximum seconds to wait for the snapshot to be ready.
            poll_interval: Seconds between status polls.
            **kwargs: Provider-specific extras forwarded to :meth:`dataset_filter`.

        Returns:
            A :class:`~llmcore.search.models.DatasetSnapshot` with ``records``
            populated.

        Raises:
            NotImplementedError: If the provider does not support datasets.
            SearchProviderError: On configuration or transport faults.
        """
        snapshot = await self.dataset_filter(
            dataset_id, filter, records_limit=records_limit, **kwargs
        )
        if not snapshot.success or not snapshot.snapshot_id:
            return snapshot
        return await self.dataset_download(
            snapshot.snapshot_id,
            format=format,
            timeout=timeout,
            poll_interval=poll_interval,
        )

    async def crawl(self, url: str, **kwargs: Any) -> Any:
        """Crawl/discover an entire domain (optional, provider-specific).

        Args:
            url: The seed URL/domain to crawl.
            **kwargs: Provider-specific crawl parameters.

        Raises:
            NotImplementedError: If the provider does not support crawling.
        """
        raise NotImplementedError(f"Search provider '{self.get_name()}' does not support crawling.")

    # ------------------------------------------------------------------
    # Lifecycle / health
    # ------------------------------------------------------------------
    async def health_check(self) -> bool:
        """Lightweight connectivity/credential check.

        Returns:
            ``True`` if the provider can reach its backend with valid
            credentials, ``False`` otherwise.  Never raises.
        """
        return False

    async def close(self) -> None:
        """Release any resources (e.g. HTTP sessions).

        The default is a no-op; providers that hold network clients override
        this.  Idempotent.
        """
        return None

    # ------------------------------------------------------------------
    # Observability (best-effort; safe no-op when tracing is unavailable)
    # ------------------------------------------------------------------
    @contextmanager
    def _span(self, operation: str, **attributes: Any):
        """Create a tracing span for a search operation (no-op on failure).

        Mirrors :meth:`llmcore.providers.base.BaseProvider._create_llm_span` but
        for the search subsystem.  Falls back to a null context if the tracing
        backend is not configured/available.

        Args:
            operation: Operation name (e.g. ``"web_search"``).
            **attributes: Additional span attributes.

        Yields:
            The span object, or ``None`` when tracing is unavailable.
        """
        span_cm = None
        try:
            from ..tracing import create_span, get_tracer

            tracer = get_tracer(f"llmcore.search.{self.get_name()}")
            span_cm = create_span(
                tracer,
                f"search.{operation}",
                **{
                    "search.provider": self.get_name(),
                    "search.operation": operation,
                    **attributes,
                },
            )
        except Exception as exc:  # pragma: no cover - tracing is optional
            logger.debug("Search span creation failed (non-fatal): %s", exc)

        if span_cm is None:
            from contextlib import nullcontext

            with nullcontext() as span:
                yield span
        else:
            with span_cm as span:
                yield span
