# src/llmcore/search/providers/serper_provider.py
"""Serper.dev web/data **search** provider for LLMCore.

Implements :class:`llmcore.search.base.BaseSearchProvider` against the Serper.dev
Google Search API (``https://google.serper.dev``) and its scraping host
(``https://scrape.serper.dev``) using :mod:`httpx` directly — consistent with the
native LLM providers and the sibling Bright Data search provider (no vendor SDK).

Supported capabilities
-----------------------
=================  ============================================================
Capability         Serper.dev surface
=================  ============================================================
``web_search``     ``POST /{type}`` where ``type`` is one of search, news,
                   images, videos, shopping, scholar, patents, maps, places,
                   autocomplete. Body: ``{q, gl, hl, num, page, tbs,
                   autocorrect, location}``.
``batch_search``   ``POST /{type}`` with a JSON **array** of query objects
                   (Serper runs them in one request) -> list of results.
``scrape``         ``POST https://scrape.serper.dev`` with
                   ``{url, includeMarkdown}`` -> structured JSON (markdown/text
                   + metadata).
=================  ============================================================

Serper does not provide an AI-ranked Discover API or a dataset marketplace, so
``discover`` and ``dataset_search`` are intentionally **not** advertised
(calling them raises ``NotImplementedError`` from the base class).

Authentication
--------------
Serper authenticates with an ``X-API-KEY`` header (NOT a Bearer token). Resolve
the key via ``[search_providers.serper].api_key`` or, preferably, the
``SERPER_API_KEY`` environment variable.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
from datetime import datetime, timezone
from typing import Any

from ...exceptions import ConfigError, SearchProviderError
from ..base import BaseSearchProvider, SearchCapability
from ..models import ScrapeResult, SearchItem, WebSearchResult

try:  # pragma: no cover - exercised indirectly
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]
    _HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)

# --- Defaults ---------------------------------------------------------------
DEFAULT_BASE_URL = "https://google.serper.dev"
DEFAULT_SCRAPE_URL = "https://scrape.serper.dev"
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_SEARCH_TYPE = "search"
DEFAULT_TOKEN_ENV_VAR = "SERPER_API_KEY"

# Search verticals Serper exposes as distinct endpoints (``POST /{type}``).
SERPER_SEARCH_TYPES = (
    "search",
    "news",
    "images",
    "videos",
    "shopping",
    "scholar",
    "patents",
    "maps",
    "places",
    "autocomplete",
)

# Which response key holds the primary result array for a given vertical.
# Anything not listed (search, scholar, patents, autocomplete) falls back to
# ``organic`` during normalization.
_RESULTS_KEY_BY_TYPE = {
    "news": "news",
    "images": "images",
    "videos": "videos",
    "shopping": "shopping",
    "places": "places",
    "maps": "places",
}


def _normalize_serper(data: Any, search_type: str) -> tuple[list[SearchItem], int | None]:
    """Normalize a Serper response into provider-agnostic :class:`SearchItem` objects.

    Serper returns Google-shaped JSON. The organic results live under
    ``organic`` for web/scholar/patents searches, and under a vertical-specific
    key (``news``, ``images``, ``videos``, ``shopping``, ``places``) otherwise.
    Only the common fields (title, link, snippet, position) are mapped; the full
    payload — including ``knowledgeGraph``, ``peopleAlsoAsk``, ``relatedSearches``
    and any vertical-specific fields — is preserved on ``WebSearchResult.raw``.

    Args:
        data: Parsed Serper response (dict) or ``None``.
        search_type: The vertical that was queried.

    Returns:
        Tuple of (mapped items, total_results-or-None). Serper does not report a
        numeric total in its standard payload, so the total is ``None``.
    """
    if not isinstance(data, dict):
        return [], None

    key = _RESULTS_KEY_BY_TYPE.get(search_type, "organic")
    arr = data.get(key)
    if not isinstance(arr, list):
        arr = data.get("organic") if isinstance(data.get("organic"), list) else []

    items: list[SearchItem] = []
    for i, item in enumerate(arr, start=1):
        if not isinstance(item, dict):
            continue
        items.append(
            SearchItem(
                position=item.get("position", i),
                title=item.get("title", ""),
                url=item.get("link", item.get("url", "")),
                description=item.get("snippet", item.get("description", "")),
                displayed_url=item.get("displayedLink", item.get("displayed_url", "")),
            )
        )
    return items, None


def _root_domain(url: str) -> str | None:
    """Extract a best-effort registered domain (last two labels) from *url*."""
    try:
        from urllib.parse import urlparse

        host = urlparse(url).netloc.split("@")[-1].split(":")[0]
        parts = host.split(".")
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return host or None
    except Exception:  # pragma: no cover - defensive
        return None


class SerperSearchProvider(BaseSearchProvider):
    """Serper.dev implementation of :class:`BaseSearchProvider`.

    Args:
        config: Settings from ``[search_providers.serper]``. Recognized keys:
            ``api_key`` / ``api_key_env_var``, ``base_url``, ``scrape_base_url``,
            ``default_search_type``, ``timeout``, ``max_retries``, ``ssl_verify``.
        log_raw_payloads: Whether to log raw request/response payloads.

    Raises:
        ConfigError: If ``httpx`` is not installed or no API key is found.
    """

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False) -> None:
        super().__init__(config, log_raw_payloads)

        if not _HTTPX_AVAILABLE:
            raise ConfigError(
                "The 'httpx' package is required for the Serper search provider. "
                "Install with: pip install llmcore[serper]"
            )

        # --- API key ---
        key = config.get("api_key") or config.get("token")
        if not key:
            env_var = config.get("api_key_env_var", DEFAULT_TOKEN_ENV_VAR)
            import os

            key = os.environ.get(env_var) or os.environ.get(DEFAULT_TOKEN_ENV_VAR)
        if not key:
            raise ConfigError(
                "Serper API key not found. Set SERPER_API_KEY or configure "
                "search_providers.serper.api_key / api_key_env_var."
            )
        self._api_key = str(key).strip()

        # --- Endpoints / tuning ---
        self._base_url = str(config.get("base_url", DEFAULT_BASE_URL)).rstrip("/")
        self._scrape_url = str(config.get("scrape_base_url", DEFAULT_SCRAPE_URL)).rstrip("/")

        search_type = str(config.get("default_search_type", DEFAULT_SEARCH_TYPE)).lower()
        if search_type not in SERPER_SEARCH_TYPES:
            logger.warning(
                "Unsupported default_search_type '%s' for Serper; falling back to '%s'.",
                search_type,
                DEFAULT_SEARCH_TYPE,
            )
            search_type = DEFAULT_SEARCH_TYPE
        self._default_search_type = search_type

        self._timeout = int(config.get("timeout", DEFAULT_TIMEOUT))
        self._max_retries = int(config.get("max_retries", DEFAULT_MAX_RETRIES))
        self._ssl_verify = bool(config.get("ssl_verify", True))

        self._client: Any | None = None  # httpx.AsyncClient, created lazily
        logger.debug(
            "SerperSearchProvider initialized (base_url=%s, scrape_url=%s, default_type=%s).",
            self._base_url,
            self._scrape_url,
            self._default_search_type,
        )

    # ------------------------------------------------------------------
    # Identity / capabilities
    # ------------------------------------------------------------------
    def get_name(self) -> str:
        """Return the provider type name (``"serper"``)."""
        return "serper"

    def get_capabilities(self) -> set[str]:
        """Return the set of supported capability strings."""
        return {
            SearchCapability.WEB_SEARCH.value,
            SearchCapability.BATCH_SEARCH.value,
            SearchCapability.SCRAPE.value,
        }

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------
    def _get_client(self) -> Any:
        """Return the lazily-created shared :class:`httpx.AsyncClient`.

        ``base_url`` is set to the search host; the scrape host is passed as an
        absolute URL per request (httpx uses absolute URLs as-is).
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={
                    "X-API-KEY": self._api_key,
                    "Content-Type": "application/json",
                    "User-Agent": "llmcore-serper-search",
                },
                timeout=self._timeout,
                verify=self._ssl_verify,
            )
        return self._client

    def _log_payload(self, label: str, payload: Any) -> None:
        """Emit a raw-payload debug log when raw logging is enabled."""
        if self.log_raw_payloads_enabled:
            try:
                logger.debug("[serper] %s: %s", label, _json.dumps(payload, default=str)[:4000])
            except Exception:  # pragma: no cover - defensive
                logger.debug("[serper] %s: <unserializable>", label)

    async def _post(self, url: str, body: Any) -> Any:
        """POST a JSON body with retry on 429 / 5xx / transport faults.

        Args:
            url: Endpoint path (relative to ``base_url``) or an absolute URL
                (used for the scrape host).
            body: JSON-serializable request body (dict for single, list for batch).

        Returns:
            The raw :class:`httpx.Response`.

        Raises:
            SearchProviderError: On authentication failure or repeated transport
                / rate-limit / server errors.
        """
        client = self._get_client()
        self._log_payload(f"POST {url} body", body)
        last_exc: Exception | None = None
        attempts = max(1, self._max_retries)
        for attempt in range(attempts):
            try:
                response = await client.post(url, json=body)
            except Exception as exc:  # httpx.TransportError, TimeoutException, …
                last_exc = exc
                if attempt < attempts - 1:
                    await asyncio.sleep(min(2**attempt, 8))
                    continue
                raise SearchProviderError(
                    self.get_name(), f"Transport error calling {url}: {exc}"
                ) from exc

            if response.status_code in (401, 403):
                raise SearchProviderError(
                    self.get_name(),
                    f"Authentication failed for {url}: {response.text[:200]}",
                    status_code=response.status_code,
                )
            # Retry on rate limit (429) and server (5xx) faults.
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < attempts - 1:
                    last_exc = SearchProviderError(
                        self.get_name(),
                        f"Retryable status {response.status_code} for {url}",
                        status_code=response.status_code,
                    )
                    await asyncio.sleep(min(2**attempt, 8))
                    continue
            return response

        if isinstance(last_exc, SearchProviderError):
            raise last_exc
        raise SearchProviderError(self.get_name(), f"Request to {url} failed after retries.")

    @staticmethod
    def _decode(response: Any) -> Any:
        """Best-effort decode of an httpx response into JSON or text."""
        text = response.text
        if not text:
            return {}
        try:
            return response.json()
        except Exception:
            try:
                return _json.loads(text)
            except (ValueError, TypeError):
                return text

    def _error_message(self, response: Any) -> str:
        """Extract Serper's ``message`` from an error response, if present."""
        data = self._decode(response)
        if isinstance(data, dict) and data.get("message"):
            return str(data["message"])
        return response.text[:200]

    def _resolve_search_type(self, search_type: str | None) -> str:
        """Validate/normalize a search vertical, raising on unknown values."""
        st = (search_type or self._default_search_type).lower()
        if st not in SERPER_SEARCH_TYPES:
            raise SearchProviderError(
                self.get_name(),
                f"Unsupported search_type '{st}'. Supported: {', '.join(SERPER_SEARCH_TYPES)}.",
            )
        return st

    # ------------------------------------------------------------------
    # web_search
    # ------------------------------------------------------------------
    def _build_query_object(
        self,
        query: str,
        *,
        count: int | None,
        country: str | None,
        language: str | None,
        time_range: str | None,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        """Assemble a Serper query object from normalized parameters."""
        q: dict[str, Any] = {"q": query}
        if count:
            q["num"] = count
        if country:
            q["gl"] = country
        if language:
            q["hl"] = language
        if time_range and "tbs" not in extra:
            q["tbs"] = f"qdr:{time_range}"
        # Pass-through recognized Serper params (page, tbs, autocorrect, location, …).
        q.update(extra)
        return q

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
        """Run a Serper search and return normalized organic results.

        ``device``, ``engine`` and ``mode`` are accepted for cross-provider
        API compatibility but ignored (Serper is Google-only and synchronous).
        Use ``search_type`` (kwarg) to select a vertical and ``tbs`` /
        ``time_range`` for time filtering.

        Args:
            query: The search query string (``q``).
            count: Number of results (``num``).
            country: ISO country code (``gl``).
            language: ISO language code (``hl``).
            device: Ignored (compatibility only).
            engine: Ignored (compatibility only).
            mode: Ignored (compatibility only).
            **kwargs: ``search_type`` (vertical), ``page``, ``tbs``,
                ``time_range`` (shorthand for ``tbs=qdr:<x>``), ``autocorrect``,
                ``location``, and any other Serper body parameter.

        Returns:
            A :class:`~llmcore.search.models.WebSearchResult`.
        """
        if not query or not isinstance(query, str):
            raise SearchProviderError(
                self.get_name(), "web_search query must be a non-empty string."
            )

        search_type = self._resolve_search_type(kwargs.pop("search_type", None))
        time_range = kwargs.pop("time_range", None)
        body = self._build_query_object(
            query,
            count=count,
            country=country,
            language=language,
            time_range=time_range,
            extra=kwargs,
        )

        trigger_at = datetime.now(timezone.utc)
        with self._span("web_search", search_type=search_type):
            response = await self._post(f"/{search_type}", body)
        fetched_at = datetime.now(timezone.utc)

        if response.status_code != 200:
            return WebSearchResult(
                success=False,
                provider=self._instance_label(),
                query=query,
                engine=f"serper:{search_type}",
                error=f"/{search_type} failed (HTTP {response.status_code}): "
                f"{self._error_message(response)}",
                trigger_sent_at=trigger_at,
                data_fetched_at=fetched_at,
            )

        payload = self._decode(response)
        return self._to_web_result(payload, query, search_type, trigger_at, fetched_at)

    def _to_web_result(
        self,
        payload: Any,
        query: str,
        search_type: str,
        trigger_at: datetime,
        fetched_at: datetime,
    ) -> WebSearchResult:
        """Build a :class:`WebSearchResult` from a decoded Serper payload."""
        items, total = _normalize_serper(payload, search_type)
        cost = payload.get("credits") if isinstance(payload, dict) else None
        return WebSearchResult(
            success=True,
            provider=self._instance_label(),
            query=query,
            engine=f"serper:{search_type}",
            items=items,
            total_results=total,
            raw=payload,
            cost=cost,
            trigger_sent_at=trigger_at,
            data_fetched_at=fetched_at,
        )

    # ------------------------------------------------------------------
    # batch_search (Serper accepts an array of query objects)
    # ------------------------------------------------------------------
    async def batch_search(
        self,
        queries: list[Any],
        *,
        count: int = 10,
        country: str | None = None,
        language: str = "en",
        search_type: str | None = None,
        **kwargs: Any,
    ) -> list[WebSearchResult]:
        """Run multiple searches in one request via Serper's array payload.

        Args:
            queries: A list of query strings, or a list of Serper query-object
                dicts (e.g. ``{"q": ..., "tbs": "qdr:m", "page": 2}``). Strings
                inherit ``count`` / ``country`` / ``language`` / ``time_range``.
            count: Default ``num`` for string queries.
            country: Default ``gl`` for string queries.
            language: Default ``hl`` for string queries.
            search_type: Vertical for the whole batch (default provider setting).
            **kwargs: Applied to string queries (e.g. ``time_range``, ``page``).

        Returns:
            A list of :class:`~llmcore.search.models.WebSearchResult`, one per
            input query, in order.
        """
        if not isinstance(queries, (list, tuple)) or not queries:
            raise SearchProviderError(
                self.get_name(), "batch_search requires a non-empty list of queries."
            )

        st = self._resolve_search_type(search_type)
        time_range = kwargs.pop("time_range", None)

        body: list[dict[str, Any]] = []
        labels: list[str] = []
        for item in queries:
            if isinstance(item, str):
                obj = self._build_query_object(
                    item,
                    count=count,
                    country=country,
                    language=language,
                    time_range=time_range,
                    extra=dict(kwargs),
                )
            elif isinstance(item, dict):
                obj = dict(item)  # assume a ready Serper query object
                if "q" not in obj:
                    raise SearchProviderError(
                        self.get_name(), "Each batch query dict must include a 'q' field."
                    )
            else:
                raise SearchProviderError(
                    self.get_name(), f"Unsupported batch query item type: {type(item).__name__}."
                )
            body.append(obj)
            labels.append(str(obj.get("q", "")))

        trigger_at = datetime.now(timezone.utc)
        with self._span("batch_search", search_type=st, batch_size=len(body)):
            response = await self._post(f"/{st}", body)
        fetched_at = datetime.now(timezone.utc)

        if response.status_code != 200:
            err = (
                f"/{st} batch failed (HTTP {response.status_code}): {self._error_message(response)}"
            )
            return [
                WebSearchResult(
                    success=False,
                    provider=self._instance_label(),
                    query=label,
                    engine=f"serper:{st}",
                    error=err,
                    trigger_sent_at=trigger_at,
                    data_fetched_at=fetched_at,
                )
                for label in labels
            ]

        decoded = self._decode(response)
        # Serper returns a list aligned with the request; tolerate a single dict.
        if isinstance(decoded, dict):
            decoded = [decoded]
        results: list[WebSearchResult] = []
        for idx, label in enumerate(labels):
            payload = decoded[idx] if isinstance(decoded, list) and idx < len(decoded) else {}
            results.append(self._to_web_result(payload, label, st, trigger_at, fetched_at))
        return results

    # ------------------------------------------------------------------
    # scrape (separate host: scrape.serper.dev)
    # ------------------------------------------------------------------
    async def scrape(
        self,
        url: str,
        *,
        response_format: str = "markdown",
        country: str | None = None,
        method: str = "GET",
        mode: str = "sync",
        **kwargs: Any,
    ) -> ScrapeResult:
        """Scrape a single URL via ``scrape.serper.dev`` (returns structured JSON).

        ``country``, ``method`` and ``mode`` are accepted for cross-provider API
        compatibility but ignored by Serper.

        Args:
            url: The target URL.
            response_format: ``"markdown"`` (default) requests and extracts
                markdown; ``"json"`` / ``"raw"`` return the full structured
                payload in ``content``.
            country: Ignored (compatibility only).
            method: Ignored (compatibility only).
            mode: Ignored (compatibility only).
            **kwargs: Extra Serper scrape body params (e.g. an optional ``q``).

        Returns:
            A :class:`~llmcore.search.models.ScrapeResult`. ``raw`` always holds
            Serper's full JSON payload (text/markdown/metadata/jsonld).
        """
        if not url or not isinstance(url, str):
            raise SearchProviderError(self.get_name(), "scrape url must be a non-empty string.")
        if response_format not in ("markdown", "json", "raw"):
            raise SearchProviderError(
                self.get_name(), "response_format must be 'markdown', 'json', or 'raw'."
            )

        body: dict[str, Any] = {"url": url}
        if response_format == "markdown":
            body["includeMarkdown"] = True
        body.update(kwargs)  # forward-compat: includeLinks, q, etc.

        trigger_at = datetime.now(timezone.utc)
        with self._span("scrape", response_format=response_format):
            response = await self._post(self._scrape_url, body)
        fetched_at = datetime.now(timezone.utc)

        if response.status_code != 200:
            return ScrapeResult(
                success=False,
                provider=self._instance_label(),
                url=url,
                status="error",
                response_format=response_format,
                error=f"scrape failed (HTTP {response.status_code}): {self._error_message(response)}",
                trigger_sent_at=trigger_at,
                data_fetched_at=fetched_at,
            )

        payload = self._decode(response)
        content: Any = payload
        fmt = "json"
        if response_format == "markdown" and isinstance(payload, dict):
            md = payload.get("markdown")
            txt = payload.get("text")
            if isinstance(md, str):
                content, fmt = md, "markdown"
            elif isinstance(txt, str):
                content, fmt = txt, "text"
        char_size = len(content) if isinstance(content, str) else None

        return ScrapeResult(
            success=True,
            provider=self._instance_label(),
            url=url,
            content=content,
            response_format=fmt,
            status="ready",
            root_domain=_root_domain(url),
            content_char_size=char_size,
            raw=payload,
            trigger_sent_at=trigger_at,
            data_fetched_at=fetched_at,
        )

    # ------------------------------------------------------------------
    # health / lifecycle
    # ------------------------------------------------------------------
    async def health_check(self) -> bool:
        """Verify credentials/connectivity with a minimal search.

        Note:
            Serper exposes no dedicated health/balance endpoint, so this issues a
            tiny ``/search`` request (``num=1``) and therefore consumes **one
            credit**. Returns ``True`` on HTTP 200, ``False`` on auth failure or
            unreachable host; never raises.
        """
        try:
            response = await self._post("/search", {"q": "ping", "num": 1})
        except SearchProviderError:
            return False
        return response.status_code == 200

    async def close(self) -> None:
        """Close the underlying :class:`httpx.AsyncClient` (idempotent)."""
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Error closing Serper httpx client: %s", exc)
            self._client = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _instance_label(self) -> str:
        """Return the configured instance name, falling back to the type name."""
        return self._provider_instance_name or self.get_name()
