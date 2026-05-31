# src/llmcore/search/providers/brightdata_provider.py
"""Bright Data web/data **search** provider for LLMCore.

Implements :class:`llmcore.search.base.BaseSearchProvider` against Bright Data's
public REST API (``https://api.brightdata.com``) using :mod:`httpx` directly —
**no dependency on the Bright Data SDK**.  This keeps llmcore a clean,
self-contained framework (the same way the native Gemini/Kimi/DeepInfra
providers talk to their vendors over ``httpx`` rather than vendoring a SDK).

Supported capabilities
-----------------------
=================  ============================================================
Capability         Bright Data product / endpoint
=================  ============================================================
``web_search``     SERP API — ``POST /request`` (sync) or
                   ``POST /unblocker/req`` + ``GET /unblocker/get_result``
                   (async).  Search URLs carry ``brd_json=1`` for parsed JSON.
``scrape``         Web Unlocker API — ``POST /request`` (sync) / async pair.
``discover``       Discover API — ``POST /discover`` + ``GET /discover``.
``dataset_search`` Marketplace Dataset API — ``POST /datasets/filter``,
                   ``GET /datasets/snapshots/{id}``,
                   ``GET /datasets/snapshots/{id}/download``,
                   ``GET /datasets/list``, ``GET /datasets/{id}/metadata``.
=================  ============================================================

Authentication
--------------
A single API token (Bearer) authenticates every endpoint.  Resolve it via
``[search_providers.brightdata].api_key`` or, preferably, the
``BRIGHTDATA_API_TOKEN`` environment variable.

Zones
-----
The SERP and Web Unlocker products run against **zones** you create in the
Bright Data control panel.  Unlike the vendor SDK, this provider does **not**
auto-create zones (that would silently mutate your account); configure
``serp_zone`` / ``unlocker_zone`` and the provider will use them.  Dataset and
Discover operations do not require a zone.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote_plus

from ...exceptions import ConfigError, SearchProviderError
from ..base import BaseSearchProvider, SearchCapability
from ..models import (
    DatasetField,
    DatasetInfo,
    DatasetMetadata,
    DatasetSnapshot,
    DiscoverItem,
    DiscoverResult,
    ScrapeResult,
    SearchItem,
    WebSearchResult,
)

# httpx is an optional runtime dependency for this provider.  It is, however,
# already pulled in transitively by several other providers and by the test
# extra (respx).  Import lazily-friendly so a missing install yields a clear
# ConfigError rather than an ImportError deep in the manager.
try:  # pragma: no cover - exercised indirectly
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]
    _HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)

# --- Defaults ---------------------------------------------------------------
DEFAULT_BASE_URL = "https://api.brightdata.com"
DEFAULT_TIMEOUT = 60
DEFAULT_ENGINE = "google"
DEFAULT_POLL_INTERVAL = 2
DEFAULT_POLL_TIMEOUT = 60
DEFAULT_MAX_RETRIES = 3
DEFAULT_TOKEN_ENV_VAR = "BRIGHTDATA_API_TOKEN"

SUPPORTED_ENGINES = ("google", "bing", "yandex")
SERP_PAGE_SIZE = 10

# Endpoint paths (relative to base_url).
_REQUEST_ENDPOINT = "/request"
_UNBLOCKER_TRIGGER_ENDPOINT = "/unblocker/req"
_UNBLOCKER_RESULT_ENDPOINT = "/unblocker/get_result"
_DISCOVER_ENDPOINT = "/discover"
_DATASETS_LIST_ENDPOINT = "/datasets/list"
_DATASETS_FILTER_ENDPOINT = "/datasets/filter"
_ZONE_ACTIVE_ENDPOINT = "/zone/get_active_zones"


# ---------------------------------------------------------------------------
# SERP URL builders (no embedded language/country tables — codes pass through)
# ---------------------------------------------------------------------------
def _build_google_url(
    query: str,
    *,
    country: str | None,
    language: str,
    device: str,
    count: int,
    start: int = 0,
    **kwargs: Any,
) -> str:
    """Build a Google search URL with Bright Data JSON parsing enabled.

    Args:
        query: Search query.
        country: ISO-3166 alpha-2 country code (passed through as ``gl``).
        language: ISO-639-1 language code (passed through as ``hl``).
        device: ``"desktop"`` or ``"mobile"``.
        count: Results per page (``num``).
        start: Pagination offset.
        **kwargs: Optional ``safe_search`` (bool) and ``time_range`` (str).

    Returns:
        A fully-qualified Google search URL.
    """
    url = f"https://www.google.com/search?q={quote_plus(query)}"
    if start > 0:
        url += f"&start={start}"
    url += f"&num={count}"
    url += "&brd_json=1"
    if language:
        url += f"&hl={language}"
    if country:
        url += f"&gl={country.lower()}"
    if device == "mobile":
        url += "&mobileaction=1"
    if "safe_search" in kwargs:
        url += f"&safe={'active' if kwargs['safe_search'] else 'off'}"
    if kwargs.get("time_range"):
        url += f"&tbs=qdr:{kwargs['time_range']}"
    return url


def _build_bing_url(
    query: str,
    *,
    country: str | None,
    language: str,
    device: str,
    count: int,
    **kwargs: Any,
) -> str:
    """Build a Bing search URL.

    Args:
        query: Search query.
        country: ISO country code; combined with *language* into ``mkt``.
        language: ISO language code.
        device: Unused (Bing market drives localization).
        count: Result count.
        **kwargs: Ignored extras.

    Returns:
        A fully-qualified Bing search URL.
    """
    url = f"https://www.bing.com/search?q={quote_plus(query)}&count={count}&brd_json=1"
    if country:
        url += f"&mkt={language}-{country.upper()}"
    return url


def _build_yandex_url(
    query: str,
    *,
    country: str | None,
    language: str,
    device: str,
    count: int,
    **kwargs: Any,
) -> str:
    """Build a Yandex search URL.

    Args:
        query: Search query.
        country: Unused — Yandex uses a numeric region id (``lr``) which would
            require an embedded lookup table; pass ``region`` in kwargs instead.
        language: Unused.
        device: Unused.
        count: Result count (``numdoc``).
        **kwargs: Optional numeric ``region`` for ``lr``.

    Returns:
        A fully-qualified Yandex search URL.
    """
    url = f"https://yandex.com/search/?text={quote_plus(query)}&numdoc={count}&brd_json=1"
    region = kwargs.get("region")
    if region is not None:
        url += f"&lr={region}"
    return url


_URL_BUILDERS = {
    "google": _build_google_url,
    "bing": _build_bing_url,
    "yandex": _build_yandex_url,
}


def _normalize_serp(engine: str, data: Any) -> tuple[list[SearchItem], int | None]:
    """Normalize a parsed SERP payload into :class:`SearchItem` objects.

    Bright Data returns engine-specific parsed JSON when ``brd_json=1`` is set.
    For Google the organic results live under the ``organic`` key; Bing/Yandex
    payloads are passed through best-effort.

    Args:
        engine: Search engine name.
        data: Parsed response payload (dict) or ``None``.

    Returns:
        Tuple of (organic items, total_results-or-None).
    """
    if not isinstance(data, dict):
        return [], None

    organic = data.get("organic")
    if not isinstance(organic, list):
        # Some engines/zones use "organic_results"; fall back gracefully.
        organic = (
            data.get("organic_results") if isinstance(data.get("organic_results"), list) else []
        )

    items: list[SearchItem] = []
    for i, item in enumerate(organic, start=1):
        if not isinstance(item, dict):
            continue
        items.append(
            SearchItem(
                position=item.get("rank", item.get("position", i)),
                title=item.get("title", ""),
                url=item.get("link", item.get("url", "")),
                description=item.get("description", item.get("snippet", "")),
                displayed_url=item.get("display_link", item.get("displayed_url", "")),
            )
        )

    total = data.get("total_results")
    if not isinstance(total, int):
        total = None
    return items, total


def _unwrap_request_body(data: Any) -> Any:
    """Unwrap the ``{status_code, headers, body}`` envelope some zones return.

    The SERP/Unlocker ``/request`` endpoint may either return the parsed SERP
    JSON directly, or wrap it in an envelope whose ``body`` holds the payload
    (a JSON string for parsed mode, or raw HTML).

    Args:
        data: The decoded ``/request`` response.

    Returns:
        The inner payload (parsed JSON when possible, else ``{"raw_html": ...}``
        for HTML, else the original object).
    """
    if isinstance(data, dict) and "body" in data and "status_code" in data:
        body = data.get("body", "")
        if isinstance(body, str):
            stripped = body.strip()
            if stripped.startswith("<"):
                return {"raw_html": body, "status_code": data.get("status_code")}
            try:
                return _json.loads(body)
            except (ValueError, TypeError):
                return {"raw_html": body, "status_code": data.get("status_code")}
        return body
    return data


class BrightDataSearchProvider(BaseSearchProvider):
    """Bright Data implementation of :class:`BaseSearchProvider`.

    Args:
        config: Settings from ``[search_providers.brightdata]``.  Recognized
            keys: ``api_key`` / ``api_key_env_var``, ``base_url``, ``serp_zone``,
            ``unlocker_zone`` (alias ``web_unlocker_zone``), ``default_engine``,
            ``timeout``, ``poll_interval``, ``poll_timeout``, ``max_retries``,
            ``ssl_verify``.
        log_raw_payloads: Whether to log raw request/response payloads.

    Raises:
        ConfigError: If ``httpx`` is not installed or no API token is found.
    """

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False) -> None:
        super().__init__(config, log_raw_payloads)

        if not _HTTPX_AVAILABLE:
            raise ConfigError(
                "The 'httpx' package is required for the Bright Data search "
                "provider. Install with: pip install llmcore[brightdata]"
            )

        # --- API token ---
        token = config.get("api_key") or config.get("token")
        if not token:
            env_var = config.get("api_key_env_var", DEFAULT_TOKEN_ENV_VAR)
            import os

            token = os.environ.get(env_var) or os.environ.get(DEFAULT_TOKEN_ENV_VAR)
        if not token:
            raise ConfigError(
                "Bright Data API token not found. Set BRIGHTDATA_API_TOKEN or "
                "configure search_providers.brightdata.api_key / api_key_env_var."
            )
        self._token = str(token).strip()

        # --- Endpoint / zones / tuning ---
        self._base_url = str(config.get("base_url", DEFAULT_BASE_URL)).rstrip("/")
        self._serp_zone = config.get("serp_zone")
        self._unlocker_zone = config.get("unlocker_zone") or config.get("web_unlocker_zone")

        engine = str(config.get("default_engine", DEFAULT_ENGINE)).lower()
        if engine not in SUPPORTED_ENGINES:
            logger.warning(
                "Unsupported default_engine '%s' for Bright Data; falling back to '%s'.",
                engine,
                DEFAULT_ENGINE,
            )
            engine = DEFAULT_ENGINE
        self._default_engine = engine

        self._timeout = int(config.get("timeout", DEFAULT_TIMEOUT))
        self._poll_interval = int(config.get("poll_interval", DEFAULT_POLL_INTERVAL))
        self._poll_timeout = int(config.get("poll_timeout", DEFAULT_POLL_TIMEOUT))
        self._max_retries = int(config.get("max_retries", DEFAULT_MAX_RETRIES))
        self._ssl_verify = bool(config.get("ssl_verify", True))

        self._client: Any | None = None  # httpx.AsyncClient, created lazily
        logger.debug(
            "BrightDataSearchProvider initialized (base_url=%s, engine=%s, "
            "serp_zone=%s, unlocker_zone=%s).",
            self._base_url,
            self._default_engine,
            self._serp_zone,
            self._unlocker_zone,
        )

    # ------------------------------------------------------------------
    # Identity / capabilities
    # ------------------------------------------------------------------
    def get_name(self) -> str:
        """Return the provider type name (``"brightdata"``)."""
        return "brightdata"

    def get_capabilities(self) -> set[str]:
        """Return the set of supported capability strings."""
        return {
            SearchCapability.WEB_SEARCH.value,
            SearchCapability.SCRAPE.value,
            SearchCapability.DISCOVER.value,
            SearchCapability.DATASET_SEARCH.value,
        }

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------
    def _get_client(self) -> Any:
        """Return the lazily-created shared :class:`httpx.AsyncClient`."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                    "User-Agent": "llmcore-brightdata-search",
                },
                timeout=self._timeout,
                verify=self._ssl_verify,
            )
        return self._client

    def _log_payload(self, label: str, payload: Any) -> None:
        """Emit a raw-payload debug log when raw logging is enabled."""
        if self.log_raw_payloads_enabled:
            try:
                logger.debug("[brightdata] %s: %s", label, _json.dumps(payload, default=str)[:4000])
            except Exception:  # pragma: no cover - defensive
                logger.debug("[brightdata] %s: <unserializable>", label)

    async def _send(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Issue an HTTP request with retry on transient faults.

        Args:
            method: HTTP verb (``"GET"`` / ``"POST"``).
            path: Endpoint path relative to ``base_url``.
            json_body: Optional JSON request body.
            params: Optional query parameters.
            timeout: Optional per-request timeout override (seconds).

        Returns:
            The raw :class:`httpx.Response`.

        Raises:
            SearchProviderError: On authentication failure or repeated transport
                errors.
        """
        client = self._get_client()
        self._log_payload(f"{method} {path} body", json_body)
        last_exc: Exception | None = None
        attempts = max(1, self._max_retries)
        for attempt in range(attempts):
            try:
                response = await client.request(
                    method,
                    path,
                    json=json_body,
                    params=params,
                    timeout=timeout if timeout is not None else self._timeout,
                )
            except Exception as exc:  # httpx.TransportError, TimeoutException, …
                last_exc = exc
                if attempt < attempts - 1:
                    await asyncio.sleep(min(2**attempt, 8))
                    continue
                raise SearchProviderError(
                    self.get_name(), f"Transport error calling {path}: {exc}"
                ) from exc

            if response.status_code in (401, 403):
                raise SearchProviderError(
                    self.get_name(),
                    f"Authentication failed for {path}: {response.text[:200]}",
                    status_code=response.status_code,
                )
            # Retry only on 5xx (server-side) faults.
            if response.status_code >= 500 and attempt < attempts - 1:
                last_exc = SearchProviderError(
                    self.get_name(),
                    f"Server error {response.status_code} for {path}",
                    status_code=response.status_code,
                )
                await asyncio.sleep(min(2**attempt, 8))
                continue
            return response

        # Exhausted retries on 5xx.
        if isinstance(last_exc, SearchProviderError):
            raise last_exc
        raise SearchProviderError(self.get_name(), f"Request to {path} failed after retries.")

    @staticmethod
    def _decode(response: Any) -> Any:
        """Best-effort decode of an httpx response into JSON or text.

        Args:
            response: An :class:`httpx.Response`.

        Returns:
            Parsed JSON when the body is JSON, otherwise the raw text.
        """
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

    # ------------------------------------------------------------------
    # web_search
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
        """Run a SERP query and return normalized organic results.

        See :meth:`BaseSearchProvider.web_search` for the full parameter
        contract.  When ``parse=False`` is passed via kwargs, the raw HTML is
        returned in ``raw`` and ``items`` is empty.
        """
        if not query or not isinstance(query, str):
            raise SearchProviderError(
                self.get_name(), "web_search query must be a non-empty string."
            )
        if not self._serp_zone:
            raise SearchProviderError(
                self.get_name(),
                "No SERP zone configured. Set search_providers.brightdata.serp_zone "
                "to a Bright Data SERP zone name.",
            )

        engine = (engine or self._default_engine).lower()
        if engine not in _URL_BUILDERS:
            raise SearchProviderError(
                self.get_name(),
                f"Unsupported engine '{engine}'. Supported: {', '.join(SUPPORTED_ENGINES)}.",
            )

        parse = kwargs.pop("parse", True)
        builder = _URL_BUILDERS[engine]
        search_url = builder(
            query,
            country=country,
            language=language,
            device=device,
            count=count,
            **kwargs,
        )
        if not parse:
            # Strip the parsed-JSON hint so the engine returns HTML.
            search_url = search_url.replace("&brd_json=1", "")

        trigger_at = datetime.now(timezone.utc)
        with self._span("web_search", engine=engine, mode=mode):
            if mode == "async":
                raw = await self._unblocker_run(self._serp_zone, search_url, response_format="json")
            else:
                raw = await self._request_run(
                    self._serp_zone,
                    search_url,
                    response_format="json" if parse else "raw",
                )
        fetched_at = datetime.now(timezone.utc)

        payload = _unwrap_request_body(raw)
        items, total = ([], None) if not parse else _normalize_serp(engine, payload)
        return WebSearchResult(
            success=True,
            provider=self._instance_label(),
            query=query,
            engine=engine,
            items=items,
            total_results=total,
            raw=payload,
            trigger_sent_at=trigger_at,
            data_fetched_at=fetched_at,
        )

    async def _request_run(self, zone: str, url: str, *, response_format: str) -> Any:
        """Execute a synchronous ``POST /request`` and return decoded data.

        Args:
            zone: Bright Data zone name.
            url: Target URL (already built; may include ``brd_json=1``).
            response_format: ``"json"`` or ``"raw"``.

        Returns:
            Decoded response payload.

        Raises:
            SearchProviderError: On non-2xx responses.
        """
        body = {"zone": zone, "url": url, "format": response_format, "method": "GET"}
        response = await self._send("POST", _REQUEST_ENDPOINT, json_body=body)
        if response.status_code != 200:
            raise SearchProviderError(
                self.get_name(),
                f"/request failed: {response.text[:200]}",
                status_code=response.status_code,
            )
        return self._decode(response)

    async def _unblocker_run(self, zone: str, url: str, *, response_format: str) -> Any:
        """Trigger an async unblocker request and poll until ready.

        Uses ``POST /unblocker/req`` (returns an ``x-response-id`` header) then
        polls ``GET /unblocker/get_result`` (HTTP 202 pending, 200 ready).

        Args:
            zone: Bright Data zone name.
            url: Target URL.
            response_format: ``"json"`` or ``"raw"``.

        Returns:
            Decoded response payload once ready.

        Raises:
            SearchProviderError: On trigger failure, polling timeout, or error.
        """
        trigger = await self._send(
            "POST",
            _UNBLOCKER_TRIGGER_ENDPOINT,
            params={"zone": zone},
            json_body={"url": url},
        )
        response_id = trigger.headers.get("x-response-id")
        if not response_id:
            raise SearchProviderError(
                self.get_name(),
                f"Async trigger returned no x-response-id (HTTP {trigger.status_code}).",
                status_code=trigger.status_code,
            )

        elapsed = 0.0
        params = {"zone": zone, "response_id": response_id}
        while True:
            poll = await self._send("GET", _UNBLOCKER_RESULT_ENDPOINT, params=params)
            if poll.status_code == 200:
                if response_format == "json":
                    return self._decode(poll)
                return poll.text
            if poll.status_code != 202:
                raise SearchProviderError(
                    self.get_name(),
                    f"Async result error (response_id={response_id}): {poll.text[:200]}",
                    status_code=poll.status_code,
                )
            if elapsed >= self._poll_timeout:
                raise SearchProviderError(
                    self.get_name(),
                    f"Async polling timed out after {self._poll_timeout}s "
                    f"(response_id={response_id}).",
                )
            await asyncio.sleep(self._poll_interval)
            elapsed += self._poll_interval

    # ------------------------------------------------------------------
    # scrape
    # ------------------------------------------------------------------
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
        """Fetch a single URL through the Web Unlocker proxy.

        See :meth:`BaseSearchProvider.scrape` for the parameter contract.
        """
        if not url or not isinstance(url, str):
            raise SearchProviderError(self.get_name(), "scrape url must be a non-empty string.")
        if not self._unlocker_zone:
            raise SearchProviderError(
                self.get_name(),
                "No unlocker zone configured. Set "
                "search_providers.brightdata.unlocker_zone to a Web Unlocker zone name.",
            )
        if response_format not in ("raw", "json"):
            raise SearchProviderError(self.get_name(), "response_format must be 'raw' or 'json'.")

        trigger_at = datetime.now(timezone.utc)
        with self._span("scrape", mode=mode, response_format=response_format):
            if mode == "async":
                data = await self._unblocker_run(
                    self._unlocker_zone, url, response_format=response_format
                )
            else:
                body: dict[str, Any] = {
                    "zone": self._unlocker_zone,
                    "url": url,
                    "format": response_format,
                    "method": method,
                }
                if country:
                    body["country"] = country.upper()
                response = await self._send("POST", _REQUEST_ENDPOINT, json_body=body)
                if response.status_code != 200:
                    return ScrapeResult(
                        success=False,
                        provider=self._instance_label(),
                        url=url,
                        status="error",
                        response_format=response_format,
                        error=f"/request failed (HTTP {response.status_code}): {response.text[:200]}",
                        trigger_sent_at=trigger_at,
                        data_fetched_at=datetime.now(timezone.utc),
                    )
                data = self._decode(response) if response_format == "json" else response.text
        fetched_at = datetime.now(timezone.utc)

        char_size = len(data) if isinstance(data, str) else None
        return ScrapeResult(
            success=True,
            provider=self._instance_label(),
            url=url,
            content=data,
            response_format=response_format,
            status="ready",
            root_domain=_root_domain(url),
            content_char_size=char_size,
            trigger_sent_at=trigger_at,
            data_fetched_at=fetched_at,
        )

    # ------------------------------------------------------------------
    # discover
    # ------------------------------------------------------------------
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
        """Run an AI-relevance-ranked Discover search (trigger + poll).

        See :meth:`BaseSearchProvider.discover` for the parameter contract.
        """
        if not query or not isinstance(query, str):
            raise SearchProviderError(self.get_name(), "discover query must be a non-empty string.")

        body: dict[str, Any] = {"query": query, "format": kwargs.get("format", "json")}
        if intent:
            body["intent"] = intent
        if include_content:
            body["include_content"] = True
        if country:
            body["country"] = country
        if city:
            body["city"] = city
        if language:
            body["language"] = language
        if filter_keywords:
            body["filter_keywords"] = filter_keywords
        if count is not None:
            body["num_results"] = count

        trigger_at = datetime.now(timezone.utc)
        with self._span("discover"):
            trigger = await self._send("POST", _DISCOVER_ENDPOINT, json_body=body)
            if trigger.status_code >= 400:
                raise SearchProviderError(
                    self.get_name(),
                    f"discover trigger failed: {trigger.text[:200]}",
                    status_code=trigger.status_code,
                )
            task_id = (self._decode(trigger) or {}).get("task_id")
            if not task_id:
                raise SearchProviderError(self.get_name(), "discover returned no task_id.")

            payload = await self._poll_discover(task_id, timeout, poll_interval)
        fetched_at = datetime.now(timezone.utc)

        results = payload.get("results", []) if isinstance(payload, dict) else []
        items = [
            DiscoverItem(
                title=r.get("title", ""),
                url=r.get("link", r.get("url", "")),
                description=r.get("description", r.get("snippet", "")),
                relevance_score=r.get("relevance_score"),
                content=r.get("content"),
            )
            for r in results
            if isinstance(r, dict)
        ]
        return DiscoverResult(
            success=True,
            provider=self._instance_label(),
            query=query,
            intent=intent,
            items=items,
            total_results=len(items),
            task_id=task_id,
            duration_seconds=payload.get("duration_seconds") if isinstance(payload, dict) else None,
            raw=payload,
            trigger_sent_at=trigger_at,
            data_fetched_at=fetched_at,
        )

    async def _poll_discover(
        self, task_id: str, timeout: int, poll_interval: int
    ) -> dict[str, Any]:
        """Poll ``GET /discover?task_id=`` until done, error, or timeout.

        Args:
            task_id: The discover task identifier.
            timeout: Maximum seconds to wait.
            poll_interval: Seconds between polls.

        Returns:
            The final response payload (``status == "done"``).

        Raises:
            SearchProviderError: On task failure or timeout.
        """
        elapsed = 0.0
        while True:
            poll = await self._send("GET", _DISCOVER_ENDPOINT, params={"task_id": task_id})
            if poll.status_code >= 400:
                raise SearchProviderError(
                    self.get_name(),
                    f"discover poll failed: {poll.text[:200]}",
                    status_code=poll.status_code,
                )
            payload = self._decode(poll)
            status = (
                payload.get("status", "processing") if isinstance(payload, dict) else "processing"
            )
            if status == "done":
                return payload  # type: ignore[return-value]
            if status in ("error", "failed"):
                msg = (
                    payload.get("error", "unknown error") if isinstance(payload, dict) else "error"
                )
                raise SearchProviderError(self.get_name(), f"discover task failed: {msg}")
            if elapsed >= timeout:
                raise SearchProviderError(
                    self.get_name(), f"discover task {task_id} timed out after {timeout}s."
                )
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

    # ------------------------------------------------------------------
    # datasets
    # ------------------------------------------------------------------
    async def list_datasets(self) -> list[DatasetInfo]:
        """List datasets available to the account (``GET /datasets/list``)."""
        with self._span("list_datasets"):
            response = await self._send("GET", _DATASETS_LIST_ENDPOINT)
        if response.status_code != 200:
            raise SearchProviderError(
                self.get_name(),
                f"datasets/list failed: {response.text[:200]}",
                status_code=response.status_code,
            )
        data = self._decode(response)
        items = (
            data
            if isinstance(data, list)
            else data.get("data", [])
            if isinstance(data, dict)
            else []
        )
        return [
            DatasetInfo(
                id=item.get("id", ""),
                name=item.get("name", ""),
                size=item.get("size", 0) or 0,
            )
            for item in items
            if isinstance(item, dict)
        ]

    async def dataset_metadata(self, dataset_id: str) -> DatasetMetadata:
        """Return the field schema for *dataset_id* (``GET /datasets/{id}/metadata``)."""
        with self._span("dataset_metadata", dataset_id=dataset_id):
            response = await self._send("GET", f"/datasets/{dataset_id}/metadata")
        if response.status_code != 200:
            raise SearchProviderError(
                self.get_name(),
                f"dataset metadata failed: {response.text[:200]}",
                status_code=response.status_code,
            )
        data = self._decode(response)
        fields: list[DatasetField] = []
        if isinstance(data, dict):
            for name, meta in (data.get("fields") or {}).items():
                if isinstance(meta, dict):
                    fields.append(
                        DatasetField(
                            name=name,
                            type=meta.get("type", "text"),
                            active=meta.get("active", True),
                            required=meta.get("required", False),
                            description=meta.get("description"),
                        )
                    )
        return DatasetMetadata(
            id=data.get("id", dataset_id) if isinstance(data, dict) else dataset_id, fields=fields
        )

    async def dataset_filter(
        self,
        dataset_id: str,
        filter: dict[str, Any],
        *,
        records_limit: int | None = None,
        **kwargs: Any,
    ) -> DatasetSnapshot:
        """Create a snapshot by filtering a dataset (``POST /datasets/filter``).

        Returns immediately with the ``snapshot_id``; use
        :meth:`dataset_download` to fetch records.
        """
        body: dict[str, Any] = {"dataset_id": dataset_id, "filter": filter}
        if records_limit is not None:
            body["records_limit"] = records_limit
        trigger_at = datetime.now(timezone.utc)
        with self._span("dataset_filter", dataset_id=dataset_id):
            response = await self._send("POST", _DATASETS_FILTER_ENDPOINT, json_body=body)
        if response.status_code >= 400:
            return DatasetSnapshot(
                success=False,
                provider=self._instance_label(),
                dataset_id=dataset_id,
                status="failed",
                error=f"datasets/filter failed (HTTP {response.status_code}): {response.text[:200]}",
                trigger_sent_at=trigger_at,
                data_fetched_at=datetime.now(timezone.utc),
            )
        data = self._decode(response)
        snapshot_id = data.get("snapshot_id") if isinstance(data, dict) else None
        if not snapshot_id:
            err = (
                data.get("error") or data.get("message") or str(data)
                if isinstance(data, dict)
                else str(data)
            )
            return DatasetSnapshot(
                success=False,
                provider=self._instance_label(),
                dataset_id=dataset_id,
                status="failed",
                error=f"No snapshot_id returned: {err}",
                trigger_sent_at=trigger_at,
                data_fetched_at=datetime.now(timezone.utc),
            )
        return DatasetSnapshot(
            success=True,
            provider=self._instance_label(),
            dataset_id=dataset_id,
            snapshot_id=snapshot_id,
            status="scheduled",
            trigger_sent_at=trigger_at,
            data_fetched_at=datetime.now(timezone.utc),
        )

    async def dataset_status(self, snapshot_id: str) -> DatasetSnapshot:
        """Return the current status of a snapshot (``GET /datasets/snapshots/{id}``)."""
        with self._span("dataset_status", snapshot_id=snapshot_id):
            response = await self._send("GET", f"/datasets/snapshots/{snapshot_id}")
        if response.status_code != 200:
            raise SearchProviderError(
                self.get_name(),
                f"dataset status failed: {response.text[:200]}",
                status_code=response.status_code,
            )
        data = self._decode(response)
        if not isinstance(data, dict):
            data = {}
        return DatasetSnapshot(
            success=True,
            provider=self._instance_label(),
            dataset_id=data.get("dataset_id"),
            snapshot_id=data.get("id", data.get("snapshot_id", snapshot_id)),
            status=data.get("status", "scheduled"),
            dataset_size=data.get("dataset_size"),
            file_size=data.get("file_size"),
            cost=data.get("cost"),
            error=data.get("error", data.get("error_message")),
        )

    async def dataset_download(
        self,
        snapshot_id: str,
        *,
        format: str = "jsonl",
        timeout: int = 300,
        poll_interval: int = 5,
        **kwargs: Any,
    ) -> DatasetSnapshot:
        """Poll a snapshot until ready, then download its records.

        See :meth:`BaseSearchProvider.dataset_download` for the contract.
        """
        trigger_at = datetime.now(timezone.utc)
        elapsed = 0.0
        last: DatasetSnapshot | None = None
        with self._span("dataset_download", snapshot_id=snapshot_id):
            while True:
                last = await self.dataset_status(snapshot_id)
                if last.status == "ready":
                    break
                if last.status == "failed":
                    last.success = False
                    last.error = last.error or "Snapshot failed."
                    return last
                if elapsed >= timeout:
                    return DatasetSnapshot(
                        success=False,
                        provider=self._instance_label(),
                        dataset_id=last.dataset_id,
                        snapshot_id=snapshot_id,
                        status=last.status,
                        error=f"Snapshot not ready after {timeout}s (status={last.status}).",
                        trigger_sent_at=trigger_at,
                        data_fetched_at=datetime.now(timezone.utc),
                    )
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

            response = await self._send(
                "GET",
                f"/datasets/snapshots/{snapshot_id}/download",
                params={"format": format},
            )
        if response.status_code >= 400:
            raise SearchProviderError(
                self.get_name(),
                f"dataset download failed: {response.text[:200]}",
                status_code=response.status_code,
            )
        records = _parse_records(response, format)
        return DatasetSnapshot(
            success=True,
            provider=self._instance_label(),
            dataset_id=last.dataset_id if last else None,
            snapshot_id=snapshot_id,
            status="ready",
            records=records,
            dataset_size=(last.dataset_size if last else None) or len(records),
            file_size=last.file_size if last else None,
            trigger_sent_at=trigger_at,
            data_fetched_at=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # health / lifecycle
    # ------------------------------------------------------------------
    async def health_check(self) -> bool:
        """Verify connectivity/credentials via ``GET /zone/get_active_zones``."""
        try:
            response = await self._send("GET", _ZONE_ACTIVE_ENDPOINT)
        except SearchProviderError:
            return False
        return response.status_code == 200

    async def close(self) -> None:
        """Close the underlying :class:`httpx.AsyncClient` (idempotent)."""
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Error closing Bright Data httpx client: %s", exc)
            self._client = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _instance_label(self) -> str:
        """Return the configured instance name, falling back to the type name."""
        return self._provider_instance_name or self.get_name()


# ---------------------------------------------------------------------------
# module-level helpers
# ---------------------------------------------------------------------------
def _root_domain(url: str) -> str | None:
    """Extract a best-effort registered domain from *url*.

    Args:
        url: A URL string.

    Returns:
        The host's last two labels (e.g. ``example.com``), or ``None``.
    """
    try:
        from urllib.parse import urlparse

        host = urlparse(url).netloc.split("@")[-1].split(":")[0]
        parts = host.split(".")
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return host or None
    except Exception:  # pragma: no cover - defensive
        return None


def _parse_records(response: Any, fmt: str) -> list[dict[str, Any]]:
    """Parse a dataset download response into a list of record dicts.

    Handles JSON arrays, newline-delimited JSON (``jsonl``), and a single JSON
    object, mirroring how Bright Data serves snapshot downloads.

    Args:
        response: An :class:`httpx.Response`.
        fmt: The requested format (``"json"``, ``"jsonl"``, or ``"csv"``).

    Returns:
        A list of record dictionaries (best-effort; raw text wrapped on failure).
    """
    text = response.text
    if not text or not text.strip():
        return []
    content_type = response.headers.get("Content-Type", "")

    if "application/json" in content_type or text.strip().startswith("["):
        try:
            data = _json.loads(text)
        except (ValueError, TypeError):
            data = None
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("data", [data])

    if "ndjson" in content_type or fmt == "jsonl" or "\n" in text.strip():
        try:
            lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
            return [_json.loads(ln) for ln in lines]
        except (ValueError, TypeError):
            pass

    try:
        data = _json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("data", [data])
    except (ValueError, TypeError):
        pass
    return [{"raw": text}]
