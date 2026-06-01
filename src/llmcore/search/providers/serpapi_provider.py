# src/llmcore/search/providers/serpapi_provider.py
"""SerpApi web **search** provider for LLMCore.

Implements :class:`llmcore.search.base.BaseSearchProvider` against the SerpApi
real-time search API (``https://serpapi.com``) using :mod:`httpx` directly —
consistent with the native LLM providers and the sibling Serper / Bright Data
search providers (no vendor SDK dependency).

SerpApi is a *meta* SERP API: a single ``GET /search`` endpoint scrapes 100+
search engines/verticals selected with the ``engine`` parameter (``google``,
``bing``, ``baidu``, ``duckduckgo``, ``yahoo``, ``yandex``, ``google_news``,
``google_images``, ``google_shopping``, ``google_scholar``, ``youtube``,
``amazon``, ``ebay``, ``walmart``, ``google_maps``, …). Each engine returns
Google/engine-shaped JSON whose organic array lives under an engine-specific key
(``organic_results`` for most search engines; ``news_results`` / ``images_results``
/ ``video_results`` / ``shopping_results`` / ``local_results`` / … for verticals).

Supported capabilities
-----------------------
=================  ============================================================
Capability         SerpApi surface
=================  ============================================================
``web_search``     ``GET /search`` (sync) or ``async=true`` + poll the
                   Search Archive (``GET /searches/{id}``). Multi-engine via
                   the ``engine`` parameter; full pass-through of every SerpApi
                   request parameter.
``batch_search``   **Client-side** bounded fan-out of concurrent ``web_search``
                   calls (SerpApi has no server-side batch endpoint). Returns
                   one result per input query, in order. Each query consumes one
                   search credit.
=================  ============================================================

SerpApi does **not** offer an arbitrary-URL unlocker/scraper, an AI-relevance
Discover API, or a dataset marketplace, so ``scrape``, ``discover`` and
``dataset_search`` are intentionally **not** advertised (calling them raises
``NotImplementedError`` from the base class).

Provider-specific extras (not part of the cross-provider ``BaseSearchProvider``
contract) are also exposed for power users:

* :meth:`SerpApiSearchProvider.search` — faithful low-level ``GET /search``
  pass-through (engine-agnostic; mirrors the official ``serpapi`` client).
* :meth:`SerpApiSearchProvider.search_archive` — fetch a past result by id.
* :meth:`SerpApiSearchProvider.account` — Account API (plan / quota; **free**).
* :meth:`SerpApiSearchProvider.locations` — Locations API (canonical locations).

Authentication
--------------
SerpApi authenticates with an ``api_key`` **query parameter** (NOT a header /
Bearer token). Resolve the key via ``[search_providers.serpapi].api_key`` or,
preferably, the ``SERPAPI_API_KEY`` environment variable (``SERPAPI_KEY`` is also
honored for compatibility with the official SDK convention).

References:
    * Google Search API: https://serpapi.com/search-api
    * Search Archive API: https://serpapi.com/search-archive-api
    * Account API: https://serpapi.com/account-api
    * Locations API: https://serpapi.com/locations-api
    * Async searches: https://serpapi.com/search-api (``async`` parameter)
    * Official Python client: https://github.com/serpapi/serpapi-python
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from ...exceptions import ConfigError, SearchProviderError
from ..base import BaseSearchProvider, SearchCapability
from ..models import SearchItem, WebSearchResult

try:  # pragma: no cover - exercised indirectly
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]
    _HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)

# --- Defaults ---------------------------------------------------------------
DEFAULT_BASE_URL = "https://serpapi.com"
SEARCH_ENDPOINT = "/search"
ACCOUNT_ENDPOINT = "/account.json"
LOCATIONS_ENDPOINT = "/locations.json"
DEFAULT_ENGINE = "google"
DEFAULT_OUTPUT = "json"
DEFAULT_TIMEOUT = 60
DEFAULT_MAX_RETRIES = 3
DEFAULT_POLL_INTERVAL = 2
DEFAULT_POLL_TIMEOUT = 60
DEFAULT_MAX_CONCURRENCY = 5
DEFAULT_TOKEN_ENV_VAR = "SERPAPI_API_KEY"
# Secondary env var honored for parity with the official `serpapi` SDK / docs.
SECONDARY_TOKEN_ENV_VARS = ("SERPAPI_KEY", "SERP_API_KEY")

# Async search lifecycle (search_metadata.status): Processing -> Success || Error.
_ASYNC_STATUS_DONE = {"success"}
_ASYNC_STATUS_ERROR = {"error"}

# A best-effort, non-exhaustive set of known SerpApi engines. Used ONLY to emit a
# soft debug warning for likely-typo'd engines — never to reject one, because
# SerpApi adds engines frequently and any unknown engine is passed through as-is.
KNOWN_ENGINES = frozenset(
    {
        # Google family
        "google",
        "google_light",
        "google_ai_mode",
        "google_ai_overview",
        "google_ads",
        "google_ads_transparency_center",
        "google_autocomplete",
        "google_events",
        "google_finance",
        "google_finance_markets",
        "google_flights",
        "google_flights_autocomplete",
        "google_forums",
        "google_hotels",
        "google_hotels_autocomplete",
        "google_hotels_photos",
        "google_hotels_reviews",
        "google_images",
        "google_images_light",
        "google_immersive_product",
        "google_jobs",
        "google_lens",
        "google_local",
        "google_local_services",
        "google_maps",
        "google_maps_autocomplete",
        "google_maps_contributor_reviews",
        "google_maps_directions",
        "google_maps_photos",
        "google_maps_posts",
        "google_maps_reviews",
        "google_news",
        "google_news_light",
        "google_patents",
        "google_patents_details",
        "google_play",
        "google_play_books",
        "google_play_games",
        "google_play_movies",
        "google_play_product",
        "google_related_questions",
        "google_reverse_image",
        "google_scholar",
        "google_scholar_author",
        "google_scholar_case_law",
        "google_shopping",
        "google_shopping_light",
        "google_short_videos",
        "google_travel_explore",
        "google_trends",
        "google_trends_autocomplete",
        "google_trends_trending_now",
        "google_videos",
        "google_videos_light",
        # Other engines
        "amazon",
        "amazon_product",
        "apple_app_store",
        "apple_maps",
        "apple_maps_places",
        "apple_product",
        "apple_reviews",
        "baidu",
        "baidu_news",
        "bing",
        "bing_copilot",
        "bing_images",
        "bing_maps",
        "bing_news",
        "bing_product",
        "bing_reverse_image",
        "bing_shopping",
        "bing_videos",
        "brave_ai_mode",
        "duckduckgo",
        "duckduckgo_light",
        "duckduckgo_maps",
        "duckduckgo_news",
        "ebay",
        "ebay_product",
        "facebook_profile",
        "home_depot",
        "home_depot_product",
        "home_depot_product_reviews",
        "instagram_profile",
        "naver",
        "naver_ai_overview",
        "open_table_reviews",
        "tripadvisor",
        "tripadvisor_place",
        "tripadvisor_reviews",
        "walmart",
        "walmart_product",
        "walmart_product_reviews",
        "yahoo",
        "yahoo_images",
        "yahoo_shopping",
        "yahoo_videos",
        "yandex",
        "yandex_images",
        "yandex_videos",
        "yelp",
        "yelp_place",
        "yelp_reviews",
        "youtube",
        "youtube_video",
        "youtube_video_transcript",
    }
)

# Engine -> free-text "query" request parameter. Most engines use ``q``; these
# differ. Anything not listed defaults to ``q``. (Derived from the per-engine
# required-parameter tables in the SerpApi documentation.)
_QUERY_PARAM_BY_ENGINE: dict[str, str] = {
    "walmart": "query",
    "naver": "query",
    "naver_ai_overview": "query",
    "apple_maps": "query",
    "apple_app_store": "term",
    "ebay": "_nkw",
    "yahoo": "p",
    "yahoo_images": "p",
    "yahoo_shopping": "p",
    "yahoo_videos": "p",
    "yandex": "text",
    "yandex_images": "text",
    "yandex_videos": "text",
    "youtube": "search_query",
    # Yelp requires a location (``find_loc``); the free-text term is ``find_desc``.
    "yelp": "find_desc",
}

# Engine -> the top-level JSON key that holds the primary result array, when it
# is NOT ``organic_results`` (the default). Used by the normalizer to map the
# most relevant array into provider-agnostic :class:`SearchItem` objects. The
# full payload is always preserved on :attr:`WebSearchResult.raw`.
_PRIMARY_RESULTS_KEY_BY_ENGINE: dict[str, str] = {
    # News
    "google_news": "news_results",
    "google_news_light": "news_results",
    "duckduckgo_news": "news_results",
    # Images
    "google_images": "images_results",
    "google_images_light": "images_results",
    "bing_images": "images_results",
    "yahoo_images": "images_results",
    "yandex_images": "images_results",
    "google_reverse_image": "image_results",
    # Videos
    "google_videos": "video_results",
    "google_videos_light": "video_results",
    "bing_videos": "video_results",
    "youtube": "video_results",
    "yandex_videos": "videos_results",
    "yahoo_videos": "videos_results",
    "google_short_videos": "short_video_results",
    # Shopping
    "google_shopping": "shopping_results",
    "google_shopping_light": "shopping_results",
    "bing_shopping": "shopping_results",
    "yahoo_shopping": "shopping_results",
    # Local / Maps
    "google_maps": "local_results",
    "google_local": "local_results",
    "bing_maps": "local_results",
    "duckduckgo_maps": "local_results",
    "apple_maps": "local_results",
    # Jobs / Events / Hotels / Retail / Travel
    "google_jobs": "jobs_results",
    "google_events": "events_results",
    "google_hotels": "properties",
    "home_depot": "products",
    "tripadvisor": "places",
    # Autocomplete
    "google_autocomplete": "suggestions",
    "google_maps_autocomplete": "suggestions",
    "google_hotels_autocomplete": "suggestions",
    "google_flights_autocomplete": "suggestions",
    "google_trends_autocomplete": "suggestions",
    # Places / Reviews (single-entity engines)
    "apple_maps_places": "place_results",
    "yelp_place": "place_results",
    "apple_reviews": "reviews",
    "yelp_reviews": "reviews",
    "tripadvisor_reviews": "reviews",
    "open_table_reviews": "reviews",
    "google_hotels_reviews": "reviews",
    "google_maps_reviews": "reviews",
    "home_depot_product_reviews": "reviews",
    "walmart_product_reviews": "reviews",
}

# Ordered fallback chain of result-array keys probed when neither the
# engine-specific key nor ``organic_results`` is present.
_FALLBACK_RESULT_KEYS: tuple[str, ...] = (
    "organic_results",
    "news_results",
    "images_results",
    "image_results",
    "video_results",
    "videos_results",
    "short_video_results",
    "shopping_results",
    "local_results",
    "jobs_results",
    "events_results",
    "products",
    "properties",
    "places",
    "place_results",
    "web_results",
    "reviews",
    "suggestions",
)


def _coerce_total_results(value: Any) -> int | None:
    """Coerce a SerpApi ``total_results`` value into an ``int`` when possible.

    SerpApi usually returns an integer, but some engines report a localized
    string (e.g. ``"About 3,140,000,000 results"``). This extracts the leading
    digit run.

    Args:
        value: The raw ``search_information.total_results`` value.

    Returns:
        The integer total, or ``None`` when it cannot be determined.
    """
    if isinstance(value, bool):  # bool is an int subclass; never a count
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        digits = "".join(ch for ch in value if ch.isdigit())
        if digits:
            try:
                return int(digits)
            except ValueError:  # pragma: no cover - defensive
                return None
    return None


def _normalize_serpapi(data: Any, engine: str) -> tuple[list[SearchItem], int | None]:
    """Normalize a SerpApi response into provider-agnostic :class:`SearchItem` objects.

    The primary result array is resolved by engine: the engine-specific key from
    :data:`_PRIMARY_RESULTS_KEY_BY_ENGINE` when defined, otherwise
    ``organic_results``, otherwise the first present key from
    :data:`_FALLBACK_RESULT_KEYS`. Only the common fields (title, link, snippet,
    position, displayed link) are mapped; the full payload — knowledge graph,
    answer boxes, related questions, pagination, and every engine-specific
    block — is preserved verbatim on :attr:`WebSearchResult.raw`.

    Args:
        data: Parsed SerpApi response (dict) or ``None``.
        engine: The engine that produced the response (e.g. ``"google"``).

    Returns:
        Tuple of (mapped items, total_results-or-None). ``total_results`` is read
        from ``search_information.total_results`` when present.
    """
    if not isinstance(data, dict):
        return [], None

    # Resolve which array to normalize.
    key = _PRIMARY_RESULTS_KEY_BY_ENGINE.get(engine)
    arr = data.get(key) if key else None
    if not isinstance(arr, list):
        for candidate in _FALLBACK_RESULT_KEYS:
            value = data.get(candidate)
            if isinstance(value, list) and value:
                arr = value
                break
    if not isinstance(arr, list):
        arr = []

    items: list[SearchItem] = []
    for i, item in enumerate(arr, start=1):
        if isinstance(item, str):
            # e.g. some autocomplete arrays are plain strings.
            items.append(SearchItem(position=i, title=item))
            continue
        if not isinstance(item, dict):
            continue
        items.append(
            SearchItem(
                position=item.get("position", i),
                title=str(item.get("title") or item.get("name") or item.get("value") or ""),
                url=str(item.get("link") or item.get("url") or item.get("serpapi_link") or ""),
                description=str(item.get("snippet") or item.get("description") or ""),
                displayed_url=str(item.get("displayed_link") or item.get("displayed_url") or ""),
            )
        )

    total = None
    info = data.get("search_information")
    if isinstance(info, dict):
        total = _coerce_total_results(info.get("total_results"))
    return items, total


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


class SerpApiSearchProvider(BaseSearchProvider):
    """SerpApi implementation of :class:`BaseSearchProvider`.

    Args:
        config: Settings from ``[search_providers.serpapi]``. Recognized keys:
            ``api_key`` / ``token`` / ``api_key_env_var``, ``base_url``,
            ``default_engine``, ``default_output`` (``json`` | ``html``),
            ``no_cache`` (bool), ``zero_trace`` (bool), ``json_restrictor``
            (str), ``timeout``, ``max_retries``, ``poll_interval``,
            ``poll_timeout``, ``max_concurrency``, ``ssl_verify``.
        log_raw_payloads: Whether to log raw request/response payloads (the
            ``api_key`` query parameter is always redacted from logs).

    Raises:
        ConfigError: If ``httpx`` is not installed or no API key is found.
    """

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False) -> None:
        super().__init__(config, log_raw_payloads)

        if not _HTTPX_AVAILABLE:
            raise ConfigError(
                "The 'httpx' package is required for the SerpApi search provider. "
                "Install with: pip install llmcore[serpapi]"
            )

        # --- API key (query parameter) ---
        key = config.get("api_key") or config.get("token")
        if not key:
            env_var = config.get("api_key_env_var", DEFAULT_TOKEN_ENV_VAR)
            key = os.environ.get(env_var)
            if not key:
                # Fall back to the documented SDK env-var spellings.
                for fallback in (DEFAULT_TOKEN_ENV_VAR, *SECONDARY_TOKEN_ENV_VARS):
                    key = os.environ.get(fallback)
                    if key:
                        break
        if not key:
            raise ConfigError(
                "SerpApi API key not found. Set SERPAPI_API_KEY (or SERPAPI_KEY) or "
                "configure search_providers.serpapi.api_key / api_key_env_var."
            )
        self._api_key = str(key).strip()

        # --- Endpoints / defaults / tuning ---
        self._base_url = str(config.get("base_url", DEFAULT_BASE_URL)).rstrip("/")

        engine = str(config.get("default_engine", DEFAULT_ENGINE)).strip().lower()
        if not engine:
            engine = DEFAULT_ENGINE
        if engine not in KNOWN_ENGINES:
            logger.debug(
                "default_engine '%s' is not in the known SerpApi engine list; "
                "passing through as-is.",
                engine,
            )
        self._default_engine = engine

        output = str(config.get("default_output", DEFAULT_OUTPUT)).strip().lower()
        if output not in ("json", "html"):
            logger.warning(
                "Unsupported default_output '%s' for SerpApi; falling back to '%s'.",
                output,
                DEFAULT_OUTPUT,
            )
            output = DEFAULT_OUTPUT
        self._default_output = output

        self._default_no_cache = bool(config.get("no_cache", False))
        self._default_zero_trace = bool(config.get("zero_trace", False))
        jr = config.get("json_restrictor")
        self._default_json_restrictor = str(jr) if jr else None

        self._timeout = int(config.get("timeout", DEFAULT_TIMEOUT))
        self._max_retries = int(config.get("max_retries", DEFAULT_MAX_RETRIES))
        self._poll_interval = int(config.get("poll_interval", DEFAULT_POLL_INTERVAL))
        self._poll_timeout = int(config.get("poll_timeout", DEFAULT_POLL_TIMEOUT))
        self._max_concurrency = max(1, int(config.get("max_concurrency", DEFAULT_MAX_CONCURRENCY)))
        self._ssl_verify = bool(config.get("ssl_verify", True))

        self._client: Any | None = None  # httpx.AsyncClient, created lazily
        logger.debug(
            "SerpApiSearchProvider initialized (base_url=%s, default_engine=%s, output=%s).",
            self._base_url,
            self._default_engine,
            self._default_output,
        )

    # ------------------------------------------------------------------
    # Identity / capabilities
    # ------------------------------------------------------------------
    def get_name(self) -> str:
        """Return the provider type name (``"serpapi"``)."""
        return "serpapi"

    def get_capabilities(self) -> set[str]:
        """Return the set of supported capability strings."""
        return {
            SearchCapability.WEB_SEARCH.value,
            SearchCapability.BATCH_SEARCH.value,
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
                    "Accept": "application/json",
                    "User-Agent": "llmcore-serpapi-search",
                },
                timeout=self._timeout,
                verify=self._ssl_verify,
            )
        return self._client

    @staticmethod
    def _redacted(params: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of *params* with the API key masked for logging."""
        safe = dict(params)
        if "api_key" in safe:
            safe["api_key"] = "***"
        return safe

    def _log_payload(self, label: str, params: dict[str, Any]) -> None:
        """Emit a raw-payload debug log (API key redacted) when enabled."""
        if self.log_raw_payloads_enabled:
            try:
                logger.debug(
                    "[serpapi] %s: %s",
                    label,
                    _json.dumps(self._redacted(params), default=str)[:4000],
                )
            except Exception:  # pragma: no cover - defensive
                logger.debug("[serpapi] %s: <unserializable>", label)

    async def _get(
        self,
        path: str,
        params: dict[str, Any],
    ) -> Any:
        """Issue a GET with the API key injected and retry on transient faults.

        The ``api_key`` query parameter is injected here (never by callers) and
        is redacted from any debug logging.

        Args:
            path: Endpoint path relative to ``base_url`` (e.g. ``"/search"``) or
                an absolute URL (used when following a pagination ``next`` link).
            params: Query parameters (without ``api_key``).

        Returns:
            The raw :class:`httpx.Response`.

        Raises:
            SearchProviderError: On authentication failure (401/403) or repeated
                transport / rate-limit (429) / server (5xx) errors.
        """
        client = self._get_client()
        request_params = {k: v for k, v in params.items() if v is not None}
        request_params.setdefault("api_key", self._api_key)
        self._log_payload(f"GET {path} params", request_params)

        last_exc: Exception | None = None
        attempts = max(1, self._max_retries)
        for attempt in range(attempts):
            try:
                response = await client.get(path, params=request_params)
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
                    f"Authentication failed for {path}: {self._error_message(response)}",
                    status_code=response.status_code,
                )
            # Retry on rate limit (429) and server (5xx) faults.
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < attempts - 1:
                    last_exc = SearchProviderError(
                        self.get_name(),
                        f"Retryable status {response.status_code} for {path}",
                        status_code=response.status_code,
                    )
                    await asyncio.sleep(min(2**attempt, 8))
                    continue
            return response

        if isinstance(last_exc, SearchProviderError):
            raise last_exc
        raise SearchProviderError(self.get_name(), f"Request to {path} failed after retries.")

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
        """Extract SerpApi's ``error`` message from a response, if present."""
        data = self._decode(response)
        if isinstance(data, dict):
            for field_name in ("error", "message"):
                if data.get(field_name):
                    return str(data[field_name])
        return (response.text or "")[:200]

    def _instance_label(self) -> str:
        """Return the configured instance name, falling back to the type name."""
        return self._provider_instance_name or self.get_name()

    # ------------------------------------------------------------------
    # Parameter assembly
    # ------------------------------------------------------------------
    def _resolve_engine(self, engine: str | None) -> str:
        """Return a normalized engine, defaulting to the configured engine."""
        eng = (engine or self._default_engine or DEFAULT_ENGINE).strip().lower()
        if eng not in KNOWN_ENGINES:
            logger.debug(
                "SerpApi engine '%s' is not in the known engine list; passing through.",
                eng,
            )
        return eng

    def _apply_serpapi_defaults(self, params: dict[str, Any]) -> None:
        """Apply configured SerpApi defaults to *params* without overriding callers.

        Mutates *params* in place. ``output`` / ``no_cache`` / ``zero_trace`` /
        ``json_restrictor`` defaults are only applied when the caller has not
        already supplied them.
        """
        params.setdefault("output", self._default_output)
        if self._default_no_cache and "no_cache" not in params:
            params["no_cache"] = True
        if self._default_zero_trace and "zero_trace" not in params:
            params["zero_trace"] = True
        if self._default_json_restrictor and "json_restrictor" not in params:
            params["json_restrictor"] = self._default_json_restrictor

    @staticmethod
    def _normalize_bool_params(params: dict[str, Any]) -> dict[str, Any]:
        """Render Python bools as SerpApi's lowercase ``"true"``/``"false"`` strings."""
        rendered: dict[str, Any] = {}
        for k, v in params.items():
            if isinstance(v, bool):
                rendered[k] = "true" if v else "false"
            else:
                rendered[k] = v
        return rendered

    def _build_search_params(
        self,
        query: str,
        *,
        engine: str,
        count: int | None,
        country: str | None,
        language: str | None,
        device: str | None,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        """Assemble SerpApi ``/search`` query parameters from normalized inputs.

        Maps the cross-provider :meth:`web_search` arguments onto SerpApi's
        parameter names and merges any provider-specific ``extra`` parameters
        (which always win over the mapped defaults):

        * ``query``     -> the engine's free-text parameter (``q`` / ``query`` /
          ``p`` / ``text`` / ``search_query`` / ``term`` / ``_nkw`` /
          ``find_desc``; see :data:`_QUERY_PARAM_BY_ENGINE`).
        * ``count``     -> ``num`` (best-effort; honored by engines such as
          ``google_scholar`` / ``google_patents`` / ``naver``; the standard
          ``google`` web engine ignores it — use ``start`` for pagination).
        * ``country``   -> ``gl``  (Google two-letter country).
        * ``language``  -> ``hl``  (Google two-letter language).
        * ``device``    -> ``device`` (only when not the default ``"desktop"``).

        Args:
            query: Free-text search query (mapped to the engine query param).
            engine: Resolved engine identifier.
            count: Desired result count (-> ``num``).
            country: ISO country code (-> ``gl``).
            language: ISO language code (-> ``hl``).
            device: ``desktop`` / ``tablet`` / ``mobile``.
            extra: Provider-specific pass-through parameters (override mapped).

        Returns:
            A dict of query parameters ready for :meth:`_get` (booleans rendered
            as SerpApi string literals; ``None`` values dropped downstream).
        """
        params: dict[str, Any] = {"engine": engine}

        query_param = _QUERY_PARAM_BY_ENGINE.get(engine, "q")
        if query:
            params[query_param] = query
        if count:
            params["num"] = count
        if country:
            params["gl"] = country
        if language:
            params["hl"] = language
        if device and device.lower() != "desktop":
            params["device"] = device.lower()

        self._apply_serpapi_defaults(params)

        # Caller-supplied params always win (e.g. explicit num/gl/location/tbm/tbs).
        params.update(extra)
        return self._normalize_bool_params(params)

    # ------------------------------------------------------------------
    # Low-level, faithful pass-through (provider-specific)
    # ------------------------------------------------------------------
    async def search(
        self,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> WebSearchResult:
        """Run a raw SerpApi ``GET /search`` (engine-agnostic pass-through).

        This is the faithful, low-level counterpart to :meth:`web_search`,
        mirroring the official ``serpapi`` client's ``search(params)``. No
        cross-provider argument mapping is performed: every key in *params* /
        *kwargs* is sent to SerpApi verbatim (except ``api_key``, which is always
        injected from configuration). The configured ``output`` /
        ``no_cache`` / ``zero_trace`` / ``json_restrictor`` defaults are applied
        unless explicitly overridden.

        Args:
            params: A dict of SerpApi request parameters (e.g.
                ``{"engine": "google_scholar", "q": "graph neural networks",
                "num": 20}``).
            **kwargs: Additional SerpApi parameters merged into *params*.

        Returns:
            A :class:`~llmcore.search.models.WebSearchResult`. ``raw`` always
            holds SerpApi's full payload; ``items`` are normalized best-effort
            for the resolved engine. For ``output=html`` the raw body is stored
            on ``raw["raw_html"]`` and ``items`` is empty.

        Raises:
            SearchProviderError: On authentication or transport faults.
        """
        merged: dict[str, Any] = dict(params or {})
        merged.update(kwargs)
        engine = self._resolve_engine(merged.get("engine"))
        merged["engine"] = engine
        self._apply_serpapi_defaults(merged)
        merged = self._normalize_bool_params(merged)
        query_label = str(
            merged.get(_QUERY_PARAM_BY_ENGINE.get(engine, "q")) or merged.get("q") or ""
        )

        trigger_at = datetime.now(timezone.utc)
        with self._span("search", engine=engine):
            response = await self._get(SEARCH_ENDPOINT, merged)
        fetched_at = datetime.now(timezone.utc)
        return self._response_to_web_result(
            response, query_label, engine, trigger_at, fetched_at
        )

    # ------------------------------------------------------------------
    # web_search (cross-provider contract)
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
        """Run a SerpApi search and return normalized organic results.

        The ``engine`` selects the SerpApi engine/vertical (defaults to the
        configured ``default_engine``, i.e. ``google``). Cross-provider
        arguments are mapped to SerpApi parameters (see
        :meth:`_build_search_params`); any other SerpApi parameter may be passed
        through ``**kwargs`` (e.g. ``location``, ``uule``, ``lat``, ``lon``,
        ``google_domain``, ``tbm``, ``tbs``, ``safe``, ``start``, ``no_cache``,
        ``output``, ``json_restrictor``).

        Args:
            query: The search query string (mapped to the engine's query param).
            count: Desired number of results (mapped to ``num``; best-effort —
                the standard ``google`` web engine ignores it, use ``start``).
            country: ISO country code (mapped to ``gl``).
            language: ISO language code (mapped to ``hl``).
            device: ``"desktop"`` (default), ``"tablet"`` or ``"mobile"``.
            engine: SerpApi engine (e.g. ``"google"``, ``"bing"``,
                ``"google_news"``). ``None`` uses the configured default.
            mode: ``"sync"`` (blocking ``GET /search``) or ``"async"`` (submit
                with ``async=true`` then poll the Search Archive).
            **kwargs: Any additional SerpApi request parameter (passed through
                verbatim; overrides mapped defaults).

        Returns:
            A :class:`~llmcore.search.models.WebSearchResult` with ``engine``
            set to ``"serpapi:<engine>"`` and ``raw`` holding the full payload.

        Raises:
            SearchProviderError: On configuration or transport faults, or on
                async submit/poll failure or timeout.
        """
        if not isinstance(query, str):
            raise SearchProviderError(self.get_name(), "web_search query must be a string.")

        resolved_engine = self._resolve_engine(engine)
        extra = dict(kwargs)

        if mode == "async":
            return await self._async_search(
                query,
                engine=resolved_engine,
                count=count,
                country=country,
                language=language,
                device=device,
                extra=extra,
            )

        params = self._build_search_params(
            query,
            engine=resolved_engine,
            count=count,
            country=country,
            language=language,
            device=device,
            extra=extra,
        )

        trigger_at = datetime.now(timezone.utc)
        with self._span("web_search", engine=resolved_engine):
            response = await self._get(SEARCH_ENDPOINT, params)
        fetched_at = datetime.now(timezone.utc)
        return self._response_to_web_result(
            response, query, resolved_engine, trigger_at, fetched_at
        )

    def _response_to_web_result(
        self,
        response: Any,
        query: str,
        engine: str,
        trigger_at: datetime,
        fetched_at: datetime,
    ) -> WebSearchResult:
        """Build a :class:`WebSearchResult` from a raw ``/search`` response."""
        engine_label = f"serpapi:{engine}"
        if response.status_code != 200:
            return WebSearchResult(
                success=False,
                provider=self._instance_label(),
                query=query,
                engine=engine_label,
                error=f"/search failed (HTTP {response.status_code}): "
                f"{self._error_message(response)}",
                trigger_sent_at=trigger_at,
                data_fetched_at=fetched_at,
            )

        payload = self._decode(response)

        # output=html (or any non-JSON body) -> preserve raw text, no items.
        if not isinstance(payload, dict):
            return WebSearchResult(
                success=True,
                provider=self._instance_label(),
                query=query,
                engine=engine_label,
                items=[],
                raw={"raw_html": payload},
                trigger_sent_at=trigger_at,
                data_fetched_at=fetched_at,
            )

        # A 200 response may still carry an engine-level error (e.g. a parameter
        # problem reported in-body). Surface it as a soft failure.
        body_error = payload.get("error")
        if body_error and not payload.get("organic_results"):
            return WebSearchResult(
                success=False,
                provider=self._instance_label(),
                query=query,
                engine=engine_label,
                error=str(body_error),
                raw=payload,
                trigger_sent_at=trigger_at,
                data_fetched_at=fetched_at,
            )

        items, total = _normalize_serpapi(payload, engine)
        return WebSearchResult(
            success=True,
            provider=self._instance_label(),
            query=query,
            engine=engine_label,
            items=items,
            total_results=total,
            raw=payload,
            trigger_sent_at=trigger_at,
            data_fetched_at=fetched_at,
        )

    # ------------------------------------------------------------------
    # Async pipeline (async=true + Search Archive polling)
    # ------------------------------------------------------------------
    async def _async_search(
        self,
        query: str,
        *,
        engine: str,
        count: int | None,
        country: str | None,
        language: str | None,
        device: str | None,
        extra: dict[str, Any],
    ) -> WebSearchResult:
        """Submit an async search (``async=true``) and poll the Search Archive.

        SerpApi returns immediately with only ``search_metadata`` (``id`` +
        ``status``); we extract the id and poll ``GET /searches/{id}`` until the
        status is ``Success`` (or ``Error``).

        Note:
            ``async`` and ``no_cache`` are mutually exclusive on SerpApi. If a
            ``no_cache`` default/override is present it is dropped (with a
            warning) so the async submit is valid.
        """
        params = self._build_search_params(
            query,
            engine=engine,
            count=count,
            country=country,
            language=language,
            device=device,
            extra=extra,
        )
        # async is incompatible with no_cache.
        if str(params.get("no_cache", "")).lower() == "true":
            logger.warning(
                "SerpApi 'async' and 'no_cache' are mutually exclusive; dropping no_cache."
            )
            params.pop("no_cache", None)
        params["async"] = "true"

        engine_label = f"serpapi:{engine}"
        trigger_at = datetime.now(timezone.utc)
        with self._span("web_search_async_submit", engine=engine):
            submit = await self._get(SEARCH_ENDPOINT, params)

        if submit.status_code != 200:
            return WebSearchResult(
                success=False,
                provider=self._instance_label(),
                query=query,
                engine=engine_label,
                error=f"async submit failed (HTTP {submit.status_code}): "
                f"{self._error_message(submit)}",
                trigger_sent_at=trigger_at,
                data_fetched_at=datetime.now(timezone.utc),
            )

        submit_payload = self._decode(submit)
        metadata = submit_payload.get("search_metadata") if isinstance(submit_payload, dict) else None
        search_id = metadata.get("id") if isinstance(metadata, dict) else None
        if not search_id:
            raise SearchProviderError(
                self.get_name(),
                "async submit did not return a search_metadata.id to poll.",
            )

        payload = await self._poll_archive(search_id)
        fetched_at = datetime.now(timezone.utc)

        meta = payload.get("search_metadata", {}) if isinstance(payload, dict) else {}
        status = str(meta.get("status", "")).lower()
        if status in _ASYNC_STATUS_ERROR:
            return WebSearchResult(
                success=False,
                provider=self._instance_label(),
                query=query,
                engine=engine_label,
                error=str(payload.get("error") or "async search reported status Error"),
                raw=payload,
                trigger_sent_at=trigger_at,
                data_fetched_at=fetched_at,
            )

        items, total = _normalize_serpapi(payload, engine)
        return WebSearchResult(
            success=True,
            provider=self._instance_label(),
            query=query,
            engine=engine_label,
            items=items,
            total_results=total,
            raw=payload,
            trigger_sent_at=trigger_at,
            data_fetched_at=fetched_at,
        )

    async def _poll_archive(self, search_id: str) -> dict[str, Any]:
        """Poll ``GET /searches/{id}`` until the search is done or times out.

        Args:
            search_id: The ``search_metadata.id`` returned by the async submit.

        Returns:
            The full result payload once ``search_metadata.status`` is
            ``Success`` (or ``Error`` — returned so the caller can surface it).

        Raises:
            SearchProviderError: On poll transport failure or after
                ``poll_timeout`` seconds without completion.
        """
        path = f"/searches/{search_id}"
        # Guard against an infinite loop when poll_interval == 0: bound the number
        # of iterations independently of the (possibly zero) elapsed accounting.
        step = self._poll_interval if self._poll_interval > 0 else 1
        max_iterations = max(1, self._poll_timeout // step + 2)
        elapsed = 0
        iterations = 0
        while True:
            with self._span("web_search_async_poll", search_id=search_id):
                poll = await self._get(path, {"output": "json"})
            if poll.status_code >= 400:
                raise SearchProviderError(
                    self.get_name(),
                    f"async poll failed (search_id={search_id}): {self._error_message(poll)}",
                    status_code=poll.status_code,
                )
            payload = self._decode(poll)
            meta = payload.get("search_metadata", {}) if isinstance(payload, dict) else {}
            status = str(meta.get("status", "")).lower()
            if status in _ASYNC_STATUS_DONE or status in _ASYNC_STATUS_ERROR:
                return payload if isinstance(payload, dict) else {}
            iterations += 1
            if elapsed >= self._poll_timeout or iterations >= max_iterations:
                raise SearchProviderError(
                    self.get_name(),
                    f"async polling timed out after {self._poll_timeout}s "
                    f"(search_id={search_id}, last status={meta.get('status')}).",
                )
            await asyncio.sleep(self._poll_interval)
            elapsed += step

    # ------------------------------------------------------------------
    # batch_search (client-side bounded fan-out)
    # ------------------------------------------------------------------
    async def batch_search(
        self,
        queries: list[Any],
        *,
        count: int = 10,
        country: str | None = None,
        language: str = "en",
        search_type: str = "search",
        **kwargs: Any,
    ) -> list[WebSearchResult]:
        """Run multiple searches concurrently (client-side fan-out).

        SerpApi has **no** server-side batch endpoint, so this issues concurrent
        :meth:`web_search` requests bounded by ``max_concurrency`` and returns
        one result per input query, in order. Each query consumes one search
        credit.

        Args:
            queries: A list of query strings, or a list of SerpApi parameter
                dicts (e.g. ``{"engine": "google_news", "q": "ai"}``). A dict
                must contain a recognizable query field (``q`` / ``query`` /
                ``p`` / ``text`` / ``search_query`` / ``term`` / ``_nkw`` /
                ``find_desc``) for the chosen engine.
            count: Default ``num`` applied to string queries.
            country: Default ``gl`` applied to string queries.
            language: Default ``hl`` applied to string queries.
            search_type: Engine to use for string queries. ``"search"`` (the
                cross-provider default) maps to the configured ``default_engine``;
                any other value is used directly as the SerpApi ``engine``.
                A per-query dict may override this with its own ``engine``.
            **kwargs: Applied to string queries (e.g. ``location``, ``tbm``,
                ``tbs``, ``start``).

        Returns:
            A list of :class:`~llmcore.search.models.WebSearchResult`, one per
            input query, in order. Individual failures are returned as
            ``success=False`` results rather than raising.

        Raises:
            SearchProviderError: If ``queries`` is empty / not a list, or a dict
                item lacks a recognizable query field.
        """
        if not isinstance(queries, (list, tuple)) or not queries:
            raise SearchProviderError(
                self.get_name(), "batch_search requires a non-empty list of queries."
            )

        batch_engine = (
            self._default_engine if search_type in (None, "search") else str(search_type)
        )
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def _run_one(item: Any) -> WebSearchResult:
            async with semaphore:
                if isinstance(item, str):
                    return await self.web_search(
                        item,
                        count=count,
                        country=country,
                        language=language,
                        engine=batch_engine,
                        **dict(kwargs),
                    )
                if isinstance(item, dict):
                    obj = dict(item)
                    engine = self._resolve_engine(obj.pop("engine", batch_engine))
                    query_param = _QUERY_PARAM_BY_ENGINE.get(engine, "q")
                    query = obj.pop(query_param, None) or obj.pop("q", None)
                    if not query:
                        raise SearchProviderError(
                            self.get_name(),
                            f"Each batch query dict must include a '{query_param}' (or 'q') field.",
                        )
                    return await self.web_search(
                        str(query),
                        engine=engine,
                        **obj,
                    )
                raise SearchProviderError(
                    self.get_name(),
                    f"Unsupported batch query item type: {type(item).__name__}.",
                )

        return await asyncio.gather(*[_run_one(q) for q in queries])

    # ------------------------------------------------------------------
    # Search Archive API (provider-specific)
    # ------------------------------------------------------------------
    async def search_archive(
        self,
        search_id: str,
        *,
        engine: str | None = None,
        **kwargs: Any,
    ) -> WebSearchResult:
        """Fetch a previously-run search from the Search Archive by id.

        Wraps ``GET /searches/{search_id}``. Useful for retrieving the result of
        an async submit, or re-reading a result without spending a credit (cached
        archive reads are free).

        Args:
            search_id: The ``search_metadata.id`` of a prior search.
            engine: Engine hint used only to normalize ``items`` (the archive
                payload already contains the engine in ``search_parameters``).
            **kwargs: Additional query parameters (e.g. ``output``).

        Returns:
            A :class:`~llmcore.search.models.WebSearchResult` reconstructed from
            the archived payload.

        Raises:
            SearchProviderError: If ``search_id`` is empty or the request fails.
        """
        if not search_id or not isinstance(search_id, str):
            raise SearchProviderError(
                self.get_name(), "search_archive requires a non-empty search_id."
            )
        params = {"output": "json"}
        params.update(kwargs)
        trigger_at = datetime.now(timezone.utc)
        with self._span("search_archive", search_id=search_id):
            response = await self._get(f"/searches/{search_id}", params)
        fetched_at = datetime.now(timezone.utc)

        payload = self._decode(response) if response.status_code == 200 else None
        resolved_engine = engine
        if resolved_engine is None and isinstance(payload, dict):
            sp = payload.get("search_parameters")
            if isinstance(sp, dict):
                resolved_engine = sp.get("engine")
        resolved_engine = self._resolve_engine(resolved_engine)

        query = ""
        if isinstance(payload, dict):
            sp = payload.get("search_parameters")
            if isinstance(sp, dict):
                query = str(sp.get("q") or sp.get("query") or "")
        return self._response_to_web_result(
            response, query, resolved_engine, trigger_at, fetched_at
        )

    # ------------------------------------------------------------------
    # Account API (provider-specific; free)
    # ------------------------------------------------------------------
    async def account(self, **kwargs: Any) -> dict[str, Any]:
        """Return SerpApi account information (plan, quota, searches left).

        Wraps ``GET /account.json`` — this endpoint is **free** and does not
        count against the monthly quota.

        Args:
            **kwargs: Additional query parameters (rarely needed).

        Returns:
            The parsed account payload (e.g. ``plan_name``, ``searches_per_month``,
            ``plan_searches_left``, ``total_searches_left``, ``this_month_usage``,
            ``account_rate_limit_per_hour``).

        Raises:
            SearchProviderError: On authentication or transport faults, or a
                non-200 response.
        """
        with self._span("account"):
            response = await self._get(ACCOUNT_ENDPOINT, dict(kwargs))
        if response.status_code != 200:
            raise SearchProviderError(
                self.get_name(),
                f"account lookup failed: {self._error_message(response)}",
                status_code=response.status_code,
            )
        payload = self._decode(response)
        return payload if isinstance(payload, dict) else {}

    # ------------------------------------------------------------------
    # Locations API (provider-specific)
    # ------------------------------------------------------------------
    async def locations(
        self,
        *,
        q: str | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return canonical SerpApi locations (for the ``location`` parameter).

        Wraps ``GET /locations.json``. Use the returned ``canonical_name`` as the
        ``location`` value for precise geolocation.

        Args:
            q: Restrict results to locations containing this string.
            limit: Limit the number of locations returned.
            **kwargs: Additional query parameters.

        Returns:
            A list of location dicts (empty list when none match).

        Raises:
            SearchProviderError: On transport faults or a non-200 response.
        """
        params: dict[str, Any] = dict(kwargs)
        if q is not None:
            params["q"] = q
        if limit is not None:
            params["limit"] = limit
        with self._span("locations"):
            response = await self._get(LOCATIONS_ENDPOINT, params)
        if response.status_code != 200:
            raise SearchProviderError(
                self.get_name(),
                f"locations lookup failed: {self._error_message(response)}",
                status_code=response.status_code,
            )
        payload = self._decode(response)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):  # pragma: no cover - defensive
            return [payload]
        return []

    # ------------------------------------------------------------------
    # health / lifecycle
    # ------------------------------------------------------------------
    async def health_check(self) -> bool:
        """Verify credentials/connectivity via the **free** Account API.

        Unlike Serper (which has no free endpoint), SerpApi exposes
        ``GET /account.json`` at no cost, so this consumes **zero** search
        credits. Returns ``True`` on HTTP 200, ``False`` on auth failure or an
        unreachable host; never raises.
        """
        try:
            response = await self._get(ACCOUNT_ENDPOINT, {})
        except SearchProviderError:
            return False
        return response.status_code == 200

    async def close(self) -> None:
        """Close the underlying :class:`httpx.AsyncClient` (idempotent)."""
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Error closing SerpApi httpx client: %s", exc)
            self._client = None
