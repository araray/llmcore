# src/llmcore/search/providers/semanticscholar_provider.py
"""Semantic Scholar (S2) academic **search** provider for LLMCore.

Implements :class:`llmcore.search.base.BaseSearchProvider` against the Semantic
Scholar APIs using :mod:`httpx` directly — consistent with the native LLM
providers and the sibling Serper / Bright Data / SerpApi search providers (no
vendor SDK dependency).

Semantic Scholar is a free, AI-powered academic search engine covering 200M+
papers across all disciplines. This provider wraps the three public S2 APIs,
which share a host (``https://api.semanticscholar.org``) but use different path
prefixes:

* **Academic Graph API** (``/graph/v1``) — paper relevance/bulk/title search,
  text-snippet search, autocomplete, paper & author details, citations,
  references, and batch lookups.
* **Recommendations API** (``/recommendations/v1``) — recommended papers from a
  single seed paper or from positive/negative example lists.
* **Datasets API** (``/datasets/v1``) — bulk-corpus release/diff download links.

Supported capabilities
-----------------------
=================  ============================================================
Capability         Semantic Scholar surface
=================  ============================================================
``web_search``     ``GET /graph/v1/paper/search`` (relevance, default), with a
                   ``search_type`` of ``relevance`` | ``bulk`` | ``match`` |
                   ``snippet`` selecting the bulk-search, title-match, or
                   text-snippet endpoints respectively.
``batch_search``   **Client-side** bounded fan-out of concurrent ``web_search``
                   calls (S2 has no multi-query search endpoint). Returns one
                   result per input query, in order.
=================  ============================================================

S2's *Recommendations* API is an item-to-item recommender (paper → papers), not
a text-query "AI search", and its *Datasets* API is a bulk-corpus download
workflow (pre-signed S3 URLs for 100M-record partitions) — neither matches the
cross-provider ``discover`` (text-query) or ``dataset_search`` (filter →
snapshot → inline records) contracts. They are therefore exposed as
**provider-specific methods** rather than advertised capabilities:

* paper graph: :meth:`paper`, :meth:`paper_batch`, :meth:`paper_authors`,
  :meth:`paper_citations`, :meth:`paper_references`, :meth:`paper_match`,
  :meth:`autocomplete`, :meth:`snippet_search`.
* authors: :meth:`author`, :meth:`author_batch`, :meth:`author_papers`,
  :meth:`author_search`.
* recommendations: :meth:`recommend_papers`, :meth:`recommend_from_examples`.
* datasets: :meth:`list_releases`, :meth:`get_release`, :meth:`get_dataset`,
  :meth:`get_dataset_diffs`.

Authentication & rate limits
-----------------------------
The S2 API key is **optional**. Without a key the provider shares the public
unauthenticated pool (a few thousand requests per 5 minutes shared across *all*
anonymous clients, and throttled under load); a free key grants a dedicated
limit (introductory 1 request/second on all endpoints). Because the key is
optional, this provider **loads and operates without one** — unlike the other
search providers, a missing key is not an error. The key, when present, is sent
via the ``x-api-key`` header (env: ``SEMANTIC_SCHOLAR_API_KEY``; ``S2_API_KEY``
is also honored).

S2 explicitly **requires** exponential backoff; this provider retries ``429`` /
``5xx`` responses with capped exponential backoff and supports optional proactive
request spacing (``min_request_interval``) plus a conservative default batch
concurrency of 1.

References:
    * Tutorial / overview: https://www.semanticscholar.org/product/api
    * Graph API: https://api.semanticscholar.org/api-docs/graph
    * Recommendations API: https://api.semanticscholar.org/api-docs/recommendations
    * Datasets API: https://api.semanticscholar.org/api-docs/datasets
    * Rate limits / backoff: https://github.com/allenai/s2-folks
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

from ...exceptions import SearchProviderError
from ..base import BaseSearchProvider, SearchCapability
from ..models import SearchItem, WebSearchResult

try:  # pragma: no cover - exercised indirectly
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]
    _HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)

# --- Endpoints ---------------------------------------------------------------
DEFAULT_BASE_URL = "https://api.semanticscholar.org"
GRAPH_PREFIX = "/graph/v1"
RECOMMENDATIONS_PREFIX = "/recommendations/v1"
DATASETS_PREFIX = "/datasets/v1"
WEBSITE_PAPER_URL = "https://www.semanticscholar.org/paper/"
WEBSITE_AUTHOR_URL = "https://www.semanticscholar.org/author/"

# --- Defaults ----------------------------------------------------------------
DEFAULT_SEARCH_TYPE = "relevance"
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 3
# Conservative: S2's unauthenticated pool is shared and key tier is ~1 RPS, so
# we do not fan out concurrently by default. Raise it when you hold a key.
DEFAULT_MAX_CONCURRENCY = 1
DEFAULT_MIN_REQUEST_INTERVAL = 0.0
DEFAULT_TOKEN_ENV_VAR = "SEMANTIC_SCHOLAR_API_KEY"
SECONDARY_TOKEN_ENV_VARS = ("S2_API_KEY",)

# Useful default field sets so results are informative out of the box (S2
# otherwise returns only ``paperId`` + ``title``). Overridable via config.
DEFAULT_PAPER_FIELDS = (
    "title,abstract,url,venue,year,publicationDate,authors,"
    "citationCount,influentialCitationCount,externalIds,openAccessPdf,publicationTypes"
)
DEFAULT_AUTHOR_FIELDS = "name,url,affiliations,homepage,paperCount,citationCount,hIndex"

# Per-endpoint maximum ``limit`` values (from the S2 API documentation).
_LIMIT_CAP = {
    "relevance": 100,
    "bulk": 1000,
    "match": 1,
    "snippet": 1000,
    "author_search": 1000,
    "recommendations": 500,
    "list": 1000,
}

# web_search ``search_type`` -> (endpoint path, limit cap key).
_SEARCH_TYPE_ENDPOINT = {
    "relevance": (f"{GRAPH_PREFIX}/paper/search", "relevance"),
    "bulk": (f"{GRAPH_PREFIX}/paper/search/bulk", "bulk"),
    "match": (f"{GRAPH_PREFIX}/paper/search/match", "match"),
    "snippet": (f"{GRAPH_PREFIX}/snippet/search", "snippet"),
}

# S2 paper-search query parameters that pass through verbatim from **kwargs.
_PAPER_FILTER_PARAMS = frozenset(
    {
        "fields",
        "offset",
        "sort",
        "token",
        "publicationTypes",
        "openAccessPdf",
        "minCitationCount",
        "publicationDateOrYear",
        "year",
        "venue",
        "fieldsOfStudy",
        # snippet-only
        "paperIds",
        "authors",
        "insertedBefore",
    }
)


def _coerce_int(value: Any) -> int | None:
    """Coerce an S2 ``total`` value (often a string) into an ``int``.

    Args:
        value: The raw value (``total`` is documented as a string).

    Returns:
        The integer total, or ``None`` when it cannot be determined.
    """
    if isinstance(value, bool):
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


def _abstract_or_tldr(paper: dict[str, Any]) -> str:
    """Return a paper's abstract, falling back to its TLDR summary."""
    abstract = paper.get("abstract")
    if abstract:
        return str(abstract)
    tldr = paper.get("tldr")
    if isinstance(tldr, dict) and tldr.get("text"):
        return str(tldr["text"])
    return ""


def _venue_label(paper: dict[str, Any]) -> str:
    """Return a best-effort venue/journal display string for a paper."""
    venue = paper.get("venue")
    if venue:
        return str(venue)
    journal = paper.get("journal")
    if isinstance(journal, dict) and journal.get("name"):
        return str(journal["name"])
    pv = paper.get("publicationVenue")
    if isinstance(pv, dict) and pv.get("name"):
        return str(pv["name"])
    return ""


def _paper_url(paper: dict[str, Any]) -> str:
    """Return a paper's S2 website URL, constructing one from its id if needed."""
    url = paper.get("url")
    if url:
        return str(url)
    pid = paper.get("paperId")
    if pid:
        return f"{WEBSITE_PAPER_URL}{pid}"
    corpus = paper.get("corpusId")
    if corpus:
        return f"{WEBSITE_PAPER_URL}CorpusID:{corpus}"
    return ""


def _paper_to_item(paper: dict[str, Any], position: int) -> SearchItem:
    """Map an S2 paper object onto a provider-agnostic :class:`SearchItem`."""
    return SearchItem(
        position=position,
        title=str(paper.get("title") or ""),
        url=_paper_url(paper),
        description=_abstract_or_tldr(paper),
        displayed_url=_venue_label(paper),
    )


def _author_url(author: dict[str, Any]) -> str:
    """Return an author's S2 website URL, constructing one from its id if needed."""
    url = author.get("url")
    if url:
        return str(url)
    aid = author.get("authorId")
    if aid:
        return f"{WEBSITE_AUTHOR_URL}{aid}"
    return ""


def _author_to_item(author: dict[str, Any], position: int) -> SearchItem:
    """Map an S2 author object onto a provider-agnostic :class:`SearchItem`."""
    affiliations = author.get("affiliations")
    desc = ", ".join(affiliations) if isinstance(affiliations, list) else ""
    return SearchItem(
        position=position,
        title=str(author.get("name") or ""),
        url=_author_url(author),
        description=desc,
        displayed_url=str(author.get("homepage") or ""),
    )


def _snippet_to_item(entry: dict[str, Any], position: int) -> SearchItem:
    """Map an S2 snippet-search entry onto a provider-agnostic :class:`SearchItem`.

    The snippet *text* (the passage relevant to the query) becomes the item
    ``description`` — directly useful for RAG grounding.
    """
    snippet = entry.get("snippet") if isinstance(entry, dict) else None
    paper = entry.get("paper") if isinstance(entry, dict) else None
    snippet = snippet if isinstance(snippet, dict) else {}
    paper = paper if isinstance(paper, dict) else {}
    return SearchItem(
        position=position,
        title=str(paper.get("title") or ""),
        url=_paper_url(paper),
        description=str(snippet.get("text") or ""),
        displayed_url=str(snippet.get("snippetKind") or ""),
    )


def _normalize_items(data: Any, kind: str) -> list[SearchItem]:
    """Normalize an S2 ``data``/``matches`` array into :class:`SearchItem` objects.

    Args:
        data: The result array (papers, authors, citations, references,
            snippets, or autocomplete matches).
        kind: One of ``"papers"``, ``"authors"``, ``"citations"``,
            ``"references"``, ``"snippets"``, ``"autocomplete"``.

    Returns:
        A list of mapped :class:`SearchItem` objects (empty when *data* is not a
        non-empty list).
    """
    if not isinstance(data, list):
        return []
    items: list[SearchItem] = []
    for i, entry in enumerate(data, start=1):
        if not isinstance(entry, dict):
            if isinstance(entry, str):
                items.append(SearchItem(position=i, title=entry))
            continue
        if kind == "snippets":
            items.append(_snippet_to_item(entry, i))
        elif kind == "authors":
            items.append(_author_to_item(entry, i))
        elif kind == "citations":
            inner = entry.get("citingPaper")
            items.append(_paper_to_item(inner if isinstance(inner, dict) else entry, i))
        elif kind == "references":
            inner = entry.get("citedPaper")
            items.append(_paper_to_item(inner if isinstance(inner, dict) else entry, i))
        elif kind == "autocomplete":
            items.append(
                SearchItem(
                    position=i,
                    title=str(entry.get("title") or ""),
                    url=(f"{WEBSITE_PAPER_URL}{entry['id']}" if entry.get("id") else ""),
                    description=str(entry.get("authorsYear") or ""),
                )
            )
        else:  # papers
            items.append(_paper_to_item(entry, i))
    return items


class SemanticScholarSearchProvider(BaseSearchProvider):
    """Semantic Scholar implementation of :class:`BaseSearchProvider`.

    Args:
        config: Settings from ``[search_providers.semanticscholar]``. Recognized
            keys: ``api_key`` / ``token`` / ``api_key_env_var`` (all **optional**
            — S2 works keyless), ``base_url``, ``default_search_type``
            (``relevance`` | ``bulk`` | ``match`` | ``snippet``),
            ``default_fields`` (paper fields), ``default_author_fields``,
            ``timeout``, ``max_retries``, ``max_concurrency`` (batch fan-out),
            ``min_request_interval`` (seconds; proactive request spacing),
            ``ssl_verify``.
        log_raw_payloads: Whether to log raw request/response payloads (the
            ``x-api-key`` header is never logged).

    Raises:
        SearchProviderError: Never raised for a missing key (S2 is keyless);
            only ``ImportError`` via ``ConfigError`` semantics if ``httpx`` is
            unavailable.
    """

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False) -> None:
        super().__init__(config, log_raw_payloads)

        if not _HTTPX_AVAILABLE:
            # Surface as ImportError so the manager logs an install hint and skips.
            raise ImportError(
                "The 'httpx' package is required for the Semantic Scholar provider. "
                "Install with: pip install llmcore[semanticscholar]"
            )

        # --- API key (OPTIONAL: S2 works without one) ---
        key = config.get("api_key") or config.get("token")
        if not key:
            env_var = config.get("api_key_env_var", DEFAULT_TOKEN_ENV_VAR)
            key = os.environ.get(env_var)
            if not key:
                for fallback in (DEFAULT_TOKEN_ENV_VAR, *SECONDARY_TOKEN_ENV_VARS):
                    key = os.environ.get(fallback)
                    if key:
                        break
        self._api_key: str | None = str(key).strip() if key else None
        if not self._api_key:
            logger.debug(
                "Semantic Scholar provider initialized WITHOUT an API key "
                "(shared, rate-limited public pool)."
            )

        # --- Endpoints / defaults / tuning ---
        self._base_url = str(config.get("base_url", DEFAULT_BASE_URL)).rstrip("/")

        st = str(config.get("default_search_type", DEFAULT_SEARCH_TYPE)).strip().lower()
        if st not in _SEARCH_TYPE_ENDPOINT:
            logger.warning(
                "Unsupported default_search_type '%s' for Semantic Scholar; "
                "falling back to '%s'.",
                st,
                DEFAULT_SEARCH_TYPE,
            )
            st = DEFAULT_SEARCH_TYPE
        self._default_search_type = st

        self._default_fields = str(config.get("default_fields", DEFAULT_PAPER_FIELDS))
        self._default_author_fields = str(
            config.get("default_author_fields", DEFAULT_AUTHOR_FIELDS)
        )

        self._timeout = int(config.get("timeout", DEFAULT_TIMEOUT))
        self._max_retries = int(config.get("max_retries", DEFAULT_MAX_RETRIES))
        self._max_concurrency = max(1, int(config.get("max_concurrency", DEFAULT_MAX_CONCURRENCY)))
        self._min_request_interval = float(
            config.get("min_request_interval", DEFAULT_MIN_REQUEST_INTERVAL)
        )
        self._ssl_verify = bool(config.get("ssl_verify", True))

        self._client: Any | None = None  # httpx.AsyncClient, created lazily
        self._throttle_lock = asyncio.Lock()
        self._last_request_ts = 0.0  # monotonic clock for request spacing
        logger.debug(
            "SemanticScholarSearchProvider initialized (base_url=%s, default_search_type=%s, "
            "authenticated=%s).",
            self._base_url,
            self._default_search_type,
            bool(self._api_key),
        )

    # ------------------------------------------------------------------
    # Identity / capabilities
    # ------------------------------------------------------------------
    def get_name(self) -> str:
        """Return the provider type name (``"semanticscholar"``)."""
        return "semanticscholar"

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
            headers = {
                "Accept": "application/json",
                "User-Agent": "llmcore-semanticscholar-search",
            }
            if self._api_key:
                headers["x-api-key"] = self._api_key
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=self._timeout,
                verify=self._ssl_verify,
            )
        return self._client

    def _log_payload(self, label: str, payload: Any) -> None:
        """Emit a raw-payload debug log when enabled (key is in headers, not here)."""
        if self.log_raw_payloads_enabled:
            try:
                logger.debug("[semanticscholar] %s: %s", label, _json.dumps(payload, default=str)[:4000])
            except Exception:  # pragma: no cover - defensive
                logger.debug("[semanticscholar] %s: <unserializable>", label)

    async def _throttle(self) -> None:
        """Enforce optional proactive request spacing (``min_request_interval``).

        Serializes request *starts* so that consecutive requests are at least
        ``min_request_interval`` seconds apart. A no-op when the interval is 0.
        """
        if self._min_request_interval <= 0:
            return
        async with self._throttle_lock:
            now = time.monotonic()
            wait = self._min_request_interval - (now - self._last_request_ts)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request_ts = time.monotonic()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        """Issue an HTTP request with retry + mandatory exponential backoff.

        S2 requires exponential backoff; ``429`` (rate limit) and ``5xx``
        responses are retried with capped exponential backoff. ``401`` / ``403``
        (auth) raise immediately; ``404`` is returned to the caller (some
        endpoints, e.g. title match, use it as a "no match" signal).

        Args:
            method: HTTP verb (``"GET"`` / ``"POST"``).
            path: Full path including the API prefix (e.g.
                ``"/graph/v1/paper/search"``).
            params: Optional query parameters (``None`` values dropped).
            json_body: Optional JSON request body (for POST endpoints).

        Returns:
            The raw :class:`httpx.Response`.

        Raises:
            SearchProviderError: On authentication failure (401/403) or repeated
                transport / rate-limit / server errors.
        """
        client = self._get_client()
        request_params = {k: v for k, v in (params or {}).items() if v is not None}
        self._log_payload(f"{method} {path} params", request_params)
        if json_body is not None:
            self._log_payload(f"{method} {path} body", json_body)

        last_exc: Exception | None = None
        attempts = max(1, self._max_retries)
        for attempt in range(attempts):
            await self._throttle()
            try:
                response = await client.request(
                    method, path, params=request_params, json=json_body
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
                    f"Authentication failed for {path}: {self._error_message(response)}",
                    status_code=response.status_code,
                )
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
        """Extract S2's ``error``/``message`` from a response, if present."""
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
    # Parameter assembly / result building
    # ------------------------------------------------------------------
    @staticmethod
    def _clamp_limit(count: int | None, cap_key: str) -> int | None:
        """Clamp a requested ``count`` to the endpoint's documented maximum."""
        if not count:
            return None
        cap = _LIMIT_CAP.get(cap_key)
        if cap is None:
            return int(count)
        return max(1, min(int(count), cap))

    @staticmethod
    def _apply_extra(params: dict[str, Any], extra: dict[str, Any]) -> None:
        """Merge S2 filter ``extra`` kwargs into *params* (in place).

        ``openAccessPdf`` is a valueless presence flag: ``True`` renders as an
        empty value (``?openAccessPdf``), ``False`` is dropped. Other booleans
        render as lowercase strings.
        """
        for k, v in extra.items():
            if k == "openAccessPdf":
                if v:
                    params["openAccessPdf"] = ""  # presence-only flag
                continue
            if isinstance(v, bool):
                params[k] = "true" if v else "false"
            else:
                params[k] = v

    def _papers_result(
        self,
        payload: Any,
        *,
        query: str,
        engine_label: str,
        item_kind: str,
        items_key: str,
        trigger_at: datetime,
        fetched_at: datetime,
        response: Any,
    ) -> WebSearchResult:
        """Build a :class:`WebSearchResult` from an S2 JSON payload.

        Args:
            payload: The decoded response body (dict, or a list for batch
                endpoints, which return a bare JSON array).
            query: The query string (or seed id) to record on the result.
            engine_label: ``"semanticscholar:<flavor>"``.
            item_kind: Normalization kind (see :func:`_normalize_items`).
            items_key: The payload key holding the result array (``"data"`` /
                ``"matches"``); ignored when *payload* is already a list.
            trigger_at: When the request was sent.
            fetched_at: When the response was received.
            response: The raw httpx response (used for the non-200 path).

        Returns:
            A :class:`WebSearchResult` (``raw`` holds the full payload).
        """
        if response.status_code != 200:
            return WebSearchResult(
                success=False,
                provider=self._instance_label(),
                query=query,
                engine=engine_label,
                error=f"{engine_label} failed (HTTP {response.status_code}): "
                f"{self._error_message(response)}",
                trigger_sent_at=trigger_at,
                data_fetched_at=fetched_at,
            )

        # Batch endpoints return a bare JSON array of papers/authors.
        if isinstance(payload, list):
            items = _normalize_items(payload, item_kind)
            return WebSearchResult(
                success=True,
                provider=self._instance_label(),
                query=query,
                engine=engine_label,
                items=items,
                total_results=len(items),
                raw={"data": payload},
                trigger_sent_at=trigger_at,
                data_fetched_at=fetched_at,
            )

        if not isinstance(payload, dict):
            return WebSearchResult(
                success=True,
                provider=self._instance_label(),
                query=query,
                engine=engine_label,
                items=[],
                raw={"raw": payload},
                trigger_sent_at=trigger_at,
                data_fetched_at=fetched_at,
            )

        body_error = payload.get("error")
        data = payload.get(items_key)
        if body_error and not data:
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

        items = _normalize_items(data, item_kind)
        total = _coerce_int(payload.get("total"))
        if total is None:
            total = len(items)
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
        """Search Semantic Scholar and return normalized paper (or snippet) results.

        The S2 search flavor is chosen with the ``search_type`` keyword:
        ``"relevance"`` (default, ``/paper/search``), ``"bulk"``
        (``/paper/search/bulk`` — large result sets with continuation
        ``token``), ``"match"`` (``/paper/search/match`` — single best title
        match), or ``"snippet"`` (``/snippet/search`` — text passages, ideal for
        RAG). Any S2 filter parameter may be passed through ``**kwargs``
        (``fields``, ``year``, ``publicationDateOrYear``, ``venue``,
        ``fieldsOfStudy``, ``publicationTypes``, ``minCitationCount``,
        ``openAccessPdf``, ``offset``, ``sort``, ``token``, and — for snippet —
        ``paperIds``, ``authors``, ``insertedBefore``).

        Args:
            query: The plain-text query (no special syntax; hyphenated terms
                yield no matches — replace hyphens with spaces).
            count: Desired number of results (mapped to ``limit``; clamped to the
                endpoint maximum: 100 relevance, 1000 bulk/snippet, 1 match).
            country: Ignored (academic search is not geolocated; accepted for
                cross-provider compatibility).
            language: Ignored (accepted for cross-provider compatibility).
            device: Ignored (accepted for cross-provider compatibility).
            engine: Ignored (S2 has a single engine; use ``search_type`` for the
                flavor).
            mode: Ignored (S2 is always synchronous).
            **kwargs: ``search_type`` plus any S2 filter parameter (passed
                through verbatim).

        Returns:
            A :class:`~llmcore.search.models.WebSearchResult` with ``engine`` set
            to ``"semanticscholar:<search_type>"`` and ``raw`` holding the full
            payload (organic ``items`` carry title/url/abstract|snippet/venue).

        Raises:
            SearchProviderError: If ``query`` is not a string, ``search_type`` is
                unknown, or on auth/transport faults.
        """
        if not isinstance(query, str):
            raise SearchProviderError(self.get_name(), "web_search query must be a string.")

        extra = dict(kwargs)
        search_type = str(extra.pop("search_type", self._default_search_type)).strip().lower()
        if search_type not in _SEARCH_TYPE_ENDPOINT:
            raise SearchProviderError(
                self.get_name(),
                f"Unsupported search_type '{search_type}'. Use one of: "
                f"{sorted(_SEARCH_TYPE_ENDPOINT)}.",
            )
        path, cap_key = _SEARCH_TYPE_ENDPOINT[search_type]

        params: dict[str, Any] = {"query": query}
        limit = self._clamp_limit(count, cap_key)
        if limit is not None and search_type != "match":  # match returns a single result
            params["limit"] = limit
        # Default fields for paper-shaped results unless the caller overrides.
        # Snippet search has its own field vocabulary, so only set if provided.
        if search_type != "snippet":
            params.setdefault("fields", self._default_fields)
        self._apply_extra(params, extra)

        engine_label = f"semanticscholar:{search_type}"
        item_kind = "snippets" if search_type == "snippet" else "papers"

        trigger_at = datetime.now(timezone.utc)
        with self._span("web_search", search_type=search_type):
            response = await self._request("GET", path, params=params)
        fetched_at = datetime.now(timezone.utc)
        payload = self._decode(response) if response.status_code == 200 else None
        return self._papers_result(
            payload,
            query=query,
            engine_label=engine_label,
            item_kind=item_kind,
            items_key="data",
            trigger_at=trigger_at,
            fetched_at=fetched_at,
            response=response,
        )

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
        """Run multiple S2 searches concurrently (client-side fan-out).

        S2 has no multi-query endpoint, so this issues concurrent
        :meth:`web_search` calls bounded by ``max_concurrency`` (default 1 —
        S2's rate limits are strict) and returns one result per input query, in
        order.

        Args:
            queries: A list of query strings, or a list of dicts each containing
                a ``query`` key plus optional per-query ``search_type`` / filter
                params.
            count: Default result count applied to string queries.
            country: Ignored (accepted for cross-provider compatibility).
            language: Ignored (accepted for cross-provider compatibility).
            search_type: S2 flavor for string queries. ``"search"`` (the
                cross-provider default) maps to the configured
                ``default_search_type``; any of ``relevance`` | ``bulk`` |
                ``match`` | ``snippet`` is used directly.
            **kwargs: Applied to string queries (S2 filter params).

        Returns:
            A list of :class:`~llmcore.search.models.WebSearchResult`, one per
            input query, in order. Individual failures are returned as
            ``success=False`` results rather than raising.

        Raises:
            SearchProviderError: If ``queries`` is empty / not a list, or a dict
                item lacks a ``query`` field.
        """
        if not isinstance(queries, (list, tuple)) or not queries:
            raise SearchProviderError(
                self.get_name(), "batch_search requires a non-empty list of queries."
            )

        batch_type = (
            self._default_search_type if search_type in (None, "search") else str(search_type)
        )
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def _run_one(item: Any) -> WebSearchResult:
            async with semaphore:
                if isinstance(item, str):
                    return await self.web_search(
                        item, count=count, search_type=batch_type, **dict(kwargs)
                    )
                if isinstance(item, dict):
                    obj = dict(item)
                    q = obj.pop("query", None) or obj.pop("q", None)
                    if not q:
                        raise SearchProviderError(
                            self.get_name(),
                            "Each batch query dict must include a 'query' field.",
                        )
                    obj.setdefault("search_type", batch_type)
                    obj.setdefault("count", count)
                    return await self.web_search(str(q), **obj)
                raise SearchProviderError(
                    self.get_name(),
                    f"Unsupported batch query item type: {type(item).__name__}.",
                )

        return await asyncio.gather(*[_run_one(q) for q in queries])

    # ------------------------------------------------------------------
    # Paper graph (provider-specific)
    # ------------------------------------------------------------------
    async def paper(self, paper_id: str, *, fields: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Return details for a single paper (``GET /graph/v1/paper/{id}``).

        Args:
            paper_id: An S2 paper id or a supported external id
                (e.g. ``"DOI:10.18653/v1/2020.acl-main.447"``, ``"ARXIV:2106.15928"``,
                ``"CorpusId:215416146"``).
            fields: Comma-separated fields to return (defaults to the configured
                paper fields).
            **kwargs: Additional query parameters.

        Returns:
            The paper object (empty dict on 404 / non-200).

        Raises:
            SearchProviderError: On auth/transport faults.
        """
        params = {"fields": fields or self._default_fields, **kwargs}
        with self._span("paper", paper_id=paper_id):
            response = await self._request("GET", f"{GRAPH_PREFIX}/paper/{paper_id}", params=params)
        if response.status_code != 200:
            return {}
        payload = self._decode(response)
        return payload if isinstance(payload, dict) else {}

    async def paper_batch(
        self, paper_ids: list[str], *, fields: str | None = None
    ) -> WebSearchResult:
        """Look up many papers at once (``POST /graph/v1/paper/batch``).

        The single most rate-limit-friendly way to enrich a set of known paper
        ids (up to 500 ids per call).

        Args:
            paper_ids: Paper ids or supported external ids.
            fields: Comma-separated fields (defaults to the configured paper
                fields).

        Returns:
            A :class:`WebSearchResult` whose ``items`` are the papers, in input
            order (``raw["data"]`` holds the full array; missing ids appear as
            ``null`` in the raw array and are skipped in ``items``).

        Raises:
            SearchProviderError: If ``paper_ids`` is empty, or on auth/transport
                faults.
        """
        if not paper_ids:
            raise SearchProviderError(self.get_name(), "paper_batch requires a non-empty id list.")
        trigger_at = datetime.now(timezone.utc)
        with self._span("paper_batch", n=len(paper_ids)):
            response = await self._request(
                "POST",
                f"{GRAPH_PREFIX}/paper/batch",
                params={"fields": fields or self._default_fields},
                json_body={"ids": list(paper_ids)},
            )
        fetched_at = datetime.now(timezone.utc)
        payload = self._decode(response) if response.status_code == 200 else None
        return self._papers_result(
            payload,
            query="batch",
            engine_label="semanticscholar:paper_batch",
            item_kind="papers",
            items_key="data",
            trigger_at=trigger_at,
            fetched_at=fetched_at,
            response=response,
        )

    async def paper_authors(
        self, paper_id: str, *, fields: str | None = None, limit: int = 100, offset: int = 0
    ) -> WebSearchResult:
        """Return a paper's authors (``GET /graph/v1/paper/{id}/authors``)."""
        return await self._graph_list(
            f"{GRAPH_PREFIX}/paper/{paper_id}/authors",
            fields=fields or self._default_author_fields,
            limit=limit,
            offset=offset,
            item_kind="authors",
            engine_label="semanticscholar:paper_authors",
            query=paper_id,
        )

    async def paper_citations(
        self, paper_id: str, *, fields: str | None = None, limit: int = 100, offset: int = 0
    ) -> WebSearchResult:
        """Return papers that cite this paper (``GET /graph/v1/paper/{id}/citations``)."""
        return await self._graph_list(
            f"{GRAPH_PREFIX}/paper/{paper_id}/citations",
            fields=fields or self._default_fields,
            limit=limit,
            offset=offset,
            item_kind="citations",
            engine_label="semanticscholar:citations",
            query=paper_id,
        )

    async def paper_references(
        self, paper_id: str, *, fields: str | None = None, limit: int = 100, offset: int = 0
    ) -> WebSearchResult:
        """Return papers referenced by this paper (``GET /graph/v1/paper/{id}/references``)."""
        return await self._graph_list(
            f"{GRAPH_PREFIX}/paper/{paper_id}/references",
            fields=fields or self._default_fields,
            limit=limit,
            offset=offset,
            item_kind="references",
            engine_label="semanticscholar:references",
            query=paper_id,
        )

    async def paper_match(self, query: str, *, fields: str | None = None, **kwargs: Any) -> WebSearchResult:
        """Return the single best title match (``GET /graph/v1/paper/search/match``).

        Useful for resolving a free-text citation / title to a canonical paper.
        Returns ``success=False`` on a 404 "no match".
        """
        return await self.web_search(query, search_type="match", fields=fields, **kwargs)

    async def autocomplete(self, query: str) -> list[dict[str, Any]]:
        """Suggest paper query completions (``GET /graph/v1/paper/autocomplete``).

        Args:
            query: Partial query string (truncated to 100 chars by S2).

        Returns:
            A list of ``{"id", "title", "authorsYear"}`` match dicts (empty on
            non-200).
        """
        with self._span("autocomplete"):
            response = await self._request(
                "GET", f"{GRAPH_PREFIX}/paper/autocomplete", params={"query": query}
            )
        if response.status_code != 200:
            return []
        payload = self._decode(response)
        matches = payload.get("matches") if isinstance(payload, dict) else None
        return matches if isinstance(matches, list) else []

    async def snippet_search(
        self, query: str, *, fields: str | None = None, limit: int = 10, **kwargs: Any
    ) -> WebSearchResult:
        """Search for text snippets/passages (``GET /graph/v1/snippet/search``).

        Returns short text passages (title/abstract/body) relevant to the query
        — ideal for retrieval-augmented generation. Each ``item.description`` is
        the snippet text; ``raw`` holds the full payload (scores, offsets,
        annotations, paper metadata).
        """
        extra = dict(kwargs)
        if fields:
            extra["fields"] = fields
        return await self.web_search(query, count=limit, search_type="snippet", **extra)

    # ------------------------------------------------------------------
    # Authors (provider-specific)
    # ------------------------------------------------------------------
    async def author(self, author_id: str, *, fields: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Return details for a single author (``GET /graph/v1/author/{id}``)."""
        params = {"fields": fields or self._default_author_fields, **kwargs}
        with self._span("author", author_id=author_id):
            response = await self._request("GET", f"{GRAPH_PREFIX}/author/{author_id}", params=params)
        if response.status_code != 200:
            return {}
        payload = self._decode(response)
        return payload if isinstance(payload, dict) else {}

    async def author_batch(
        self, author_ids: list[str], *, fields: str | None = None
    ) -> WebSearchResult:
        """Look up many authors at once (``POST /graph/v1/author/batch``)."""
        if not author_ids:
            raise SearchProviderError(self.get_name(), "author_batch requires a non-empty id list.")
        trigger_at = datetime.now(timezone.utc)
        with self._span("author_batch", n=len(author_ids)):
            response = await self._request(
                "POST",
                f"{GRAPH_PREFIX}/author/batch",
                params={"fields": fields or self._default_author_fields},
                json_body={"ids": list(author_ids)},
            )
        fetched_at = datetime.now(timezone.utc)
        payload = self._decode(response) if response.status_code == 200 else None
        return self._papers_result(
            payload,
            query="batch",
            engine_label="semanticscholar:author_batch",
            item_kind="authors",
            items_key="data",
            trigger_at=trigger_at,
            fetched_at=fetched_at,
            response=response,
        )

    async def author_papers(
        self, author_id: str, *, fields: str | None = None, limit: int = 100, offset: int = 0
    ) -> WebSearchResult:
        """Return an author's papers (``GET /graph/v1/author/{id}/papers``)."""
        return await self._graph_list(
            f"{GRAPH_PREFIX}/author/{author_id}/papers",
            fields=fields or self._default_fields,
            limit=limit,
            offset=offset,
            item_kind="papers",
            engine_label="semanticscholar:author_papers",
            query=author_id,
        )

    async def author_search(
        self, query: str, *, fields: str | None = None, limit: int = 100, offset: int = 0
    ) -> WebSearchResult:
        """Search for authors by name (``GET /graph/v1/author/search``)."""
        return await self._graph_list(
            f"{GRAPH_PREFIX}/author/search",
            fields=fields or self._default_author_fields,
            limit=self._clamp_limit(limit, "author_search"),
            offset=offset,
            item_kind="authors",
            engine_label="semanticscholar:author_search",
            query=query,
            extra_params={"query": query},
        )

    async def _graph_list(
        self,
        path: str,
        *,
        fields: str | None,
        limit: int | None,
        offset: int,
        item_kind: str,
        engine_label: str,
        query: str,
        extra_params: dict[str, Any] | None = None,
    ) -> WebSearchResult:
        """Shared helper for paginated Graph list endpoints returning ``{data}``."""
        params: dict[str, Any] = {"fields": fields, "offset": offset}
        if limit is not None:
            params["limit"] = limit
        if extra_params:
            params.update(extra_params)
        trigger_at = datetime.now(timezone.utc)
        with self._span("graph_list", path=path):
            response = await self._request("GET", path, params=params)
        fetched_at = datetime.now(timezone.utc)
        payload = self._decode(response) if response.status_code == 200 else None
        return self._papers_result(
            payload,
            query=query,
            engine_label=engine_label,
            item_kind=item_kind,
            items_key="data",
            trigger_at=trigger_at,
            fetched_at=fetched_at,
            response=response,
        )

    # ------------------------------------------------------------------
    # Recommendations (provider-specific)
    # ------------------------------------------------------------------
    async def recommend_papers(
        self,
        paper_id: str,
        *,
        limit: int = 100,
        fields: str | None = None,
        pool: str = "recent",
    ) -> WebSearchResult:
        """Recommend papers similar to a seed paper.

        Wraps ``GET /recommendations/v1/papers/forpaper/{paper_id}``.

        Args:
            paper_id: The seed paper id (or supported external id).
            limit: Max recommendations (clamped to 500).
            fields: Comma-separated paper fields (defaults to configured fields).
            pool: Candidate pool — ``"recent"`` (default) or ``"all-cs"``.

        Returns:
            A :class:`WebSearchResult` whose ``items`` are the recommended papers
            (``raw["recommendedPapers"]`` holds the full array).

        Raises:
            SearchProviderError: On auth/transport faults.
        """
        params = {
            "limit": self._clamp_limit(limit, "recommendations"),
            "fields": fields or self._default_fields,
            "from": pool,
        }
        trigger_at = datetime.now(timezone.utc)
        with self._span("recommend_papers", paper_id=paper_id):
            response = await self._request(
                "GET", f"{RECOMMENDATIONS_PREFIX}/papers/forpaper/{paper_id}", params=params
            )
        fetched_at = datetime.now(timezone.utc)
        payload = self._decode(response) if response.status_code == 200 else None
        return self._papers_result(
            payload,
            query=paper_id,
            engine_label="semanticscholar:recommendations",
            item_kind="papers",
            items_key="recommendedPapers",
            trigger_at=trigger_at,
            fetched_at=fetched_at,
            response=response,
        )

    async def recommend_from_examples(
        self,
        positive_paper_ids: list[str],
        negative_paper_ids: list[str] | None = None,
        *,
        limit: int = 100,
        fields: str | None = None,
    ) -> WebSearchResult:
        """Recommend papers from positive/negative example lists.

        Wraps ``POST /recommendations/v1/papers``.

        Args:
            positive_paper_ids: Papers to recommend *more like*.
            negative_paper_ids: Papers to recommend *less like* (optional).
            limit: Max recommendations (clamped to 500).
            fields: Comma-separated paper fields (defaults to configured fields).

        Returns:
            A :class:`WebSearchResult` whose ``items`` are the recommended papers.

        Raises:
            SearchProviderError: If ``positive_paper_ids`` is empty, or on
                auth/transport faults.
        """
        if not positive_paper_ids:
            raise SearchProviderError(
                self.get_name(), "recommend_from_examples requires at least one positive id."
            )
        body = {
            "positivePaperIds": list(positive_paper_ids),
            "negativePaperIds": list(negative_paper_ids or []),
        }
        params = {
            "limit": self._clamp_limit(limit, "recommendations"),
            "fields": fields or self._default_fields,
        }
        trigger_at = datetime.now(timezone.utc)
        with self._span("recommend_from_examples"):
            response = await self._request(
                "POST", f"{RECOMMENDATIONS_PREFIX}/papers/", params=params, json_body=body
            )
        fetched_at = datetime.now(timezone.utc)
        payload = self._decode(response) if response.status_code == 200 else None
        return self._papers_result(
            payload,
            query="recommendations",
            engine_label="semanticscholar:recommendations",
            item_kind="papers",
            items_key="recommendedPapers",
            trigger_at=trigger_at,
            fetched_at=fetched_at,
            response=response,
        )

    # ------------------------------------------------------------------
    # Datasets API (provider-specific; bulk-corpus download links)
    # ------------------------------------------------------------------
    async def list_releases(self) -> list[str]:
        """List available dataset releases (``GET /datasets/v1/release/``).

        Returns:
            A list of release-id date stamps (e.g. ``["2023-08-01", ...]``);
            empty on non-200.
        """
        with self._span("list_releases"):
            response = await self._request("GET", f"{DATASETS_PREFIX}/release/")
        if response.status_code != 200:
            return []
        payload = self._decode(response)
        return payload if isinstance(payload, list) else []

    async def get_release(self, release_id: str = "latest") -> dict[str, Any]:
        """Return metadata for a release (``GET /datasets/v1/release/{id}``).

        Args:
            release_id: A release date stamp or ``"latest"``.

        Returns:
            The release metadata (``release_id``, ``README``, ``datasets``);
            empty dict on non-200.
        """
        with self._span("get_release", release_id=release_id):
            response = await self._request("GET", f"{DATASETS_PREFIX}/release/{release_id}")
        if response.status_code != 200:
            return {}
        payload = self._decode(response)
        return payload if isinstance(payload, dict) else {}

    async def get_dataset(self, dataset_name: str, *, release_id: str = "latest") -> dict[str, Any]:
        """Return a dataset's download links (``GET /release/{id}/dataset/{name}``).

        Args:
            dataset_name: Dataset name (e.g. ``"papers"``, ``"abstracts"``,
                ``"citations"``, ``"authors"``, ``"tldrs"``, ``"embeddings"``).
            release_id: A release date stamp or ``"latest"``.

        Returns:
            Dataset metadata including temporary pre-signed ``files`` URLs;
            empty dict on non-200.
        """
        with self._span("get_dataset", dataset_name=dataset_name, release_id=release_id):
            response = await self._request(
                "GET", f"{DATASETS_PREFIX}/release/{release_id}/dataset/{dataset_name}"
            )
        if response.status_code != 200:
            return {}
        payload = self._decode(response)
        return payload if isinstance(payload, dict) else {}

    async def get_dataset_diffs(
        self, start_release_id: str, end_release_id: str, dataset_name: str
    ) -> dict[str, Any]:
        """Return incremental update diffs between two releases for a dataset.

        Wraps ``GET /datasets/v1/diffs/{start}/to/{end}/{dataset_name}``.

        Args:
            start_release_id: The release currently held by the client.
            end_release_id: The target release (or ``"latest"``).
            dataset_name: Dataset name.

        Returns:
            A diff list with ``update_files`` / ``delete_files`` URLs per
            sequential release; empty dict on non-200.
        """
        path = f"{DATASETS_PREFIX}/diffs/{start_release_id}/to/{end_release_id}/{dataset_name}"
        with self._span("get_dataset_diffs"):
            response = await self._request("GET", path)
        if response.status_code != 200:
            return {}
        payload = self._decode(response)
        return payload if isinstance(payload, dict) else {}

    # ------------------------------------------------------------------
    # health / lifecycle
    # ------------------------------------------------------------------
    async def health_check(self) -> bool:
        """Verify connectivity via a tiny autocomplete request.

        S2 has no free balance/account endpoint, so this issues a minimal
        ``GET /graph/v1/paper/autocomplete`` (cheap; counts toward the rate
        limit but returns quickly). Returns ``True`` on HTTP 200, ``False`` on
        auth failure or an unreachable host; never raises.
        """
        try:
            response = await self._request(
                "GET", f"{GRAPH_PREFIX}/paper/autocomplete", params={"query": "covid"}
            )
        except SearchProviderError:
            return False
        return response.status_code == 200

    async def close(self) -> None:
        """Close the underlying :class:`httpx.AsyncClient` (idempotent)."""
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Error closing Semantic Scholar httpx client: %s", exc)
            self._client = None
