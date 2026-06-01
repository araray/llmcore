# tests/search/test_serpapi_provider.py
"""Tests for :mod:`llmcore.search.providers.serpapi_provider`.

All HTTP is intercepted with ``respx`` (no network). We assert the exact
endpoint (``GET /search``), that the ``api_key`` is sent as a **query
parameter** (never a header), the cross-provider -> SerpApi parameter mapping
(``q``/``num``/``gl``/``hl``/``device``/``engine``), engine-aware response
normalization (organic vs. vertical result arrays), the async submit + Search
Archive polling pipeline, client-side batch fan-out, the free Account-API
health check, and the Locations API — using fixtures modeled on real SerpApi
responses.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from confy.loader import Config

from llmcore.exceptions import ConfigError, SearchProviderError
from llmcore.search.manager import SearchProviderManager
from llmcore.search.models import WebSearchResult
from llmcore.search.providers.serpapi_provider import (
    DEFAULT_BASE_URL,
    SerpApiSearchProvider,
    _coerce_total_results,
    _normalize_serpapi,
)

BASE = DEFAULT_BASE_URL

# A representative slice of a real SerpApi Google ``/search`` response.
SERPAPI_GOOGLE_RESPONSE = {
    "search_metadata": {"id": "abc123", "status": "Success"},
    "search_parameters": {"engine": "google", "q": "apple inc", "gl": "us", "hl": "en"},
    "search_information": {"query_displayed": "apple inc", "total_results": 3140000000},
    "knowledge_graph": {"title": "Apple Inc.", "type": "Technology company"},
    "organic_results": [
        {
            "position": 1,
            "title": "Apple",
            "link": "https://www.apple.com/",
            "displayed_link": "https://www.apple.com",
            "snippet": "Discover the innovative world of Apple ...",
        },
        {
            "position": 2,
            "title": "Apple Inc. - Wikipedia",
            "link": "https://en.wikipedia.org/wiki/Apple_Inc.",
            "displayed_link": "https://en.wikipedia.org > wiki > Apple_Inc.",
            "snippet": "Apple Inc. is an American multinational technology company ...",
        },
    ],
    "related_questions": [{"question": "Who owns Apple?"}],
    "serpapi_pagination": {"current": 1, "next": "https://serpapi.com/search.json?..."},
}


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Patch asyncio.sleep so retry/backoff/poll loops run instantly."""

    async def _instant(_seconds):
        return None

    monkeypatch.setattr("asyncio.sleep", _instant)
    yield


@pytest.fixture
def provider():
    """A SerpApiSearchProvider with an explicit key and fast retries/polls."""
    return SerpApiSearchProvider(
        {
            "api_key": "k_test_1234567890",
            "_instance_name": "serpapi",
            "timeout": 5,
            "max_retries": 3,
            "poll_interval": 0,
            "poll_timeout": 5,
        }
    )


# ---------------------------------------------------------------------------
# Construction / identity / config
# ---------------------------------------------------------------------------
def test_identity_and_capabilities(provider):
    assert provider.get_name() == "serpapi"
    assert provider.get_capabilities() == {"web_search", "batch_search"}
    assert provider.supports("web_search") is True
    assert provider.supports("batch_search") is True
    assert provider.supports("scrape") is False
    assert provider.supports("discover") is False
    assert provider.supports("dataset_search") is False
    assert provider._base_url == BASE
    assert provider._default_engine == "google"
    assert provider._default_output == "json"


def test_key_from_env(monkeypatch):
    monkeypatch.delenv("SERPAPI_KEY", raising=False)
    monkeypatch.setenv("SERPAPI_API_KEY", "k_env_999")
    p = SerpApiSearchProvider({"_instance_name": "serpapi"})
    assert p._api_key == "k_env_999"


def test_key_from_secondary_env(monkeypatch):
    """The SDK-convention SERPAPI_KEY is honored when SERPAPI_API_KEY is absent."""
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    monkeypatch.setenv("SERPAPI_KEY", "k_secondary_123")
    p = SerpApiSearchProvider({"_instance_name": "serpapi"})
    assert p._api_key == "k_secondary_123"


def test_custom_env_var(monkeypatch):
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    monkeypatch.delenv("SERPAPI_KEY", raising=False)
    monkeypatch.setenv("MY_SERPAPI", "k_custom_777")
    p = SerpApiSearchProvider({"api_key_env_var": "MY_SERPAPI", "_instance_name": "serpapi"})
    assert p._api_key == "k_custom_777"


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    monkeypatch.delenv("SERPAPI_KEY", raising=False)
    monkeypatch.delenv("SERP_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="API key not found"):
        SerpApiSearchProvider({"_instance_name": "serpapi"})


def test_invalid_default_output_falls_back():
    p = SerpApiSearchProvider({"api_key": "k", "default_output": "xml"})
    assert p._default_output == "json"


def test_config_defaults_applied():
    p = SerpApiSearchProvider(
        {
            "api_key": "k",
            "default_engine": "bing",
            "no_cache": True,
            "zero_trace": True,
            "json_restrictor": "organic_results[].{title,link}",
            "max_concurrency": 9,
        }
    )
    assert p._default_engine == "bing"
    assert p._default_no_cache is True
    assert p._default_zero_trace is True
    assert p._default_json_restrictor == "organic_results[].{title,link}"
    assert p._max_concurrency == 9


# ---------------------------------------------------------------------------
# web_search (sync)
# ---------------------------------------------------------------------------
@respx.mock
async def test_web_search_basic_param_mapping_and_auth(provider):
    route = respx.get(f"{BASE}/search").mock(
        return_value=httpx.Response(200, json=SERPAPI_GOOGLE_RESPONSE)
    )
    res = await provider.web_search("apple inc", count=10, country="us", language="en")

    assert isinstance(res, WebSearchResult)
    assert res.success is True
    assert res.provider == "serpapi"
    assert res.engine == "serpapi:google"
    assert res.total_results == 3140000000
    assert [i.title for i in res.items] == ["Apple", "Apple Inc. - Wikipedia"]
    assert res.items[0].url == "https://www.apple.com/"
    assert res.items[0].position == 1
    assert res.items[0].description.startswith("Discover")
    # Full payload preserved for power users.
    assert res.raw["knowledge_graph"]["title"] == "Apple Inc."
    assert res.raw["related_questions"][0]["question"] == "Who owns Apple?"

    req = route.calls.last.request
    assert req.method == "GET"
    assert req.url.path == "/search"
    # api_key is a QUERY PARAMETER (never a header).
    assert "x-api-key" not in {k.lower() for k in req.headers}
    assert "authorization" not in {k.lower() for k in req.headers}
    params = dict(req.url.params)
    assert params["api_key"] == "k_test_1234567890"
    assert params["engine"] == "google"
    assert params["q"] == "apple inc"
    assert params["num"] == "10"
    assert params["gl"] == "us"
    assert params["hl"] == "en"
    assert params["output"] == "json"
    # desktop is the default device -> not sent.
    assert "device" not in params


@respx.mock
async def test_web_search_device_and_engine_and_passthrough(provider):
    route = respx.get(f"{BASE}/search").mock(
        return_value=httpx.Response(200, json=SERPAPI_GOOGLE_RESPONSE)
    )
    await provider.web_search(
        "q",
        engine="google",
        device="mobile",
        location="Austin, TX",
        tbm="nws",
        start=20,
        safe="active",
    )
    params = dict(route.calls.last.request.url.params)
    assert params["device"] == "mobile"
    assert params["location"] == "Austin, TX"
    assert params["tbm"] == "nws"
    assert params["start"] == "20"
    assert params["safe"] == "active"


@respx.mock
async def test_web_search_engine_specific_query_param_yahoo(provider):
    """Yahoo uses ``p`` (not ``q``) as the free-text query parameter."""
    route = respx.get(f"{BASE}/search").mock(
        return_value=httpx.Response(200, json={"organic_results": []})
    )
    await provider.web_search("coffee", engine="yahoo")
    params = dict(route.calls.last.request.url.params)
    assert params["engine"] == "yahoo"
    assert params["p"] == "coffee"
    assert "q" not in params


@respx.mock
async def test_web_search_vertical_news_normalization(provider):
    news_resp = {
        "search_information": {"total_results": 42},
        "news_results": [
            {"position": 1, "title": "Apple announces", "link": "https://news/a", "snippet": "s"},
            {"position": 2, "title": "Apple earnings", "link": "https://news/b", "snippet": "s2"},
        ],
    }
    respx.get(f"{BASE}/search").mock(return_value=httpx.Response(200, json=news_resp))
    res = await provider.web_search("apple", engine="google_news")
    assert res.engine == "serpapi:google_news"
    assert [i.title for i in res.items] == ["Apple announces", "Apple earnings"]
    assert res.items[1].url == "https://news/b"
    assert res.total_results == 42


@respx.mock
async def test_web_search_bool_kwargs_rendered_as_strings(provider):
    route = respx.get(f"{BASE}/search").mock(
        return_value=httpx.Response(200, json=SERPAPI_GOOGLE_RESPONSE)
    )
    await provider.web_search("q", no_cache=True, nfpr=False)
    params = dict(route.calls.last.request.url.params)
    assert params["no_cache"] == "true"
    assert params["nfpr"] == "false"


@respx.mock
async def test_web_search_default_no_cache_applied():
    """A configured no_cache default is applied unless overridden."""
    p = SerpApiSearchProvider({"api_key": "k", "no_cache": True, "timeout": 5})
    route = respx.get(f"{BASE}/search").mock(
        return_value=httpx.Response(200, json=SERPAPI_GOOGLE_RESPONSE)
    )
    await p.web_search("q")
    assert dict(route.calls.last.request.url.params)["no_cache"] == "true"
    # Explicit override wins.
    await p.web_search("q", no_cache=False)
    assert dict(route.calls.last.request.url.params)["no_cache"] == "false"
    await p.close()


@respx.mock
async def test_web_search_html_output_preserves_raw(provider):
    respx.get(f"{BASE}/search").mock(
        return_value=httpx.Response(200, text="<html><body>SERP</body></html>")
    )
    res = await provider.web_search("q", output="html")
    assert res.success is True
    assert res.items == []
    assert res.raw["raw_html"].startswith("<html>")


@respx.mock
async def test_web_search_in_body_error_is_soft_failure(provider):
    respx.get(f"{BASE}/search").mock(
        return_value=httpx.Response(200, json={"error": "Google hasn't returned any results."})
    )
    res = await provider.web_search("asdkfjhaskdjfh")
    assert res.success is False
    assert "hasn't returned any results" in (res.error or "")


@respx.mock
async def test_web_search_http_error_returns_error_result(provider):
    respx.get(f"{BASE}/search").mock(
        return_value=httpx.Response(400, json={"error": "Invalid API parameters"})
    )
    res = await provider.web_search("q")
    assert res.success is False
    assert "Invalid API parameters" in (res.error or "")


# ---------------------------------------------------------------------------
# Low-level faithful search()
# ---------------------------------------------------------------------------
@respx.mock
async def test_raw_search_passthrough(provider):
    route = respx.get(f"{BASE}/search").mock(
        return_value=httpx.Response(200, json=SERPAPI_GOOGLE_RESPONSE)
    )
    res = await provider.search({"engine": "google_scholar", "q": "graph neural networks", "num": 20})
    params = dict(route.calls.last.request.url.params)
    assert params["engine"] == "google_scholar"
    assert params["q"] == "graph neural networks"
    assert params["num"] == "20"
    assert res.engine == "serpapi:google_scholar"
    assert res.success is True


# ---------------------------------------------------------------------------
# Async pipeline (async=true + Search Archive polling)
# ---------------------------------------------------------------------------
@respx.mock
async def test_async_search_submit_then_poll(provider):
    submit = {"search_metadata": {"id": "S123", "status": "Processing"}}
    processing = {"search_metadata": {"id": "S123", "status": "Processing"}}
    done = dict(SERPAPI_GOOGLE_RESPONSE)
    done["search_metadata"] = {"id": "S123", "status": "Success"}

    search_route = respx.get(f"{BASE}/search").mock(
        return_value=httpx.Response(200, json=submit)
    )
    archive_route = respx.get(f"{BASE}/searches/S123").mock(
        side_effect=[
            httpx.Response(200, json=processing),
            httpx.Response(200, json=done),
        ]
    )

    res = await provider.web_search("apple inc", mode="async")
    assert res.success is True
    assert res.items[0].title == "Apple"
    # The submit carried async=true ...
    submit_params = dict(search_route.calls.last.request.url.params)
    assert submit_params["async"] == "true"
    # ... and we polled the archive endpoint at least twice.
    assert archive_route.call_count == 2


@respx.mock
async def test_async_search_drops_no_cache():
    """async + no_cache are mutually exclusive; no_cache is dropped on submit."""
    p = SerpApiSearchProvider({"api_key": "k", "no_cache": True, "timeout": 5, "poll_interval": 0})
    submit = {"search_metadata": {"id": "S9", "status": "Processing"}}
    done = dict(SERPAPI_GOOGLE_RESPONSE)
    done["search_metadata"] = {"id": "S9", "status": "Success"}
    search_route = respx.get(f"{BASE}/search").mock(return_value=httpx.Response(200, json=submit))
    respx.get(f"{BASE}/searches/S9").mock(return_value=httpx.Response(200, json=done))

    await p.web_search("q", mode="async")
    params = dict(search_route.calls.last.request.url.params)
    assert params["async"] == "true"
    assert "no_cache" not in params
    await p.close()


@respx.mock
async def test_async_search_status_error_is_soft_failure(provider):
    submit = {"search_metadata": {"id": "E1", "status": "Processing"}}
    errored = {"search_metadata": {"id": "E1", "status": "Error"}, "error": "Engine failed"}
    respx.get(f"{BASE}/search").mock(return_value=httpx.Response(200, json=submit))
    respx.get(f"{BASE}/searches/E1").mock(return_value=httpx.Response(200, json=errored))
    res = await provider.web_search("q", mode="async")
    assert res.success is False
    assert "Engine failed" in (res.error or "")


@respx.mock
async def test_async_search_missing_id_raises(provider):
    respx.get(f"{BASE}/search").mock(
        return_value=httpx.Response(200, json={"search_metadata": {"status": "Processing"}})
    )
    with pytest.raises(SearchProviderError, match=r"did not return a search_metadata\.id"):
        await provider.web_search("q", mode="async")


@respx.mock
async def test_async_poll_timeout_raises(provider):
    submit = {"search_metadata": {"id": "T1", "status": "Processing"}}
    processing = {"search_metadata": {"id": "T1", "status": "Processing"}}
    respx.get(f"{BASE}/search").mock(return_value=httpx.Response(200, json=submit))
    respx.get(f"{BASE}/searches/T1").mock(return_value=httpx.Response(200, json=processing))
    # poll_timeout=5, poll_interval=0 -> loop reaches the timeout guard quickly.
    with pytest.raises(SearchProviderError, match="timed out"):
        await provider.web_search("q", mode="async")


# ---------------------------------------------------------------------------
# batch_search (client-side fan-out)
# ---------------------------------------------------------------------------
@respx.mock
async def test_batch_search_strings(provider):
    respx.get(f"{BASE}/search").mock(return_value=httpx.Response(200, json=SERPAPI_GOOGLE_RESPONSE))
    results = await provider.batch_search(["apple inc", "tesla inc"], count=5, country="us")
    assert isinstance(results, list) and len(results) == 2
    assert results[0].query == "apple inc"
    assert results[1].query == "tesla inc"
    assert all(r.success for r in results)


@respx.mock
async def test_batch_search_dicts_with_engine(provider):
    route = respx.get(f"{BASE}/search").mock(
        return_value=httpx.Response(200, json={"news_results": [{"title": "N", "link": "u"}]})
    )
    results = await provider.batch_search(
        [
            {"engine": "google_news", "q": "ai"},
            {"engine": "google_news", "q": "robotics"},
        ]
    )
    assert len(results) == 2
    assert all(r.engine == "serpapi:google_news" for r in results)
    # Last request used the google_news engine.
    assert dict(route.calls.last.request.url.params)["engine"] == "google_news"


@respx.mock
async def test_batch_search_search_type_as_engine(provider):
    route = respx.get(f"{BASE}/search").mock(
        return_value=httpx.Response(200, json={"images_results": [{"title": "I", "link": "u"}]})
    )
    results = await provider.batch_search(["cats"], search_type="google_images")
    assert results[0].engine == "serpapi:google_images"
    assert dict(route.calls.last.request.url.params)["engine"] == "google_images"


async def test_batch_search_dict_without_query_raises(provider):
    with pytest.raises(SearchProviderError, match="must include a"):
        await provider.batch_search([{"engine": "google", "tbm": "nws"}])


async def test_batch_search_empty_raises(provider):
    with pytest.raises(SearchProviderError, match="non-empty list"):
        await provider.batch_search([])


async def test_batch_search_bad_item_type_raises(provider):
    with pytest.raises(SearchProviderError, match="Unsupported batch query item"):
        await provider.batch_search([123])


# ---------------------------------------------------------------------------
# Search Archive / Account / Locations (provider-specific)
# ---------------------------------------------------------------------------
@respx.mock
async def test_search_archive(provider):
    archived = dict(SERPAPI_GOOGLE_RESPONSE)
    route = respx.get(f"{BASE}/searches/abc123").mock(
        return_value=httpx.Response(200, json=archived)
    )
    res = await provider.search_archive("abc123")
    assert res.success is True
    assert res.engine == "serpapi:google"  # inferred from search_parameters
    assert res.query == "apple inc"
    assert dict(route.calls.last.request.url.params)["api_key"] == "k_test_1234567890"


async def test_search_archive_requires_id(provider):
    with pytest.raises(SearchProviderError, match="non-empty search_id"):
        await provider.search_archive("")


@respx.mock
async def test_account(provider):
    account_resp = {
        "account_id": "5ac",
        "plan_name": "Developer Plan",
        "searches_per_month": 5000,
        "plan_searches_left": 4990,
        "total_searches_left": 4990,
        "this_month_usage": 10,
    }
    route = respx.get(f"{BASE}/account.json").mock(
        return_value=httpx.Response(200, json=account_resp)
    )
    acct = await provider.account()
    assert acct["plan_name"] == "Developer Plan"
    assert acct["total_searches_left"] == 4990
    assert dict(route.calls.last.request.url.params)["api_key"] == "k_test_1234567890"


@respx.mock
async def test_account_http_error_raises(provider):
    respx.get(f"{BASE}/account.json").mock(return_value=httpx.Response(500, text="boom"))
    with pytest.raises(SearchProviderError, match="account lookup failed"):
        await provider.account()


@respx.mock
async def test_locations(provider):
    locs = [
        {"id": "1", "name": "Austin", "canonical_name": "Austin,Texas,United States"},
        {"id": "2", "name": "Austin County", "canonical_name": "Austin County,Texas,..."},
    ]
    route = respx.get(f"{BASE}/locations.json").mock(return_value=httpx.Response(200, json=locs))
    out = await provider.locations(q="Austin", limit=2)
    assert isinstance(out, list) and len(out) == 2
    assert out[0]["canonical_name"].startswith("Austin")
    params = dict(route.calls.last.request.url.params)
    assert params["q"] == "Austin"
    assert params["limit"] == "2"
    # The locations endpoint does NOT require api_key, and the provider injects it
    # by default (harmless); just assert the query params we set are present.


# ---------------------------------------------------------------------------
# auth / retries / health / lifecycle
# ---------------------------------------------------------------------------
@respx.mock
async def test_auth_error_raises_with_status(provider):
    respx.get(f"{BASE}/search").mock(
        return_value=httpx.Response(401, json={"error": "Invalid API key"})
    )
    with pytest.raises(SearchProviderError) as exc:
        await provider.web_search("q")
    assert exc.value.status_code == 401


@respx.mock
async def test_retry_on_429_then_success(provider):
    respx.get(f"{BASE}/search").mock(
        side_effect=[
            httpx.Response(429, json={"error": "rate"}),
            httpx.Response(200, json=SERPAPI_GOOGLE_RESPONSE),
        ]
    )
    res = await provider.web_search("q")
    assert res.success is True
    assert res.items[0].title == "Apple"


@respx.mock
async def test_retry_on_5xx_exhausts_then_error_result(provider):
    respx.get(f"{BASE}/search").mock(return_value=httpx.Response(503, json={"error": "down"}))
    res = await provider.web_search("q")
    # 5xx is retried then returned (not raised) -> soft failure result.
    assert res.success is False
    assert "503" in (res.error or "")


@respx.mock
async def test_health_check_uses_free_account_endpoint(provider):
    route = respx.get(f"{BASE}/account.json").mock(
        return_value=httpx.Response(200, json={"plan_name": "X"})
    )
    assert await provider.health_check() is True
    # Health check must hit the (free) Account API, NOT /search.
    assert route.called
    assert route.calls.last.request.url.path == "/account.json"


@respx.mock
async def test_health_check_failure(provider):
    respx.get(f"{BASE}/account.json").mock(return_value=httpx.Response(401, text="nope"))
    assert await provider.health_check() is False


@respx.mock
async def test_close_idempotent(provider):
    respx.get(f"{BASE}/account.json").mock(return_value=httpx.Response(200, json={}))
    await provider.health_check()
    assert provider._client is not None
    await provider.close()
    assert provider._client is None
    await provider.close()  # second close is a no-op


async def test_web_search_non_string_query_raises(provider):
    with pytest.raises(SearchProviderError, match="must be a string"):
        await provider.web_search(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# manager wiring + pure helpers
# ---------------------------------------------------------------------------
def test_manager_loads_serpapi(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "k")
    cfg = Config(defaults={"search_providers": {"serpapi": {}}})
    m = SearchProviderManager(cfg)
    assert m.get_available_search_providers() == ["serpapi"]
    assert m.get_default_search_provider().get_name() == "serpapi"


def test_manager_serpapi_alias(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "k")
    cfg = Config(defaults={"search_providers": {"web": {"type": "serp_api"}}})
    m = SearchProviderManager(cfg)
    assert m.get_search_provider("web").get_name() == "serpapi"


def test_manager_skips_serpapi_without_key(monkeypatch):
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    monkeypatch.delenv("SERPAPI_KEY", raising=False)
    monkeypatch.delenv("SERP_API_KEY", raising=False)
    cfg = Config(defaults={"search_providers": {"serpapi": {}}})
    m = SearchProviderManager(cfg)
    assert m.get_available_search_providers() == []


def test_normalize_serpapi_falls_back_to_organic():
    # Unknown engine -> falls back to organic_results.
    data = {"organic_results": [{"title": "t", "link": "u", "snippet": "s", "position": 1}]}
    items, total = _normalize_serpapi(data, "some_new_engine")
    assert len(items) == 1 and items[0].url == "u"
    assert total is None


def test_normalize_serpapi_string_items():
    # Some autocomplete arrays are plain strings.
    items, _ = _normalize_serpapi({"suggestions": ["coffee near me", "coffee shop"]}, "google_autocomplete")
    assert [i.title for i in items] == ["coffee near me", "coffee shop"]


def test_normalize_serpapi_non_dict():
    assert _normalize_serpapi(None, "google") == ([], None)
    assert _normalize_serpapi("oops", "google") == ([], None)


def test_coerce_total_results_variants():
    assert _coerce_total_results(50) == 50
    assert _coerce_total_results("About 3,140,000 results") == 3140000
    assert _coerce_total_results(12.0) == 12
    assert _coerce_total_results(True) is None
    assert _coerce_total_results(None) is None
    assert _coerce_total_results("none") is None
