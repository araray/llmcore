# tests/search/test_serper_provider.py
"""Tests for :mod:`llmcore.search.providers.serper_provider`.

All HTTP is intercepted with ``respx`` (no network). We assert the exact
endpoints, request bodies (including Serper's array/batch payloads), the
``X-API-KEY`` auth header, the scrape host, and response normalization using a
fixture modeled on a real Serper ``/search`` response.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from confy.loader import Config

from llmcore.exceptions import ConfigError, SearchProviderError
from llmcore.search.manager import SearchProviderManager
from llmcore.search.models import ScrapeResult, WebSearchResult
from llmcore.search.providers.serper_provider import (
    DEFAULT_BASE_URL,
    DEFAULT_SCRAPE_URL,
    SerperSearchProvider,
    _normalize_serper,
)

BASE = DEFAULT_BASE_URL
SCRAPE = DEFAULT_SCRAPE_URL

# A representative slice of a real Serper /search response (Google-shaped).
SERPER_SEARCH_RESPONSE = {
    "searchParameters": {"q": "apple inc", "gl": "us", "hl": "en", "type": "search"},
    "knowledgeGraph": {"title": "Apple", "type": "Technology company"},
    "organic": [
        {
            "title": "Apple",
            "link": "https://www.apple.com/",
            "snippet": "Discover the innovative world of Apple ...",
            "position": 1,
        },
        {
            "title": "Apple Inc. - Wikipedia",
            "link": "https://en.wikipedia.org/wiki/Apple_Inc.",
            "snippet": "Apple Inc. is an American multinational technology company ...",
            "position": 2,
        },
    ],
    "peopleAlsoAsk": [{"question": "What does Apple Inc mean?", "link": "https://example"}],
    "relatedSearches": [{"query": "Apple Inc competitors"}],
    "credits": 1,
}


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Patch asyncio.sleep so retry/backoff loops run instantly."""

    async def _instant(_seconds):
        return None

    monkeypatch.setattr("asyncio.sleep", _instant)
    yield


@pytest.fixture
def provider():
    """A SerperSearchProvider with an explicit key and fast retries."""
    return SerperSearchProvider(
        {"api_key": "k_test_1234567890", "_instance_name": "serper", "timeout": 5, "max_retries": 3}
    )


# ---------------------------------------------------------------------------
# Construction / identity / config
# ---------------------------------------------------------------------------
def test_identity_and_capabilities(provider):
    assert provider.get_name() == "serper"
    assert provider.get_capabilities() == {"web_search", "batch_search", "scrape"}
    assert provider.supports("batch_search") is True
    assert provider.supports("discover") is False
    assert provider.supports("dataset_search") is False
    assert provider._base_url == BASE


def test_token_from_env(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "k_env_999")
    p = SerperSearchProvider({"_instance_name": "serper"})
    assert p._api_key == "k_env_999"


def test_custom_env_var(monkeypatch):
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    monkeypatch.setenv("MY_SERPER", "k_custom_777")
    p = SerperSearchProvider({"api_key_env_var": "MY_SERPER", "_instance_name": "serper"})
    assert p._api_key == "k_custom_777"


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="API key not found"):
        SerperSearchProvider({"_instance_name": "serper"})


def test_invalid_default_search_type_falls_back():
    p = SerperSearchProvider({"api_key": "k", "default_search_type": "weird"})
    assert p._default_search_type == "search"


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------
@respx.mock
async def test_web_search_basic(provider):
    route = respx.post(f"{BASE}/search").mock(
        return_value=httpx.Response(200, json=SERPER_SEARCH_RESPONSE)
    )
    res = await provider.web_search("apple inc", count=10, country="us", language="en")

    assert isinstance(res, WebSearchResult)
    assert res.success is True
    assert res.provider == "serper"
    assert res.engine == "serper:search"
    assert res.cost == 1  # from "credits"
    assert [i.title for i in res.items] == ["Apple", "Apple Inc. - Wikipedia"]
    assert res.items[0].url == "https://www.apple.com/"
    assert res.items[0].position == 1
    # Full payload preserved (SERP features available to power users).
    assert res.raw["knowledgeGraph"]["title"] == "Apple"
    assert res.raw["peopleAlsoAsk"][0]["question"].startswith("What does")

    req = route.calls.last.request
    assert req.method == "POST"
    assert str(req.url) == f"{BASE}/search"
    assert req.headers["x-api-key"] == "k_test_1234567890"
    assert req.headers["content-type"] == "application/json"
    body = json.loads(req.content)
    assert body == {"q": "apple inc", "num": 10, "gl": "us", "hl": "en"}


@respx.mock
async def test_web_search_ignores_engine_device_mode(provider):
    """Cross-provider params must not leak into the Serper body."""
    route = respx.post(f"{BASE}/search").mock(
        return_value=httpx.Response(200, json=SERPER_SEARCH_RESPONSE)
    )
    await provider.web_search("q", engine="bing", device="mobile", mode="async")
    body = json.loads(route.calls.last.request.content)
    assert "engine" not in body and "device" not in body and "mode" not in body


@respx.mock
async def test_web_search_time_range_and_passthrough(provider):
    route = respx.post(f"{BASE}/search").mock(
        return_value=httpx.Response(200, json=SERPER_SEARCH_RESPONSE)
    )
    await provider.web_search("q", time_range="d", page=2, autocorrect=False, location="Austin, TX")
    body = json.loads(route.calls.last.request.content)
    assert body["tbs"] == "qdr:d"
    assert body["page"] == 2
    assert body["autocorrect"] is False
    assert body["location"] == "Austin, TX"


@respx.mock
async def test_web_search_news_vertical(provider):
    news_resp = {
        "news": [
            {
                "title": "Apple announces",
                "link": "https://news.example/a",
                "snippet": "s",
                "position": 1,
            },
            {
                "title": "Apple earnings",
                "link": "https://news.example/b",
                "snippet": "s2",
                "position": 2,
            },
        ]
    }
    route = respx.post(f"{BASE}/news").mock(return_value=httpx.Response(200, json=news_resp))
    res = await provider.web_search("apple", search_type="news")
    assert route.called
    assert res.engine == "serper:news"
    assert [i.title for i in res.items] == ["Apple announces", "Apple earnings"]


@respx.mock
async def test_web_search_bad_type_raises(provider):
    with pytest.raises(SearchProviderError, match="Unsupported search_type"):
        await provider.web_search("q", search_type="altavista")


@respx.mock
async def test_web_search_http_error_returns_error_result(provider):
    respx.post(f"{BASE}/search").mock(
        return_value=httpx.Response(400, json={"message": "Bad request", "statusCode": 400})
    )
    res = await provider.web_search("q")
    assert res.success is False
    assert "Bad request" in (res.error or "")


# ---------------------------------------------------------------------------
# batch_search (array payload)
# ---------------------------------------------------------------------------
@respx.mock
async def test_batch_search_strings(provider):
    batch_resp = [
        SERPER_SEARCH_RESPONSE,
        {
            "organic": [
                {"title": "Tesla", "link": "https://tesla.com", "snippet": "t", "position": 1}
            ]
        },
    ]
    route = respx.post(f"{BASE}/search").mock(return_value=httpx.Response(200, json=batch_resp))

    results = await provider.batch_search(["apple inc", "tesla inc"], count=5, country="us")
    assert isinstance(results, list) and len(results) == 2
    assert results[0].query == "apple inc"
    assert results[1].query == "tesla inc"
    assert results[1].items[0].title == "Tesla"

    # Body must be a JSON array of query objects.
    body = json.loads(route.calls.last.request.content)
    assert isinstance(body, list) and len(body) == 2
    assert body[0] == {"q": "apple inc", "num": 5, "gl": "us", "hl": "en"}


@respx.mock
async def test_batch_search_dicts(provider):
    route = respx.post(f"{BASE}/patents").mock(
        return_value=httpx.Response(200, json=[{"organic": []}, {"organic": []}])
    )
    queries = [
        {"q": "some query", "tbs": "qdr:m", "page": 2},
        {"q": "google inc", "tbs": "qdr:m", "page": 2},
    ]
    results = await provider.batch_search(queries, search_type="patents")
    assert len(results) == 2
    body = json.loads(route.calls.last.request.content)
    assert body == queries  # dicts passed through unchanged


async def test_batch_search_dict_without_q_raises(provider):
    with pytest.raises(SearchProviderError, match="must include a 'q' field"):
        await provider.batch_search([{"tbs": "qdr:m"}])


async def test_batch_search_empty_raises(provider):
    with pytest.raises(SearchProviderError, match="non-empty list"):
        await provider.batch_search([])


@respx.mock
async def test_batch_search_http_error_returns_failed_list(provider):
    respx.post(f"{BASE}/search").mock(
        return_value=httpx.Response(429, json={"message": "Too Many Requests"})
    )
    # max_retries=3 -> retries exhausted -> 429 returned -> failed results
    results = await provider.batch_search(["a", "b"], count=1)
    assert len(results) == 2
    assert all(r.success is False for r in results)
    assert all("429" in (r.error or "") for r in results)


# ---------------------------------------------------------------------------
# scrape (separate host)
# ---------------------------------------------------------------------------
@respx.mock
async def test_scrape_markdown(provider):
    route = respx.post(SCRAPE).mock(
        return_value=httpx.Response(
            200,
            json={"markdown": "# Title\n\nbody", "text": "Title body", "metadata": {"title": "T"}},
        )
    )
    res = await provider.scrape("https://shop.example.com/item/42")
    assert isinstance(res, ScrapeResult)
    assert res.success is True
    assert res.response_format == "markdown"
    assert res.content == "# Title\n\nbody"
    assert res.content_char_size == len("# Title\n\nbody")
    assert res.root_domain == "example.com"
    assert res.raw["metadata"]["title"] == "T"  # full payload preserved

    body = json.loads(route.calls.last.request.content)
    assert body == {"url": "https://shop.example.com/item/42", "includeMarkdown": True}
    assert route.calls.last.request.headers["x-api-key"] == "k_test_1234567890"


@respx.mock
async def test_scrape_json_no_markdown_flag(provider):
    respx.post(SCRAPE).mock(
        return_value=httpx.Response(200, json={"text": "hello", "metadata": {}})
    )
    res = await provider.scrape("https://x.example", response_format="json")
    assert res.success is True
    assert res.response_format == "json"
    assert res.content == {"text": "hello", "metadata": {}}


@respx.mock
async def test_scrape_passthrough_kwargs(provider):
    route = respx.post(SCRAPE).mock(return_value=httpx.Response(200, json={"markdown": "x"}))
    await provider.scrape("https://x.example", q="google inc")
    body = json.loads(route.calls.last.request.content)
    assert body["q"] == "google inc"
    assert body["includeMarkdown"] is True


@respx.mock
async def test_scrape_http_error(provider):
    respx.post(SCRAPE).mock(return_value=httpx.Response(500, text="boom"))
    res = await provider.scrape("https://x.example")
    assert res.success is False
    assert res.status == "error"
    assert "500" in (res.error or "")


# ---------------------------------------------------------------------------
# auth / retries / health / lifecycle
# ---------------------------------------------------------------------------
@respx.mock
async def test_auth_error_raises(provider):
    respx.post(f"{BASE}/search").mock(
        return_value=httpx.Response(403, json={"message": "Unauthorized", "statusCode": 403})
    )
    with pytest.raises(SearchProviderError) as exc:
        await provider.web_search("q")
    assert exc.value.status_code == 403


@respx.mock
async def test_retry_on_429_then_success(provider):
    respx.post(f"{BASE}/search").mock(
        side_effect=[
            httpx.Response(429, json={"message": "rate"}),
            httpx.Response(200, json=SERPER_SEARCH_RESPONSE),
        ]
    )
    res = await provider.web_search("q")
    assert res.success is True
    assert res.items[0].title == "Apple"


@respx.mock
async def test_health_check_ok(provider):
    respx.post(f"{BASE}/search").mock(return_value=httpx.Response(200, json={"organic": []}))
    assert await provider.health_check() is True


@respx.mock
async def test_health_check_failure(provider):
    respx.post(f"{BASE}/search").mock(return_value=httpx.Response(401, text="nope"))
    assert await provider.health_check() is False


@respx.mock
async def test_close_idempotent(provider):
    respx.post(f"{BASE}/search").mock(return_value=httpx.Response(200, json={"organic": []}))
    await provider.health_check()
    assert provider._client is not None
    await provider.close()
    assert provider._client is None
    await provider.close()


# ---------------------------------------------------------------------------
# manager wiring + pure helpers
# ---------------------------------------------------------------------------
def test_manager_loads_serper(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "k")
    cfg = Config(defaults={"search_providers": {"serper": {}}})
    m = SearchProviderManager(cfg)
    assert m.get_available_search_providers() == ["serper"]
    assert m.get_default_search_provider().get_name() == "serper"


def test_manager_serper_alias(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "k")
    cfg = Config(defaults={"search_providers": {"web": {"type": "serper_dev"}}})
    m = SearchProviderManager(cfg)
    assert m.get_search_provider("web").get_name() == "serper"


def test_normalize_serper_falls_back_to_organic():
    # Unknown vertical key -> falls back to "organic".
    data = {"organic": [{"title": "t", "link": "u", "snippet": "s", "position": 1}]}
    items, total = _normalize_serper(data, "scholar")
    assert len(items) == 1 and items[0].url == "u"
    assert total is None


def test_normalize_serper_non_dict():
    assert _normalize_serper(None, "search") == ([], None)
