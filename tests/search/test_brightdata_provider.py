# tests/search/test_brightdata_provider.py
"""Tests for :mod:`llmcore.search.providers.brightdata_provider`.

Design notes
------------
* All HTTP is intercepted with ``respx`` (httpx interception) — **no network**.
  We assert the exact endpoints, request bodies, query params, and auth headers
  the provider sends, plus correct normalization of the responses.
* ``asyncio.sleep`` is patched to a no-op (autouse) so polling/retry paths run
  instantly.
* Construction uses an explicit token so no environment is required.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from llmcore.exceptions import ConfigError, SearchProviderError
from llmcore.search.models import (
    DiscoverResult,
    ScrapeResult,
    WebSearchResult,
)
from llmcore.search.providers.brightdata_provider import (
    DEFAULT_BASE_URL,
    BrightDataSearchProvider,
    _normalize_serp,
    _parse_records,
    _root_domain,
    _unwrap_request_body,
)

BASE = DEFAULT_BASE_URL


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Patch asyncio.sleep so polling/retry loops run instantly."""

    async def _instant(_seconds):
        return None

    monkeypatch.setattr("asyncio.sleep", _instant)
    yield


@pytest.fixture
def provider():
    """A BrightDataSearchProvider with both zones configured and fast polling."""
    return BrightDataSearchProvider(
        {
            "api_key": "tok_test_1234567890",
            "serp_zone": "z_serp",
            "unlocker_zone": "z_unlocker",
            "_instance_name": "brightdata",
            "timeout": 5,
            "poll_interval": 0,
            "poll_timeout": 2,
            "max_retries": 3,
        }
    )


# ---------------------------------------------------------------------------
# Construction / identity / config resolution
# ---------------------------------------------------------------------------
def test_identity_and_capabilities(provider):
    assert provider.get_name() == "brightdata"
    assert provider.get_capabilities() == {"web_search", "scrape", "discover", "dataset_search"}
    assert provider.supports("web_search") is True
    assert provider.supports("crawl") is False
    assert provider._base_url == BASE


def test_token_from_env(monkeypatch):
    monkeypatch.setenv("BRIGHTDATA_API_TOKEN", "tok_from_env_999")
    p = BrightDataSearchProvider({"_instance_name": "brightdata"})
    assert p._token == "tok_from_env_999"


def test_custom_env_var(monkeypatch):
    monkeypatch.delenv("BRIGHTDATA_API_TOKEN", raising=False)
    monkeypatch.setenv("MY_BD_TOKEN", "tok_custom_777")
    p = BrightDataSearchProvider({"api_key_env_var": "MY_BD_TOKEN", "_instance_name": "brightdata"})
    assert p._token == "tok_custom_777"


def test_missing_token_raises(monkeypatch):
    monkeypatch.delenv("BRIGHTDATA_API_TOKEN", raising=False)
    with pytest.raises(ConfigError, match="token not found"):
        BrightDataSearchProvider({"_instance_name": "brightdata"})


def test_invalid_default_engine_falls_back(monkeypatch):
    p = BrightDataSearchProvider({"api_key": "tok_test_1234567890", "default_engine": "altavista"})
    assert p._default_engine == "google"


# ---------------------------------------------------------------------------
# web_search — sync
# ---------------------------------------------------------------------------
@respx.mock
async def test_web_search_sync_google(provider):
    route = respx.post(f"{BASE}/request").mock(
        return_value=httpx.Response(
            200,
            json={
                "organic": [
                    {
                        "rank": 1,
                        "title": "Result One",
                        "link": "https://one.example.com",
                        "description": "first",
                        "display_link": "one.example.com",
                    },
                    {
                        "rank": 2,
                        "title": "Result Two",
                        "link": "https://two.example.com",
                        "description": "second",
                    },
                ],
                "total_results": 12345,
            },
        )
    )

    result = await provider.web_search("python asyncio", count=2, country="US", language="en")

    assert isinstance(result, WebSearchResult)
    assert result.success is True
    assert result.provider == "brightdata"
    assert result.engine == "google"
    assert result.total_results == 12345
    assert [i.title for i in result.items] == ["Result One", "Result Two"]
    assert result.items[0].position == 1
    assert result.items[0].url == "https://one.example.com"
    assert result.items[0].displayed_url == "one.example.com"
    assert result.elapsed_ms() is not None

    # Verify the exact request the provider sent.
    req = route.calls.last.request
    assert req.method == "POST"
    assert str(req.url) == f"{BASE}/request"
    assert req.headers["authorization"] == "Bearer tok_test_1234567890"
    assert req.headers["content-type"] == "application/json"
    body = json.loads(req.content)
    assert body["zone"] == "z_serp"
    assert body["format"] == "json"
    assert body["method"] == "GET"
    assert "brd_json=1" in body["url"]
    assert "q=python+asyncio" in body["url"]
    assert "num=2" in body["url"]
    assert "hl=en" in body["url"]
    assert "gl=us" in body["url"]


@respx.mock
async def test_web_search_unwraps_body_envelope(provider):
    inner = {"organic": [{"rank": 1, "title": "Wrapped", "link": "https://w.example"}]}
    respx.post(f"{BASE}/request").mock(
        return_value=httpx.Response(
            200,
            json={"status_code": 200, "body": json.dumps(inner)},
        )
    )
    result = await provider.web_search("q", count=1)
    assert result.success is True
    assert len(result.items) == 1
    assert result.items[0].title == "Wrapped"


@respx.mock
async def test_web_search_no_parse_returns_raw_html(provider):
    respx.post(f"{BASE}/request").mock(
        return_value=httpx.Response(200, text="<html><body>raw</body></html>")
    )
    result = await provider.web_search("q", count=3, parse=False)
    assert result.success is True
    assert result.items == []
    # raw HTML preserved (string passthrough)
    assert "raw" in (result.raw if isinstance(result.raw, str) else "")


async def test_web_search_requires_serp_zone():
    p = BrightDataSearchProvider({"api_key": "tok_test_1234567890"})  # no serp_zone
    with pytest.raises(SearchProviderError, match="No SERP zone configured"):
        await p.web_search("q")


@respx.mock
async def test_web_search_bad_engine_raises(provider):
    with pytest.raises(SearchProviderError, match="Unsupported engine"):
        await provider.web_search("q", engine="altavista")


# ---------------------------------------------------------------------------
# web_search — async (unblocker trigger + poll)
# ---------------------------------------------------------------------------
@respx.mock
async def test_web_search_async_mode(provider):
    trigger = respx.post(url__startswith=f"{BASE}/unblocker/req").mock(
        return_value=httpx.Response(200, headers={"x-response-id": "resp-123"})
    )
    # First poll: pending (202); second: ready (200) with JSON.
    poll = respx.get(url__startswith=f"{BASE}/unblocker/get_result").mock(
        side_effect=[
            httpx.Response(202),
            httpx.Response(200, json={"organic": [{"rank": 1, "title": "Async", "link": "u"}]}),
        ]
    )

    result = await provider.web_search("q", count=1, mode="async")
    assert result.success is True
    assert [i.title for i in result.items] == ["Async"]

    # Trigger body carries the URL; zone is a query param.
    treq = trigger.calls.last.request
    assert json.loads(treq.content)["url"].startswith("https://www.google.com/search")
    assert "zone=z_serp" in str(treq.url)
    assert poll.call_count == 2


@respx.mock
async def test_web_search_async_missing_response_id(provider):
    respx.post(url__startswith=f"{BASE}/unblocker/req").mock(
        return_value=httpx.Response(200)  # no x-response-id header
    )
    with pytest.raises(SearchProviderError, match="no x-response-id"):
        await provider.web_search("q", mode="async")


# ---------------------------------------------------------------------------
# scrape
# ---------------------------------------------------------------------------
@respx.mock
async def test_scrape_sync_raw(provider):
    route = respx.post(f"{BASE}/request").mock(
        return_value=httpx.Response(200, text="<html>hello</html>")
    )
    result = await provider.scrape("https://shop.example.com/item/42", country="de")
    assert isinstance(result, ScrapeResult)
    assert result.success is True
    assert result.content == "<html>hello</html>"
    assert result.response_format == "raw"
    assert result.status == "ready"
    assert result.root_domain == "example.com"
    assert result.content_char_size == len("<html>hello</html>")

    body = json.loads(route.calls.last.request.content)
    assert body["zone"] == "z_unlocker"
    assert body["url"] == "https://shop.example.com/item/42"
    assert body["format"] == "raw"
    assert body["method"] == "GET"
    assert body["country"] == "DE"  # uppercased


@respx.mock
async def test_scrape_sync_json(provider):
    respx.post(f"{BASE}/request").mock(
        return_value=httpx.Response(200, json={"title": "X", "price": 9.99})
    )
    result = await provider.scrape("https://x.example", response_format="json")
    assert result.success is True
    assert result.content == {"title": "X", "price": 9.99}
    assert result.content_char_size is None


@respx.mock
async def test_scrape_http_error_returns_error_result(provider):
    respx.post(f"{BASE}/request").mock(return_value=httpx.Response(422, text="bad"))
    result = await provider.scrape("https://x.example")
    assert result.success is False
    assert result.status == "error"
    assert "422" in (result.error or "")


async def test_scrape_requires_unlocker_zone():
    p = BrightDataSearchProvider({"api_key": "tok_test_1234567890", "serp_zone": "z"})
    with pytest.raises(SearchProviderError, match="No unlocker zone configured"):
        await p.scrape("https://x.example")


# ---------------------------------------------------------------------------
# discover
# ---------------------------------------------------------------------------
@respx.mock
async def test_discover_trigger_and_poll(provider):
    trigger = respx.route(method="POST", url=f"{BASE}/discover").mock(
        return_value=httpx.Response(200, json={"task_id": "task-9"})
    )
    poll = respx.route(method="GET", url__startswith=f"{BASE}/discover").mock(
        side_effect=[
            httpx.Response(200, json={"status": "processing"}),
            httpx.Response(
                200,
                json={
                    "status": "done",
                    "duration_seconds": 3.5,
                    "results": [
                        {
                            "title": "AI Trends",
                            "link": "https://ai.example",
                            "description": "d",
                            "relevance_score": 0.91,
                            "content": "# markdown",
                        }
                    ],
                },
            ),
        ]
    )

    result = await provider.discover(
        "ai trends 2026", intent="latest developments", include_content=True
    )
    assert isinstance(result, DiscoverResult)
    assert result.success is True
    assert result.task_id == "task-9"
    assert result.duration_seconds == 3.5
    assert len(result.items) == 1
    assert result.items[0].relevance_score == 0.91
    assert result.items[0].content == "# markdown"

    body = json.loads(trigger.calls.last.request.content)
    assert body["query"] == "ai trends 2026"
    assert body["intent"] == "latest developments"
    assert body["include_content"] is True
    assert poll.call_count == 2


@respx.mock
async def test_discover_task_failure(provider):
    respx.route(method="POST", url=f"{BASE}/discover").mock(
        return_value=httpx.Response(200, json={"task_id": "t"})
    )
    respx.route(method="GET", url__startswith=f"{BASE}/discover").mock(
        return_value=httpx.Response(200, json={"status": "failed", "error": "boom"})
    )
    with pytest.raises(SearchProviderError, match="discover task failed: boom"):
        await provider.discover("q")


@respx.mock
async def test_discover_no_task_id(provider):
    respx.route(method="POST", url=f"{BASE}/discover").mock(
        return_value=httpx.Response(200, json={})
    )
    with pytest.raises(SearchProviderError, match="no task_id"):
        await provider.discover("q")


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
@respx.mock
async def test_list_datasets(provider):
    respx.get(f"{BASE}/datasets/list").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"id": "gd_1", "name": "LinkedIn Profiles", "size": 1000},
                {"id": "gd_2", "name": "Amazon Products", "size": 500},
            ],
        )
    )
    datasets = await provider.list_datasets()
    assert [d.id for d in datasets] == ["gd_1", "gd_2"]
    assert datasets[0].name == "LinkedIn Profiles"
    assert datasets[0].size == 1000


@respx.mock
async def test_dataset_metadata(provider):
    respx.get(f"{BASE}/datasets/gd_1/metadata").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "gd_1",
                "fields": {
                    "name": {"type": "text", "required": True},
                    "followers": {"type": "number", "active": True},
                },
            },
        )
    )
    meta = await provider.dataset_metadata("gd_1")
    assert meta.id == "gd_1"
    assert set(meta.field_names()) == {"name", "followers"}
    name_field = next(f for f in meta.fields if f.name == "name")
    assert name_field.type == "text"
    assert name_field.required is True


@respx.mock
async def test_dataset_filter_returns_snapshot_id(provider):
    route = respx.post(f"{BASE}/datasets/filter").mock(
        return_value=httpx.Response(200, json={"snapshot_id": "snap-1"})
    )
    snap = await provider.dataset_filter(
        "gd_1",
        {"name": "industry", "operator": "=", "value": "Technology"},
        records_limit=100,
    )
    assert snap.success is True
    assert snap.snapshot_id == "snap-1"
    assert snap.status == "scheduled"

    body = json.loads(route.calls.last.request.content)
    assert body["dataset_id"] == "gd_1"
    assert body["filter"]["value"] == "Technology"
    assert body["records_limit"] == 100


@respx.mock
async def test_dataset_filter_no_snapshot_id(provider):
    respx.post(f"{BASE}/datasets/filter").mock(
        return_value=httpx.Response(200, json={"error": "bad filter"})
    )
    snap = await provider.dataset_filter("gd_1", {"x": 1})
    assert snap.success is False
    assert "bad filter" in (snap.error or "")


@respx.mock
async def test_dataset_status(provider):
    respx.get(f"{BASE}/datasets/snapshots/snap-1").mock(
        return_value=httpx.Response(
            200, json={"id": "snap-1", "status": "building", "dataset_id": "gd_1"}
        )
    )
    snap = await provider.dataset_status("snap-1")
    assert snap.snapshot_id == "snap-1"
    assert snap.status == "building"
    assert snap.dataset_id == "gd_1"


@respx.mock
async def test_dataset_download_polls_then_downloads_jsonl(provider):
    respx.get(f"{BASE}/datasets/snapshots/snap-1").mock(
        side_effect=[
            httpx.Response(200, json={"id": "snap-1", "status": "building"}),
            httpx.Response(200, json={"id": "snap-1", "status": "ready", "dataset_size": 2}),
        ]
    )
    respx.get(url__startswith=f"{BASE}/datasets/snapshots/snap-1/download").mock(
        return_value=httpx.Response(
            200,
            headers={"Content-Type": "application/x-ndjson"},
            text='{"a": 1}\n{"a": 2}\n',
        )
    )
    snap = await provider.dataset_download("snap-1", format="jsonl")
    assert snap.success is True
    assert snap.status == "ready"
    assert snap.record_count == 2
    assert snap.records == [{"a": 1}, {"a": 2}]


@respx.mock
async def test_dataset_search_convenience(provider):
    respx.post(f"{BASE}/datasets/filter").mock(
        return_value=httpx.Response(200, json={"snapshot_id": "snap-2"})
    )
    respx.get(f"{BASE}/datasets/snapshots/snap-2").mock(
        return_value=httpx.Response(
            200, json={"id": "snap-2", "status": "ready", "dataset_size": 1}
        )
    )
    respx.get(url__startswith=f"{BASE}/datasets/snapshots/snap-2/download").mock(
        return_value=httpx.Response(200, json=[{"name": "Acme"}])
    )
    snap = await provider.dataset_search("gd_1", {"name": "x", "operator": "is_not_null"})
    assert snap.success is True
    assert snap.records == [{"name": "Acme"}]


@respx.mock
async def test_dataset_download_failed_status(provider):
    respx.get(f"{BASE}/datasets/snapshots/snap-x").mock(
        return_value=httpx.Response(200, json={"id": "snap-x", "status": "failed", "error": "nope"})
    )
    snap = await provider.dataset_download("snap-x")
    assert snap.success is False
    assert "nope" in (snap.error or "")


# ---------------------------------------------------------------------------
# health / errors / retries / lifecycle
# ---------------------------------------------------------------------------
@respx.mock
async def test_health_check_ok(provider):
    respx.get(f"{BASE}/zone/get_active_zones").mock(return_value=httpx.Response(200, json=[]))
    assert await provider.health_check() is True


@respx.mock
async def test_health_check_failure_returns_false(provider):
    respx.get(f"{BASE}/zone/get_active_zones").mock(return_value=httpx.Response(401, text="nope"))
    assert await provider.health_check() is False


@respx.mock
async def test_auth_error_raises_searchprovidererror(provider):
    respx.post(f"{BASE}/request").mock(return_value=httpx.Response(401, text="unauthorized"))
    with pytest.raises(SearchProviderError) as exc_info:
        await provider.web_search("q")
    assert exc_info.value.status_code == 401


@respx.mock
async def test_retry_on_5xx_then_success(provider):
    respx.post(f"{BASE}/request").mock(
        side_effect=[
            httpx.Response(503, text="busy"),
            httpx.Response(200, json={"organic": [{"rank": 1, "title": "ok", "link": "u"}]}),
        ]
    )
    result = await provider.web_search("q", count=1)
    assert result.success is True
    assert result.items[0].title == "ok"


@respx.mock
async def test_close_is_idempotent(provider):
    respx.get(f"{BASE}/zone/get_active_zones").mock(return_value=httpx.Response(200, json=[]))
    await provider.health_check()  # forces client creation
    assert provider._client is not None
    await provider.close()
    assert provider._client is None
    await provider.close()  # no error second time


# ---------------------------------------------------------------------------
# pure helper functions
# ---------------------------------------------------------------------------
def test_root_domain():
    assert _root_domain("https://www.example.com/path?x=1") == "example.com"
    assert _root_domain("http://localhost:8080") == "localhost"
    assert _root_domain("not a url") is None or isinstance(_root_domain("not a url"), str)


def test_normalize_serp_handles_non_dict():
    assert _normalize_serp("google", None) == ([], None)
    assert _normalize_serp("google", "html string") == ([], None)


def test_unwrap_request_body_html():
    out = _unwrap_request_body({"status_code": 200, "body": "<html>x</html>"})
    assert out["raw_html"] == "<html>x</html>"


def test_parse_records_json_array():
    resp = httpx.Response(200, json=[{"a": 1}, {"a": 2}])
    assert _parse_records(resp, "json") == [{"a": 1}, {"a": 2}]


def test_parse_records_jsonl():
    resp = httpx.Response(
        200, headers={"Content-Type": "application/x-ndjson"}, text='{"a":1}\n{"a":2}'
    )
    assert _parse_records(resp, "jsonl") == [{"a": 1}, {"a": 2}]
