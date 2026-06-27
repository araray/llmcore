"""HTTP + SSE conformance — exercises the secondary transport via ASGI."""

from __future__ import annotations

import json

import pytest

CHAT = "/llmcore.v1/InferenceService/Chat"
CHAT_STREAM = "/llmcore.v1/InferenceService/ChatStream"
COUNT = "/llmcore.v1/InferenceService/CountTokens"
COST = "/llmcore.v1/InferenceService/EstimateCost"
EMBED = "/llmcore.v1/InferenceService/Embed"
PROVIDERS = "/llmcore.v1/CatalogService/ListProviders"
MODELS = "/llmcore.v1/CatalogService/ListModels"
DETAILS = "/llmcore.v1/CatalogService/GetProviderDetails"
INFO = "/llmcore.v1/ControlService/GetInfo"


async def _sse_collect(client, url, payload):
    """Return (data_chunks, terminal_event) from an SSE response."""
    data_chunks, terminal = [], None
    async with client.stream("POST", url, json=payload) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        event = "message"
        async for line in resp.aiter_lines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                body = json.loads(line.split(":", 1)[1].strip())
                if event == "done":
                    terminal = ("done", body)
                elif event == "error":
                    terminal = ("error", body)
                else:
                    data_chunks.append(body)
                event = "message"
    return data_chunks, terminal


@pytest.mark.asyncio
async def test_chat(http_client):
    r = await http_client.post(CHAT, json={"message": "hello world"})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "echo: hello world"
    assert body["usage"]["prompt_tokens"] == 2
    assert body["usage"]["provider"] == "fake"


@pytest.mark.asyncio
async def test_chat_stream_sse(http_client):
    chunks, terminal = await _sse_collect(http_client, CHAT_STREAM, {"message": "streaming please"})
    assert terminal is not None and terminal[0] == "done"
    assert "".join(c.get("text", "") for c in chunks) == "echo: streaming please"


@pytest.mark.asyncio
async def test_count_and_cost(http_client):
    r = await http_client.post(COUNT, json={"text": "one two three"})
    assert r.json()["tokens"] == 3
    r = await http_client.post(
        COST,
        json={"provider_name": "fake", "model_name": "fake-1",
              "prompt_tokens": 1000000, "completion_tokens": 1000000},
    )
    body = r.json()
    assert body["currency"] == "USD"
    assert body["total_cost"] == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_catalog_and_control(http_client):
    assert (await http_client.post(PROVIDERS, json={})).json()["providers"] == ["fake"]
    assert (await http_client.post(MODELS, json={"provider_name": "fake"})).json()["models"] == [
        "fake-1", "fake-2",
    ]
    details = (await http_client.post(DETAILS, json={"provider_name": "fake"})).json()
    assert details["id"] == "fake-1" and details["context_length"] == 8192
    info = (await http_client.post(INFO, json={})).json()
    assert info["contract_version"] == "llmcore.v1"
    assert "tier0" in info["capabilities"]
    health = (await http_client.get("/healthz")).json()
    assert health["ok"] is True


@pytest.mark.asyncio
async def test_embed_returns_501(http_client):
    r = await http_client.post(EMBED, json={"input": ["x"]})
    assert r.status_code == 501
    err = r.json()["error"]
    assert err["category"] == "ERROR_CATEGORY_UNSUPPORTED"
    assert err["code"] == "unsupported.capability"


@pytest.mark.asyncio
async def test_error_provider_rate_limited(http_client):
    r = await http_client.post(CHAT, json={"message": "__error__:provider_rate_limited"})
    assert r.status_code == 429
    err = r.json()["error"]
    assert err["category"] == "ERROR_CATEGORY_PROVIDER"
    assert err["code"] == "provider.rate_limited"
    assert err["retryable"] is True
    assert err["retry_after_ms"] == pytest.approx(2000.0)


@pytest.mark.asyncio
async def test_invalid_json_is_400(http_client):
    r = await http_client.post(CHAT, content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 400
    assert r.json()["error"]["category"] == "ERROR_CATEGORY_INVALID_ARGUMENT"


@pytest.mark.asyncio
async def test_error_mid_stream_emits_error_event(http_client):
    chunks, terminal = await _sse_collect(
        http_client, CHAT_STREAM, {"message": "__error_mid__:internal"}
    )
    assert chunks  # one chunk before failure
    assert terminal is not None and terminal[0] == "error"
    assert terminal[1]["category"] == "ERROR_CATEGORY_INTERNAL"
