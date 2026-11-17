# tests/media/test_higgsfield_current_api.py
"""Higgsfield against the live API and higgsfield-client 0.2.0 (2026-11-16).

- the live API enforces Soul ``resolution`` 720p/1080p and seven aspect
  ratios, not the 2K/4K and ten ratios its openapi.json still lists (live
  422); size="1024x768" now yields a completed Soul image on both backends;
- video durations are integer enums (Hailuo 6/10, Kling 5/10; live 400);
- the media protocol passes ``first_frame``, which the provider ignored;
- the SDK raised a bare HiggsfieldClientError (status only on __cause__).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx

from llmcore.exceptions import ProviderError
from llmcore.media import MediaRef
from llmcore.providers.higgsfield_provider import HiggsfieldProvider, _nearest_ratio

API = "https://api.higgsfield.ai"
CONFIG: dict[str, Any] = {"api_key": "id:secret", "_instance_name": "higgsfield", "backend": "httpx"}
QUEUED = {"request_id": "r1", "status": "queued", "status_url": f"{API}/requests/r1/status"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("HIGGSFIELD_API_KEY", "HIGGSFIELD_KEY"):
        monkeypatch.delenv(name, raising=False)


def _body(route: respx.Route) -> dict[str, Any]:
    return json.loads(route.calls[-1].request.content)


def test_nearest_ratio_uses_the_live_enum():
    assert _nearest_ratio(1024, 768) == "4:3"
    assert _nearest_ratio(2560, 1080) == "16:9"  # 21:9 is not accepted live
    assert _nearest_ratio(1000, 800) in ("4:3", "3:2")  # 5:4 is not accepted live


@respx.mock
async def test_image_size_maps_to_resolution_and_ratio():
    route = respx.post(f"{API}/higgsfield-ai/soul/standard").mock(return_value=httpx.Response(200, json=QUEUED))
    p = HiggsfieldProvider(CONFIG)
    await p.generate_image_media("apple", size="1920x1080")
    assert _body(route)["resolution"] == "1080p" and _body(route)["aspect_ratio"] == "16:9"
    await p.generate_image_media("apple", size="720p", aspect_ratio="1:1")
    assert _body(route)["resolution"] == "720p" and _body(route)["aspect_ratio"] == "1:1"


@respx.mock
async def test_video_first_frame_duration_and_unsupported_args(caplog):
    i2v = respx.post(f"{API}/minimax/hailuo-2.3/standard/image-to-video").mock(
        return_value=httpx.Response(200, json=QUEUED))
    kling = respx.post(f"{API}/kling-video/v2.5-turbo/pro/text-to-video").mock(
        return_value=httpx.Response(200, json=QUEUED))
    p = HiggsfieldProvider(CONFIG)
    await p.generate_video_media("waves", first_frame=MediaRef.from_url("https://x/a.png"),
                                 duration_seconds=8.5, seed=3, negative_prompt="blur")
    body = _body(i2v)
    assert body["image_url"] == "https://x/a.png" and body["duration"] == 10
    assert "seed" not in body and "negative_prompt" not in body
    assert "does not take negative_prompt, seed" in caplog.text
    await p.generate_video_media("waves", model="kling-video/v2.5-turbo/pro/text-to-video",
                                 duration_seconds=6, negative_prompt="blur", cfg_scale=0.7)
    body = _body(kling)
    assert body["duration"] == 5 and body["negative_prompt"] == "blur" and body["cfg_scale"] == 0.7


@respx.mock
async def test_delete_request_data():
    respx.delete(f"{API}/requests/r1").mock(return_value=httpx.Response(200, json={}))
    respx.delete(f"{API}/requests/gone").mock(return_value=httpx.Response(404, json={"detail": "x"}))
    respx.delete(f"{API}/requests/busy").mock(return_value=httpx.Response(409, json={"detail": "running"}))
    p = HiggsfieldProvider(CONFIG)
    assert await p.delete_request_data("r1") is True
    assert await p.delete_request_data("gone") is False
    with pytest.raises(ProviderError):
        await p.delete_request_data("busy")


async def test_sdk_errors_carry_the_http_status():
    p = HiggsfieldProvider(CONFIG)
    p._backend = "sdk"
    p._sdk = MagicMock()
    request = httpx.Request("POST", f"{API}/higgsfield-ai/soul/standard")
    response = httpx.Response(422, text='{"detail": "resolution"}', request=request)
    error = Exception("[{'loc': ['body', 'resolution']}]")
    error.__cause__ = httpx.HTTPStatusError("422", request=request, response=response)
    p._sdk.submit = AsyncMock(side_effect=error)
    with pytest.raises(ProviderError) as info:
        await p.generate_image_media("x")
    assert info.value.status_code == 422 and not info.value.retryable
