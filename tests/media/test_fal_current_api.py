# tests/media/test_fal_current_api.py
"""fal against its live catalogue and fal-client 1.0.3 (verified 2026-11-16).

- fal-ai/minimax-video is gone (404 on the Platform API); Hailuo 2.3 takes
  ``duration`` as "6" / "10" and splits text- and image-to-video;
- fal-ai/mmaudio-v2 requires video_url, so a text-only SFX request needs
  .../text-to-audio; the bare fal-ai/kokoro id has no schema;
- FLUX takes image_size as a preset or {width, height}, not "WxH"
  (live: size="512x384" now yields a 512x384 image);
- the SDK backend dropped the webhook URL and let FalClientHTTPError escape.
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
from llmcore.providers.fal_provider import FalProvider, _image_size

QUEUE = "https://queue.fal.run"
CONFIG: dict[str, Any] = {"api_key": "fal-test-key", "_instance_name": "fal", "backend": "httpx"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("FAL_KEY", "FAL_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _submitted(route: respx.Route) -> dict[str, Any]:
    return json.loads(route.calls[0].request.content)


def _ok() -> httpx.Response:
    return httpx.Response(200, json={"request_id": "r1"})


def test_image_size_translation():
    assert _image_size("1024x768") == {"width": 1024, "height": 768}
    assert _image_size("landscape_4_3") == "landscape_4_3"
    assert _image_size(None) is None


@respx.mock
async def test_video_defaults_follow_hailuo():
    t2v = respx.post(f"{QUEUE}/fal-ai/minimax/hailuo-2.3/standard/text-to-video").mock(return_value=_ok())
    i2v = respx.post(f"{QUEUE}/fal-ai/minimax/hailuo-2.3/standard/image-to-video").mock(return_value=_ok())
    p = FalProvider(CONFIG)
    await p.generate_video_media("waves", duration_seconds=9)
    assert _submitted(t2v)["duration"] == "10"
    job = await p.generate_video_media("waves", first_frame=MediaRef.from_url("https://x/a.png"))
    assert job.model.endswith("image-to-video")
    assert _submitted(i2v)["image_url"] == "https://x/a.png"


@respx.mock
async def test_text_only_sfx_uses_text_to_audio_unless_pinned():
    text = respx.post(f"{QUEUE}/fal-ai/mmaudio-v2/text-to-audio").mock(return_value=_ok())
    foley = respx.post(f"{QUEUE}/fal-ai/mmaudio-v2").mock(return_value=_ok())
    p = FalProvider(CONFIG)
    await p.generate_sfx_media("a door creaking", duration_seconds=2)
    assert text.called and not foley.called
    await p.generate_sfx_media("steps", video=MediaRef.from_url("https://x/v.mp4"))
    assert _submitted(foley)["video_url"] == "https://x/v.mp4"


async def test_sdk_backend_forwards_webhook_and_maps_errors():
    p = FalProvider(CONFIG)
    p._backend = "sdk"
    p._sdk = MagicMock()
    p._sdk.submit = AsyncMock(return_value=MagicMock(request_id="r1"))
    await p._submit("fal-ai/flux/schnell", {"prompt": "x"}, webhook_url="https://hook/1")
    assert p._sdk.submit.await_args.kwargs["webhook_url"] == "https://hook/1"

    http_error = type("FalClientHTTPError", (Exception,), {"status_code": 422})("Input should be <= 12")
    p._sdk.result = AsyncMock(side_effect=http_error)
    with pytest.raises(ProviderError) as info:
        await p._fetch_result("fal-ai/flux/schnell", "r1")
    assert info.value.status_code == 422 and info.value.retryable is False


async def test_sdk_completed_with_error_fails_the_job():
    completed = type("Completed", (), {"error": "boom", "error_type": "runner_error", "metrics": {}, "position": None})()
    p = FalProvider(CONFIG)
    p._backend = "sdk"
    p._sdk = MagicMock()
    p._sdk.status = AsyncMock(return_value=completed)
    p._sdk.result = AsyncMock()
    from llmcore.media.models import MediaCapability, MediaJob, MediaJobStatus

    job = MediaJob(capability=MediaCapability("image_generate"), provider="fal", model="fal-ai/flux/schnell",
                   status=MediaJobStatus.QUEUED, provider_job_id="r1", provider_metadata={"endpoint": "fal-ai/flux/schnell"})
    job = await p.poll_media_job(job)
    assert job.status == MediaJobStatus.FAILED and job.error == "boom"
    p._sdk.result.assert_not_awaited()
