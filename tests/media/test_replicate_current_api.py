# tests/media/test_replicate_current_api.py
"""Replicate against the live API and replicate 1.0.7 (verified 2026-11-16).

- the SDK backend always ran by model, so community models (whisper,
  kokoro, musicgen) 404'd; live: jaaari/kokoro-82m now succeeds via sdk;
- ReplicateError escaped raw on the SDK backend (live 422 and 429 now map);
- Cancel-After bounds a prediction's runtime (documented header);
- Cog enums sit behind allOf/$ref (Hailuo 2.3 duration: [6, 10]).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx

from llmcore.exceptions import ProviderError
from llmcore.providers.replicate_provider import ReplicateProvider, _snap_to_enum

API = "https://api.replicate.com"
VERSION = "a" * 64
CONFIG: dict[str, Any] = {"api_key": "r8-test", "_instance_name": "replicate", "backend": "httpx"}

HAILUO = {
    "latest_version": {
        "id": VERSION,
        "openapi_schema": {"components": {"schemas": {
            "Input": {"properties": {
                "prompt": {"type": "string"},
                "duration": {"default": 6, "allOf": [{"$ref": "#/components/schemas/duration"}]},
                "first_frame_image": {"type": "string"},
            }},
            "duration": {"enum": [6, 10], "type": "integer"},
        }}},
    }
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("REPLICATE_API_TOKEN", "REPLICATE_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_snap_to_enum():
    assert _snap_to_enum(8.5, [6, 10]) == 10
    assert _snap_to_enum(7, [6, 10]) == 6
    assert _snap_to_enum(8, [6, 10]) == 10  # ties go up
    assert _snap_to_enum("768p", ["768p", "1080p"]) == "768p"
    assert _snap_to_enum(5, None) == 5


@respx.mock
async def test_ref_enums_and_cancel_after():
    respx.get(f"{API}/v1/models/minimax/hailuo-2.3").mock(return_value=httpx.Response(200, json=HAILUO))
    create = respx.post(f"{API}/v1/predictions").mock(
        return_value=httpx.Response(201, json={"id": "p1", "status": "starting", "urls": {}}))
    p = ReplicateProvider({**CONFIG, "cancel_after": "10m"})
    await p.generate_video_media("waves", duration_seconds=8.5)
    request = create.calls[0].request
    body = json.loads(request.content)
    assert body["version"] == VERSION and body["input"]["duration"] == 10
    assert request.headers["Cancel-After"] == "10m"


async def test_sdk_runs_community_models_by_version_and_maps_errors():
    p = ReplicateProvider({**CONFIG})
    p._backend = "sdk"
    p._sdk = MagicMock()
    p._version_cache["jaaari/kokoro-82m"] = VERSION
    p._schema_cache["jaaari/kokoro-82m"] = {"text": {}}
    p._sdk.predictions.async_create = AsyncMock(return_value={"id": "p1", "status": "starting", "urls": {}})
    await p.synthesize_speech_media("hello", webhook_url="https://hook/1")
    kwargs = p._sdk.predictions.async_create.await_args.kwargs
    assert kwargs["version"] == VERSION and "model" not in kwargs
    assert kwargs["webhook_events_filter"] == ["completed"]

    err = type("ReplicateError", (Exception,), {"status": 422})("Input validation failed")
    p._sdk.predictions.async_create = AsyncMock(side_effect=err)
    with pytest.raises(ProviderError) as info:
        await p.synthesize_speech_media("hello")
    assert info.value.status_code == 422 and not info.value.retryable
