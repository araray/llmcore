# tests/providers/test_typesafe_current_api.py
"""TypeSafe against Jev 1.14 and typesafe-sdk 0.7.5 (live-verified 2026-11-16).

- jev-latest answers as jev-1.14.0; Box questions locate things in images
  (live: a red square at x 400-550 / 600, y 50-200 / 400 came back as
  left 0.67, right 0.92, top 0.11, bottom 0.51);
- images and Box need TypeSafe-Danger-Zone-Images: true;
- the SDK path marked every failure retryable, a 400 included.
"""

from __future__ import annotations

import base64
import json

import httpx
import pytest
import respx

from llmcore.providers.typesafe_provider import (
    Box,
    Noul,
    TypeSafeProvider,
    _sdk_error_status,
    _wire_asset,
)

URL = "https://api.typesafe.ai/v1/systemone"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


@respx.mock
async def test_assets_box_and_headers():
    route = respx.post(URL).mock(return_value=httpx.Response(200, json={
        "model": "jev-1.14.0",
        "answers": {"red": {"type": "noul", "noul": 0.99},
                    "where": {"type": "box", "box": {"top": 0.11, "bottom": 0.51, "left": 0.67, "right": 0.92},
                              "confidence": 0.95}},
        "usage": {"input_tokens": 900, "output_tokens": 10}}))
    p = TypeSafeProvider({"api_key": "k", "backend": "httpx", "long_context": True})
    result = await p.system_one({"picture": "img-1"},
                                {"red": Noul(instructions="Red square in `picture`?"),
                                 "where": Box(instructions="The red square in `picture`.")},
                                assets={"img-1": PNG})
    request = route.calls[0].request
    body = json.loads(request.content)
    assert body["assets"] == {"img-1": {"type": "image", "format": "png",
                                        "data_base64": base64.b64encode(PNG).decode()}}
    assert request.headers["TypeSafe-Danger-Zone-Images"] == "true"
    assert request.headers["TypeSafe-Danger-Zone-Long-Context"] == "true"
    assert result.boxes["where"].box["left"] == 0.67 and result.boxes["where"].confidence == 0.95
    assert p.get_max_context_length("jev-latest") == 262_144


def test_assets_are_png_or_jpeg():
    assert _wire_asset("a", b"\xff\xd8\xff\xe0")["format"] == "jpeg"
    assert _wire_asset("a", {"format": "png", "data_base64": "x"}) == {"type": "image", "format": "png", "data_base64": "x"}
    with pytest.raises(ValueError, match="PNG or JPEG"):
        _wire_asset("a", b"GIF89a")


def test_sdk_errors_are_classified():
    bad = type("TypeSafeBadRequestError", (Exception,), {})()
    limited = type("TypeSafeRateLimitError", (Exception,), {})()
    local = type("ValidationError", (Exception,), {})()
    timeout = type("TypeSafeAPITimeoutError", (Exception,), {})()
    assert _sdk_error_status(bad) == (400, False)
    assert _sdk_error_status(limited) == (429, True)
    assert _sdk_error_status(local) == (None, False)
    assert _sdk_error_status(timeout) == (None, True)
