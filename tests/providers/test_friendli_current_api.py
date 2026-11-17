# tests/providers/test_friendli_current_api.py
"""FriendliProvider with the real friendli 0.16 SDK (released 2026-11-04).

0.16 removed ``chat.stream``: streaming is ``chat.complete(stream=True)``.
The provider called ``chat.stream``, so ``backend = "sdk"`` streaming raised
AttributeError. This runs the real SDK end to end over a mocked transport.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

friendli = pytest.importorskip("friendli")

from llmcore.models import Message, Role  # noqa: E402
from llmcore.providers.friendli_provider import FriendliProvider  # noqa: E402

URL = "https://api.friendli.ai/serverless/v1/chat/completions"


def _sse(*chunks: dict) -> bytes:
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks).encode() + b"data: [DONE]\n\n"


def _chunk(delta: dict, finish: str | None = None) -> dict:
    return {"id": "x", "object": "chat.completion.chunk", "created": 1, "model": "zai-org/GLM-5.3",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


@respx.mock
async def test_sdk_streaming_uses_complete():
    assert not hasattr(friendli.AsyncFriendli(token="x").serverless.chat, "stream")
    route = respx.post(URL).mock(return_value=httpx.Response(
        200, headers={"content-type": "text/event-stream"},
        content=_sse(_chunk({"role": "assistant", "content": "", "reasoning_content": "Thin"}),
                     _chunk({"reasoning_content": "king"}),
                     _chunk({"content": "391"}, "stop"))))
    p = FriendliProvider({"api_key": "flp_test", "backend": "sdk", "team": "team1"})
    out = [c async for c in await p.chat_completion([Message(role=Role.USER, content="17*23?")],
                                                    "zai-org/GLM-5.3", stream=True)]
    assert json.loads(route.calls[0].request.content)["stream"] is True
    assert "".join(p.extract_delta_content(c) for c in out) == "391"
    # The SDK's chunk models drop reasoning_content (a known limitation of
    # this backend, warned about at init); the direct backend keeps it:


@respx.mock
async def test_direct_stream_carries_the_reasoning_for_replay():
    respx.post(URL).mock(return_value=httpx.Response(
        200, headers={"content-type": "text/event-stream"},
        content=_sse(_chunk({"role": "assistant", "content": "", "reasoning_content": "Thin"}),
                     _chunk({"reasoning_content": "king"}),
                     _chunk({"content": "391"}, "stop"))))
    p = FriendliProvider({"api_key": "flp_test", "backend": "httpx"})
    out = [c async for c in await p.chat_completion([Message(role=Role.USER, content="17*23?")],
                                                    "zai-org/GLM-5.3", stream=True)]
    assert p.assistant_replay_metadata(out[-1]) == {"reasoning_content": "Thinking"}


def test_non_stream_replay_metadata():
    p = FriendliProvider({"api_key": "flp_test", "backend": "httpx"})
    response = {"choices": [{"message": {"content": "391", "reasoning_content": "thinking"}}]}
    assert p.assistant_replay_metadata(response) == {"reasoning_content": "thinking"}


@pytest.mark.parametrize(
    ("model", "kwargs", "expected"),
    [("zai-org/GLM-5.3", {"effort": "medium"}, {"reasoning_effort": "high"}),
     ("zai-org/GLM-5.3", {"effort": "minimal"}, {"reasoning_effort": "low"}),
     ("zai-org/GLM-5.1", {"effort": "high"}, {}),  # toggle only: no effort level
     ("zai-org/GLM-5.1", {"effort": "none"}, {"enable_thinking": False}),
     # no toggle: "none" is the lowest level, not a disable (reasoning leaked, live)
     ("zai-org/GLM-5.3", {"effort": "none"}, {"reasoning_effort": "low"})],
)
def test_effort_follows_the_catalogue(model, kwargs, expected):
    p = FriendliProvider({"api_key": "flp_test", "backend": "httpx"})
    p._catalog = {  # live /serverless/v1/models reasoning_options, 2026-11-16
        "zai-org/GLM-5.3": {"reasoning_options": [{"type": "effort", "values": ["low", "high", "max"]}]},
        "zai-org/GLM-5.1": {"reasoning_options": [{"type": "toggle"}]},
    }
    assert p._map_effort(model, dict(kwargs)) == expected


@respx.mock
async def test_replay_survives_a_trailing_usage_chunk():
    """Live (2026-11-16): with usage on, GLM-5.3 ends with a usage-only chunk,
    and assistant_replay_metadata(chunks[-1]) returned nothing."""
    usage_only = {"id": "x", "object": "chat.completion.chunk", "created": 1, "model": "zai-org/GLM-5.3",
                  "choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}
    body = ("".join(f"data: {json.dumps(c)}\n\n" for c in (
        _chunk({"role": "assistant", "content": "", "reasoning_content": "Thin"}),
        _chunk({"reasoning_content": "king"}),
        _chunk({"content": "pong"}, "stop"),
        usage_only)) + "data: [DONE]\n\n").encode()
    respx.post(URL).mock(return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body))
    p = FriendliProvider({"api_key": "flp_test", "backend": "httpx"})
    out = [c async for c in await p.chat_completion([Message(role=Role.USER, content="hi")], "zai-org/GLM-5.3", stream=True)]
    assert out[-1].get("usage")
    assert p.assistant_replay_metadata(out[-1]) == {"reasoning_content": "Thinking"}
    assert "".join(p.extract_delta_content(c) for c in out) == "pong"
