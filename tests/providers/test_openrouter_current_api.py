# tests/providers/test_openrouter_current_api.py
"""OpenRouter against the current API, and the OpenAI-compatible replay layer.

Measured live on 2026-11-16:

- Gemini 3 returns its reasoning only as an encrypted ``reasoning_details``
  item, which OpenRouter's docs say to send back unmodified and in order;
  streams deliver the items whole inside deltas;
- usage always carries ``cost`` / ``cost_details`` / ``is_byok``;
- ``/models`` is paginated and lists only text-output models by default
  (468 of 724);
- failures can arrive inside an HTTP 200 (``error`` in a body or chunk).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx

pytest.importorskip("openai")

from llmcore.exceptions import ProviderError  # noqa: E402
from llmcore.models import Message, Role, Tool  # noqa: E402
from llmcore.providers.openrouter_provider import OPENROUTER_MODELS_URL, OpenRouterProvider  # noqa: E402

from .httpx2_wire import R, Wire, sse  # noqa: E402

BASE = "https://openrouter.ai/api/v1"
CTX = [Message(role=Role.USER, content="Weather in Lisbon?")]
WEATHER = Tool(name="get_weather", description="w",
               parameters={"type": "object", "properties": {"city": {"type": "string"}}})
ENCRYPTED = {"type": "reasoning.encrypted", "data": "AY89", "format": "google-gemini-v1", "index": 0}
CALL = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Lisbon\"}"}}


def _provider(wire: Wire, **cfg: Any) -> OpenRouterProvider:
    p = OpenRouterProvider({"api_key": "k", **cfg})
    p._client = wire.client(BASE)
    return p


def _chat(message: dict[str, Any], finish: str = "stop", **extra: Any) -> dict[str, Any]:
    return {"id": "gen-1", "object": "chat.completion", "created": 1, "model": "google/gemini-3-flash-preview",
            "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 26, "completion_tokens": 17, "total_tokens": 43, "cost": 6.4e-05,
                      "is_byok": False, "cost_details": {"upstream_inference_cost": 6.4e-05}}, **extra}


def _chunk(delta: dict[str, Any], finish: str | None = None, **extra: Any) -> dict[str, Any]:
    return {"id": "gen-1", "object": "chat.completion.chunk", "created": 1, "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}


class TestReasoningDetailsReplay:
    async def test_non_stream_details_are_replayed_verbatim_with_cost(self):
        wire = Wire().on("/api/v1/chat/completions",
                         R(200, json=_chat({"content": "", "tool_calls": [CALL], "reasoning_details": [ENCRYPTED]},
                                           "tool_calls")),
                         R(200, json=_chat({"content": "22C"})))
        p = _provider(wire)
        first = await p.chat_completion(CTX, "google/gemini-3-flash-preview", tools=[WEATHER])
        meta = p.assistant_replay_metadata(first)
        assert meta == {"reasoning_details": [ENCRYPTED]}
        assert p.extract_usage_details(first)["cost"] == 6.4e-05
        history = CTX + [Message(role=Role.ASSISTANT, content="", tool_calls=[CALL], metadata=meta),
                         Message(role=Role.TOOL, content="22C", tool_call_id="c1")]
        await p.chat_completion(history, "google/gemini-3-flash-preview", tools=[WEATHER])
        assert wire.sent()["messages"][1]["reasoning_details"] == [ENCRYPTED]

    async def test_stream_accumulates_details_and_text(self):
        body = sse(_chunk({"role": "assistant", "content": "", "reasoning": "Need ",
                           "reasoning_details": [{"type": "reasoning.text", "text": "Need ", "index": 0}]}),
                   _chunk({"reasoning": "the tool.", "reasoning_details": [{"type": "reasoning.text", "text": "the tool.", "index": 0, "signature": "s"}]}),
                   _chunk({"tool_calls": [{"index": 0, **CALL}]}),
                   _chunk({}, "tool_calls"))
        wire = Wire().on("/api/v1/chat/completions", R(200, content=body, headers={"content-type": "text/event-stream"}))
        p = _provider(wire)
        out = [c async for c in await p.chat_completion(CTX, "google/gemini-3-flash-preview", tools=[WEATHER], stream=True)]
        meta = p.assistant_replay_metadata(next(c for c in out if c["choices"] and c["choices"][0].get("finish_reason")))
        assert meta["reasoning_details"] == [{"type": "reasoning.text", "text": "Need the tool.", "index": 0, "signature": "s"}]
        assert meta["reasoning_content"] == "Need the tool."
        # OpenRouter takes the text back as "reasoning" (its replay field).
        payload = p._build_message_payload(Message(role=Role.ASSISTANT, content="", tool_calls=[CALL], metadata=meta),
                                           "google/gemini-3-flash-preview")
        assert payload["reasoning"] == "Need the tool." and payload["reasoning_details"][0]["signature"] == "s"


class TestInBandErrors:
    async def test_body_error_raises(self):
        wire = Wire().on("/api/v1/chat/completions", R(200, json={
            "id": "gen-1", "error": {"code": 502, "message": "Upstream timed out",
                                     "metadata": {"provider_name": "Google"}}}))
        with pytest.raises(ProviderError, match="Upstream timed out") as info:
            await _provider(wire).chat_completion(CTX, "google/gemini-3-flash-preview")
        assert info.value.retryable is True

    async def test_stream_error_chunk_raises(self):
        body = sse(_chunk({"role": "assistant", "content": "Par"}),
                   {"id": "gen-1", "object": "chat.completion.chunk", "created": 1, "model": "m",
                    "error": {"code": "server_error", "message": "Provider disconnected"},
                    "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}]})
        wire = Wire().on("/api/v1/chat/completions", R(200, content=body, headers={"content-type": "text/event-stream"}))
        stream = await _provider(wire).chat_completion(CTX, "google/gemini-3-flash-preview", stream=True)
        with pytest.raises(ProviderError, match="Provider disconnected"):
            async for _ in stream:
                pass


class TestRequestShaping:
    async def test_profile_params_search_plugin_and_preferences(self):
        wire = Wire().on("/api/v1/chat/completions", R(200, json=_chat({"content": "ok"})))
        p = _provider(wire, provider_preferences={"zdr": True, "data_collection": "deny"})
        await p.chat_completion(CTX, "openai/gpt-5.4-mini", native_search=True,
                                provider={"order": ["OpenAI"]}, models=["openai/gpt-5.4-nano"],
                                plugins=[{"id": "response-healing"}], session_id="s1", top_k=5)
        body = wire.sent()
        assert body["provider"] == {"zdr": True, "data_collection": "deny", "order": ["OpenAI"]}
        assert body["plugins"] == [{"id": "response-healing"}, {"id": "web"}]
        assert body["models"] == ["openai/gpt-5.4-nano"] and body["session_id"] == "s1" and body["top_k"] == 5

    def test_effort_clamps_to_the_listed_efforts(self):
        p = OpenRouterProvider({"api_key": "k"})
        p._models_cache = [
            {"id": "google/gemini-3-flash-preview", "reasoning": {"supported_efforts": ["high", "medium", "low", "minimal", "none"]}},
            {"id": "openai/gpt-4o-mini", "reasoning": None},
        ]
        assert p._shape_request("google/gemini-3-flash-preview", {"effort": "max"})["effort"] == "high"
        assert "effort" not in p._shape_request("openai/gpt-4o-mini", {"effort": "high"})

    @respx.mock
    async def test_listing_is_paginated_and_includes_every_modality(self):
        route = respx.get(OPENROUTER_MODELS_URL).mock(side_effect=[
            httpx.Response(200, json={"data": [{"id": "a/one", "context_length": 1000}], "total_count": 2}),
            httpx.Response(200, json={"data": [{"id": "a/two", "context_length": 2000,
                                                "architecture": {"output_modalities": ["image"]}}], "total_count": 2}),
        ])
        p = OpenRouterProvider({"api_key": "k"})
        details = await p.get_models_details()
        assert [d.id for d in details] == ["a/one", "a/two"]
        assert route.calls[0].request.url.params["output_modalities"] == "all"
        assert route.calls[1].request.url.params["offset"] == "1"
        assert p.get_max_context_length("a/two") == 2000


class TestSdkBackend:
    async def test_sdk_sends_the_full_body_and_returns_wire_dicts(self):
        pytest.importorskip("openrouter")
        p = OpenRouterProvider({"api_key": "k", "backend": "sdk"})
        result = MagicMock()
        result.model_dump.return_value = _chat({"content": "ok", "reasoning_details": [ENCRYPTED]})
        p._openrouter_client = MagicMock()
        p._openrouter_client.chat.send_async = AsyncMock(return_value=result)
        import inspect

        from openrouter import OpenRouter
        p._openrouter_client.chat.send_async.__signature__ = inspect.signature(OpenRouter(api_key="x").chat.send_async)
        history = CTX + [Message(role=Role.ASSISTANT, content="", tool_calls=[CALL],
                                 metadata={"reasoning_details": [ENCRYPTED]}),
                         Message(role=Role.TOOL, content="22C", tool_call_id="c1")]
        out = await p.chat_completion(history, "google/gemini-3-flash-preview", tools=[WEATHER],
                                      effort="low", provider={"zdr": True})
        sent = p._openrouter_client.chat.send_async.call_args.kwargs
        assert sent["messages"][1]["tool_calls"] == [CALL]
        assert sent["messages"][1]["reasoning_details"] == [ENCRYPTED]
        assert sent["reasoning"] == {"effort": "low"} and sent["provider"] == {"zdr": True}
        assert p.assistant_replay_metadata(out) == {"reasoning_details": [ENCRYPTED]}
        with pytest.raises(ValueError, match="no parameter"):
            await p.chat_completion(CTX, "m", not_a_real_param=1)


def test_openai_itself_never_gets_compat_fields():
    from llmcore.providers.openai_provider import OpenAIProvider

    p = OpenAIProvider({"api_key": "k"})
    payload = p._build_message_payload(
        Message(role=Role.ASSISTANT, content="x", metadata={"reasoning_content": "r", "reasoning_details": [ENCRYPTED]}),
        "gpt-6-luna")
    assert "reasoning_content" not in payload and "reasoning_details" not in payload
    assert json.dumps(payload)  # still a plain message
