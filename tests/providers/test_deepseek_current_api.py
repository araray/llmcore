# tests/providers/test_deepseek_current_api.py
"""DeepSeekProvider against the current API (2026-11).

The test account had no balance (every chat call answered 402), so these
follow api-docs.deepseek.com as of 2026-11-16 plus the live GET /models:

- efforts are low/high/max and llmcore's levels map as DeepSeek documents;
- with tools, every earlier assistant turn's reasoning_content must be sent
  back (400 otherwise), so streams have to accumulate it;
- deepseek-flash (V4.1-Flash) has built-in deepseek_web_search /
  deepseek_read_url tools and a safety_check parameter;
- a native /responses endpoint exists.

The real AsyncOpenAI client is used over an httpx2 MockTransport (openai 3.x
speaks httpx2, which respx cannot intercept), so the exact wire body is checked.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("openai")
from .httpx2_wire import R, Wire  # noqa: E402

from llmcore.exceptions import ProviderError  # noqa: E402
from llmcore.models import Message, Role, Tool  # noqa: E402
from llmcore.providers.deepseek_provider import DeepSeekProvider  # noqa: E402

BASE = "https://api.deepseek.com"
CTX = [Message(role=Role.USER, content="Weather in Lisbon?")]
WEATHER = Tool(name="get_weather", description="w",
               parameters={"type": "object", "properties": {"city": {"type": "string"}}})


def _provider(wire: Wire, **cfg: Any) -> DeepSeekProvider:
    p = DeepSeekProvider({"api_key": "k", **cfg})
    p._client = wire.client(BASE)
    return p


def _chat(message: dict[str, Any], finish: str = "stop") -> dict[str, Any]:
    return {"id": "x", "object": "chat.completion", "created": 1, "model": "deepseek-flash",
            "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}}


class TestRequests:
    @pytest.mark.parametrize(
        ("effort", "thinking", "sent_effort"),
        [("minimal", "enabled", "low"), ("low", "enabled", "low"), ("medium", "enabled", "high"),
         ("xhigh", "enabled", "high"), ("max", "enabled", "max"), ("none", "disabled", None)],
    )
    async def test_effort_follows_deepseeks_mapping(self, effort, thinking, sent_effort):
        wire = Wire().on("/chat/completions", R(200, json=_chat({"content": "ok"})))
        await _provider(wire).chat_completion(CTX, "deepseek-flash", effort=effort)
        body = wire.sent()
        assert body["thinking"] == {"type": thinking}
        assert body.get("reasoning_effort") == sent_effort

    async def test_builtin_tools_and_safety_check(self):
        wire = Wire().on("/chat/completions", R(200, json=_chat({"content": "ok"})))
        p = _provider(wire)
        assert p.supports_native_search("deepseek-flash") and not p.supports_native_search("deepseek-v4-pro")
        await p.chat_completion(CTX, "deepseek-flash", tools=[WEATHER], native_search=True,
                                extra_tools=[{"type": "deepseek_read_url"}], safety_check={"level": "high"})
        body = wire.sent()
        assert [t["type"] for t in body["tools"]] == ["function", "deepseek_read_url", "deepseek_web_search"]
        assert body["safety_check"] == {"level": "high"}


class TestReasoningReplay:
    async def test_non_stream_reasoning_is_replayed_on_the_next_turn(self):
        call = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Lisbon\"}"}}
        wire = Wire().on("/chat/completions", *[
            R(200, json=_chat({"content": "", "reasoning_content": "Need the tool.", "tool_calls": [call]}, "tool_calls")),
            R(200, json=_chat({"content": "22C"})),
        ])
        p = _provider(wire)
        first = await p.chat_completion(CTX, "deepseek-flash", tools=[WEATHER])
        meta = p.assistant_replay_metadata(first)
        assert meta == {"reasoning_content": "Need the tool."}
        history = CTX + [Message(role=Role.ASSISTANT, content="", tool_calls=[call], metadata=meta),
                         Message(role=Role.TOOL, content="22C", tool_call_id="c1")]
        await p.chat_completion(history, "deepseek-flash", tools=[WEATHER])
        assert wire.sent()["messages"][1]["reasoning_content"] == "Need the tool."

    async def test_stream_accumulates_reasoning_for_replay(self):
        def chunk(delta: dict[str, Any], finish: str | None = None) -> str:
            return "data: " + json.dumps({"id": "x", "object": "chat.completion.chunk", "created": 1,
                                          "model": "deepseek-flash",
                                          "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}) + "\n\n"
        sse = (chunk({"role": "assistant", "reasoning_content": "Need "}) + chunk({"reasoning_content": "the tool."})
               + chunk({"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                                        "function": {"name": "get_weather", "arguments": "{}"}}]})
               + chunk({}, "tool_calls") + "data: [DONE]\n\n")
        wire = Wire().on("/chat/completions", R(
            200, content=sse.encode(), headers={"content-type": "text/event-stream"}))
        p = _provider(wire)
        out = [c async for c in await p.chat_completion(CTX, "deepseek-flash", tools=[WEATHER], stream=True)]
        finish = next(c for c in out if c["choices"] and c["choices"][0].get("finish_reason"))
        assert p.assistant_replay_metadata(finish) == {"reasoning_content": "Need the tool."}
        # Deltas are not duplicated by the replay copy.
        assert "".join(p.extract_delta_reasoning_content(c) or "" for c in out) == "Need the tool."


class TestErrorsDiscoveryResponses:
    async def test_402_is_not_retryable(self):
        wire = Wire().on("/chat/completions", R(402, json={
            "error": {"message": "Insufficient Balance", "type": "insufficient_balance_error"}}))
        with pytest.raises(ProviderError, match="Insufficient balance") as info:
            await _provider(wire).chat_completion(CTX, "deepseek-flash")
        assert info.value.retryable is False

    async def test_missing_reasoning_400_explains_the_fix(self):
        wire = Wire().on("/chat/completions", R(400, json={
            "error": {"message": "The reasoning_content in the thinking mode must be passed back to the API.",
                      "type": "invalid_request_error"}}))
        with pytest.raises(ProviderError, match="assistant_replay_metadata"):
            await _provider(wire).chat_completion(CTX, "deepseek-flash")

    async def test_discovery_reads_context_vision_and_efforts(self):
        wire = Wire().on("/models", R(200, json={"object": "list", "data": [
            {"id": "deepseek-flash", "object": "model", "owned_by": "deepseek", "name": "DeepSeek-V4.1-Flash",
             "context_window": 1048576, "max_output_tokens": 393216, "input_modalities": ["text", "image"],
             "effort": {"supported_levels": ["low", "high", "max"], "default_level": "high"}}]}))
        p = _provider(wire)
        (details,) = await p.get_models_details()
        assert details.supports_vision and details.context_length == 1048576
        assert details.metadata["effort_levels"] == ["low", "high", "max"]

    async def test_responses_route(self):
        wire = Wire().on("/responses", R(200, json={
            "id": "r1", "object": "response", "created_at": 1, "model": "deepseek-flash", "status": "completed",
            "output": [{"type": "reasoning", "id": "rs1", "summary": [], "encrypted_content": "enc"},
                       {"type": "message", "id": "m1", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "hi", "annotations": []}]}],
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}}))
        p = _provider(wire, responses_api="always")
        out = await p.chat_completion(CTX, "deepseek-flash", effort="max", native_search=True)
        body = wire.sent()
        assert body["reasoning"] == {"effort": "max"} and body["store"] is False
        assert {"type": "deepseek_web_search"} in body["tools"]
        assert p.extract_response_content(out) == "hi"
        assert p.assistant_replay_metadata(out)["reasoning_items"][0]["encrypted_content"] == "enc"
