# tests/providers/test_anthropic_current_api.py
"""AnthropicProvider against the current API (version 2026-10-21, Claude 5.x).

Each behaviour here was measured live on 2026-11-16:

- sampling parameters are rejected on every model under 2026-10-21;
- "no thinking" is spelled per model (disabled / between_tools / neither);
- a forced tool_choice is a 400 on Opus 5.5, Sonnet 5.5 and Fable;
- thinking blocks carry signatures that a tool loop has to send back.

The direct transport is mocked with respx, so the exact wire body is checked.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from llmcore.exceptions import ContextLengthError, ProviderError
from llmcore.models import Message, Role, Tool
from llmcore.providers.anthropic_provider import AnthropicProvider

URL = "https://api.anthropic.com/v1/messages"
CTX = [Message(role=Role.USER, content="hi")]
WEATHER = Tool(name="get_weather", description="w",
               parameters={"type": "object", "properties": {"city": {"type": "string"}}})


@pytest.fixture(autouse=True)
def _clean_class_state():
    AnthropicProvider._LIVE_CAPABILITIES.clear()
    AnthropicProvider._NO_FORCED_TOOL_CHOICE.clear()
    AnthropicProvider._warned_sampling = False
    yield
    AnthropicProvider._LIVE_CAPABILITIES.clear()
    AnthropicProvider._NO_FORCED_TOOL_CHOICE.clear()


def _provider(**cfg: Any) -> AnthropicProvider:
    return AnthropicProvider({"api_key": "k", "backend": "httpx", "capability_lookup": False, **cfg})


def _message(*blocks: dict[str, Any], stop: str = "end_turn") -> dict[str, Any]:
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "m",
            "content": list(blocks) or [{"type": "text", "text": "ok"}], "stop_reason": stop,
            "usage": {"input_tokens": 3, "output_tokens": 2}}


def _sent(route: respx.Route, i: int = -1) -> dict[str, Any]:
    return json.loads(route.calls[i].request.content)


class TestRequestShape:
    @respx.mock
    async def test_api_version_and_sampling_dropped(self):
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=_message()))
        await _provider().chat_completion(CTX, "claude-haiku-5-5", temperature=0.3, top_p=0.5)
        body = _sent(route)
        assert route.calls[-1].request.headers["anthropic-version"] == "2026-10-21"
        assert "temperature" not in body and "top_p" not in body

    @pytest.mark.parametrize(
        ("model", "thinking", "effort"),
        [("claude-haiku-5-5", {"type": "disabled"}, None),
         ("claude-sonnet-5-5", {"type": "between_tools"}, None),
         ("claude-opus-5-5", None, "low")],  # no off switch: lowest effort
    )
    @respx.mock
    async def test_effort_none_per_model(self, model, thinking, effort):
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=_message()))
        await _provider().chat_completion(CTX, model, effort="none")
        body = _sent(route)
        assert body.get("thinking") == thinking
        assert (body.get("output_config") or {}).get("effort") == effort

    @respx.mock
    async def test_live_capabilities_override_the_snapshot(self):
        respx.get("https://api.anthropic.com/v1/models").mock(return_value=httpx.Response(200, json={
            "data": [{"id": "claude-new-9", "max_input_tokens": 2_000_000, "max_tokens": 256_000,
                      "capabilities": {
                          "thinking": {"supported": True, "types": {
                              "adaptive": {"supported": True}, "disabled": {"supported": True}}},
                          "effort": {"supported": True, "low": {"supported": True}}}}]}))
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=_message()))
        p = AnthropicProvider({"api_key": "k", "backend": "httpx"})
        await p.chat_completion(CTX, "claude-new-9", effort="high")
        # Only "low" is accepted, so "high" moves to it; the window is the live one.
        assert _sent(route)["output_config"] == {"effort": "low"}
        assert p.get_max_context_length("claude-new-9") == 2_000_000

    @respx.mock
    async def test_thinking_display(self):
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=_message()))
        await _provider().chat_completion(CTX, "claude-opus-5-5", effort="high", thinking_display="summarized")
        assert _sent(route)["thinking"] == {"type": "adaptive", "display": "summarized"}

    @respx.mock
    async def test_native_search_and_extra_tools(self):
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=_message()))
        await _provider().chat_completion(
            CTX, "claude-haiku-5-5", tools=[WEATHER], native_search=True,
            extra_tools=[{"type": "web_fetch_20250910", "name": "web_fetch"}])
        types = [t.get("type", "custom") for t in _sent(route)["tools"]]
        assert types == ["custom", "web_fetch_20250910", "web_search_20250305"]


class TestForcedToolChoice:
    @respx.mock
    async def test_rejected_forced_choice_becomes_auto_and_is_remembered(self):
        rejection = httpx.Response(400, json={"type": "error", "error": {
            "type": "invalid_request_error",
            "message": 'tool_choice: type "tool" and "any" are not supported for this model.'}})
        route = respx.post(URL).mock(side_effect=[rejection, httpx.Response(200, json=_message()),
                                                  httpx.Response(200, json=_message())])
        p = _provider()
        await p.chat_completion(CTX, "claude-opus-5-5", tools=[WEATHER], tool_choice="get_weather")
        assert _sent(route, 0)["tool_choice"] == {"type": "tool", "name": "get_weather"}
        assert _sent(route, 1)["tool_choice"] == {"type": "auto"}
        await p.chat_completion(CTX, "claude-opus-5-5", tools=[WEATHER], tool_choice="any")
        assert route.call_count == 3 and _sent(route, 2)["tool_choice"] == {"type": "auto"}


class TestThinkingRoundTrip:
    THINKING = {"type": "thinking", "thinking": "", "signature": "sig=="}
    TOOL_USE = {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Lisbon"}}

    @respx.mock
    async def test_raw_blocks_are_kept_and_replayed_verbatim(self):
        route = respx.post(URL).mock(side_effect=[
            httpx.Response(200, json=_message(self.THINKING, self.TOOL_USE, stop="tool_use")),
            httpx.Response(200, json=_message()),
        ])
        p = _provider()
        first = await p.chat_completion(CTX, "claude-sonnet-5-5", tools=[WEATHER])
        meta = p.assistant_replay_metadata(first)
        assert meta == {"anthropic_content": [self.THINKING, self.TOOL_USE]}
        history = CTX + [
            Message(role=Role.ASSISTANT, content="", metadata=meta,
                    tool_calls=first["choices"][0]["message"]["tool_calls"]),
            Message(role=Role.TOOL, content="22C", tool_call_id="toolu_1"),
        ]
        await p.chat_completion(history, "claude-sonnet-5-5", tools=[WEATHER])
        assistant = _sent(route, 1)["messages"][1]
        assert assistant == {"role": "assistant", "content": [self.THINKING, self.TOOL_USE]}

    @respx.mock
    async def test_stream_final_chunk_carries_the_blocks(self):
        events = [
            {"type": "message_start", "message": {"id": "m", "usage": {"input_tokens": 1}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig=="}},
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"city\": \"Lisbon\"}"}},
            {"type": "content_block_stop", "index": 1},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 5}},
            {"type": "message_stop"},
        ]
        sse = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
        respx.post(URL).mock(return_value=httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"}))
        p = _provider()
        chunks = [c async for c in await p.chat_completion(CTX, "claude-sonnet-5-5", tools=[WEATHER], stream=True)]
        blocks = p.assistant_replay_metadata(chunks[-1])["anthropic_content"]
        assert blocks[0] == {"type": "thinking", "thinking": "hmm", "signature": "sig=="}
        assert blocks[1]["input"] == {"city": "Lisbon"}

    def test_plain_text_answers_need_no_replay_data(self):
        normalized = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        assert _provider().assistant_replay_metadata(normalized) == {}


class TestErrorsAndRules:
    @respx.mock
    async def test_only_prompt_too_long_is_a_context_error(self):
        respx.post(URL).mock(side_effect=[
            httpx.Response(400, json={"type": "error", "error": {"type": "invalid_request_error",
                                      "message": "prompt is too long: 1200000 tokens > 1000000 maximum"}}),
            httpx.Response(400, json={"type": "error", "error": {"type": "invalid_request_error",
                                      "message": "max_tokens: 999999 > 128000, the maximum allowed"}}),
        ])
        p = _provider()
        with pytest.raises(ContextLengthError):
            await p.chat_completion(CTX, "claude-haiku-5-5")
        with pytest.raises(ProviderError) as info:
            await p.chat_completion(CTX, "claude-haiku-5-5", max_tokens=999999)
        assert not isinstance(info.value, ContextLengthError)

    def test_api_2026_10_21_role_rules(self):
        doc = {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "x"}}
        assert AnthropicProvider._fit_blocks_to_role([{"type": "text", "text": "a"}, doc], "assistant") == [
            {"type": "text", "text": "a"}]
        cited = {"type": "text", "text": "a", "citations": [{"type": "char_location"}]}
        assert AnthropicProvider._fit_blocks_to_role([cited], "user") == [{"type": "text", "text": "a"}]

    def test_window_exceeded_stop_is_length(self):
        assert AnthropicProvider._normalize_stop_reason("model_context_window_exceeded") == "length"

    @pytest.mark.parametrize(
        ("model", "generation"),
        [("claude-sonnet-4-20250514", (4, 0)), ("claude-opus-5", (5, 0)),
         ("claude-opus-4-5-20251101", (4, 5)), ("claude-sonnet-5-5", (5, 5))],
    )
    def test_generation_parsing(self, model, generation):
        assert AnthropicProvider._model_generation(model) == generation
