# tests/providers/test_kimi_current_api.py
"""KimiProvider against the current Moonshot API (K3 era, 2026-11).

Measured live on 2026-11-16:

- K3 fixes sampling ("invalid temperature: only 1 is allowed for this model");
- K3 depth is ``reasoning_effort`` (low/high/max; "none" turns it off);
  kimi-k2.7-code refuses ``thinking: disabled``;
- a K3 tool loop stops reasoning unless ``reasoning_content`` is replayed;
- usage reports ``prompt_tokens_details.cached_tokens`` / ``cache_write_tokens``;
- ``/v1/responses`` exists; K3 takes dynamic-tool system messages.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("openai")

from llmcore.models import Message, Role, Tool  # noqa: E402
from llmcore.providers.kimi_provider import KimiProvider  # noqa: E402

from .httpx2_wire import R, Wire, sse  # noqa: E402

BASE = "https://api.moonshot.ai/v1"
CTX = [Message(role=Role.USER, content="17*23?")]
WEATHER = Tool(name="get_weather", description="w",
               parameters={"type": "object", "properties": {"city": {"type": "string"}}})


@pytest.fixture(autouse=True)
def _clean_learned_state():
    KimiProvider._REJECTED_PARAMS.clear()
    yield
    KimiProvider._REJECTED_PARAMS.clear()


def _provider(wire: Wire, **cfg: Any) -> KimiProvider:
    p = KimiProvider({"api_key": "k", **cfg})
    p._client = wire.client(BASE)
    return p


def _chat(message: dict[str, Any], finish: str = "stop", usage: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"id": "x", "object": "chat.completion", "created": 1, "model": "kimi-k3",
            "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish}],
            "usage": usage or {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}}


class TestFamilies:
    @pytest.mark.parametrize(
        ("model", "effort", "reasoning_effort", "thinking"),
        [("kimi-k3.1-pro", "medium", "high", None),     # K3: low/high/max
         ("kimi-k3", "minimal", "low", None),
         ("kimi-k3", "none", "none", None),
         ("kimi-k2.6", "none", None, {"type": "disabled"}),
         ("kimi-k2.7-code", "none", None, {"type": "enabled"})],  # always thinks
    )
    async def test_effort_per_family(self, model, effort, reasoning_effort, thinking):
        wire = Wire().on("/v1/chat/completions", R(200, json=_chat({"content": "391"})))
        await _provider(wire).chat_completion(CTX, model, effort=effort)
        body = wire.sent()
        assert body.get("reasoning_effort") == reasoning_effort
        assert body.get("thinking") == thinking

    async def test_k3_strips_fixed_sampling_and_renames_max_tokens(self):
        wire = Wire().on("/v1/chat/completions", R(200, json=_chat({"content": "391"})))
        await _provider(wire).chat_completion(CTX, "kimi-k3", temperature=0.2, top_p=0.5, max_tokens=99)
        body = wire.sent()
        assert "temperature" not in body and "top_p" not in body
        assert body["max_completion_tokens"] == 99 and "max_tokens" not in body

    async def test_a_fixed_value_refusal_is_learned(self):
        refusal = R(400, json={"error": {"message": "invalid presence_penalty: only 0 is allowed for this model",
                                         "type": "invalid_request_error"}})
        wire = Wire().on("/v1/chat/completions", refusal, R(200, json=_chat({"content": "ok"})))
        p = _provider(wire)
        await p.chat_completion(CTX, "kimi-new-model", presence_penalty=0.5)
        assert "presence_penalty" in wire.sent(0) and "presence_penalty" not in wire.sent(1)
        assert KimiProvider._REJECTED_PARAMS == {"kimi-new-model": {"presence_penalty"}}

    def test_dynamic_tool_system_message(self):
        p = KimiProvider({"api_key": "k"})
        payload = p._build_message_payload(Message(role=Role.SYSTEM, content="", metadata={"tools": [WEATHER]}))
        assert payload == {"role": "system", "tools": [{"type": "function", "function": WEATHER.model_dump()}]}

    def test_kimi_code_base_url_reads_its_own_key(self, monkeypatch):
        monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
        monkeypatch.setenv("KIMI_API_KEY", "code-key")
        p = KimiProvider({"base_url": "https://api.kimi.com/coding/v1"})
        assert p._api_key == "code-key"


class TestReplayAndUsage:
    async def test_stream_accumulates_reasoning_for_replay(self):
        def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
            return {"id": "x", "object": "chat.completion.chunk", "created": 1, "model": "kimi-k3",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        body = sse(chunk({"role": "assistant", "reasoning_content": "Use "}), chunk({"reasoning_content": "the tool."}),
                   chunk({"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                                          "function": {"name": "get_weather", "arguments": "{}"}}]}),
                   chunk({}, "tool_calls"))
        wire = Wire().on("/v1/chat/completions", R(200, content=body, headers={"content-type": "text/event-stream"}))
        p = _provider(wire)
        out = [c async for c in await p.chat_completion(CTX, "kimi-k3", tools=[WEATHER], stream=True)]
        finish = next(c for c in out if c["choices"] and c["choices"][0].get("finish_reason"))
        assert p.assistant_replay_metadata(finish) == {"reasoning_content": "Use the tool."}

    async def test_replayed_reasoning_reaches_the_wire(self):
        wire = Wire().on("/v1/chat/completions", R(200, json=_chat({"content": "ok"})))
        call = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
        history = CTX + [Message(role=Role.ASSISTANT, content="", tool_calls=[call],
                                 metadata={"reasoning_content": "Use the tool."}),
                         Message(role=Role.TOOL, content="22C", tool_call_id="c1")]
        await _provider(wire).chat_completion(history, "kimi-k3", tools=[WEATHER])
        assert wire.sent()["messages"][1]["reasoning_content"] == "Use the tool."

    def test_cache_breakdown(self):
        usage = {"prompt_tokens": 300, "completion_tokens": 5, "total_tokens": 305,
                 "prompt_tokens_details": {"cached_tokens": 200, "cache_write_tokens": 64}}
        details = KimiProvider({"api_key": "k"}).extract_usage_details({"usage": usage})
        assert details["cached_tokens"] == 200 and details["cache_write_tokens"] == 64

    async def test_responses_route(self):
        wire = Wire().on("/v1/responses", R(200, json={
            "id": "r1", "object": "response", "created_at": 1, "model": "kimi-k3", "status": "completed",
            "output": [{"type": "reasoning", "id": "rs1", "summary": [{"type": "summary_text", "text": "391"}]},
                       {"type": "message", "id": "m1", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "391", "annotations": []}]}],
            "usage": {"input_tokens": 9, "output_tokens": 4, "total_tokens": 13}}))
        p = _provider(wire, responses_api="always")
        out = await p.chat_completion(CTX, "kimi-k3", effort="low", temperature=0.3)
        body = wire.sent()
        assert body["reasoning"] == {"effort": "low"} and "temperature" not in body
        assert p.extract_response_content(out) == "391"
        assert p.assistant_replay_metadata(out)["reasoning_items"][0]["id"] == "rs1"
