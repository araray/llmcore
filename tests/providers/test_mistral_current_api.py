# tests/providers/test_mistral_current_api.py
"""MistralProvider against the current API (mistralai 3.7, 2026-11).

Measured live on 2026-11-16, or taken from docs.mistral.ai where the test key
had no quota (Small/Medium answered 429 with a 0 req/min limit):

- reasoning answers carry ``content`` as thinking + text chunks, and the docs
  require the whole turn (thinking chunk included) to be replayed;
- ``reasoning_effort`` values differ per model ("none"/"high" on Small 4);
- an unsupported parameter is a 400 "<name> support is not enabled for this
  model" (code 3051);
- ``/v1/responses`` exists and is stateless (``store`` must be false);
- TTS needs a voice or ref_audio ("Either ref_audio or voice must be provided").
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx

from llmcore.exceptions import ProviderError
from llmcore.models import Message, Role, Tool
from llmcore.providers.mistral_provider import MistralProvider

BASE = "https://api.mistral.ai/v1"
CTX = [Message(role=Role.USER, content="17*23?")]
THINKING = {"type": "thinking", "thinking": [{"type": "text", "text": "17*23 = 391"}]}
WEATHER = Tool(name="get_weather", description="w",
               parameters={"type": "object", "properties": {"city": {"type": "string"}}})


@pytest.fixture(autouse=True)
def _clean_learned_state():
    MistralProvider._REJECTED_PARAMS.clear()
    yield
    MistralProvider._REJECTED_PARAMS.clear()


def _provider(**cfg: Any) -> MistralProvider:
    return MistralProvider({"api_key": "k", "backend": "httpx", **cfg})


def _chat(content: Any, finish: str = "stop") -> dict[str, Any]:
    return {"id": "x", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": finish}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}}


def _sent(route: respx.Route, i: int = -1) -> dict[str, Any]:
    return json.loads(route.calls[i].request.content)


def _sse(*events: dict[str, Any]) -> bytes:
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode() + b"data: [DONE]\n\n"


class TestReasoning:
    @respx.mock
    async def test_chunked_answer_is_flattened_and_replayed_verbatim(self):
        chunks = [THINKING, {"type": "text", "text": "391"}]
        route = respx.post(f"{BASE}/chat/completions").mock(side_effect=[
            httpx.Response(200, json=_chat(chunks)), httpx.Response(200, json=_chat("ok"))])
        p = _provider()
        first = await p.chat_completion(CTX, "mistral-small-latest", effort="high")
        assert p.extract_response_content(first) == "391"
        assert p.extract_reasoning_content(first) == "17*23 = 391"
        meta = p.assistant_replay_metadata(first)
        assert meta == {"mistral_content": chunks}
        history = CTX + [Message(role=Role.ASSISTANT, content="391", metadata=meta),
                         Message(role=Role.USER, content="and doubled?")]
        await p.chat_completion(history, "mistral-small-latest")
        assert _sent(route)["messages"][1] == {"role": "assistant", "content": chunks}

    @respx.mock
    async def test_stream_deltas_flatten_and_the_last_chunk_carries_the_turn(self):
        events = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": [
                {"type": "thinking", "thinking": [{"type": "text", "text": "17*23"}]}]}}]},
            {"choices": [{"index": 0, "delta": {"content": [
                {"type": "thinking", "thinking": [{"type": "text", "text": " = 391"}], "closed": True},
                {"type": "text", "text": "39"}]}}]},
            {"choices": [{"index": 0, "delta": {"content": "1"}, "finish_reason": "stop"}]},
        ]
        respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(
            200, content=_sse(*events), headers={"content-type": "text/event-stream"}))
        p = _provider()
        out = [c async for c in await p.chat_completion(CTX, "mistral-small-latest", stream=True)]
        assert "".join(p.extract_delta_content(c) for c in out) == "391"
        assert "".join(p.extract_delta_reasoning_content(c) or "" for c in out) == "17*23 = 391"
        assert p.assistant_replay_metadata(out[-1])["mistral_content"] == [
            {"type": "thinking", "thinking": [{"type": "text", "text": "17*23 = 391"}]},
            {"type": "text", "text": "391"}]

    @pytest.mark.parametrize(
        ("model", "effort", "sent"),
        [("mistral-small-latest", "medium", "high"),   # Small 4: none/high only
         ("mistral-small-latest", "minimal", "none"),
         ("mistral-large-4", "max", "high"),            # Large 4: low/medium/high
         ("ministral-8b-latest", "high", None)],        # no reasoning: not sent
    )
    @respx.mock
    async def test_effort_maps_to_each_models_values(self, model, effort, sent):
        route = respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(200, json=_chat("ok")))
        await _provider().chat_completion(CTX, model, effort=effort)
        assert _sent(route).get("reasoning_effort") == sent

    @respx.mock
    async def test_live_capabilities_decide_reasoning(self):
        respx.get(f"{BASE}/models").mock(return_value=httpx.Response(200, json={"data": [
            {"id": "new-model-1", "aliases": ["new-model-latest"], "max_context_length": 99999,
             "capabilities": {"completion_chat": True, "reasoning": True}}]}))
        route = respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(200, json=_chat("ok")))
        p = _provider()
        await p.get_models_details()
        await p.chat_completion(CTX, "new-model-latest", effort="high")
        assert _sent(route)["reasoning_effort"] == "high"
        assert p.get_max_context_length("new-model-latest") == 99999


class TestErrorsAndLearning:
    @respx.mock
    async def test_rejected_parameter_is_dropped_retried_and_remembered(self):
        rejection = httpx.Response(400, json={
            "object": "error", "message": "min_tokens support is not enabled for this model",
            "type": "invalid_request_invalid_args", "code": "3051"})
        route = respx.post(f"{BASE}/chat/completions").mock(side_effect=[
            rejection, httpx.Response(200, json=_chat("ok")), httpx.Response(200, json=_chat("ok"))])
        p = _provider()
        await p.chat_completion(CTX, "ministral-3b-latest", min_tokens=1, top_k=5)
        assert "min_tokens" in _sent(route, 0) and "min_tokens" not in _sent(route, 1)
        assert _sent(route, 1)["top_k"] == 5
        await p.chat_completion(CTX, "ministral-3b-latest", min_tokens=1)
        assert route.call_count == 3 and "min_tokens" not in _sent(route, 2)

    @pytest.mark.parametrize("wording", [
        "min_tokens support is not enabled for this model",
        "top_k sampling is not enabled for this model",
        "reasoning_effort is not enabled for this model",
    ])
    def test_every_measured_rejection_wording_is_learned(self, wording):
        body = {"model": "m", "messages": [], "min_tokens": 1, "top_k": 5, "reasoning_effort": "high"}
        name = wording.split()[0]
        assert _provider()._learn_rejection(ProviderError("mistral", f"API Error (400): {wording}"), body, "m")
        assert name not in body

    async def test_sdk_errors_map_like_rest_errors(self):
        p = _provider()
        sdk_error = type("SDKError", (Exception,), {})("boom")
        sdk_error.raw_response = httpx.Response(429, headers={"x-ratelimit-limit-req-minute": "0"},
                                                json={"message": "Rate limit exceeded"})
        mapped = p._sdk_error(sdk_error, "mistral-small-latest")
        assert "no quota" in str(mapped) and mapped.retryable is False

    @respx.mock
    async def test_zero_quota_429_is_explained_and_not_retryable(self):
        respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(
            429, headers={"x-ratelimit-limit-req-minute": "0"},
            json={"object": "error", "message": "Rate limit exceeded", "code": "1300"}))
        with pytest.raises(ProviderError) as info:
            await _provider().chat_completion(CTX, "mistral-medium-latest")
        assert "no quota" in str(info.value) and info.value.retryable is False

    @respx.mock
    async def test_stream_errors_carry_the_servers_message(self):
        respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(
            400, json={"object": "error", "message": "Invalid tool schema"}))
        with pytest.raises(ProviderError, match="Invalid tool schema"):
            await _provider().chat_completion(CTX, "ministral-3b-latest", stream=True)

    @respx.mock
    async def test_fim_stream_is_an_iterator(self):
        respx.post(f"{BASE}/fim/completions").mock(return_value=httpx.Response(
            200, content=_sse({"choices": [{"index": 0, "delta": {"content": "pass"}}]}),
            headers={"content-type": "text/event-stream"}))
        stream = await _provider().fim_completion("def f():\n    ", stream=True)
        assert [c["choices"][0]["delta"]["content"] async for c in stream] == ["pass"]


class TestResponsesLane:
    @respx.mock
    async def test_tool_call_round_trip(self):
        call = {"type": "function_call", "id": "fc_1", "call_id": "c1", "name": "get_weather",
                "arguments": "{\"city\": \"Lisbon\"}", "status": "completed"}
        text = {"type": "message", "id": "m1", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": "22C", "annotations": []}]}
        usage = {"input_tokens": 9, "output_tokens": 4, "total_tokens": 13}
        route = respx.post(f"{BASE}/responses").mock(side_effect=[
            httpx.Response(200, json={"id": "r1", "object": "response", "status": "completed",
                                      "output": [call], "usage": usage}),
            httpx.Response(200, json={"id": "r2", "object": "response", "status": "completed",
                                      "output": [text], "usage": usage}),
        ])
        p = _provider(responses_api="always")
        first = await p.chat_completion(CTX, "ministral-8b-latest", tools=[WEATHER])
        assert _sent(route, 0)["store"] is False
        calls = first["choices"][0]["message"]["tool_calls"]
        assert calls[0]["function"]["name"] == "get_weather"
        history = CTX + [Message(role=Role.ASSISTANT, content="", tool_calls=calls),
                         Message(role=Role.TOOL, content="22C", tool_call_id="c1")]
        second = await p.chat_completion(history, "ministral-8b-latest", tools=[WEATHER])
        assert _sent(route, 1)["input"][-1] == {"type": "function_call_output", "call_id": "c1", "output": "22C"}
        assert p.extract_response_content(second) == "22C"

    def test_chat_only_parameters_are_refused(self):
        with pytest.raises(ValueError, match="safe_prompt"):
            import asyncio
            asyncio.run(_provider(responses_api="always").chat_completion(CTX, "ministral-8b-latest", safe_prompt=True))


class TestMediaAndSdk:
    @respx.mock
    async def test_tts_needs_a_voice_and_passes_new_options(self):
        p = _provider()
        with pytest.raises(ProviderError, match="voice="):
            await p.generate_speech("hi", model="voxtral-mini-tts-3")
        route = respx.post(f"{BASE}/audio/speech").mock(return_value=httpx.Response(200, json={"audio_data": ""}))
        await p.generate_speech("hi", voice="v1", model="voxtral-mini-tts-3", expressiveness=0.7)
        assert _sent(route) == {"input": "hi", "response_format": "mp3", "model": "voxtral-mini-tts-3",
                                "voice_id": "v1", "expressiveness": 0.7}
        cloned = await p.generate_speech("hi", model="voxtral-mini-tts-latest", ref_audio="UklGRg==")
        assert cloned.voice == "ref_audio" and _sent(route)["ref_audio"] == "UklGRg=="

    @respx.mock
    async def test_stt_forwards_extra_fields(self, tmp_path):
        audio = tmp_path / "clip.mp3"
        audio.write_bytes(b"ID3")
        route = respx.post(f"{BASE}/audio/transcriptions").mock(
            return_value=httpx.Response(200, json={"text": "hi", "language": "en"}))
        await _provider().transcribe_audio(str(audio), diarize=True, speaker_references=["ref1"])
        body = route.calls[-1].request.content
        assert b'name="speaker_references"' in body and b"audio/mpeg" in body

    async def test_sdk_embeddings_keep_their_options(self):
        p = _provider()
        p._backend = "sdk"
        p._sdk = MagicMock()
        p._sdk.embeddings.create_async = AsyncMock(return_value={"data": []})
        await p.create_embeddings(["a"], model="codestral-embed", output_dimension=256)
        kwargs = p._sdk.embeddings.create_async.call_args.kwargs
        assert kwargs["output_dimension"] == 256 and kwargs["inputs"] == ["a"]
