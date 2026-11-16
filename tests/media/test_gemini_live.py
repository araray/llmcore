# tests/media/test_gemini_live.py
"""Gemini Live API sessions: ``asr_stream`` and ``voice_agent`` (2026-11-16).

Live: gemini-3.5-transcribe-live streamed interim results and the exact
sentence; gemini-3.8-live ran get_weather({"city": "Lisbon"}) and answered
"The weather in Lisbon is clear and 22 degrees Celsius." with 24 kHz audio.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace as NS
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("google.genai")

from llmcore.exceptions import ProviderError  # noqa: E402
from llmcore.media.models import MediaCapability, MediaExecution  # noqa: E402
from llmcore.models_multimodal import StreamEventType, VoiceAgentEventType  # noqa: E402
from llmcore.providers.gemini_provider import GeminiProvider  # noqa: E402


class FakeSession:
    """receive() yields one turn per call, then nothing once exhausted."""

    def __init__(self, turns: list[list[Any]]) -> None:
        self.turns = list(turns)
        self.sent: list[tuple[str, dict[str, Any]]] = []

    async def send_realtime_input(self, **kwargs: Any) -> None:
        self.sent.append(("realtime", kwargs))

    async def send_client_content(self, **kwargs: Any) -> None:
        self.sent.append(("client", kwargs))

    async def send_tool_response(self, **kwargs: Any) -> None:
        self.sent.append(("tool", kwargs))

    async def receive(self):
        turn = self.turns.pop(0) if self.turns else []
        for message in turn:
            yield message


def _content(**fields: Any) -> NS:
    base = {k: None for k in ("model_turn", "turn_complete", "interrupted", "input_transcription",
                              "output_transcription", "interim_input_transcription")}
    return NS(**{**base, **fields})


def _message(**fields: Any) -> NS:
    base = {"setup_complete": None, "tool_call": None, "server_content": None}
    msg = NS(**{**base, **fields})
    msg.model_dump = lambda **_: {}
    return msg


def _provider(session: FakeSession, seen: dict[str, Any]) -> GeminiProvider:
    p = GeminiProvider({"api_key": "k"})

    @asynccontextmanager
    async def connect(model: str, config: Any):
        seen.update(model=model, config=config)
        yield session

    p._client = MagicMock()
    p._client.aio.live.connect = connect
    return p


def test_declared_on_the_developer_api():
    p = GeminiProvider({"api_key": "k"})
    assert {MediaCapability.ASR_STREAM, MediaCapability.VOICE_AGENT} <= p.media_capabilities()
    assert p.media_execution(MediaCapability.VOICE_AGENT) is MediaExecution.STREAM


async def test_live_transcription():
    session = FakeSession([[
        _message(server_content=_content(interim_input_transcription=NS(text="Gemini"))),
        _message(server_content=_content(input_transcription=NS(text=" Gemini Live transcribes this."))),
    ]])
    seen: dict[str, Any] = {}
    p = _provider(session, seen)
    async with p.open_transcription_session(sample_rate_hz=16000) as s:
        await s.send_audio(b"\x00\x01")
        await s.finalize()
        events = [e async for e in s]
    assert seen["model"] == "gemini-3.5-transcribe-live"
    blob = session.sent[0][1]["audio"]
    assert blob.mime_type == "audio/pcm;rate=16000" and session.sent[1] == ("realtime", {"audio_stream_end": True})
    assert [(e.type, e.text) for e in events] == [
        (StreamEventType.INTERIM, "Gemini"), (StreamEventType.FINAL, "Gemini Live transcribes this.")]


async def test_voice_agent_tool_loop():
    call = NS(id="c1", name="get_weather", args={"city": "Lisbon"})
    session = FakeSession([
        [_message(tool_call=NS(function_calls=[call]))],
        [_message(server_content=_content(model_turn=NS(parts=[NS(inline_data=NS(data=b"pcm"))]))),
         _message(server_content=_content(output_transcription=NS(text="Clear, 22 C."))),
         _message(server_content=_content(turn_complete=True))],
    ])
    seen: dict[str, Any] = {}
    p = _provider(session, seen)
    tool = {"name": "get_weather", "description": "Weather.", "parameters": {"type": "object", "properties": {}}}
    async with p.open_voice_agent(prompt="Be terse.", functions=[tool], voice="Kore") as agent:
        await agent.inject_user_message("Weather in Lisbon?")
        events = []
        async for e in agent:
            events.append(e)
            if e.type is VoiceAgentEventType.FUNCTION_CALL_REQUEST:
                await agent.respond_to_function_call(e.function_call.id, {"temp_c": 22})
        with pytest.raises(ProviderError, match="system instruction"):
            await agent.update_prompt("new")

    config = seen["config"]
    assert seen["model"] == "gemini-3.8-live" and config.system_instruction == "Be terse."
    assert config.tools[0].function_declarations[0].name == "get_weather"
    assert [e.type for e in events] == [
        VoiceAgentEventType.FUNCTION_CALL_REQUEST, VoiceAgentEventType.AGENT_STARTED_SPEAKING,
        VoiceAgentEventType.AUDIO, VoiceAgentEventType.AGENT_AUDIO_DONE, VoiceAgentEventType.CONVERSATION_TEXT]
    assert events[-1].content == "Clear, 22 C."
    tool_reply = [kw for kind, kw in session.sent if kind == "tool"][0]["function_responses"][0]
    assert (tool_reply.id, tool_reply.name, tool_reply.response) == ("c1", "get_weather", {"temp_c": 22})
