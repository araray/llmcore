# tests/media/test_openai_voice_agent.py
"""OpenAI Realtime voice agent (``voice_agent``), live-verified 2026-11-16.

Live (gpt-realtime-2.1-mini): a typed "weather in Lisbon?" turn gave a spoken
preamble, function_call_request get_weather({"city": "Lisbon"}), and after the
tool result a spoken answer ("Lisbon is currently 22 degrees Celsius with
clear skies...") with ~8 s of 24 kHz audio.
"""

from __future__ import annotations

import base64
import json
from typing import Any
from unittest.mock import patch

import pytest

from llmcore.media.models import MediaCapability, MediaExecution
from llmcore.models_multimodal import VoiceAgentEventType
from llmcore.providers.openai_provider import OpenAIProvider


class FakeSocket:
    def __init__(self, incoming: list[dict[str, Any]]) -> None:
        self.sent: list[dict[str, Any]] = []
        self.incoming = [json.dumps(m) for m in incoming]
        self.closed = False

    async def send(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self):
        async def gen():
            for m in self.incoming:
                yield m
        return gen()


@pytest.fixture
def provider(monkeypatch) -> OpenAIProvider:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return OpenAIProvider({"api_key": "sk-test"})


def test_declared(provider):
    assert MediaCapability.VOICE_AGENT in provider.media_capabilities()
    assert provider.media_execution(MediaCapability.VOICE_AGENT) is MediaExecution.STREAM


async def test_session_setup_tool_loop_and_events(provider):
    pcm = b"\x01\x02" * 4
    socket = FakeSocket([
        {"type": "session.created"},
        {"type": "session.updated"},
        {"type": "response.created", "response": {"id": "r1"}},
        {"type": "response.output_audio.delta", "response_id": "r1", "delta": base64.b64encode(pcm).decode()},
        {"type": "response.output_audio.delta", "response_id": "r1", "delta": base64.b64encode(pcm).decode()},
        {"type": "response.output_audio.done", "response_id": "r1"},
        {"type": "response.output_audio_transcript.done", "response_id": "r1", "transcript": "Let me check."},
        {"type": "response.function_call_arguments.done", "call_id": "c1", "name": "get_weather",
         "arguments": json.dumps({"city": "Lisbon"})},
        {"type": "conversation.item.input_audio_transcription.completed", "transcript": "weather?"},
        {"type": "error", "error": {"message": "boom"}},
    ])
    seen: dict[str, Any] = {}

    async def connect(url: str, additional_headers: dict[str, str]):
        seen.update(url=url, headers=additional_headers)
        return socket

    tool = {"name": "get_weather", "description": "Weather.", "parameters": {"type": "object", "properties": {}}}
    with patch("websockets.asyncio.client.connect", connect):
        async with provider.open_voice_agent(prompt="Be terse.", functions=[tool], voice="marin",
                                             greeting="Hello!") as agent:
            await agent.inject_user_message("Weather in Lisbon?")
            events = [e async for e in agent]
            await agent.respond_to_function_call("c1", {"temp_c": 22})

    assert seen["url"] == "wss://api.openai.com/v1/realtime?model=gpt-realtime-2.1-mini"
    update, greet, item, respond, *tail = socket.sent
    session = update["session"]
    assert session["type"] == "realtime" and session["instructions"] == "Be terse."
    assert session["tools"] == [{"type": "function", "name": "get_weather", "description": "Weather.",
                                 "parameters": {"type": "object", "properties": {}}}]
    assert session["audio"]["output"] == {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "marin"}
    assert session["audio"]["input"]["transcription"] == {"model": "gpt-4o-transcribe"}
    assert greet["type"] == "response.create" and "Hello!" in greet["response"]["instructions"]
    assert item["item"]["content"] == [{"type": "input_text", "text": "Weather in Lisbon?"}]
    assert tail[0]["item"] == {"type": "function_call_output", "call_id": "c1", "output": '{"temp_c": 22}'}
    assert tail[1] == {"type": "response.create"}

    types = [e.type for e in events]
    assert types == [
        VoiceAgentEventType.WELCOME, VoiceAgentEventType.SETTINGS_APPLIED, VoiceAgentEventType.AGENT_THINKING,
        VoiceAgentEventType.AGENT_STARTED_SPEAKING, VoiceAgentEventType.AUDIO, VoiceAgentEventType.AUDIO,
        VoiceAgentEventType.AGENT_AUDIO_DONE, VoiceAgentEventType.CONVERSATION_TEXT,
        VoiceAgentEventType.FUNCTION_CALL_REQUEST, VoiceAgentEventType.CONVERSATION_TEXT, VoiceAgentEventType.ERROR,
    ]
    assert events[4].audio == pcm
    call = events[8].function_call
    assert (call.id, call.name, call.arguments) == ("c1", "get_weather", {"city": "Lisbon"})
    assert events[9].role == "user" and events[-1].content == "boom"
    assert socket.closed
