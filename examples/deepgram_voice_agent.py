# examples/deepgram_voice_agent.py
"""Bidirectional Voice Agent (STT -> LLM -> TTS over one socket) with Deepgram.

The Voice Agent runs the whole conversational loop server-side. This example is
self-contained: it synthesizes a user utterance as linear16 PCM and feeds it to
the agent as "microphone" audio, registers a client-side ``get_weather`` tool
that the agent can call, and prints every event (including the agent's TTS audio
frames, which it writes to a file).

Two entry points are shown:
* ``run_voice_agent`` — the high-level driver (auto-answers function calls).
* ``open_voice_agent`` — the low-level session (full manual control), in a
  commented-out block at the bottom.

Notes:
* Set DEEPGRAM_API_KEY. The ``think`` (LLM) leg defaults to ``open_ai`` /
  ``gpt-4o-mini``; depending on your Deepgram account you may need to configure
  the LLM provider/endpoint (see the ``settings`` override below). No system
  prompt is hardcoded — supply ``prompt=`` explicitly.

Run:
    pip install "llmcore[deepgram]"
    export DEEPGRAM_API_KEY="dg_..."
    python examples/deepgram_voice_agent.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from llmcore.models_multimodal import VoiceAgentEventType, VoiceAgentFunctionCall
from llmcore.providers.deepgram_provider import DeepgramProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000
FRAME_BYTES = 4800  # ~100 ms of 24 kHz mono linear16

# A Deepgram-format function definition the agent may call.
WEATHER_FUNCTION = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
}


async def _weather_handler(call: VoiceAgentFunctionCall) -> str:
    """Client-side tool: return a (fake) weather report as a JSON string."""
    city = call.arguments.get("city", "your area")
    logger.info("Tool call: get_weather(city=%r)", city)
    return json.dumps({"city": city, "temperature_f": 72, "conditions": "sunny"})


async def _mic_audio(provider: DeepgramProvider) -> AsyncIterator[bytes]:
    """Synthesize a user utterance and yield it as linear16 PCM frames."""
    speech = await provider.generate_speech(
        "Hi! What is the weather like in San Francisco right now?",
        voice="aura-2-thalia-en",
        response_format="linear16",
        sample_rate=SAMPLE_RATE,
    )
    pcm = speech.audio_data
    for i in range(0, len(pcm), FRAME_BYTES):
        yield pcm[i : i + FRAME_BYTES]
        await asyncio.sleep(0.05)


async def main() -> None:
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        logger.error("Set DEEPGRAM_API_KEY to run this example.")
        return

    # Full [providers.deepgram] defaults (incl. the agent audio/listen/think/
    # speak shape) ship in default_config.toml; here we pass a minimal agent
    # config inline so the example is self-contained.
    provider = DeepgramProvider(
        {
            "api_key": api_key,
            "_instance_name": "deepgram",
            "agent": {
                "audio": {
                    "input": {"encoding": "linear16", "sample_rate": SAMPLE_RATE},
                    "output": {"encoding": "linear16", "sample_rate": SAMPLE_RATE},
                },
                "listen": {"provider": {"type": "deepgram", "model": "nova-3"}},
                "think": {"provider": {"type": "open_ai", "model": "gpt-4o-mini"}},
                "speak": {"provider": {"type": "deepgram", "model": "aura-2-thalia-en"}},
            },
        }
    )

    agent_audio = bytearray()

    async def _on_event(event: Any) -> None:
        if event.type == VoiceAgentEventType.CONVERSATION_TEXT:
            print(f"[{event.role}] {event.content}")
        elif event.type == VoiceAgentEventType.AUDIO and event.audio:
            agent_audio.extend(event.audio)
        elif event.type in (
            VoiceAgentEventType.AGENT_STARTED_SPEAKING,
            VoiceAgentEventType.AGENT_AUDIO_DONE,
            VoiceAgentEventType.WELCOME,
        ):
            print(f"-- {event.type.value} --")

    try:
        async for _ in provider.run_voice_agent(
            _mic_audio(provider),
            on_event=_on_event,
            function_handler=_weather_handler,
            functions=[WEATHER_FUNCTION],
            greeting="Hello! How can I help you today?",
            prompt="You are a concise, friendly voice assistant.",
        ):
            pass  # all handling done in _on_event / function_handler
    finally:
        if agent_audio:
            with open("deepgram_agent_audio.pcm", "wb") as fh:  # noqa: ASYNC230
                fh.write(bytes(agent_audio))
            logger.info("Wrote %d bytes of agent audio.", len(agent_audio))
        await provider.close()

    # --- Low-level alternative (manual control) ---------------------------
    # async with provider.open_voice_agent(prompt="...", functions=[WEATHER_FUNCTION]) as s:
    #     asyncio.create_task(pump_microphone_into(s))   # your audio source
    #     async for event in s:
    #         if event.type == VoiceAgentEventType.FUNCTION_CALL_REQUEST:
    #             for fn in event.raw["functions"]:
    #                 out = await _weather_handler(VoiceAgentFunctionCall(
    #                     id=fn["id"], name=fn["name"],
    #                     arguments=json.loads(fn["arguments"]),
    #                     client_side=fn["client_side"], raw=fn))
    #                 await s.respond_to_function_call(fn["id"], fn["name"], out)


if __name__ == "__main__":
    asyncio.run(main())
