# examples/deepgram_flux.py
"""Conversational STT with Deepgram Flux (listen.v2) via the Deepgram provider.

Flux is turn-aware: instead of interim/final hypotheses it emits StartOfTurn /
EagerEndOfTurn / EndOfTurn events with end-of-turn confidence — ideal for
voice-agent turn-taking. This example synthesizes a short utterance as linear16
PCM and streams it through the Flux socket.

Run:
    pip install "llmcore[deepgram]"
    export DEEPGRAM_API_KEY="dg_..."
    python examples/deepgram_flux.py
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator

from llmcore.models_multimodal import StreamEventType
from llmcore.providers.deepgram_provider import DeepgramProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_BYTES = 3200


async def _frames(pcm: bytes) -> AsyncIterator[bytes]:
    for i in range(0, len(pcm), FRAME_BYTES):
        yield pcm[i : i + FRAME_BYTES]
        await asyncio.sleep(0.05)


async def main() -> None:
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        logger.error("Set DEEPGRAM_API_KEY to run this example.")
        return

    provider = DeepgramProvider({"api_key": api_key, "_instance_name": "deepgram"})
    try:
        speech = await provider.generate_speech(
            "What is the weather like in San Francisco today?",
            voice="aura-2-thalia-en",
            response_format="linear16",
            sample_rate=SAMPLE_RATE,
        )
        logger.info("Streaming %d bytes through Flux...", len(speech.audio_data))

        async for event in provider.transcribe_stream_flux(
            _frames(speech.audio_data),
            model="flux-general-en",
            encoding="linear16",
            sample_rate=SAMPLE_RATE,
            eot_threshold=0.7,
        ):
            if event.type == StreamEventType.START_OF_TURN:
                print("-- start of turn --")
            elif event.type == StreamEventType.EAGER_END_OF_TURN:
                print(f"[eager-eot conf={event.confidence}] {event.text}")
            elif event.type == StreamEventType.END_OF_TURN:
                print(f"[END OF TURN conf={event.confidence}] {event.text}")
    finally:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
