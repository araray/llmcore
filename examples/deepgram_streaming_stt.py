# examples/deepgram_streaming_stt.py
"""Live (streaming) speech-to-text with the Deepgram provider.

This example is fully self-contained: it first synthesizes a short utterance
with Deepgram TTS as raw linear16 PCM, then streams those bytes back through the
live STT socket as if they were microphone frames, and prints interim/final
events as they arrive.

Shows how to:
1. Drive ``transcribe_stream`` with an async byte source (one-call fan-in/out).
2. Consume :class:`TranscriptionStreamEvent`s (INTERIM / FINAL / UTTERANCE_END).
3. (Alternative) Manage the socket manually with ``open_transcription_socket``.

Run:
    pip install "llmcore[deepgram]"
    export DEEPGRAM_API_KEY="dg_..."
    python examples/deepgram_streaming_stt.py
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
FRAME_BYTES = 3200  # ~100 ms of 16 kHz mono linear16


async def _synthesize_pcm(provider: DeepgramProvider, text: str) -> bytes:
    """Synthesize ``text`` as raw linear16 PCM (so the STT socket can ingest it)."""
    result = await provider.generate_speech(
        text, voice="aura-2-thalia-en", response_format="linear16", sample_rate=SAMPLE_RATE
    )
    return result.audio_data


async def _frames(pcm: bytes) -> AsyncIterator[bytes]:
    """Yield PCM in real-time-ish frames with small gaps."""
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
        pcm = await _synthesize_pcm(
            provider, "Testing the Deepgram live transcription pipeline end to end."
        )
        logger.info("Synthesized %d bytes of linear16 PCM; streaming to STT...", len(pcm))

        async for event in provider.transcribe_stream(
            _frames(pcm),
            model="nova-3",
            language="en",
            encoding="linear16",
            sample_rate=SAMPLE_RATE,
            interim_results=True,
            smart_format=True,
        ):
            if event.type in (StreamEventType.INTERIM, StreamEventType.FINAL):
                tag = "FINAL" if event.type == StreamEventType.FINAL else "interim"
                print(f"[{tag}] {event.text}")
            elif event.type == StreamEventType.UTTERANCE_END:
                print(f"[utterance-end @ {event.end}]")
    finally:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
