"""Concise live smoke checks for Deepgram audio support.

This example verifies both sides of the audio path without writing files:
text-to-speech produces audio bytes, then speech-to-text transcribes a small
public sample URL.

Run:
    set -a
    source /av/data/dbs/.env
    set +a
    python examples/live_audio_smoke.py
"""

from __future__ import annotations

import asyncio
import logging
import os

from llmcore import ProviderError
from llmcore.providers.deepgram_provider import DeepgramProvider

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
for noisy_logger in ("httpx", "httpcore", "llmcore"):
    logging.getLogger(noisy_logger).setLevel(logging.ERROR)

SAMPLE_URL = "https://dpgr.am/spacewalk.wav"


async def main() -> None:
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        logger.error("Set DEEPGRAM_API_KEY to run this example.")
        raise SystemExit(1)

    provider = DeepgramProvider({"api_key": api_key, "_instance_name": "deepgram"})
    try:
        async with asyncio.timeout(60):
            speech = await provider.generate_speech(
                "LLMCore live audio smoke test.",
                voice="aura-2-thalia-en",
                response_format="mp3",
            )
        logger.info(
            "OK deepgram TTS: %d bytes, voice=%s, format=%s",
            len(speech.audio_data),
            speech.voice,
            speech.format,
        )

        async with asyncio.timeout(60):
            transcript = await provider.transcribe_audio(
                b"",
                url=SAMPLE_URL,
                model="nova-3",
                language="en",
                smart_format=True,
                punctuate=True,
            )
        logger.info(
            "OK deepgram STT: duration=%ss, text=%s",
            transcript.duration_seconds,
            transcript.text[:160],
        )
    except (ProviderError, TimeoutError) as exc:
        logger.error("FAIL deepgram audio: %s", exc)
        raise SystemExit(1) from exc
    finally:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
