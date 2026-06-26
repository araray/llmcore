# examples/deepgram_batch_tts.py
"""Batch text-to-speech with the Deepgram provider.

Shows how to:
1. Synthesize speech from text with ``generate_speech`` (collects the SDK's
   async audio generator into a single :class:`SpeechResult`).
2. Select an Aura voice (passed as the model) and an output encoding.
3. Write the resulting audio to a file.

Run:
    pip install "llmcore[deepgram]"
    export DEEPGRAM_API_KEY="dg_..."
    python examples/deepgram_batch_tts.py
"""

from __future__ import annotations

import asyncio
import logging
import os

from llmcore.providers.deepgram_provider import DeepgramProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TEXT = (
    "Deepgram brings real-time voice to the Wairu ecosystem. "
    "This audio was synthesized with the Aura text-to-speech engine."
)


async def main() -> None:
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        logger.error("Set DEEPGRAM_API_KEY to run this example.")
        return

    provider = DeepgramProvider({"api_key": api_key, "_instance_name": "deepgram"})
    try:
        result = await provider.generate_speech(
            TEXT,
            voice="aura-2-thalia-en",  # voice is passed as the model
            response_format="mp3",
            sample_rate=24000,
        )
        out = "deepgram_tts_out.mp3"
        with open(out, "wb") as fh:  # noqa: ASYNC230
            fh.write(result.audio_data)
        logger.info(
            "Wrote %d bytes to %s (voice=%s, format=%s)",
            len(result.audio_data), out, result.voice, result.format,
        )
    finally:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
