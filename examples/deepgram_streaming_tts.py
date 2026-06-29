# examples/deepgram_streaming_tts.py
"""Live (streaming) text-to-speech with the Deepgram provider.

``stream_speech`` is dual-mode:
* pass a single string -> Deepgram's REST streaming endpoint yields audio chunks;
* pass an async iterable of text pieces -> a TTS WebSocket is opened and each
  piece is sent + flushed, with audio bytes streamed back as they synthesize.

This example demonstrates both and writes each to a file.

Run:
    pip install "llmcore[deepgram]"
    export DEEPGRAM_API_KEY="dg_..."
    python examples/deepgram_streaming_tts.py
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator

from llmcore.providers.deepgram_provider import DeepgramProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def _text_pieces() -> AsyncIterator[str]:
    for piece in [
        "Streaming text to speech ",
        "lets you start playback ",
        "before the full text is ready.",
    ]:
        yield piece
        await asyncio.sleep(0.1)


async def main() -> None:
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        logger.error("Set DEEPGRAM_API_KEY to run this example.")
        return

    provider = DeepgramProvider({"api_key": api_key, "_instance_name": "deepgram"})
    try:
        # --- REST streaming (single string) -------------------------------
        chunks: list[bytes] = []
        async for audio in provider.stream_speech(
            "Hello from the Deepgram REST streaming endpoint.",
            model="aura-2-thalia-en",
            response_format="mp3",
        ):
            chunks.append(audio)
        with open("deepgram_stream_rest.mp3", "wb") as fh:  # noqa: ASYNC230
            fh.write(b"".join(chunks))
        logger.info("REST stream: %d chunks, %d bytes", len(chunks), sum(map(len, chunks)))

        # --- WebSocket streaming (incremental text) -----------------------
        ws_bytes = bytearray()
        async for audio in provider.stream_speech(
            _text_pieces(),
            model="aura-2-thalia-en",
            response_format="linear16",
            sample_rate=24000,
        ):
            ws_bytes.extend(audio)
        with open("deepgram_stream_ws.pcm", "wb") as fh:  # noqa: ASYNC230
            fh.write(bytes(ws_bytes))
        logger.info("WS stream: %d bytes of linear16 PCM", len(ws_bytes))
    finally:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
