# examples/deepgram_batch_stt.py
"""Batch (pre-recorded) speech-to-text with the Deepgram provider.

Shows how to:
1. Construct the Deepgram provider directly (media methods are provider-direct;
   the LLMCore facade has no media methods).
2. Transcribe raw audio bytes with ``transcribe_audio``.
3. Transcribe a remote URL (via the ``url=`` keyword).
4. Use nova-3 keyterm prompting and request smart formatting / diarization.
5. Read the normalized :class:`TranscriptionResult` (text, segments, metadata).

Run:
    pip install "llmcore[deepgram]"
    export DEEPGRAM_API_KEY="dg_..."
    python examples/deepgram_batch_stt.py [path/to/audio.wav]

If no path is given, a small public Deepgram sample is fetched over HTTP.
This performs real network calls and will incur Deepgram usage.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import urllib.request

from llmcore.providers.deepgram_provider import DeepgramProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_URL = "https://dpgr.am/spacewalk.wav"


async def main() -> None:
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        logger.error("Set DEEPGRAM_API_KEY to run this example.")
        return

    provider = DeepgramProvider({"api_key": api_key, "_instance_name": "deepgram"})
    try:
        # --- 1) Transcribe local bytes (or fetch the sample) ---------------
        if len(sys.argv) > 1:
            path = sys.argv[1]
            logger.info("Reading local audio: %s", path)
            audio = open(path, "rb").read()  # noqa: ASYNC230
        else:
            logger.info("Fetching sample audio: %s", SAMPLE_URL)
            audio = urllib.request.urlopen(SAMPLE_URL).read()  # noqa: ASYNC210

        result = await provider.transcribe_audio(
            audio,
            model="nova-3",
            language="en",
            smart_format=True,
            punctuate=True,
            diarize=True,
            prompt=["Deepgram", "spacewalk"],  # nova-3 keyterm prompting
        )
        print("\n=== Transcript (bytes) ===")
        print(result.text)
        print(f"model={result.model} duration={result.duration_seconds}s "
              f"segments={len(result.segments)}")
        if result.segments:
            seg = result.segments[0]
            print(f"first segment: [{seg.start:.2f}-{seg.end:.2f}] "
                  f"speaker={seg.speaker}: {seg.text[:80]}")

        # --- 2) Transcribe a remote URL (no local download) ----------------
        url_result = await provider.transcribe_audio(
            b"", url=SAMPLE_URL, model="nova-3", smart_format=True
        )
        print("\n=== Transcript (URL) ===")
        print(url_result.text[:200], "...")
    finally:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
