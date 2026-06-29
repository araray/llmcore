# examples/deepgram_text_intelligence.py
"""Text intelligence (summary / topics / sentiment / intents) with Deepgram.

Uses ``analyze_text`` (read.v1) on plain text. Exactly one of ``text`` or
``url`` must be supplied.

Run:
    pip install "llmcore[deepgram]"
    export DEEPGRAM_API_KEY="dg_..."
    python examples/deepgram_text_intelligence.py
"""

from __future__ import annotations

import asyncio
import logging
import os

from llmcore.providers.deepgram_provider import DeepgramProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TEXT = (
    "I called yesterday about my internet being down for three days. "
    "The technician never showed up and now I want a refund and to cancel "
    "my plan. This is the third time this has happened this year."
)


async def main() -> None:
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        logger.error("Set DEEPGRAM_API_KEY to run this example.")
        return

    provider = DeepgramProvider({"api_key": api_key, "_instance_name": "deepgram"})
    try:
        result = await provider.analyze_text(
            TEXT, summarize=True, topics=True, sentiment=True, intents=True
        )
        print("\n=== Summary ===")
        print(result.summary)
        print("\n=== Topics ===")
        for seg in result.topics:
            print(" ", seg.get("topics") or seg)
        print("\n=== Intents ===")
        for seg in result.intents:
            print(" ", seg.get("intents") or seg)
        print("\n=== Sentiments ===")
        print(result.sentiments)
        print(f"\nrequest_id={result.request_id} model={result.model}")
    finally:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
