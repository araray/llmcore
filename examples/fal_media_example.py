#!/usr/bin/env python3
"""fal media example — the queue lifecycle, URL inputs and cross-capability chaining.

fal is a media *marketplace*: every capability is a queue submission, and inputs
are addressed by URL. This example walks the three things that makes different
from the other media adapters.

Run:
    export FAL_KEY=...            # FAL_API_KEY also works
    python examples/fal_media_example.py
"""

from __future__ import annotations

import asyncio
import os

from llmcore import LLMCore
from llmcore.media import MediaRef

CONFIG = {
    "providers": {
        "fal": {
            "type": "fal",
            "timeout": 300,
            # Endpoint paths, not model names. Override any of them; the
            # gallery at https://fal.ai/models moves faster than llmcore does.
            "models": {"image_generate": "fal-ai/flux/schnell"},
        }
    }
}


async def main() -> None:
    if not (os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY")):
        raise SystemExit("Set FAL_KEY (or FAL_API_KEY) first.")

    llm = await LLMCore.create(CONFIG)
    try:
        # 1. Everything is a job -- even image generation, which answers
        #    synchronously on OpenAI. `wait()` absorbs the difference, so this
        #    same code works against either provider.
        print("submitting an image job...")
        job = await llm.media.images.generate(
            "a calico cat asleep on a stack of books, soft window light",
            provider="fal",
        )
        print(f"  {job.status.value}, queue position {job.queue_position}")

        result = await llm.media.wait(job, timeout=300)
        image = result.artifacts[0]
        print(f"  done: {image.width}x{image.height} {image.mime_type}")
        print(f"  {image.uri}")
        if result.usage:
            print(f"  compute: {result.usage.compute_seconds:.2f}s")

        # 2. Chaining costs nothing extra: a fal artifact is already a URL, so
        #    it feeds straight back in without being downloaded and re-uploaded.
        print("\nupscaling that image in place...")
        upscaled = await llm.media.wait(
            await llm.media.images.upscale(
                image=MediaRef.from_url(image.uri), provider="fal"
            ),
            timeout=300,
        )
        big = upscaled.artifacts[0]
        print(f"  {image.width}x{image.height} -> {big.width}x{big.height}")

        # 3. Cross-capability chaining: speech out, then straight back in.
        print("\nTTS -> ASR round trip...")
        speech = await llm.media.wait(
            await llm.media.audio.speak(
                "The abstraction did not bend.", provider="fal"
            ),
            timeout=300,
        )
        transcript = await llm.media.wait(
            await llm.media.audio.transcribe(
                audio=MediaRef.from_url(speech.artifacts[0].uri), provider="fal"
            ),
            timeout=300,
        )
        print(f"  heard back: {transcript.artifacts[0].text!r}")

        # 4. Keep a local copy. fal URLs are CDN-hosted and not permanent, but
        #    fal does not publish an expiry -- so the default ON_EXPIRY policy
        #    has nothing to trigger on and leaves the artifact remote. Pass
        #    force=True (or configure `artifact_materialize = "always"`) when you want
        #    the bytes on disk.
        local = await llm.media.artifacts.materialize(big, force=True)
        print(f"\nsaved the upscale to {local.uri}")
        print(f"  sha256 {local.checksum_sha256}")
        print(f"  source {local.provider_metadata['source_uri']}")
    finally:
        await llm.close()


if __name__ == "__main__":
    asyncio.run(main())
