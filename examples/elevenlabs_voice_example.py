#!/usr/bin/env python3
"""ElevenLabs example — speech, transcription, and consent you can act on.

The interesting part is not that llmcore can synthesize speech; several
providers do. It is that every synthesized artifact tells you *whose voice it
is* and whether ElevenLabs considers that voice cleared for use, without a
second API call — and that "the provider didn't say" stays distinguishable from
"the provider said no".

Run:
    export ELEVENLABS_API_KEY=...
    python examples/elevenlabs_voice_example.py
"""

from __future__ import annotations

import asyncio
import os

from llmcore import LLMCore
from llmcore.media import MediaRef

CONFIG = {
    "providers": {
        "elevenlabs": {
            "type": "elevenlabs",
            "timeout": 120,
            # Per-capability models; these are the defaults.
            "models": {"tts": "eleven_v4", "asr": "scribe_v1"},
        }
    }
}


def describe(consent) -> str:
    """Render a consent record the way a reviewer would want to read it."""
    if consent is None:
        return "no consent record (not a cloned-voice provider, or disabled)"
    if consent.verification_satisfied is None:
        return f"{consent.category or 'unknown'}: provider said nothing about verification"
    state = "cleared" if consent.verification_satisfied else "NOT CLEARED"
    return (
        f"{consent.category or 'unknown'} voice {consent.voice_name!r} — {state} "
        f"(cloned={consent.is_cloned}, owner={consent.is_owner})"
    )


async def main() -> None:
    if not os.getenv("ELEVENLABS_API_KEY"):
        raise SystemExit("Set ELEVENLABS_API_KEY first.")

    llm = await LLMCore.create(CONFIG)
    try:
        # 1. Speech, with provenance attached to the artifact itself.
        print("synthesizing...")
        speech = await llm.media.audio.speak(
            "Consent is not an afterthought.", provider="elevenlabs"
        )
        audio = speech.artifacts[0]
        print(f"  {len(audio.data)} bytes {audio.mime_type} @ {audio.sample_rate_hz} Hz")
        print(f"  model: {audio.provenance.generator}")
        print(f"  voice: {describe(audio.provenance.consent)}")
        print(f"  billed: {speech.usage.characters} characters")

        # 2. A policy a caller can actually enforce. Note the three branches:
        #    treating "unknown" as "fine" is a decision, not a default.
        consent = audio.provenance.consent
        if consent and consent.verification_satisfied is False:
            print("\n  REFUSING to ship: the voice is not cleared for use.")
            return
        if consent and consent.verification_satisfied is None:
            print("\n  WARNING: consent state unknown — decide before shipping.")

        # 3. Streaming, for when first-byte latency matters more than a handle.
        print("\nstreaming...")
        chunks = 0
        async for _chunk in llm.media.audio.stream_tts(
            "Streaming starts before synthesis finishes.", provider="elevenlabs"
        ):
            chunks += 1
        print(f"  {chunks} chunks")

        # 4. Round trip: the speech we just made, transcribed back.
        print("\ntranscribing it back...")
        transcript = await llm.media.audio.transcribe(
            audio=MediaRef.from_bytes(audio.data, mime_type=audio.mime_type, filename="s.mp3"),
            provider="elevenlabs",
        )
        text_artifact = transcript.artifacts[0]
        print(f"  {text_artifact.text!r}")
        print(f"  language: {text_artifact.provider_metadata.get('language_code')}")

        # 5. Sound effects carry no consent record -- nothing here is a voice,
        #    and an empty field would imply a question that does not apply.
        print("\nsound effect...")
        sfx = await llm.media.audio.sfx(
            prompt="a heavy wooden door creaking open",
            provider="elevenlabs",
            duration_seconds=3,
        )
        print(f"  {len(sfx.artifacts[0].data)} bytes")
        print(f"  consent: {describe(sfx.artifacts[0].provenance.consent)}")
    finally:
        await llm.close()


if __name__ == "__main__":
    asyncio.run(main())
