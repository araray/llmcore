# tests/media/test_media_bridge.py
"""Tests for the legacy-result ↔ MediaArtifact bridge.

``models_multimodal`` predates ``llmcore.media`` and is public API — seven
providers return those types today.  Rather than replacing them, each converts
to and from :class:`~llmcore.media.MediaArtifact` so the legacy provider methods
and the media routers describe the same asset (spec §4.3).  These tests pin the
round trips, including the data that must survive them.
"""

from __future__ import annotations

import base64
import hashlib

import pytest

from llmcore.media.models import MediaArtifact, MediaKind
from llmcore.models_multimodal import (
    GeneratedImage,
    ImageGenerationResult,
    OCRResult,
    SpeechResult,
    TranscriptionResult,
    TranscriptionSegment,
)

AUDIO = b"ID3\x04audio-bytes"


class TestSpeechResultBridge:
    def _speech(self, **kw) -> SpeechResult:
        return SpeechResult(
            audio_data=AUDIO,
            format=kw.pop("format", "mp3"),
            model="aura-2-thalia-en",
            voice="thalia",
            duration_seconds=1.5,
            **kw,
        )

    def test_to_artifact_carries_bytes_and_checksum(self):
        a = self._speech().to_artifact()
        assert a.kind is MediaKind.AUDIO
        assert a.data == AUDIO
        assert a.checksum_sha256 == hashlib.sha256(AUDIO).hexdigest()
        assert a.duration_seconds == 1.5

    @pytest.mark.parametrize(
        ("fmt", "mime"),
        [("mp3", "audio/mpeg"), ("linear16", "audio/wav"), ("opus", "audio/opus"),
         ("flac", "audio/flac"), ("mulaw", "audio/basic")],
    )
    def test_format_maps_to_mime_type(self, fmt, mime):
        assert self._speech(format=fmt).to_artifact().mime_type == mime

    def test_unknown_format_degrades_gracefully(self):
        assert self._speech(format="weird").to_artifact().mime_type == "audio/weird"

    def test_voice_and_format_survive_in_metadata(self):
        meta = self._speech().to_artifact().provider_metadata
        assert meta["voice"] == "thalia" and meta["format"] == "mp3"

    def test_round_trip(self):
        back = SpeechResult.from_artifact(self._speech().to_artifact())
        assert back.audio_data == AUDIO
        assert back.voice == "thalia" and back.format == "mp3"
        assert back.duration_seconds == 1.5

    def test_from_artifact_requires_inline_bytes(self):
        remote = MediaArtifact(kind=MediaKind.AUDIO, uri="https://x/a.mp3")
        with pytest.raises(ValueError, match="materialize"):
            SpeechResult.from_artifact(remote)

    def test_from_artifact_overrides_win(self):
        a = self._speech().to_artifact()
        assert SpeechResult.from_artifact(a, voice="other").voice == "other"


class TestTranscriptionResultBridge:
    def _transcript(self) -> TranscriptionResult:
        return TranscriptionResult(
            text="hello there",
            language="en",
            duration_seconds=2.0,
            model="nova-3",
            segments=[
                TranscriptionSegment(text="hello", start=0.0, end=1.0, speaker="0"),
                TranscriptionSegment(text="there", start=1.0, end=2.0, speaker="1"),
            ],
        )

    def test_to_artifact_is_text_kind(self):
        a = self._transcript().to_artifact()
        assert a.kind is MediaKind.TEXT
        assert a.text == "hello there"
        assert a.mime_type == "text/plain"
        assert a.duration_seconds == 2.0

    def test_diarization_survives(self):
        segs = self._transcript().to_artifact().provider_metadata["segments"]
        assert [s["speaker"] for s in segs] == ["0", "1"]

    def test_round_trip_preserves_segments(self):
        back = TranscriptionResult.from_artifact(self._transcript().to_artifact())
        assert back.text == "hello there" and back.language == "en"
        assert len(back.segments) == 2
        assert back.segments[1].end == 2.0 and back.segments[1].speaker == "1"

    def test_round_trip_of_empty_transcript(self):
        empty = TranscriptionResult(text="", model="nova-3")
        back = TranscriptionResult.from_artifact(empty.to_artifact())
        assert back.text == "" and back.segments == []


class TestImageBridge:
    def test_base64_data_is_decoded_to_bytes(self):
        img = GeneratedImage(data=base64.b64encode(b"PNGBYTES").decode(), format="png")
        a = img.to_artifact()
        assert a.data == b"PNGBYTES"
        assert a.mime_type == "image/png"
        assert a.checksum_sha256 == hashlib.sha256(b"PNGBYTES").hexdigest()

    def test_url_only_image_keeps_uri(self):
        a = GeneratedImage(url="https://x/y.jpeg", format="jpeg").to_artifact()
        assert a.uri == "https://x/y.jpeg" and a.data is None
        assert a.mime_type == "image/jpeg"

    def test_bad_base64_falls_back_to_uri_path(self):
        a = GeneratedImage(data="!!!not-base64!!!", url="https://x/y.png").to_artifact()
        assert a.data is None and a.uri == "https://x/y.png"

    def test_revised_prompt_is_kept(self):
        a = GeneratedImage(url="u", revised_prompt="a cat, oil painting").to_artifact()
        assert a.provider_metadata["revised_prompt"] == "a cat, oil painting"

    def test_result_maps_every_image(self):
        result = ImageGenerationResult(
            images=[
                GeneratedImage(data=base64.b64encode(b"a").decode()),
                GeneratedImage(url="https://x/b.png"),
            ],
            model="gpt-image",
        )
        arts = result.to_artifacts()
        assert len(arts) == 2
        assert arts[0].data == b"a" and arts[1].uri == "https://x/b.png"

    def test_empty_result(self):
        assert ImageGenerationResult(images=[], model="m").to_artifacts() == []


class TestOCRBridge:
    def test_markdown_and_text_pages_are_joined(self):
        a = OCRResult(
            pages=[{"markdown": "# Title"}, {"text": "body"}],
            model="mistral-ocr",
            pages_processed=2,
        ).to_artifact()
        assert a.kind is MediaKind.TEXT
        assert a.text == "# Title\nbody"
        assert a.mime_type == "text/markdown"
        assert a.provider_metadata["pages_processed"] == 2

    def test_pages_are_preserved_verbatim(self):
        pages = [{"markdown": "x", "images": [{"id": "img-1"}]}]
        a = OCRResult(pages=pages, model="m", pages_processed=1).to_artifact()
        assert a.provider_metadata["pages"] == pages

    def test_empty_pages(self):
        a = OCRResult(pages=[], model="m", pages_processed=0).to_artifact()
        assert a.text == ""
