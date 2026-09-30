# tests/media/test_openai_media_adapter.py
"""OpenAI as a media adapter (spec phase M3): images, speech, embeddings.

Also guards the subclassing hazard this phase introduced. ``DeepInfraProvider``,
``VLLMProvider``, ``PoeProvider`` and ``OpenRouterProvider`` all extend
``OpenAIProvider``, so they inherit the media protocol *methods* — but not the
endpoints behind them. Each must declare what it can actually serve, or the
router will confidently call an endpoint that 404s.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmcore.exceptions import ProviderError
from llmcore.media import MediaCapability, MediaExecution, MediaManager, MediaRef
from llmcore.media.protocols import (
    CAPABILITY_PROTOCOLS,
    ASRProvider,
    ImageEditProvider,
    ImageGenerationProvider,
    MediaCapableProvider,
    StreamingTTSProvider,
    TTSProvider,
)
from llmcore.models_multimodal import (
    GeneratedImage,
    ImageGenerationResult,
    SpeechResult,
    TranscriptionResult,
)


@pytest.fixture
def provider():
    """An OpenAIProvider with the SDK client mocked out."""
    with patch("llmcore.providers.openai_provider.AsyncOpenAI") as client_cls:
        client_cls.return_value = MagicMock()
        from llmcore.providers.openai_provider import OpenAIProvider

        return OpenAIProvider({"api_key": "sk-test", "_instance_name": "openai"})


def _image_result(n: int = 1) -> ImageGenerationResult:
    return ImageGenerationResult(
        images=[GeneratedImage(url=f"https://x/{i}.png", format="png") for i in range(n)],
        model="gpt-image-1",
        metadata={"created": 1},
    )


def _speech() -> SpeechResult:
    return SpeechResult(
        audio_data=b"MP3", format="mp3", model="gpt-4o-mini-tts", voice="alloy"
    )


def _transcript(duration: float | None = 60.0) -> TranscriptionResult:
    return TranscriptionResult(
        text="hello", language="en", duration_seconds=duration, model="whisper-1"
    )


# ---------------------------------------------------------------------------
# Conformance
# ---------------------------------------------------------------------------


class TestConformance:
    def test_is_media_capable(self, provider):
        assert isinstance(provider, MediaCapableProvider)

    @pytest.mark.parametrize(
        "protocol",
        [ImageGenerationProvider, ImageEditProvider, TTSProvider, StreamingTTSProvider, ASRProvider],
    )
    def test_implements_protocols(self, provider, protocol):
        assert isinstance(provider, protocol)

    def test_declared_capabilities(self, provider):
        assert provider.media_capabilities() == frozenset(
            {
                MediaCapability.IMAGE_GENERATE,
                MediaCapability.IMAGE_EDIT,
                MediaCapability.TTS,
                MediaCapability.TTS_STREAM,
                MediaCapability.ASR,
            }
        )

    def test_sora_video_is_not_declared(self, provider):
        """openai 3.1 deprecated the Sora video APIs; we must not offer them."""
        caps = provider.media_capabilities()
        assert MediaCapability.VIDEO_GENERATE not in caps
        assert MediaCapability.VIDEO_EDIT not in caps

    def test_every_declared_capability_is_backed(self, provider):
        unmet = [
            c.value
            for c in provider.media_capabilities()
            if not isinstance(provider, CAPABILITY_PROTOCOLS[c])
        ]
        assert unmet == []

    def test_execution_classes(self, provider):
        assert provider.media_execution(MediaCapability.TTS_STREAM) is MediaExecution.STREAM
        for cap in (MediaCapability.IMAGE_GENERATE, MediaCapability.TTS, MediaCapability.ASR):
            assert provider.media_execution(cap) is MediaExecution.REQUEST_RESPONSE


# ---------------------------------------------------------------------------
# The subclassing guard
# ---------------------------------------------------------------------------


class TestSubclassCapabilityDeclaration:
    """Every OpenAIProvider subclass must declare its own capabilities.

    Inheriting OpenAI's declaration would advertise /v1/images and /v1/audio on
    providers that do not serve them.
    """

    @staticmethod
    def _subclasses() -> list[type]:
        from llmcore.providers.openai_provider import OpenAIProvider

        # Import every module that defines one, then walk the tree.
        import llmcore.providers.deepinfra_provider  # noqa: F401
        import llmcore.providers.openrouter_provider  # noqa: F401
        import llmcore.providers.poe_provider  # noqa: F401
        import llmcore.providers.vllm_provider  # noqa: F401

        found: list[type] = []
        stack = list(OpenAIProvider.__subclasses__())
        while stack:
            cls = stack.pop()
            found.append(cls)
            stack.extend(cls.__subclasses__())
        return found

    def test_subclasses_were_found(self):
        assert len(self._subclasses()) >= 4

    def test_every_subclass_declares_its_own(self):
        offenders = [
            cls.__name__
            for cls in self._subclasses()
            if "_MEDIA_CAPABILITIES" not in cls.__dict__
        ]
        assert offenders == [], (
            "These OpenAIProvider subclasses inherit OpenAI's media capability "
            f"declaration instead of stating their own: {offenders}"
        )

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("DeepInfraProvider", {"image_generate", "tts", "asr"}),
            ("VLLMProvider", set()),
            ("PoeProvider", set()),
            ("OpenRouterProvider", set()),
        ],
    )
    def test_subclass_declarations(self, name, expected):
        cls = next(c for c in self._subclasses() if c.__name__ == name)
        assert set(cls.__dict__["_MEDIA_CAPABILITIES"]) == expected

    def test_declarations_use_valid_capability_names(self):
        for cls in self._subclasses():
            for raw in cls.__dict__.get("_MEDIA_CAPABILITIES", frozenset()):
                MediaCapability(raw)  # raises on a typo


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


class TestImageGeneration:
    async def test_delegates_and_normalizes(self, provider):
        provider.generate_image = AsyncMock(return_value=_image_result(2))
        result = await provider.generate_image_media("a tabby", n=2, size="1024x1024")
        assert result.capability is MediaCapability.IMAGE_GENERATE
        assert len(result.artifacts) == 2
        assert result.usage.basis == "per_image" and result.usage.images == 2
        kwargs = provider.generate_image.call_args.kwargs
        assert kwargs["n"] == 2 and kwargs["size"] == "1024x1024"

    async def test_unsupported_params_are_not_forwarded(self, provider):
        """seed / negative_prompt have no OpenAI equivalent."""
        provider.generate_image = AsyncMock(return_value=_image_result())
        await provider.generate_image_media("x", seed=7, negative_prompt="blurry")
        kwargs = provider.generate_image.call_args.kwargs
        assert "seed" not in kwargs and "negative_prompt" not in kwargs

    async def test_reference_images_route_to_edit(self, provider):
        """OpenAI expresses reference-conditioned generation as an edit."""
        provider.edit_image_media = AsyncMock(return_value="edited")
        ref = MediaRef.from_bytes(b"png", mime_type="image/png")
        out = await provider.generate_image_media("x", reference_images=[ref])
        assert out == "edited"
        assert provider.edit_image_media.call_args.kwargs["image"] is ref


class TestImageEdit:
    async def test_calls_the_edits_endpoint(self, provider):
        response = MagicMock()
        response.data = [MagicMock(b64_json=None, url="https://x/e.png", revised_prompt=None)]
        response.created = 1
        provider._client.images.edit = AsyncMock(return_value=response)

        result = await provider.edit_image_media(
            "make it night",
            image=MediaRef.from_bytes(b"PNGDATA", mime_type="image/png"),
            size="512x512",
        )
        assert result.capability is MediaCapability.IMAGE_EDIT
        assert result.artifact.uri == "https://x/e.png"
        kwargs = provider._client.images.edit.call_args.kwargs
        assert kwargs["prompt"] == "make it night"
        assert kwargs["size"] == "512x512"
        # uploads, not URLs
        assert kwargs["image"][1] == b"PNGDATA"

    async def test_mask_is_uploaded(self, provider):
        response = MagicMock()
        response.data = []
        provider._client.images.edit = AsyncMock(return_value=response)
        await provider.edit_image_media(
            "x",
            image=MediaRef.from_bytes(b"IMG"),
            mask=MediaRef.from_bytes(b"MASK"),
        )
        assert provider._client.images.edit.call_args.kwargs["mask"][1] == b"MASK"

    async def test_remote_ref_is_fetched(self, provider):
        """The edits endpoint takes an upload, so a URL must be materialized."""
        response = MagicMock()
        response.data = []
        provider._client.images.edit = AsyncMock(return_value=response)

        async def fake_fetch(url):
            return b"DOWNLOADED"

        with patch("llmcore.media.artifacts.default_fetcher", return_value=fake_fetch):
            await provider.edit_image_media(
                "x", image=MediaRef.from_url("https://x/in.png")
            )
        assert provider._client.images.edit.call_args.kwargs["image"][1] == b"DOWNLOADED"

    async def test_api_error_maps_to_provider_error(self, provider):
        from llmcore.providers.openai_provider import OpenAIError

        provider._client.images.edit = AsyncMock(side_effect=OpenAIError("nope"))
        with pytest.raises(ProviderError, match="Image edit error"):
            await provider.edit_image_media("x", image=MediaRef.from_bytes(b"i"))


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


class TestSpeech:
    async def test_synthesize_delegates(self, provider):
        provider.generate_speech = AsyncMock(return_value=_speech())
        result = await provider.synthesize_speech_media("hello", voice="nova", speed=1.1)
        assert result.capability is MediaCapability.TTS
        assert result.artifact.data == b"MP3"
        assert result.usage.characters == 5
        kwargs = provider.generate_speech.call_args.kwargs
        assert kwargs["voice"] == "nova" and kwargs["speed"] == 1.1

    async def test_sample_rate_is_ignored_not_forwarded(self, provider):
        """OpenAI TTS has no sample-rate parameter; forwarding it would 400."""
        provider.generate_speech = AsyncMock(return_value=_speech())
        await provider.synthesize_speech_media("hi", sample_rate_hz=24000)
        assert "sample_rate_hz" not in provider.generate_speech.call_args.kwargs
        assert "sample_rate" not in provider.generate_speech.call_args.kwargs

    async def test_omitted_options_are_not_forced(self, provider):
        provider.generate_speech = AsyncMock(return_value=_speech())
        await provider.synthesize_speech_media("hi")
        kwargs = provider.generate_speech.call_args.kwargs
        for absent in ("voice", "response_format", "speed"):
            assert absent not in kwargs

    async def test_stream_uses_streaming_response(self, provider):
        class _Resp:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def iter_bytes(self):
                for chunk in (b"a", b"b"):
                    yield chunk

        provider._client.audio.speech.with_streaming_response.create = MagicMock(
            return_value=_Resp()
        )
        stream = provider.stream_speech_media("hi", voice="nova")  # no await
        assert [c async for c in stream] == [b"a", b"b"]
        kwargs = provider._client.audio.speech.with_streaming_response.create.call_args.kwargs
        assert kwargs["voice"] == "nova" and kwargs["input"] == "hi"


class TestTranscription:
    async def test_delegates_with_bytes(self, provider):
        provider.transcribe_audio = AsyncMock(return_value=_transcript())
        result = await provider.transcribe_media(audio=MediaRef.from_bytes(b"WAV"))
        assert provider.transcribe_audio.call_args[0][0] == b"WAV"
        assert result.text == "hello"
        assert result.usage.audio_minutes == 1.0

    async def test_timestamps_map_to_granularities(self, provider):
        provider.transcribe_audio = AsyncMock(return_value=_transcript())
        await provider.transcribe_media(audio=MediaRef.from_bytes(b"x"), timestamps=True)
        assert provider.transcribe_audio.call_args.kwargs["timestamp_granularities"] == [
            "segment"
        ]

    async def test_remote_audio_is_fetched(self, provider):
        provider.transcribe_audio = AsyncMock(return_value=_transcript())

        async def fake_fetch(url):
            return b"REMOTE"

        with patch("llmcore.media.artifacts.default_fetcher", return_value=fake_fetch):
            await provider.transcribe_media(audio=MediaRef.from_url("https://x/a.wav"))
        assert provider.transcribe_audio.call_args[0][0] == b"REMOTE"

    async def test_artifact_chaining_names_the_real_format(self, provider):
        """Regression: chaining TTS output into ASR must not claim it is wav.

        OpenAI infers the container from the upload filename, so mp3 bytes
        uploaded as ``audio.wav`` are rejected with
        "This model does not support the format you provided" — which is
        exactly what happened the first time a TTS artifact was fed back in.
        """
        from llmcore.media.models import MediaArtifact, MediaKind

        provider.transcribe_audio = AsyncMock(return_value=_transcript())
        mp3 = MediaArtifact(kind=MediaKind.AUDIO, data=b"ID3", mime_type="audio/mpeg")
        await provider.transcribe_media(audio=MediaRef.from_artifact(mp3))
        assert provider.transcribe_audio.call_args.kwargs["filename"] == "audio.mp3"

    @pytest.mark.parametrize(
        ("ref", "expected"),
        [
            (MediaRef.from_bytes(b"x", mime_type="audio/mpeg"), "audio.mp3"),
            (MediaRef.from_bytes(b"x", mime_type="audio/wav"), "audio.wav"),
            (MediaRef.from_bytes(b"x", filename="clip.flac"), "clip.flac"),
            (MediaRef.from_url("https://x/a.ogg?t=1"), "audio.oga"),
            (MediaRef.from_bytes(b"x"), "audio.wav"),
        ],
    )
    def test_filename_inference(self, ref, expected):
        from llmcore.providers.openai_provider import _audio_filename_for

        assert _audio_filename_for(ref) == expected

    async def test_transcribe_audio_honours_filename(self, provider):
        """The legacy method now labels raw bytes correctly too."""
        captured: dict[str, Any] = {}

        async def _create(**kwargs):
            captured.update(kwargs)
            return MagicMock(text="hi", language=None, duration=None, segments=[])

        provider._client.audio.transcriptions.create = _create
        await provider.transcribe_audio(b"ID3", filename="clip.mp3")
        name, _stream, mime = captured["file"]
        assert name == "clip.mp3" and mime == "audio/mpeg"

    async def test_transcribe_audio_default_filename_is_unchanged(self, provider):
        """Existing callers keep the previous behaviour."""
        captured: dict[str, Any] = {}

        async def _create(**kwargs):
            captured.update(kwargs)
            return MagicMock(text="hi", language=None, duration=None, segments=[])

        provider._client.audio.transcriptions.create = _create
        await provider.transcribe_audio(b"RIFF")
        assert captured["file"][0] == "audio.wav"

    async def test_missing_duration_leaves_minutes_unset(self, provider):
        provider.transcribe_audio = AsyncMock(return_value=_transcript(duration=None))
        result = await provider.transcribe_media(audio=MediaRef.from_bytes(b"x"))
        assert result.usage.audio_minutes is None


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class TestEmbeddings:
    async def test_create_embeddings(self, provider):
        response = MagicMock()
        response.model_dump.return_value = {"data": [{"embedding": [0.1]}], "model": "m"}
        provider._client.embeddings.create = AsyncMock(return_value=response)
        out = await provider.create_embeddings(["hello"], dimensions=256)
        assert out["data"][0]["embedding"] == [0.1]
        kwargs = provider._client.embeddings.create.call_args.kwargs
        assert kwargs["input"] == ["hello"] and kwargs["dimensions"] == 256

    async def test_default_model(self, provider):
        response = MagicMock()
        response.model_dump.return_value = {}
        provider._client.embeddings.create = AsyncMock(return_value=response)
        await provider.create_embeddings("hi")
        assert (
            provider._client.embeddings.create.call_args.kwargs["model"]
            == "text-embedding-3-small"
        )

    async def test_error_maps(self, provider):
        from llmcore.providers.openai_provider import OpenAIError

        provider._client.embeddings.create = AsyncMock(side_effect=OpenAIError("bad"))
        with pytest.raises(ProviderError, match="Embeddings error"):
            await provider.create_embeddings("hi")


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


class TestRouting:
    async def test_image_generate_routes_to_openai(self, provider):
        media = MediaManager({"openai": provider})
        assert media.who_can(MediaCapability.IMAGE_GENERATE) == ["openai"]
        provider.generate_image = AsyncMock(return_value=_image_result())
        result = await media.images.generate("a tabby")
        assert result.provider == "openai"

    async def test_edit_routes_through_the_router(self, provider):
        response = MagicMock()
        response.data = []
        provider._client.images.edit = AsyncMock(return_value=response)
        media = MediaManager({"openai": provider})
        out = await media.images.edit("night", image=MediaRef.from_bytes(b"i"))
        assert out.capability is MediaCapability.IMAGE_EDIT

    def test_a_chat_only_subclass_is_not_an_adapter(self):
        """An OpenRouter instance must not advertise image or audio."""
        with patch("llmcore.providers.openai_provider.AsyncOpenAI") as c:
            c.return_value = MagicMock()
            from llmcore.providers.openrouter_provider import OpenRouterProvider

            router_provider = OpenRouterProvider({"api_key": "k"})
        media = MediaManager({"openrouter": router_provider})
        assert media.who_can(MediaCapability.IMAGE_GENERATE) == []
        assert media.capabilities() == {}
