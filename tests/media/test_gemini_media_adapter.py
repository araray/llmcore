# tests/media/test_gemini_media_adapter.py
"""Gemini as a media adapter (spec phase M4): Imagen, Veo, native TTS, embeddings.

This is the phase that matters most for the abstraction: **Veo makes Gemini the
first true async-job provider**, so the ``MediaJob`` lifecycle is exercised
against a real vendor's long-running-operation shape rather than the in-repo
fake. If the job model is wrong anywhere, it shows here.

The google-genai client is mocked; its real ``types`` are used, so the config
objects the adapter builds must actually validate.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmcore.exceptions import MediaJobError, ProviderError
from llmcore.media import (
    JobPolicy,
    MediaCapability,
    MediaExecution,
    MediaJobStatus,
    MediaKind,
    MediaManager,
    MediaRef,
)
from llmcore.media.protocols import (
    CAPABILITY_PROTOCOLS,
    ImageEditProvider,
    ImageGenerationProvider,
    ImageUpscaleProvider,
    MediaCapableProvider,
    MediaJobPoller,
    TTSProvider,
    VideoGenerationProvider,
)

FAST = JobPolicy(poll_initial_seconds=0.0, poll_max_seconds=0.0, job_timeout_seconds=5.0, jitter=0.0)


@pytest.fixture
def provider():
    """A GeminiProvider whose google-genai client is mocked."""
    with patch("llmcore.providers.gemini_provider.genai") as genai_mod:
        genai_mod.Client.return_value = MagicMock()
        from llmcore.providers.gemini_provider import GeminiProvider

        return GeminiProvider({"api_key": "test-key", "_instance_name": "gemini"})


@pytest.fixture
def vertex_provider():
    """A GeminiProvider in Vertex AI mode, where Imagen's endpoints exist."""
    with patch("llmcore.providers.gemini_provider.genai") as genai_mod:
        genai_mod.Client.return_value = MagicMock()
        from llmcore.providers.gemini_provider import GeminiProvider

        return GeminiProvider(
            {
                "vertex_ai": True,
                "project": "p",
                "location": "us-central1",
                "_instance_name": "gemini",
            }
        )


def _generated_image(data: bytes = b"PNGDATA", mime: str = "image/png") -> Any:
    image = MagicMock(image_bytes=data, mime_type=mime, gcs_uri=None)
    return MagicMock(image=image, enhanced_prompt="enhanced", rai_filtered_reason=None)


def _image_response(n: int = 1) -> Any:
    return MagicMock(generated_images=[_generated_image() for _ in range(n)])


def _video_operation(*, done: bool = False, error: Any = None, videos: int = 0) -> Any:
    generated = [
        MagicMock(video=MagicMock(video_bytes=b"MP4", mime_type="video/mp4", uri=None))
        for _ in range(videos)
    ]
    operation = MagicMock(
        done=done,
        error=error,
        response=MagicMock(generated_videos=generated) if done else None,
        result=None,
    )
    # `name` is reserved by the MagicMock constructor (it names the mock), so it
    # must be configured afterwards to become a real attribute.
    operation.configure_mock(name="operations/abc123")
    return operation


# ---------------------------------------------------------------------------
# Conformance
# ---------------------------------------------------------------------------


class TestConformance:
    def test_is_media_capable_and_a_job_poller(self, provider):
        assert isinstance(provider, MediaCapableProvider)
        assert isinstance(provider, MediaJobPoller)

    @pytest.mark.parametrize(
        "protocol",
        [
            ImageGenerationProvider,
            ImageEditProvider,
            ImageUpscaleProvider,
            TTSProvider,
            VideoGenerationProvider,
        ],
    )
    def test_implements_protocols(self, provider, protocol):
        assert isinstance(provider, protocol)

    def test_declared_capabilities_developer_api(self, provider):
        assert provider.media_capabilities() == frozenset(
            {
                MediaCapability.IMAGE_GENERATE,
                MediaCapability.TTS,
                MediaCapability.VIDEO_GENERATE,
            }
        )

    def test_declared_capabilities_vertex(self, vertex_provider):
        assert vertex_provider.media_capabilities() == frozenset(
            {
                MediaCapability.IMAGE_GENERATE,
                MediaCapability.IMAGE_EDIT,
                MediaCapability.IMAGE_UPSCALE,
                MediaCapability.TTS,
                MediaCapability.VIDEO_GENERATE,
            }
        )

    @pytest.mark.parametrize("mode", ["developer", "vertex"])
    def test_every_declared_capability_is_backed(self, provider, vertex_provider, mode):
        provider = vertex_provider if mode == "vertex" else provider
        unmet = [
            c.value
            for c in provider.media_capabilities()
            if not isinstance(provider, CAPABILITY_PROTOCOLS[c])
        ]
        assert unmet == []

    def test_video_is_the_only_async_job(self, provider):
        assert (
            provider.media_execution(MediaCapability.VIDEO_GENERATE)
            is MediaExecution.ASYNC_JOB
        )
        for cap in (MediaCapability.IMAGE_GENERATE, MediaCapability.TTS):
            assert provider.media_execution(cap) is MediaExecution.REQUEST_RESPONSE

    def test_developer_api_hides_vertex_only_capabilities(self, provider):
        """Verified live: the Developer API rejects Imagen's edit/upscale endpoints.

        Declaring them unconditionally would make the router call an endpoint
        that always errors with "only supported in Gemini Enterprise Agent
        Platform mode".
        """
        assert provider._vertex_ai is False
        caps = provider.media_capabilities()
        assert MediaCapability.IMAGE_EDIT not in caps
        assert MediaCapability.IMAGE_UPSCALE not in caps
        # ...but the capabilities that DO work in both modes are present.
        assert MediaCapability.IMAGE_GENERATE in caps
        assert MediaCapability.VIDEO_GENERATE in caps
        assert MediaCapability.TTS in caps

    def test_vertex_mode_exposes_the_full_image_surface(self, vertex_provider):
        caps = vertex_provider.media_capabilities()
        assert MediaCapability.IMAGE_EDIT in caps
        assert MediaCapability.IMAGE_UPSCALE in caps

    def test_model_defaults_are_configurable(self):
        with patch("llmcore.providers.gemini_provider.genai") as genai_mod:
            genai_mod.Client.return_value = MagicMock()
            from llmcore.providers.gemini_provider import GeminiProvider

            p = GeminiProvider({"api_key": "k", "default_video_model": "veo-custom"})
        assert p._media_model("video", None) == "veo-custom"
        assert p._media_model("video", "explicit") == "explicit"


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


class TestImageGeneration:
    """The Imagen (Vertex) path; the Developer-API path is TestDualImagePath."""

    async def test_generates_and_normalizes(self, vertex_provider):
        vertex_provider._client.aio.models.generate_images = AsyncMock(return_value=_image_response(2))
        result = await vertex_provider.generate_image_media("a tabby", n=2)
        assert result.capability is MediaCapability.IMAGE_GENERATE
        assert len(result.artifacts) == 2
        assert result.artifacts[0].data == b"PNGDATA"
        assert result.artifacts[0].checksum_sha256
        assert result.usage.images == 2

    async def test_negative_prompt_is_forwarded(self, vertex_provider):
        """Imagen supports it natively — unlike OpenAI, it must not be dropped."""
        vertex_provider._client.aio.models.generate_images = AsyncMock(return_value=_image_response())
        await vertex_provider.generate_image_media("x", negative_prompt="blurry", size="2K")
        config = vertex_provider._client.aio.models.generate_images.call_args.kwargs["config"]
        assert config.negative_prompt == "blurry"
        assert config.image_size == "2K"

    async def test_seed_is_ignored(self, vertex_provider):
        vertex_provider._client.aio.models.generate_images = AsyncMock(return_value=_image_response())
        await vertex_provider.generate_image_media("x", seed=7)
        config = vertex_provider._client.aio.models.generate_images.call_args.kwargs["config"]
        assert not hasattr(config, "seed") or config.seed is None

    async def test_provenance_is_declared(self, vertex_provider):
        """Google watermarks generated imagery; that belongs on the artifact."""
        vertex_provider._client.aio.models.generate_images = AsyncMock(return_value=_image_response())
        result = await vertex_provider.generate_image_media("x")
        assert result.artifact.provenance.watermarked is True
        assert result.artifact.provenance.generator == "google"

    async def test_reference_images_route_to_edit(self, vertex_provider):
        vertex_provider.edit_image_media = AsyncMock(return_value="edited")
        ref = MediaRef.from_bytes(b"png", mime_type="image/png")
        assert await vertex_provider.generate_image_media("x", reference_images=[ref]) == "edited"

    async def test_api_error_maps_to_provider_error(self, vertex_provider):
        vertex_provider._client.aio.models.generate_images = AsyncMock(side_effect=RuntimeError("quota"))
        with pytest.raises(ProviderError, match="Image generation failed"):
            await vertex_provider.generate_image_media("x")


class TestDualImagePath:
    """Image generation has two transports; the auth mode picks one."""

    async def test_developer_api_uses_generate_content(self, provider):
        part = MagicMock(inline_data=MagicMock(data=b"PNGBYTES", mime_type="image/png"))
        provider._client.aio.models.generate_content = AsyncMock(
            return_value=MagicMock(candidates=[MagicMock(content=MagicMock(parts=[part]))])
        )
        provider._client.aio.models.generate_images = AsyncMock()

        result = await provider.generate_image_media("a red panda")

        provider._client.aio.models.generate_images.assert_not_called()
        config = provider._client.aio.models.generate_content.call_args.kwargs["config"]
        assert config.response_modalities == ["IMAGE"]
        assert result.artifact.data == b"PNGBYTES"
        assert result.raw["transport"] == "generate_content"
        assert result.model == "gemini-2.5-flash-image"

    async def test_vertex_uses_the_imagen_endpoint(self, vertex_provider):
        vertex_provider._client.aio.models.generate_images = AsyncMock(
            return_value=_image_response()
        )
        vertex_provider._client.aio.models.generate_content = AsyncMock()

        result = await vertex_provider.generate_image_media("x", negative_prompt="blurry")

        vertex_provider._client.aio.models.generate_content.assert_not_called()
        config = vertex_provider._client.aio.models.generate_images.call_args.kwargs["config"]
        assert config.negative_prompt == "blurry"
        assert result.model.startswith("imagen")

    async def test_developer_path_still_marks_provenance(self, provider):
        part = MagicMock(inline_data=MagicMock(data=b"P", mime_type="image/png"))
        provider._client.aio.models.generate_content = AsyncMock(
            return_value=MagicMock(candidates=[MagicMock(content=MagicMock(parts=[part]))])
        )
        result = await provider.generate_image_media("x")
        assert result.artifact.provenance.watermarked is True

    async def test_developer_path_handles_an_empty_response(self, provider):
        provider._client.aio.models.generate_content = AsyncMock(
            return_value=MagicMock(candidates=[])
        )
        result = await provider.generate_image_media("x")
        assert result.artifacts == ()


class TestImageEditAndUpscale:
    # Imagen's edit/upscale endpoints are Vertex-only.
    async def test_edit_builds_reference_images(self, vertex_provider):
        vertex_provider._client.aio.models.edit_image = AsyncMock(return_value=_image_response())
        result = await vertex_provider.edit_image_media(
            "night", image=MediaRef.from_bytes(b"IMG", mime_type="image/png")
        )
        assert result.capability is MediaCapability.IMAGE_EDIT
        refs = vertex_provider._client.aio.models.edit_image.call_args.kwargs["reference_images"]
        assert len(refs) == 1
        assert refs[0].reference_image.image_bytes == b"IMG"

    async def test_edit_adds_a_mask_reference(self, vertex_provider):
        vertex_provider._client.aio.models.edit_image = AsyncMock(return_value=_image_response())
        await vertex_provider.edit_image_media(
            "x", image=MediaRef.from_bytes(b"IMG"), mask=MediaRef.from_bytes(b"MASK")
        )
        refs = vertex_provider._client.aio.models.edit_image.call_args.kwargs["reference_images"]
        assert len(refs) == 2
        assert refs[1].reference_image.image_bytes == b"MASK"

    async def test_remote_ref_is_fetched(self, vertex_provider):
        vertex_provider._client.aio.models.edit_image = AsyncMock(return_value=_image_response())

        async def fake_fetch(url):
            return b"DOWNLOADED"

        with patch("llmcore.media.artifacts.default_fetcher", return_value=fake_fetch):
            await vertex_provider.edit_image_media("x", image=MediaRef.from_url("https://x/i.png"))
        refs = vertex_provider._client.aio.models.edit_image.call_args.kwargs["reference_images"]
        assert refs[0].reference_image.image_bytes == b"DOWNLOADED"

    async def test_upscale_maps_scale_to_a_factor(self, vertex_provider):
        vertex_provider._client.aio.models.upscale_image = AsyncMock(return_value=_image_response())
        await vertex_provider.upscale_image_media(image=MediaRef.from_bytes(b"i"), scale=4)
        assert vertex_provider._client.aio.models.upscale_image.call_args.kwargs["upscale_factor"] == "x4"

    async def test_upscale_defaults_to_x2(self, vertex_provider):
        vertex_provider._client.aio.models.upscale_image = AsyncMock(return_value=_image_response())
        await vertex_provider.upscale_image_media(image=MediaRef.from_bytes(b"i"))
        assert vertex_provider._client.aio.models.upscale_image.call_args.kwargs["upscale_factor"] == "x2"


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------


class TestSpeech:
    @staticmethod
    def _audio_response(data: bytes = b"PCMDATA") -> Any:
        part = MagicMock(inline_data=MagicMock(data=data, mime_type="audio/L16;rate=24000"))
        return MagicMock(candidates=[MagicMock(content=MagicMock(parts=[part]))])

    async def test_extracts_inline_audio(self, provider):
        provider._client.aio.models.generate_content = AsyncMock(
            return_value=self._audio_response()
        )
        result = await provider.synthesize_speech_media("hello", voice="Puck")
        assert result.capability is MediaCapability.TTS
        assert result.artifact.kind is MediaKind.AUDIO
        assert result.artifact.data == b"PCMDATA"
        assert result.artifact.sample_rate_hz == 24000
        assert result.usage.characters == 5

    async def test_voice_is_passed_in_the_speech_config(self, provider):
        provider._client.aio.models.generate_content = AsyncMock(
            return_value=self._audio_response()
        )
        await provider.synthesize_speech_media("hi", voice="Puck")
        config = provider._client.aio.models.generate_content.call_args.kwargs["config"]
        assert config.response_modalities == ["AUDIO"]
        assert config.speech_config.voice_config.prebuilt_voice_config.voice_name == "Puck"

    async def test_unsupported_options_are_not_forwarded(self, provider):
        """Gemini TTS returns raw PCM: no container format, no speed."""
        provider._client.aio.models.generate_content = AsyncMock(
            return_value=self._audio_response()
        )
        await provider.synthesize_speech_media("hi", audio_format="mp3", speed=1.5)
        config = provider._client.aio.models.generate_content.call_args.kwargs["config"]
        assert getattr(config, "speed", None) is None

    async def test_empty_response_yields_an_empty_artifact(self, provider):
        provider._client.aio.models.generate_content = AsyncMock(
            return_value=MagicMock(candidates=[])
        )
        result = await provider.synthesize_speech_media("hi")
        assert result.artifact.data == b""


# ---------------------------------------------------------------------------
# Veo — the async-job validation
# ---------------------------------------------------------------------------


class TestVideoSubmission:
    async def test_returns_a_job_not_a_result(self, provider):
        provider._client.aio.models.generate_videos = AsyncMock(return_value=_video_operation())
        job = await provider.generate_video_media("dunes")
        assert job.capability is MediaCapability.VIDEO_GENERATE
        assert job.status is MediaJobStatus.RUNNING
        assert job.provider_job_id == "operations/abc123"
        assert "operation" in job.provider_metadata

    async def test_maps_video_config(self, provider):
        provider._client.aio.models.generate_videos = AsyncMock(return_value=_video_operation())
        await provider.generate_video_media(
            "x", duration_seconds=8, fps=24, aspect_ratio="16:9", with_audio=True
        )
        config = provider._client.aio.models.generate_videos.call_args.kwargs["config"]
        assert config.duration_seconds == 8
        assert config.fps == 24
        assert config.aspect_ratio == "16:9"
        assert config.generate_audio is True

    async def test_uses_the_non_deprecated_source_argument(self, provider):
        """google-genai deprecated prompt=/image= in favour of source=."""
        provider._client.aio.models.generate_videos = AsyncMock(return_value=_video_operation())
        await provider.generate_video_media("dunes")
        kwargs = provider._client.aio.models.generate_videos.call_args.kwargs
        assert "prompt" not in kwargs and "image" not in kwargs
        assert kwargs["source"].prompt == "dunes"

    async def test_first_frame_conditions_the_opening_image(self, provider):
        provider._client.aio.models.generate_videos = AsyncMock(return_value=_video_operation())
        await provider.generate_video_media(
            "x", first_frame=MediaRef.from_bytes(b"START", mime_type="image/png")
        )
        source = provider._client.aio.models.generate_videos.call_args.kwargs["source"]
        assert source.image.image_bytes == b"START"

    async def test_last_frame_is_a_generative_transition(self, provider):
        """Distinct from interpolation: it invents content toward that frame."""
        provider._client.aio.models.generate_videos = AsyncMock(return_value=_video_operation())
        await provider.generate_video_media("x", last_frame=MediaRef.from_bytes(b"END"))
        config = provider._client.aio.models.generate_videos.call_args.kwargs["config"]
        assert config.last_frame.image_bytes == b"END"

    async def test_immediately_done_operation_succeeds(self, provider):
        provider._client.aio.models.generate_videos = AsyncMock(
            return_value=_video_operation(done=True, videos=1)
        )
        job = await provider.generate_video_media("x")
        assert job.succeeded and job.artifacts[0].data == b"MP4"


class TestVideoJobLifecycle:
    async def test_poll_transitions_running_to_succeeded(self, provider):
        provider._client.aio.models.generate_videos = AsyncMock(return_value=_video_operation())
        job = await provider.generate_video_media("x")
        assert job.status is MediaJobStatus.RUNNING

        provider._client.aio.operations.get = AsyncMock(
            return_value=_video_operation(done=True, videos=1)
        )
        job = await provider.poll_media_job(job)
        assert job.succeeded
        assert job.progress == 1.0
        assert job.artifacts[0].kind is MediaKind.VIDEO
        assert job.artifacts[0].provenance.watermarked is True

    async def test_poll_maps_operation_error_to_failure(self, provider):
        provider._client.aio.models.generate_videos = AsyncMock(return_value=_video_operation())
        job = await provider.generate_video_media("x")
        provider._client.aio.operations.get = AsyncMock(
            return_value=_video_operation(error=MagicMock(message="safety block"))
        )
        job = await provider.poll_media_job(job)
        assert job.status is MediaJobStatus.FAILED
        assert "safety block" in (job.error or "")

    async def test_poll_on_terminal_job_is_a_noop(self, provider):
        provider._client.aio.models.generate_videos = AsyncMock(
            return_value=_video_operation(done=True, videos=1)
        )
        job = await provider.generate_video_media("x")
        provider._client.aio.operations.get = AsyncMock()
        assert (await provider.poll_media_job(job)) is job
        provider._client.aio.operations.get.assert_not_called()

    async def test_lost_operation_handle_fails_loudly(self, provider):
        provider._client.aio.models.generate_videos = AsyncMock(return_value=_video_operation())
        job = await provider.generate_video_media("x")
        job.provider_metadata.pop("operation")
        job = await provider.poll_media_job(job)
        assert job.status is MediaJobStatus.FAILED
        assert "cannot be polled" in (job.error or "")

    async def test_cancel_refuses_honestly(self, provider):
        """Veo cannot be cancelled; reporting success would imply billing stopped."""
        provider._client.aio.models.generate_videos = AsyncMock(return_value=_video_operation())
        job = await provider.generate_video_media("x")
        with pytest.raises(ProviderError, match="cannot be cancelled"):
            await provider.cancel_media_job(job)

    async def test_full_lifecycle_through_the_job_manager(self, provider):
        """The point of M4: drive a REAL vendor's operation shape end to end."""
        provider._client.aio.models.generate_videos = AsyncMock(return_value=_video_operation())
        media = MediaManager({"gemini": provider}, job_policy=FAST)

        job = await media.video.generate("a drone shot over dunes", duration_seconds=8)
        assert media.jobs.get(job.id) is job  # tracked on submission

        calls = {"n": 0}

        async def _get(_op):
            calls["n"] += 1
            return _video_operation(done=calls["n"] >= 2, videos=1)

        provider._client.aio.operations.get = _get
        result = await media.wait(job, timeout=5)
        assert calls["n"] == 2  # polled until done
        assert result.capability is MediaCapability.VIDEO_GENERATE
        assert result.artifacts[0].mime_type == "video/mp4"

    async def test_failed_job_raises_through_the_manager(self, provider):
        provider._client.aio.models.generate_videos = AsyncMock(return_value=_video_operation())
        media = MediaManager({"gemini": provider}, job_policy=FAST)
        job = await media.video.generate("x")
        provider._client.aio.operations.get = AsyncMock(
            return_value=_video_operation(error=MagicMock(message="blocked"))
        )
        with pytest.raises(MediaJobError, match="blocked"):
            await media.jobs.wait(job)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class TestEmbeddings:
    async def test_returns_an_openai_shaped_payload(self, provider):
        provider._client.aio.models.embed_content = AsyncMock(
            return_value=MagicMock(embeddings=[MagicMock(values=[0.1, 0.2])])
        )
        out = await provider.create_embeddings(["hello"])
        assert out["data"][0]["embedding"] == [0.1, 0.2]
        assert out["data"][0]["index"] == 0
        assert out["model"]

    async def test_dimensions_and_task_type_are_forwarded(self, provider):
        provider._client.aio.models.embed_content = AsyncMock(
            return_value=MagicMock(embeddings=[])
        )
        await provider.create_embeddings("x", dimensions=256, task_type="RETRIEVAL_DOCUMENT")
        config = provider._client.aio.models.embed_content.call_args.kwargs["config"]
        assert config.output_dimensionality == 256
        assert config.task_type == "RETRIEVAL_DOCUMENT"


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


class TestRouting:
    def test_gemini_is_the_default_video_route(self, provider):
        media = MediaManager({"gemini": provider})
        assert media.who_can(MediaCapability.VIDEO_GENERATE) == ["gemini"]
        assert media.resolve(MediaCapability.VIDEO_GENERATE) is provider

    def test_gemini_does_not_claim_asr(self, provider):
        media = MediaManager({"gemini": provider})
        assert media.who_can(MediaCapability.ASR) == []

    async def test_image_routes_through_the_router(self, provider):
        part = MagicMock(inline_data=MagicMock(data=b"P", mime_type="image/png"))
        provider._client.aio.models.generate_content = AsyncMock(
            return_value=MagicMock(candidates=[MagicMock(content=MagicMock(parts=[part]))])
        )
        media = MediaManager({"gemini": provider})
        assert (await media.images.generate("x")).provider == "gemini"

    def test_developer_mode_does_not_route_image_edit(self, provider):
        media = MediaManager({"gemini": provider})
        assert media.who_can(MediaCapability.IMAGE_EDIT) == []
