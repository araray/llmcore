# tests/media/test_elevenlabs_media_adapter.py
"""ElevenLabs as a media adapter (spec phase M6).

The gate for this phase is **consent and provenance as first-class metadata**.
Synthetic speech differs from every other media kind llmcore generates: a
cloned voice belongs to a person who either did or did not agree to it, and
ElevenLabs reports that state only on the *voice* resource. These tests pin the
behaviour that makes it reachable from the artifact, and — more importantly —
that "the provider said nothing" stays distinguishable from "the provider said
no".
"""

from __future__ import annotations

import base64
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from llmcore.exceptions import ConfigError, ProviderError
from llmcore.media import (
    MediaCapability,
    MediaExecution,
    MediaKind,
    MediaManager,
    MediaRef,
    VoiceConsent,
)
from llmcore.media.protocols import CAPABILITY_PROTOCOLS, MediaCapableProvider
from llmcore.providers.elevenlabs_provider import ElevenLabsProvider

API = "https://api.elevenlabs.io"
VOICE = "EXAVITQu4vr4xnSDxMaL"
BASE_CONFIG: dict[str, Any] = {
    "api_key": "el-test-key",
    "_instance_name": "elevenlabs",
    "backend": "httpx",
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("ELEVENLABS_API_KEY", "ELEVEN_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    yield


@pytest.fixture
def provider() -> ElevenLabsProvider:
    return ElevenLabsProvider(dict(BASE_CONFIG))


def _voice_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "voice_id": VOICE,
        "name": "Sarah",
        "category": "premade",
        "is_owner": False,
        "safety_control": None,
        "voice_verification": {
            "requires_verification": False,
            "is_verified": False,
            "verification_failures": [],
            "verification_attempts_count": 0,
        },
    }
    payload.update(overrides)
    return payload


def _mock_voice(**overrides: Any):
    return respx.get(f"{API}/v1/voices/{VOICE}").mock(
        return_value=httpx.Response(200, json=_voice_payload(**overrides))
    )


# ---------------------------------------------------------------------------
# Construction & conformance
# ---------------------------------------------------------------------------


class TestConstruction:
    @pytest.mark.parametrize("env_var", ["ELEVENLABS_API_KEY", "ELEVEN_API_KEY"])
    def test_key_from_either_env_spelling(self, monkeypatch, env_var):
        monkeypatch.setenv(env_var, "from-env")
        assert ElevenLabsProvider({"backend": "httpx"})._api_key == "from-env"

    def test_explicit_key_wins(self, monkeypatch):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "env")
        assert ElevenLabsProvider({"api_key": "explicit", "backend": "httpx"})._api_key == (
            "explicit"
        )

    def test_missing_key_raises(self):
        with pytest.raises(ConfigError, match="ElevenLabs API key not found"):
            ElevenLabsProvider({"backend": "httpx"})

    def test_backend_defaults_to_direct_rest(self, provider):
        assert provider._backend == "httpx"

    def test_auto_prefers_httpx(self):
        with patch.multiple(
            "llmcore.providers.elevenlabs_provider",
            httpx_available=True,
            elevenlabs_sdk_available=True,
        ):
            assert ElevenLabsProvider._resolve_backend("auto") == "httpx"

    def test_no_transport_raises(self):
        with patch.multiple(
            "llmcore.providers.elevenlabs_provider",
            httpx_available=False,
            elevenlabs_sdk_available=False,
        ):
            with pytest.raises(ConfigError, match="requires 'httpx'"):
                ElevenLabsProvider(dict(BASE_CONFIG))

    def test_models_are_configurable(self):
        p = ElevenLabsProvider({**BASE_CONFIG, "models": {"tts": "eleven_v3"}})
        assert p._model_for("tts", None) == "eleven_v3"
        assert p._model_for("tts", "explicit") == "explicit"
        assert p._model_for("asr", None) == "scribe_v1"  # untouched default


class TestConformance:
    def test_is_media_capable(self, provider):
        assert isinstance(provider, MediaCapableProvider)

    def test_declares_the_voice_spread(self, provider):
        assert {c.value for c in provider.media_capabilities()} == {
            "tts", "tts_stream", "asr", "sfx", "music", "voice_design",
        }

    def test_every_declared_capability_is_backed(self, provider):
        unmet = [
            c.value
            for c in provider.media_capabilities()
            if not isinstance(provider, CAPABILITY_PROTOCOLS[c])
        ]
        assert unmet == []

    def test_only_streaming_tts_is_a_stream(self, provider):
        for cap in provider.media_capabilities():
            expected = (
                MediaExecution.STREAM
                if cap is MediaCapability.TTS_STREAM
                else MediaExecution.REQUEST_RESPONSE
            )
            assert provider.media_execution(cap) is expected

    def test_nothing_is_an_async_job(self, provider):
        """ElevenLabs has no queue; a job handle would be a lie."""
        assert all(
            provider.media_execution(c) is not MediaExecution.ASYNC_JOB
            for c in provider.media_capabilities()
        )

    def test_chat_is_refused_with_a_pointer(self, provider):
        import asyncio

        from llmcore.models import Message, Role

        with pytest.raises(ProviderError, match=r"llm\.media"):
            asyncio.run(provider.chat_completion([Message(role=Role.USER, content="hi")]))


# ---------------------------------------------------------------------------
# The gate: consent as first-class metadata
# ---------------------------------------------------------------------------


class TestVoiceConsentSemantics:
    """``None`` means the provider said nothing. That is not ``False``."""

    def test_silence_is_not_consent(self):
        c = VoiceConsent()
        assert c.verification_satisfied is None
        assert c.is_cloned is None

    def test_premade_voice_needs_no_verification(self):
        c = VoiceConsent(category="premade", requires_verification=False, is_verified=False)
        assert c.verification_satisfied is True
        assert c.is_cloned is False

    def test_unverified_clone_is_not_satisfied(self):
        c = VoiceConsent(category="cloned", requires_verification=True, is_verified=False)
        assert c.verification_satisfied is False
        assert c.is_cloned is True

    def test_verified_clone_is_satisfied(self):
        c = VoiceConsent(category="cloned", requires_verification=True, is_verified=True)
        assert c.verification_satisfied is True

    @pytest.mark.parametrize(
        ("category", "cloned"),
        [
            ("cloned", True),
            ("professional", True),
            ("famous", True),
            ("premade", False),
            ("generated", False),
        ],
    )
    def test_which_categories_imitate_a_person(self, category, cloned):
        assert VoiceConsent(category=category).is_cloned is cloned

    def test_failures_are_preserved(self):
        c = VoiceConsent(
            requires_verification=True, is_verified=False, verification_failures=("no_captcha",)
        )
        assert c.verification_failures == ("no_captcha",)
        assert c.verification_satisfied is False


class TestConsentResolution:
    @respx.mock
    async def test_synthesis_carries_consent(self, provider):
        """The whole point: no second API call to learn whose voice this is."""
        _mock_voice()
        respx.post(f"{API}/v1/text-to-speech/{VOICE}").mock(
            return_value=httpx.Response(200, content=b"ID3audio")
        )
        result = await provider.synthesize_speech_media("hello")
        consent = result.artifacts[0].provenance.consent
        assert consent.voice_id == VOICE
        assert consent.voice_name == "Sarah"
        assert consent.category == "premade"
        assert consent.verification_satisfied is True
        assert consent.provider_declared is True
        await provider.close()

    @respx.mock
    async def test_unverified_clone_is_surfaced_not_hidden(self, provider):
        _mock_voice(
            category="cloned",
            voice_verification={
                "requires_verification": True,
                "is_verified": False,
                "verification_failures": ["captcha_failed"],
                "verification_attempts_count": 2,
            },
        )
        respx.post(f"{API}/v1/text-to-speech/{VOICE}").mock(
            return_value=httpx.Response(200, content=b"audio")
        )
        result = await provider.synthesize_speech_media("hello")
        consent = result.artifacts[0].provenance.consent
        assert consent.is_cloned is True
        assert consent.verification_satisfied is False
        assert consent.verification_failures == ("captcha_failed",)
        await provider.close()

    @respx.mock
    async def test_lookup_is_cached(self, provider):
        route = _mock_voice()
        respx.post(f"{API}/v1/text-to-speech/{VOICE}").mock(
            return_value=httpx.Response(200, content=b"audio")
        )
        await provider.synthesize_speech_media("one")
        await provider.synthesize_speech_media("two")
        assert route.call_count == 1
        await provider.close()

    @respx.mock
    async def test_refresh_bypasses_the_cache(self, provider):
        route = _mock_voice()
        await provider.get_voice_consent(VOICE)
        await provider.get_voice_consent(VOICE, refresh=True)
        assert route.call_count == 2
        await provider.close()

    @respx.mock
    async def test_failed_lookup_loses_metadata_not_audio(self, provider):
        """A consent lookup failure must not fail the synthesis the caller asked for.

        The resulting record reads as *we do not know*, which is honest, rather
        than as *this is fine*.
        """
        respx.get(f"{API}/v1/voices/{VOICE}").mock(return_value=httpx.Response(500, text="boom"))
        respx.post(f"{API}/v1/text-to-speech/{VOICE}").mock(
            return_value=httpx.Response(200, content=b"audio")
        )
        result = await provider.synthesize_speech_media("hello")
        consent = result.artifacts[0].provenance.consent
        assert result.artifacts[0].data == b"audio"
        assert consent.provider_declared is False
        assert consent.verification_satisfied is None
        assert consent.category is None
        await provider.close()

    @respx.mock
    async def test_consent_resolution_can_be_disabled(self):
        p = ElevenLabsProvider({**BASE_CONFIG, "resolve_consent": False})
        route = respx.get(f"{API}/v1/voices/{VOICE}")
        respx.post(f"{API}/v1/text-to-speech/{VOICE}").mock(
            return_value=httpx.Response(200, content=b"audio")
        )
        result = await p.synthesize_speech_media("hello")
        assert route.call_count == 0
        assert result.artifacts[0].provenance.consent is None
        await p.close()

    @respx.mock
    async def test_generated_audio_carries_no_consent_record(self, provider):
        """SFX and music are nobody's voice; an empty consent field would imply
        a question that does not apply."""
        respx.post(f"{API}/v1/sound-generation").mock(
            return_value=httpx.Response(200, content=b"sfx")
        )
        result = await provider.generate_sfx_media("a creaking door")
        assert result.artifacts[0].provenance.consent is None
        assert result.artifacts[0].provenance.generator == "eleven_text_to_sound_v2"
        await provider.close()


# ---------------------------------------------------------------------------
# Speech
# ---------------------------------------------------------------------------


class TestSpeech:
    @respx.mock
    async def test_tts_request_shape(self, provider):
        _mock_voice()
        route = respx.post(f"{API}/v1/text-to-speech/{VOICE}").mock(
            return_value=httpx.Response(200, content=b"audio")
        )
        await provider.synthesize_speech_media("hello", speed=1.2)
        import json

        body = json.loads(route.calls[0].request.content)
        assert body["text"] == "hello"
        assert body["model_id"] == "eleven_v4"
        assert body["voice_settings"]["speed"] == 1.2
        assert route.calls[0].request.url.params["output_format"] == "mp3_44100_128"
        assert route.calls[0].request.headers["xi-api-key"] == "el-test-key"
        await provider.close()

    @respx.mock
    @pytest.mark.parametrize(
        ("fmt", "mime", "rate"),
        [
            ("mp3_44100_128", "audio/mpeg", 44100),
            ("pcm_24000", "audio/pcm", 24000),
            ("opus_48000_128", "audio/opus", 48000),
            ("ulaw_8000", "audio/basic", 8000),
        ],
    )
    async def test_output_format_drives_mime_and_rate(self, fmt, mime, rate):
        p = ElevenLabsProvider({**BASE_CONFIG, "output_format": fmt, "resolve_consent": False})
        respx.post(f"{API}/v1/text-to-speech/{VOICE}").mock(
            return_value=httpx.Response(200, content=b"audio")
        )
        artifact = (await p.synthesize_speech_media("x")).artifacts[0]
        assert artifact.mime_type == mime
        assert artifact.sample_rate_hz == rate
        await p.close()

    @respx.mock
    async def test_usage_counts_characters_not_tokens(self, provider):
        _mock_voice()
        respx.post(f"{API}/v1/text-to-speech/{VOICE}").mock(
            return_value=httpx.Response(200, content=b"audio")
        )
        result = await provider.synthesize_speech_media("hello")
        assert result.usage.basis == "per_character"
        assert result.usage.characters == 5
        await provider.close()

    @respx.mock
    async def test_streaming_yields_chunks(self, provider):
        respx.post(f"{API}/v1/text-to-speech/{VOICE}/stream").mock(
            return_value=httpx.Response(200, content=b"abcdefgh")
        )
        chunks = [c async for c in provider.stream_speech_media("hello")]
        assert b"".join(chunks) == b"abcdefgh"
        await provider.close()

    @respx.mock
    async def test_streaming_maps_errors(self, provider):
        respx.post(f"{API}/v1/text-to-speech/{VOICE}/stream").mock(
            return_value=httpx.Response(422, text="bad voice settings")
        )
        with pytest.raises(ProviderError, match="rejected the request parameters"):
            [c async for c in provider.stream_speech_media("hello")]
        await provider.close()

    @respx.mock
    async def test_voice_override_per_call(self, provider):
        respx.get(f"{API}/v1/voices/other").mock(
            return_value=httpx.Response(200, json=_voice_payload(voice_id="other", name="Other"))
        )
        route = respx.post(f"{API}/v1/text-to-speech/other").mock(
            return_value=httpx.Response(200, content=b"audio")
        )
        result = await provider.synthesize_speech_media("x", voice="other")
        assert route.call_count == 1
        assert result.artifacts[0].provenance.consent.voice_name == "Other"
        await provider.close()


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


class TestTranscription:
    @respx.mock
    async def test_local_bytes_upload_as_multipart(self, provider):
        route = respx.post(f"{API}/v1/speech-to-text").mock(
            return_value=httpx.Response(
                200, json={"text": "hello there", "language_code": "eng"}
            )
        )
        result = await provider.transcribe_media(
            audio=MediaRef.from_bytes(b"ID3", mime_type="audio/mpeg", filename="a.mp3")
        )
        assert result.artifacts[0].kind is MediaKind.TEXT
        assert result.artifacts[0].text == "hello there"
        assert b"multipart/form-data" in route.calls[0].request.headers["content-type"].encode()
        await provider.close()

    @respx.mock
    async def test_remote_audio_is_handed_over_as_a_url(self, provider):
        """The bytes must not round-trip through this process."""
        route = respx.post(f"{API}/v1/speech-to-text").mock(
            return_value=httpx.Response(200, json={"text": "remote"})
        )
        await provider.transcribe_media(
            audio=MediaRef.from_url("https://example.invalid/a.mp3")
        )
        from urllib.parse import parse_qs

        body = parse_qs(route.calls[0].request.content.decode())
        assert body["cloud_storage_url"] == ["https://example.invalid/a.mp3"]
        assert "multipart" not in route.calls[0].request.headers["content-type"]
        await provider.close()

    @respx.mock
    async def test_diarize_and_timestamps_map_to_api_fields(self, provider):
        route = respx.post(f"{API}/v1/speech-to-text").mock(
            return_value=httpx.Response(200, json={"text": "x"})
        )
        await provider.transcribe_media(
            audio=MediaRef.from_bytes(b"a"), diarize=True, timestamps=True, language="en"
        )
        body = route.calls[0].request.content.decode()
        assert "diarize" in body and "true" in body
        assert "word" in body
        assert "language_code" in body
        await provider.close()

    @respx.mock
    async def test_word_timings_are_preserved(self, provider):
        respx.post(f"{API}/v1/speech-to-text").mock(
            return_value=httpx.Response(
                200,
                json={
                    "text": "hi",
                    "language_code": "eng",
                    "language_probability": 0.95,
                    "words": [{"text": "hi", "start": 0.0, "end": 0.4}],
                },
            )
        )
        artifact = (
            await provider.transcribe_media(audio=MediaRef.from_bytes(b"a"))
        ).artifacts[0]
        assert artifact.provider_metadata["words"][0]["end"] == 0.4
        assert artifact.provider_metadata["language_probability"] == 0.95
        await provider.close()


# ---------------------------------------------------------------------------
# SFX, music, voice design
# ---------------------------------------------------------------------------


class TestSoundAndMusic:
    @respx.mock
    async def test_sfx_refuses_video_rather_than_ignoring_it(self, provider):
        """Silently dropping the video would return audio unrelated to the
        footage the caller passed."""
        with pytest.raises(ProviderError, match="text-conditioned only"):
            await provider.generate_sfx_media(
                "footsteps", video=MediaRef.from_url("https://e/x.mp4")
            )
        await provider.close()

    @respx.mock
    async def test_sfx_requires_a_prompt(self, provider):
        with pytest.raises(ProviderError, match="requires a text prompt"):
            await provider.generate_sfx_media(None)
        await provider.close()

    @respx.mock
    async def test_sfx_duration_is_passed(self, provider):
        route = respx.post(f"{API}/v1/sound-generation").mock(
            return_value=httpx.Response(200, content=b"sfx")
        )
        await provider.generate_sfx_media("a door", duration_seconds=3.0)
        import json

        assert json.loads(route.calls[0].request.content)["duration_seconds"] == 3.0
        await provider.close()

    @respx.mock
    async def test_music_converts_seconds_to_milliseconds(self, provider):
        """The protocol speaks seconds; the API takes milliseconds."""
        route = respx.post(f"{API}/v1/music").mock(
            return_value=httpx.Response(200, content=b"music")
        )
        await provider.generate_music_media("a motif", duration_seconds=10)
        import json

        body = json.loads(route.calls[0].request.content)
        assert body["music_length_ms"] == 10000
        assert body["model_id"] == "music_v2_5"
        await provider.close()


class TestVoiceDesign:
    @respx.mock
    async def test_previews_become_artifacts(self, provider):
        audio = base64.b64encode(b"preview-audio").decode()
        respx.post(f"{API}/v1/text-to-voice/design").mock(
            return_value=httpx.Response(
                200,
                json={
                    "text": "auto generated sample",
                    "previews": [
                        {
                            "audio_base_64": audio,
                            "generated_voice_id": "gen-1",
                            "media_type": "audio/mpeg",
                            "duration_secs": 4.2,
                        },
                        {"audio_base_64": audio, "generated_voice_id": "gen-2"},
                    ],
                },
            )
        )
        result = await provider.design_voice_media("a calm elderly storyteller with a rasp")
        assert len(result.artifacts) == 2
        assert result.artifacts[0].data == b"preview-audio"
        assert result.artifacts[0].duration_seconds == 4.2
        assert result.artifacts[0].provider_metadata["generated_voice_id"] == "gen-1"
        assert result.raw["text"] == "auto generated sample"
        await provider.close()

    @respx.mock
    async def test_designed_voices_state_their_provenance(self, provider):
        """A designed voice imitates no one — say so, rather than leaving the
        consent question open, which would read as 'unknown'."""
        respx.post(f"{API}/v1/text-to-voice/design").mock(
            return_value=httpx.Response(
                200,
                json={
                    "previews": [
                        {
                            "audio_base_64": base64.b64encode(b"a").decode(),
                            "generated_voice_id": "gen-1",
                        }
                    ]
                },
            )
        )
        consent = (
            await provider.design_voice_media("a calm storyteller with a slight rasp")
        ).artifacts[0].provenance.consent
        assert consent.category == "generated"
        assert consent.is_cloned is False
        assert consent.verification_satisfied is True
        assert consent.voice_id == "gen-1"
        await provider.close()

    @respx.mock
    async def test_previews_without_audio_are_skipped(self, provider):
        respx.post(f"{API}/v1/text-to-voice/design").mock(
            return_value=httpx.Response(200, json={"previews": [{"generated_voice_id": "x"}]})
        )
        result = await provider.design_voice_media("a calm storyteller with a rasp")
        assert result.artifacts == []
        await provider.close()

    @respx.mock
    async def test_auto_generate_text_unless_given(self, provider):
        route = respx.post(f"{API}/v1/text-to-voice/design").mock(
            return_value=httpx.Response(200, json={"previews": []})
        )
        await provider.design_voice_media("a calm storyteller with a rasp")
        import json

        assert json.loads(route.calls[0].request.content)["auto_generate_text"] is True

        await provider.design_voice_media("a calm storyteller with a rasp", text="Say this.")
        body = json.loads(route.calls[1].request.content)
        assert body["text"] == "Say this."
        assert "auto_generate_text" not in body
        await provider.close()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TestErrorMapping:
    @respx.mock
    @pytest.mark.parametrize(
        ("status", "body", "match"),
        [
            (401, "bad key", "authentication failed"),
            (404, "no voice", "not found"),
            (422, "bad param", "rejected the request parameters"),
            (429, "slow down", "rate limit"),
            (500, "boom", "API error"),
        ],
    )
    async def test_status_mapping(self, provider, status, body, match):
        respx.post(f"{API}/v1/sound-generation").mock(
            return_value=httpx.Response(status, text=body)
        )
        with pytest.raises(ProviderError, match=match):
            await provider.generate_sfx_media("x")
        await provider.close()

    @respx.mock
    @pytest.mark.parametrize(
        ("status", "body"),
        [
            (402, '{"detail":{"code":"paid_plan_required"}}'),
            (403, '{"detail":{"code":"feature_not_available"}}'),
        ],
    )
    async def test_plan_gating_is_not_reported_as_an_auth_failure(
        self, provider, status, body
    ):
        """Live-found: a valid key on a free plan returned 403 feature_not_available,
        which was being reported as 'check your API key' — sending the caller
        after a problem they do not have."""
        respx.post(f"{API}/v1/music").mock(return_value=httpx.Response(status, text=body))
        with pytest.raises(ProviderError) as exc:
            await provider.generate_music_media("x")
        message = str(exc.value)
        assert "plan" in message
        assert "the API key is valid" in message
        assert "ELEVENLABS_API_KEY" not in message
        assert exc.value.retryable is False
        await provider.close()

    @respx.mock
    async def test_a_genuine_403_still_reads_as_auth(self, provider):
        respx.post(f"{API}/v1/music").mock(
            return_value=httpx.Response(403, text='{"detail":"invalid api key"}')
        )
        with pytest.raises(ProviderError, match="authentication failed"):
            await provider.generate_music_media("x")
        await provider.close()

    @respx.mock
    async def test_server_errors_are_retryable(self, provider):
        respx.post(f"{API}/v1/sound-generation").mock(return_value=httpx.Response(503, text="x"))
        with pytest.raises(ProviderError) as exc:
            await provider.generate_sfx_media("x")
        assert exc.value.retryable is True
        await provider.close()


# ---------------------------------------------------------------------------
# Discovery & routing
# ---------------------------------------------------------------------------


class TestDiscovery:
    @respx.mock
    async def test_model_details_from_the_api(self, provider):
        respx.get(f"{API}/v1/models").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "model_id": "eleven_v4",
                        "name": "Eleven v4",
                        "can_do_text_to_speech": True,
                        "maximum_text_length_per_request": 10000,
                        "languages": [{"language_id": "en"}, {"language_id": "pt"}],
                    }
                ],
            )
        )
        details = await provider.get_models_details()
        assert details[0].id == "eleven_v4"
        assert details[0].context_length == 10000
        assert details[0].metadata["languages"] == ["en", "pt"]
        await provider.close()

    @respx.mock
    async def test_discovery_degrades_rather_than_failing(self, provider):
        """A model-list outage should not take down a whole session."""
        respx.get(f"{API}/v1/models").mock(side_effect=httpx.ConnectError("down"))
        details = await provider.get_models_details()
        assert {d.id for d in details} >= {"eleven_v4", "scribe_v1"}
        await provider.close()

    def test_routing_prefers_elevenlabs_for_voice(self, provider):
        media = MediaManager({"elevenlabs": provider})
        assert media.who_can(MediaCapability.TTS) == ["elevenlabs"]
        assert media.who_can(MediaCapability.MUSIC) == ["elevenlabs"]
        assert media.who_can(MediaCapability.VOICE_DESIGN) == ["elevenlabs"]

    @respx.mock
    async def test_router_dispatch_end_to_end(self, provider):
        _mock_voice()
        respx.post(f"{API}/v1/text-to-speech/{VOICE}").mock(
            return_value=httpx.Response(200, content=b"audio")
        )
        media = MediaManager({"elevenlabs": provider})
        result = await media.audio.speak("hello", provider="elevenlabs")
        assert result.artifacts[0].data == b"audio"
        assert result.artifacts[0].provenance.consent.category == "premade"
        await provider.close()


# ---------------------------------------------------------------------------
# SDK backend
# ---------------------------------------------------------------------------


class TestSdkBackend:
    @pytest.fixture
    def sdk_provider(self) -> ElevenLabsProvider:
        with patch(
            "llmcore.providers.elevenlabs_provider.elevenlabs_sdk"
        ) as mod, patch(
            "llmcore.providers.elevenlabs_provider.elevenlabs_sdk_available", True
        ):
            mod.AsyncElevenLabs.return_value = MagicMock()
            return ElevenLabsProvider({**BASE_CONFIG, "backend": "sdk"})

    def test_uses_the_sdk_client(self, sdk_provider):
        assert sdk_provider._backend == "sdk"
        assert sdk_provider._sdk is not None

    async def test_tts_goes_through_the_sdk(self, sdk_provider):
        async def _chunks(*_a, **_k):
            for piece in (b"ab", b"cd"):
                yield piece

        sdk_provider._sdk.text_to_speech.convert = _chunks
        sdk_provider._resolve_consent = False
        result = await sdk_provider.synthesize_speech_media("hello")
        assert result.artifacts[0].data == b"abcd"

    async def test_consent_lookup_goes_through_the_sdk(self, sdk_provider):
        voice = MagicMock()
        voice.model_dump.return_value = _voice_payload(category="cloned")
        sdk_provider._sdk.voices.get = AsyncMock(return_value=voice)
        consent = await sdk_provider.get_voice_consent(VOICE)
        assert consent.category == "cloned"
        assert consent.provider_declared is True

    async def test_transcription_goes_through_the_sdk(self, sdk_provider):
        result_obj = MagicMock()
        result_obj.model_dump.return_value = {"text": "sdk transcript"}
        sdk_provider._sdk.speech_to_text.convert = AsyncMock(return_value=result_obj)
        result = await sdk_provider.transcribe_media(audio=MediaRef.from_bytes(b"a"))
        assert result.artifacts[0].text == "sdk transcript"
