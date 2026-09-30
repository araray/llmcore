# src/llmcore/providers/elevenlabs_provider.py
"""ElevenLabs media provider for the LLMCore media subsystem.

ElevenLabs is the deepest *voice* vendor llmcore reaches: text-to-speech and
streaming TTS, batch speech-to-text, sound effects, music, and voice design —
generating candidate voices from a description rather than speech from text.

Why this adapter carries consent metadata
-----------------------------------------

Synthetic speech raises a question no other media kind does. A generated image
resembles no one in particular; a cloned voice belongs to a person who either
did or did not agree to it. ElevenLabs tracks that state per voice —
``category`` (premade / cloned / professional / generated), ``is_owner``,
``safety_control``, and a ``voice_verification`` record — but only on the
*voice* resource, not on the audio a caller gets back.

So a caller who wants to refuse audio from an unverified clone would have to
know to make a second API call. This adapter makes that unnecessary: every
synthesized artifact carries a :class:`~llmcore.media.VoiceConsent` on its
provenance, resolved from a short-lived voice cache. The spec's gate for this
phase is exactly that — *consent/provenance as first-class metadata*.

The tri-state matters. ``None`` means the provider said nothing, which is not
the same as ``False`` (the provider said no). Deciding that silence is
acceptable is a policy choice belonging to the caller, so
:attr:`VoiceConsent.verification_satisfied` reports ``None`` rather than
guessing.

Transport (selectable via ``backend``)
--------------------------------------

* ``"httpx"`` — direct REST against ``api.elevenlabs.io``. **The default**:
  these are plain multipart/JSON endpoints returning audio bytes, and it keeps
  the vendor SDK off the critical path.
* ``"sdk"`` — the official ``elevenlabs`` package.

References:
  - https://elevenlabs.io/docs/api-reference
  - SDK clone: /av/avalon/xrepos/elevenlabs-python (v2.70.0)
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator, AsyncIterator
from typing import TYPE_CHECKING, Any

try:
    import httpx

    httpx_available = True
except ImportError:  # pragma: no cover
    httpx_available = False
    httpx = None  # type: ignore

try:
    import elevenlabs as elevenlabs_sdk

    elevenlabs_sdk_available = True
except ImportError:
    elevenlabs_sdk_available = False
    elevenlabs_sdk = None  # type: ignore

from ..exceptions import ConfigError, ProviderError
from ..models import Message, ModelDetails, Tool
from .base import BaseProvider, ContextPayload

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..media.models import MediaCapability, MediaExecution, MediaRef, MediaResult

logger = logging.getLogger(__name__)

#: REST root.
_BASE_URL = "https://api.elevenlabs.io"

#: Environment variables checked, in order, for the credential. ElevenLabs'
#: own convention is ``ELEVENLABS_API_KEY``; ``ELEVEN_API_KEY`` is the older
#: spelling their SDK still honours.
_API_KEY_ENV_VARS: tuple[str, ...] = ("ELEVENLABS_API_KEY", "ELEVEN_API_KEY")

#: Per-capability default models, verified against the live ``/v1/models``
#: lineup on 2026-09-30. ``eleven_v4`` is GA (no alpha access, standard rate)
#: and covers 85 languages, so it is the default rather than an older release.
_DEFAULT_MODELS: dict[str, str] = {
    "tts": "eleven_v4",
    "tts_stream": "eleven_v4",
    "asr": "scribe_v1",
    "sfx": "eleven_text_to_sound_v2",
    "music": "music_v2_5",
    "voice_design": "eleven_ttv_v3",
}

#: A premade ElevenLabs voice ("Sarah"), used when no voice is configured.
#: Premade voices need no verification, so the default path is consent-clean.
_DEFAULT_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"

#: Default output encoding. ElevenLabs returns raw audio bytes, not JSON.
_DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"

#: ``output_format`` prefix → MIME type.
_FORMAT_MIME: dict[str, str] = {
    "mp3": "audio/mpeg",
    "pcm": "audio/pcm",
    "ulaw": "audio/basic",
    "alaw": "audio/x-alaw-basic",
    "opus": "audio/opus",
}


class ElevenLabsProvider(BaseProvider):
    """ElevenLabs voice/audio provider.

    Media-only by design: ElevenLabs serves speech and audio generation, not
    chat completions, so :meth:`chat_completion` raises rather than pretending.
    The real surface is ``llm.media`` (see ``docs/MEDIA_SUBSYSTEM_SPEC.md``).

    Configuration keys (under ``[providers.elevenlabs]``):

    - ``api_key`` / ``api_key_env_var`` — credential; falls back to
      ``ELEVENLABS_API_KEY`` then ``ELEVEN_API_KEY``.
    - ``backend`` — ``"httpx"`` (default) or ``"sdk"``.
    - ``base_url`` — override the REST root.
    - ``timeout`` — HTTP timeout in seconds (default 120).
    - ``voice_id`` — default voice for TTS.
    - ``output_format`` — default audio encoding (default ``mp3_44100_128``).
    - ``models`` — per-capability model overrides.
    - ``resolve_consent`` — whether to look up voice consent metadata and
      attach it to synthesized artifacts (default ``True``).
    """

    _MEDIA_CAPABILITIES: frozenset[str] = frozenset(
        {"tts", "tts_stream", "asr", "sfx", "music", "voice_design"}
    )

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False):
        """Initialize the ElevenLabs provider.

        Args:
            config: Provider configuration from ``[providers.elevenlabs]``.
            log_raw_payloads: Whether to log raw request/response payloads.

        Raises:
            ConfigError: If no transport is installed or no API key is found.
        """
        super().__init__(config, log_raw_payloads)

        if not (httpx_available or elevenlabs_sdk_available):
            raise ConfigError(
                "The ElevenLabs provider requires 'httpx' (direct REST, preferred) or "
                "the 'elevenlabs' SDK. Install with: pip install llmcore[elevenlabs]"
            )

        api_key = config.get("api_key")
        if not api_key:
            env_var = config.get("api_key_env_var")
            candidates = (env_var, *_API_KEY_ENV_VARS) if env_var else _API_KEY_ENV_VARS
            for name in candidates:
                if name and os.environ.get(name):
                    api_key = os.environ[name]
                    break
        if not api_key:
            raise ConfigError(
                "ElevenLabs API key not found. Set ELEVENLABS_API_KEY or configure "
                "providers.elevenlabs.api_key / api_key_env_var. Create a key at "
                "https://elevenlabs.io/app/settings/api-keys."
            )
        self._api_key = str(api_key)

        self._base_url = str(config.get("base_url", _BASE_URL)).rstrip("/")
        self._timeout = float(config.get("timeout", 120))
        self._voice_id = str(config.get("voice_id") or _DEFAULT_VOICE_ID)
        self._output_format = str(config.get("output_format") or _DEFAULT_OUTPUT_FORMAT)
        self._resolve_consent = bool(config.get("resolve_consent", True))
        self._models: dict[str, str] = {
            **_DEFAULT_MODELS,
            **{str(k): str(v) for k, v in (config.get("models") or {}).items()},
        }
        self.default_model = config.get("default_model") or self._models["tts"]

        self._backend = self._resolve_backend(config.get("backend"))
        self._http: Any = None
        self._sdk: Any = None
        if self._backend == "sdk":
            self._sdk = elevenlabs_sdk.AsyncElevenLabs(api_key=self._api_key)

        #: voice_id → VoiceConsent, populated lazily. Consent state changes
        #: rarely, and a lookup per synthesis would double the request count.
        self._consent_cache: dict[str, Any] = {}

        logger.debug(
            "ElevenLabs provider initialized (backend=%s, voice=%s, consent=%s).",
            self._backend,
            self._voice_id,
            self._resolve_consent,
        )

    @staticmethod
    def _resolve_backend(requested: str | None) -> str:
        """Resolve the transport, preferring direct REST."""
        available = {"httpx": httpx_available, "sdk": elevenlabs_sdk_available}
        req = (requested or "auto").lower()
        if req not in ("auto", "httpx", "sdk"):
            logger.warning("Unknown ElevenLabs backend '%s'; using auto-detection.", req)
            req = "auto"
        if req != "auto" and available.get(req):
            return req
        if req != "auto":
            logger.warning(
                "Requested ElevenLabs backend '%s' is unavailable; falling back.", req
            )
        for backend in ("httpx", "sdk"):
            if available[backend]:
                return backend
        raise ConfigError("No usable ElevenLabs transport is installed.")

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def _get_http(self) -> Any:
        """Return the lazily-built HTTP client."""
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self._base_url,
                headers={"xi-api-key": self._api_key},
                timeout=self._timeout,
            )
        return self._http

    def _raise_status(self, status: int, body: str, context: str) -> None:
        """Map an ElevenLabs HTTP error onto a :class:`ProviderError`."""
        logger.error("ElevenLabs error %s on %s: %s", status, context, body)

        # Plan gating must not be reported as an auth failure. ElevenLabs
        # answers 402, or 403 with a feature_not_available / paid_plan_required
        # code, when the credential is perfectly valid but the account's plan
        # does not include the feature — telling that caller to check their API
        # key sends them after a problem they do not have.
        plan_gated = status == 402 or (
            status == 403
            and any(
                code in body
                for code in ("feature_not_available", "paid_plan_required", "status_limited")
            )
        )
        if plan_gated:
            raise ProviderError(
                self.get_name(),
                f"ElevenLabs rejected '{context}' as unavailable on this account's plan "
                f"(the API key is valid). Music, voice design and some other features "
                f"require a paid plan. Error: {body}",
                model_name=context,
                status_code=status,
                retryable=False,
            )
        if status in (401, 403):
            raise ProviderError(
                self.get_name(),
                f"ElevenLabs authentication failed. Check ELEVENLABS_API_KEY. Error: {body}",
                model_name=context,
                status_code=status,
            )
        if status == 404:
            raise ProviderError(
                self.get_name(),
                f"ElevenLabs resource '{context}' not found. Check the voice or model id. "
                f"Error: {body}",
                model_name=context,
                status_code=status,
            )
        if status == 422:
            raise ProviderError(
                self.get_name(),
                f"ElevenLabs rejected the request parameters for '{context}': {body}",
                model_name=context,
                status_code=status,
            )
        if status == 429:
            raise ProviderError(
                self.get_name(),
                f"ElevenLabs rate limit or concurrency limit reached. Error: {body}",
                model_name=context,
                status_code=status,
                retryable=True,
            )
        raise ProviderError(
            self.get_name(),
            f"ElevenLabs API error ({status}): {body}",
            model_name=context,
            status_code=status,
            retryable=status >= 500,
        )

    def _model_for(self, capability: str, model: str | None) -> str:
        """Return the model id for *capability*, honouring an explicit override."""
        return model or self._models.get(capability) or self.default_model

    @staticmethod
    def _mime_for(output_format: str) -> str:
        """Map an ElevenLabs ``output_format`` onto a MIME type."""
        return _FORMAT_MIME.get(output_format.split("_")[0].lower(), "audio/mpeg")

    @staticmethod
    def _sample_rate_for(output_format: str) -> int | None:
        """Extract the sample rate encoded in an ``output_format`` string."""
        parts = output_format.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            return int(parts[1])
        return None

    # ------------------------------------------------------------------
    # BaseProvider surface (ElevenLabs is media-only)
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        """Return the configured instance name (default: ``"elevenlabs"``)."""
        return self._provider_instance_name or "elevenlabs"

    async def chat_completion(
        self,
        context: ContextPayload,
        model: str | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Not supported: ElevenLabs serves voice and audio, not chat.

        Raises:
            ProviderError: Always, with a pointer to the media surface.
        """
        raise ProviderError(
            self.get_name(),
            "ElevenLabs is a voice/audio provider and has no chat-completions API. "
            "Use llm.media.audio instead.",
            status_code=400,
            retryable=False,
        )

    async def get_models_details(self) -> list[ModelDetails]:
        """List the TTS models ElevenLabs advertises.

        Falls back to the configured defaults when the API is unreachable, so
        model discovery degrades rather than failing a whole session.
        """
        try:
            resp = await self._get_http().get("/v1/models")
            if resp.status_code >= 400:
                self._raise_status(resp.status_code, resp.text, "models")
            payload = resp.json()
        except ProviderError:
            raise
        except Exception as e:  # degraded discovery is deliberate
            logger.warning("ElevenLabs model discovery failed (%s); using defaults.", e)
            payload = [{"model_id": m} for m in sorted(set(self._models.values()))]

        return [
            ModelDetails(
                id=str(m.get("model_id")),
                provider_name=self.get_name(),
                context_length=int(m.get("maximum_text_length_per_request") or 0),
                supports_streaming=bool(m.get("can_do_text_to_speech")),
                supports_tools=False,
                model_type="media",
                metadata={
                    "name": m.get("name"),
                    "languages": [
                        lang.get("language_id") for lang in (m.get("languages") or [])
                    ],
                    "can_do_text_to_speech": m.get("can_do_text_to_speech"),
                    "requires_alpha_access": m.get("requires_alpha_access"),
                },
            )
            for m in payload
        ]

    def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
        """ElevenLabs parameters vary per capability, so none are advertised."""
        return {}

    def get_max_context_length(self, model: str | None = None) -> int:
        """ElevenLabs models take characters, not a token context window."""
        return 0

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """ElevenLabs bills per character, so tokens are not a meaningful unit."""
        return 0

    async def count_message_tokens(
        self, messages: list[Message], model: str | None = None
    ) -> int:
        """ElevenLabs bills per character, so tokens are not a meaningful unit."""
        return 0

    def extract_response_content(self, response: dict[str, Any]) -> str:
        """Return any text ElevenLabs produced (transcripts carry one)."""
        return str(response.get("text") or "")

    def extract_delta_content(self, chunk: dict[str, Any]) -> str:
        """ElevenLabs streams audio bytes, not token deltas."""
        return ""

    # ------------------------------------------------------------------
    # Media capability declaration
    # ------------------------------------------------------------------

    def media_capabilities(self) -> frozenset[MediaCapability]:
        """Return the media capabilities this provider serves."""
        from ..media.models import MediaCapability

        return frozenset(MediaCapability(c) for c in self._MEDIA_CAPABILITIES)

    def media_execution(
        self, capability: MediaCapability, model: str | None = None
    ) -> MediaExecution:
        """Report how *capability* executes.

        Everything here answers in one request; only streaming TTS is a byte
        stream. ElevenLabs has no queue, so nothing is an async job.
        """
        from ..media.models import MediaCapability, MediaExecution

        if capability is MediaCapability.TTS_STREAM:
            return MediaExecution.STREAM
        return MediaExecution.REQUEST_RESPONSE

    # ------------------------------------------------------------------
    # Consent
    # ------------------------------------------------------------------

    async def get_voice_consent(self, voice_id: str, *, refresh: bool = False) -> Any:
        """Return the :class:`VoiceConsent` ElevenLabs reports for *voice_id*.

        Results are cached for the life of the provider because consent state
        changes rarely and a lookup per synthesis would double the request
        count. Pass ``refresh=True`` after verifying a voice.

        A failed lookup yields a consent record with everything ``None`` and
        ``provider_declared=False``, rather than raising: the caller asked for
        speech, and losing the metadata should not lose the audio. The
        ``None``s then correctly read as *we do not know*.
        """
        from ..media.models import VoiceConsent

        if not refresh and voice_id in self._consent_cache:
            return self._consent_cache[voice_id]

        try:
            voice = await self._fetch_voice(voice_id)
        except Exception as e:  # metadata must not fail synthesis
            logger.warning("ElevenLabs consent lookup failed for %s: %s", voice_id, e)
            unknown = VoiceConsent(voice_id=voice_id, provider_declared=False)
            self._consent_cache[voice_id] = unknown
            return unknown

        verification = voice.get("voice_verification") or {}
        consent = VoiceConsent(
            voice_id=voice_id,
            voice_name=voice.get("name"),
            category=voice.get("category"),
            requires_verification=verification.get("requires_verification"),
            is_verified=verification.get("is_verified"),
            verification_failures=tuple(verification.get("verification_failures") or ()),
            is_owner=voice.get("is_owner"),
            safety_control=voice.get("safety_control"),
            provider_declared=True,
        )
        self._consent_cache[voice_id] = consent
        return consent

    async def _fetch_voice(self, voice_id: str) -> dict[str, Any]:
        """Fetch one voice resource."""
        if self._backend == "sdk":
            voice = await self._sdk.voices.get(voice_id)
            dump = getattr(voice, "model_dump", None)
            return dict(dump()) if callable(dump) else dict(voice)
        resp = await self._get_http().get(f"/v1/voices/{voice_id}")
        if resp.status_code >= 400:
            self._raise_status(resp.status_code, resp.text, f"voices/{voice_id}")
        return dict(resp.json())

    async def _provenance_for(self, voice_id: str | None, model: str) -> Any:
        """Build the provenance record attached to synthesized audio."""
        from ..media.models import MediaProvenance

        consent = None
        if voice_id and self._resolve_consent:
            consent = await self.get_voice_consent(voice_id)
        return MediaProvenance(generator=model, consent=consent, provider_declared=True)

    # ------------------------------------------------------------------
    # Speech
    # ------------------------------------------------------------------

    async def synthesize_speech_media(
        self,
        text: str,
        *,
        model: str | None = None,
        voice: str | None = None,
        audio_format: str | None = None,
        sample_rate_hz: int | None = None,
        speed: float | None = None,
        **kwargs: Any,
    ) -> MediaResult:
        """Synthesize *text* into a single audio artifact.

        The returned artifact carries a :class:`VoiceConsent` on its provenance
        describing the voice used, so a caller can gate on consent without a
        second API call.
        """
        from ..media.models import (
            MediaArtifact,
            MediaCapability,
            MediaKind,
            MediaResult,
            MediaUsage,
        )

        model_id = self._model_for("tts", model)
        voice_id = voice or self._voice_id
        output_format = audio_format or self._output_format

        body: dict[str, Any] = {"text": text, "model_id": model_id}
        voice_settings = dict(kwargs.pop("voice_settings", None) or {})
        if speed is not None:
            voice_settings["speed"] = speed
        if voice_settings:
            body["voice_settings"] = voice_settings
        body.update({k: v for k, v in kwargs.items() if v is not None})

        audio = await self._tts_bytes(voice_id, body, output_format, stream=False)

        artifact = MediaArtifact(
            kind=MediaKind.AUDIO,
            data=audio,
            mime_type=self._mime_for(output_format),
            sample_rate_hz=sample_rate_hz or self._sample_rate_for(output_format),
            provenance=await self._provenance_for(voice_id, model_id),
            provider_metadata={"voice_id": voice_id, "output_format": output_format},
        )
        return MediaResult(
            capability=MediaCapability.TTS,
            provider=self.get_name(),
            model=model_id,
            artifacts=[artifact],
            usage=MediaUsage(
                provider=self.get_name(),
                model=model_id,
                basis="per_character",
                characters=len(text),
            ),
        )

    def stream_speech_media(
        self,
        text: str,
        *,
        model: str | None = None,
        voice: str | None = None,
        audio_format: str | None = None,
        sample_rate_hz: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[bytes]:
        """Yield audio chunks as they are synthesized.

        A plain method returning an async iterator, per the protocol — callers
        write ``async for chunk in provider.stream_speech_media(...)``.
        """
        model_id = self._model_for("tts_stream", model)
        voice_id = voice or self._voice_id
        output_format = audio_format or self._output_format
        body: dict[str, Any] = {"text": text, "model_id": model_id}
        body.update({k: v for k, v in kwargs.items() if v is not None})
        return self._tts_stream(voice_id, body, output_format)

    async def _tts_bytes(
        self, voice_id: str, body: dict[str, Any], output_format: str, *, stream: bool
    ) -> bytes:
        """Run a one-shot TTS request and return the audio bytes."""
        if self._backend == "sdk":
            chunks = self._sdk.text_to_speech.convert(
                voice_id=voice_id,
                text=body["text"],
                model_id=body.get("model_id"),
                output_format=output_format,
            )
            return b"".join([c async for c in chunks])

        resp = await self._get_http().post(
            f"/v1/text-to-speech/{voice_id}",
            params={"output_format": output_format},
            json=body,
        )
        if resp.status_code >= 400:
            self._raise_status(resp.status_code, resp.text, f"text-to-speech/{voice_id}")
        return resp.content

    async def _tts_stream(
        self, voice_id: str, body: dict[str, Any], output_format: str
    ) -> AsyncIterator[bytes]:
        """Stream TTS audio chunks."""
        if self._backend == "sdk":
            chunks = self._sdk.text_to_speech.stream(
                voice_id=voice_id,
                text=body["text"],
                model_id=body.get("model_id"),
                output_format=output_format,
            )
            async for chunk in chunks:
                if chunk:
                    yield chunk
            return

        client = self._get_http()
        async with client.stream(
            "POST",
            f"/v1/text-to-speech/{voice_id}/stream",
            params={"output_format": output_format},
            json=body,
        ) as resp:
            if resp.status_code >= 400:
                await resp.aread()
                self._raise_status(
                    resp.status_code, resp.text, f"text-to-speech/{voice_id}/stream"
                )
            async for chunk in resp.aiter_bytes():
                if chunk:
                    yield chunk

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    async def transcribe_media(
        self,
        *,
        audio: MediaRef,
        model: str | None = None,
        language: str | None = None,
        diarize: bool | None = None,
        timestamps: bool | None = None,
        **kwargs: Any,
    ) -> MediaResult:
        """Transcribe *audio*; the text lands on the artifact's ``text`` field."""
        from ..media.models import (
            MediaArtifact,
            MediaCapability,
            MediaKind,
            MediaResult,
            MediaUsage,
        )

        model_id = self._model_for("asr", model)
        data: dict[str, Any] = {"model_id": model_id}
        if language:
            data["language_code"] = language
        if diarize is not None:
            data["diarize"] = str(bool(diarize)).lower()
        if timestamps is not None:
            data["timestamps_granularity"] = "word" if timestamps else "none"
        data.update({k: str(v) for k, v in kwargs.items() if v is not None})

        payload = await self._transcribe(audio, data)

        words = payload.get("words") or []
        artifact = MediaArtifact(
            kind=MediaKind.TEXT,
            text=payload.get("text"),
            mime_type="text/plain",
            provider_metadata={
                "language_code": payload.get("language_code"),
                "language_probability": payload.get("language_probability"),
                "words": words,
            },
        )
        return MediaResult(
            capability=MediaCapability.ASR,
            provider=self.get_name(),
            model=model_id,
            artifacts=[artifact],
            usage=MediaUsage(
                provider=self.get_name(), model=model_id, basis="per_request", raw=payload
            ),
        )

    async def _transcribe(self, audio: MediaRef, data: dict[str, Any]) -> dict[str, Any]:
        """Post audio to the speech-to-text endpoint.

        A remote ``MediaRef`` is handed to ElevenLabs as a ``cloud_storage_url``
        so the bytes never round-trip through this process; local refs are
        uploaded as multipart.
        """
        if audio.is_remote and audio.url:
            data = {**data, "cloud_storage_url": audio.url}
            if self._backend == "sdk":
                result = await self._sdk.speech_to_text.convert(**data)
                return self._as_dict(result)
            resp = await self._get_http().post("/v1/speech-to-text", data=data)
            if resp.status_code >= 400:
                self._raise_status(resp.status_code, resp.text, "speech-to-text")
            return dict(resp.json())

        raw = audio.read_bytes()
        filename = audio.filename or "audio.mp3"
        mime = audio.mime_type or "audio/mpeg"

        if self._backend == "sdk":
            result = await self._sdk.speech_to_text.convert(file=(filename, raw, mime), **data)
            return self._as_dict(result)

        resp = await self._get_http().post(
            "/v1/speech-to-text", data=data, files={"file": (filename, raw, mime)}
        )
        if resp.status_code >= 400:
            self._raise_status(resp.status_code, resp.text, "speech-to-text")
        return dict(resp.json())

    @staticmethod
    def _as_dict(obj: Any) -> dict[str, Any]:
        """Normalize an SDK model into a plain dict."""
        dump = getattr(obj, "model_dump", None)
        if callable(dump):
            return dict(dump())
        return dict(obj) if isinstance(obj, dict) else {"text": getattr(obj, "text", None)}

    # ------------------------------------------------------------------
    # Sound effects and music
    # ------------------------------------------------------------------

    async def generate_sfx_media(
        self,
        prompt: str | None = None,
        *,
        model: str | None = None,
        video: MediaRef | None = None,
        duration_seconds: float | None = None,
        **kwargs: Any,
    ) -> MediaResult:
        """Generate sound effects from *prompt*.

        ElevenLabs generates from text only. *video* is accepted to satisfy the
        protocol and refused explicitly rather than silently ignored, because
        dropping it would return audio unrelated to the footage the caller
        passed — use a foley-capable provider (fal's MMAudio) for that.
        """
        from ..media.models import MediaCapability

        if video is not None:
            raise ProviderError(
                self.get_name(),
                "ElevenLabs sound generation is text-conditioned only and cannot take a "
                "video reference. Route video-conditioned foley to a provider that "
                "supports it (e.g. fal's MMAudio).",
                status_code=400,
                retryable=False,
            )
        if not prompt:
            raise ProviderError(
                self.get_name(),
                "ElevenLabs sound generation requires a text prompt.",
                status_code=400,
                retryable=False,
            )

        model_id = self._model_for("sfx", model)
        body: dict[str, Any] = {"text": prompt, "model_id": model_id}
        if duration_seconds is not None:
            body["duration_seconds"] = duration_seconds
        body.update({k: v for k, v in kwargs.items() if v is not None})

        audio = await self._audio_post("/v1/sound-generation", body, "sound-generation")
        return self._audio_result(MediaCapability.SFX, model_id, audio, prompt)

    async def generate_music_media(
        self,
        prompt: str,
        *,
        model: str | None = None,
        duration_seconds: float | None = None,
        **kwargs: Any,
    ) -> MediaResult:
        """Generate music from *prompt*."""
        from ..media.models import MediaCapability

        model_id = self._model_for("music", model)
        body: dict[str, Any] = {"prompt": prompt, "model_id": model_id}
        if duration_seconds is not None:
            # The API takes milliseconds; the protocol speaks seconds.
            body["music_length_ms"] = int(duration_seconds * 1000)
        body.update({k: v for k, v in kwargs.items() if v is not None})

        audio = await self._audio_post("/v1/music", body, "music")
        return self._audio_result(MediaCapability.MUSIC, model_id, audio, prompt)

    async def _audio_post(self, path: str, body: dict[str, Any], context: str) -> bytes:
        """POST a JSON body to an endpoint that answers with audio bytes."""
        if self._backend == "sdk":
            if context == "music":
                chunks = self._sdk.music.compose(
                    prompt=body.get("prompt"),
                    model_id=body.get("model_id"),
                    music_length_ms=body.get("music_length_ms"),
                    output_format=self._output_format,
                )
            else:
                chunks = self._sdk.text_to_sound_effects.convert(
                    text=body.get("text"),
                    model_id=body.get("model_id"),
                    duration_seconds=body.get("duration_seconds"),
                    output_format=self._output_format,
                )
            return b"".join([c async for c in chunks])

        resp = await self._get_http().post(
            path, params={"output_format": self._output_format}, json=body
        )
        if resp.status_code >= 400:
            self._raise_status(resp.status_code, resp.text, context)
        return resp.content

    def _audio_result(
        self, capability: Any, model_id: str, audio: bytes, prompt: str | None
    ) -> MediaResult:
        """Wrap generated audio bytes in a :class:`MediaResult`."""
        from ..media.models import (
            MediaArtifact,
            MediaKind,
            MediaProvenance,
            MediaResult,
            MediaUsage,
        )

        artifact = MediaArtifact(
            kind=MediaKind.AUDIO,
            data=audio,
            mime_type=self._mime_for(self._output_format),
            sample_rate_hz=self._sample_rate_for(self._output_format),
            # No consent record: nothing here is anyone's voice.
            provenance=MediaProvenance(generator=model_id, provider_declared=True),
            provider_metadata={"prompt": prompt, "output_format": self._output_format},
        )
        return MediaResult(
            capability=capability,
            provider=self.get_name(),
            model=model_id,
            artifacts=[artifact],
            usage=MediaUsage(
                provider=self.get_name(), model=model_id, basis="per_request"
            ),
        )

    # ------------------------------------------------------------------
    # Voice design
    # ------------------------------------------------------------------

    async def design_voice_media(
        self,
        description: str,
        *,
        model: str | None = None,
        text: str | None = None,
        **kwargs: Any,
    ) -> MediaResult:
        """Generate candidate voices matching *description*.

        Returns one artifact per preview, each carrying the
        ``generated_voice_id`` needed to keep it. A designed voice is
        synthetic rather than cloned from a person, so each preview's consent
        record says so explicitly instead of leaving the question open.
        """
        from ..media.models import (
            MediaArtifact,
            MediaCapability,
            MediaKind,
            MediaProvenance,
            MediaResult,
            MediaUsage,
            VoiceConsent,
        )

        model_id = self._model_for("voice_design", model)
        body: dict[str, Any] = {"voice_description": description, "model_id": model_id}
        if text:
            body["text"] = text
        else:
            body["auto_generate_text"] = True
        body.update({k: v for k, v in kwargs.items() if v is not None})

        payload = await self._design_voice(body)

        artifacts: list[MediaArtifact] = []
        for preview in payload.get("previews") or []:
            audio_b64 = preview.get("audio_base_64") or preview.get("audio_base64")
            if not audio_b64:
                continue
            import base64

            generated_id = preview.get("generated_voice_id")
            artifacts.append(
                MediaArtifact(
                    kind=MediaKind.AUDIO,
                    data=base64.b64decode(audio_b64),
                    mime_type=preview.get("media_type") or "audio/mpeg",
                    duration_seconds=preview.get("duration_secs"),
                    provenance=MediaProvenance(
                        generator=model_id,
                        provider_declared=True,
                        # A designed voice imitates no one, so consent is not
                        # open: state the category rather than leaving None,
                        # which would read as "unknown".
                        consent=VoiceConsent(
                            voice_id=generated_id,
                            category="generated",
                            requires_verification=False,
                            is_owner=True,
                            provider_declared=True,
                        ),
                    ),
                    provider_metadata={
                        "generated_voice_id": generated_id,
                        "description": description,
                    },
                )
            )

        return MediaResult(
            capability=MediaCapability.VOICE_DESIGN,
            provider=self.get_name(),
            model=model_id,
            artifacts=artifacts,
            usage=MediaUsage(
                provider=self.get_name(), model=model_id, basis="per_request"
            ),
            raw=payload,
        )

    async def _design_voice(self, body: dict[str, Any]) -> dict[str, Any]:
        """Call the voice-design endpoint."""
        if self._backend == "sdk":
            result = await self._sdk.text_to_voice.design(
                voice_description=body["voice_description"],
                model_id=body.get("model_id"),
                text=body.get("text"),
                auto_generate_text=body.get("auto_generate_text"),
            )
            return self._as_dict(result)

        resp = await self._get_http().post("/v1/text-to-voice/design", json=body)
        if resp.status_code >= 400:
            self._raise_status(resp.status_code, resp.text, "text-to-voice/design")
        return dict(resp.json())

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Release the HTTP client and any SDK resources."""
        if self._http is not None:
            try:
                await self._http.aclose()
            finally:
                self._http = None
        self._sdk = None
