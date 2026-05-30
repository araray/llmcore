# src/llmcore/providers/deepinfra_provider.py
"""
DeepInfra provider implementation for the LLMCore library.

DeepInfra (https://deepinfra.com) is a serverless inference platform that hosts
100+ open-weight models behind an **OpenAI-compatible** API.  Because its chat,
embeddings, and image-generation surfaces are drop-in OpenAI-compatible, this
provider subclasses :class:`~llmcore.providers.openai_provider.OpenAIProvider`
and overrides only the parts where DeepInfra diverges from OpenAI.

Surfaces implemented
--------------------
* **Chat completions** — inherited from OpenAIProvider (text, streaming, tool
  calling, vision via ``Message.metadata["inline_images"]``, structured output
  via ``response_format``).  DeepInfra-specific request parameters
  (``reasoning_effort``, ``reasoning``, ``service_tier``, ``prompt_cache_key``,
  ``top_k``, ``min_p``, ``repetition_penalty``) are routed through the OpenAI
  SDK ``extra_body`` mechanism so the implementation is independent of the
  installed openai SDK version.
* **Model discovery** — overridden to read DeepInfra's rich ``GET /v1/models``
  endpoint, which (unlike vanilla OpenAI) returns ``context_length``,
  ``max_tokens``, ``pricing`` and ``tags`` per model.
* **Text-to-Speech (TTS)** — overridden, ``POST /v1/audio/speech`` (httpx).
* **Speech-to-Text (STT)** — overridden, ``POST /v1/audio/transcriptions`` (httpx).
* **Audio translation** — :meth:`translate_audio`, ``POST /v1/audio/translations``.
* **Image generation** — thin wrapper over the inherited OpenAI implementation
  (``POST /v1/openai/images/generations``) with DeepInfra defaults
  (FLUX Schnell, ``b64_json``).
* **Embeddings** — :meth:`create_embeddings` via ``POST /v1/openai/embeddings``.

Endpoint notes
--------------
The official "drop-in" base URL is ``https://api.deepinfra.com/v1/openai`` and
the OpenAI Python SDK appends ``/chat/completions``, ``/embeddings`` and
``/images/generations`` to it (verified against the DeepInfra docs).  The audio
endpoints, however, are documented under the canonical
``https://api.deepinfra.com/v1/audio/*`` paths (NOT under ``/v1/openai/audio``),
so the TTS/STT/translation methods issue raw httpx requests against the
canonical paths derived by stripping the trailing ``/openai`` from ``base_url``.

References
----------
* Quickstart / Chat: https://docs.deepinfra.com/chat/overview
* Models list:        https://docs.deepinfra.com/api-reference/models/openai-models
* Reasoning:          https://docs.deepinfra.com/chat/reasoning
* Prompt caching:     https://docs.deepinfra.com/chat/prompt-caching
* Vision:             https://docs.deepinfra.com/chat/vision
* Embeddings:         https://docs.deepinfra.com/apis/embeddings
* TTS:                https://docs.deepinfra.com/api-reference/audio/openai-audio-speech
* STT:                https://docs.deepinfra.com/api-reference/audio/openai-audio-transcriptions
* Image generation:   https://docs.deepinfra.com/apis/image-generation
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import pathlib
from collections.abc import AsyncGenerator
from typing import Any

try:
    import httpx

    httpx_available = True
except ImportError:  # pragma: no cover - exercised only when httpx is absent
    httpx_available = False
    httpx = None  # type: ignore[assignment]

from ..exceptions import ProviderError
from ..model_cards.registry import get_model_card_registry
from ..models import ModelDetails, Tool
from .base import ContextPayload
from .openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (verified against the DeepInfra documentation, 2026-05-29)
# ---------------------------------------------------------------------------

#: OpenAI-compatible base URL (chat / embeddings / images).
DEFAULT_DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"

#: Canonical native API root (audio / models discovery).  Derived at runtime
#: from ``base_url`` by stripping a trailing ``/openai`` so custom proxies keep
#: working; this constant is the documented default.
DEFAULT_DEEPINFRA_NATIVE_BASE = "https://api.deepinfra.com/v1"

#: Default chat model.  Matches every example in the DeepInfra docs.  Override
#: via ``[providers.deepinfra].default_model`` or per-request ``model=``.
DEFAULT_DEEPINFRA_MODEL = "deepseek-ai/DeepSeek-V3"

#: Default text-to-speech model (DeepInfra TTS docs use Kokoro).
DEFAULT_DEEPINFRA_TTS_MODEL = "hexgrad/Kokoro-82M"

#: Default speech-to-text model (DeepInfra speech-recognition docs name this as
#: the "best accuracy" Whisper variant).
DEFAULT_DEEPINFRA_STT_MODEL = "openai/whisper-large"

#: Default image-generation model.  ASSUMPTION (labelled): the image-generation
#: doc states the default is "FLUX Schnell" without an explicit ID; this is the
#: canonical Black Forest Labs repo ID DeepInfra exposes.  Override per-request.
DEFAULT_DEEPINFRA_IMAGE_MODEL = "black-forest-labs/FLUX-1-schnell"

#: Default embedding model (DeepInfra embeddings doc).
DEFAULT_DEEPINFRA_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"

#: Conservative fallback context window when neither a model card nor the live
#: discovery cache knows the model.  Override via ``default_context_length``.
DEFAULT_DEEPINFRA_CONTEXT_LENGTH = 32_768

#: Environment variables searched (in order) for the API token.
_DEEPINFRA_API_KEY_ENV_VARS = ("DEEPINFRA_TOKEN", "DEEPINFRA_API_KEY")

#: Request parameters that are NOT part of the typed OpenAI SDK
#: ``chat.completions.create`` signature and must therefore be delivered via
#: ``extra_body``.  Routing them this way keeps the implementation independent
#: of the installed openai SDK version and matches the DeepInfra docs, which use
#: ``extra_body`` for exactly these fields.
_DEEPINFRA_EXTRA_BODY_KEYS = frozenset(
    {
        "service_tier",
        "reasoning_effort",
        "reasoning",
        "prompt_cache_key",
        "top_k",
        "min_p",
        "repetition_penalty",
    }
)

#: Substrings in a model's ``tags``/``id`` that signal vision/multimodal input.
_VISION_HINTS = ("vision", "multimodal", "-vl", "vl-", "image-text-to-text")

#: Tag -> model_type mapping used by :meth:`get_models_details`.
_TAG_TYPE_MAP: dict[str, str] = {
    "embeddings": "embedding",
    "embedding": "embedding",
    "reranker": "rerank",
    "rerank": "rerank",
    "automatic-speech-recognition": "stt",
    "speech-recognition": "stt",
    "text-to-speech": "tts",
    "text-to-image": "image-generation",
    "text-to-video": "text-to-video",
    "ocr": "ocr",
}


class DeepInfraProvider(OpenAIProvider):
    """LLMCore provider for the DeepInfra serverless inference API.

    DeepInfra exposes an OpenAI-compatible surface, so the bulk of the
    behaviour is inherited from :class:`OpenAIProvider`.  This subclass:

    * resolves the ``DEEPINFRA_TOKEN`` / ``DEEPINFRA_API_KEY`` credentials and
      the DeepInfra base URL,
    * advertises DeepInfra-specific request parameters and routes the
      non-OpenAI ones through ``extra_body``,
    * reads the richer ``/v1/models`` discovery endpoint,
    * resolves context windows from the model-card registry / live discovery,
    * implements TTS, STT, audio translation and provider-level embeddings, and
    * applies DeepInfra-appropriate defaults to image generation.

    Multimodal *input* (vision) is unchanged from the parent and is supplied via
    ``Message.metadata["inline_images"]`` (image URLs or ``data:`` URIs) or
    ``Message.metadata["content_parts"]``.
    """

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False):
        """Initialise the DeepInfra provider.

        Args:
            config: Provider configuration.  Recognised keys include
                ``api_key``, ``api_key_env_var``, ``base_url``,
                ``default_model``, ``timeout``, ``max_retries`` and
                ``default_context_length``.
            log_raw_payloads: Whether to log raw request/response payloads.

        Raises:
            ConfigError: If no API key can be resolved.
        """
        # Work on a shallow copy so we never mutate the caller's dict in a way
        # that leaks DeepInfra defaults back into shared config structures.
        cfg = dict(config)

        # 1) Base URL: default to the DeepInfra OpenAI-compatible endpoint.
        cfg.setdefault("base_url", DEFAULT_DEEPINFRA_BASE_URL)

        # 2) Default model.
        cfg.setdefault("default_model", DEFAULT_DEEPINFRA_MODEL)

        # 3) Credential resolution.  The manager only auto-loads env vars for
        #    OpenAI-compatible *shim* sections, so a dedicated-class section
        #    such as ``[providers.deepinfra]`` must resolve its own token here.
        if not cfg.get("api_key"):
            env_var = cfg.get("api_key_env_var")
            resolved: str | None = None
            if env_var:
                resolved = os.environ.get(env_var)
            if not resolved:
                for candidate in _DEEPINFRA_API_KEY_ENV_VARS:
                    resolved = os.environ.get(candidate)
                    if resolved:
                        break
            if resolved:
                cfg["api_key"] = resolved

        # Parent validates the key and builds the AsyncOpenAI client.
        super().__init__(cfg, log_raw_payloads=log_raw_payloads)

        # Native API root for audio + discovery (strip a trailing ``/openai``).
        self._native_base = self._derive_native_base(self.base_url)

        # Configurable fallback context window for unknown models.
        try:
            self._default_context_length = int(
                config.get("default_context_length", DEFAULT_DEEPINFRA_CONTEXT_LENGTH)
            )
        except (TypeError, ValueError):
            self._default_context_length = DEFAULT_DEEPINFRA_CONTEXT_LENGTH

        # Populated lazily by get_models_details(); used by get_max_context_length.
        self._discovery_context_cache: dict[str, int] = {}

        logger.debug(
            "DeepInfraProvider initialised: base_url=%s native_base=%s default_model=%s",
            self.base_url,
            self._native_base,
            self.default_model,
        )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        """Return the provider instance name (``"deepinfra"`` by default)."""
        return self._provider_instance_name or "deepinfra"

    @staticmethod
    def _derive_native_base(base_url: str | None) -> str:
        """Derive the canonical native API root from the OpenAI base URL.

        ``https://api.deepinfra.com/v1/openai`` -> ``https://api.deepinfra.com/v1``.
        Any other base URL has a trailing ``/openai`` stripped if present,
        otherwise it is returned unchanged.

        Args:
            base_url: The configured OpenAI-compatible base URL.

        Returns:
            The native API base URL (no trailing slash).
        """
        if not base_url:
            return DEFAULT_DEEPINFRA_NATIVE_BASE
        trimmed = base_url.rstrip("/")
        if trimmed.endswith("/openai"):
            trimmed = trimmed[: -len("/openai")]
        return trimmed

    def _auth_headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        """Build Bearer auth headers for raw httpx calls.

        Args:
            extra: Additional headers to merge in.

        Returns:
            A header dict including ``Authorization``.
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if extra:
            headers.update(extra)
        return headers

    @staticmethod
    def _require_httpx() -> None:
        """Raise a clear error if httpx is unavailable.

        Raises:
            ImportError: If the ``httpx`` package is not installed.
        """
        if not httpx_available:
            raise ImportError(
                "httpx is required for DeepInfra audio/discovery endpoints. "
                "Install with: pip install httpx"
            )

    # ------------------------------------------------------------------
    # Supported parameters
    # ------------------------------------------------------------------

    def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
        """Return the supported request-parameter schema for *model*.

        Extends the OpenAI parameter set with DeepInfra-specific sampling and
        control parameters and the ``extra_body`` passthrough used internally
        to deliver them.  ``max_tokens`` is always supported because DeepInfra
        reasoning models (e.g. ``deepseek-ai/DeepSeek-R1``) do not use the
        OpenAI o-series ``max_completion_tokens`` convention.

        Args:
            model: Model name; ``None`` resolves to the default model.

        Returns:
            A JSON-schema-like dict of supported parameters.
        """
        params = super().get_supported_parameters(model)

        # DeepInfra reasoning models accept the standard ``max_tokens``; ensure
        # it is present even if the parent stripped it for an o-series prefix.
        params.setdefault("max_tokens", {"type": "integer", "minimum": 1})

        # DeepInfra-specific sampling / control parameters.
        params["top_k"] = {"type": "integer", "minimum": 0}
        params["min_p"] = {"type": "number", "minimum": 0.0, "maximum": 1.0}
        params["repetition_penalty"] = {"type": "number", "minimum": 0.0}
        params["reasoning"] = {"type": "object"}
        # DeepInfra adds "none" to disable chain-of-thought entirely.
        params["reasoning_effort"] = {
            "type": "string",
            "enum": ["none", "low", "medium", "high"],
        }
        params["service_tier"] = {"type": "string", "enum": ["default", "priority"]}
        # Internal passthrough container (allowed so pre-flight validation in
        # the parent's chat_completion does not reject it).
        params["extra_body"] = {"type": "object"}
        return params

    # ------------------------------------------------------------------
    # Chat completion (route DeepInfra-specific params via extra_body)
    # ------------------------------------------------------------------

    async def chat_completion(
        self,
        context: ContextPayload,
        model: str | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Perform a chat completion, routing DeepInfra extras via ``extra_body``.

        Any parameter listed in :data:`_DEEPINFRA_EXTRA_BODY_KEYS` is moved out
        of the top-level kwargs and merged into ``extra_body`` so it is sent in
        the JSON request body regardless of the installed openai SDK version.
        All other behaviour (message building, vision, tool calling, streaming,
        error mapping) is inherited unchanged from :class:`OpenAIProvider`.

        Args:
            context: List of :class:`~llmcore.models.Message` objects.
            model: Model identifier (or ``MODEL:VERSION`` / ``deploy_id:ID``).
            stream: Stream the response as an async generator of chunks.
            tools: Optional tool definitions for function calling.
            tool_choice: Optional tool-choice strategy.
            **kwargs: Additional request parameters (validated against
                :meth:`get_supported_parameters`).

        Returns:
            A response dict, or an async generator of chunk dicts when
            ``stream=True``.
        """
        extra_body: dict[str, Any] = dict(kwargs.pop("extra_body", None) or {})
        for key in list(kwargs.keys()):
            if key in _DEEPINFRA_EXTRA_BODY_KEYS:
                extra_body[key] = kwargs.pop(key)
        if extra_body:
            kwargs["extra_body"] = extra_body

        return await super().chat_completion(
            context,
            model=model,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Model discovery (rich /v1/models endpoint)
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_model(model_id: str, tags: list[str]) -> tuple[str, bool, bool]:
        """Classify a discovered model from its ID and tags.

        Args:
            model_id: The model identifier (e.g. ``Qwen/Qwen2.5-VL-7B-Instruct``).
            tags: The ``metadata.tags`` list from the discovery response.

        Returns:
            A ``(model_type, supports_vision, supports_tools)`` tuple.
        """
        lowered_tags = [str(t).lower() for t in (tags or [])]
        lowered_id = model_id.lower()

        # Model type from tags (first match wins).
        model_type = "chat"
        for tag in lowered_tags:
            if tag in _TAG_TYPE_MAP:
                model_type = _TAG_TYPE_MAP[tag]
                break

        # Vision support from tags or ID hints.
        supports_vision = any(hint in lowered_id for hint in _VISION_HINTS) or any(
            hint in tag for tag in lowered_tags for hint in _VISION_HINTS
        )

        # Tool calling: broadly supported for text-generation / chat LLMs on
        # DeepInfra (function calling is a first-class platform feature).  Not
        # applicable to embedding/tts/stt/image/rerank/video model types.
        supports_tools = model_type == "chat"

        return model_type, supports_vision, supports_tools

    async def get_models_details(self) -> list[ModelDetails]:
        """Discover available models via DeepInfra's rich ``GET /v1/models``.

        Unlike vanilla OpenAI, DeepInfra's ``/v1/models`` returns per-model
        metadata (``context_length``, ``max_tokens``, ``pricing``, ``tags``,
        ``description``).  This method maps that to :class:`ModelDetails`,
        deriving capability flags from tags/IDs, and caches discovered context
        lengths for :meth:`get_max_context_length`.

        Falls back to the inherited OpenAI-SDK discovery (``models.list()``) if
        httpx is unavailable or the request fails.

        Returns:
            A list of :class:`ModelDetails`.
        """
        if not httpx_available:
            logger.debug("httpx unavailable; falling back to OpenAI SDK model discovery.")
            return await super().get_models_details()

        url = f"{self._native_base}/models"
        provider_name = self.get_name()
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                resp = await client.get(url, headers=self._auth_headers())
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:  # degrade gracefully to SDK discovery
            logger.warning(
                "DeepInfra /v1/models discovery failed (%s); falling back to SDK list.", e
            )
            try:
                return await super().get_models_details()
            except Exception as e2:
                raise ProviderError(provider_name, f"Failed to fetch models: {e2}") from e2

        try:
            registry = get_model_card_registry()
        except Exception:
            registry = None

        result: list[ModelDetails] = []
        for m in data.get("data", []):
            model_id = m.get("id")
            if not model_id:
                continue
            meta = m.get("metadata") or {}
            tags = meta.get("tags") or []
            ctx = meta.get("context_length")
            max_out = meta.get("max_tokens")

            model_type, supports_vision, supports_tools = self._classify_model(model_id, tags)

            # Prefer an authoritative model card if present.
            if registry is not None:
                card = registry.get(provider_name, model_id)
                if card and card.capabilities:
                    supports_tools = (
                        card.capabilities.tool_use or card.capabilities.function_calling
                    )
                    supports_vision = card.capabilities.vision
                if card and not ctx:
                    ctx = card.get_context_length()

            if not ctx:
                ctx = self._default_context_length
            self._discovery_context_cache[model_id] = int(ctx)

            result.append(
                ModelDetails(
                    id=model_id,
                    provider_name=provider_name,
                    context_length=int(ctx),
                    max_output_tokens=int(max_out) if max_out else None,
                    model_type=model_type,
                    supports_streaming=True,
                    supports_tools=supports_tools,
                    supports_vision=supports_vision,
                    metadata={
                        "owned_by": m.get("owned_by"),
                        "created": m.get("created"),
                        "tags": tags,
                        "description": meta.get("description"),
                        "pricing": meta.get("pricing"),
                    },
                )
            )

        logger.info("Discovered %d DeepInfra models.", len(result))
        return result

    # ------------------------------------------------------------------
    # Context length resolution
    # ------------------------------------------------------------------

    def get_max_context_length(self, model: str | None = None) -> int:
        """Resolve the maximum context window for *model*.

        Resolution order:

        1. Model-card registry (authoritative).
        2. Live-discovery cache (populated by :meth:`get_models_details`).
        3. Configured fallback (``default_context_length``).

        The OpenAI token tables / prefix heuristics in the parent are *not*
        consulted because DeepInfra model IDs never match them.

        Args:
            model: Model name; ``None`` resolves to the default model.

        Returns:
            The maximum input context length in tokens.
        """
        model_name = model or self.default_model
        provider_name = self.get_name()

        try:
            registry = get_model_card_registry()
            card = registry.get(provider_name, model_name)
            if card is not None:
                return card.get_context_length()
        except Exception as e:
            logger.debug("Model card lookup failed for '%s': %s", model_name, e)

        cached = self._discovery_context_cache.get(model_name)
        if cached:
            return cached

        return self._default_context_length

    # ------------------------------------------------------------------
    # Text-to-Speech (TTS) — POST /v1/audio/speech
    # ------------------------------------------------------------------

    async def generate_speech(
        self,
        text: str,
        *,
        voice: str | None = None,
        model: str | None = None,
        response_format: str = "mp3",
        speed: float = 1.0,
        instructions: str | None = None,
        service_tier: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate speech audio from text (DeepInfra OpenAI-compatible TTS).

        Issues a raw httpx ``POST {native_base}/audio/speech`` request because
        the OpenAI-compatible audio endpoints are documented under
        ``/v1/audio/*`` rather than ``/v1/openai/audio/*``.

        Args:
            text: Text to synthesise (sent as the ``input`` field).
            voice: Optional preset voice (model-specific; omitted if ``None``).
            model: TTS model; defaults to :data:`DEFAULT_DEEPINFRA_TTS_MODEL`.
            response_format: One of ``mp3``, ``opus``, ``flac``, ``wav``, ``pcm``.
            speed: Playback speed multiplier (0.25-4.0).
            instructions: Ignored by most DeepInfra TTS models; forwarded via
                ``extra_body`` for forward compatibility.
            service_tier: ``"default"`` or ``"priority"``.
            **kwargs: Extra model-specific parameters (forwarded in ``extra_body``).

        Returns:
            A :class:`~llmcore.models_multimodal.SpeechResult` with raw audio.

        Raises:
            ProviderError: On API or transport errors.
            ImportError: If httpx is not installed.
        """
        from ..models_multimodal import SpeechResult

        self._require_httpx()
        tts_model = model or DEFAULT_DEEPINFRA_TTS_MODEL
        url = f"{self._native_base}/audio/speech"

        payload: dict[str, Any] = {
            "model": tts_model,
            "input": text,
            "response_format": response_format,
            "speed": speed,
        }
        if voice is not None:
            payload["voice"] = voice
        if service_tier is not None:
            payload["service_tier"] = service_tier

        extra_body = dict(kwargs.pop("extra_body", None) or {})
        if instructions is not None:
            extra_body["instructions"] = instructions
        extra_body.update(kwargs)
        if extra_body:
            payload["extra_body"] = extra_body

        logger.debug(
            "DeepInfra TTS: model=%s voice=%s format=%s len=%d",
            tts_model,
            voice,
            response_format,
            len(text),
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                resp = await client.post(
                    url,
                    headers=self._auth_headers({"Content-Type": "application/json"}),
                    json=payload,
                )
                if resp.status_code >= 400:
                    raise ProviderError(
                        self.get_name(),
                        f"TTS Error ({resp.status_code}): {resp.text[:500]}",
                    )
                audio_bytes = resp.content
        except ProviderError:
            raise
        except Exception as e:
            logger.error("DeepInfra TTS request failed: %s", e, exc_info=True)
            raise ProviderError(self.get_name(), f"TTS error: {e}") from e

        return SpeechResult(
            audio_data=audio_bytes,
            format=response_format,
            model=tts_model,
            voice=voice or "",
            metadata={},
        )

    # ------------------------------------------------------------------
    # Speech-to-Text (STT) — POST /v1/audio/transcriptions
    # ------------------------------------------------------------------

    async def transcribe_audio(
        self,
        audio_data: bytes | str,
        *,
        model: str | None = None,
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "json",
        temperature: float | None = None,
        timestamp_granularities: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Transcribe audio to text (DeepInfra OpenAI-compatible Whisper STT).

        Args:
            audio_data: Raw audio bytes or a path to an audio file (mp3/wav).
            model: STT model; defaults to :data:`DEFAULT_DEEPINFRA_STT_MODEL`.
            language: Input language hint (ISO-639-1, e.g. ``"en"``).
            prompt: Optional text to bias the transcription style.
            response_format: ``json``, ``verbose_json``, ``text``, ``srt`` or ``vtt``.
            temperature: Sampling temperature (default 0 server-side).
            timestamp_granularities: ``["segment"]`` and/or ``["word"]`` —
                forces ``verbose_json`` when set with the default ``json`` format.
            **kwargs: Extra form fields forwarded to the endpoint.

        Returns:
            A :class:`~llmcore.models_multimodal.TranscriptionResult`.

        Raises:
            ProviderError: On API or transport errors.
            ImportError: If httpx is not installed.
        """
        return await self._transcribe_or_translate(
            "transcriptions",
            audio_data,
            model=model,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities,
            **kwargs,
        )

    async def translate_audio(
        self,
        audio_data: bytes | str,
        *,
        model: str | None = None,
        prompt: str | None = None,
        response_format: str = "json",
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Translate audio into English text (``POST /v1/audio/translations``).

        Args:
            audio_data: Raw audio bytes or a path to an audio file.
            model: STT model; defaults to :data:`DEFAULT_DEEPINFRA_STT_MODEL`.
            prompt: Optional text to bias the translation style.
            response_format: ``json``, ``verbose_json``, ``text``, ``srt`` or ``vtt``.
            temperature: Sampling temperature.
            **kwargs: Extra form fields forwarded to the endpoint.

        Returns:
            A :class:`~llmcore.models_multimodal.TranscriptionResult` (English).

        Raises:
            ProviderError: On API or transport errors.
            ImportError: If httpx is not installed.
        """
        return await self._transcribe_or_translate(
            "translations",
            audio_data,
            model=model,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            **kwargs,
        )

    async def _transcribe_or_translate(
        self,
        endpoint: str,
        audio_data: bytes | str,
        *,
        model: str | None = None,
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "json",
        temperature: float | None = None,
        timestamp_granularities: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Shared multipart implementation for transcription and translation.

        Args:
            endpoint: ``"transcriptions"`` or ``"translations"``.
            audio_data: Raw bytes or a file path.
            model: STT model (defaults to the configured STT default).
            language: Language hint (transcription only).
            prompt: Optional style prompt.
            response_format: Output format.
            temperature: Sampling temperature.
            timestamp_granularities: Timing granularities (transcription only).
            **kwargs: Extra form fields.

        Returns:
            A :class:`~llmcore.models_multimodal.TranscriptionResult`.
        """
        from ..models_multimodal import TranscriptionResult, TranscriptionSegment

        self._require_httpx()
        stt_model = model or DEFAULT_DEEPINFRA_STT_MODEL
        url = f"{self._native_base}/audio/{endpoint}"

        # Resolve audio bytes + filename. File reads are offloaded to a worker
        # thread so we never perform blocking I/O on the event loop.
        if isinstance(audio_data, str):
            path = pathlib.Path(audio_data)
            content = await asyncio.to_thread(path.read_bytes)
            filename = path.name
        else:
            content = audio_data
            filename = "audio.wav"

        # ``timestamp_granularities`` requires verbose_json.
        if timestamp_granularities and response_format == "json":
            response_format = "verbose_json"

        form: dict[str, Any] = {"model": stt_model, "response_format": response_format}
        if language is not None:
            form["language"] = language
        if prompt is not None:
            form["prompt"] = prompt
        if temperature is not None:
            form["temperature"] = str(temperature)
        if timestamp_granularities:
            # Repeated form field — httpx accepts a list value per key.
            form["timestamp_granularities[]"] = timestamp_granularities
        for k, v in kwargs.items():
            form[k] = v

        files = {"file": (filename, io.BytesIO(content), "application/octet-stream")}

        logger.debug(
            "DeepInfra STT(%s): model=%s language=%s format=%s",
            endpoint,
            stt_model,
            language,
            response_format,
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                resp = await client.post(
                    url,
                    headers=self._auth_headers(),
                    data=form,
                    files=files,
                )
                if resp.status_code >= 400:
                    raise ProviderError(
                        self.get_name(),
                        f"STT Error ({resp.status_code}): {resp.text[:500]}",
                    )
                # Parse depending on the requested format.
                if response_format in ("json", "verbose_json"):
                    body = resp.json()
                else:
                    body = {"text": resp.text}
        except ProviderError:
            raise
        except Exception as e:
            logger.error("DeepInfra STT request failed: %s", e, exc_info=True)
            raise ProviderError(self.get_name(), f"STT error: {e}") from e

        text = body.get("text", "") if isinstance(body, dict) else str(body)
        segments: list[TranscriptionSegment] = []
        for seg in (body.get("segments") or []) if isinstance(body, dict) else []:
            if isinstance(seg, dict):
                segments.append(
                    TranscriptionSegment(
                        text=seg.get("text", ""),
                        start=float(seg.get("start", 0.0) or 0.0),
                        end=float(seg.get("end", 0.0) or 0.0),
                    )
                )

        return TranscriptionResult(
            text=text,
            language=(body.get("language") if isinstance(body, dict) else None) or language,
            duration_seconds=(body.get("duration") if isinstance(body, dict) else None),
            segments=segments,
            model=stt_model,
            metadata={},
        )

    # ------------------------------------------------------------------
    # Image generation — POST /v1/openai/images/generations (inherited)
    # ------------------------------------------------------------------

    async def generate_image(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
        size: str | None = None,
        quality: str | None = None,
        response_format: str = "b64_json",
        style: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate images from a text prompt.

        Thin wrapper over the inherited OpenAI implementation that applies the
        DeepInfra default model (FLUX Schnell) and forces ``b64_json`` (the only
        response format DeepInfra supports for image generation).

        Args:
            prompt: Text description of the desired image(s).
            model: Image model; defaults to :data:`DEFAULT_DEEPINFRA_IMAGE_MODEL`.
            n: Number of images to generate.
            size: Image dimensions (e.g. ``"1024x1024"``).
            quality: Accepted for compatibility (ignored by DeepInfra).
            response_format: Forced to ``b64_json``.
            style: Accepted for compatibility (ignored by DeepInfra).
            **kwargs: Extra params forwarded to the endpoint.

        Returns:
            An :class:`~llmcore.models_multimodal.ImageGenerationResult`.
        """
        img_model = model or DEFAULT_DEEPINFRA_IMAGE_MODEL
        return await super().generate_image(
            prompt,
            model=img_model,
            n=n,
            size=size,
            quality=quality,
            response_format="b64_json",
            style=style,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Embeddings — POST /v1/openai/embeddings
    # ------------------------------------------------------------------

    async def create_embeddings(
        self,
        input_texts: str | list[str],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create text embeddings via DeepInfra's OpenAI-compatible endpoint.

        DeepInfra supports only ``encoding_format="float"`` (no ``dimensions``).

        Args:
            input_texts: A string or list of strings to embed.
            model: Embedding model; defaults to
                :data:`DEFAULT_DEEPINFRA_EMBEDDING_MODEL`.
            **kwargs: Extra parameters (e.g. ``encoding_format``).

        Returns:
            The raw API response as a dict (``data``, ``model``, ``usage``).

        Raises:
            ProviderError: On API errors.
        """
        if not self._client:
            raise ProviderError(self.get_name(), "Client not initialized.")
        emb_model = model or DEFAULT_DEEPINFRA_EMBEDDING_MODEL
        kwargs.setdefault("encoding_format", "float")
        try:
            resp = await self._client.embeddings.create(
                model=emb_model, input=input_texts, **kwargs
            )
            return resp.model_dump(exclude_none=True)
        except Exception as e:
            logger.error("DeepInfra embeddings failed: %s", e, exc_info=True)
            raise ProviderError(self.get_name(), f"Embeddings error: {e}") from e
