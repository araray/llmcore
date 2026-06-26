# src/llmcore/providers/deepgram_provider.py
"""LLMCore provider for Deepgram — speech / audio (STT, TTS, Voice Agent).

Deepgram (https://deepgram.com) is a **real-time voice / audio** platform. It is
fundamentally different from the text-completion LLM providers in llmcore: its
primary surfaces are WebSocket streams for speech-to-text (STT), text-to-speech
(TTS), and a bidirectional **Voice Agent** (STT → LLM → TTS over one socket).
There is **no text chat-completion surface**, so :meth:`chat_completion` raises a
clear, actionable error and callers use the media methods instead.

This provider wraps the official, async-first, WebSocket-native
``deepgram-sdk`` (v7.x) and exposes:

Batch
    * :meth:`transcribe_audio` — pre-recorded STT → :class:`TranscriptionResult`
    * :meth:`generate_speech` — TTS → :class:`SpeechResult`
    * :meth:`analyze_text` — text intelligence → :class:`TextAnalysisResult`

Streaming / real-time
    * :meth:`transcribe_stream` / :meth:`open_transcription_socket` — live STT
    * :meth:`transcribe_stream_flux` / :meth:`open_flux_socket` — Flux (v2) STT
    * :meth:`stream_speech` / :meth:`open_speech_socket` — live TTS
    * :meth:`open_voice_agent` / :meth:`run_voice_agent` — the Voice Agent

Auth / account
    * :meth:`grant_token` — short-lived token (temp key) for browsers/clients
    * :meth:`list_models`, :meth:`get_projects`, :meth:`get_balances`,
      :meth:`get_usage` — a pragmatic management subset
    * :attr:`client` — the raw ``AsyncDeepgramClient`` escape hatch

Design notes
------------
* **Native SDK** (not OpenAI-compatible): follows the *Gemini* provider template
  — lazy SDK import with an availability flag, ``__init__`` client construction,
  ``ImportError`` when the SDK is absent (so :class:`ProviderManager` surfaces a
  friendly *"install llmcore[deepgram]"*), and ``ConfigError`` for bad config.
* **Provider-direct access**: media methods are called on the provider instance
  (``mgr.get_provider("deepgram").transcribe_audio(...)``), consistent with the
  other media providers (the :class:`LLMCore` facade has no media methods today).
* **Token semantics**: Deepgram bills per audio-minute (STT) / per-character
  (TTS), so :meth:`count_tokens` returns a documented *character-count* heuristic
  and :meth:`get_max_context_length` returns a configurable nominal value (the
  REST-TTS input character cap by default). These are **not** billing units.
* **Configuration**: every capability is wired through ``[providers.deepgram]``
  in ``config/default_config.toml`` with safe defaults and per-call overrides.

Tested against ``deepgram-sdk`` v7.3.1.

References:
    * Deepgram docs: https://developers.deepgram.com/ (append ``.md`` for Markdown)
    * Pre-recorded STT: https://developers.deepgram.com/docs/pre-recorded-audio
    * TTS REST: https://developers.deepgram.com/docs/text-to-speech
    * Streaming STT: https://developers.deepgram.com/docs/live-streaming-audio
    * Streaming TTS: https://developers.deepgram.com/docs/streaming-text-to-speech
    * Flux: https://developers.deepgram.com/docs/flux/quickstart
    * Voice Agent: https://developers.deepgram.com/docs/voice-agent
    * Text Intelligence: https://developers.deepgram.com/docs/text-intelligence
    * Token Auth: https://developers.deepgram.com/docs/token-based-authentication
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import logging
import os
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..exceptions import ConfigError, ProviderError
from ..models import Message, ModelDetails
from ..models_multimodal import (
    SpeechResult,
    StreamEventType,
    TextAnalysisResult,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionStreamEvent,
    VoiceAgentEvent,
    VoiceAgentEventType,
    VoiceAgentFunctionCall,
)
from .base import BaseProvider, ContextPayload

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator

    from ..models import Tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy SDK import (mirrors the gemini/native-provider pattern)
# ---------------------------------------------------------------------------

deepgram_available = False
try:
    import deepgram as _deepgram
    from deepgram import AsyncDeepgramClient
    from deepgram.agent.v1.types.agent_v1inject_agent_message import (
        AgentV1InjectAgentMessage,
    )
    from deepgram.agent.v1.types.agent_v1inject_user_message import (
        AgentV1InjectUserMessage,
    )
    from deepgram.agent.v1.types.agent_v1send_function_call_response import (
        AgentV1SendFunctionCallResponse,
    )
    from deepgram.agent.v1.types.agent_v1settings import AgentV1Settings
    from deepgram.agent.v1.types.agent_v1update_prompt import AgentV1UpdatePrompt
    from deepgram.agent.v1.types.agent_v1update_speak import AgentV1UpdateSpeak
    from deepgram.agent.v1.types.agent_v1update_think import AgentV1UpdateThink
    from deepgram.core.api_error import ApiError as _DeepgramApiError
    from deepgram.environment import DeepgramClientEnvironment
    from deepgram.speak.v1.types.speak_v1text import SpeakV1Text

    deepgram_available = True
except ImportError:  # pragma: no cover - exercised only when SDK is absent
    _deepgram = None  # type: ignore[assignment]
    AsyncDeepgramClient = None  # type: ignore[assignment,misc]
    DeepgramClientEnvironment = None  # type: ignore[assignment,misc]
    SpeakV1Text = None  # type: ignore[assignment,misc]
    AgentV1Settings = None  # type: ignore[assignment,misc]
    AgentV1UpdatePrompt = None  # type: ignore[assignment,misc]
    AgentV1UpdateThink = None  # type: ignore[assignment,misc]
    AgentV1UpdateSpeak = None  # type: ignore[assignment,misc]
    AgentV1InjectUserMessage = None  # type: ignore[assignment,misc]
    AgentV1InjectAgentMessage = None  # type: ignore[assignment,misc]
    AgentV1SendFunctionCallResponse = None  # type: ignore[assignment,misc]

    class _DeepgramApiError(Exception):  # type: ignore[no-redef]
        """Fallback so ``except _DeepgramApiError`` is valid without the SDK."""

        status_code: int | None = None
        headers: Any | None = None
        body: Any | None = None


# ---------------------------------------------------------------------------
# Constants (verified against deepgram-sdk v7.3.1)
# ---------------------------------------------------------------------------

#: Environment variables searched (in order) for the API key.
_DEEPGRAM_API_KEY_ENV_VARS = ("DEEPGRAM_API_KEY",)

#: Nominal default models per capability. Overridable via config / per call.
_DEFAULT_STT_MODEL = "nova-3"
_DEFAULT_FLUX_MODEL = "flux-general-en"
_DEFAULT_TTS_MODEL = "aura-2-thalia-en"

#: Documented REST text-to-speech input character cap. Used as the nominal
#: "context length" for a provider that has no token context window.
_DEFAULT_FALLBACK_CONTEXT_LENGTH = 2000

#: Accepted keyword params for batch STT (``listen.v1.media.transcribe_file`` /
#: ``transcribe_url``). Used to filter merged config + call kwargs so unknown
#: keys never reach the SDK. ``request``/``url`` are handled separately.
_STT_BATCH_PARAMS = frozenset(
    {
        "callback",
        "callback_method",
        "extra",
        "sentiment",
        "summarize",
        "tag",
        "topics",
        "custom_topic",
        "custom_topic_mode",
        "intents",
        "custom_intent",
        "custom_intent_mode",
        "detect_entities",
        "detect_language",
        "diarize",
        "diarize_model",
        "dictation",
        "encoding",
        "filler_words",
        "keyterm",
        "keywords",
        "language",
        "measurements",
        "model",
        "multichannel",
        "numerals",
        "paragraphs",
        "profanity_filter",
        "punctuate",
        "redact",
        "replace",
        "search",
        "smart_format",
        "utterances",
        "utt_split",
        "version",
        "mip_opt_out",
        "channels",
        "sample_rate",
    }
)

#: Accepted keyword params for batch TTS (``speak.v1.audio.generate``).
_TTS_BATCH_PARAMS = frozenset(
    {
        "callback",
        "callback_method",
        "mip_opt_out",
        "tag",
        "bit_rate",
        "container",
        "encoding",
        "model",
        "sample_rate",
        "speed",
    }
)

#: Accepted keyword params for streaming STT (``listen.v1.connect``). Superset
#: of the batch params plus live-only knobs. ``model`` is REQUIRED by the SDK.
_STT_STREAM_PARAMS = frozenset(
    {
        "callback",
        "callback_method",
        "channels",
        "detect_entities",
        "diarize",
        "dictation",
        "encoding",
        "endpointing",
        "extra",
        "interim_results",
        "keyterm",
        "keywords",
        "language",
        "mip_opt_out",
        "model",
        "multichannel",
        "numerals",
        "profanity_filter",
        "punctuate",
        "redact",
        "replace",
        "sample_rate",
        "search",
        "smart_format",
        "tag",
        "utterance_end_ms",
        "vad_events",
        "version",
    }
)

#: Accepted keyword params for streaming TTS (``speak.v1.connect``).
_TTS_STREAM_PARAMS = frozenset(
    {"encoding", "mip_opt_out", "model", "sample_rate", "speed"}
)

#: Accepted keyword params for Flux / v2 streaming STT (``listen.v2.connect``).
#: ``model`` is REQUIRED by the SDK.
_FLUX_PARAMS = frozenset(
    {
        "model",
        "encoding",
        "sample_rate",
        "eager_eot_threshold",
        "eot_threshold",
        "eot_timeout_ms",
        "keyterm",
        "language_hint",
        "mip_opt_out",
        "tag",
    }
)

#: Params that the Deepgram API expects as a single comma-joined string even
#: though config / callers may naturally supply a list.
_COMMA_JOIN_PARAMS = frozenset({"redact"})

#: Keys consumed by the provider itself (never forwarded to the SDK as request
#: params) when present at the top level of ``[providers.deepgram]``.
_RESERVED_CONFIG_KEYS = frozenset(
    {
        "type",
        "_instance_name",
        "api_key",
        "api_key_env_var",
        "access_token",
        "session_id",
        "timeout",
        "max_retries",
        "mip_opt_out",
        "fallback_context_length",
        "base_url",
        "ws_url",
        "agent_ws_url",
        "agent_rest_url",
        "default_stt_model",
        "default_flux_model",
        "default_tts_model",
        "stt",
        "flux",
        "tts",
        "agent",
    }
)


class DeepgramProvider(BaseProvider):
    """Deepgram speech/audio provider.

    Wraps :class:`deepgram.AsyncDeepgramClient` and implements the llmcore
    :class:`~llmcore.providers.base.BaseProvider` contract for a media provider.

    The text-centric abstract methods are satisfied honestly:
    :meth:`chat_completion` raises (Deepgram has no text completion);
    token/context methods return documented heuristics. The media methods
    (:meth:`transcribe_audio`, :meth:`generate_speech`, and the streaming/agent
    methods) are the real surface.
    """

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False):
        """Initialize the Deepgram provider.

        Args:
            config: The ``[providers.deepgram]`` configuration dictionary. Keys
                of interest:

                * ``api_key`` / ``api_key_env_var`` — API key or the env var
                  holding it (falls back to ``DEEPGRAM_API_KEY``).
                * ``access_token`` — Bearer token; takes precedence over
                  ``api_key`` when set.
                * ``session_id`` — value for the ``x-deepgram-session-id``
                  header (auto-UUID if omitted).
                * ``timeout`` / ``max_retries`` — transport tuning.
                * ``mip_opt_out`` — global Model Improvement Program opt-out.
                * ``fallback_context_length`` — nominal context length reported
                  by :meth:`get_max_context_length`.
                * ``base_url`` / ``ws_url`` / ``agent_ws_url`` /
                  ``agent_rest_url`` — self-hosted endpoint overrides.
                * ``default_stt_model`` / ``default_flux_model`` /
                  ``default_tts_model`` — per-capability default models.
                * ``stt`` / ``flux`` / ``tts`` / ``agent`` — nested capability
                  default dicts (see ``default_config.toml``).
            log_raw_payloads: Whether to log raw request/response payloads.

        Raises:
            ImportError: If the ``deepgram-sdk`` package is not installed.
            ConfigError: If no credentials are available or the client cannot be
                constructed.
        """
        super().__init__(config, log_raw_payloads)
        if not deepgram_available:
            raise ImportError(
                "Deepgram library (`deepgram-sdk`) not installed. "
                "Install with 'pip install llmcore[deepgram]'."
            )

        # --- Credential resolution (access_token > api_key/env) ---
        self._access_token: str | None = config.get("access_token")
        self._api_key_env_var: str | None = config.get("api_key_env_var")
        api_key = config.get("api_key")
        if not api_key and self._api_key_env_var:
            api_key = os.environ.get(self._api_key_env_var)
        if not api_key:
            for env_var in _DEEPGRAM_API_KEY_ENV_VARS:
                api_key = os.environ.get(env_var)
                if api_key:
                    break
        self.api_key: str | None = api_key

        if not self.api_key and not self._access_token:
            raise ConfigError(
                "Deepgram credentials not found. Set the DEEPGRAM_API_KEY "
                "environment variable, configure 'api_key'/'api_key_env_var', "
                "or provide an 'access_token' in [providers.deepgram]."
            )

        # --- Transport / behaviour config ---
        self._session_id: str | None = config.get("session_id")
        self._timeout: float = float(config.get("timeout", 60))
        self._max_retries: int = int(config.get("max_retries", 2))
        self.mip_opt_out: bool = bool(config.get("mip_opt_out", False))
        self.fallback_context_length: int = int(
            config.get("fallback_context_length", _DEFAULT_FALLBACK_CONTEXT_LENGTH)
        )

        # --- Default models per capability ---
        self.default_stt_model: str = config.get("default_stt_model", _DEFAULT_STT_MODEL)
        self.default_flux_model: str = config.get(
            "default_flux_model", _DEFAULT_FLUX_MODEL
        )
        self.default_tts_model: str = config.get("default_tts_model", _DEFAULT_TTS_MODEL)
        # The provider's nominal "default model" (used by generic callers).
        self.default_model: str = self.default_stt_model

        # --- Capability default dicts (nested sub-tables) ---
        self._stt_defaults: dict[str, Any] = dict(config.get("stt", {}) or {})
        self._stt_stream_defaults: dict[str, Any] = dict(
            self._stt_defaults.pop("streaming", {}) or {}
        )
        self._flux_defaults: dict[str, Any] = dict(config.get("flux", {}) or {})
        self._tts_defaults: dict[str, Any] = dict(config.get("tts", {}) or {})
        self._tts_stream_defaults: dict[str, Any] = dict(
            self._tts_defaults.pop("streaming", {}) or {}
        )
        self._agent_defaults: dict[str, Any] = dict(config.get("agent", {}) or {})

        # Honesty flags for generic chat-surface consumers (audio streaming and
        # voice-agent tool calling are documented separately, not via these).
        self.supports_streaming: bool = False
        self.supports_tools: bool = False

        # --- Construct the async client ---
        try:
            client_kwargs: dict[str, Any] = {
                "timeout": self._timeout,
                "max_retries": self._max_retries,
            }
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
            if self._access_token:
                client_kwargs["access_token"] = self._access_token
            if self._session_id:
                client_kwargs["session_id"] = self._session_id
            environment = self._build_environment(config)
            if environment is not None:
                client_kwargs["environment"] = environment

            self._client = AsyncDeepgramClient(**client_kwargs)
            logger.debug(
                "Deepgram client initialised (instance=%s, stt=%s, tts=%s).",
                self._provider_instance_name or "deepgram",
                self.default_stt_model,
                self.default_tts_model,
            )
        except Exception as exc:  # surface as ConfigError
            raise ConfigError(f"Deepgram client configuration failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_environment(config: dict[str, Any]) -> Any | None:
        """Build a custom ``DeepgramClientEnvironment`` for self-hosting.

        Any omitted URL falls back to the corresponding production endpoint, so
        callers can override just one leg (e.g. a self-hosted REST base) without
        respecifying the rest.

        Args:
            config: The provider config dict.

        Returns:
            A ``DeepgramClientEnvironment`` if any URL override is present,
            otherwise ``None`` (meaning "use production defaults").
        """
        base = config.get("base_url")
        production = config.get("ws_url")
        agent = config.get("agent_ws_url")
        agent_rest = config.get("agent_rest_url")
        if not any((base, production, agent, agent_rest)):
            return None
        prod_env = DeepgramClientEnvironment.PRODUCTION
        return DeepgramClientEnvironment(
            base=base or prod_env.base,
            production=production or prod_env.production,
            agent=agent or prod_env.agent,
            agent_rest=agent_rest or prod_env.agent_rest,
        )

    @staticmethod
    def _normalize_params(params: dict[str, Any]) -> dict[str, Any]:
        """Normalize merged params for the SDK.

        * Drops ``None`` values (so SDK defaults apply).
        * Joins list-valued params that the API expects as comma-joined strings
          (e.g. ``redact``).

        Args:
            params: The merged parameter dict.

        Returns:
            A new, normalized dict safe to splat into an SDK call.
        """
        out: dict[str, Any] = {}
        for key, value in params.items():
            if value is None:
                continue
            if key in _COMMA_JOIN_PARAMS and isinstance(value, (list, tuple)):
                out[key] = ",".join(str(v) for v in value)
            else:
                out[key] = value
        return out

    def _merge_params(
        self,
        *layers: dict[str, Any],
        allowed: frozenset[str],
        model: str | None = None,
        default_model: str | None = None,
    ) -> dict[str, Any]:
        """Merge config + call layers, filter to ``allowed``, and normalize.

        Later layers win over earlier ones. ``mip_opt_out`` defaults to the
        provider-level setting unless a layer overrides it. ``model`` (and a
        ``default_model`` fallback) are injected last.

        Args:
            *layers: Parameter dicts in increasing-precedence order.
            allowed: The set of parameter names the target SDK call accepts.
            model: Explicit model override (highest precedence for ``model``).
            default_model: Fallback model if none is otherwise set.

        Returns:
            The merged, filtered, normalized parameter dict.
        """
        merged: dict[str, Any] = {}
        # Seed mip_opt_out from the provider default (overridable by layers).
        if "mip_opt_out" in allowed:
            merged["mip_opt_out"] = self.mip_opt_out
        for layer in layers:
            for key, value in (layer or {}).items():
                # ``None`` means "not specified" — it must not clobber a value
                # supplied by an earlier (lower-precedence) layer. Explicit
                # ``False``/``0`` are real values and are preserved.
                if key in allowed and value is not None:
                    merged[key] = value
        if model is not None:
            merged["model"] = model
        if "model" not in merged and default_model is not None:
            merged["model"] = default_model
        return self._normalize_params({k: v for k, v in merged.items() if k in allowed})

    @staticmethod
    def _map_api_error(
        exc: Exception, *, model: str | None = None
    ) -> ProviderError:
        """Map a Deepgram SDK exception to an llmcore :class:`ProviderError`.

        Args:
            exc: The original exception (typically ``deepgram ... ApiError``).
            model: The model in play, for richer error context.

        Returns:
            A :class:`ProviderError` carrying the status code, headers, and the
            original exception (retryability inferred from the status).
        """
        status = getattr(exc, "status_code", None)
        headers = getattr(exc, "headers", None)
        body = getattr(exc, "body", None)
        message = f"Deepgram API error: {body if body is not None else exc}"
        return ProviderError(
            "deepgram",
            message,
            model_name=model,
            status_code=status,
            headers=headers,
            original_exception=exc,
        )

    # ------------------------------------------------------------------
    # BaseProvider: identity & capabilities
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        """Return the provider instance name (or ``"deepgram"``)."""
        return self._provider_instance_name or "deepgram"

    async def get_models_details(self) -> list[ModelDetails]:
        """Return known Deepgram models as :class:`ModelDetails`.

        Sourced from the model-card registry (provider == ``"deepgram"``), with
        a small static fallback if the registry has no Deepgram cards. Context
        lengths are nominal (see module docstring); ``model_type`` reflects the
        card (``"stt"`` / ``"tts"``).

        Returns:
            A list of :class:`ModelDetails` (possibly empty if neither source
            yields anything).
        """
        details: list[ModelDetails] = []
        try:
            from ..model_cards.registry import get_model_card_registry

            registry = get_model_card_registry()
            registry.load()
            # list_cards returns ModelCardSummary objects (model_id, provider,
            # model_type, context_length, tags, display_name, ...).
            summaries = registry.list_cards(provider="deepgram")
            for summary in summaries:
                tags = list(getattr(summary, "tags", []) or [])
                details.append(
                    ModelDetails(
                        id=summary.model_id,
                        provider_name=self.get_name(),
                        display_name=summary.display_name,
                        context_length=summary.context_length,
                        supports_streaming="streaming" in tags,
                        supports_tools=False,
                        family=summary.model_id.split("-", 1)[0] or None,
                        model_type=summary.model_type,
                        metadata={"tags": tags, "status": getattr(summary, "status", None)},
                    )
                )
        except Exception as exc:  # registry is best-effort
            logger.debug("Deepgram model-card lookup failed: %s", exc)

        if details:
            return details

        # Static fallback: the canonical default models.
        fallback = [
            (self.default_stt_model, "stt"),
            (self.default_flux_model, "stt"),
            (self.default_tts_model, "tts"),
        ]
        return [
            ModelDetails(
                id=model_id,
                provider_name=self.get_name(),
                context_length=self.fallback_context_length,
                supports_streaming=True,
                supports_tools=False,
                model_type=model_type,
            )
            for model_id, model_type in fallback
        ]

    def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
        """Return the supported request parameters, grouped by capability.

        The returned schema lets callers pre-flight which knobs are accepted for
        batch STT, batch TTS, streaming, Flux, and the Voice Agent.

        Args:
            model: Unused (parameters are capability- not model-scoped) — present
                for interface compatibility.

        Returns:
            A dict keyed by capability with sorted parameter-name lists and a
            ``notes`` entry documenting non-token billing.
        """
        return {
            "stt_batch": sorted(_STT_BATCH_PARAMS),
            "tts_batch": sorted(_TTS_BATCH_PARAMS),
            "stt_streaming": sorted(
                _STT_BATCH_PARAMS
                | {
                    "interim_results",
                    "endpointing",
                    "utterance_end_ms",
                    "vad_events",
                    "channels",
                    "no_delay",
                }
            ),
            "flux": [
                "model",
                "encoding",
                "sample_rate",
                "eot_threshold",
                "eager_eot_threshold",
                "eot_timeout_ms",
                "keyterm",
                "language_hint",
                "mip_opt_out",
                "tag",
            ],
            "agent": [
                "audio",
                "agent",
                "greeting",
                "context",
                "flags",
                "tags",
                "experimental",
                "mip_opt_out",
            ],
            "notes": (
                "Deepgram bills per audio-minute (STT) / per-character (TTS); "
                "token counts are not billing units."
            ),
        }

    def get_max_context_length(self, model: str | None = None) -> int:
        """Return a nominal context length for a media provider.

        Deepgram has no token context window. This returns the model card's
        ``max_input_tokens`` when available (TTS cards use the REST input
        character cap), otherwise the configured ``fallback_context_length``.

        Args:
            model: Optional model id to look up a card-specific nominal.

        Returns:
            A nominal context length (int). Capability-dependent; documented.
        """
        if model:
            try:
                from ..model_cards.registry import get_model_card_registry

                registry = get_model_card_registry()
                registry.load()
                card = registry.get("deepgram", model)
                if card is not None:
                    return card.context.max_input_tokens
            except Exception as exc:  # best-effort
                logger.debug("Deepgram context-length lookup failed: %s", exc)
        return self.fallback_context_length

    # ------------------------------------------------------------------
    # BaseProvider: text-completion surface (intentionally unsupported)
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
        """Not supported — Deepgram is a speech/audio provider.

        Raises:
            ProviderError: Always (status 400, non-retryable), directing callers
                to the audio methods.
        """
        raise ProviderError(
            "deepgram",
            "Deepgram is a speech/audio provider and does not implement text "
            "chat_completion. Use transcribe_audio()/generate_speech()/"
            "transcribe_stream()/stream_speech()/open_voice_agent() instead.",
            model_name=model,
            status_code=400,
            retryable=False,
        )

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """Return a character-count heuristic (Deepgram has no token billing).

        Args:
            text: The text to measure.
            model: Unused; present for interface compatibility.

        Returns:
            ``len(text)`` — a documented stand-in (TTS is billed per character).
        """
        return len(text or "")

    async def count_message_tokens(
        self, messages: list[Message], model: str | None = None
    ) -> int:
        """Return a character-count heuristic over messages, with small overhead.

        Args:
            messages: The messages to measure.
            model: Unused; present for interface compatibility.

        Returns:
            Sum of per-message content lengths plus a constant per-message
            overhead — a documented stand-in (not a billing unit).
        """
        total = 0
        for msg in messages or []:
            content = getattr(msg, "content", "") or ""
            total += len(content) + 4  # nominal per-message overhead
        return total

    def extract_response_content(self, response: dict[str, Any]) -> str:
        """Best-effort transcript/text extraction from a response dict.

        Handles both an llmcore :class:`TranscriptionResult`-shaped dict
        (``{"text": ...}``) and a raw Deepgram response
        (``results.channels[0].alternatives[0].transcript``).

        Args:
            response: A response dict.

        Returns:
            The extracted text, or an empty string if none is found.
        """
        if not isinstance(response, dict):
            return ""
        if isinstance(response.get("text"), str):
            return response["text"]
        try:
            results = response.get("results") or {}
            channels = results.get("channels") or []
            if channels:
                alts = channels[0].get("alternatives") or []
                if alts:
                    return alts[0].get("transcript", "") or ""
        except (AttributeError, IndexError, KeyError, TypeError):
            return ""
        return ""

    def extract_delta_content(self, chunk: dict[str, Any]) -> str:
        """Extract incremental transcript text from a stream-event dict.

        Args:
            chunk: A stream event dict (e.g. a serialized
                :class:`TranscriptionStreamEvent`).

        Returns:
            The text delta, or an empty string.
        """
        if not isinstance(chunk, dict):
            return ""
        if isinstance(chunk.get("text"), str):
            return chunk["text"]
        return self.extract_response_content(chunk)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def warm_up(self) -> None:
        """Cheap readiness check (no network call).

        Logs the configured endpoints/models so operators can confirm the
        instance is wired correctly. Network warmth is intentionally avoided to
        keep this side-effect-free.
        """
        logger.debug(
            "Deepgram provider ready (instance=%s, stt=%s, flux=%s, tts=%s).",
            self.get_name(),
            self.default_stt_model,
            self.default_flux_model,
            self.default_tts_model,
        )

    async def close(self) -> None:
        """Close the underlying async HTTP client (best-effort).

        WebSocket sockets are per-call context managers that close themselves;
        this only releases the REST/httpx connection pool. Guarded so it never
        raises during teardown.
        """
        try:
            wrapper = getattr(self._client, "_client_wrapper", None)
            http_wrapper = getattr(wrapper, "httpx_client", None)
            raw = getattr(http_wrapper, "httpx_client", None)
            if raw is not None and hasattr(raw, "aclose"):
                await raw.aclose()
                logger.debug("Deepgram httpx client closed.")
        except Exception as exc:  # teardown must not raise
            logger.debug("Deepgram close() ignored error: %s", exc)

    # ------------------------------------------------------------------
    # Batch media: Speech-to-Text
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
    ) -> TranscriptionResult:
        """Transcribe pre-recorded audio (batch STT).

        Maps to ``listen.v1.media.transcribe_file`` (bytes / file path) or
        ``listen.v1.media.transcribe_url`` (when ``url=`` is given in kwargs).
        Config defaults from ``[providers.deepgram.stt]`` are merged with explicit
        arguments and ``kwargs`` (e.g. ``diarize``, ``smart_format``,
        ``punctuate``, ``numerals``, ``utterances``, ``redact``, ``keyterm``,
        ``summarize``, ``topics``, ``intents``, ``sentiment`` …).

        Args:
            audio_data: Raw audio ``bytes`` or a path to an audio file.
            model: STT model (defaults to ``default_stt_model``).
            language: Language hint (e.g. ``"en"``, ``"multi"``).
            prompt: If given, mapped to nova-3 ``keyterm`` prompting (a list with
                one entry) — Deepgram has no free-text prompt for STT.
            response_format: Accepted for interface compatibility; Deepgram
                always returns structured JSON (other formats are ignored).
            temperature: Ignored (no Deepgram analog); logged at DEBUG.
            timestamp_granularities: Accepted for interface compatibility;
                Deepgram returns word/utterance timings when ``utterances``/
                diarization are enabled.
            **kwargs: Additional Deepgram params, plus ``url=`` to transcribe a
                remote URL instead of bytes.

        Returns:
            A :class:`TranscriptionResult` with the transcript, duration,
            per-utterance segments (speaker-labelled when diarized), and
            intelligence (summary/topics/intents/sentiments) in ``metadata``.

        Raises:
            ProviderError: On API errors (status/headers preserved).
            FileNotFoundError: If ``audio_data`` is a path that does not exist.
        """
        if temperature is not None:
            logger.debug("Deepgram transcribe_audio: 'temperature' ignored (no analog).")

        url = kwargs.pop("url", None)
        call_kwargs: dict[str, Any] = dict(kwargs)
        if prompt:
            # Map a free-text prompt onto nova-3 keyterm prompting.
            existing = call_kwargs.get("keyterm")
            if existing is None:
                call_kwargs["keyterm"] = [prompt]
            elif isinstance(existing, (list, tuple)):
                call_kwargs["keyterm"] = [*existing, prompt]
            else:
                call_kwargs["keyterm"] = [existing, prompt]

        explicit: dict[str, Any] = {"language": language}
        params = self._merge_params(
            self._stt_defaults,
            explicit,
            call_kwargs,
            allowed=_STT_BATCH_PARAMS,
            model=model,
            default_model=self.default_stt_model,
        )
        used_model = params.get("model", self.default_stt_model)

        try:
            if url is not None:
                response = await self._client.listen.v1.media.transcribe_url(
                    url=url, **params
                )
            else:
                request_bytes = self._coerce_audio_bytes(audio_data)
                response = await self._client.listen.v1.media.transcribe_file(
                    request=request_bytes, **params
                )
        except _DeepgramApiError as exc:
            raise self._map_api_error(exc, model=used_model) from exc
        except (ConfigError, ProviderError, FileNotFoundError):
            raise
        except Exception as exc:  # wrap unexpected SDK errors
            raise ProviderError(
                "deepgram",
                f"Deepgram transcription failed: {exc}",
                model_name=used_model,
                original_exception=exc,
            ) from exc

        return self._build_transcription_result(response, used_model, language)

    @staticmethod
    def _coerce_audio_bytes(audio_data: bytes | str) -> bytes:
        """Coerce ``audio_data`` to raw bytes.

        Args:
            audio_data: Raw bytes, or a filesystem path string.

        Returns:
            The audio bytes.

        Raises:
            FileNotFoundError: If a path string does not exist.
            ProviderError: If the input type is unsupported.
        """
        if isinstance(audio_data, bytes):
            return audio_data
        if isinstance(audio_data, str):
            path = Path(audio_data)
            if not path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_data}")
            return path.read_bytes()
        raise ProviderError(
            "deepgram",
            f"Unsupported audio_data type: {type(audio_data).__name__} "
            "(expected bytes or a file path str).",
            status_code=400,
            retryable=False,
        )

    def _build_transcription_result(
        self, response: Any, model: str, language: str | None
    ) -> TranscriptionResult:
        """Build a :class:`TranscriptionResult` from a Deepgram batch response.

        Args:
            response: The SDK ``ListenV1Response`` (or accepted async response).
            model: The model used.
            language: The requested language hint, if any.

        Returns:
            The populated :class:`TranscriptionResult`.
        """
        raw = self._to_dict(response)
        results = raw.get("results") or {}
        metadata = raw.get("metadata") or {}

        # Primary transcript: channel 0, alternative 0.
        transcript = ""
        confidence = None
        channels = results.get("channels") or []
        if channels:
            alts = channels[0].get("alternatives") or []
            if alts:
                transcript = alts[0].get("transcript", "") or ""
                confidence = alts[0].get("confidence")

        # Segments from utterances (when requested); speaker-labelled if diarized.
        segments: list[TranscriptionSegment] = []
        for utt in results.get("utterances") or []:
            segments.append(
                TranscriptionSegment(
                    text=utt.get("transcript", "") or "",
                    start=float(utt.get("start", 0.0) or 0.0),
                    end=float(utt.get("end", 0.0) or 0.0),
                    speaker=(
                        str(utt["speaker"]) if utt.get("speaker") is not None else None
                    ),
                )
            )

        duration = metadata.get("duration")
        result_metadata: dict[str, Any] = {
            "request_id": metadata.get("request_id"),
            "models": metadata.get("models"),
            "confidence": confidence,
        }
        # Surface audio-intelligence add-ons when present.
        for key in ("summary", "topics", "intents", "sentiments"):
            if results.get(key) is not None:
                result_metadata[key] = results[key]

        return TranscriptionResult(
            text=transcript,
            language=language,
            duration_seconds=float(duration) if duration is not None else None,
            segments=segments,
            model=model,
            metadata={k: v for k, v in result_metadata.items() if v is not None},
        )

    # ------------------------------------------------------------------
    # Batch media: Text-to-Speech
    # ------------------------------------------------------------------

    async def generate_speech(
        self,
        text: str,
        *,
        voice: str = "aura-2-thalia-en",
        model: str | None = None,
        response_format: str = "mp3",
        speed: float = 1.0,
        instructions: str | None = None,
        **kwargs: Any,
    ) -> SpeechResult:
        """Synthesize speech from text (batch TTS).

        Maps to ``speak.v1.audio.generate`` (an async byte generator) and
        collects the full audio. In Deepgram the **voice is encoded in the model
        id** (e.g. ``aura-2-thalia-en``); ``voice`` and ``model`` are reconciled:
        an explicit ``model`` wins, else ``voice`` (if it looks like a model id)
        is used, else the configured ``tts.model``/``default_tts_model``.

        Args:
            text: The text to synthesize.
            voice: A voice id (== model id), used when ``model`` is not given.
            model: Explicit TTS model id (highest precedence).
            response_format: Output encoding (``mp3``/``linear16``/``opus``/
                ``flac``/``aac``/``mulaw``/``alaw``). Mapped to ``encoding``.
            speed: Speaking-rate multiplier.
            instructions: Ignored (no Deepgram analog); logged at DEBUG.
            **kwargs: Additional Deepgram params (``container``, ``sample_rate``,
                ``bit_rate``, ``tag`` …).

        Returns:
            A :class:`SpeechResult` with the synthesized audio bytes, the
            resolved format, model, and voice.

        Raises:
            ProviderError: On API errors.
        """
        if instructions is not None:
            logger.debug("Deepgram generate_speech: 'instructions' ignored (no analog).")

        # Reconcile voice/model: explicit model > voice-as-model > config default.
        resolved_model = model or voice or self._tts_defaults.get("model") or self.default_tts_model

        explicit: dict[str, Any] = {
            "encoding": response_format,
            "speed": speed,
        }
        params = self._merge_params(
            self._tts_defaults,
            explicit,
            kwargs,
            allowed=_TTS_BATCH_PARAMS,
            model=resolved_model,
            default_model=self.default_tts_model,
        )
        used_model = params.get("model", resolved_model)
        used_encoding = params.get("encoding", "mp3")

        try:
            chunks: list[bytes] = []
            async for chunk in self._client.speak.v1.audio.generate(
                text=text, **params
            ):
                if chunk:
                    chunks.append(chunk)
            audio = b"".join(chunks)
        except _DeepgramApiError as exc:
            raise self._map_api_error(exc, model=used_model) from exc
        except (ConfigError, ProviderError):
            raise
        except Exception as exc:  # wrap unexpected SDK errors
            raise ProviderError(
                "deepgram",
                f"Deepgram speech synthesis failed: {exc}",
                model_name=used_model,
                original_exception=exc,
            ) from exc

        return SpeechResult(
            audio_data=audio,
            format=used_encoding,
            model=used_model,
            voice=used_model,  # voice is encoded in the model id
            metadata={"bytes": len(audio)},
        )

    # ------------------------------------------------------------------
    # Shared serialization helper
    # ------------------------------------------------------------------

    @staticmethod
    def _to_dict(obj: Any) -> dict[str, Any]:
        """Coerce an SDK Pydantic model (or dict) to a plain dict.

        Thin instance-facing wrapper over :func:`_to_plain_dict`.

        Args:
            obj: A Pydantic v2 model, a dict, or any object.

        Returns:
            A plain dict (empty if coercion is not possible).
        """
        return _to_plain_dict(obj)

    # ------------------------------------------------------------------
    # Streaming media: shared orchestration
    # ------------------------------------------------------------------

    async def _run_stream(
        self,
        connect: Any,
        params: dict[str, Any],
        mapper: Any,
        audio: AsyncIterable[bytes] | None,
        *,
        finalize_on_close: bool,
        keepalive_interval: float | None,
        supports_finalize: bool,
        model: str | None,
    ) -> AsyncGenerator[TranscriptionStreamEvent, None]:
        """Drive a live STT/Flux socket (fan-in audio, fan-out events).

        Opens the socket, spawns a producer task that pumps ``audio`` and then
        (optionally) finalizes and closes the send side, optionally spawns a
        keepalive task, and yields mapped events as the server returns them.
        Teardown is cancellation-safe; a producer-side exception is re-raised
        after the receive loop ends so callers observe send failures.

        Args:
            connect: The SDK ``connect`` context-manager factory.
            params: Normalized connect parameters (must include ``model``).
            mapper: ``raw_dict -> TranscriptionStreamEvent | None``.
            audio: Async iterable of PCM/encoded audio chunks, or ``None`` to
                drive the send side externally (used by the socket wrappers).
            finalize_on_close: Whether to send ``Finalize`` before closing.
            keepalive_interval: If > 0, send ``KeepAlive`` on this cadence.
            supports_finalize: Whether the socket exposes ``send_finalize``.
            model: The model in play (for error context).

        Yields:
            :class:`TranscriptionStreamEvent` objects.

        Raises:
            ProviderError: On connect/transport failures or producer errors.
        """
        producer_error: dict[str, BaseException] = {}

        try:
            async with connect(**params) as socket:

                async def _pump() -> None:
                    try:
                        if audio is not None:
                            async for chunk in audio:
                                if chunk:
                                    await socket.send_media(chunk)
                        if finalize_on_close and supports_finalize:
                            await socket.send_finalize()
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:  # capture for re-raise
                        producer_error["error"] = exc
                    finally:
                        # Always close the send side so the server drains and
                        # the receive loop terminates (best-effort).
                        with suppress(Exception):
                            await socket.send_close_stream()

                async def _keepalive() -> None:
                    try:
                        while True:
                            await asyncio.sleep(keepalive_interval)  # type: ignore[arg-type]
                            await socket.send_keep_alive()
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:  # keepalive failure is non-fatal
                        logger.debug("Deepgram keepalive stopped: %s", exc)

                pump_task = (
                    asyncio.create_task(_pump()) if audio is not None else None
                )
                ka_task = (
                    asyncio.create_task(_keepalive())
                    if keepalive_interval and keepalive_interval > 0
                    else None
                )

                try:
                    async for message in socket:
                        if isinstance(message, (bytes, bytearray)):
                            continue  # STT/Flux sockets only emit JSON events
                        event = mapper(self._to_dict(message))
                        if event is not None:
                            yield event
                finally:
                    for task in (pump_task, ka_task):
                        if task is not None and not task.done():
                            task.cancel()
                            with suppress(asyncio.CancelledError, Exception):
                                await task
        except _DeepgramApiError as exc:
            raise self._map_api_error(exc, model=model) from exc
        except (ConfigError, ProviderError):
            raise
        except Exception as exc:  # wrap unexpected transport errors
            raise ProviderError(
                "deepgram",
                f"Deepgram streaming failed: {exc}",
                model_name=model,
                original_exception=exc,
            ) from exc

        if "error" in producer_error:
            exc = producer_error["error"]
            if isinstance(exc, _DeepgramApiError):
                raise self._map_api_error(exc, model=model) from exc
            raise ProviderError(
                "deepgram",
                f"Deepgram audio producer failed: {exc}",
                model_name=model,
                original_exception=exc,
            ) from exc

    # ------------------------------------------------------------------
    # Streaming media: Speech-to-Text (live)
    # ------------------------------------------------------------------

    async def transcribe_stream(
        self,
        audio: AsyncIterable[bytes] | None = None,
        *,
        model: str | None = None,
        language: str | None = None,
        encoding: str | None = None,
        sample_rate: int | None = None,
        interim_results: bool | None = None,
        endpointing: int | bool | None = None,
        utterance_end_ms: int | None = None,
        vad_events: bool | None = None,
        keepalive_interval: float | None = None,
        finalize_on_close: bool = True,
        **kwargs: Any,
    ) -> AsyncGenerator[TranscriptionStreamEvent, None]:
        """Transcribe a live audio stream (``listen.v1.connect``).

        Pumps ``audio`` (an async iterable of raw/encoded PCM chunks) into a
        Deepgram streaming socket and yields :class:`TranscriptionStreamEvent`
        objects (interim + final transcripts, utterance-end, speech-started,
        metadata). Config defaults from ``[providers.deepgram.stt]`` and
        ``[providers.deepgram.stt.streaming]`` are merged with the explicit
        arguments and ``kwargs``.

        Args:
            audio: Async iterable of audio chunks. If ``None``, use
                :meth:`open_transcription_socket` and drive sends yourself.
            model: STT model (defaults to ``default_stt_model``).
            language: Language hint.
            encoding: Audio encoding (e.g. ``"linear16"``, ``"mulaw"``,
                ``"opus"``). Required when sending raw PCM.
            sample_rate: Audio sample rate in Hz (required for raw PCM).
            interim_results: Emit partial hypotheses before finals.
            endpointing: Silence (ms) before finalizing an utterance, or
                ``False`` to disable.
            utterance_end_ms: Emit ``UtteranceEnd`` after this much silence.
            vad_events: Emit ``SpeechStarted`` voice-activity events.
            keepalive_interval: If > 0, send periodic ``KeepAlive`` frames to
                hold the socket open across audio gaps (defaults to the
                configured ``stt.streaming.keepalive_interval``).
            finalize_on_close: Send ``Finalize`` before closing so trailing
                audio is flushed to a final transcript.
            **kwargs: Additional Deepgram streaming params.

        Yields:
            :class:`TranscriptionStreamEvent` objects.

        Raises:
            ProviderError: On connect/transport/producer errors.
        """
        explicit: dict[str, Any] = {
            "language": language,
            "encoding": encoding,
            "sample_rate": sample_rate,
            "interim_results": interim_results,
            "endpointing": endpointing,
            "utterance_end_ms": utterance_end_ms,
            "vad_events": vad_events,
        }
        params = self._merge_params(
            self._stt_defaults,
            self._stt_stream_defaults,
            explicit,
            kwargs,
            allowed=_STT_STREAM_PARAMS,
            model=model,
            default_model=self.default_stt_model,
        )
        used_model = params.get("model", self.default_stt_model)
        interval = (
            keepalive_interval
            if keepalive_interval is not None
            else self._stt_stream_defaults.get("keepalive_interval")
        )

        async for event in self._run_stream(
            self._client.listen.v1.connect,
            params,
            _map_stt_message,
            audio,
            finalize_on_close=finalize_on_close,
            keepalive_interval=interval,
            supports_finalize=True,
            model=used_model,
        ):
            yield event

    @asynccontextmanager
    async def open_transcription_socket(
        self,
        *,
        model: str | None = None,
        language: str | None = None,
        encoding: str | None = None,
        sample_rate: int | None = None,
        interim_results: bool | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[DeepgramTranscriptionStream]:
        """Open a low-level live-STT socket for manual full-duplex control.

        Unlike :meth:`transcribe_stream` (which pumps an audio iterable for
        you), this yields a :class:`DeepgramTranscriptionStream` so you can
        interleave ``send_audio``/``finalize``/``keepalive`` with iterating
        mapped events — e.g. for a microphone loop with backpressure.

        Args:
            model: STT model (defaults to ``default_stt_model``).
            language: Language hint.
            encoding: Audio encoding for raw PCM.
            sample_rate: Sample rate (Hz) for raw PCM.
            interim_results: Emit partial hypotheses.
            **kwargs: Additional Deepgram streaming params.

        Yields:
            A :class:`DeepgramTranscriptionStream` bound to the open socket.

        Raises:
            ProviderError: On connect/transport errors.
        """
        explicit: dict[str, Any] = {
            "language": language,
            "encoding": encoding,
            "sample_rate": sample_rate,
            "interim_results": interim_results,
        }
        params = self._merge_params(
            self._stt_defaults,
            self._stt_stream_defaults,
            explicit,
            kwargs,
            allowed=_STT_STREAM_PARAMS,
            model=model,
            default_model=self.default_stt_model,
        )
        used_model = params.get("model", self.default_stt_model)
        try:
            async with self._client.listen.v1.connect(**params) as socket:
                yield DeepgramTranscriptionStream(socket)
        except _DeepgramApiError as exc:
            raise self._map_api_error(exc, model=used_model) from exc

    # ------------------------------------------------------------------
    # Streaming media: Flux (v2 conversational STT)
    # ------------------------------------------------------------------

    async def transcribe_stream_flux(
        self,
        audio: AsyncIterable[bytes] | None = None,
        *,
        model: str | None = None,
        encoding: str | None = None,
        sample_rate: int | None = None,
        eot_threshold: float | None = None,
        eager_eot_threshold: float | None = None,
        eot_timeout_ms: int | None = None,
        keepalive_interval: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[TranscriptionStreamEvent, None]:
        """Transcribe a live stream with **Flux** (``listen.v2.connect``).

        Flux is Deepgram's conversational, turn-aware STT: instead of
        interim/final hypotheses it emits *turn* events (``StartOfTurn``,
        ``EagerEndOfTurn``, ``TurnResumed``, ``EndOfTurn``, ``Update``) with
        end-of-turn confidence — ideal for voice agents. Config defaults come
        from ``[providers.deepgram.flux]``.

        Args:
            audio: Async iterable of audio chunks (``None`` → use
                :meth:`open_flux_socket`).
            model: Flux model (defaults to ``default_flux_model``).
            encoding: Audio encoding for raw PCM.
            sample_rate: Sample rate (Hz) for raw PCM.
            eot_threshold: End-of-turn confidence threshold (0-1).
            eager_eot_threshold: Eager end-of-turn threshold for low-latency
                speculative turn-ends.
            eot_timeout_ms: Max silence (ms) before forcing end-of-turn.
            keepalive_interval: Reserved; Flux has no KeepAlive frame, so this
                is ignored (kept for signature symmetry).
            **kwargs: Additional Flux params (``keyterm``, ``language_hint`` …).

        Yields:
            :class:`TranscriptionStreamEvent` objects (turn events).

        Raises:
            ProviderError: On connect/transport/producer errors.
        """
        del keepalive_interval  # Flux has no keepalive frame.
        explicit: dict[str, Any] = {
            "encoding": encoding,
            "sample_rate": sample_rate,
            "eot_threshold": eot_threshold,
            "eager_eot_threshold": eager_eot_threshold,
            "eot_timeout_ms": eot_timeout_ms,
        }
        params = self._merge_params(
            self._flux_defaults,
            explicit,
            kwargs,
            allowed=_FLUX_PARAMS,
            model=model,
            default_model=self.default_flux_model,
        )
        used_model = params.get("model", self.default_flux_model)

        async for event in self._run_stream(
            self._client.listen.v2.connect,
            params,
            _map_flux_message,
            audio,
            finalize_on_close=False,
            keepalive_interval=None,
            supports_finalize=False,
            model=used_model,
        ):
            yield event

    @asynccontextmanager
    async def open_flux_socket(
        self,
        *,
        model: str | None = None,
        encoding: str | None = None,
        sample_rate: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[DeepgramFluxStream]:
        """Open a low-level Flux (v2) socket for manual full-duplex control.

        Args:
            model: Flux model (defaults to ``default_flux_model``).
            encoding: Audio encoding for raw PCM.
            sample_rate: Sample rate (Hz) for raw PCM.
            **kwargs: Additional Flux params.

        Yields:
            A :class:`DeepgramFluxStream` bound to the open socket.

        Raises:
            ProviderError: On connect/transport errors.
        """
        explicit: dict[str, Any] = {"encoding": encoding, "sample_rate": sample_rate}
        params = self._merge_params(
            self._flux_defaults,
            explicit,
            kwargs,
            allowed=_FLUX_PARAMS,
            model=model,
            default_model=self.default_flux_model,
        )
        used_model = params.get("model", self.default_flux_model)
        try:
            async with self._client.listen.v2.connect(**params) as socket:
                yield DeepgramFluxStream(socket)
        except _DeepgramApiError as exc:
            raise self._map_api_error(exc, model=used_model) from exc

    # ------------------------------------------------------------------
    # Streaming media: Text-to-Speech (live)
    # ------------------------------------------------------------------

    async def stream_speech(
        self,
        text: str | AsyncIterable[str],
        *,
        model: str | None = None,
        response_format: str | None = None,
        sample_rate: int | None = None,
        speed: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[bytes, None]:
        """Synthesize speech as a byte stream.

        Dual-mode by input type:

        * **``str``** → REST chunked synthesis (``speak.v1.audio.generate``):
          lowest-overhead path for a known, complete text; audio is yielded in
          chunks as it is produced.
        * **``AsyncIterable[str]``** → WebSocket synthesis
          (``speak.v1.connect``): send text incrementally (e.g. from an LLM
          token stream) and receive audio in real time.

        Config defaults come from ``[providers.deepgram.tts]`` (and
        ``[providers.deepgram.tts.streaming]`` for the WS path).

        Args:
            text: A complete string (REST) or an async iterable of text pieces
                (WebSocket).
            model: TTS model/voice (defaults to ``default_tts_model``).
            response_format: Output encoding (mapped to ``encoding``).
            sample_rate: Output sample rate (Hz).
            speed: Speaking-rate multiplier.
            **kwargs: Additional Deepgram TTS params.

        Yields:
            Audio ``bytes`` chunks.

        Raises:
            ProviderError: On API/transport errors.
        """
        resolved_model = model or self._tts_defaults.get("model") or self.default_tts_model

        if isinstance(text, str):
            # REST chunked path.
            explicit: dict[str, Any] = {
                "encoding": response_format,
                "sample_rate": sample_rate,
                "speed": speed,
            }
            params = self._merge_params(
                self._tts_defaults,
                explicit,
                kwargs,
                allowed=_TTS_BATCH_PARAMS,
                model=resolved_model,
                default_model=self.default_tts_model,
            )
            used_model = params.get("model", resolved_model)
            try:
                async for chunk in self._client.speak.v1.audio.generate(
                    text=text, **params
                ):
                    if chunk:
                        yield chunk
            except _DeepgramApiError as exc:
                raise self._map_api_error(exc, model=used_model) from exc
            except (ConfigError, ProviderError):
                raise
            except Exception as exc:
                raise ProviderError(
                    "deepgram",
                    f"Deepgram TTS streaming failed: {exc}",
                    model_name=used_model,
                    original_exception=exc,
                ) from exc
            return

        # WebSocket path (incremental text in, audio out).
        explicit = {
            "encoding": response_format,
            "sample_rate": sample_rate,
            "speed": speed,
        }
        params = self._merge_params(
            self._tts_defaults,
            self._tts_stream_defaults,
            explicit,
            kwargs,
            allowed=_TTS_STREAM_PARAMS,
            model=resolved_model,
            default_model=self.default_tts_model,
        )
        used_model = params.get("model", resolved_model)
        producer_error: dict[str, BaseException] = {}

        try:
            async with self._client.speak.v1.connect(**params) as socket:

                async def _pump() -> None:
                    try:
                        async for piece in text:
                            if piece:
                                await socket.send_text(
                                    SpeakV1Text(type="Speak", text=piece)
                                )
                        await socket.send_flush()
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        producer_error["error"] = exc
                    finally:
                        # Close once the text iterable is exhausted so the
                        # server drains the final audio and ends the stream.
                        with suppress(Exception):
                            await socket.send_close()

                pump_task = asyncio.create_task(_pump())
                try:
                    async for message in socket:
                        if isinstance(message, (bytes, bytearray)):
                            yield bytes(message)
                        # Control frames (Metadata/Flushed/Cleared/Warning) skipped.
                finally:
                    if not pump_task.done():
                        pump_task.cancel()
                        with suppress(asyncio.CancelledError, Exception):
                            await pump_task
        except _DeepgramApiError as exc:
            raise self._map_api_error(exc, model=used_model) from exc
        except (ConfigError, ProviderError):
            raise
        except Exception as exc:
            raise ProviderError(
                "deepgram",
                f"Deepgram TTS streaming failed: {exc}",
                model_name=used_model,
                original_exception=exc,
            ) from exc

        if "error" in producer_error:
            exc = producer_error["error"]
            if isinstance(exc, _DeepgramApiError):
                raise self._map_api_error(exc, model=used_model) from exc
            raise ProviderError(
                "deepgram",
                f"Deepgram TTS text producer failed: {exc}",
                model_name=used_model,
                original_exception=exc,
            ) from exc

    @asynccontextmanager
    async def open_speech_socket(
        self,
        *,
        model: str | None = None,
        response_format: str | None = None,
        sample_rate: int | None = None,
        speed: float | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[DeepgramSpeechStream]:
        """Open a low-level live-TTS socket for manual control.

        Yields a :class:`DeepgramSpeechStream` so you can interleave
        ``send_text``/``flush``/``clear``/``close`` with iterating audio bytes.

        Args:
            model: TTS model/voice (defaults to ``default_tts_model``).
            response_format: Output encoding (mapped to ``encoding``).
            sample_rate: Output sample rate (Hz).
            speed: Speaking-rate multiplier.
            **kwargs: Additional Deepgram TTS params.

        Yields:
            A :class:`DeepgramSpeechStream` bound to the open socket.

        Raises:
            ProviderError: On connect/transport errors.
        """
        explicit: dict[str, Any] = {
            "encoding": response_format,
            "sample_rate": sample_rate,
            "speed": speed,
        }
        params = self._merge_params(
            self._tts_defaults,
            self._tts_stream_defaults,
            explicit,
            kwargs,
            allowed=_TTS_STREAM_PARAMS,
            model=model,
            default_model=self.default_tts_model,
        )
        used_model = params.get("model", self.default_tts_model)
        try:
            async with self._client.speak.v1.connect(**params) as socket:
                yield DeepgramSpeechStream(socket)
        except _DeepgramApiError as exc:
            raise self._map_api_error(exc, model=used_model) from exc

    # ------------------------------------------------------------------
    # Voice Agent (agent.v1) — bidirectional STT -> LLM -> TTS over one socket
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_function(func: Any) -> dict[str, Any]:
        """Coerce a function definition to a Deepgram-shaped dict.

        Accepts an llmcore :class:`~llmcore.models.Tool` (``name`` /
        ``description`` / ``parameters``), a Pydantic model, or a plain dict
        (passed through unchanged).

        Args:
            func: The function definition.

        Returns:
            A plain dict suitable for ``agent.think.functions[]``.
        """
        if isinstance(func, dict):
            return func
        return _to_plain_dict(func)

    def _build_agent_settings(
        self,
        *,
        settings: dict[str, Any] | None = None,
        prompt: str | None = None,
        functions: list[Any] | None = None,
        greeting: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Assemble the ``Settings`` payload for the Voice Agent.

        Precedence (lowest -> highest): config ``[agent]`` defaults, ``**kwargs``
        (deep-merged), the ``prompt`` / ``functions`` / ``greeting`` conveniences,
        then an explicit ``settings=`` dict (deep-merged last so callers can
        always override).

        The config ``[agent]`` table is laid out with ``audio`` plus the agent
        sub-sections (``listen`` / ``think`` / ``speak`` / ``greeting`` /
        ``context``) at the top level for ergonomics; this method remaps them
        into the SDK's required shape: ``audio`` stays top-level while the
        sub-sections nest under ``agent``.

        Critically, ``prompt`` is **never** defaulted — it is only set when the
        caller supplies one (honouring the "no hardcoded prompt" invariant).

        Args:
            settings: A full/partial ``Settings`` dict, deep-merged last.
            prompt: System prompt -> ``agent.think.prompt`` (never defaulted).
            functions: Tool/function definitions -> ``agent.think.functions``.
            greeting: Spoken greeting -> ``agent.greeting``.
            **kwargs: Top-level ``Settings`` overrides (deep-merged).

        Returns:
            A plain dict ready for ``AgentV1Settings.model_validate``.
        """
        cfg = copy.deepcopy(self._agent_defaults)

        # ``audio`` is a top-level Settings key; everything else in our config
        # layout belongs under ``agent``.
        audio = cfg.pop("audio", {}) or {}
        top_passthrough: dict[str, Any] = {}
        for key in ("flags", "tags", "experimental", "mip_opt_out"):
            if key in cfg:
                top_passthrough[key] = cfg.pop(key)

        settings_dict: dict[str, Any] = {
            "type": "Settings",
            "audio": audio,
            "agent": cfg,  # remaining: listen/think/speak/greeting/context
        }
        settings_dict.update(top_passthrough)
        settings_dict.setdefault("mip_opt_out", self.mip_opt_out)

        # Top-level overrides from kwargs (deep-merged).
        if kwargs:
            settings_dict = _deep_merge(settings_dict, kwargs)

        agent_section = settings_dict.setdefault("agent", {})
        if prompt is not None:
            agent_section.setdefault("think", {})["prompt"] = prompt
        if functions is not None:
            agent_section.setdefault("think", {})["functions"] = [
                self._coerce_function(f) for f in functions
            ]
        if greeting is not None:
            agent_section["greeting"] = greeting

        # Explicit settings win (deep-merged last).
        if settings:
            settings_dict = _deep_merge(settings_dict, settings)

        return settings_dict

    @asynccontextmanager
    async def open_voice_agent(
        self,
        *,
        settings: dict[str, Any] | None = None,
        prompt: str | None = None,
        functions: list[Any] | None = None,
        greeting: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[DeepgramVoiceAgentSession]:
        """Open a Voice Agent session (full manual control).

        Connects, sends the assembled ``Settings`` once, and yields a
        :class:`DeepgramVoiceAgentSession`. The caller drives the session:
        push microphone audio with :meth:`~DeepgramVoiceAgentSession.send_audio`,
        iterate the session for :class:`VoiceAgentEvent`s (including ``AUDIO``
        events carrying TTS bytes), and respond to function calls. Teardown
        happens on context exit (the agent protocol has no explicit close frame).

        Args:
            settings: Full/partial ``Settings`` dict (deep-merged last).
            prompt: System prompt -> ``agent.think.prompt`` (never defaulted).
            functions: Tool/function definitions -> ``agent.think.functions``.
            greeting: Spoken greeting -> ``agent.greeting``.
            **kwargs: Top-level ``Settings`` overrides (deep-merged).

        Yields:
            A :class:`DeepgramVoiceAgentSession`.

        Raises:
            ProviderError: On API/transport errors.
        """
        settings_dict = self._build_agent_settings(
            settings=settings,
            prompt=prompt,
            functions=functions,
            greeting=greeting,
            **kwargs,
        )
        settings_obj = AgentV1Settings.model_validate(settings_dict)
        try:
            async with self._client.agent.v1.connect() as socket:
                await socket.send_settings(settings_obj)
                yield DeepgramVoiceAgentSession(socket)
        except _DeepgramApiError as exc:
            raise self._map_api_error(exc) from exc

    async def run_voice_agent(
        self,
        audio_source: AsyncIterable[bytes] | None = None,
        *,
        on_event: Any = None,
        function_handler: Any = None,
        settings: dict[str, Any] | None = None,
        prompt: str | None = None,
        functions: list[Any] | None = None,
        greeting: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[VoiceAgentEvent, None]:
        """Drive a Voice Agent end-to-end and stream its events.

        Concurrently pumps ``audio_source`` (if given) into the session and
        yields every :class:`VoiceAgentEvent`. When ``function_handler`` is
        supplied, each ``FUNCTION_CALL_REQUEST`` is auto-answered: every function
        in the request is dispatched to ``function_handler`` (sync or async) and
        the stringified result is sent back via ``FunctionCallResponse`` before
        the event is yielded. ``on_event`` (sync or async) is invoked for every
        event as a side-channel hook.

        The generator runs until the socket closes or the consumer stops
        iterating (``aclose``), at which point the audio pump is torn down.

        Args:
            audio_source: Async iterable of microphone audio chunks (or ``None``
                to drive audio yourself elsewhere).
            on_event: Optional ``callable(event)`` (awaited if it returns an
                awaitable).
            function_handler: Optional ``callable(VoiceAgentFunctionCall)`` whose
                return value (awaited if awaitable) is sent back as the function
                result.
            settings: Full/partial ``Settings`` dict (deep-merged last).
            prompt: System prompt -> ``agent.think.prompt`` (never defaulted).
            functions: Tool/function definitions -> ``agent.think.functions``.
            greeting: Spoken greeting -> ``agent.greeting``.
            **kwargs: Top-level ``Settings`` overrides (deep-merged).

        Yields:
            :class:`VoiceAgentEvent` objects in arrival order.

        Raises:
            ProviderError: On API/transport errors, or wrapping an exception
                raised by the audio source.
        """
        async with self.open_voice_agent(
            settings=settings,
            prompt=prompt,
            functions=functions,
            greeting=greeting,
            **kwargs,
        ) as session:
            producer_error: dict[str, BaseException] = {}

            async def _pump() -> None:
                try:
                    if audio_source is not None:
                        async for chunk in audio_source:
                            if chunk:
                                await session.send_audio(chunk)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # captured for re-raise
                    producer_error["error"] = exc

            pump_task = (
                asyncio.ensure_future(_pump()) if audio_source is not None else None
            )
            try:
                async for event in session:
                    if on_event is not None:
                        result = on_event(event)
                        if inspect.isawaitable(result):
                            await result
                    if (
                        function_handler is not None
                        and event.type == VoiceAgentEventType.FUNCTION_CALL_REQUEST
                    ):
                        await self._dispatch_function_calls(
                            session, event, function_handler
                        )
                    yield event
            finally:
                if pump_task is not None and not pump_task.done():
                    pump_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await pump_task

            if "error" in producer_error:
                err = producer_error["error"]
                raise ProviderError(
                    "deepgram",
                    f"Voice-agent audio producer failed: {err}",
                    retryable=False,
                    original_exception=err
                    if isinstance(err, Exception)
                    else None,
                ) from err

    @staticmethod
    async def _dispatch_function_calls(
        session: DeepgramVoiceAgentSession,
        event: VoiceAgentEvent,
        function_handler: Any,
    ) -> None:
        """Dispatch every function in a request and send each response back.

        Args:
            session: The active session.
            event: A ``FUNCTION_CALL_REQUEST`` event (``raw['functions']`` holds
                the full list; ``event.function_call`` is the first, for
                convenience).
            function_handler: ``callable(VoiceAgentFunctionCall)`` (sync/async).
        """
        functions = event.raw.get("functions") or []
        for fn in functions:
            call = VoiceAgentFunctionCall(
                id=fn.get("id", "") or "",
                name=fn.get("name", "") or "",
                arguments=_safe_json(fn.get("arguments")),
                client_side=bool(fn.get("client_side", True)),
                raw=fn,
            )
            result = function_handler(call)
            if inspect.isawaitable(result):
                result = await result
            await session.respond_to_function_call(
                call.id, call.name, "" if result is None else str(result)
            )

    # ------------------------------------------------------------------
    # Text Intelligence (read.v1)
    # ------------------------------------------------------------------

    async def analyze_text(
        self,
        text: str | None = None,
        *,
        url: str | None = None,
        summarize: bool | str | None = None,
        topics: bool | None = None,
        sentiment: bool | None = None,
        intents: bool | None = None,
        language: str | None = None,
        **kwargs: Any,
    ) -> TextAnalysisResult:
        """Run Deepgram text-intelligence (summary / topics / sentiment / intents).

        Exactly one of ``text`` or ``url`` must be provided.

        Args:
            text: Plain text to analyze.
            url: URL of text to analyze (mutually exclusive with ``text``).
            summarize: Enable summarization (``True`` or a mode string).
            topics: Enable topic detection.
            sentiment: Enable sentiment analysis.
            intents: Enable intent recognition.
            language: BCP-47 language hint (passthrough).
            **kwargs: Additional ``read`` params (``custom_topic``, ``tag`` …).

        Returns:
            A :class:`TextAnalysisResult`.

        Raises:
            ProviderError: If neither/both of ``text``/``url`` are given, or on
                API errors.
        """
        if (text is None) == (url is None):
            raise ProviderError(
                "deepgram",
                "analyze_text requires exactly one of `text` or `url`.",
                status_code=400,
                retryable=False,
            )
        request: dict[str, str] = {"url": url} if url is not None else {"text": text}

        params = self._normalize_params(
            {
                "summarize": summarize,
                "topics": topics,
                "sentiment": sentiment,
                "intents": intents,
                "language": language,
                **kwargs,
            }
        )
        try:
            response = await self._client.read.v1.text.analyze(request=request, **params)
        except _DeepgramApiError as exc:
            raise self._map_api_error(exc) from exc

        raw = self._to_dict(response)
        results = raw.get("results") or {}
        metadata = raw.get("metadata") or {}
        return TextAnalysisResult(
            summary=results.get("summary", {}).get("text")
            if isinstance(results.get("summary"), dict)
            else results.get("summary"),
            topics=self._extract_intel_section(results.get("topics")),
            intents=self._extract_intel_section(results.get("intents")),
            sentiments=results.get("sentiments"),
            language=language,
            model=(metadata.get("models") or [None])[0]
            if metadata.get("models")
            else None,
            request_id=metadata.get("request_id"),
            metadata=metadata,
            raw=raw,
        )

    @staticmethod
    def _extract_intel_section(section: Any) -> list[dict[str, Any]]:
        """Flatten a topics/intents section to a list of segment dicts.

        Deepgram nests these as ``{"segments": [{"topics"|"intents": [...]}, ...]}``.
        This returns the list of segment dicts (or an empty list).

        Args:
            section: The raw ``results.topics`` / ``results.intents`` value.

        Returns:
            A list of dicts (never ``None``).
        """
        if isinstance(section, dict):
            segments = section.get("segments")
            if isinstance(segments, list):
                return segments
        if isinstance(section, list):
            return section
        return []

    # ------------------------------------------------------------------
    # Token auth + account (auth.v1 / manage.v1) — pragmatic subset
    # ------------------------------------------------------------------

    async def grant_token(self, ttl_seconds: float | None = None) -> dict[str, Any]:
        """Mint a short-lived access token (temporary key) for clients/browsers.

        Args:
            ttl_seconds: Optional token lifetime in seconds (server default if
                omitted).

        Returns:
            ``{"access_token": str, "expires_in": float | None, "raw": dict}``.

        Raises:
            ProviderError: On API errors.
        """
        call_kwargs: dict[str, Any] = {}
        if ttl_seconds is not None:
            call_kwargs["ttl_seconds"] = ttl_seconds
        try:
            response = await self._client.auth.v1.tokens.grant(**call_kwargs)
        except _DeepgramApiError as exc:
            raise self._map_api_error(exc) from exc
        raw = self._to_dict(response)
        return {
            "access_token": raw.get("access_token"),
            "expires_in": raw.get("expires_in"),
            "raw": raw,
        }

    async def get_projects(self) -> dict[str, Any]:
        """List the projects visible to the current credential.

        Returns:
            The serialized ``ListProjectsV1Response`` as a dict.

        Raises:
            ProviderError: On API errors.

        Note:
            The full management API (balances, usage, keys, members, …) is
            available via the :attr:`client` escape hatch
            (``provider.client.manage.v1...``).
        """
        try:
            response = await self._client.manage.v1.projects.list()
        except _DeepgramApiError as exc:
            raise self._map_api_error(exc) from exc
        return self._to_dict(response)

    # ------------------------------------------------------------------
    # Escape hatch
    # ------------------------------------------------------------------

    @property
    def client(self) -> Any:
        """The underlying ``AsyncDeepgramClient`` (escape hatch for power users)."""
        return self._client


# ---------------------------------------------------------------------------
# Module-level helpers: serialization + event mapping
# ---------------------------------------------------------------------------


def _to_plain_dict(obj: Any) -> dict[str, Any]:
    """Coerce an SDK Pydantic model (or dict) to a plain (recursively) dict.

    Pydantic v2 ``model_dump`` is recursive, so nested message structures
    (``channel.alternatives[].transcript`` etc.) become nested dicts.

    Args:
        obj: A Pydantic model, a dict, or any object.

    Returns:
        A plain dict (empty if coercion is not possible).
    """
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    for attr in ("model_dump", "dict"):
        method = getattr(obj, attr, None)
        if callable(method):
            try:
                return method()
            except Exception:  # fall through to __dict__
                break
    return dict(getattr(obj, "__dict__", {}) or {})


def _map_stt_message(raw: dict[str, Any]) -> TranscriptionStreamEvent | None:
    """Map a ``listen.v1`` streaming message dict to a stream event.

    Args:
        raw: The message as a plain dict (already serialized).

    Returns:
        A :class:`TranscriptionStreamEvent`, or ``None`` for unrecognized
        messages (defensively skipped so the stream stays clean).
    """
    msg_type = raw.get("type")

    if msg_type == "Results":
        channel = raw.get("channel") or {}
        alternatives = channel.get("alternatives") or []
        alt = alternatives[0] if alternatives else {}
        is_final = raw.get("is_final")
        start = raw.get("start")
        duration = raw.get("duration")
        end = (
            start + duration
            if isinstance(start, (int, float)) and isinstance(duration, (int, float))
            else None
        )
        return TranscriptionStreamEvent(
            type=StreamEventType.FINAL if is_final else StreamEventType.INTERIM,
            text=alt.get("transcript", "") or "",
            is_final=is_final,
            speech_final=raw.get("speech_final"),
            start=start,
            end=end,
            confidence=alt.get("confidence"),
            words=alt.get("words") or [],
            channel_index=raw.get("channel_index"),
            raw=raw,
        )

    if msg_type == "UtteranceEnd":
        return TranscriptionStreamEvent(
            type=StreamEventType.UTTERANCE_END,
            end=raw.get("last_word_end"),
            channel_index=raw.get("channel"),
            raw=raw,
        )

    if msg_type == "SpeechStarted":
        return TranscriptionStreamEvent(
            type=StreamEventType.SPEECH_STARTED,
            start=raw.get("timestamp"),
            channel_index=raw.get("channel"),
            raw=raw,
        )

    if msg_type == "Metadata":
        return TranscriptionStreamEvent(type=StreamEventType.METADATA, raw=raw)

    return None


#: Flux ``event`` value -> llmcore stream-event type.
_FLUX_EVENT_MAP = {
    "Update": StreamEventType.UPDATE,
    "StartOfTurn": StreamEventType.START_OF_TURN,
    "EagerEndOfTurn": StreamEventType.EAGER_END_OF_TURN,
    "TurnResumed": StreamEventType.TURN_RESUMED,
    "EndOfTurn": StreamEventType.END_OF_TURN,
}


def _map_flux_message(raw: dict[str, Any]) -> TranscriptionStreamEvent | None:
    """Map a ``listen.v2`` (Flux) streaming message dict to a stream event.

    Args:
        raw: The message as a plain dict (already serialized).

    Returns:
        A :class:`TranscriptionStreamEvent`, or ``None`` for unrecognized
        messages.
    """
    msg_type = raw.get("type")

    if msg_type == "TurnInfo":
        event = raw.get("event")
        return TranscriptionStreamEvent(
            type=_FLUX_EVENT_MAP.get(event, StreamEventType.OTHER),
            text=raw.get("transcript", "") or "",
            is_final=event == "EndOfTurn",
            start=raw.get("audio_window_start"),
            end=raw.get("audio_window_end"),
            confidence=raw.get("end_of_turn_confidence"),
            words=raw.get("words") or [],
            raw=raw,
        )

    if msg_type == "Connected":
        return TranscriptionStreamEvent(type=StreamEventType.OPEN, raw=raw)

    if msg_type == "Error":
        return TranscriptionStreamEvent(type=StreamEventType.ERROR, raw=raw)

    return None


# ---------------------------------------------------------------------------
# Socket wrappers: thin, llmcore-shaped facades over raw SDK sockets
# ---------------------------------------------------------------------------


class DeepgramTranscriptionStream:
    """Full-duplex facade over a ``listen.v1`` streaming socket.

    Returned by :meth:`DeepgramProvider.open_transcription_socket`. Send audio
    with :meth:`send_audio`, optionally :meth:`finalize`/:meth:`keepalive`, and
    iterate the instance to receive mapped :class:`TranscriptionStreamEvent`s.
    """

    def __init__(self, socket: Any) -> None:
        """Bind to an open SDK socket.

        Args:
            socket: The raw ``AsyncV1SocketClient``.
        """
        self._socket = socket

    async def send_audio(self, chunk: bytes) -> None:
        """Send a chunk of audio bytes to the server."""
        await self._socket.send_media(chunk)

    async def finalize(self) -> None:
        """Force finalization of any buffered audio (``Finalize``)."""
        await self._socket.send_finalize()

    async def keepalive(self) -> None:
        """Send a ``KeepAlive`` to hold the socket open across audio gaps."""
        await self._socket.send_keep_alive()

    async def close(self) -> None:
        """Signal end-of-stream (``CloseStream``) so the server drains/closes."""
        await self._socket.send_close_stream()

    async def __aiter__(self) -> AsyncIterator[TranscriptionStreamEvent]:
        """Yield mapped transcription events from the socket."""
        async for message in self._socket:
            if isinstance(message, (bytes, bytearray)):
                continue
            event = _map_stt_message(_to_plain_dict(message))
            if event is not None:
                yield event

    @property
    def socket(self) -> Any:
        """The underlying raw SDK socket (escape hatch)."""
        return self._socket


class DeepgramFluxStream:
    """Full-duplex facade over a ``listen.v2`` (Flux) streaming socket.

    Returned by :meth:`DeepgramProvider.open_flux_socket`. Iterating yields
    turn-aware :class:`TranscriptionStreamEvent`s.
    """

    def __init__(self, socket: Any) -> None:
        """Bind to an open SDK socket.

        Args:
            socket: The raw ``AsyncV2SocketClient``.
        """
        self._socket = socket

    async def send_audio(self, chunk: bytes) -> None:
        """Send a chunk of audio bytes to the server."""
        await self._socket.send_media(chunk)

    async def configure(self, message: Any) -> None:
        """Send a mid-stream ``Configure`` message (raw passthrough)."""
        await self._socket.send_configure(message)

    async def close(self) -> None:
        """Signal end-of-stream (``CloseStream``)."""
        await self._socket.send_close_stream()

    async def __aiter__(self) -> AsyncIterator[TranscriptionStreamEvent]:
        """Yield mapped Flux turn events from the socket."""
        async for message in self._socket:
            if isinstance(message, (bytes, bytearray)):
                continue
            event = _map_flux_message(_to_plain_dict(message))
            if event is not None:
                yield event

    @property
    def socket(self) -> Any:
        """The underlying raw SDK socket (escape hatch)."""
        return self._socket


class DeepgramSpeechStream:
    """Full-duplex facade over a ``speak.v1`` streaming TTS socket.

    Returned by :meth:`DeepgramProvider.open_speech_socket`. Push text with
    :meth:`send_text`, :meth:`flush` to force synthesis, and iterate to receive
    audio ``bytes`` as they are produced.
    """

    def __init__(self, socket: Any) -> None:
        """Bind to an open SDK socket.

        Args:
            socket: The raw ``AsyncV1SocketClient`` (speak).
        """
        self._socket = socket

    async def send_text(self, text: str) -> None:
        """Queue a piece of text for synthesis."""
        if SpeakV1Text is None:  # pragma: no cover - SDK guaranteed when live
            raise ProviderError(
                "deepgram", "Deepgram SDK not available.", retryable=False
            )
        await self._socket.send_text(SpeakV1Text(type="Speak", text=text))

    async def flush(self) -> None:
        """Flush queued text so the server synthesizes it now (``Flush``)."""
        await self._socket.send_flush()

    async def clear(self) -> None:
        """Discard any unsynthesized buffered text (``Clear``)."""
        await self._socket.send_clear()

    async def close(self) -> None:
        """Close the synthesis stream (``Close``)."""
        await self._socket.send_close()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Yield audio ``bytes`` chunks (control frames are skipped)."""
        async for message in self._socket:
            if isinstance(message, (bytes, bytearray)):
                yield bytes(message)

    @property
    def socket(self) -> Any:
        """The underlying raw SDK socket (escape hatch)."""
        return self._socket


# ---------------------------------------------------------------------------
# Voice-agent helpers + session wrapper
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` (override wins).

    Nested dicts are merged key-by-key; any non-dict value (or a type mismatch)
    replaces the base value outright.

    Args:
        base: The lower-precedence mapping.
        override: The higher-precedence mapping.

    Returns:
        A new merged dict (inputs are not mutated).
    """
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = _deep_merge(existing, value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _safe_json(value: Any) -> dict[str, Any]:
    """Parse a JSON string to a dict, tolerating ``None``/already-parsed/garbage.

    Deepgram sends function-call ``arguments`` as a JSON **string**. This returns
    a dict for any valid JSON object string, an empty dict otherwise.

    Args:
        value: A JSON string, a dict, or ``None``.

    Returns:
        A dict (empty if parsing fails or the result is not a dict).
    """
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


#: Agent message ``type`` -> llmcore event type (no extra payload fields).
_AGENT_SIMPLE_EVENT_MAP = {
    "Welcome": VoiceAgentEventType.WELCOME,
    "SettingsApplied": VoiceAgentEventType.SETTINGS_APPLIED,
    "UserStartedSpeaking": VoiceAgentEventType.USER_STARTED_SPEAKING,
    "AgentThinking": VoiceAgentEventType.AGENT_THINKING,
    "AgentStartedSpeaking": VoiceAgentEventType.AGENT_STARTED_SPEAKING,
    "AgentAudioDone": VoiceAgentEventType.AGENT_AUDIO_DONE,
    "PromptUpdated": VoiceAgentEventType.PROMPT_UPDATED,
    "ThinkUpdated": VoiceAgentEventType.THINK_UPDATED,
    "SpeakUpdated": VoiceAgentEventType.SPEAK_UPDATED,
    "InjectionRefused": VoiceAgentEventType.INJECTION_REFUSED,
    "Error": VoiceAgentEventType.ERROR,
    "Warning": VoiceAgentEventType.WARNING,
}


def _map_agent_message(raw: dict[str, Any]) -> VoiceAgentEvent | None:
    """Map a Voice-Agent control message dict to a :class:`VoiceAgentEvent`.

    Binary (audio) frames are handled by the session wrapper, not here.

    Args:
        raw: The message as a plain dict (already serialized).

    Returns:
        A :class:`VoiceAgentEvent`, or ``None`` for unrecognized/ignored
        messages (e.g. ``History``).
    """
    msg_type = raw.get("type")

    if msg_type == "ConversationText":
        return VoiceAgentEvent(
            type=VoiceAgentEventType.CONVERSATION_TEXT,
            role=raw.get("role"),
            content=raw.get("content"),
            raw=raw,
        )

    if msg_type == "FunctionCallRequest":
        functions = raw.get("functions") or []
        first = functions[0] if functions else {}
        function_call = (
            VoiceAgentFunctionCall(
                id=first.get("id", "") or "",
                name=first.get("name", "") or "",
                arguments=_safe_json(first.get("arguments")),
                client_side=bool(first.get("client_side", True)),
                raw=first,
            )
            if first
            else None
        )
        return VoiceAgentEvent(
            type=VoiceAgentEventType.FUNCTION_CALL_REQUEST,
            function_call=function_call,
            raw=raw,
        )

    mapped = _AGENT_SIMPLE_EVENT_MAP.get(msg_type)
    if mapped is not None:
        return VoiceAgentEvent(type=mapped, content=raw.get("content"), raw=raw)

    return None


class DeepgramVoiceAgentSession:
    """Full-duplex facade over an ``agent.v1`` Voice-Agent socket.

    Returned by :meth:`DeepgramProvider.open_voice_agent`. Push microphone audio
    with :meth:`send_audio`, steer the agent at runtime (:meth:`update_prompt`,
    :meth:`update_think`, :meth:`update_speak`, :meth:`inject_user_message`,
    :meth:`inject_agent_message`), answer tool calls with
    :meth:`respond_to_function_call`, and iterate the instance to receive mapped
    :class:`VoiceAgentEvent`s (``AUDIO`` events carry TTS ``bytes``).

    The agent protocol has no explicit close frame; teardown is handled by the
    owning ``open_voice_agent`` context manager.
    """

    def __init__(self, socket: Any) -> None:
        """Bind to an open SDK socket (after ``send_settings``).

        Args:
            socket: The raw ``AsyncV1SocketClient`` (agent).
        """
        self._socket = socket

    async def send_audio(self, chunk: bytes) -> None:
        """Send a chunk of microphone audio to the agent."""
        await self._socket.send_media(chunk)

    async def inject_user_message(self, content: str) -> None:
        """Inject a text user message (as if the user spoke it)."""
        await self._socket.send_inject_user_message(
            AgentV1InjectUserMessage(type="InjectUserMessage", content=content)
        )

    async def inject_agent_message(self, message: str) -> None:
        """Make the agent speak an exact message immediately."""
        await self._socket.send_inject_agent_message(
            AgentV1InjectAgentMessage(type="InjectAgentMessage", message=message)
        )

    async def update_prompt(self, prompt: str) -> None:
        """Replace the agent's system prompt mid-conversation."""
        await self._socket.send_update_prompt(
            AgentV1UpdatePrompt(type="UpdatePrompt", prompt=prompt)
        )

    async def update_think(self, think: dict[str, Any]) -> None:
        """Update the ``think`` (LLM) configuration mid-conversation.

        Args:
            think: A partial ``think`` config (provider/model/prompt/functions …).
        """
        await self._socket.send_update_think(
            AgentV1UpdateThink.model_validate({"type": "UpdateThink", "think": think})
        )

    async def update_speak(self, speak: dict[str, Any]) -> None:
        """Update the ``speak`` (TTS) configuration mid-conversation.

        Args:
            speak: A partial ``speak`` config (provider/model …).
        """
        await self._socket.send_update_speak(
            AgentV1UpdateSpeak.model_validate({"type": "UpdateSpeak", "speak": speak})
        )

    async def respond_to_function_call(
        self, function_id: str, name: str, content: str
    ) -> None:
        """Send a client-side function result back to the agent.

        Args:
            function_id: The ``id`` from the originating request.
            name: The function name.
            content: The (stringified) function result.
        """
        await self._socket.send_function_call_response(
            AgentV1SendFunctionCallResponse(
                type="FunctionCallResponse",
                id=function_id,
                name=name,
                content=content,
            )
        )

    async def keepalive(self) -> None:
        """Send a ``KeepAlive`` to hold the socket open across silence."""
        await self._socket.send_keep_alive()

    async def __aiter__(self) -> AsyncIterator[VoiceAgentEvent]:
        """Yield mapped agent events; binary frames become ``AUDIO`` events."""
        async for message in self._socket:
            if isinstance(message, (bytes, bytearray)):
                yield VoiceAgentEvent(
                    type=VoiceAgentEventType.AUDIO, audio=bytes(message), raw={}
                )
                continue
            event = _map_agent_message(_to_plain_dict(message))
            if event is not None:
                yield event

    @property
    def socket(self) -> Any:
        """The underlying raw SDK socket (escape hatch)."""
        return self._socket
