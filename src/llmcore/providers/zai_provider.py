# src/llmcore/providers/zai_provider.py
"""
Z.ai (Zhipu AI / GLM) API provider implementation for the LLMCore library.

Handles interactions with the Z.ai Open Platform, which serves the GLM family
of models (``glm-5.2``, ``glm-5.1``, ``glm-4.7``, the ``glm-*v`` vision models,
``embedding-3``, etc.).  The chat endpoint is OpenAI-compatible
(``POST /chat/completions``), so this provider drives it through the ``openai``
Python SDK pointed at the Z.ai base URL, mirroring the DeepSeek/Kimi providers.

GLM-specific extensions handled here:

- **Thinking mode**: ``thinking = {"type": "enabled" | "disabled"}`` toggle to
  switch deep-reasoning on or off.
- **Reasoning effort**: ``reasoning_effort`` accepting
  ``none | minimal | low | medium | high | xhigh | max`` (GLM-5.2+).
- **Reasoning content**: ``reasoning_content`` field on assistant messages,
  surfaced in both streaming deltas and non-streaming responses.
- **Sampling control**: ``do_sample`` toggle plus the Z.ai open-interval
  ``(0, 1)`` constraint on ``temperature`` / ``top_p`` (auto-clamped).
- **Platform extras**: ``request_id``, ``user_id``, ``seed``,
  ``watermark_enabled``, ``sensitive_word_check`` and ``tool_stream`` pass
  through unchanged via ``extra_body``.
- **Cache-aware usage**: ``prompt_tokens_details.cached_tokens`` and
  ``completion_tokens_details.reasoning_tokens`` exposed via
  :meth:`ZaiProvider.extract_usage_details`.
- **Embeddings**: ``embedding-3`` / ``embedding-2`` via the OpenAI-compatible
  ``/embeddings`` endpoint.

Transport (selectable via the ``backend`` config key):

- ``"sdk"`` — the official synchronous ``zai-sdk`` (``ZaiClient`` /
  ``ZhipuAiClient``), bridged to async via ``asyncio.to_thread``.  This is the
  **default** when the SDK is installed.
- ``"openai"`` — the ``openai`` Python SDK (AsyncOpenAI) pointed at the Z.ai
  base URL (OpenAI-compatibility mode); native async.
- ``"httpx"`` — direct async HTTP calls against the REST endpoints.

When unset, the backend is auto-resolved in that order of preference based on
which libraries are installed (``zai-sdk`` → ``openai`` → ``httpx``).  An
explicitly requested backend that is unavailable falls back with a warning.

References:
  - https://docs.z.ai/
  - https://docs.z.ai/guides/llm/glm-5.2
  - https://docs.z.ai/api-reference/
  - Z.ai Python SDK (``zai-sdk``)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator
from typing import Any, Literal

# --- Optional native Z.ai SDK (zai-sdk) ---
# The official SDK is synchronous (httpx.Client based); we bridge its calls to
# async via ``asyncio.to_thread``.  Preferred backend when installed.
try:
    from zai import ZaiClient, ZhipuAiClient
    from zai.core import (
        APIAuthenticationError as ZaiAuthError,
    )
    from zai.core import (
        APIReachLimitError as ZaiRateLimitError,
    )
    from zai.core import (
        APIStatusError as ZaiStatusError,
    )
    from zai.core import (
        APITimeoutError as ZaiTimeoutError,
    )
    from zai.core import (
        ZaiError,
    )

    zai_sdk_available = True
except ImportError:
    zai_sdk_available = False
    ZaiClient = None  # type: ignore
    ZhipuAiClient = None  # type: ignore
    ZaiError = Exception  # type: ignore
    ZaiStatusError = Exception  # type: ignore
    ZaiAuthError = Exception  # type: ignore
    ZaiRateLimitError = Exception  # type: ignore
    ZaiTimeoutError = Exception  # type: ignore

# --- Optional OpenAI SDK (OpenAI-compatible fallback backend) ---
try:
    from openai import AsyncOpenAI
    from openai._exceptions import (
        APIConnectionError as OpenAIAPIConnectionError,
    )
    from openai._exceptions import (
        APIError as OpenAIAPIError,
    )
    from openai._exceptions import (
        APIStatusError as OpenAIAPIStatusError,
    )
    from openai._exceptions import (
        APITimeoutError as OpenAIAPITimeoutError,
    )
    from openai._exceptions import (
        OpenAIError,
    )

    openai_available = True
except ImportError:
    openai_available = False
    AsyncOpenAI = None  # type: ignore
    OpenAIError = Exception  # type: ignore
    OpenAIAPIError = Exception  # type: ignore
    OpenAIAPIStatusError = Exception  # type: ignore
    OpenAIAPIConnectionError = Exception  # type: ignore
    OpenAIAPITimeoutError = Exception  # type: ignore

try:
    import tiktoken

    tiktoken_available = True
except ImportError:
    tiktoken_available = False
    tiktoken = None  # type: ignore

try:
    import httpx

    httpx_available = True
except ImportError:
    httpx_available = False
    httpx = None  # type: ignore

from ..exceptions import ConfigError, ContextLengthError, ProviderError
from ..model_cards.registry import get_model_card_registry
from ..models import Message, ModelDetails, Tool, ToolCall
from ..models import Role as LLMCoreRole
from ..models_multimodal import (
    GeneratedImage,
    ImageGenerationResult,
    OCRResult,
    SpeechResult,
    TranscriptionResult,
)
from ..tokens import EstimateCounter as _EstimateCounter
from .base import BaseProvider, ContextPayload

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Overseas OpenAI-compatible API endpoint (Z.ai Open Platform).
_BASE_URL_OVERSEAS = "https://api.z.ai/api/paas/v4"

#: Mainland-China endpoint (Zhipu AI / BigModel).  Selectable via ``region``.
_BASE_URL_CHINA = "https://open.bigmodel.cn/api/paas/v4"

#: Default model when none configured.
_DEFAULT_MODEL = "glm-5.2"

#: Default embedding model.
_DEFAULT_EMBEDDING_MODEL = "embedding-3"

#: Default media models for the optional multimodal APIs.
_DEFAULT_IMAGE_MODEL = "cogview-4"
_DEFAULT_TTS_MODEL = "glm-tts"
_DEFAULT_STT_MODEL = "glm-asr-2512"
_DEFAULT_OCR_MODEL = "glm-ocr"
_DEFAULT_VIDEO_MODEL = "cogvideox-3"

#: Hardcoded context length map for the GLM family.
_CONTEXT_LENGTHS: dict[str, int] = {
    # GLM-5 generation (1M context)
    "glm-5.2": 1_000_000,
    "glm-5.1": 1_000_000,
    "glm-5": 1_000_000,
    "glm-5-turbo": 1_000_000,
    "glm-5v-turbo": 65_536,
    # GLM-4.x generation
    "glm-4.7": 131_072,
    "glm-4.6": 204_800,
    "glm-4.6v": 65_536,
    "glm-4.5": 131_072,
    "glm-4.5v": 65_536,
    "glm-4-32b-0414-128k": 131_072,
    # Embeddings
    "embedding-3": 8_192,
    "embedding-2": 8_192,
}

#: Models that natively accept image inputs.
_VISION_MODELS: frozenset[str] = frozenset(
    {"glm-5v-turbo", "glm-4.6v", "glm-4.5v", "glm-ocr"}
)

#: Valid thinking-mode types.
ThinkingType = Literal["enabled", "disabled"]

#: Reasoning-effort levels accepted by GLM-5.2+ (passed through verbatim).
_VALID_EFFORTS: frozenset[str] = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)


class ZaiProvider(BaseProvider):
    """First-class Z.ai (GLM) provider with native thinking-mode support.

    Configuration keys (under ``[providers.zai]``):

    - ``api_key`` / ``api_key_env_var`` — API credential.
    - ``backend`` — Transport: ``"sdk"`` (native zai-sdk, default when
      installed), ``"openai"`` (OpenAI-compatibility mode), or ``"httpx"``
      (direct REST).  Omit/``"auto"`` to auto-detect (sdk → openai → httpx).
    - ``base_url`` — Override the API URL (default: the overseas endpoint
      ``https://api.z.ai/api/paas/v4``).
    - ``region`` — ``"overseas"`` (default) or ``"china"``; selects the
      default ``base_url`` when one is not given explicitly.
    - ``default_model`` — Default model ID (default: ``glm-5.2``).
    - ``default_embedding_model`` — Default embedding model
      (default: ``embedding-3``).
    - ``timeout`` — HTTP request timeout in seconds (default: 300).
    - ``thinking`` — Default thinking mode: ``"enabled"`` or ``"disabled"``
      (default: ``"enabled"``).
    - ``reasoning_effort`` — Default reasoning effort
      (default: ``"high"``); one of
      ``none | minimal | low | medium | high | xhigh | max``.
    """

    default_model: str
    default_embedding_model: str
    _backend: str  # "sdk" | "openai" | "httpx"
    _sdk_client: Any  # zai.ZaiClient | None
    _client: AsyncOpenAI | None
    _encoding: Any  # tiktoken.Encoding | None
    _default_thinking: ThinkingType
    _default_reasoning_effort: str
    _api_key: str
    _base_url: str
    _timeout: float
    _http: Any  # httpx.AsyncClient | None

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False):
        """Initialize the Z.ai provider.

        Args:
            config: Provider-specific configuration dict from ``[providers.zai]``.
            log_raw_payloads: Whether to log raw request/response payloads.

        Raises:
            ConfigError: If the ``openai`` SDK is missing or no API key found.
        """
        super().__init__(config, log_raw_payloads)

        if not (zai_sdk_available or openai_available or httpx_available):
            raise ConfigError(
                "The Z.ai provider requires one of: the 'zai' SDK (preferred), "
                "the 'openai' SDK (compatibility mode), or 'httpx' (direct API). "
                "Install with: pip install llmcore[zai]"
            )

        # --- API key ---
        api_key = config.get("api_key") or os.environ.get(
            config.get("api_key_env_var", "ZAI_API_KEY")
        )
        if not api_key:
            api_key = os.environ.get("ZAI_API_KEY") or os.environ.get("ZHIPUAI_API_KEY")
        if not api_key:
            raise ConfigError(
                "Z.ai API key not found. Set ZAI_API_KEY or configure "
                "providers.zai.api_key / api_key_env_var."
            )

        # --- Model / timeout ---
        self.default_model = config.get("default_model", _DEFAULT_MODEL)
        self.default_embedding_model = config.get(
            "default_embedding_model", _DEFAULT_EMBEDDING_MODEL
        )
        # Optional media model defaults (image/TTS/STT/OCR/video).
        self.default_image_model = config.get("default_image_model", _DEFAULT_IMAGE_MODEL)
        self.default_tts_model = config.get("default_tts_model", _DEFAULT_TTS_MODEL)
        self.default_stt_model = config.get("default_stt_model", _DEFAULT_STT_MODEL)
        self.default_ocr_model = config.get("default_ocr_model", _DEFAULT_OCR_MODEL)
        self.default_video_model = config.get("default_video_model", _DEFAULT_VIDEO_MODEL)
        timeout = config.get("timeout", 300)
        self._timeout = float(timeout)

        # --- Thinking mode defaults ---
        thinking_raw = config.get("thinking", "enabled")
        if thinking_raw not in ("enabled", "disabled"):
            logger.warning("Invalid thinking value '%s'; defaulting to 'enabled'.", thinking_raw)
            thinking_raw = "enabled"
        self._default_thinking = thinking_raw  # type: ignore[assignment]

        effort_raw = str(config.get("reasoning_effort", "high")).lower()
        if effort_raw not in _VALID_EFFORTS:
            logger.warning(
                "Invalid reasoning_effort '%s'; defaulting to 'high'.", effort_raw
            )
            effort_raw = "high"
        self._default_reasoning_effort = effort_raw

        # --- Endpoint / region ---
        region = str(config.get("region", "overseas")).lower()
        default_base = _BASE_URL_CHINA if region == "china" else _BASE_URL_OVERSEAS
        base_url = config.get("base_url", default_base)
        self._region = region
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        # Lazily-created raw httpx client (used as the "httpx" backend and for
        # media endpoints not covered by the OpenAI-compatibility surface).
        self._http = None
        self._sdk_client = None
        self._client = None

        # --- Backend resolution: SDK preferred, then openai-compat, then httpx ---
        self._backend = self._resolve_backend(config.get("backend"))

        try:
            if self._backend == "sdk":
                # Pick the regional client; both accept an explicit base_url.
                client_cls = ZhipuAiClient if region == "china" else ZaiClient
                self._sdk_client = client_cls(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
                )
            elif self._backend == "openai":
                self._client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
                )
            # The "httpx" backend uses the lazily-created raw client; nothing to
            # initialize eagerly here.
            logger.debug(
                "Z.ai client initialized (backend=%s, base_url=%s, "
                "default_model=%s, thinking=%s, reasoning_effort=%s).",
                self._backend,
                base_url,
                self.default_model,
                self._default_thinking,
                self._default_reasoning_effort,
            )
        except Exception as e:
            raise ConfigError(f"Z.ai client initialization failed: {e}")

        # --- Tokenizer ---
        self._encoding = None
        if tiktoken_available:
            try:
                self._encoding = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                logger.warning("Failed to load tiktoken for Z.ai: %s", e)

    # =========================================================================
    # Backend resolution and SDK bridging
    # =========================================================================

    @staticmethod
    def _resolve_backend(requested: str | None) -> str:
        """Resolve the transport backend, honoring availability.

        Preference order when unset/``"auto"``: ``sdk`` → ``openai`` → ``httpx``.
        An explicitly requested backend that is unavailable falls back through
        the same chain with a warning.
        """
        available = {
            "sdk": zai_sdk_available,
            "openai": openai_available,
            "httpx": httpx_available,
        }
        order = ["sdk", "openai", "httpx"]

        req = (requested or "auto").lower()
        if req not in ("auto", *order):
            logger.warning("Unknown Z.ai backend '%s'; using auto-detection.", req)
            req = "auto"

        if req != "auto":
            if available.get(req):
                return req
            logger.warning(
                "Requested Z.ai backend '%s' is unavailable; falling back. "
                "Install with: pip install llmcore[zai]",
                req,
            )

        for backend in order:
            if available[backend]:
                if req != "auto" and backend != req:
                    logger.info("Z.ai backend resolved to '%s'.", backend)
                return backend
        # Should be unreachable given the __init__ guard.
        raise ConfigError("No usable Z.ai transport backend is installed.")

    async def _run_sdk(self, fn: Any) -> Any:
        """Run a synchronous SDK call in a worker thread (async bridge)."""
        return await asyncio.to_thread(fn)

    # =========================================================================
    # BaseProvider interface
    # =========================================================================

    def get_name(self) -> str:
        return self._provider_instance_name or "zai"

    async def get_models_details(self) -> list[ModelDetails]:
        """Discover available models via the ``GET /models`` endpoint.

        Z.ai exposes an OpenAI-compatible model listing.  When the endpoint is
        unavailable (or the active backend cannot list models) the static
        :data:`_CONTEXT_LENGTHS` table is used as a fallback so capability
        discovery always returns the known GLM models.
        """
        try:
            registry = get_model_card_registry()
        except Exception:
            registry = None

        try:
            raw_models = await self._list_models_raw()
            result: list[ModelDetails] = []
            for m in raw_models:
                mid = m.get("id")
                if not mid:
                    continue
                result.append(
                    self._build_model_details(
                        mid,
                        registry,
                        owned_by=m.get("owned_by"),
                        created=m.get("created"),
                    )
                )
            if result:
                return result
        except Exception as e:
            logger.warning(
                "Z.ai model listing failed (%s); falling back to static table.", e
            )

        # Fallback: synthesize details from the static context-length table.
        return [
            self._build_model_details(mid, registry)
            for mid in _CONTEXT_LENGTHS
            if not mid.startswith("embedding")
        ]

    async def _list_models_raw(self) -> list[dict[str, Any]]:
        """Return the raw ``/models`` listing as dicts, per active backend."""
        if self._backend == "openai" and self._client is not None:
            resp = await self._client.models.list()
            return [
                {"id": m.id, "owned_by": getattr(m, "owned_by", None), "created": getattr(m, "created", None)}
                for m in resp.data
            ]
        # SDK and httpx backends both read /models over raw HTTP (the zai SDK
        # does not expose a model-listing resource).
        resp = await self._raw_get("/models")
        return resp.json().get("data", [])

    def _build_model_details(
        self,
        mid: str,
        registry: Any | None,
        owned_by: str | None = None,
        created: int | None = None,
    ) -> ModelDetails:
        """Build a :class:`ModelDetails` for a GLM model id."""
        provider = self.get_name()
        ctx = self.get_max_context_length(mid)

        supports_tools = not mid.startswith("embedding")
        supports_vision = mid in _VISION_MODELS
        if registry:
            card = registry.get(provider, mid)
            if card and card.capabilities:
                supports_tools = card.capabilities.tool_use or card.capabilities.function_calling
                supports_vision = card.capabilities.vision

        return ModelDetails(
            id=mid,
            context_length=ctx,
            supports_streaming=True,
            supports_tools=supports_tools,
            supports_vision=supports_vision,
            provider_name=provider,
            metadata={
                "owned_by": owned_by or "zai",
                "created": created,
                "supports_thinking": not mid.startswith("embedding"),
            },
        )

    def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
        """Return the set of supported API parameters for GLM models."""
        return {
            # Note: Z.ai enforces an open interval (0, 1) on temperature/top_p;
            # out-of-range values are auto-clamped before dispatch.
            "temperature": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "top_p": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "max_tokens": {"type": "integer", "minimum": 1},
            "stop": {"type": "array", "items": {"type": "string"}},
            "response_format": {"type": "object"},
            "seed": {"type": "integer"},
            "do_sample": {"type": "boolean"},
            "stream_options": {"type": "object"},
            "user": {"type": "string"},
            # GLM / Z.ai-specific
            "thinking": {
                "type": "object",
                "description": 'Thinking mode toggle: {"type": "enabled"} or {"type": "disabled"}',
            },
            "reasoning_effort": {
                "type": "string",
                "enum": sorted(_VALID_EFFORTS),
            },
            "request_id": {"type": "string"},
            "user_id": {"type": "string"},
            "watermark_enabled": {"type": "boolean"},
            "sensitive_word_check": {"type": "object"},
            "tool_stream": {"type": "boolean"},
        }

    def get_max_context_length(self, model: str | None = None) -> int:
        """Return the maximum context length for a GLM model."""
        model_name = model or self.default_model
        limit = _CONTEXT_LENGTHS.get(model_name)
        if limit is not None:
            return limit

        # Try model card registry
        try:
            registry = get_model_card_registry()
            card = registry.get(self.get_name(), model_name)
            if card is not None:
                return card.get_context_length()
        except Exception:
            pass

        # GLM-5 family all have 1M context
        if model_name.startswith("glm-5"):
            return 1_000_000

        # Conservative fallback for unknown models
        logger.warning(
            "Unknown context length for Z.ai model '%s'. Falling back to 131072.",
            model_name,
        )
        return 131_072

    # =========================================================================
    # Chat completion
    # =========================================================================

    @staticmethod
    def _clamp_open_interval(value: Any) -> Any:
        """Clamp a sampling value into Z.ai's open interval ``(0, 1)``.

        Z.ai rejects ``temperature``/``top_p`` values of exactly 0 or 1, so we
        nudge them to the nearest in-range value (matching the official SDK).
        """
        try:
            v = float(value)
        except (TypeError, ValueError):
            return value
        if v <= 0:
            return 0.01
        if v >= 1:
            return 0.99
        return v

    def _resolve_thinking_params(
        self, kwargs: dict[str, Any]
    ) -> tuple[dict[str, str] | None, str | None]:
        """Resolve thinking mode and reasoning effort from kwargs + defaults.

        Returns:
            (thinking_obj, reasoning_effort) — both suitable for direct
            injection into the API request body.  ``reasoning_effort`` is only
            returned when thinking is enabled.
        """
        thinking = kwargs.pop("thinking", None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)

        # Build thinking object
        if thinking is None:
            thinking_obj: dict[str, str] | None = {"type": self._default_thinking}
        elif isinstance(thinking, dict):
            thinking_obj = thinking
        elif isinstance(thinking, str):
            thinking_obj = {"type": thinking}
        elif isinstance(thinking, bool):
            thinking_obj = {"type": "enabled" if thinking else "disabled"}
        else:
            thinking_obj = {"type": self._default_thinking}

        # Resolve reasoning effort
        if reasoning_effort is None:
            effort = self._default_reasoning_effort
        else:
            effort_str = str(reasoning_effort).lower()
            effort = effort_str if effort_str in _VALID_EFFORTS else self._default_reasoning_effort

        if thinking_obj and thinking_obj.get("type") == "enabled":
            return thinking_obj, effort
        return thinking_obj, None

    def _build_message_payload(self, msg: Message) -> dict[str, Any]:
        """Build a Z.ai-format message dict from an llmcore Message.

        Preserves ``reasoning_content`` on assistant messages (required for
        multi-turn tool calls under thinking mode) and supports structured
        multimodal content via ``content_parts`` metadata.
        """
        role_str = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        metadata = msg.metadata or {}

        content: Any = msg.content
        if "content_parts" in metadata:
            content = metadata["content_parts"]

        msg_dict: dict[str, Any] = {"role": role_str, "content": content}

        # Tool message needs tool_call_id
        if msg.role == LLMCoreRole.TOOL and msg.tool_call_id:
            msg_dict["tool_call_id"] = msg.tool_call_id

        # Assistant message may carry tool_calls and reasoning_content
        if role_str == "assistant":
            if "tool_calls" in metadata:
                msg_dict["tool_calls"] = metadata["tool_calls"]
                if not msg.content:
                    msg_dict["content"] = None
            if "reasoning_content" in metadata:
                msg_dict["reasoning_content"] = metadata["reasoning_content"]

        name = metadata.get("name")
        if name:
            msg_dict["name"] = name

        return msg_dict

    #: kwargs that must travel inside ``extra_body`` (not native OpenAI params).
    _EXTRA_BODY_KEYS: frozenset[str] = frozenset(
        {
            "do_sample",
            "request_id",
            "user_id",
            "watermark_enabled",
            "sensitive_word_check",
            "tool_stream",
        }
    )

    async def chat_completion(
        self,
        context: ContextPayload,
        model: str | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Perform a chat completion against the Z.ai API.

        Dispatches to the active backend (``sdk`` / ``openai`` / ``httpx``).
        Across all backends it provides:

        - Automatic thinking-mode parameter injection.
        - GLM platform extras (``do_sample``, ``request_id``, ``seed`` …).
        - ``reasoning_content`` extraction in streaming and non-streaming modes.
        - Open-interval clamping of ``temperature`` / ``top_p``.

        Additional kwargs accepted:

        - ``thinking``: ``dict | str | bool | None`` — override thinking mode.
        - ``reasoning_effort``: ``str | None`` — override reasoning effort.
        """
        model_name = model or self.default_model

        # Validate kwargs (only known parameters)
        supported = self.get_supported_parameters(model_name)
        for key in kwargs:
            if key not in supported:
                raise ValueError(f"Unsupported parameter '{key}' for Z.ai provider.")

        # Validate context
        if not (isinstance(context, list) and all(isinstance(msg, Message) for msg in context)):
            raise ProviderError(self.get_name(), "Context must be list[Message].")

        messages_payload = [self._build_message_payload(msg) for msg in context]
        if not messages_payload:
            raise ProviderError(self.get_name(), "No valid messages.")

        # --- Tool schema ---
        tools_payload = None
        if tools:
            tools_payload = [{"type": "function", "function": t.model_dump()} for t in tools]

        # --- Resolve thinking + split remaining kwargs ---
        thinking_obj, effort = self._resolve_thinking_params(kwargs)

        # ``sampling`` = native chat params; ``extras`` = GLM platform params.
        sampling: dict[str, Any] = {}
        extras: dict[str, Any] = {}
        for key, val in kwargs.items():
            if key in ("thinking", "reasoning_effort"):
                continue  # already handled
            if key in ("temperature", "top_p"):
                sampling[key] = self._clamp_open_interval(val)
            elif key in self._EXTRA_BODY_KEYS:
                extras[key] = val
            else:
                sampling[key] = val

        if tool_choice:
            sampling["tool_choice"] = tool_choice
        if stream:
            sampling.setdefault("stream_options", {"include_usage": True})

        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "RAW ZAI REQUEST (backend=%s): %s",
                self._backend,
                json.dumps(
                    {
                        "model": model_name,
                        "messages": messages_payload,
                        "stream": stream,
                        "thinking": thinking_obj,
                        "reasoning_effort": effort,
                        "tools": tools_payload,
                        **sampling,
                        **extras,
                    },
                    indent=2,
                    default=str,
                ),
            )

        try:
            if self._backend == "sdk":
                return await self._chat_via_sdk(
                    model_name, messages_payload, stream, tools_payload,
                    thinking_obj, effort, sampling, extras,
                )
            if self._backend == "openai":
                return await self._chat_via_openai(
                    model_name, messages_payload, stream, tools_payload,
                    thinking_obj, effort, sampling, extras,
                )
            return await self._chat_via_httpx(
                model_name, messages_payload, stream, tools_payload,
                thinking_obj, effort, sampling, extras,
            )
        except (ProviderError, ContextLengthError, ValueError):
            raise
        except Exception as e:
            self._raise_chat_error(e, model_name)

    async def _chat_via_sdk(
        self,
        model_name: str,
        messages: list[dict[str, Any]],
        stream: bool,
        tools_payload: list[dict[str, Any]] | None,
        thinking_obj: dict[str, str] | None,
        effort: str | None,
        sampling: dict[str, Any],
        extras: dict[str, Any],
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Chat via the native (synchronous) zai-sdk, bridged to async."""
        call_kwargs: dict[str, Any] = {**sampling, **extras}
        if tools_payload:
            call_kwargs["tools"] = tools_payload
        if thinking_obj:
            call_kwargs["thinking"] = thinking_obj
        if effort:
            call_kwargs["reasoning_effort"] = effort

        resp = await self._run_sdk(
            lambda: self._sdk_client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=stream,
                **call_kwargs,
            )
        )

        if stream:
            return self._bridge_sdk_stream(resp)
        return self._normalize_obj(resp)

    async def _bridge_sdk_stream(
        self, iterator: Any
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Bridge the SDK's synchronous stream iterator to an async generator."""
        sentinel = object()

        def _next() -> Any:
            try:
                return next(iterator)
            except StopIteration:
                return sentinel

        while True:
            chunk = await asyncio.to_thread(_next)
            if chunk is sentinel:
                break
            chunk_dict = self._normalize_obj(chunk)
            if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                logger.debug("RAW ZAI SDK STREAM CHUNK: %s", json.dumps(chunk_dict, default=str))
            yield chunk_dict

    async def _chat_via_openai(
        self,
        model_name: str,
        messages: list[dict[str, Any]],
        stream: bool,
        tools_payload: list[dict[str, Any]] | None,
        thinking_obj: dict[str, str] | None,
        effort: str | None,
        sampling: dict[str, Any],
        extras: dict[str, Any],
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Chat via the OpenAI-compatible AsyncOpenAI client (extra_body extras)."""
        api_kwargs: dict[str, Any] = dict(sampling)
        extra_body: dict[str, Any] = dict(extras)
        if thinking_obj:
            extra_body["thinking"] = thinking_obj
        if effort:
            extra_body["reasoning_effort"] = effort
        if extra_body:
            api_kwargs["extra_body"] = extra_body
        if tools_payload:
            api_kwargs["tools"] = tools_payload

        resp = await self._client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=stream,
            **api_kwargs,
        )  # type: ignore

        if stream:

            async def stream_wrapper() -> AsyncGenerator[dict[str, Any], None]:
                async for chunk in resp:  # type: ignore
                    yield self._normalize_obj(chunk)

            return stream_wrapper()
        return self._normalize_obj(resp)

    async def _chat_via_httpx(
        self,
        model_name: str,
        messages: list[dict[str, Any]],
        stream: bool,
        tools_payload: list[dict[str, Any]] | None,
        thinking_obj: dict[str, str] | None,
        effort: str | None,
        sampling: dict[str, Any],
        extras: dict[str, Any],
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Chat via direct httpx calls against ``/chat/completions``."""
        body: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "stream": stream,
            **sampling,
            **extras,
        }
        if tools_payload:
            body["tools"] = tools_payload
        if thinking_obj:
            body["thinking"] = thinking_obj
        if effort:
            body["reasoning_effort"] = effort

        if stream:
            return self._httpx_sse_stream(body)
        resp = await self._raw_post("/chat/completions", json=body)
        return resp.json()

    async def _httpx_sse_stream(
        self, body: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream ``/chat/completions`` over httpx, parsing SSE ``data:`` lines."""
        client = self._get_http()
        try:
            async with client.stream("POST", "/chat/completions", json=body) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        continue
        except httpx.HTTPStatusError as e:
            self._raise_chat_error(e, body.get("model", ""))
        except httpx.HTTPError as e:
            raise ProviderError(self.get_name(), f"Streaming error: {e}")

    @staticmethod
    def _normalize_obj(obj: Any) -> dict[str, Any]:
        """Normalize an SDK/OpenAI pydantic response to a plain dict."""
        if hasattr(obj, "model_dump"):
            return obj.model_dump(exclude_none=True)
        if isinstance(obj, dict):
            return obj
        return dict(obj)

    def _raise_chat_error(self, e: Exception, model_name: str) -> None:
        """Map a backend exception to an llmcore ProviderError/ContextLengthError."""
        status = getattr(e, "status_code", None)
        if status is None:
            resp = getattr(e, "response", None)
            status = getattr(resp, "status_code", None)
        msg = str(e)
        if status is not None:
            logger.error("Z.ai status error (%s): %s", status, msg)
            if status == 400 and "context" in msg.lower() and "length" in msg.lower():
                raise ContextLengthError(
                    provider_name=self.get_name(),
                    model=model_name,
                    max_tokens=self.get_max_context_length(model_name),
                    requested_tokens=None,
                    message=msg,
                )
            if status in (401, 403):
                raise ProviderError(
                    self.get_name(),
                    f"Authentication failed for Z.ai. Check ZAI_API_KEY / "
                    f"providers.zai.api_key. Error: {msg}",
                )
            if status == 400 and any(
                p in msg.lower() for p in ("model not exist", "does not exist", "model_not_found")
            ):
                raise ProviderError(
                    self.get_name(),
                    f"Model '{model_name}' not found on Z.ai. Known GLM models "
                    f"include glm-5.2, glm-5.1, glm-4.7, glm-4.6v. Error: {msg}",
                )
            raise ProviderError(self.get_name(), f"API Error ({status}): {msg}")
        if isinstance(e, (ZaiTimeoutError, OpenAIAPITimeoutError)):
            raise ProviderError(self.get_name(), f"Timeout: {e}")
        logger.error("Unexpected Z.ai error: %s", e, exc_info=True)
        raise ProviderError(self.get_name(), f"Error: {e}")

    # =========================================================================
    # Embeddings
    # =========================================================================

    async def create_embeddings(
        self,
        input_texts: str | list[str],
        *,
        model: str | None = None,
        dimensions: int | None = None,
        encoding_format: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create text embeddings via the OpenAI-compatible ``/embeddings`` endpoint.

        Args:
            input_texts: Single string or list of strings to embed.
            model: Embedding model (default: ``embedding-3``).
            dimensions: Desired output dimensionality (if the model supports it).
            encoding_format: Response encoding (``float`` or ``base64``).
            **kwargs: Extra API parameters.

        Returns:
            Raw API response dict with ``data``, ``model`` and ``usage``.
        """
        embed_model = model or self.default_embedding_model
        api_kwargs: dict[str, Any] = dict(kwargs)
        if dimensions is not None:
            api_kwargs["dimensions"] = dimensions
        if encoding_format is not None:
            api_kwargs["encoding_format"] = encoding_format

        try:
            if self._backend == "sdk":
                resp = await self._run_sdk(
                    lambda: self._sdk_client.embeddings.create(
                        model=embed_model, input=input_texts, **api_kwargs
                    )
                )
                return self._normalize_obj(resp)
            if self._backend == "openai":
                resp = await self._client.embeddings.create(
                    model=embed_model,
                    input=input_texts,
                    **api_kwargs,
                )
                return self._normalize_obj(resp)
            # httpx backend: direct POST /embeddings.
            body = {"model": embed_model, "input": input_texts, **api_kwargs}
            resp = await self._raw_post("/embeddings", json=body)
            return resp.json()
        except (ProviderError, ContextLengthError):
            raise
        except Exception as e:
            self._raise_chat_error(e, embed_model)

    # =========================================================================
    # Multimodal media APIs (image / TTS / STT / OCR / video / web search)
    # =========================================================================

    def _get_http(self) -> Any:
        """Return (lazily creating) the raw httpx client for media endpoints."""
        if not httpx_available:
            raise ProviderError(
                self.get_name(),
                "The 'httpx' package is required for Z.ai media APIs.",
            )
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self._base_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout,
            )
        return self._http

    async def _raw_post(
        self,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: Any | None = None,
    ) -> Any:
        """POST to a Z.ai endpoint and return the parsed httpx response."""
        client = self._get_http()
        try:
            resp = await client.post(path, json=json, data=data, files=files)
            resp.raise_for_status()
            return resp
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            body = e.response.text
            if status in (401, 403):
                raise ProviderError(
                    self.get_name(),
                    f"Authentication failed for Z.ai. Check ZAI_API_KEY. Error: {body}",
                )
            raise ProviderError(self.get_name(), f"API Error ({status}): {body}")
        except httpx.HTTPError as e:
            raise ProviderError(self.get_name(), f"HTTP error: {e}")

    async def _raw_get(self, path: str) -> Any:
        """GET a Z.ai endpoint and return the parsed httpx response."""
        client = self._get_http()
        try:
            resp = await client.get(path)
            resp.raise_for_status()
            return resp
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                self.get_name(), f"API Error ({e.response.status_code}): {e.response.text}"
            )
        except httpx.HTTPError as e:
            raise ProviderError(self.get_name(), f"HTTP error: {e}")

    async def _media_json(
        self,
        *,
        sdk_call: Any,
        path: str,
        json: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: Any | None = None,
    ) -> dict[str, Any]:
        """Run a JSON-returning media op via the SDK (preferred) or httpx."""
        if self._backend == "sdk":
            try:
                sdk_resp = await self._run_sdk(sdk_call)
            except (ProviderError, ContextLengthError):
                raise
            except Exception as e:
                self._raise_chat_error(e, "")
            return self._normalize_obj(sdk_resp)
        resp = await self._raw_post(path, json=json, data=data, files=files)
        return resp.json()

    async def generate_image(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
        size: str | None = None,
        quality: str | None = None,
        response_format: str = "url",
        style: str | None = None,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        """Generate images via ``POST /images/generations`` (CogView / GLM-Image).

        Z.ai returns image URLs by default; ``response_format`` is accepted for
        API parity but the platform primarily returns ``url`` entries.
        """
        image_model = model or self.default_image_model
        body: dict[str, Any] = {"model": image_model, "prompt": prompt}
        if size is not None:
            body["size"] = size
        if quality is not None:
            body["quality"] = quality
        body.update(kwargs)

        payload = await self._media_json(
            sdk_call=lambda: self._sdk_client.images.generations(
                **{k: v for k, v in body.items()}
            ),
            path="/images/generations",
            json=body,
        )
        images: list[GeneratedImage] = []
        for item in payload.get("data", []):
            images.append(
                GeneratedImage(
                    data=item.get("b64_json"),
                    url=item.get("url"),
                    revised_prompt=item.get("revised_prompt"),
                    format="png",
                )
            )
        return ImageGenerationResult(
            images=images,
            model=image_model,
            metadata={"created": payload.get("created"), "content_filter": payload.get("content_filter")},
        )

    async def generate_speech(
        self,
        text: str,
        *,
        voice: str = "tongtong",
        model: str | None = None,
        response_format: str = "wav",
        speed: float = 1.0,
        instructions: str | None = None,
        **kwargs: Any,
    ) -> SpeechResult:
        """Generate speech audio via ``POST /audio/speech`` (GLM-TTS).

        Note: Z.ai limits ``text`` to ~1024 characters and returns ``wav`` or
        ``pcm`` audio.  ``speed``/``instructions`` are accepted for interface
        parity but ignored by the GLM-TTS endpoint.
        """
        tts_model = model or self.default_tts_model
        body: dict[str, Any] = {
            "model": tts_model,
            "input": text,
            "voice": voice,
            "response_format": response_format,
        }
        body.update(kwargs)

        # The endpoint returns raw audio bytes (or JSON on error, already raised).
        if self._backend == "sdk":
            sdk_resp = await self._run_sdk(lambda: self._sdk_client.audio.speech(**body))
            audio_bytes = getattr(sdk_resp, "content", None)
            if audio_bytes is None and hasattr(sdk_resp, "read"):
                audio_bytes = sdk_resp.read()
        else:
            resp = await self._raw_post("/audio/speech", json=body)
            audio_bytes = resp.content
        return SpeechResult(
            audio_data=audio_bytes or b"",
            format=response_format,
            model=tts_model,
            voice=voice,
        )

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
        """Transcribe audio via ``POST /audio/transcriptions`` (GLM-ASR).

        Args:
            audio_data: Raw audio bytes or a path to an audio file.
            model: STT model (default: ``glm-asr-2512``).
            prompt: Optional decoding hint.
        """
        stt_model = model or self.default_stt_model

        if isinstance(audio_data, str):
            # Treat as a file path.
            try:
                import aiofiles

                async with aiofiles.open(audio_data, "rb") as f:
                    file_bytes = await f.read()
                filename = os.path.basename(audio_data)
            except ImportError:
                with open(audio_data, "rb") as f:
                    file_bytes = f.read()
                filename = os.path.basename(audio_data)
        else:
            file_bytes = audio_data
            filename = "audio.wav"

        data: dict[str, Any] = {"model": stt_model}
        if prompt:
            data["prompt"] = prompt
        if temperature is not None:
            data["temperature"] = str(temperature)
        data.update({k: str(v) for k, v in kwargs.items()})
        files = {"file": (filename, file_bytes)}

        payload = await self._media_json(
            sdk_call=lambda: self._sdk_client.audio.transcriptions.create(
                file=(filename, file_bytes), **data
            ),
            path="/audio/transcriptions",
            data=data,
            files=files,
        )
        return TranscriptionResult(
            text=payload.get("text", ""),
            language=language or payload.get("language"),
            model=stt_model,
            metadata={k: v for k, v in payload.items() if k != "text"},
        )

    async def ocr(
        self,
        document: str | bytes | dict[str, Any],
        *,
        model: str | None = None,
        pages: list[int] | None = None,
        include_image_base64: bool | None = None,
        image_limit: int | None = None,
        image_min_size: int | None = None,
        **kwargs: Any,
    ) -> OCRResult:
        """Process a document with GLM-OCR via ``POST /layout_parsing``.

        Args:
            document: A URL string or base64-encoded image/PDF.  A dict with a
                ``file`` key is also accepted.
            model: OCR model (default: ``glm-ocr``).
            pages: Optional 0-based page range; mapped to
                ``start_page_id``/``end_page_id``.
        """
        ocr_model = model or self.default_ocr_model

        if isinstance(document, dict):
            file_ref = document.get("file") or document.get("url")
        elif isinstance(document, bytes):
            import base64

            file_ref = base64.b64encode(document).decode("ascii")
        else:
            file_ref = document

        body: dict[str, Any] = {"model": ocr_model, "file": file_ref}
        if pages:
            body["start_page_id"] = pages[0]
            body["end_page_id"] = pages[-1]
        if include_image_base64 is not None:
            body["return_crop_images"] = include_image_base64
        body.update(kwargs)

        payload = await self._media_json(
            sdk_call=lambda: self._sdk_client.layout_parsing.create(**body),
            path="/layout_parsing",
            json=body,
        )
        raw_pages = payload.get("pages") or payload.get("data") or []
        if isinstance(raw_pages, dict):
            raw_pages = [raw_pages]
        return OCRResult(
            pages=raw_pages if isinstance(raw_pages, list) else [],
            model=ocr_model,
            pages_processed=len(raw_pages) if isinstance(raw_pages, list) else 0,
            metadata={k: v for k, v in payload.items() if k not in ("pages", "data")},
        )

    async def generate_video(
        self,
        prompt: str | None = None,
        *,
        model: str | None = None,
        image_url: str | None = None,
        quality: str | None = None,
        size: str | None = None,
        duration: int | None = None,
        fps: int | None = None,
        with_audio: bool | None = None,
        wait: bool = False,
        poll_interval: float = 5.0,
        max_wait_seconds: float = 300.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a video via ``POST /videos/generations`` (CogVideoX).

        Z.ai video generation is asynchronous: the initial call returns a task
        ``id`` with status ``PROCESSING``.  When ``wait=True`` this polls
        ``GET /async-result/{id}`` until the task completes (or ``max_wait_seconds``
        elapses) and returns the final result dict; otherwise it returns the
        initial task descriptor for the caller to poll via
        :meth:`retrieve_video_result`.

        Args:
            prompt: Text prompt (text-to-video).
            image_url: Optional source image (image-to-video).
            model: Video model (default: ``cogvideox-3``).
            wait: If True, block until the task finishes.

        Returns:
            The raw Z.ai video task/result dictionary.
        """
        video_model = model or self.default_video_model
        body: dict[str, Any] = {"model": video_model}
        if prompt is not None:
            body["prompt"] = prompt
        if image_url is not None:
            body["image_url"] = image_url
        if quality is not None:
            body["quality"] = quality
        if size is not None:
            body["size"] = size
        if duration is not None:
            body["duration"] = duration
        if fps is not None:
            body["fps"] = fps
        if with_audio is not None:
            body["with_audio"] = with_audio
        body.update(kwargs)

        task = await self._media_json(
            sdk_call=lambda: self._sdk_client.videos.generations(**body),
            path="/videos/generations",
            json=body,
        )
        if not wait:
            return task

        task_id = task.get("id")
        if not task_id:
            return task

        elapsed = 0.0
        while elapsed < max_wait_seconds:
            result = await self.retrieve_video_result(task_id)
            status = str(result.get("task_status", "")).upper()
            if status in ("SUCCESS", "FAIL", "FAILED"):
                return result
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        raise ProviderError(
            self.get_name(),
            f"Video task '{task_id}' did not complete within {max_wait_seconds}s.",
        )

    async def retrieve_video_result(self, task_id: str) -> dict[str, Any]:
        """Fetch the status/result of an async video task.

        Uses the SDK ``videos.retrieve_videos_result`` when available, otherwise
        ``GET /async-result/{id}`` directly.
        """
        if self._backend == "sdk":
            sdk_resp = await self._run_sdk(
                lambda: self._sdk_client.videos.retrieve_videos_result(id=task_id)
            )
            return self._normalize_obj(sdk_resp)
        resp = await self._raw_get(f"/async-result/{task_id}")
        return resp.json()

    async def web_search(
        self,
        query: str,
        *,
        search_engine: str = "search_std",
        count: int | None = None,
        search_domain_filter: str | None = None,
        search_recency_filter: str | None = None,
        content_size: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run a web search via ``POST /web_search`` (Z.ai Web Search API).

        Args:
            query: The search query string.
            search_engine: Engine id (default ``search_std``).
            count: Maximum number of results.

        Returns:
            The raw Z.ai web-search response dict (``search_result`` list, etc.).
        """
        body: dict[str, Any] = {"search_query": query, "search_engine": search_engine}
        if count is not None:
            body["count"] = count
        if search_domain_filter is not None:
            body["search_domain_filter"] = search_domain_filter
        if search_recency_filter is not None:
            body["search_recency_filter"] = search_recency_filter
        if content_size is not None:
            body["content_size"] = content_size
        body.update(kwargs)

        return await self._media_json(
            sdk_call=lambda: self._sdk_client.web_search.web_search(**body),
            path="/web_search",
            json=body,
        )

    # =========================================================================
    # Response extraction
    # =========================================================================

    def extract_response_content(self, response: dict[str, Any]) -> str:
        """Extract the final text content from a non-streaming response."""
        try:
            choices = response.get("choices", [])
            if not choices:
                return ""
            return choices[0].get("message", {}).get("content") or ""
        except (KeyError, IndexError, TypeError) as e:
            logger.warning("Failed to extract Z.ai content: %s", e)
            return ""

    def extract_delta_content(self, chunk: dict[str, Any]) -> str:
        """Extract text delta from a streaming chunk."""
        try:
            choices = chunk.get("choices", [])
            if not choices:
                return ""
            return choices[0].get("delta", {}).get("content") or ""
        except (KeyError, IndexError, TypeError):
            return ""

    def extract_reasoning_content(self, response: dict[str, Any]) -> str | None:
        """Extract thinking/reasoning content from a non-streaming response.

        This is the ``reasoning_content`` field GLM returns when thinking mode
        is enabled.
        """
        try:
            choices = response.get("choices", [])
            if not choices:
                return None
            return choices[0].get("message", {}).get("reasoning_content")
        except (KeyError, IndexError, TypeError):
            return None

    def extract_delta_reasoning_content(self, chunk: dict[str, Any]) -> str | None:
        """Extract reasoning delta from a streaming chunk."""
        try:
            choices = chunk.get("choices", [])
            if not choices:
                return None
            return choices[0].get("delta", {}).get("reasoning_content")
        except (KeyError, IndexError, TypeError):
            return None

    def extract_tool_calls(self, response: dict[str, Any]) -> list[ToolCall]:
        """Extract tool calls from a Z.ai response."""
        out: list[ToolCall] = []
        try:
            choices = response.get("choices", [])
            if not choices:
                return out
            raw_calls = choices[0].get("message", {}).get("tool_calls")
            if not raw_calls:
                return out
            for tc in raw_calls:
                tc_type = tc.get("type", "function")
                if tc_type == "function":
                    func = tc.get("function", {})
                    args_str = func.get("arguments", "{}")
                    try:
                        args_dict = json.loads(args_str)
                    except (json.JSONDecodeError, TypeError):
                        args_dict = {"_raw": args_str}
                    out.append(
                        ToolCall(
                            id=tc.get("id", ""),
                            name=func.get("name", ""),
                            arguments=args_dict,
                        )
                    )
        except (KeyError, IndexError, TypeError) as e:
            logger.warning("Failed to extract Z.ai tool calls: %s", e)
        return out

    def extract_usage_details(self, response: dict[str, Any]) -> dict[str, Any]:
        """Extract extended usage details including cache and reasoning tokens.

        Z.ai usage mirrors the OpenAI schema and adds:
        - ``prompt_tokens_details.cached_tokens`` — tokens served from cache
        - ``completion_tokens_details.reasoning_tokens`` — thinking tokens
        """
        usage = response.get("usage", {})
        if not usage:
            return {}

        result: dict[str, Any] = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }

        pd = usage.get("prompt_tokens_details") or {}
        if pd:
            result["cached_tokens"] = pd.get("cached_tokens")

        cd = usage.get("completion_tokens_details") or {}
        if cd:
            result["reasoning_tokens"] = cd.get("reasoning_tokens")

        return result

    def extract_finish_reason(self, response: dict[str, Any]) -> str | None:
        """Extract the finish reason from a non-streaming response.

        GLM finish reasons: ``stop``, ``length``, ``tool_calls``,
        ``sensitive``, ``network_error``.
        """
        try:
            choices = response.get("choices", [])
            if not choices:
                return None
            return choices[0].get("finish_reason")
        except (KeyError, IndexError, TypeError):
            return None

    # =========================================================================
    # Token counting
    # =========================================================================

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """Count tokens using tiktoken cl100k_base (approximate for GLM)."""
        if not text:
            return 0
        if not self._encoding:
            return _EstimateCounter().count(text)
        return await asyncio.to_thread(lambda: len(self._encoding.encode(text)))

    async def count_message_tokens(self, messages: list[Message], model: str | None = None) -> int:
        """Estimate total tokens for a list of messages.

        Uses tiktoken cl100k_base as a proxy; for precise counts use the
        official GLM tokenizer.
        """
        if not self._encoding:
            counter = _EstimateCounter()
            total = sum(
                counter.count(m.content)
                + counter.count(m.role.value if hasattr(m.role, "value") else str(m.role))
                for m in messages
            )
            return total + len(messages) * 4

        n = 0
        for m in messages:
            n += 3  # message overhead
            try:
                rv = m.role.value if hasattr(m.role, "value") else str(m.role)
                n += len(self._encoding.encode(rv))
                n += len(self._encoding.encode(m.content))
            except Exception:
                rv = m.role.value if hasattr(m.role, "value") else str(m.role)
                n += max(1, int((len(rv) + len(m.content)) * 0.3))
        return n + 3

    # =========================================================================
    # Resource cleanup
    # =========================================================================

    async def close(self) -> None:
        """Close the SDK / OpenAI / media HTTP clients."""
        if self._sdk_client is not None:
            try:
                close_fn = getattr(self._sdk_client, "close", None)
                if callable(close_fn):
                    await asyncio.to_thread(close_fn)
            except Exception as e:
                logger.error("Error closing Z.ai SDK client: %s", e)
            self._sdk_client = None
        if self._client:
            try:
                await self._client.close()
            except Exception as e:
                logger.error("Error closing Z.ai client: %s", e)
        self._client = None
        if self._http is not None:
            try:
                await self._http.aclose()
            except Exception as e:
                logger.error("Error closing Z.ai media client: %s", e)
            self._http = None
        logger.info("ZaiProvider closed.")
