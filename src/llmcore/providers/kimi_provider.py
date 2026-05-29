# src/llmcore/providers/kimi_provider.py
"""Kimi (Moonshot AI) provider implementation for the LLMCore library.

Handles interactions with the Kimi Open Platform (Moonshot AI) using the
OpenAI-compatible Chat Completions endpoint plus several Moonshot-specific
extensions and auxiliary REST endpoints.

Supported features
------------------
- **Chat completion** (text, streaming) via the ``openai`` SDK (AsyncOpenAI)
  pointed at the Moonshot base URL.
- **Thinking mode** — the Moonshot ``thinking`` request field, a *top-level*
  body parameter injected through the SDK's ``extra_body``::

      {"type": "enabled" | "disabled", "keep": "all" | null}

  ``type`` toggles whether the model emits ``reasoning_content`` this turn;
  ``keep`` (``kimi-k2.6`` only) enables *Preserved Thinking* — historical
  ``reasoning_content`` is forwarded to the model across turns.
- **Reasoning content** — the ``reasoning_content`` field on assistant
  messages (both streaming delta and non-streaming message), exposed via
  :meth:`extract_reasoning_content` / :meth:`extract_delta_reasoning_content`.
- **Vision (multimodal input)** — image *and video* understanding through the
  content-parts protocol.  Images/videos are supplied via message metadata
  (``inline_images`` / ``inline_videos``) or pre-built ``content_parts``.
  Sources may be base64 data URIs (``data:image/png;base64,...``) or Moonshot
  file references (``ms://<file_id>``).
- **Structured output** — ``response_format`` of ``{"type": "json_object"}``
  or ``{"type": "json_schema", "json_schema": {...}}`` (MFJS schema).
- **Partial Mode** — prefix-completion via ``"partial": true`` on the last
  assistant message (supplied through ``metadata["partial"]``).
- **Tool calling** — OpenAI-compatible ``tools`` / ``tool_choice`` with
  ``reasoning_content`` preservation for multi-step tool loops under thinking.
- **Accurate token counting** via the Moonshot
  ``POST /v1/tokenizers/estimate-token-count`` endpoint (tiktoken fallback).
- **Dynamic model discovery** via ``GET /v1/models`` which, unlike vanilla
  OpenAI, returns rich capability metadata (``context_length``,
  ``supports_image_in``, ``supports_video_in``, ``supports_reasoning``).
- **Balance check** (``GET /v1/users/me/balance``) and **file upload**
  (``POST /v1/files``) auxiliary helpers for vision file references.

What this provider does *not* do
--------------------------------
Moonshot exposes **no audio/speech endpoints** (no TTS / STT / transcription)
and no image-generation endpoint.  The corresponding :class:`BaseProvider`
methods therefore retain their default ``NotImplementedError`` behaviour.

Transport
---------
Two clients are maintained:

- ``AsyncOpenAI`` for ``/v1/chat/completions`` (SSE parsing, retries, timeout).
- A raw ``httpx.AsyncClient`` for the Moonshot-specific REST endpoints
  (model discovery with extra fields, token estimate, balance, file upload).

References
----------
- https://platform.kimi.ai/docs  (Kimi Open Platform)
- https://platform.kimi.ai/docs/api/chat
- https://platform.kimi.ai/docs/api/list-models
- https://platform.kimi.ai/docs/guide/use-kimi-k2-thinking-model
- https://platform.kimi.ai/docs/guide/use-kimi-vision-model
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator
from typing import Any, Literal

try:
    import httpx

    httpx_available = True
except ImportError:  # pragma: no cover - httpx is a hard dep of openai
    httpx_available = False
    httpx = None  # type: ignore

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

from ..exceptions import ConfigError, ContextLengthError, ProviderError
from ..model_cards.registry import get_model_card_registry
from ..models import Message, ModelDetails, Tool, ToolCall
from ..models import Role as LLMCoreRole
from .base import BaseProvider, ContextPayload

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: International (``.ai``) OpenAI-compatible API endpoint.  NOTE: keys from
#: ``platform.kimi.ai`` and ``platform.kimi.com`` (``api.moonshot.cn``) are
#: **not** interchangeable — mixing them yields a 401.
_BASE_URL_DEFAULT = "https://api.moonshot.ai/v1"

#: Default model when none configured.  ``kimi-k2.6`` is the current flagship
#: multimodal (text+image+video) model with thinking enabled by default.
_DEFAULT_MODEL = "kimi-k2.6"

#: Hardcoded context-length fallback map (used only when dynamic discovery and
#: the model-card registry do not provide a value).
_CONTEXT_LENGTHS: dict[str, int] = {
    "kimi-k2.6": 262_144,
    "kimi-k2.5": 262_144,
    "moonshot-v1-8k": 8_192,
    "moonshot-v1-32k": 32_768,
    "moonshot-v1-128k": 131_072,
    "moonshot-v1-auto": 131_072,
    "moonshot-v1-8k-vision-preview": 8_192,
    "moonshot-v1-32k-vision-preview": 32_768,
    "moonshot-v1-128k-vision-preview": 131_072,
    # Deprecated (retired 2026-05-25) — kept for graceful error reporting.
    "kimi-k2-0905-preview": 262_144,
    "kimi-k2-0711-preview": 131_072,
    "kimi-k2-turbo-preview": 262_144,
    "kimi-k2-thinking": 262_144,
    "kimi-k2-thinking-turbo": 262_144,
}

#: Valid thinking mode types.
ThinkingType = Literal["enabled", "disabled"]

#: Valid ``thinking.keep`` values (Preserved Thinking).
ThinkingKeep = Literal["all"]

#: Sampling parameters that are *fixed* (and rejected when overridden) by the
#: ``kimi-k2.5`` / ``kimi-k2.6`` model families.  They are silently stripped for
#: those models to avoid ``invalid_request_error``.
_FIXED_SAMPLING_KEYS = frozenset(
    {"temperature", "top_p", "n", "presence_penalty", "frequency_penalty"}
)


def _model_family(model: str) -> str:
    """Return a coarse family label for capability gating.

    Args:
        model: The model identifier.

    Returns:
        One of ``"k2.6"``, ``"k2.5"``, ``"k2-thinking"``, ``"k2"``,
        ``"moonshot-v1"``, or ``"unknown"``.
    """
    m = model.lower()
    if m.startswith("kimi-k2.6"):
        return "k2.6"
    if m.startswith("kimi-k2.5"):
        return "k2.5"
    if m.startswith("kimi-k2-thinking"):
        return "k2-thinking"
    if m.startswith("kimi-k2"):
        return "k2"
    if m.startswith("moonshot-v1"):
        return "moonshot-v1"
    return "unknown"


def _model_supports_thinking_param(model: str) -> bool:
    """Whether the ``thinking`` request field is accepted by *model*.

    Only the ``kimi-k2.5`` and ``kimi-k2.6`` families accept the ``thinking``
    parameter.  ``kimi-k2-thinking*`` always reasons (no toggle) and the
    ``moonshot-v1`` series has no thinking mode.
    """
    return _model_family(model) in ("k2.5", "k2.6")


def _model_supports_thinking_keep(model: str) -> bool:
    """Whether the ``thinking.keep`` (Preserved Thinking) field is accepted.

    Per Moonshot docs, ``keep`` is a ``kimi-k2.6``-only extension.
    """
    return _model_family(model) == "k2.6"


def _model_has_fixed_sampling(model: str) -> bool:
    """Whether *model* rejects custom sampling parameters.

    ``kimi-k2.5`` / ``kimi-k2.6`` use fixed temperature/top_p/n/penalties and
    return ``invalid_request_error`` if these are supplied with non-default
    values.
    """
    return _model_family(model) in ("k2.5", "k2.6", "k2-thinking")


class KimiProvider(BaseProvider):
    """First-class Kimi (Moonshot AI) provider with native thinking + vision.

    Configuration keys (under ``[providers.kimi]``):

    - ``api_key`` / ``api_key_env_var`` — API credential (env default:
      ``MOONSHOT_API_KEY``).
    - ``base_url`` — Override the API URL (default:
      ``https://api.moonshot.ai/v1``).  Use ``https://api.moonshot.cn/v1`` for
      the China platform (keys are **not** interchangeable).
    - ``default_model`` — Default model ID (default: ``kimi-k2.6``).
    - ``timeout`` — HTTP request timeout in seconds (default: 300).
    - ``thinking`` — Default thinking mode: ``"enabled"`` or ``"disabled"``
      (default: ``"enabled"``).  Applied only to thinking-capable models.
    - ``thinking_keep`` — Default Preserved-Thinking setting: ``"all"`` or
      ``null`` (default: ``null``).  Applied only to ``kimi-k2.6``.
    """

    default_model: str
    _client: AsyncOpenAI | None
    _http: Any  # httpx.AsyncClient | None
    _encoding: Any  # tiktoken.Encoding | None
    _default_thinking: ThinkingType
    _default_keep: ThinkingKeep | None
    _base_url: str
    _api_key: str

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False):
        """Initialize the Kimi provider.

        Args:
            config: Provider-specific configuration dict from
                ``[providers.kimi]``.
            log_raw_payloads: Whether to log raw request/response payloads.

        Raises:
            ConfigError: If the ``openai`` SDK is missing or no API key found.
        """
        super().__init__(config, log_raw_payloads)

        if not openai_available:
            raise ConfigError(
                "The 'openai' package is required for the Kimi provider. "
                "Install with: pip install openai"
            )

        # --- API key ---
        api_key = config.get("api_key") or os.environ.get(
            config.get("api_key_env_var", "MOONSHOT_API_KEY")
        )
        if not api_key:
            api_key = os.environ.get("MOONSHOT_API_KEY")
        if not api_key:
            raise ConfigError(
                "Kimi (Moonshot) API key not found. Set MOONSHOT_API_KEY or "
                "configure providers.kimi.api_key / api_key_env_var."
            )
        self._api_key = api_key

        # --- Model / timeout ---
        self.default_model = config.get("default_model", _DEFAULT_MODEL)
        timeout = config.get("timeout", 300)

        # --- Thinking-mode defaults ---
        thinking_raw = config.get("thinking", "enabled")
        if thinking_raw not in ("enabled", "disabled"):
            logger.warning("Invalid thinking value '%s'; defaulting to 'enabled'.", thinking_raw)
            thinking_raw = "enabled"
        self._default_thinking = thinking_raw  # type: ignore[assignment]

        keep_raw = config.get("thinking_keep", None)
        if keep_raw in (None, "", "null", "none", "None"):
            self._default_keep = None
        elif keep_raw == "all":
            self._default_keep = "all"
        else:
            logger.warning("Invalid thinking_keep value '%s'; defaulting to null.", keep_raw)
            self._default_keep = None

        # --- Base URL (normalize to ensure a single trailing /v1) ---
        base_url = config.get("base_url", _BASE_URL_DEFAULT)
        self._base_url = base_url.rstrip("/")

        # --- Clients ---
        try:
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=self._base_url,
                timeout=timeout,
            )
            # Raw httpx client for Moonshot-specific endpoints.  The origin is
            # the base_url without the trailing ``/v1`` so we can address
            # ``/v1/...`` paths uniformly.
            self._http = (
                httpx.AsyncClient(
                    base_url=self._origin(),
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=timeout,
                )
                if httpx_available
                else None
            )
            logger.debug(
                "Kimi clients initialized (base_url=%s, default_model=%s, "
                "thinking=%s, keep=%s).",
                self._base_url,
                self.default_model,
                self._default_thinking,
                self._default_keep,
            )
        except Exception as e:
            raise ConfigError(f"Kimi client initialization failed: {e}")

        # --- Tokenizer fallback (approximate; estimate endpoint is preferred) ---
        self._encoding = None
        if tiktoken_available:
            try:
                self._encoding = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                logger.warning("Failed to load tiktoken for Kimi: %s", e)

    def _origin(self) -> str:
        """Return the scheme+host origin (base_url with a trailing ``/v1``
        segment stripped), suitable as an httpx ``base_url`` for ``/v1/*`` paths.
        """
        b = self._base_url
        if b.endswith("/v1"):
            return b[: -len("/v1")]
        return b

    # =========================================================================
    # BaseProvider interface
    # =========================================================================

    def get_name(self) -> str:
        return self._provider_instance_name or "kimi"

    async def get_models_details(self) -> list[ModelDetails]:
        """Discover available models via ``GET /v1/models``.

        Moonshot's model list response carries richer capability metadata than
        vanilla OpenAI: ``context_length``, ``supports_image_in``,
        ``supports_video_in`` and ``supports_reasoning``.  These are mapped onto
        :class:`ModelDetails` (with video capability recorded in ``metadata``).

        Falls back to the OpenAI SDK ``models.list()`` if the raw HTTP call
        is unavailable, and consults the model-card registry to fill gaps.
        """
        provider = self.get_name()
        try:
            registry = get_model_card_registry()
        except Exception:
            registry = None

        raw_models: list[dict[str, Any]] = []

        # Prefer the raw HTTP call to capture the extra capability fields.
        if self._http is not None:
            try:
                resp = await self._http.get("/v1/models")
                resp.raise_for_status()
                raw_models = resp.json().get("data", []) or []
            except Exception as e:
                logger.debug("Raw Kimi /v1/models failed (%s); falling back to SDK.", e)

        if not raw_models and self._client is not None:
            try:
                sdk_resp = await self._client.models.list()
                for m in sdk_resp.data:
                    entry = {"id": m.id, "owned_by": getattr(m, "owned_by", None),
                             "created": getattr(m, "created", None)}
                    # OpenAI SDK keeps unknown fields in ``model_extra``.
                    extra = getattr(m, "model_extra", None)
                    if isinstance(extra, dict):
                        entry.update(extra)
                    raw_models.append(entry)
            except OpenAIError as e:
                raise ProviderError(self.get_name(), f"Failed to fetch models: {e}")

        result: list[ModelDetails] = []
        for m in raw_models:
            mid = m.get("id", "")
            if not mid:
                continue

            ctx = m.get("context_length") or self.get_max_context_length(mid)
            supports_vision = bool(m.get("supports_image_in", False))
            supports_video = bool(m.get("supports_video_in", False))
            supports_reasoning = bool(m.get("supports_reasoning", False))
            supports_tools = True  # All current Kimi/Moonshot models support tools.

            # Registry overlay (authoritative when present).
            if registry:
                try:
                    card = registry.get(provider, mid)
                except Exception:
                    card = None
                if card and card.capabilities:
                    caps = card.capabilities
                    supports_tools = caps.tool_use or caps.function_calling
                    supports_vision = supports_vision or caps.vision
                    supports_video = supports_video or caps.video_input
                    supports_reasoning = supports_reasoning or caps.reasoning

            result.append(
                ModelDetails(
                    id=mid,
                    context_length=int(ctx),
                    supports_streaming=True,
                    supports_tools=supports_tools,
                    supports_vision=supports_vision,
                    supports_reasoning=supports_reasoning,
                    provider_name=provider,
                    metadata={
                        "owned_by": m.get("owned_by"),
                        "created": m.get("created"),
                        "supports_video_in": supports_video,
                        "supports_thinking": _model_supports_thinking_param(mid),
                    },
                )
            )
        return result

    def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
        """Return the set of supported API parameters for Kimi models."""
        return {
            "temperature": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "top_p": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "n": {"type": "integer", "minimum": 1, "maximum": 5},
            "max_tokens": {"type": "integer", "minimum": 1},
            "max_completion_tokens": {"type": "integer", "minimum": 1},
            "presence_penalty": {"type": "number", "minimum": -2.0, "maximum": 2.0},
            "frequency_penalty": {"type": "number", "minimum": -2.0, "maximum": 2.0},
            "stop": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
            "response_format": {"type": "object"},
            "stream_options": {"type": "object"},
            "prompt_cache_key": {"type": "string"},
            "safety_identifier": {"type": "string"},
            # Kimi/Moonshot-specific
            "thinking": {
                "type": "object",
                "description": (
                    'Thinking-mode control (kimi-k2.5/k2.6): '
                    '{"type": "enabled"|"disabled", "keep": "all"|null}.'
                ),
            },
            "thinking_keep": {
                "type": "string",
                "enum": ["all"],
                "description": "Preserved Thinking (kimi-k2.6): keep historical reasoning_content.",
            },
        }

    def get_max_context_length(self, model: str | None = None) -> int:
        """Return the maximum context length for a Kimi/Moonshot model."""
        model_name = model or self.default_model
        limit = _CONTEXT_LENGTHS.get(model_name)
        if limit is not None:
            return limit

        # Try the model-card registry.
        try:
            registry = get_model_card_registry()
            card = registry.get(self.get_name(), model_name)
            if card is not None:
                return card.get_context_length()
        except Exception:
            pass

        # Family-based heuristic.
        fam = _model_family(model_name)
        if fam in ("k2.5", "k2.6", "k2", "k2-thinking"):
            return 262_144
        if fam == "moonshot-v1":
            return 131_072

        logger.warning(
            "Unknown context length for Kimi model '%s'. Falling back to 131072.",
            model_name,
        )
        return 131_072

    # =========================================================================
    # Chat completion
    # =========================================================================

    def _resolve_thinking(
        self, model_name: str, kwargs: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Resolve the ``thinking`` request object from kwargs + defaults.

        Pops ``thinking`` and ``thinking_keep`` from *kwargs* so they are not
        forwarded verbatim to the OpenAI SDK.

        Args:
            model_name: The target model (capability gating).
            kwargs: Mutable request kwargs (modified in place).

        Returns:
            A ``{"type": ..., "keep"?: ...}`` dict suitable for injection into
            the request body, or ``None`` when the model does not support the
            ``thinking`` parameter.
        """
        thinking = kwargs.pop("thinking", "__unset__")
        keep = kwargs.pop("thinking_keep", "__unset__")

        if not _model_supports_thinking_param(model_name):
            # Silently ignore for non-thinking-capable models.
            if thinking != "__unset__" or keep != "__unset__":
                logger.debug(
                    "Model '%s' does not accept the 'thinking' parameter; ignoring.",
                    model_name,
                )
            return None

        # Resolve type.
        if thinking == "__unset__" or thinking is None:
            t_type: str = self._default_thinking
            t_keep: str | None = self._default_keep
        elif isinstance(thinking, dict):
            t_type = thinking.get("type", self._default_thinking)
            t_keep = thinking.get("keep", self._default_keep)
        elif isinstance(thinking, bool):
            t_type = "enabled" if thinking else "disabled"
            t_keep = self._default_keep
        elif isinstance(thinking, str):
            t_type = thinking
            t_keep = self._default_keep
        else:
            t_type = self._default_thinking
            t_keep = self._default_keep

        if t_type not in ("enabled", "disabled"):
            logger.warning("Invalid thinking.type '%s'; using 'enabled'.", t_type)
            t_type = "enabled"

        # Explicit per-request keep override.
        if keep != "__unset__":
            t_keep = keep

        obj: dict[str, Any] = {"type": t_type}

        # ``keep`` is a kimi-k2.6-only extension and only meaningful with
        # type=enabled.
        if (
            t_keep == "all"
            and t_type == "enabled"
            and _model_supports_thinking_keep(model_name)
        ):
            obj["keep"] = "all"
        elif t_keep == "all" and not _model_supports_thinking_keep(model_name):
            logger.debug(
                "thinking.keep='all' ignored for model '%s' (kimi-k2.6 only).",
                model_name,
            )

        return obj

    @staticmethod
    def _normalize_media_part(item: Any, kind: str) -> dict[str, Any] | None:
        """Normalize an inline image/video entry into a content-part dict.

        Accepted *item* forms:
            - ``str`` — a base64 data URI (``data:image/png;base64,...``) or a
              Moonshot file reference (``ms://<file_id>``).
            - ``dict`` with a ``url`` key (optionally pre-shaped as
              ``{"image_url"|"video_url": {...}}`` — passed through).

        Args:
            item: The raw inline entry.
            kind: ``"image_url"`` or ``"video_url"``.

        Returns:
            A content-part dict, or ``None`` if the item cannot be normalized.
        """
        if isinstance(item, str):
            return {"type": kind, kind: {"url": item}}
        if isinstance(item, dict):
            # Already a full content part (passthrough).
            if item.get("type") in ("image_url", "video_url"):
                return item
            # ``{"url": "..."}`` or ``{"image_url": {...}}`` shapes.
            if kind in item:
                inner = item[kind]
                if isinstance(inner, str):
                    inner = {"url": inner}
                return {"type": kind, kind: inner}
            if "url" in item:
                return {"type": kind, kind: {"url": item["url"]}}
        logger.warning("Skipping unrecognized inline %s entry: %r", kind, item)
        return None

    def _build_message_payload(self, msg: Message) -> dict[str, Any]:
        """Build a Moonshot-format message dict from an llmcore Message.

        Handles:
        - Multimodal user content (``inline_images`` / ``inline_videos`` /
          ``content_parts`` from ``metadata``).
        - Assistant ``tool_calls`` + ``reasoning_content`` preservation
          (required for multi-step thinking + tool loops, and for Preserved
          Thinking with ``thinking.keep="all"``).
        - Partial Mode (``metadata["partial"] = True`` on the last assistant
          message).
        - Tool result messages (``tool_call_id``).
        """
        role_str = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        metadata = msg.metadata or {}

        # --- Resolve content (string or multimodal array) ---
        content: Any = msg.content

        if "content_parts" in metadata:
            # Caller supplied fully-formed Moonshot content parts.
            content = metadata["content_parts"]
        else:
            inline_images = metadata.get("inline_images") or []
            inline_videos = metadata.get("inline_videos") or []
            if inline_images or inline_videos:
                parts: list[dict[str, Any]] = []
                # Text first or last is acceptable; Moonshot docs put media
                # first then the text instruction.  Preserve that ordering.
                for img in inline_images:
                    p = self._normalize_media_part(img, "image_url")
                    if p:
                        parts.append(p)
                for vid in inline_videos:
                    p = self._normalize_media_part(vid, "video_url")
                    if p:
                        parts.append(p)
                if msg.content:
                    parts.append({"type": "text", "text": msg.content})
                content = parts

        msg_dict: dict[str, Any] = {"role": role_str, "content": content}

        # Tool message needs tool_call_id.
        if msg.role == LLMCoreRole.TOOL and msg.tool_call_id:
            msg_dict["tool_call_id"] = msg.tool_call_id

        # Assistant message may carry tool_calls, reasoning_content, partial.
        if role_str == "assistant":
            if "tool_calls" in metadata:
                msg_dict["tool_calls"] = metadata["tool_calls"]
                if not msg.content:
                    msg_dict["content"] = None
            if "reasoning_content" in metadata:
                # Preserved Thinking: forward historical reasoning as-is.
                msg_dict["reasoning_content"] = metadata["reasoning_content"]
            if metadata.get("partial"):
                msg_dict["partial"] = True

        # Optional name field (also used by Partial Mode role-play personas).
        name = metadata.get("name")
        if name:
            msg_dict["name"] = name

        return msg_dict

    async def chat_completion(
        self,
        context: ContextPayload,
        model: str | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Perform a chat completion against the Kimi (Moonshot) API.

        Extends the OpenAI-compatible flow with:

        - Automatic ``thinking`` injection (capability-gated by model family).
        - Fixed-sampling-parameter stripping for ``kimi-k2.5`` / ``kimi-k2.6``.
        - ``reasoning_content`` extraction in both streaming and non-streaming
          modes (see :meth:`extract_reasoning_content`).
        - Multimodal (image + video) content assembly.

        Additional kwargs accepted:

        - ``thinking``: ``dict | str | bool | None`` — override thinking mode.
        - ``thinking_keep``: ``"all" | None`` — Preserved Thinking (k2.6).
        - ``response_format``: structured-output control.
        - ``prompt_cache_key`` / ``safety_identifier``: passthrough.

        Returns:
            A response dict (``stream=False``) or an async generator of chunk
            dicts (``stream=True``).
        """
        if not self._client:
            raise ProviderError(self.get_name(), "Client not initialized.")

        model_name = model or self.default_model

        # Validate kwargs (reject genuinely unknown parameters early).
        supported = self.get_supported_parameters(model_name)
        for key in list(kwargs.keys()):
            if key not in supported:
                raise ValueError(f"Unsupported parameter '{key}' for Kimi provider.")

        # Validate context.
        if not (isinstance(context, list) and all(isinstance(m, Message) for m in context)):
            raise ProviderError(self.get_name(), "Context must be list[Message].")

        messages_payload = [self._build_message_payload(m) for m in context]
        if not messages_payload:
            raise ProviderError(self.get_name(), "No valid messages.")

        # --- Tool schema ---
        tools_payload = None
        if tools:
            tools_payload = [{"type": "function", "function": t.model_dump()} for t in tools]

        # --- Build API kwargs ---
        api_kwargs: dict[str, Any] = {}
        extra_body: dict[str, Any] = {}

        # Thinking mode (top-level body field via extra_body).
        thinking_obj = self._resolve_thinking(model_name, kwargs)
        if thinking_obj is not None:
            extra_body["thinking"] = thinking_obj

        # Strip fixed sampling params for k2.5/k2.6 families.
        strip_sampling = _model_has_fixed_sampling(model_name)
        for key, val in kwargs.items():
            if strip_sampling and key in _FIXED_SAMPLING_KEYS:
                logger.debug(
                    "Skipping '%s' (fixed/ignored for model '%s').", key, model_name
                )
                continue
            # prompt_cache_key / safety_identifier are Moonshot extensions; the
            # OpenAI SDK does not type them, so route via extra_body.
            if key in ("prompt_cache_key", "safety_identifier"):
                extra_body[key] = val
            else:
                api_kwargs[key] = val

        if tools_payload:
            api_kwargs["tools"] = tools_payload
        if tool_choice:
            api_kwargs["tool_choice"] = tool_choice

        # Request usage in the final stream chunk.
        if stream:
            api_kwargs.setdefault("stream_options", {"include_usage": True})

        if extra_body:
            api_kwargs["extra_body"] = extra_body

        # --- Logging ---
        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "RAW KIMI REQUEST: %s",
                json.dumps(
                    {
                        "model": model_name,
                        "messages": messages_payload,
                        "stream": stream,
                        **api_kwargs,
                    },
                    indent=2,
                    default=str,
                ),
            )

        # --- API call ---
        try:
            resp = await self._client.chat.completions.create(
                model=model_name,
                messages=messages_payload,
                stream=stream,
                **api_kwargs,
            )  # type: ignore[arg-type]

            if stream:

                async def stream_wrapper() -> AsyncGenerator[dict[str, Any], None]:
                    async for chunk in resp:  # type: ignore[union-attr]
                        chunk_dict = chunk.model_dump(exclude_none=True)
                        if self.log_raw_payloads_enabled and logger.isEnabledFor(
                            logging.DEBUG
                        ):
                            logger.debug(
                                "RAW KIMI STREAM CHUNK: %s",
                                json.dumps(chunk_dict, default=str),
                            )
                        yield chunk_dict

                return stream_wrapper()

            response_dict = resp.model_dump(exclude_none=True)  # type: ignore[union-attr]
            if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "RAW KIMI RESPONSE: %s",
                    json.dumps(response_dict, indent=2, default=str),
                )
            return response_dict

        except OpenAIAPIStatusError as e:
            status = e.status_code
            msg = str(e)
            logger.error("Kimi status error (%d): %s", status, msg, exc_info=True)

            low = msg.lower()
            if status == 400 and (
                "token limit" in low
                or "context" in low
                or "token length too long" in low
            ):
                raise ContextLengthError(
                    model_name=model_name,
                    limit=self.get_max_context_length(model_name),
                    actual=0,
                    message=msg,
                )
            if status == 404 and ("not found" in low or "model" in low):
                raise ProviderError(
                    self.get_name(),
                    f"Model '{model_name}' not found on Kimi (or no access). "
                    f"Current models: kimi-k2.6, kimi-k2.5, moonshot-v1-8k/32k/128k "
                    f"(+ -vision-preview). The kimi-k2-* preview series retired "
                    f"2026-05-25. Original error: {msg}",
                )
            if status == 429 and ("quota" in low or "balance" in low or "suspended" in low):
                raise ProviderError(
                    self.get_name(),
                    f"Kimi quota/balance issue. Check balance at "
                    f"https://platform.kimi.ai/console. Error: {msg}",
                )
            if status == 401:
                raise ProviderError(
                    self.get_name(),
                    f"Kimi authentication failed. Verify MOONSHOT_API_KEY and that "
                    f"the key matches the platform of base_url "
                    f"({self._base_url}). Keys from platform.kimi.ai and "
                    f"platform.kimi.com are not interchangeable. Error: {msg}",
                )
            raise ProviderError(self.get_name(), f"API Error ({status}): {msg}")
        except OpenAIAPITimeoutError as e:
            raise ProviderError(self.get_name(), f"Timeout: {e}")
        except OpenAIAPIConnectionError as e:
            raise ProviderError(self.get_name(), f"Connection error: {e}")
        except OpenAIAPIError as e:
            raise ProviderError(self.get_name(), f"API Error: {e}")
        except OpenAIError as e:
            raise ProviderError(self.get_name(), f"Error: {e}")
        except Exception as e:
            logger.error("Unexpected Kimi error: %s", e, exc_info=True)
            raise ProviderError(self.get_name(), f"Unexpected error: {e}")

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
            logger.warning("Failed to extract Kimi content: %s", e)
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
        """Extract the ``reasoning_content`` from a non-streaming response.

        Present when thinking mode is enabled.  Returns ``None`` otherwise.
        """
        try:
            choices = response.get("choices", [])
            if not choices:
                return None
            return choices[0].get("message", {}).get("reasoning_content")
        except (KeyError, IndexError, TypeError):
            return None

    def extract_delta_reasoning_content(self, chunk: dict[str, Any]) -> str | None:
        """Extract the ``reasoning_content`` delta from a streaming chunk.

        Per Moonshot docs, ``reasoning_content`` deltas always precede the
        first ``content`` delta in a thinking-enabled stream.
        """
        try:
            choices = chunk.get("choices", [])
            if not choices:
                return None
            return choices[0].get("delta", {}).get("reasoning_content")
        except (KeyError, IndexError, TypeError):
            return None

    def extract_tool_calls(self, response: dict[str, Any]) -> list[ToolCall]:
        """Extract tool calls from a Kimi response."""
        out: list[ToolCall] = []
        try:
            choices = response.get("choices", [])
            if not choices:
                return out
            raw_calls = choices[0].get("message", {}).get("tool_calls")
            if not raw_calls:
                return out
            for tc in raw_calls:
                if tc.get("type", "function") != "function":
                    continue
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
            logger.warning("Failed to extract Kimi tool calls: %s", e)
        return out

    def extract_usage_details(self, response: dict[str, Any]) -> dict[str, Any]:
        """Extract usage details including Moonshot's ``cached_tokens``.

        Moonshot reports ``cached_tokens`` in the usage object (context-cache
        hits), useful for cost accounting with prompt caching.
        """
        usage = response.get("usage", {})
        if not usage:
            return {}
        result: dict[str, Any] = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "cached_tokens": usage.get("cached_tokens"),
        }
        cd = usage.get("completion_tokens_details", {})
        if cd:
            result["reasoning_tokens"] = cd.get("reasoning_tokens")
        return result

    def extract_finish_reason(self, response: dict[str, Any]) -> str | None:
        """Extract finish reason (``stop`` | ``length`` | ``tool_calls``)."""
        try:
            choices = response.get("choices", [])
            if not choices:
                return None
            return choices[0].get("finish_reason")
        except (KeyError, IndexError, TypeError):
            return None

    # =========================================================================
    # Token counting (real Moonshot estimate endpoint)
    # =========================================================================

    async def estimate_tokens(
        self, messages_payload: list[dict[str, Any]], model: str | None = None
    ) -> int | None:
        """Call ``POST /v1/tokenizers/estimate-token-count`` for an exact count.

        Args:
            messages_payload: Moonshot-format message dicts.
            model: Model id (defaults to the provider default).

        Returns:
            The estimated total token count, or ``None`` on failure (caller
            should fall back to a heuristic count).
        """
        if self._http is None:
            return None
        model_name = model or self.default_model
        try:
            resp = await self._http.post(
                "/v1/tokenizers/estimate-token-count",
                json={"model": model_name, "messages": messages_payload},
            )
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                logger.debug("Kimi estimate error: %s", data["error"])
                return None
            return int(data.get("data", {}).get("total_tokens"))
        except Exception as e:
            logger.debug("Kimi estimate-token-count failed: %s", e)
            return None

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """Count tokens for a single text string.

        Prefers the Moonshot estimate endpoint (exact); falls back to tiktoken
        ``cl100k_base`` and finally a character heuristic.
        """
        if not text:
            return 0
        exact = await self.estimate_tokens(
            [{"role": "user", "content": text}], model=model
        )
        if exact is not None:
            return max(0, exact)
        if self._encoding:
            return await asyncio.to_thread(lambda: len(self._encoding.encode(text)))
        return max(1, int(len(text) * 0.3))

    async def count_message_tokens(
        self, messages: list[Message], model: str | None = None
    ) -> int:
        """Count total tokens for a list of messages.

        Uses the Moonshot estimate endpoint over the fully-built payload
        (accounting for role overhead and multimodal parts); falls back to a
        tiktoken/heuristic per-message sum.
        """
        payload = [self._build_message_payload(m) for m in messages]
        exact = await self.estimate_tokens(payload, model=model)
        if exact is not None:
            return max(0, exact)

        # Heuristic fallback (text content only; multimodal parts are
        # under-counted, which the estimate endpoint avoids).
        if self._encoding:
            n = 0
            for m in messages:
                n += 3
                rv = m.role.value if hasattr(m.role, "value") else str(m.role)
                try:
                    n += len(self._encoding.encode(rv))
                    n += len(self._encoding.encode(m.content or ""))
                except Exception:
                    n += max(1, int((len(rv) + len(m.content or "")) * 0.3))
            return n + 3

        total = sum(
            len(m.content or "")
            + len(m.role.value if hasattr(m.role, "value") else str(m.role))
            for m in messages
        )
        return max(1, int(total * 0.3) + len(messages) * 4)

    # =========================================================================
    # Auxiliary Moonshot REST helpers
    # =========================================================================

    async def check_balance(self) -> dict[str, Any]:
        """Return account balance via ``GET /v1/users/me/balance``.

        Returns:
            The ``data`` object: ``available_balance`` / ``voucher_balance`` /
            ``cash_balance`` (USD).

        Raises:
            ProviderError: If the HTTP client is unavailable or the call fails.
        """
        if self._http is None:
            raise ProviderError(self.get_name(), "httpx client unavailable for balance check.")
        try:
            resp = await self._http.get("/v1/users/me/balance")
            resp.raise_for_status()
            return resp.json().get("data", {})
        except Exception as e:
            raise ProviderError(self.get_name(), f"Balance check failed: {e}")

    async def upload_file(
        self, data: bytes, filename: str, purpose: str = "file-extract"
    ) -> str:
        """Upload a file to Moonshot storage and return its file id.

        For vision, reference the returned id as ``ms://<file_id>`` in an
        ``inline_images`` / ``inline_videos`` entry.

        Args:
            data: Raw file bytes.
            filename: Original file name (used for content-type inference).
            purpose: Upload purpose. ``"file-extract"`` for document QA;
                ``"video"`` / image purposes for vision uploads.

        Returns:
            The uploaded file id.

        Raises:
            ProviderError: On upload failure.
        """
        if self._http is None:
            raise ProviderError(self.get_name(), "httpx client unavailable for file upload.")
        try:
            resp = await self._http.post(
                "/v1/files",
                files={"file": (filename, data)},
                data={"purpose": purpose},
            )
            resp.raise_for_status()
            return resp.json().get("id", "")
        except Exception as e:
            raise ProviderError(self.get_name(), f"File upload failed: {e}")

    # =========================================================================
    # Resource cleanup
    # =========================================================================

    async def close(self) -> None:
        """Close the OpenAI SDK client and the raw httpx client."""
        errors: list[Exception] = []
        if self._client:
            try:
                await self._client.close()
            except Exception as e:
                errors.append(e)
                logger.error("Error closing Kimi OpenAI client: %s", e)
        if self._http is not None:
            try:
                await self._http.aclose()
            except Exception as e:
                errors.append(e)
                logger.error("Error closing Kimi httpx client: %s", e)
        self._client = None
        self._http = None
        if not errors:
            logger.info("KimiProvider closed.")
