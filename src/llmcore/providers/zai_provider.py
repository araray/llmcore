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

Transport: Uses the ``openai`` Python SDK (AsyncOpenAI) pointed at the Z.ai
base URL.  The SDK handles SSE parsing, keep-alive tolerance, and retry/timeout
plumbing.

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
    _client: AsyncOpenAI | None
    _encoding: Any  # tiktoken.Encoding | None
    _default_thinking: ThinkingType
    _default_reasoning_effort: str

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False):
        """Initialize the Z.ai provider.

        Args:
            config: Provider-specific configuration dict from ``[providers.zai]``.
            log_raw_payloads: Whether to log raw request/response payloads.

        Raises:
            ConfigError: If the ``openai`` SDK is missing or no API key found.
        """
        super().__init__(config, log_raw_payloads)

        if not openai_available:
            raise ConfigError(
                "The 'openai' package is required for the Z.ai provider. "
                "Install with: pip install openai"
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
        timeout = config.get("timeout", 300)

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

        # --- Client ---
        region = str(config.get("region", "overseas")).lower()
        default_base = _BASE_URL_CHINA if region == "china" else _BASE_URL_OVERSEAS
        base_url = config.get("base_url", default_base)
        try:
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
            )
            logger.debug(
                "Z.ai client initialized (base_url=%s, default_model=%s, "
                "thinking=%s, reasoning_effort=%s).",
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
    # BaseProvider interface
    # =========================================================================

    def get_name(self) -> str:
        return self._provider_instance_name or "zai"

    async def get_models_details(self) -> list[ModelDetails]:
        """Discover available models via the ``GET /models`` endpoint.

        Z.ai exposes an OpenAI-compatible model listing.  When the endpoint is
        unavailable the static :data:`_CONTEXT_LENGTHS` table is used as a
        fallback so capability discovery always returns the known GLM models.
        """
        if not self._client:
            raise ProviderError(self.get_name(), "Client not initialized.")

        provider = self.get_name()
        try:
            registry = get_model_card_registry()
        except Exception:
            registry = None

        try:
            resp = await self._client.models.list()
            result: list[ModelDetails] = []
            for m in resp.data:
                mid = m.id
                result.append(
                    self._build_model_details(
                        mid,
                        registry,
                        owned_by=getattr(m, "owned_by", None),
                        created=getattr(m, "created", None),
                    )
                )
            if result:
                return result
        except OpenAIError as e:
            logger.warning(
                "Z.ai model listing failed (%s); falling back to static table.", e
            )

        # Fallback: synthesize details from the static context-length table.
        return [
            self._build_model_details(mid, registry)
            for mid in _CONTEXT_LENGTHS
            if not mid.startswith("embedding")
        ]

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

        Extends the standard OpenAI-compatible flow with:

        - Automatic thinking-mode parameter injection.
        - GLM platform extras (``do_sample``, ``request_id``, ``seed`` …)
          routed through ``extra_body``.
        - ``reasoning_content`` extraction in streaming and non-streaming modes.
        - Open-interval clamping of ``temperature`` / ``top_p``.

        Additional kwargs accepted:

        - ``thinking``: ``dict | str | bool | None`` — override thinking mode.
        - ``reasoning_effort``: ``str | None`` — override reasoning effort.
        """
        if not self._client:
            raise ProviderError(self.get_name(), "Client not initialized.")

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

        # --- Build API kwargs ---
        api_kwargs: dict[str, Any] = {}
        extra_body: dict[str, Any] = {}

        # Thinking mode
        thinking_obj, effort = self._resolve_thinking_params(kwargs)
        if thinking_obj:
            extra_body["thinking"] = thinking_obj
        if effort:
            extra_body["reasoning_effort"] = effort

        # Copy remaining kwargs, routing platform extras into extra_body and
        # clamping sampling parameters to the open interval (0, 1).
        for key, val in kwargs.items():
            if key in ("thinking", "reasoning_effort"):
                continue  # already handled
            if key in ("temperature", "top_p"):
                api_kwargs[key] = self._clamp_open_interval(val)
            elif key in self._EXTRA_BODY_KEYS:
                extra_body[key] = val
            else:
                api_kwargs[key] = val

        if extra_body:
            api_kwargs["extra_body"] = extra_body
        if tools_payload:
            api_kwargs["tools"] = tools_payload
        if tool_choice:
            api_kwargs["tool_choice"] = tool_choice

        # Stream options — request usage in final chunk
        if stream:
            api_kwargs.setdefault("stream_options", {"include_usage": True})

        # --- Logging ---
        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "RAW ZAI REQUEST: %s",
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
            )  # type: ignore

            if stream:

                async def stream_wrapper() -> AsyncGenerator[dict[str, Any], None]:
                    async for chunk in resp:  # type: ignore
                        chunk_dict = chunk.model_dump(exclude_none=True)
                        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                "RAW ZAI STREAM CHUNK: %s",
                                json.dumps(chunk_dict, default=str),
                            )
                        yield chunk_dict

                return stream_wrapper()
            else:
                response_dict = resp.model_dump(exclude_none=True)  # type: ignore
                if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "RAW ZAI RESPONSE: %s",
                        json.dumps(response_dict, indent=2, default=str),
                    )
                return response_dict

        except OpenAIAPIStatusError as e:
            status = e.status_code
            msg = str(e)
            logger.error("Z.ai status error (%d): %s", status, msg, exc_info=True)

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
        except OpenAIAPITimeoutError as e:
            raise ProviderError(self.get_name(), f"Timeout: {e}")
        except OpenAIAPIConnectionError as e:
            raise ProviderError(self.get_name(), f"Connection error: {e}")
        except OpenAIAPIError as e:
            raise ProviderError(self.get_name(), f"API Error: {e}")
        except OpenAIError as e:
            raise ProviderError(self.get_name(), f"Error: {e}")
        except Exception as e:
            logger.error("Unexpected Z.ai error: %s", e, exc_info=True)
            raise ProviderError(self.get_name(), f"Unexpected error: {e}")

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
        if not self._client:
            raise ProviderError(self.get_name(), "Client not initialized.")

        embed_model = model or self.default_embedding_model
        api_kwargs: dict[str, Any] = dict(kwargs)
        if dimensions is not None:
            api_kwargs["dimensions"] = dimensions
        if encoding_format is not None:
            api_kwargs["encoding_format"] = encoding_format

        try:
            resp = await self._client.embeddings.create(
                model=embed_model,
                input=input_texts,
                **api_kwargs,
            )
            return resp.model_dump(exclude_none=True)
        except OpenAIAPIStatusError as e:
            raise ProviderError(self.get_name(), f"Embeddings API Error ({e.status_code}): {e}")
        except OpenAIError as e:
            raise ProviderError(self.get_name(), f"Embeddings error: {e}")
        except Exception as e:
            raise ProviderError(self.get_name(), f"Unexpected embeddings error: {e}")

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
        """Close the HTTP client."""
        if self._client:
            try:
                await self._client.close()
            except Exception as e:
                logger.error("Error closing Z.ai client: %s", e)
        self._client = None
        logger.info("ZaiProvider closed.")
