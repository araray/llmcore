# src/llmcore/providers/deepseek_provider.py
"""
DeepSeek API provider implementation for the LLMCore library.

Handles interactions with the DeepSeek API (``deepseek-flash`` = V4.1-Flash,
``deepseek-v4-pro``). Uses the OpenAI-compatible chat completions endpoint with
DeepSeek-specific extensions:

- **Thinking mode**: on by default; ``thinking.type = "enabled" | "disabled"``
  plus ``reasoning_effort = "low" | "high" | "max"``. llmcore's ``effort``
  maps onto them as DeepSeek documents (minimal/low -> low, medium/high/xhigh
  -> high, max -> max, none -> thinking off).
- **Reasoning content**: ``reasoning_content`` on assistant messages. With
  tools, DeepSeek requires it on every earlier assistant turn (a 400
  otherwise); :meth:`assistant_replay_metadata` returns it, streams included.
- **Built-in tools** (``deepseek-flash``): ``native_search=True`` adds
  ``deepseek_web_search``; ``extra_tools`` passes others such as
  ``{"type": "deepseek_read_url"}``; ``safety_check={"level": "high"}``.
- **Cache-aware token accounting**: ``prompt_cache_hit_tokens`` and
  ``prompt_cache_miss_tokens`` in the usage object.
- **Beta endpoint**: Chat Prefix Completion and strict tool schemas via
  ``https://api.deepseek.com/beta``.
- **Responses API**: ``responses_api = "always"`` routes chat through
  DeepSeek's native ``/responses`` (added 2026-08 for Codex).

Transport: Uses the ``openai`` Python SDK (AsyncOpenAI) pointed at the
DeepSeek base URL.  The SDK handles SSE parsing, keep-alive tolerance,
and retry/timeout plumbing.

References:
  - https://api-docs.deepseek.com/
  - https://api-docs.deepseek.com/guides/thinking_mode
  - https://api-docs.deepseek.com/api/create-chat-completion
  - https://api-docs.deepseek.com/updates (audited 2026-11-16)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
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
from . import openai_responses as rsp
from .base import BaseProvider, ContextPayload, carry_replay_to_tail

logger = logging.getLogger(__name__)

# =============================================================================
# DSML text-format tool calls (defensive normalization)
# =============================================================================
# deepseek-v4 models occasionally emit tool calls as DSML MARKUP inside
# message.content instead of the structured tool_calls field, e.g. (bars are
# U+FF5C FULLWIDTH VERTICAL LINE, deepseek's special-token style):
#
#   <|.|DSML|.|tool_calls>            (|.| = U+FF5C shown as ASCII here)
#   <|.|DSML|.|invoke name="list_dir">
#   <|.|DSML|.|parameter name="path" string="true">.</|.|DSML|.|parameter>
#   </|.|DSML|.|invoke>
#   </|.|DSML|.|tool_calls>
#
# Observed on multi-round tool turns (wairu Darwin-vs-Lite benchmark, bug 3):
# without normalization the markup leaks to every consumer as the final
# answer text. chat_completion() normalizes such responses into standard
# OpenAI shape (structured tool_calls + stripped content) at the provider
# boundary, so llmcore phases, host dispatchers, and extract_tool_calls all
# see a normal response. ASCII pipes are tolerated defensively.

_DSML_BAR = "[\uff5c|]{1,2}"
_DSML_BLOCK_RE = re.compile(
    rf"<{_DSML_BAR}DSML{_DSML_BAR}tool_calls>(.*?)</{_DSML_BAR}DSML{_DSML_BAR}tool_calls>",
    re.DOTALL,
)
_DSML_INVOKE_RE = re.compile(
    rf'<{_DSML_BAR}DSML{_DSML_BAR}invoke\s+name="([^"]+)"\s*>(.*?)</{_DSML_BAR}DSML{_DSML_BAR}invoke>',
    re.DOTALL,
)
_DSML_PARAM_RE = re.compile(
    rf'<{_DSML_BAR}DSML{_DSML_BAR}parameter\s+name="([^"]+)"'
    rf'(?:\s+string="(true|false)")?\s*>(.*?)</{_DSML_BAR}DSML{_DSML_BAR}parameter>',
    re.DOTALL,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Stable OpenAI-compatible API endpoint.
_BASE_URL_STABLE = "https://api.deepseek.com"

#: Beta endpoint for prefix completion, FIM, and strict tool schema.
_BASE_URL_BETA = "https://api.deepseek.com/beta"

#: Default model when none configured. ``deepseek-flash`` is V4.1-Flash
#: (multimodal, the built-in web/URL tools); V4 Flash itself was retired on
#: 2026-09-10.
_DEFAULT_MODEL = "deepseek-flash"

#: Context windows from GET /models (2026-11-16): ``context_window`` 1048576.
#: Live discovery (``get_models_details``) refreshes them.
_CONTEXT_LENGTHS: dict[str, int] = {
    "deepseek-flash": 1_048_576,
    "deepseek-v4-pro": 1_048_576,
    # Retired 2026-09-10; still accepted and served by V4.1-Flash.
    "deepseek-v4-flash": 1_048_576,
    "deepseek-v4-flash-vision-exp": 1_048_576,
}

#: Models whose built-in tools (deepseek_web_search, deepseek_read_url) are
#: documented (2026-11-13 changelog: V4.1-Flash).
_BUILTIN_TOOL_MODELS = ("deepseek-flash", "deepseek-v4-flash")

#: Valid thinking mode types.
ThinkingType = Literal["enabled", "disabled"]

#: Valid reasoning effort levels (GET /models: supported_levels).
ReasoningEffort = Literal["low", "high", "max"]

#: llmcore effort -> DeepSeek effort, as api-docs.deepseek.com/guides/
#: thinking_mode maps them server-side. "none" turns thinking off instead.
_EFFORT_MAP: dict[str, str] = {
    "minimal": "low",
    "low": "low",
    "medium": "high",
    "high": "high",
    "xhigh": "high",
    "max": "max",
    "ultra": "max",
}


class DeepSeekProvider(BaseProvider):
    """First-class DeepSeek provider with native thinking-mode support.

    Configuration keys (under ``[providers.deepseek]``):

    - ``api_key`` / ``api_key_env_var`` — API credential.
    - ``base_url`` — Override the stable API URL (default: ``https://api.deepseek.com``).
    - ``default_model`` — Default model ID (default: ``deepseek-v4-pro``).
    - ``timeout`` — HTTP request timeout in seconds (default: 300).
    - ``thinking`` — Default thinking mode: ``"enabled"`` or ``"disabled"``
      (default: ``"enabled"``).
    - ``reasoning_effort`` — Default reasoning effort; accepts the canonical
      values ``low | medium | high | xhigh | max`` and folds them to the
      DeepSeek wire vocabulary via ``_EFFORT_MAP`` (``minimal`` → ``"low"``,
      ``medium``/``xhigh`` → ``"high"``; unrecognized values fall back to
      ``"high"``; default: ``"high"``).
    - ``responses_api`` — ``"never"`` (default, Chat Completions) or
      ``"always"`` (DeepSeek's native ``/responses``).
    """

    default_model: str
    _client: AsyncOpenAI | None
    _beta_client: AsyncOpenAI | None
    _encoding: Any  # tiktoken.Encoding | None
    _default_thinking: ThinkingType
    _default_reasoning_effort: ReasoningEffort

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False):
        """Initialize the DeepSeek provider.

        Args:
            config: Provider-specific configuration dict from
                ``[providers.deepseek]``.
            log_raw_payloads: Whether to log raw request/response payloads.

        Raises:
            ConfigError: If the ``openai`` SDK is missing or no API key found.
        """
        super().__init__(config, log_raw_payloads)

        if not openai_available:
            raise ConfigError(
                "The 'openai' package is required for the DeepSeek provider. "
                "Install with: pip install openai"
            )

        # --- API key ---
        api_key = config.get("api_key") or os.environ.get(
            config.get("api_key_env_var", "DEEPSEEK_API_KEY")
        )
        if not api_key:
            api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ConfigError(
                "DeepSeek API key not found. Set DEEPSEEK_API_KEY or configure "
                "providers.deepseek.api_key / api_key_env_var."
            )

        #: Kept for the /v1/tools/* media endpoints (TTS, ASR, OCR).
        self._api_key = str(api_key)
        self._tools_base = str(config.get("tools_base_url", "https://api.deepseek.com/v1/tools")).rstrip("/")

        # --- Model / timeout ---
        self.default_model = config.get("default_model", _DEFAULT_MODEL)
        timeout = config.get("timeout", 300)

        # --- Thinking mode defaults ---
        thinking_raw = config.get("thinking", "enabled")
        if thinking_raw not in ("enabled", "disabled"):
            logger.warning("Invalid thinking value '%s'; defaulting to 'enabled'.", thinking_raw)
            thinking_raw = "enabled"
        self._default_thinking = thinking_raw  # type: ignore[assignment]

        effort_raw = config.get("reasoning_effort", "high")
        self._default_reasoning_effort = _EFFORT_MAP.get(effort_raw, "high")  # type: ignore[assignment]

        responses_api = str(config.get("responses_api", "never")).lower()
        if responses_api not in ("never", "always"):
            raise ConfigError(
                f"DeepSeek responses_api must be 'never' or 'always', not {responses_api!r}."
            )
        self.responses_api = responses_api
        #: model id -> {"context": int, "efforts": [...], "vision": bool} from /models
        self._discovered: dict[str, dict[str, Any]] = {}

        # --- Clients ---
        base_url = config.get("base_url", _BASE_URL_STABLE)
        try:
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
            )
            # Separate client for beta endpoint
            self._beta_client = AsyncOpenAI(
                api_key=api_key,
                base_url=config.get("beta_base_url", _BASE_URL_BETA),
                timeout=timeout,
            )
            logger.debug(
                "DeepSeek clients initialized (stable=%s, default_model=%s, "
                "thinking=%s, reasoning_effort=%s).",
                base_url,
                self.default_model,
                self._default_thinking,
                self._default_reasoning_effort,
            )
        except Exception as e:
            raise ConfigError(f"DeepSeek client initialization failed: {e}")

        # --- Tokenizer ---
        self._encoding = None
        if tiktoken_available:
            try:
                self._encoding = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                logger.warning("Failed to load tiktoken for DeepSeek: %s", e)

    # =========================================================================
    # BaseProvider interface
    # =========================================================================

    def get_name(self) -> str:
        return self._provider_instance_name or "deepseek"

    async def get_models_details(self) -> list[ModelDetails]:
        """Discover available models via the ``GET /models`` endpoint."""
        if not self._client:
            raise ProviderError(self.get_name(), "Client not initialized.")
        try:
            resp = await self._client.models.list()
            result: list[ModelDetails] = []
            provider = self.get_name()
            try:
                registry = get_model_card_registry()
            except Exception:
                registry = None

            for m in resp.data:
                mid = m.id
                # Only include deepseek models
                if not mid.startswith("deepseek"):
                    continue
                # The listing carries context_window, max_output_tokens,
                # input_modalities and effort levels (2026-11).
                extra = getattr(m, "model_extra", None) or {}
                inputs = extra.get("input_modalities") or []
                efforts = (extra.get("effort") or {}).get("supported_levels") or []
                if extra.get("context_window"):
                    self._discovered[mid] = {
                        "context": int(extra["context_window"]),
                        "efforts": list(efforts),
                        "vision": "image" in inputs,
                    }
                ctx = self.get_max_context_length(mid)

                # Check model card for capabilities
                supports_tools = True  # All V4 models support tools
                supports_vision = "image" in inputs
                if registry:
                    card = registry.get(provider, mid)
                    if card and card.capabilities:
                        supports_tools = (
                            card.capabilities.tool_use or card.capabilities.function_calling
                        )
                        supports_vision = supports_vision or card.capabilities.vision

                result.append(
                    ModelDetails(
                        id=mid,
                        context_length=ctx,
                        supports_streaming=True,
                        supports_tools=supports_tools,
                        supports_vision=supports_vision,
                        provider_name=provider,
                        supports_reasoning=True,
                        metadata={
                            "owned_by": m.owned_by,
                            "created": m.created,
                            "supports_thinking": True,
                            "name": extra.get("name"),
                            "max_output_tokens": extra.get("max_output_tokens"),
                            "effort_levels": efforts,
                        },
                    )
                )
            return result
        except OpenAIError as e:
            raise ProviderError(self.get_name(), f"Failed to fetch models: {e}")

    def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
        """Return the set of supported API parameters for DeepSeek models."""
        return {
            "temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0},
            "top_p": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "max_tokens": {"type": "integer", "minimum": 1},
            "presence_penalty": {"type": "number", "minimum": -2.0, "maximum": 2.0},
            "frequency_penalty": {"type": "number", "minimum": -2.0, "maximum": 2.0},
            "stop": {"type": "array", "items": {"type": "string"}},
            "response_format": {"type": "object"},
            "logprobs": {"type": "boolean"},
            "top_logprobs": {"type": "integer", "minimum": 0, "maximum": 20},
            "stream_options": {"type": "object"},
            # DeepSeek-specific
            "thinking": {
                "type": "object",
                "description": 'Thinking mode toggle: {"type": "enabled"} or {"type": "disabled"}',
            },
            "reasoning_effort": {
                "type": "string",
                "enum": ["minimal", "low", "medium", "high", "xhigh", "max"],
            },
            # llmcore's cross-provider knob: "none" turns thinking off.
            "effort": {
                "type": "string",
                "enum": ["none", "minimal", "low", "medium", "high", "xhigh", "max"],
            },
            # Built-in tools and safety (deepseek-flash, 2026-11-13).
            "native_search": {"type": "boolean"},
            "extra_tools": {"type": "array", "items": {"type": "object"}},
            "safety_check": {"type": "object"},
            "parallel_tool_calls": {"type": "boolean"},
        }

    def get_max_context_length(self, model: str | None = None) -> int:
        """Return the maximum context length for a DeepSeek model."""
        model_name = model or self.default_model
        if model_name in self._discovered:
            return self._discovered[model_name]["context"]
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

        # V4 / V4.1 models all have a 1M context
        if "v4" in model_name or model_name.startswith("deepseek-flash"):
            return 1_048_576

        # Conservative fallback for unknown models
        logger.warning(
            "Unknown context length for DeepSeek model '%s'. Falling back to 131072.",
            model_name,
        )
        return 131_072

    # =========================================================================
    # Chat completion
    # =========================================================================

    def _resolve_thinking_params(
        self, kwargs: dict[str, Any]
    ) -> tuple[dict[str, str] | None, str | None]:
        """Resolve thinking mode and reasoning effort from kwargs + defaults.

        Returns:
            (thinking_obj, reasoning_effort) — both suitable for direct
            injection into the API request body.
        """
        # Extract thinking from kwargs (remove so it doesn't go to OpenAI SDK)
        thinking = kwargs.pop("thinking", None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        effort = kwargs.pop("effort", None)
        if effort is not None:
            effort = str(effort).strip().lower()
            if effort == "none":
                thinking = thinking if thinking is not None else {"type": "disabled"}
            elif reasoning_effort is None:
                reasoning_effort = effort

        # Build thinking object
        if thinking is None:
            # Use provider-level default
            thinking_obj: dict[str, str] | None = {"type": self._default_thinking}
        elif isinstance(thinking, dict):
            thinking_obj = thinking
        elif isinstance(thinking, str):
            thinking_obj = {"type": thinking}
        elif isinstance(thinking, bool):
            # Boolean convenience: True → enabled, False → disabled
            thinking_obj = {"type": "enabled" if thinking else "disabled"}
        else:
            thinking_obj = {"type": self._default_thinking}

        # Resolve reasoning effort
        if reasoning_effort is None:
            effort = self._default_reasoning_effort
        else:
            effort = _EFFORT_MAP.get(str(reasoning_effort).lower(), "high")

        # Only set reasoning_effort when thinking is enabled
        if thinking_obj and thinking_obj.get("type") == "enabled":
            return thinking_obj, effort
        else:
            return thinking_obj, None

    def _build_message_payload(self, msg: Message) -> dict[str, Any]:
        """Build a DeepSeek-format message dict from an llmcore Message.

        Key differences from plain OpenAI:
        - Preserves ``reasoning_content`` on assistant messages (required
          for tool-call multi-turn under thinking mode).
        - Supports ``prefix: true`` on assistant messages for Chat Prefix
          Completion.
        """
        role_str = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        metadata = msg.metadata or {}

        content: Any = msg.content
        # Support content_parts for structured content
        if "content_parts" in metadata:
            content = metadata["content_parts"]

        msg_dict: dict[str, Any] = {"role": role_str, "content": content}

        # Tool message needs tool_call_id
        if msg.role == LLMCoreRole.TOOL and msg.tool_call_id:
            msg_dict["tool_call_id"] = msg.tool_call_id

        # Assistant message may carry tool_calls and reasoning_content.
        # First-class Message.tool_calls (R-2) takes precedence over the
        # legacy metadata["tool_calls"] channel — without this, a role="tool"
        # result sent by a caller using the native tool-role protocol has no
        # preceding assistant tool_calls and the API rejects the request.
        if role_str == "assistant":
            tool_calls = getattr(msg, "tool_calls", None) or metadata.get("tool_calls")
            if tool_calls:
                msg_dict["tool_calls"] = tool_calls
                if not msg.content:
                    msg_dict["content"] = None

            # Preserve reasoning_content for thinking-mode tool-call turns
            if "reasoning_content" in metadata:
                msg_dict["reasoning_content"] = metadata["reasoning_content"]

            # Chat Prefix Completion support
            if metadata.get("prefix"):
                msg_dict["prefix"] = True

        # Optional name field
        name = metadata.get("name")
        if name:
            msg_dict["name"] = name

        return msg_dict

    @staticmethod
    def _parse_dsml_arguments(body: str) -> dict[str, Any]:
        """Parse one DSML invoke body into an arguments dict.

        ``string="true"`` params stay raw strings; anything else is
        JSON-parsed when possible (booleans/numbers/objects), else kept raw.
        """
        args: dict[str, Any] = {}
        for pname, is_string, raw in _DSML_PARAM_RE.findall(body):
            value = raw.strip()
            if is_string == "true":
                args[pname] = value
            else:
                try:
                    args[pname] = json.loads(value)
                except (ValueError, TypeError):
                    args[pname] = value
        return args

    def _normalize_dsml_tool_calls(self, response: dict[str, Any]) -> dict[str, Any]:
        """Rewrite DSML text-format tool calls into structured tool_calls.

        No-op when the structured field is already populated (never
        double-parse) or when the content carries no DSML block. Mutates and
        returns ``response``; any parser error passes the response through
        untouched (defensive normalization must never break a reply).
        """
        try:
            for choice in response.get("choices") or []:
                message = choice.get("message")
                if not isinstance(message, dict) or message.get("tool_calls"):
                    continue
                content = message.get("content")
                if not content or "DSML" not in content:
                    continue

                calls: list[dict[str, Any]] = []
                for block in _DSML_BLOCK_RE.finditer(content):
                    for name, body in _DSML_INVOKE_RE.findall(block.group(1)):
                        calls.append(
                            {
                                "id": f"dsml_call_{len(calls)}",
                                "type": "function",
                                "function": {
                                    "name": name.strip(),
                                    "arguments": json.dumps(
                                        self._parse_dsml_arguments(body)
                                    ),
                                },
                            }
                        )
                if not calls:
                    continue
                stripped = _DSML_BLOCK_RE.sub("", content)
                message["tool_calls"] = calls
                message["content"] = stripped.strip() or None
                if choice.get("finish_reason") == "stop":
                    choice["finish_reason"] = "tool_calls"
                # Deliberately visible: this is model/API misbehavior we paper
                # over, and its frequency is worth tracking.
                logger.warning(
                    "DeepSeek emitted %d DSML text-format tool call(s); "
                    "normalized into structured tool_calls",
                    len(calls),
                )
        except Exception as e:
            logger.error("DSML normalization failed (response passed through): %s", e)
        return response

    async def chat_completion(
        self,
        context: ContextPayload,
        model: str | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Perform a chat completion against the DeepSeek API.

        Extends the standard OpenAI-compatible flow with:

        - Automatic thinking-mode parameter injection.
        - Beta endpoint routing for prefix completion and strict tools.
        - ``reasoning_content`` extraction in streaming and non-streaming modes.

        Additional kwargs accepted:

        - ``thinking``: ``dict | str | bool | None`` — override thinking mode
          for this request.
        - ``reasoning_effort``: ``str | None`` — override reasoning effort.
        - ``use_beta``: ``bool`` — force the beta endpoint.
        """
        if not self._client:
            raise ProviderError(self.get_name(), "Client not initialized.")

        model_name = model or self.default_model

        # Validate kwargs (only known parameters)
        supported = self.get_supported_parameters(model_name)
        # Allow 'use_beta' as a passthrough
        for key in kwargs:
            if key not in supported and key != "use_beta":
                raise ValueError(f"Unsupported parameter '{key}' for DeepSeek provider.")

        # Validate context
        if not (isinstance(context, list) and all(isinstance(msg, Message) for msg in context)):
            raise ProviderError(self.get_name(), "Context must be list[Message].")

        messages_payload = [self._build_message_payload(msg) for msg in context]
        if not messages_payload:
            raise ProviderError(self.get_name(), "No valid messages.")

        # --- Built-in tools (deepseek_web_search, deepseek_read_url) ---
        builtin_tools = [dict(t) for t in kwargs.pop("extra_tools", None) or []]
        if kwargs.pop("native_search", False):
            if self.supports_native_search(model_name):
                if not any(t.get("type") == "deepseek_web_search" for t in builtin_tools):
                    builtin_tools.append({"type": "deepseek_web_search"})
            else:
                logger.debug("DeepSeek model %s has no built-in web search; ignored.", model_name)

        # Thinking mode
        thinking_obj, effort = self._resolve_thinking_params(kwargs)

        if self.responses_api == "always":
            return await self._responses_completion(
                context, model_name, stream=stream, tools=tools, tool_choice=tool_choice,
                builtin_tools=builtin_tools, thinking=thinking_obj, effort=effort, params=kwargs,
            )

        # --- Tool schema ---
        tools_payload = None
        if tools:
            tools_payload = [{"type": "function", "function": t.model_dump()} for t in tools]
        if builtin_tools:
            tools_payload = (tools_payload or []) + builtin_tools

        # --- Build API kwargs ---
        api_kwargs = {}
        if thinking_obj:
            api_kwargs["extra_body"] = api_kwargs.get("extra_body", {})
            api_kwargs["extra_body"]["thinking"] = thinking_obj
        if effort:
            api_kwargs["extra_body"] = api_kwargs.get("extra_body", {})
            api_kwargs["extra_body"]["reasoning_effort"] = effort

        # When thinking is enabled, sampling params are ignored by DeepSeek
        is_thinking = thinking_obj and thinking_obj.get("type") == "enabled"
        sampling_keys = {"temperature", "top_p", "presence_penalty", "frequency_penalty"}

        # Copy remaining kwargs
        use_beta = kwargs.pop("use_beta", False)
        safety_check = kwargs.pop("safety_check", None)
        if safety_check is not None:
            api_kwargs.setdefault("extra_body", {})["safety_check"] = safety_check
        for key, val in kwargs.items():
            if key in ("thinking", "reasoning_effort"):
                continue  # Already handled
            if is_thinking and key in sampling_keys:
                # Silently skip — DeepSeek ignores these in thinking mode
                logger.debug("Skipping '%s' (ignored in thinking mode).", key)
                continue
            api_kwargs[key] = val

        if tools_payload:
            api_kwargs["tools"] = tools_payload
        if tool_choice:
            api_kwargs["tool_choice"] = tool_choice

        # Detect if we need the beta endpoint
        needs_beta = use_beta
        if not needs_beta:
            # Check for prefix completion
            if messages_payload and messages_payload[-1].get("prefix"):
                needs_beta = True
            # Check for strict tool schema
            if tools_payload:
                for t in tools_payload:
                    if t.get("function", {}).get("strict"):
                        needs_beta = True
                        break

        client = self._beta_client if needs_beta else self._client

        # Stream options — request usage in final chunk
        if stream:
            api_kwargs.setdefault("stream_options", {"include_usage": True})

        # --- Logging ---
        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "RAW DEEPSEEK REQUEST: %s",
                json.dumps(
                    {
                        "model": model_name,
                        "messages": messages_payload,
                        "stream": stream,
                        "endpoint": "beta" if needs_beta else "stable",
                        **api_kwargs,
                    },
                    indent=2,
                    default=str,
                ),
            )

        # --- API call ---
        try:
            resp = await client.chat.completions.create(
                model=model_name,
                messages=messages_payload,
                stream=stream,
                **api_kwargs,
            )  # type: ignore

            if stream:

                async def stream_wrapper() -> AsyncGenerator[dict[str, Any], None]:
                    # The thinking arrives in pieces; the chunk with the
                    # finish reason gets the whole of it as
                    # ``replay_reasoning_content``, which a tool loop has to
                    # send back (assistant_replay_metadata).
                    reasoning: list[str] = []
                    try:
                        _replay_tail: dict[str, Any] = {}
                        async for chunk in resp:  # type: ignore
                            chunk_dict = chunk.model_dump(exclude_none=True)
                            if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                                logger.debug(
                                    "RAW DEEPSEEK STREAM CHUNK: %s",
                                    json.dumps(chunk_dict, default=str),
                                )
                            for choice in chunk_dict.get("choices") or []:
                                if choice.get("index", 0) != 0:
                                    continue
                                delta = choice.get("delta") or {}
                                if delta.get("reasoning_content"):
                                    reasoning.append(delta["reasoning_content"])
                                if choice.get("finish_reason") and reasoning:
                                    choice.setdefault("delta", delta)["replay_reasoning_content"] = (
                                        "".join(reasoning)
                                    )
                            carry_replay_to_tail(chunk_dict, _replay_tail)
                            yield chunk_dict
                    except OpenAIError as e:
                        raise self._map_error(e, model_name) from e

                return stream_wrapper()
            else:
                response_dict = resp.model_dump(exclude_none=True)  # type: ignore
                if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "RAW DEEPSEEK RESPONSE: %s",
                        json.dumps(response_dict, indent=2, default=str),
                    )
                # Bug 3 (Darwin-vs-Lite eval): normalize DSML text-format
                # tool calls into the structured field at the provider
                # boundary so no consumer ever sees the raw markup.
                return self._normalize_dsml_tool_calls(response_dict)

        except OpenAIError as e:
            raise self._map_error(e, model_name) from e
        except Exception as e:
            logger.error("Unexpected DeepSeek error: %s", e, exc_info=True)
            raise ProviderError(self.get_name(), f"Unexpected error: {e}") from e

    def _map_error(self, e: Exception, model_name: str) -> ProviderError:
        """An SDK error as a ProviderError, with DeepSeek's specific cases."""
        if isinstance(e, OpenAIAPITimeoutError):
            return ProviderError(self.get_name(), f"Timeout: {e}", model_name=model_name)
        if isinstance(e, OpenAIAPIConnectionError):
            return ProviderError(self.get_name(), f"Connection error: {e}", model_name=model_name)
        if not isinstance(e, OpenAIAPIStatusError):
            return ProviderError(self.get_name(), f"API Error: {e}", model_name=model_name)
        status = e.status_code
        msg = str(e)
        low = msg.lower()
        if status == 400 and any(
            p in low for p in ("context_length", "maximum context length", "context window")
        ):
            return ContextLengthError(
                model_name=model_name,
                limit=self.get_max_context_length(model_name),
                actual=0,
                message=msg,
            )
        if status in (400, 404) and any(
            p in low for p in ("model not exist", "does not exist", "model_not_found")
        ):
            return ProviderError(
                self.get_name(),
                f"Model '{model_name}' not found on DeepSeek. Current models: "
                f"deepseek-flash, deepseek-v4-pro. Original error: {msg}",
                model_name=model_name,
                status_code=status,
                retryable=False,
            )
        if status == 400 and "reasoning_content" in low:
            return ProviderError(
                self.get_name(),
                "DeepSeek needs every earlier assistant turn's reasoning_content when "
                "tools are in use: put provider.assistant_replay_metadata(response) in "
                f"that turn's Message.metadata. Original error: {msg}",
                model_name=model_name,
                status_code=status,
                retryable=False,
            )
        if status == 402:
            return ProviderError(
                self.get_name(),
                f"Insufficient balance on DeepSeek account. "
                f"Check balance at https://platform.deepseek.com. Error: {msg}",
                model_name=model_name,
                status_code=status,
                retryable=False,
            )
        return ProviderError(
            self.get_name(), f"API Error ({status}): {msg}", model_name=model_name, status_code=status
        )

    # =========================================================================
    # Media: /v1/tools/{tts,asr,ocr}
    # =========================================================================

    _MEDIA_CAPABILITIES: frozenset[str] = frozenset({"tts", "tts_stream", "asr", "ocr"})

    def media_capabilities(self) -> Any:
        """Speech, transcription and OCR tool APIs (api-docs.deepseek.com/guides)."""
        from ..media.models import MediaCapability

        return frozenset(MediaCapability(c) for c in self._MEDIA_CAPABILITIES)

    def media_execution(self, capability: Any, model: str | None = None) -> Any:
        from ..media.models import MediaCapability, MediaExecution

        if capability is MediaCapability.TTS_STREAM:
            return MediaExecution.STREAM
        return MediaExecution.REQUEST_RESPONSE

    def _tools_error(self, status: int, body: str, what: str) -> ProviderError:
        """Map a /v1/tools/* failure (402 balance, 413 size, 429 concurrency ...)."""
        hints = {
            400: "the input was rejected (unreadable content, unsupported format, or content review)",
            402: "the account has insufficient balance",
            408: "the upload did not finish in time",
            413: "the file is too large",
            429: "too many requests in progress for this account",
            503: "the service is busy",
            504: "processing timed out",
        }
        return ProviderError(
            self.get_name(),
            f"DeepSeek {what} failed ({status}): {hints.get(status, 'error')}. {body[:300]}",
            status_code=status,
            retryable=status in (429, 503, 504),
        )

    def _tools_client(self, timeout: float = 300.0) -> Any:
        import httpx

        return httpx.AsyncClient(timeout=timeout, headers={"Authorization": f"Bearer {self._api_key}"})

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
    ) -> Any:
        """Text to speech (``POST /v1/tools/tts``): voices ``stella`` / ``tide``,
        ``wav`` (default) or ``pcm`` (16-bit mono 24 kHz). The language is
        detected from the text; speed control is not offered."""
        from ..media.models import MediaArtifact, MediaCapability, MediaKind, MediaResult, MediaUsage

        audio = b"".join([chunk async for chunk in self.stream_speech_media(
            text, voice=voice, audio_format=audio_format, **kwargs)])
        fmt = audio_format or "wav"
        return MediaResult(
            capability=MediaCapability.TTS,
            provider=self.get_name(),
            model="deepseek-tts",
            artifacts=(MediaArtifact(
                kind=MediaKind.AUDIO, data=audio,
                mime_type="audio/wav" if fmt == "wav" else "audio/L16;rate=24000",
                sample_rate_hz=24000,
            ),),
            usage=MediaUsage(provider=self.get_name(), model="deepseek-tts", basis="per_character",
                             characters=len(text)),
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
    ) -> Any:
        """Stream TTS audio as it is synthesized. A response that ends early
        is a failure per DeepSeek's docs (the stream is cut off)."""
        body: dict[str, Any] = {"text": text, "voice_id": voice or "stella", "format": audio_format or "wav"}
        body.update(kwargs)

        async def _stream() -> Any:
            async with self._tools_client() as client:
                async with client.stream("POST", f"{self._tools_base}/tts", json=body) as resp:
                    if resp.status_code >= 400:
                        raise self._tools_error(resp.status_code, (await resp.aread()).decode("utf-8", "replace"), "TTS")
                    async for chunk in resp.aiter_bytes():
                        if chunk:
                            yield chunk

        return _stream()

    async def transcribe_media(
        self,
        *,
        audio: Any,
        model: str | None = None,
        language: str | None = None,
        diarize: bool | None = None,
        timestamps: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        """Speech to text (``POST /v1/tools/asr``): 16-bit PCM WAV or Ogg Opus,
        at most 8 MiB and 120 s; the language is detected from the audio."""
        from ..media.models import MediaCapability, MediaResult, MediaUsage
        from ..models_multimodal import TranscriptionResult

        if audio.is_remote:
            from ..media.artifacts import default_fetcher

            data = await default_fetcher()(audio.url or "")
        else:
            data = audio.read_bytes()
        async with self._tools_client(timeout=660.0) as client:
            resp = await client.post(f"{self._tools_base}/asr",
                                     files={"file": (audio.filename or "audio.wav", data, audio.mime_type or "audio/wav")})
        if resp.status_code >= 400:
            raise self._tools_error(resp.status_code, resp.text, "ASR")
        payload = resp.json()
        seconds = (payload.get("duration_ms") or 0) / 1000 or None
        transcript = TranscriptionResult(text=str(payload.get("text") or ""), duration_seconds=seconds,
                                         model="deepseek-asr", metadata={"request_id": payload.get("request_id")})
        return MediaResult(
            capability=MediaCapability.ASR, provider=self.get_name(), model="deepseek-asr",
            artifacts=(transcript.to_artifact(),),
            usage=MediaUsage(provider=self.get_name(), model="deepseek-asr", basis="per_second", seconds=seconds),
            raw=payload,
        )

    async def ocr_media(
        self,
        *,
        document: Any,
        model: str | None = None,
        pages: Any = None,
        purpose: str = "text_recognition",
        **kwargs: Any,
    ) -> Any:
        """OCR (``POST /v1/tools/ocr``): PDF, Word, PowerPoint or an image;
        ``purpose`` is ``text_recognition`` or ``rich_layout_recognition``.
        The service keeps the connection alive with leading newlines, which
        JSON parsing ignores."""
        from ..media.models import MediaCapability, MediaResult, MediaUsage
        from ..models_multimodal import OCRResult

        if pages is not None:
            logger.debug("DeepSeek OCR has no page selection; all pages are recognized.")
        if document.is_remote:
            from ..media.artifacts import default_fetcher

            data = await default_fetcher()(document.url or "")
        else:
            data = document.read_bytes()
        async with self._tools_client(timeout=200.0) as client:
            resp = await client.post(
                f"{self._tools_base}/ocr", data={"purpose": purpose},
                files={"file": (document.filename or "document", data, document.mime_type or "application/octet-stream")})
        if resp.status_code >= 400:
            raise self._tools_error(resp.status_code, resp.text, "OCR")
        payload = json.loads(resp.text)
        result = OCRResult(
            pages=[{"index": p.get("index"), "markdown": p.get("text", ""), "finish_reason": p.get("finish_reason")}
                   for p in payload.get("pages") or []],
            model="deepseek-ocr", pages_processed=int(payload.get("page_count") or 0),
            metadata={"id": payload.get("id"), "purpose": payload.get("purpose"), "usage": payload.get("usage")},
        )
        usage = payload.get("usage") or {}
        return MediaResult(
            capability=MediaCapability.OCR, provider=self.get_name(), model="deepseek-ocr",
            artifacts=(result.to_artifact(),),
            usage=MediaUsage(provider=self.get_name(), model="deepseek-ocr", basis="per_page",
                             input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens")),
            raw=payload,
        )

    def supports_native_search(self, model: str | None = None) -> bool:
        """``deepseek_web_search`` is documented for V4.1-Flash (deepseek-flash)."""
        return (model or self.default_model).startswith(_BUILTIN_TOOL_MODELS)

    async def get_balance(self) -> dict[str, Any]:
        """``GET /user/balance``: ``is_available`` and per-currency balances."""
        if not self._client:
            raise ProviderError(self.get_name(), "Client not initialized.")
        try:
            resp = await self._client.get("/user/balance", cast_to=object)
        except OpenAIError as e:
            raise self._map_error(e, "balance") from e
        return dict(resp) if isinstance(resp, dict) else {}

    # =========================================================================
    # Responses API
    # =========================================================================

    async def _responses_completion(
        self,
        context: list[Message],
        model_name: str,
        *,
        stream: bool,
        tools: list[Tool] | None,
        tool_choice: Any,
        builtin_tools: list[dict[str, Any]],
        thinking: dict[str, Any] | None,
        effort: str | None,
        params: dict[str, Any],
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Chat through DeepSeek's native ``/responses`` (Codex-compatible).

        Thinking maps to ``reasoning.effort`` ("none" when it is off).
        Reasoning items come back with ``store: false`` and are replayed via
        ``metadata["reasoning_items"]``.
        """
        params = dict(params)
        params.pop("use_beta", None)
        params.pop("stream_options", None)
        safety_check = params.pop("safety_check", None)
        thinking_off = bool(thinking) and thinking.get("type") == "disabled"
        body = rsp.build_responses_request(
            model_name,
            context,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
            extra_tools=builtin_tools,
            reasoning={"effort": "none" if thinking_off else (effort or self._default_reasoning_effort)},
            **params,
        )
        if safety_check is not None:
            body["safety_check"] = safety_check
        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
            logger.debug("RAW DEEPSEEK RESPONSES REQUEST: %s", json.dumps(body, default=str))
        try:
            result = await rsp.sdk_create(self._client, body, stream=stream)
        except OpenAIError as e:
            raise self._map_error(e, model_name) from e
        except rsp.ResponsesFailed as exc:
            raise ProviderError(self.get_name(), f"Response failed: {exc}", model_name=model_name) from None
        if not stream:
            return result

        async def _events() -> AsyncGenerator[dict[str, Any], None]:
            try:
                async for chunk in result:  # type: ignore[union-attr]
                    yield chunk
            except OpenAIError as e:
                raise self._map_error(e, model_name) from e
            except rsp.ResponsesFailed as exc:
                raise ProviderError(
                    self.get_name(), f"Response failed: {exc}", model_name=model_name
                ) from None

        return _events()

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
            logger.warning("Failed to extract DeepSeek content: %s", e)
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

        This is the ``reasoning_content`` field that DeepSeek returns when
        thinking mode is enabled.

        Args:
            response: Full API response dict.

        Returns:
            The reasoning text, or None if not present.
        """
        try:
            choices = response.get("choices", [])
            if not choices:
                return None
            return choices[0].get("message", {}).get("reasoning_content")
        except (KeyError, IndexError, TypeError):
            return None

    def extract_delta_reasoning_content(self, chunk: dict[str, Any]) -> str | None:
        """Extract reasoning delta from a streaming chunk.

        Args:
            chunk: A single streaming chunk dict.

        Returns:
            The reasoning text delta, or None if not present in this chunk.
        """
        try:
            choices = chunk.get("choices", [])
            if not choices:
                return None
            return choices[0].get("delta", {}).get("reasoning_content")
        except (KeyError, IndexError, TypeError):
            return None

    def assistant_replay_metadata(self, response: dict[str, Any]) -> dict[str, Any]:
        """The reasoning to send back with the next assistant turn.

        With tools, DeepSeek requires ``reasoning_content`` on every earlier
        assistant turn ("must be fully passed back ... the API will return a
        400 error"). On the Responses route it is the reasoning items. For a
        stream, pass the chunk that carries the finish reason.
        """
        try:
            choice = (response.get("choices") or [{}])[0]
            source = choice.get("message") or choice.get("delta") or {}
        except (AttributeError, IndexError, TypeError):
            return {}
        if source.get("reasoning_items"):
            return {"reasoning_items": list(source["reasoning_items"])}
        text = source.get("replay_reasoning_content") or (
            choice.get("message") or {}
        ).get("reasoning_content")
        return {"reasoning_content": text} if text else {}

    def extract_tool_calls(self, response: dict[str, Any]) -> list[ToolCall]:
        """Extract tool calls from a DeepSeek response."""
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
            logger.warning("Failed to extract DeepSeek tool calls: %s", e)
        return out

    def extract_usage_details(self, response: dict[str, Any]) -> dict[str, Any]:
        """Extract extended usage details including cache and reasoning tokens.

        DeepSeek usage includes:
        - ``prompt_cache_hit_tokens`` — tokens served from context cache
        - ``prompt_cache_miss_tokens`` — tokens not in cache
        - ``completion_tokens_details.reasoning_tokens`` — thinking tokens

        These are crucial for accurate cost accounting with DeepSeek's
        tiered cache-hit/miss pricing.
        """
        usage = response.get("usage", {})
        if not usage:
            return {}

        result: dict[str, Any] = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            # DeepSeek-specific cache fields
            "prompt_cache_hit_tokens": usage.get("prompt_cache_hit_tokens"),
            "prompt_cache_miss_tokens": usage.get("prompt_cache_miss_tokens"),
        }

        # Reasoning tokens
        cd = usage.get("completion_tokens_details", {})
        if cd:
            result["reasoning_tokens"] = cd.get("reasoning_tokens")

        return result

    def extract_finish_reason(self, response: dict[str, Any]) -> str | None:
        """Extract finish reason from a non-streaming response.

        DeepSeek finish reasons:
        - ``stop`` — natural completion
        - ``length`` — token limit reached
        - ``content_filter`` — filtered
        - ``tool_calls`` — tool invocation
        - ``insufficient_system_resource`` — provider resource issue
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
        """Count tokens using tiktoken cl100k_base (approximate for DeepSeek)."""
        if not text:
            return 0
        if not self._encoding:
            return _EstimateCounter().count(text)
        return await asyncio.to_thread(lambda: len(self._encoding.encode(text)))

    async def count_message_tokens(self, messages: list[Message], model: str | None = None) -> int:
        """Estimate total tokens for a list of messages.

        Uses tiktoken cl100k_base as a proxy; for precise counts use the
        official DeepSeek tokenizer.
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
        """Close HTTP clients."""
        errors: list[Exception] = []
        for name, client in [("stable", self._client), ("beta", self._beta_client)]:
            if client:
                try:
                    await client.close()
                except Exception as e:
                    errors.append(e)
                    logger.error("Error closing DeepSeek %s client: %s", name, e)
        self._client = None
        self._beta_client = None
        if not errors:
            logger.info("DeepSeekProvider closed.")
