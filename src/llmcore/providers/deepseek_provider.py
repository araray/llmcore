# src/llmcore/providers/deepseek_provider.py
"""
DeepSeek API provider implementation for the LLMCore library.

Handles interactions with the DeepSeek API (V4-Flash, V4-Pro models).
Uses the OpenAI-compatible chat completions endpoint with DeepSeek-specific
extensions:

- **Thinking mode**: ``thinking.type = "enabled" | "disabled"`` toggle plus
  ``reasoning_effort = "high" | "max"`` control.
- **Reasoning content**: ``reasoning_content`` field on assistant messages
  (both streaming delta and non-streaming response).
- **Cache-aware token accounting**: ``prompt_cache_hit_tokens`` and
  ``prompt_cache_miss_tokens`` in the usage object.
- **Beta endpoints**: Chat Prefix Completion, FIM Completion, and strict
  tool schema mode via ``https://api.deepseek.com/beta``.
- **Tool calls under thinking mode**: preserves ``reasoning_content`` when
  tool calls are present (required by the DeepSeek API for multi-turn).

Transport: Uses the ``openai`` Python SDK (AsyncOpenAI) pointed at the
DeepSeek base URL.  The SDK handles SSE parsing, keep-alive tolerance,
and retry/timeout plumbing.

References:
  - https://api-docs.deepseek.com/
  - https://api-docs.deepseek.com/guides/thinking_mode
  - https://api-docs.deepseek.com/api/create-chat-completion
  - DeepSeek-V4 technical report (2026)
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
from .base import BaseProvider, ContextPayload

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Stable OpenAI-compatible API endpoint.
_BASE_URL_STABLE = "https://api.deepseek.com"

#: Beta endpoint for prefix completion, FIM, and strict tool schema.
_BASE_URL_BETA = "https://api.deepseek.com/beta"

#: Default model when none configured.
_DEFAULT_MODEL = "deepseek-v4-pro"

#: Hardcoded context length map.  V4 models natively support 1M tokens.
_CONTEXT_LENGTHS: dict[str, int] = {
    "deepseek-v4-flash": 1_000_000,
    "deepseek-v4-pro": 1_000_000,
    # Legacy compatibility aliases (retiring 2026-07-24)
    "deepseek-chat": 131_072,
    "deepseek-reasoner": 131_072,
    # Older snapshot IDs
    "deepseek-v3.2": 131_072,
    "deepseek-r1": 131_072,
}

#: Valid thinking mode types.
ThinkingType = Literal["enabled", "disabled"]

#: Valid reasoning effort levels.
ReasoningEffort = Literal["high", "max"]

#: Reasoning effort normalization map (DeepSeek folds low/medium → high).
_EFFORT_MAP: dict[str, str] = {
    "low": "high",
    "medium": "high",
    "high": "high",
    "xhigh": "max",
    "max": "max",
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
    - ``reasoning_effort`` — Default reasoning effort: ``"high"`` or ``"max"``
      (default: ``"high"``).
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
                ctx = self.get_max_context_length(mid)

                # Check model card for capabilities
                supports_tools = True  # All V4 models support tools
                supports_vision = False  # No vision support yet
                if registry:
                    card = registry.get(provider, mid)
                    if card and card.capabilities:
                        supports_tools = (
                            card.capabilities.tool_use or card.capabilities.function_calling
                        )
                        supports_vision = card.capabilities.vision

                result.append(
                    ModelDetails(
                        id=mid,
                        context_length=ctx,
                        supports_streaming=True,
                        supports_tools=supports_tools,
                        supports_vision=supports_vision,
                        provider_name=provider,
                        metadata={
                            "owned_by": m.owned_by,
                            "created": m.created,
                            "supports_thinking": True,
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
                "enum": ["low", "medium", "high", "xhigh", "max"],
            },
        }

    def get_max_context_length(self, model: str | None = None) -> int:
        """Return the maximum context length for a DeepSeek model."""
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

        # V4 models all have 1M context
        if "v4" in model_name:
            return 1_000_000

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
            effort = _EFFORT_MAP.get(str(reasoning_effort), "high")

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

        # Assistant message may carry tool_calls and reasoning_content
        if role_str == "assistant":
            if "tool_calls" in metadata:
                msg_dict["tool_calls"] = metadata["tool_calls"]
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

        # --- Tool schema ---
        tools_payload = None
        if tools:
            tools_payload = [{"type": "function", "function": t.model_dump()} for t in tools]

        # --- Build API kwargs ---
        api_kwargs = {}

        # Thinking mode
        thinking_obj, effort = self._resolve_thinking_params(kwargs)
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
                    async for chunk in resp:  # type: ignore
                        chunk_dict = chunk.model_dump(exclude_none=True)
                        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                "RAW DEEPSEEK STREAM CHUNK: %s",
                                json.dumps(chunk_dict, default=str),
                            )
                        yield chunk_dict

                return stream_wrapper()
            else:
                response_dict = resp.model_dump(exclude_none=True)  # type: ignore
                if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "RAW DEEPSEEK RESPONSE: %s",
                        json.dumps(response_dict, indent=2, default=str),
                    )
                return response_dict

        except OpenAIAPIStatusError as e:
            status = e.status_code
            msg = str(e)
            logger.error("DeepSeek status error (%d): %s", status, msg, exc_info=True)

            if status == 400 and "context_length" in msg.lower():
                raise ContextLengthError(
                    provider_name=self.get_name(),
                    model=model_name,
                    max_tokens=self.get_max_context_length(model_name),
                    requested_tokens=None,
                    message=msg,
                )
            if status == 400 and any(
                p in msg.lower() for p in ("model not exist", "does not exist", "model_not_found")
            ):
                raise ProviderError(
                    self.get_name(),
                    f"Model '{model_name}' not found on DeepSeek. "
                    f"Current V4 models: deepseek-v4-flash, deepseek-v4-pro. "
                    f"Legacy aliases deepseek-chat/deepseek-reasoner retire "
                    f"2026-07-24. Original error: {msg}",
                )
            if status == 402:
                raise ProviderError(
                    self.get_name(),
                    f"Insufficient balance on DeepSeek account. "
                    f"Check balance at https://platform.deepseek.com. Error: {msg}",
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
            logger.error("Unexpected DeepSeek error: %s", e, exc_info=True)
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
        if not self._encoding:
            # Rough heuristic: ~0.3 tokens per character (English)
            return max(1, int(len(text) * 0.3))
        if not text:
            return 0
        return await asyncio.to_thread(lambda: len(self._encoding.encode(text)))

    async def count_message_tokens(self, messages: list[Message], model: str | None = None) -> int:
        """Estimate total tokens for a list of messages.

        Uses tiktoken cl100k_base as a proxy; for precise counts use the
        official DeepSeek tokenizer.
        """
        if not self._encoding:
            total = sum(
                len(m.content) + len(m.role.value if hasattr(m.role, "value") else str(m.role))
                for m in messages
            )
            return max(1, int(total * 0.3) + len(messages) * 4)

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
