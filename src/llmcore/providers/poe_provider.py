# src/llmcore/providers/poe_provider.py
"""
LLMCore provider for Poe — a unified gateway to hundreds of AI models and bots.

Poe's API offers two connectivity modes, both supported here:

1. **OpenAI-compatible** (``/v1/chat/completions``): Fully OpenAI-compatible
   chat completions.  Uses ``openai`` SDK with ``base_url`` override.
2. **Native SSE** (``fastapi_poe`` client): Poe's native SSE-based bot query
   protocol.  Offers custom parameters, file attachments, and Poe-specific
   features not available through the OpenAI-compatible surface.

The provider **subclasses OpenAIProvider** for (1) and wraps the
``fastapi_poe`` async client for (2), falling back transparently between them.

Key characteristics of the Poe API:
- Uses **bot names** as model IDs (e.g., ``Claude-Sonnet-4.6``, ``GPT-5.4``).
- Point-based billing; per-token pricing exposed via ``/v1/models``.
- Rate limit: 500 RPM (request-based, no token-based limits).
- Supports: streaming, tools, vision (base64), file uploads,
  web search (Responses API), video generation.
- Some OpenAI parameters are ignored (response_format, seed,
  frequency_penalty, presence_penalty).

Tested against Poe API as of April 2026 / fastapi_poe v0.0.83.

References:
    https://creator.poe.com/api-reference/overview
    https://creator.poe.com/docs/poe-protocol-specification
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# --- Optional fastapi_poe (native SSE client) imports ---
_fastapi_poe_available = False
try:
    from fastapi_poe.client import (
        BotError as PoeBotError,
    )
    from fastapi_poe.client import (
        BotMessage as PoeBotMessage,
    )
    from fastapi_poe.client import (
        get_bot_response,
    )
    from fastapi_poe.types import (
        ProtocolMessage as PoeProtocolMessage,
    )
    from fastapi_poe.types import (
        ToolDefinition as PoeToolDefinition,
    )

    _fastapi_poe_available = True
except ImportError:
    PoeBotError = Exception  # type: ignore[assignment, misc]
    PoeBotMessage = None  # type: ignore[assignment, misc]
    PoeProtocolMessage = None  # type: ignore[assignment, misc]
    PoeToolDefinition = None  # type: ignore[assignment, misc]
    get_bot_response = None  # type: ignore[assignment]

from ..exceptions import ConfigError, ProviderError
from ..model_cards.registry import get_model_card_registry
from ..models import Message, ModelDetails, Tool
from ..models import Role as LLMCoreRole
from .base import BaseProvider, ContextPayload

# Inherit from OpenAIProvider for the OpenAI-compatible path.
try:
    from .openai_provider import OpenAIProvider, openai_available
except ImportError:
    OpenAIProvider = None  # type: ignore[assignment, misc]
    openai_available = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POE_BASE_URL = "https://api.poe.com/v1"
POE_MODELS_URL = "https://api.poe.com/v1/models"
POE_BALANCE_URL = "https://api.poe.com/usage/current_balance"
POE_NATIVE_BASE_URL = "https://api.poe.com/bot/"
DEFAULT_POE_MODEL = "GPT-4o-Mini"

# Poe ignores these OpenAI parameters — strip them before forwarding to
# avoid confusion in raw payload logs and potential future validation.
_POE_IGNORED_PARAMS = frozenset(
    {
        "response_format",
        "seed",
        "frequency_penalty",
        "presence_penalty",
        "logit_bias",
        "user",
        "service_tier",
        "audio",
        "modalities",
    }
)


class PoeProvider(OpenAIProvider):
    """LLMCore provider for the Poe API.

    Extends ``OpenAIProvider`` since the ``/v1/chat/completions`` surface is
    fully OpenAI-compatible.  Adds Poe-specific features:

    - Rich model discovery via ``/v1/models`` (pricing, modalities).
    - Balance checking via ``/usage/current_balance``.
    - Native ``fastapi_poe`` SSE backend for custom parameters & attachments.
    - Video generation awareness (Sora / Veo bots).
    - ``extra_body`` passthrough for Poe-specific bot parameters.

    Configuration (``[providers.poe]``)::

        [providers.poe]
        type = "poe"
        api_key_env_var = "POE_API_KEY"
        default_model = "GPT-4o-Mini"
        # Optional:
        backend = "openai"          # "openai" (default) or "native"
        timeout = 120
    """

    # Poe-specific state
    _backend: str = "openai"
    _poe_api_key: str = ""
    _models_cache: list[dict[str, Any]] | None = None
    _httpx_session: httpx.AsyncClient | None = None

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False):
        """Initialise the PoeProvider.

        Args:
            config: Configuration dictionary from ``[providers.poe]``.
                Required: ``api_key`` or ``api_key_env_var`` or env ``POE_API_KEY``.
                Optional:
                    ``default_model``: Default bot name (default: ``GPT-4o-Mini``).
                    ``backend``: ``"openai"`` (default) or ``"native"``.
                    ``timeout``: Request timeout in seconds (default: 120).
            log_raw_payloads: Whether to log raw request/response payloads.
        """
        # -- Resolve API key ------------------------------------------------
        api_key = config.get("api_key")
        api_key_env_var = config.get("api_key_env_var", "POE_API_KEY")
        if not api_key:
            api_key = os.environ.get(api_key_env_var)
        if not api_key:
            api_key = os.environ.get("POE_API_KEY")

        if not api_key:
            raise ConfigError(
                "Poe API key not found. Set POE_API_KEY environment variable "
                "or configure api_key in [providers.poe]."
            )

        self._poe_api_key = api_key
        self._backend = config.get("backend", "openai")

        # -- Build config for the OpenAI-compatible base --------------------
        openai_config = {
            **config,
            "api_key": api_key,
            "base_url": config.get("base_url", POE_BASE_URL),
            "default_model": config.get("default_model", DEFAULT_POE_MODEL),
            "timeout": config.get("timeout", 120),
        }

        if OpenAIProvider is None or not openai_available:
            raise ImportError(
                "OpenAI library not installed. Poe provider requires the "
                "'openai' package. Install with: pip install llmcore[openai]"
            )

        super().__init__(openai_config, log_raw_payloads)

        # -- Validate native backend availability ---------------------------
        if self._backend == "native" and not _fastapi_poe_available:
            logger.warning(
                "fastapi_poe package not installed. "
                "Falling back to OpenAI-compatible mode.  "
                "Install with: pip install fastapi-poe"
            )
            self._backend = "openai"

        logger.info(
            "PoeProvider initialized (backend=%s, model=%s)",
            self._backend,
            openai_config["default_model"],
        )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        """Returns ``'poe'`` as the provider name."""
        return self._provider_instance_name or "poe"

    # ------------------------------------------------------------------
    # Supported Parameters (override to strip Poe-ignored params)
    # ------------------------------------------------------------------

    def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
        """Return supported parameters, adding ``extra_body`` and removing
        parameters that Poe silently ignores.

        Args:
            model: Model/bot name.  Currently unused (Poe params are uniform).

        Returns:
            Parameter schema dict.
        """
        base = super().get_supported_parameters(model)
        # Remove params Poe ignores
        for p in _POE_IGNORED_PARAMS:
            base.pop(p, None)
        # Ensure extra_body is listed (used for custom bot params)
        base.setdefault("extra_body", {"type": "object"})
        return base

    # ------------------------------------------------------------------
    # Model Discovery
    # ------------------------------------------------------------------

    async def get_models_details(self) -> list[ModelDetails]:
        """Discover available models/bots from the Poe ``/v1/models`` API.

        The listing includes pricing, modalities, and ownership information.

        Returns:
            List of ``ModelDetails`` populated with Poe metadata.
        """
        details_list: list[ModelDetails] = []
        try:
            models_data = await self._fetch_models()
            for model in models_data:
                model_id = model.get("id", "")
                if not model_id:
                    continue

                architecture = model.get("architecture", {}) or {}
                pricing = model.get("pricing", {}) or {}
                input_mods = architecture.get("input_modalities", [])
                output_mods = architecture.get("output_modalities", [])

                supports_vision = "image" in input_mods
                supports_tools = True  # Poe passes tools through

                # Parse pricing (per-token strings)
                input_price_per_token = float(pricing.get("prompt", "0") or "0")
                output_price_per_token = float(pricing.get("completion", "0") or "0")

                details = ModelDetails(
                    id=model_id,
                    display_name=model.get("description", model_id)[:120],
                    context_length=self.get_max_context_length(model_id),
                    supports_streaming=True,
                    supports_tools=supports_tools,
                    supports_vision=supports_vision,
                    provider_name=self.get_name(),
                    metadata={
                        "owned_by": model.get("owned_by"),
                        "input_modalities": input_mods,
                        "output_modalities": output_mods,
                        "modality": architecture.get("modality"),
                        "pricing_per_token": {
                            "input": input_price_per_token,
                            "output": output_price_per_token,
                        },
                        "pricing_per_million": {
                            "input": input_price_per_token * 1_000_000,
                            "output": output_price_per_token * 1_000_000,
                        },
                    },
                )
                details_list.append(details)

            logger.info("Discovered %d models/bots from Poe.", len(details_list))
        except Exception as e:
            logger.error("Failed to list models from Poe: %s", e, exc_info=True)
            raise ProviderError(self.get_name(), f"Failed to list models: {e}")
        return details_list

    async def _fetch_models(self) -> list[dict[str, Any]]:
        """Fetch the model listing from the Poe ``/v1/models`` endpoint.

        Requires authentication (unlike OpenRouter).  Caches the result
        for the lifetime of the provider instance.

        Returns:
            List of model dicts from the API response.
        """
        if self._models_cache is not None:
            return self._models_cache

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    POE_MODELS_URL,
                    headers={"Authorization": f"Bearer {self._poe_api_key}"},
                )
                resp.raise_for_status()
                data = resp.json()

            models = data.get("data", [])
            self._models_cache = models
            logger.debug("Fetched %d models from Poe /v1/models.", len(models))
            return models
        except Exception as e:
            logger.error("Failed to fetch Poe models: %s", e)
            return []

    # ------------------------------------------------------------------
    # Balance Checking (Poe-specific)
    # ------------------------------------------------------------------

    async def get_balance(self) -> int:
        """Fetch the current point balance from Poe.

        Returns:
            Current point balance as an integer.

        Raises:
            ProviderError: If the balance check fails.
        """
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    POE_BALANCE_URL,
                    headers={"Authorization": f"Bearer {self._poe_api_key}"},
                )
                resp.raise_for_status()
                data = resp.json()
            return int(data.get("current_point_balance", 0))
        except Exception as e:
            raise ProviderError(
                self.get_name(),
                f"Failed to check balance: {e}",
            )

    # ------------------------------------------------------------------
    # Context Length
    # ------------------------------------------------------------------

    def get_max_context_length(self, model: str | None = None) -> int:
        """Return the maximum context length for a Poe bot.

        Resolution:
        1. Model Card Registry (``poe`` provider key).
        2. Fallback: 128000 (conservative default for frontier models).

        Args:
            model: Bot name.

        Returns:
            Max input tokens.
        """
        model_name = model or self.default_model

        # 1. Model card registry
        try:
            registry = get_model_card_registry()
            card = registry.get("poe", model_name)
            if card is not None:
                return card.get_context_length()
        except Exception:
            pass

        # 2. Fallback
        return 128000

    # ------------------------------------------------------------------
    # Chat Completion (dual path)
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
        """Send a chat completion request via Poe.

        Dispatches to either the OpenAI-compatible or the native
        ``fastapi_poe`` backend based on the ``backend`` configuration.

        All standard OpenAI parameters are forwarded.  Poe-specific bot
        parameters can be passed via ``extra_body`` (e.g.
        ``extra_body={"reasoning_effort": "high"}``).

        Parameters that Poe ignores (``response_format``, ``seed``, etc.)
        are silently stripped to avoid validation errors.

        Args:
            context: List of LLMCore Message objects.
            model: Poe bot name (e.g., ``"Claude-Sonnet-4.6"``).
            stream: Whether to stream the response.
            tools: Optional list of Tool definitions.
            tool_choice: Tool choice mode.
            **kwargs: Generation parameters + ``extra_body`` for bot params.

        Returns:
            Dict or async generator with OpenAI-normalised response.
        """
        # Strip Poe-ignored params from kwargs to avoid validation errors
        # in the parent OpenAIProvider.get_supported_parameters check.
        for p in list(kwargs.keys()):
            if p in _POE_IGNORED_PARAMS:
                kwargs.pop(p)

        if self._backend == "native" and _fastapi_poe_available:
            return await self._chat_completion_native(
                context, model, stream, tools, tool_choice, **kwargs
            )

        # Default: OpenAI-compatible path via parent
        return await super().chat_completion(context, model, stream, tools, tool_choice, **kwargs)

    # ------------------------------------------------------------------
    # Native fastapi_poe Backend
    # ------------------------------------------------------------------

    async def _chat_completion_native(
        self,
        context: ContextPayload,
        model: str | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Chat completion using the native ``fastapi_poe`` SSE client.

        Falls back to the OpenAI-compatible path on any error.

        Args:
            context: List of LLMCore Message objects.
            model: Poe bot name.
            stream: Whether to stream.
            tools: Tool definitions.
            tool_choice: Tool choice mode (ignored by native path).
            **kwargs: Additional parameters.

        Returns:
            OpenAI-normalised response dict or async generator.
        """
        bot_name = model or self.default_model

        # Convert LLMCore messages → Poe ProtocolMessages
        poe_messages: list[Any] = []
        for msg in context:
            role_str = msg.role if isinstance(msg.role, str) else msg.role.value
            poe_msg = PoeProtocolMessage(role=role_str, content=msg.content)
            poe_messages.append(poe_msg)

        # Convert tools to Poe format
        poe_tools = None
        if tools:
            poe_tools = [
                PoeToolDefinition(
                    type="function",
                    function={
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                )
                for t in tools
            ]

        try:
            if stream:
                return self._native_stream_wrapper(poe_messages, bot_name, poe_tools, **kwargs)
            else:
                return await self._native_non_stream(poe_messages, bot_name, poe_tools, **kwargs)
        except Exception as e:
            logger.warning(
                "Poe native backend error: %s. Falling back to OpenAI mode.",
                e,
                exc_info=True,
            )
            # Fallback to OpenAI-compatible
            return await OpenAIProvider.chat_completion(
                self, context, model, stream, tools, tool_choice, **kwargs
            )

    async def _native_non_stream(
        self,
        messages: list[Any],
        bot_name: str,
        tools: list[Any] | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Collect all native SSE chunks into a single OpenAI-format response."""
        chunks: list[str] = []
        async for partial in get_bot_response(
            messages=messages,
            bot_name=bot_name,
            api_key=self._poe_api_key,
            tools=tools,
            temperature=kwargs.get("temperature"),
            base_url=POE_NATIVE_BASE_URL,
        ):
            if hasattr(partial, "is_suggested_reply") and partial.is_suggested_reply:
                continue
            if hasattr(partial, "is_replace_response") and partial.is_replace_response:
                chunks.clear()
            chunks.append(partial.text)

        content = "".join(chunks)
        return {
            "id": f"poe-native-{id(content)}",
            "object": "chat.completion",
            "model": bot_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    async def _native_stream_wrapper(
        self,
        messages: list[Any],
        bot_name: str,
        tools: list[Any] | None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Wrap native SSE partials into OpenAI-format stream chunks."""

        async def _gen() -> AsyncGenerator[dict[str, Any], None]:
            async for partial in get_bot_response(
                messages=messages,
                bot_name=bot_name,
                api_key=self._poe_api_key,
                tools=tools,
                temperature=kwargs.get("temperature"),
                base_url=POE_NATIVE_BASE_URL,
            ):
                if hasattr(partial, "is_suggested_reply") and partial.is_suggested_reply:
                    continue
                yield {
                    "id": f"poe-native-stream",
                    "object": "chat.completion.chunk",
                    "model": bot_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": partial.text},
                            "finish_reason": None,
                        }
                    ],
                }
            # Final done chunk
            yield {
                "id": f"poe-native-stream",
                "object": "chat.completion.chunk",
                "model": bot_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }

        return _gen()

    # ------------------------------------------------------------------
    # Token Counting
    # ------------------------------------------------------------------

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """Estimate token count using tiktoken (cl100k_base).

        Poe does not expose a tokeniser — we use tiktoken as a reasonable
        approximation since most Poe-hosted models use BPE tokenisers with
        similar token-to-character ratios.

        Args:
            text: Text to count.
            model: Ignored (all models use cl100k_base approximation).

        Returns:
            Estimated token count.
        """
        return await super().count_tokens(text, model)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the OpenAI client and any native httpx sessions."""
        if self._httpx_session:
            try:
                await self._httpx_session.aclose()
            except Exception as e:
                logger.debug("Error closing Poe httpx session: %s", e)
            self._httpx_session = None

        await super().close()
        logger.debug("PoeProvider closed.")
