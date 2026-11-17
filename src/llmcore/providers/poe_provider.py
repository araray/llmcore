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

Key characteristics of the Poe API (live, 2026-11-16):
- Uses **bot names** as model IDs, lowercase since 2026 (``gpt-5.4-mini``,
  ``claude-haiku-4.5``; matching is case-insensitive).
- ``/v1/models`` lists each bot's context window, reasoning support, custom
  ``parameters`` (with JSON schemas), supported features and endpoints.
- ``effort=`` maps onto whichever reasoning parameter a bot declares:
  ``reasoning_effort`` / ``output_effort`` / ``thinking_level`` (enums),
  ``thinking_budget`` (scaled within its range) or ``enable_thinking``.
  ``native_search=True`` sets a bot's ``web_search`` parameter.
- Point-based billing; per-token pricing exposed via ``/v1/models``.
- Rate limit: 500 RPM (request-based, no token-based limits).
- Ignored parameters (response_format, seed, penalties, store, ...) are
  stripped; a 413 is a context overflow; 529 means overloaded.

Audited against the live API and fastapi_poe 0.0.83 on 2026-11-16.

References:
    https://creator.poe.com/api-reference/overview
    https://creator.poe.com/docs/poe-protocol-specification
"""

from __future__ import annotations

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
from ..models import ModelDetails, Tool
from .base import ContextPayload, flatten_tool_messages_for_text_protocol

# Inherit from OpenAIProvider for the OpenAI-compatible path.
try:
    from .openai_provider import OpenAIProvider, nearest_effort, openai_available
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
DEFAULT_POE_MODEL = "gpt-5.4-mini"

#: Reasoning parameters a bot may declare, in the order effort= prefers them.
_EFFORT_PARAMS = ("reasoning_effort", "output_effort", "thinking_level", "thinking_budget", "enable_thinking")
#: Share of a thinking_budget range each effort level gets.
_BUDGET_SHARE = {"none": 0.0, "minimal": 0.05, "low": 0.25, "medium": 0.5, "high": 0.75, "xhigh": 0.9, "max": 1.0}

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
        # Also documented as ignored (2026-11).
        "store",
        "metadata",
        "prediction",
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
        default_model = "gpt-5.4-mini"
        # Optional:
        backend = "openai"          # "openai" (default) or "native"
        timeout = 120
    """

    #: Poe's media bots are reached as chat *bots*, not through /v1/images or
    #: /v1/audio, so none of the OpenAI media endpoints exist here. Surfacing
    #: them through the media routers is tracked separately.
    _MEDIA_CAPABILITIES: frozenset[Any] = frozenset()


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
                    ``default_model``: Default bot name (default: ``gpt-5.4-mini``).
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

    def _poe_parameters(self, model: str | None = None) -> dict[str, Any]:
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
                window = model.get("context_window") or {}
                features = model.get("supported_features") or []

                supports_vision = "image" in input_mods
                supports_tools = "tools" in features

                # Parse pricing (per-token strings)
                input_price_per_token = float(pricing.get("prompt", "0") or "0")
                output_price_per_token = float(pricing.get("completion", "0") or "0")

                created = model.get("created")
                details = ModelDetails(
                    id=model_id,
                    display_name=(model.get("metadata") or {}).get("display_name")
                    or (model.get("description") or model_id)[:120],
                    context_length=self.get_max_context_length(model_id),
                    max_output_tokens=window.get("max_output_tokens"),
                    supports_streaming=True,
                    supports_tools=supports_tools,
                    supports_vision=supports_vision,
                    supports_reasoning=bool(model.get("reasoning")),
                    provider_name=self.get_name(),
                    metadata={
                        "owned_by": model.get("owned_by"),
                        # Poe reports milliseconds.
                        "created": int(created) // 1000 if isinstance(created, (int, float)) else created,
                        "description": model.get("description"),
                        "reasoning": model.get("reasoning"),
                        "parameters": model.get("parameters") or [],
                        "supported_features": features,
                        "supported_endpoints": model.get("supported_endpoints") or [],
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

    def _listed(self, bot: str) -> dict[str, Any] | None:
        """The cached listing entry for ``bot`` (ids are lowercase)."""
        wanted = bot.lower()
        for entry in self._models_cache or []:
            if str(entry.get("id", "")).lower() == wanted:
                return entry
        return None

    # ------------------------------------------------------------------
    # Balance Checking (Poe-specific)
    # ------------------------------------------------------------------

    async def get_balance(self) -> int:
        """The current point balance (see :meth:`get_balance_details`)."""
        return int((await self.get_balance_details()).get("current_point_balance", 0))

    async def get_balance_details(self) -> dict[str, Any]:
        """``GET /usage/current_balance``: points by source, USD values,
        the next daily grant and auto-recharge settings.

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
                return dict(resp.json())
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

        The fallback stays here even though cards no longer invent a window,
        because the two answer different questions. A card states what is
        *known* about a model, and for an image or audio bot the honest answer
        is nothing. This method is a **budgeting assumption** with an ``int``
        contract -- callers use it to decide how much history to send -- so it
        has to produce a number. The distinction matters: the card does not
        claim 128,000, and nothing reading cards will believe it does.

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
                # `get_context_length()` returns None for a card that states
                # no window. Returning that would break this method's `int`
                # contract -- which is how it broke `ModelDetails` once.
                known = card.get_context_length()
                if known:
                    return known
        except Exception:
            pass

        # 2. The live listing's context window
        listed = self._listed(model_name)
        window = (listed or {}).get("context_window") or {}
        known = window.get("context_length") or (listed or {}).get("context_length")
        if known:
            return int(known)

        # 3. Fallback
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
        kwargs = self._bot_parameters(model or self.default_model, kwargs)

        if self._backend == "native" and _fastapi_poe_available:
            return await self._chat_completion_native(
                context, model, stream, tools, tool_choice, **kwargs
            )

        # Default: OpenAI-compatible path via parent
        return await super().chat_completion(context, model, stream, tools, tool_choice, **kwargs)

    def _bot_parameters(self, bot: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Map ``effort`` and ``native_search`` onto the bot's own parameters.

        Each bot declares its custom parameters with a JSON schema in
        ``/v1/models``. ``effort`` goes to the first reasoning parameter the bot
        has (enum values snap to the nearest; a ``thinking_budget`` is scaled
        within its range); ``native_search`` sets ``web_search``. Unlisted
        bots get ``reasoning_effort``, which Poe honours (measured: "minimal"
        0 reasoning tokens, "high" 52 on gpt-5.4-nano). Both travel in
        ``extra_body``.
        """
        kwargs = dict(kwargs)
        effort = kwargs.pop("effort", None)
        search = kwargs.pop("native_search", False)
        if effort is None and not search:
            return kwargs
        listed = self._listed(bot)
        params = {p.get("name"): p for p in (listed or {}).get("parameters") or [] if isinstance(p, dict)}
        extra = dict(kwargs.get("extra_body") or {})
        if effort is not None:
            requested = str(effort).strip().lower()
            if listed is None:
                extra.setdefault("reasoning_effort", requested)
            for name in _EFFORT_PARAMS:
                spec = params.get(name)
                if spec is None or name in extra:
                    continue
                schema = spec.get("schema") or {}
                if schema.get("enum"):
                    extra[name] = requested if requested in schema["enum"] else nearest_effort(requested, schema["enum"])
                elif schema.get("type") == "number":
                    low = float(schema.get("minimum", 0))
                    high = float(schema.get("maximum", low))
                    extra[name] = int(low + (high - low) * _BUDGET_SHARE.get(requested, 0.5))
                elif schema.get("type") == "boolean":
                    extra[name] = requested != "none"
                break
            else:
                if listed is not None:
                    logger.debug("Poe bot %s declares no reasoning parameter; effort ignored.", bot)
        if search:
            if listed is None or "web_search" in params:
                extra["web_search"] = True
            else:
                logger.debug("Poe bot %s has no web_search parameter; native_search ignored.", bot)
        if extra:
            kwargs["extra_body"] = extra
        return kwargs

    def supports_native_search(self, model: str | None = None) -> bool:
        """True for bots that declare a ``web_search`` parameter."""
        listed = self._listed(model or self.default_model)
        return bool(listed) and any(
            p.get("name") == "web_search" for p in listed.get("parameters") or [] if isinstance(p, dict)
        )

    def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:  # type: ignore[override]
        params = self._poe_parameters(model)
        params["effort"] = {
            "type": "string",
            "enum": ["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        }
        params["native_search"] = {"type": "boolean"}
        return params

    def _is_context_overflow(self, status: int | None, text: str, code: str | None) -> bool:
        """Poe answers an over-long request with 413 ``request_too_large``."""
        if status == 413 or code == "request_too_large":
            return True
        return super()._is_context_overflow(status, text, code)

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

        # Poe's protocol has no tool role — render tool results as user text
        # (R-2 fallback). The OpenAI-compatible path maps them natively.
        context = flatten_tool_messages_for_text_protocol(list(context))

        try:
            # The native protocol's roles are system/user/bot/tool: an
            # "assistant" turn failed validation before the fallback could run.
            poe_messages: list[Any] = []
            for msg in context:
                role_str = msg.role if isinstance(msg.role, str) else msg.role.value
                if role_str == "assistant":
                    role_str = "bot"
                parameters = (kwargs.get("extra_body") or {}) if role_str == "user" else {}
                poe_messages.append(
                    PoeProtocolMessage(role=role_str, content=msg.content, parameters=parameters)
                )

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

            if stream:
                # Pull the first chunk now: an error inside a lazy generator
                # would escape the fallback below.
                return await self._primed(
                    await self._native_stream_wrapper(poe_messages, bot_name, poe_tools, **kwargs)
                )
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
            stop_sequences=kwargs.get("stop"),
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
                stop_sequences=kwargs.get("stop"),
                base_url=POE_NATIVE_BASE_URL,
            ):
                if hasattr(partial, "is_suggested_reply") and partial.is_suggested_reply:
                    continue
                if getattr(partial, "is_replace_response", False):
                    # The bot replaced everything said so far; a stream can
                    # only say so, not unsay it.
                    logger.debug("Poe bot %s replaced its response mid-stream.", bot_name)
                yield {
                    "id": "poe-native-stream",
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
                "id": "poe-native-stream",
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
