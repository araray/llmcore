# src/llmcore/providers/openrouter_provider.py
"""
LLMCore provider for OpenRouter — a unified gateway to 300+ AI models.

OpenRouter's API is fully OpenAI-compatible (same ``/chat/completions``
request/response format), so this provider **subclasses OpenAIProvider**
and overrides only what differs:

- Authentication: ``OPENROUTER_API_KEY`` (env) or config key
- Base URL: ``https://openrouter.ai/api/v1``
- Model IDs: ``provider/model-name`` format (e.g., ``openai/gpt-4o``)
- Custom headers: ``HTTP-Referer``, ``X-Title`` for rankings
- Model discovery: Rich ``/models`` endpoint with pricing + modalities
- Provider preferences: Route to specific upstream providers
- Multi-model routing: ``models=[...]`` for automatic fallback

Dual backend support:
- **httpx mode** (default): Uses the ``openai`` Python SDK with
  ``base_url`` override — no extra dependencies beyond ``openai``.
- **SDK mode**: Uses the ``openrouter`` Python SDK when installed
  and ``backend: sdk`` is configured.

Tested against OpenRouter API v1 / openrouter SDK v0.8.1.
"""

import json
import logging
import os
from collections.abc import AsyncGenerator
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# --- Optional SDK imports ---
openrouter_sdk_available = False
try:
    import openrouter as openrouter_sdk

    openrouter_sdk_available = True
except ImportError:
    openrouter_sdk = None  # type: ignore

from ..exceptions import ConfigError, ProviderError
from ..models import Message, ModelDetails, Tool
from ..models import Role as LLMCoreRole
from .base import BaseProvider, ContextPayload

# We inherit from OpenAIProvider for the core chat logic.
# Import is deferred to avoid circular imports if needed.
try:
    from .openai_provider import OpenAIProvider, openai_available
except ImportError:
    OpenAIProvider = None  # type: ignore
    openai_available = False

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"


class OpenRouterProvider(OpenAIProvider):
    """
    LLMCore provider for OpenRouter.

    Extends ``OpenAIProvider`` since the API is fully OpenAI-compatible.
    Adds OpenRouter-specific features: rich model discovery, provider
    preferences, multi-model routing, and custom ranking headers.

    Configuration (``[providers.openrouter]``)::

        [providers.openrouter]
        type = "openrouter"
        api_key_env_var = "OPENROUTER_API_KEY"
        default_model = "openai/gpt-4o-mini"
        # Optional OpenRouter-specific settings:
        app_url = "https://myapp.com"          # HTTP-Referer for rankings
        app_title = "My App"                   # X-Title for dashboard
        backend = "openai"                     # "openai" (default) or "sdk"
        timeout = 120
    """

    # OpenRouter-specific state
    _app_url: str | None = None
    _app_title: str | None = None
    _backend: str = "openai"  # "openai" or "sdk"
    _openrouter_client: Any | None = None  # openrouter.OpenRouter SDK client
    _models_cache: list[dict[str, Any]] | None = None

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False):
        """
        Initializes the OpenRouterProvider.

        Args:
            config: Configuration dictionary from ``[providers.openrouter]``.
                Required: ``api_key`` or ``api_key_env_var`` or env
                    ``OPENROUTER_API_KEY``.
                Optional:
                    ``default_model``: Default model (default: ``openai/gpt-4o-mini``).
                    ``app_url``: Your app URL for OpenRouter rankings.
                    ``app_title``: Your app name for the OpenRouter dashboard.
                    ``backend``: ``"openai"`` (default) or ``"sdk"``.
                    ``timeout``: Request timeout in seconds (default: 120).
            log_raw_payloads: Whether to log raw request/response payloads.
        """
        # Resolve API key before calling super().__init__
        api_key = config.get("api_key")
        api_key_env_var = config.get("api_key_env_var", "OPENROUTER_API_KEY")
        if not api_key:
            api_key = os.environ.get(api_key_env_var)
        if not api_key:
            api_key = os.environ.get("OPENROUTER_API_KEY")

        if not api_key:
            raise ConfigError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY "
                "environment variable or configure api_key in config."
            )

        # Store OpenRouter-specific config before calling super
        self._app_url = config.get("app_url")
        self._app_title = config.get("app_title")
        self._backend = config.get("backend", "openai")

        # Build config for OpenAIProvider base class
        openai_config = {
            **config,
            "api_key": api_key,
            "base_url": config.get("base_url", OPENROUTER_BASE_URL),
            "default_model": config.get("default_model", DEFAULT_OPENROUTER_MODEL),
            "timeout": config.get("timeout", 120),
        }

        # Initialize the OpenAI-compatible base
        if OpenAIProvider is None or not openai_available:
            raise ImportError(
                "OpenAI library not installed. OpenRouter provider requires "
                "'openai' package. Install with 'pip install llmcore[openai]'."
            )

        super().__init__(openai_config, log_raw_payloads)

        # Set OpenRouter-specific headers on the underlying httpx client
        if self._client and hasattr(self._client, "_custom_headers"):
            extra_headers = {}
            if self._app_url:
                extra_headers["HTTP-Referer"] = self._app_url
            if self._app_title:
                extra_headers["X-Title"] = self._app_title
            if extra_headers:
                self._client = self._client.with_options(
                    default_headers=extra_headers
                )

        # Optionally initialize the native OpenRouter SDK client
        if self._backend == "sdk":
            if not openrouter_sdk_available:
                logger.warning(
                    "OpenRouter SDK ('openrouter' package) not installed. "
                    "Falling back to OpenAI-compatible mode."
                )
                self._backend = "openai"
            else:
                try:
                    self._openrouter_client = openrouter_sdk.OpenRouter(
                        api_key=api_key,
                        http_referer=self._app_url,
                        x_open_router_title=self._app_title,
                    )
                    logger.debug("OpenRouter SDK client initialized.")
                except Exception as e:
                    logger.warning(
                        f"Failed to init OpenRouter SDK client: {e}. "
                        f"Falling back to OpenAI-compatible mode."
                    )
                    self._backend = "openai"

        logger.info(
            f"OpenRouterProvider initialized (backend={self._backend}, "
            f"model={self.default_model})"
        )

    def get_name(self) -> str:
        """Returns 'openrouter' as the provider name."""
        return self._provider_instance_name or "openrouter"

    # ------------------------------------------------------------------
    # Model Discovery (overrides OpenAIProvider)
    # ------------------------------------------------------------------

    async def get_models_details(self) -> list[ModelDetails]:
        """Discovers available models from the OpenRouter ``/models`` API.

        OpenRouter's model listing is much richer than OpenAI's — it includes
        pricing, input/output modalities, context lengths, provider info,
        supported parameters, and knowledge cutoff dates.

        Returns:
            List of ``ModelDetails`` populated with OpenRouter metadata.
        """
        details_list = []
        try:
            models_data = await self._fetch_models()
            for model in models_data:
                model_id = model.get("id", "")
                if not model_id:
                    continue

                context_length = model.get("context_length") or 4096
                top_provider = model.get("top_provider", {}) or {}
                max_output = top_provider.get("max_completion_tokens")
                architecture = model.get("architecture", {}) or {}
                input_modalities = architecture.get("input_modalities", [])
                output_modalities = architecture.get("output_modalities", [])
                supported_params = model.get("supported_parameters", [])

                # Derive capabilities from modalities and parameters
                supports_vision = "image" in input_modalities
                supports_audio_in = "audio" in input_modalities
                supports_tools = "tools" in supported_params
                supports_reasoning = (
                    "reasoning" in supported_params
                    or "include_reasoning" in supported_params
                )
                supports_json = (
                    "response_format" in supported_params
                    or "structured_outputs" in supported_params
                )

                # Parse pricing (OpenRouter prices are per-token strings)
                pricing = model.get("pricing", {}) or {}
                input_price_per_token = float(pricing.get("prompt", "0") or "0")
                output_price_per_token = float(
                    pricing.get("completion", "0") or "0"
                )

                details = ModelDetails(
                    id=model_id,
                    display_name=model.get("name"),
                    context_length=int(context_length),
                    max_output_tokens=int(max_output) if max_output else None,
                    supports_streaming=True,
                    supports_tools=supports_tools,
                    supports_vision=supports_vision,
                    supports_reasoning=supports_reasoning,
                    provider_name=self.get_name(),
                    metadata={
                        "display_name": model.get("name"),
                        "description": model.get("description"),
                        "input_modalities": input_modalities,
                        "output_modalities": output_modalities,
                        "supported_parameters": supported_params,
                        "pricing_per_token": {
                            "input": input_price_per_token,
                            "output": output_price_per_token,
                        },
                        "pricing_per_million": {
                            "input": input_price_per_token * 1_000_000,
                            "output": output_price_per_token * 1_000_000,
                        },
                        "knowledge_cutoff": model.get("knowledge_cutoff"),
                        "expiration_date": model.get("expiration_date"),
                        "is_moderated": top_provider.get("is_moderated", False),
                        "canonical_slug": model.get("canonical_slug"),
                    },
                )
                details_list.append(details)

            logger.info(
                f"Discovered {len(details_list)} models from OpenRouter."
            )
        except Exception as e:
            logger.error(
                f"Failed to list models from OpenRouter: {e}", exc_info=True
            )
            raise ProviderError(self.get_name(), f"Failed to list models: {e}")
        return details_list

    async def _fetch_models(self) -> list[dict[str, Any]]:
        """Fetch the model listing from OpenRouter API.

        Uses a simple httpx GET (no auth required for model listing).
        Caches the result for the lifetime of the provider instance.

        Returns:
            List of model dicts from the ``/models`` endpoint.
        """
        if self._models_cache is not None:
            return self._models_cache

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(OPENROUTER_MODELS_URL)
                resp.raise_for_status()
                data = resp.json()

            models = data.get("data", [])
            self._models_cache = models
            logger.debug(f"Fetched {len(models)} models from OpenRouter API.")
            return models
        except Exception as e:
            logger.error(f"Failed to fetch OpenRouter models: {e}")
            return []

    # ------------------------------------------------------------------
    # Context Length (overrides OpenAIProvider)
    # ------------------------------------------------------------------

    def get_max_context_length(self, model: str | None = None) -> int:
        """Returns the maximum context length for an OpenRouter model.

        Resolution:
        1. Model Card Registry (``openrouter`` provider key)
        2. Cached model listing from API
        3. Fallback: 128000
        """
        model_name = model or self.default_model

        # 1. Model card registry
        try:
            from ..model_cards.registry import get_model_card_registry

            registry = get_model_card_registry()
            card = registry.get("openrouter", model_name)
            if card is not None:
                return card.get_context_length()
        except Exception:
            pass

        # 2. Cached models from API
        if self._models_cache:
            for m in self._models_cache:
                if m.get("id") == model_name:
                    ctx = m.get("context_length")
                    if ctx:
                        return int(ctx)

        # 3. Fallback
        return 128000

    # ------------------------------------------------------------------
    # Chat Completion (extends OpenAIProvider with OpenRouter features)
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
        """Sends a chat completion request via OpenRouter.

        Supports all standard OpenAI parameters plus OpenRouter extensions:

        - ``provider``: Dict with provider preferences (``order``,
          ``allow_fallbacks``, ``require``, ``ignore``, etc.)
        - ``models``: List of model IDs for automatic fallback routing.
        - ``reasoning``: Dict with reasoning config (``effort``, ``max_tokens``).

        These are extracted from ``**kwargs`` and passed as top-level
        request body fields.

        Args:
            context: List of LLMCore Message objects.
            model: OpenRouter model ID (e.g., ``"anthropic/claude-sonnet-4"``).
            stream: If True, returns an async generator.
            tools: Optional list of Tool definitions.
            tool_choice: Tool choice mode.
            **kwargs: Standard generation params + OpenRouter extensions.

        Returns:
            Dict or async generator with OpenAI-normalized response.
        """
        if self._backend == "sdk" and self._openrouter_client:
            return await self._chat_completion_sdk(
                context, model, stream, tools, tool_choice, **kwargs
            )

        # Default: use the inherited OpenAI-compatible path
        # Extract OpenRouter-specific kwargs that need special handling
        # (these are top-level body fields, not generation params)
        openrouter_kwargs = {}
        for key in ("provider", "models", "reasoning"):
            if key in kwargs:
                openrouter_kwargs[key] = kwargs.pop(key)

        # The parent OpenAIProvider.chat_completion handles the rest
        # OpenRouter-specific fields need to go into extra_body
        if openrouter_kwargs:
            existing_extra = kwargs.get("extra_body", {})
            if isinstance(existing_extra, dict):
                existing_extra.update(openrouter_kwargs)
            else:
                existing_extra = openrouter_kwargs
            kwargs["extra_body"] = existing_extra

        return await super().chat_completion(
            context, model, stream, tools, tool_choice, **kwargs
        )

    async def _chat_completion_sdk(
        self,
        context: ContextPayload,
        model: str | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Chat completion using the native OpenRouter SDK.

        Falls back to OpenAI-compatible mode on any SDK error.

        Args:
            context: List of LLMCore Message objects.
            model: Model identifier.
            stream: Whether to stream.
            tools: Tool definitions.
            tool_choice: Tool choice mode.
            **kwargs: Additional parameters.

        Returns:
            OpenAI-normalized response dict or async generator.
        """
        if not self._openrouter_client:
            logger.warning("OpenRouter SDK client not available, falling back.")
            return await super().chat_completion(
                context, model, stream, tools, tool_choice, **kwargs
            )

        model_name = model or self.default_model

        # Convert LLMCore messages to dicts
        messages = []
        for msg in context:
            msg_dict: dict[str, Any] = {
                "role": msg.role if isinstance(msg.role, str) else msg.role.value,
                "content": msg.content,
            }
            if msg.role == LLMCoreRole.TOOL and msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
            messages.append(msg_dict)

        # Convert tools to OpenAI function format
        sdk_tools = None
        if tools:
            sdk_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]

        try:
            # Use SDK's async method
            sdk_kwargs: dict[str, Any] = {
                "messages": messages,
                "model": model_name,
                "stream": stream,
            }
            if sdk_tools:
                sdk_kwargs["tools"] = sdk_tools
            if tool_choice:
                sdk_kwargs["tool_choice"] = tool_choice

            # Pass through standard params
            for key in (
                "temperature",
                "top_p",
                "max_tokens",
                "max_completion_tokens",
                "frequency_penalty",
                "presence_penalty",
                "seed",
                "stop",
                "response_format",
            ):
                if key in kwargs:
                    sdk_kwargs[key] = kwargs[key]

            # OpenRouter-specific params
            for key in ("provider", "models", "reasoning"):
                if key in kwargs:
                    sdk_kwargs[key] = kwargs[key]

            result = await self._openrouter_client.chat.send_async(**sdk_kwargs)

            if stream:
                # Wrap SDK EventStream into our async generator format
                async def sdk_stream_wrapper():
                    async for chunk in result:
                        yield {
                            "choices": [
                                {
                                    "delta": {
                                        "content": getattr(
                                            getattr(
                                                chunk.choices[0], "delta", None
                                            )
                                            if chunk.choices
                                            else None,
                                            "content",
                                            "",
                                        )
                                        or ""
                                    }
                                }
                            ]
                        }

                return sdk_stream_wrapper()
            else:
                # Normalize SDK response to dict
                if hasattr(result, "model_dump"):
                    return result.model_dump()
                elif hasattr(result, "dict"):
                    return result.dict()
                else:
                    # Already a dict-like ChatResult
                    return {
                        "id": getattr(result, "id", None),
                        "model": getattr(result, "model", model_name),
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": result.choices[0].message.content
                                    if result.choices
                                    else "",
                                },
                                "finish_reason": getattr(
                                    result.choices[0], "finish_reason", None
                                )
                                if result.choices
                                else None,
                            }
                        ],
                        "usage": {
                            "prompt_tokens": getattr(
                                result.usage, "prompt_tokens", 0
                            )
                            if result.usage
                            else 0,
                            "completion_tokens": getattr(
                                result.usage, "completion_tokens", 0
                            )
                            if result.usage
                            else 0,
                            "total_tokens": getattr(
                                result.usage, "total_tokens", 0
                            )
                            if result.usage
                            else 0,
                        },
                    }
        except Exception as e:
            logger.error(
                f"OpenRouter SDK error: {e}. Falling back to OpenAI mode.",
                exc_info=True,
            )
            return await super().chat_completion(
                context, model, stream, tools, tool_choice, **kwargs
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close both the OpenAI and OpenRouter SDK clients."""
        # Close OpenRouter SDK client if present
        if self._openrouter_client:
            try:
                if hasattr(self._openrouter_client, "__aexit__"):
                    await self._openrouter_client.__aexit__(None, None, None)
                elif hasattr(self._openrouter_client, "close"):
                    self._openrouter_client.close()
            except Exception as e:
                logger.debug(f"Error closing OpenRouter SDK client: {e}")
            self._openrouter_client = None

        # Close the OpenAI base client
        await super().close()
        logger.debug("OpenRouterProvider closed.")
