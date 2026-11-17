# src/llmcore/providers/openrouter_provider.py
"""
LLMCore provider for OpenRouter — a unified gateway to 300+ AI models.

OpenRouter's API is fully OpenAI-compatible (same ``/chat/completions``
request/response format), so this provider **subclasses OpenAIProvider**
and overrides only what differs:

- Authentication: ``OPENROUTER_API_KEY`` (env) or config key
- Base URL: ``https://openrouter.ai/api/v1``
- Model IDs: ``provider/model-name`` format (e.g., ``openai/gpt-5.4-mini``)
- Custom headers: ``HTTP-Referer``, ``X-OpenRouter-Title`` (+ legacy
  ``X-Title``), ``X-OpenRouter-Categories`` for rankings
- Model discovery: the full paginated ``/models`` catalogue (every output
  modality), with each model's reasoning efforts, used to clamp ``effort``
- Provider preferences (``provider=``, or ``provider_preferences`` in config,
  e.g. ``{"zdr": true, "data_collection": "deny"}``), multi-model fallback
  (``models=[...]``), plugins, prompt caching and the other extensions in the
  OpenRouter vendor profile (``openai_provider._VENDOR_PROFILES``)

Shared with the other OpenAI-compatible vendors (OpenAIProvider): the
``reasoning_details`` of each assistant turn are returned by
``assistant_replay_metadata()`` and replayed unmodified (Gemini 3 and Claude
send encrypted reasoning only), failures reported inside an HTTP 200 raise,
``native_search=True`` adds the ``web`` plugin, and usage carries OpenRouter's
``cost``.

Backends:
- **openai** (default): the ``openai`` SDK (or ``transport = "httpx"``).
- **sdk**: the ``openrouter`` SDK (1.3.x), sending the same request body.

Audited against the live API and openrouter SDK 1.3.179 on 2026-11-16.
"""

import inspect
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
from ..models import ModelDetails, Tool
from .base import ContextPayload

# We inherit from OpenAIProvider for the core chat logic.
# Import is deferred to avoid circular imports if needed.
try:
    from .openai_provider import OpenAIProvider, nearest_effort, openai_available
except ImportError:
    OpenAIProvider = None  # type: ignore
    openai_available = False

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-5.4-mini"


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
        default_model = "openai/gpt-5.4-mini"
        # Optional OpenRouter-specific settings:
        app_url = "https://myapp.com"          # HTTP-Referer for rankings
        app_title = "My App"                   # X-OpenRouter-Title for dashboard
        app_categories = "cli-agent"           # X-OpenRouter-Categories
        provider_preferences = { zdr = true }  # merged into every request
        backend = "openai"                     # "openai" (default) or "sdk"
        timeout = 120
    """

    #: OpenRouter is a chat-completions gateway; it exposes no image or audio
    #: endpoints of its own.
    _MEDIA_CAPABILITIES: frozenset[Any] = frozenset()


    # OpenRouter-specific state
    _app_url: str | None = None
    _app_title: str | None = None
    _backend: str = "openai"  # "openai" or "sdk"
    _openrouter_client: Any | None = None  # openrouter.OpenRouter SDK client
    _models_cache: list[dict[str, Any]] | None = None
    _app_categories: str | None = None
    _provider_preferences: dict[str, Any] = {}  # replaced per instance

    def _listed(self, model_id: str) -> dict[str, Any] | None:
        """The cached listing entry for ``model_id`` (indexed lazily)."""
        by_id = getattr(self, "_models_by_id", None)
        if not by_id and self._models_cache:
            by_id = self._models_by_id = {m.get("id"): m for m in self._models_cache if m.get("id")}
        return (by_id or {}).get(model_id)

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False):
        """
        Initializes the OpenRouterProvider.

        Args:
            config: Configuration dictionary from ``[providers.openrouter]``.
                Required: ``api_key`` or ``api_key_env_var`` or env
                    ``OPENROUTER_API_KEY``.
                Optional:
                    ``default_model``: Default model (default: ``openai/gpt-5.4-mini``).
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
        self._app_categories = config.get("app_categories")
        self._backend = config.get("backend", "openai")
        self._provider_preferences = dict(config.get("provider_preferences") or {})
        self._models_by_id: dict[str, dict[str, Any]] = {}

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
                # X-OpenRouter-Title is current; X-Title is still honoured.
                extra_headers["X-OpenRouter-Title"] = self._app_title
                extra_headers["X-Title"] = self._app_title
            if self._app_categories:
                extra_headers["X-OpenRouter-Categories"] = str(self._app_categories)
            if extra_headers:
                self._client = self._client.with_options(
                    default_headers=extra_headers
                )
                # The direct transport has to send these too, or the two
                # transports would identify this app differently to OpenRouter.
                self._direct_headers.update(extra_headers)

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
                    sdk_kwargs: dict[str, Any] = {
                        "api_key": api_key,
                        "http_referer": self._app_url,
                        "x_open_router_title": self._app_title,
                        "timeout_ms": int(float(openai_config["timeout"]) * 1000),
                    }
                    if self._app_categories:
                        sdk_kwargs["x_open_router_categories"] = str(self._app_categories)
                    if openai_config["base_url"].rstrip("/") != OPENROUTER_BASE_URL:
                        sdk_kwargs["server_url"] = openai_config["base_url"]
                    self._openrouter_client = openrouter_sdk.OpenRouter(**sdk_kwargs)
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

                context_length = model.get("context_length") or self.get_max_context_length(model_id)
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
                reasoning = model.get("reasoning") or {}
                supports_reasoning = bool(reasoning) or (
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
                        "pricing": pricing,
                        "reasoning": reasoning,
                        "supports_audio_input": supports_audio_in,
                        "supports_structured_output": supports_json,
                        "endpoints": model.get("endpoints"),
                        "default_parameters": model.get("default_parameters"),
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
        """The whole OpenRouter catalogue (no auth needed).

        ``/models`` is paginated (``limit`` up to 1000, ``offset``,
        ``total_count``) and lists only text-output models unless asked for
        ``output_modalities=all``: 468 of 724 models on 2026-11-16. Cached for
        the provider's lifetime; a failure raises instead of returning ``[]``.
        """
        if self._models_cache is not None:
            return self._models_cache
        models: list[dict[str, Any]] = []
        offset = 0
        async with httpx.AsyncClient(timeout=60) as client:
            while True:
                resp = await client.get(
                    OPENROUTER_MODELS_URL,
                    params={"output_modalities": "all", "limit": 1000, "offset": offset},
                )
                resp.raise_for_status()
                data = resp.json()
                page = data.get("data") or []
                models.extend(page)
                total = data.get("total_count")
                offset += len(page)
                if not page or total is None or offset >= int(total):
                    break
        self._models_cache = models
        self._models_by_id = {m.get("id"): m for m in models if m.get("id")}
        logger.debug("Fetched %d models from OpenRouter.", len(models))
        return models

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
        cached = self._listed(model_name)
        if cached and cached.get("context_length"):
            return int(cached["context_length"])

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
        kwargs = self._shape_request(model or self.default_model, kwargs)
        if self._backend == "sdk" and self._openrouter_client:
            return await self._chat_completion_sdk(
                context, model, stream, tools, tool_choice, **kwargs
            )
        return await super().chat_completion(
            context, model, stream, tools, tool_choice, **kwargs
        )

    def _shape_request(self, model_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        """OpenRouter-specific request shaping, shared by both backends.

        - ``provider_preferences`` from config merge under a caller's
          ``provider=`` (the caller wins key by key).
        - ``effort`` is clamped to the model's ``supported_efforts`` (from the
          cached listing), or dropped for a model without reasoning.
        """
        kwargs = dict(kwargs)
        if self._provider_preferences:
            kwargs["provider"] = {**self._provider_preferences, **dict(kwargs.get("provider") or {})}
        listed = self._listed(model_name)
        if "effort" in kwargs and listed is not None:
            efforts = (listed.get("reasoning") or {}).get("supported_efforts") or []
            requested = str(kwargs["effort"]).strip().lower()
            if not listed.get("reasoning"):
                logger.debug("OpenRouter model %s has no reasoning; effort dropped.", model_name)
                kwargs.pop("effort")
            elif efforts and requested not in efforts:
                kwargs["effort"] = nearest_effort(requested, efforts) or requested
        return kwargs

    async def _chat_completion_sdk(
        self,
        context: ContextPayload,
        model: str | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Chat completion through the ``openrouter`` SDK.

        Sends the same body as the default path (messages built by the shared
        payload builder, so tool calls, images and replayed reasoning survive)
        and returns the same dict shapes. A parameter the SDK has no field for
        is an error, not a silent drop.
        """
        model_name = model or self.default_model
        body: dict[str, Any] = {
            "model": model_name,
            "messages": [self._build_message_payload(m, model_name) for m in context],
            "stream": stream,
        }
        if tools:
            body["tools"] = [{"type": "function", "function": t.model_dump()} for t in tools]
        if tool_choice:
            body["tool_choice"] = tool_choice
        params = dict(kwargs)
        body.update(params.pop("extra_body", None) or {})
        effort = params.pop("effort", None)
        if effort is not None:
            body["reasoning"] = {**dict(params.pop("reasoning", None) or {}), "effort": effort}
        if params.pop("native_search", False):
            self._apply_native_search(params, model_name)
        body.update(params)

        allowed = set(inspect.signature(self._openrouter_client.chat.send_async).parameters)
        unknown = sorted(k for k in body if k not in allowed)
        if unknown:
            raise ValueError(f"The openrouter SDK has no parameter {unknown[0]!r}; use backend='openai'.")
        try:
            result = await self._openrouter_client.chat.send_async(**body)
        except Exception as e:
            raise ProviderError(
                self.get_name(),
                f"OpenRouter SDK error: {e}",
                model_name=model_name,
                status_code=getattr(e, "status_code", None),
            ) from e

        if not stream:
            response = _sdk_dump(result)
            self._raise_in_band(response, model_name)
            return response

        async def _chunks() -> AsyncGenerator[dict[str, Any], None]:
            try:
                async for event in result:
                    yield _sdk_dump(getattr(event, "data", event))
            except ProviderError:
                raise
            except Exception as e:
                raise ProviderError(
                    self.get_name(), f"OpenRouter SDK stream error: {e}", model_name=model_name
                ) from e

        return self._compat_stream(_chunks(), model_name)

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


def _sdk_dump(obj: Any) -> dict[str, Any]:
    """An SDK model as the wire-shaped dict the extractors read."""
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return dict(dump(by_alias=True, exclude_none=True, mode="json"))
    return dict(obj) if isinstance(obj, dict) else {}
