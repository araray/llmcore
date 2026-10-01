# src/llmcore/providers/anthropic_provider.py
"""
Anthropic API provider implementation for the LLMCore library.

Supports:
- Chat completion (text, streaming)
- Tool/function calling with normalized OpenAI-compatible response format
- Multimodal content (vision: images via base64/URL; documents: PDF/text)
- Extended thinking (enabled/adaptive/disabled with budget_tokens)
- Structured output (output_config with JSON schema)
- Reasoning effort control (output_config.effort)
- Prompt caching (cache_control on content blocks + system)
- Proper token counting via messages.count_tokens() API
- Dynamic model discovery via models.list() API
- Streaming tool call support (input_json_delta accumulation)
- Service tier selection
- Metadata / user_id passthrough

Tested against anthropic SDK v0.94.0.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from collections.abc import AsyncGenerator
from typing import Any

try:
    import httpx

    httpx_available = True
except ImportError:  # pragma: no cover
    httpx_available = False
    httpx = None  # type: ignore

logger = logging.getLogger(__name__)

# --- Guarded SDK imports ---
anthropic_available = False
try:
    import anthropic
    from anthropic import AsyncAnthropic
    from anthropic._exceptions import (
        APIConnectionError as AnthropicConnectionError,
    )
    from anthropic._exceptions import (
        APIStatusError,
        OverloadedError,
    )
    from anthropic._exceptions import (
        APITimeoutError as AnthropicTimeoutError,
    )
    from anthropic._exceptions import (
        AuthenticationError as AnthropicAuthError,
    )
    from anthropic._exceptions import (
        BadRequestError as AnthropicBadRequestError,
    )
    from anthropic._exceptions import (
        RateLimitError as AnthropicRateLimitError,
    )

    anthropic_available = True
except ImportError:
    anthropic = None  # type: ignore[assignment]
    AsyncAnthropic = None  # type: ignore[assignment,misc]
    APIStatusError = Exception  # type: ignore[assignment,misc]
    AnthropicBadRequestError = Exception  # type: ignore[assignment,misc]
    AnthropicAuthError = Exception  # type: ignore[assignment,misc]
    AnthropicRateLimitError = Exception  # type: ignore[assignment,misc]
    AnthropicConnectionError = Exception  # type: ignore[assignment,misc]
    AnthropicTimeoutError = Exception  # type: ignore[assignment,misc]
    OverloadedError = Exception  # type: ignore[assignment,misc]

from ..exceptions import ConfigError, ContextLengthError, ProviderError
from ..models import Message, ModelDetails, Tool, ToolCall
from ..models import Role as LLMCoreRole
from ..tokens import EstimateCounter as _EstimateCounter
from .base import BaseProvider, ContextPayload

# ---------------------------------------------------------------------------
# Default context lengths — fallback when dynamic discovery hasn't run.
# Updated April 2026 to cover current-generation models.
# ---------------------------------------------------------------------------
DEFAULT_ANTHROPIC_TOKEN_LIMITS: dict[str, int] = {
    # Claude 4.6
    "claude-opus-4-6": 200000,
    "claude-sonnet-4-6": 200000,
    # Claude 4.5
    "claude-opus-4-5": 200000,
    "claude-opus-4-5-20251101": 200000,
    "claude-sonnet-4-5": 200000,
    "claude-sonnet-4-5-20250929": 200000,
    "claude-haiku-4-5": 200000,
    "claude-haiku-4-5-20251001": 200000,
    # Claude 4.1
    "claude-opus-4-1": 200000,
    "claude-opus-4-1-20250805": 200000,
    # Claude 4.0
    "claude-opus-4-0": 200000,
    "claude-opus-4-20250514": 200000,
    "claude-sonnet-4-0": 200000,
    "claude-sonnet-4-20250514": 200000,
    # Claude 3 (legacy, some still available)
    "claude-3-haiku-20240307": 200000,
}

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"


#: Public API root for the direct transport.
_ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"

#: Version header the Messages API requires on every direct request. The SDK
#: sends this for you; calling the API directly means owning it.
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(BaseProvider):
    """
    LLMCore provider for interacting with the Anthropic API (Claude models).

    Handles ``list[Message]`` context type and standardized tool-calling.
    Normalizes all responses to OpenAI-compatible structure for consistency
    with other llmcore providers.

    Supports multimodal content via ``Message.metadata`` conventions:

    - ``metadata["inline_images"]``: list of image dicts::

          [{"data": "<base64>", "media_type": "image/png"}]
          [{"url": "https://..."}]

    - ``metadata["inline_documents"]``: list of document dicts::

          [{"data": "<base64>", "media_type": "application/pdf"}]
          [{"data": "plain text...", "media_type": "text/plain", "title": "..."}]
          [{"url": "https://...pdf", "media_type": "application/pdf"}]

    - ``metadata["content_parts"]``: pre-constructed Anthropic content blocks
      (passed through directly).

    - ``metadata["tool_calls"]``: list of raw Anthropic tool_use dicts for
      assistant messages in multi-turn tool calling.

    - ``metadata["cache_control"]``: dict applied to the last content block
      in the message (e.g., ``{"type": "ephemeral"}``).
    """

    _client: AsyncAnthropic | None = None
    _api_key_env_var: str | None = None
    _discovered_context_lengths: dict[str, int] | None = None

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False):
        """
        Initialize the AnthropicProvider.

        Args:
            config: Configuration dictionary from ``[providers.anthropic]``
                containing: ``api_key``, ``api_key_env_var``, ``base_url``,
                ``default_model``, ``timeout``.
            log_raw_payloads: Whether to log raw request/response payloads.
        """
        super().__init__(config, log_raw_payloads)
        if not anthropic_available:
            raise ImportError(
                "Anthropic library not installed. "
                "Please install `anthropic` or `llmcore[anthropic]`."
            )

        self._api_key_env_var = config.get("api_key_env_var")
        api_key = config.get("api_key")
        if not api_key and self._api_key_env_var:
            api_key = os.environ.get(self._api_key_env_var)
        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY")

        self.api_key = api_key
        self.base_url = config.get("base_url")
        self.default_model = config.get("default_model", DEFAULT_MODEL)
        self.timeout = float(config.get("timeout", 120.0))

        if not self.api_key:
            raise ConfigError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment "
                "variable or configure api_key in config."
            )

        try:
            self._client = AsyncAnthropic(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
            logger.debug("Anthropic async client initialized.")
        except Exception as e:
            raise ConfigError(f"Anthropic client initialization failed: {e}")

        # --- Transport -------------------------------------------------------
        #
        # llmcore's rule is to call the API directly and fall back to the vendor
        # SDK, and this provider had only the SDK path. The direct path is now
        # available with `backend = "httpx"`.
        #
        # The default stays "sdk" here, unlike the media providers: the
        # anthropic SDK owns prompt-caching headers, beta-feature headers and
        # retry/backoff behaviour that the direct path would have to track by
        # hand, and this provider's existing tests mock AsyncAnthropic. Direct
        # is opt-in rather than imposed.
        self._backend = self._resolve_backend(config.get("backend"))
        self._http: Any = None

    # ------------------------------------------------------------------
    # Provider Identity & Discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_backend(requested: str | None) -> str:
        """Resolve the transport. ``"sdk"`` by default; ``"httpx"`` is opt-in."""
        available = {"sdk": True, "httpx": httpx_available}
        req = (requested or "sdk").lower()
        if req not in ("sdk", "httpx"):
            logger.warning("Unknown Anthropic backend '%s'; using the SDK.", req)
            return "sdk"
        if available.get(req):
            return req
        logger.warning(
            "Requested Anthropic backend '%s' is unavailable; using the SDK.", req
        )
        return "sdk"

    def _direct_base_url(self) -> str:
        """REST root for the direct transport."""
        return str(self.base_url or _ANTHROPIC_DEFAULT_BASE_URL).rstrip("/")

    def _get_http(self) -> Any:
        """Return the lazily-built direct HTTP client."""
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self._direct_base_url(),
                headers={
                    "x-api-key": self.api_key or "",
                    "anthropic-version": _ANTHROPIC_VERSION,
                    "content-type": "application/json",
                },
                timeout=self.timeout,
            )
        return self._http

    @staticmethod
    def _drop_none(value: Any) -> Any:
        """Recursively strip ``None`` values from a decoded JSON payload.

        The SDK path returns ``model_dump(exclude_none=True)``, so raw wire JSON
        — which does include nulls like ``"stop_sequence": null`` — would differ
        from it key-for-key. Stripping them here is what makes the two
        transports produce *identical* dicts, which is the whole point: every
        ``extract_*`` method parses these dicts, so a shape difference would
        leak the transport choice into callers.
        """
        if isinstance(value, dict):
            return {
                k: AnthropicProvider._drop_none(v)
                for k, v in value.items()
                if v is not None
            }
        if isinstance(value, list):
            return [AnthropicProvider._drop_none(v) for v in value]
        return value

    def _raise_direct_status(self, status: int, body: str, model_name: str) -> None:
        """Map a direct-transport failure onto the same errors the SDK path uses.

        Raises:
            ContextLengthError: If the prompt exceeded the model's window.
            ProviderError: For every other failure.
        """
        lowered = body.lower()
        if status == 400 and (
            "context" in lowered and ("long" in lowered or "exceed" in lowered)
        ):
            raise ContextLengthError(
                model_name=model_name,
                limit=self.get_max_context_length(model_name),
                actual=0,
                message=body,
            )
        if status in (401, 403):
            raise ProviderError(
                self.get_name(),
                f"Anthropic authentication failed. Check ANTHROPIC_API_KEY. Error: {body}",
                model_name=model_name,
                status_code=status,
            )
        if status == 404:
            raise ProviderError(
                self.get_name(),
                f"Anthropic model '{model_name}' not found. Error: {body}",
                model_name=model_name,
                status_code=status,
            )
        if status == 429:
            raise ProviderError(
                self.get_name(),
                f"Anthropic rate limit reached. Error: {body}",
                model_name=model_name,
                status_code=status,
                retryable=True,
            )
        raise ProviderError(
            self.get_name(),
            f"API Error ({status}): {body}",
            model_name=model_name,
            status_code=status,
            retryable=status >= 500,
        )

    @staticmethod
    async def _sdk_message_events(raw_stream: Any) -> AsyncGenerator[dict[str, Any], None]:
        """Yield SDK stream events as plain dicts.

        The counterpart to :meth:`_direct_message_events`: both produce the same
        event dicts so the OpenAI-shaped normalization is written once.
        """
        async for event in raw_stream:
            yield event.model_dump(exclude_none=True)

    async def _direct_messages(
        self, body: dict[str, Any], model_name: str
    ) -> dict[str, Any]:
        """POST ``/v1/messages`` and return the SDK-equivalent response dict."""
        try:
            resp = await self._get_http().post("/v1/messages", json=body)
        except httpx.TimeoutException as e:
            raise ProviderError(self.get_name(), f"Timeout: {e}") from e
        except httpx.HTTPError as e:
            raise ProviderError(self.get_name(), f"Connection error: {e}") from e
        if resp.status_code >= 400:
            self._raise_direct_status(resp.status_code, resp.text, model_name)
        return self._drop_none(resp.json())

    async def _direct_message_events(
        self, body: dict[str, Any], model_name: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Yield raw Messages-API stream events as dicts.

        Produces the *same* event dicts the SDK path produces, so the whole
        OpenAI-shaped normalization downstream is shared between transports
        rather than written twice.
        """
        client = self._get_http()
        async with client.stream(
            "POST", "/v1/messages", json={**body, "stream": True}
        ) as resp:
            if resp.status_code >= 400:
                await resp.aread()
                self._raise_direct_status(resp.status_code, resp.text, model_name)
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    # SSE also carries `event:` lines, which duplicate the
                    # payload's own `type` field, so they are skipped.
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    event = json.loads(payload)
                except ValueError:
                    logger.warning("Skipping unparseable Anthropic SSE chunk.")
                    continue
                yield self._drop_none(event)

    def get_name(self) -> str:
        """Returns the provider instance name."""
        return self._provider_instance_name or "anthropic"

    async def get_models_details(self) -> list[ModelDetails]:
        """Discover available models via the Anthropic Models API.

        Falls back to the static ``DEFAULT_ANTHROPIC_TOKEN_LIMITS`` table
        if the API call fails (e.g., due to permissions).
        """
        if not self._client:
            raise ProviderError(self.get_name(), "Anthropic client not initialized.")

        details_list: list[ModelDetails] = []

        try:
            models_page = await self._client.models.list(limit=100)
            discovered_ctx: dict[str, int] = {}

            for model_info in models_page.data:
                model_id = model_info.id
                context_length = model_info.max_input_tokens or 200000

                # Extract capabilities
                caps = model_info.capabilities
                supports_tools = True  # All current Claude models support tools
                supports_vision = False
                supports_pdf = False
                supports_thinking = False

                if caps:
                    if hasattr(caps, "image_input"):
                        img_cap = caps.image_input
                        supports_vision = (
                            getattr(img_cap, "supported", False) if img_cap is not None else False
                        )
                    if hasattr(caps, "pdf_input"):
                        pdf_cap = caps.pdf_input
                        supports_pdf = (
                            getattr(pdf_cap, "supported", False) if pdf_cap is not None else False
                        )
                    if hasattr(caps, "thinking") and caps.thinking:
                        thinking_types = getattr(caps.thinking, "types", None)
                        if thinking_types:
                            supports_thinking = (
                                getattr(thinking_types, "enabled", None) is not None
                                or getattr(thinking_types, "adaptive", None) is not None
                            )

                discovered_ctx[model_id] = context_length

                metadata: dict[str, Any] = {
                    "display_name": model_info.display_name,
                    "max_output_tokens": model_info.max_tokens,
                }
                if supports_vision:
                    metadata["supports_vision"] = True
                if supports_pdf:
                    metadata["supports_pdf"] = True
                if supports_thinking:
                    metadata["supports_thinking"] = True

                details_list.append(
                    ModelDetails(
                        id=model_id,
                        context_length=context_length,
                        supports_streaming=True,
                        supports_tools=supports_tools,
                        provider_name=self.get_name(),
                        metadata=metadata,
                    )
                )

            self._discovered_context_lengths = discovered_ctx
            logger.info(f"Discovered {len(details_list)} Anthropic models via API.")
            return details_list

        except Exception as e:
            logger.warning(
                f"Failed to list models via Anthropic API ({e}). "
                f"Falling back to static model table."
            )
            # Fallback to static table
            for model_id, context_length in DEFAULT_ANTHROPIC_TOKEN_LIMITS.items():
                details_list.append(
                    ModelDetails(
                        id=model_id,
                        context_length=context_length,
                        supports_streaming=True,
                        supports_tools=True,
                        provider_name=self.get_name(),
                        metadata={},
                    )
                )
            return details_list

    # ------------------------------------------------------------------
    # Extended thinking: what each model generation actually accepts
    # ------------------------------------------------------------------

    #: Generation at which `thinking.budget_tokens` stopped being accepted and
    #: `{"type": "adaptive"}` took over. Claude 4.6 deprecated it; 5.x and
    #: Fable 5.x **reject it with a 400**. Forwarding a caller's budget to one
    #: of those models turns a reasonable request into an error, which is the
    #: defect recorded in PROVIDER_MODERNIZATION_PLAN.md and the routing spec.
    _ADAPTIVE_FROM = (4, 6)

    #: llmcore's canonical reasoning-effort vocabulary mapped onto Anthropic's
    #: `output_config.effort`, which has fewer rungs. `none`/`minimal` fold
    #: down and `xhigh` folds up, because folding to the nearest supported rung
    #: is the documented policy for an unsupported effort level -- dropping it
    #: silently would lose the caller's intent entirely.
    _EFFORT_TO_OUTPUT_CONFIG: dict[str, str] = {
        # "none" never reaches this table -- it disables thinking outright.
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "high",
        "max": "max",
    }

    #: For pre-4.6 models, which take a token budget instead of an effort
    #: level. Anthropic requires >= 1024.
    _EFFORT_TO_BUDGET: dict[str, int] = {
        "none": 0,
        "minimal": 1024,
        "low": 2048,
        "medium": 8192,
        "high": 16384,
        "xhigh": 24576,
        "max": 32768,
    }

    @staticmethod
    def _model_generation(model: str | None) -> tuple[int, int] | None:
        """Extract ``(major, minor)`` from a model id, or ``None`` if unclear.

        Ids look like ``claude-sonnet-4-6`` or ``claude-opus-5-5-20260101``.
        Returning ``None`` for anything unrecognised matters: an unknown model
        must not be assumed modern, because guessing wrong in that direction
        sends `budget_tokens` to a model that rejects it.
        """
        if not model:
            return None
        match = re.search(r"-(\d+)-(\d+)(?:-|$)", model)
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))

    @classmethod
    def _supports_adaptive_thinking(cls, model: str | None) -> bool:
        """Whether *model* takes ``{"type": "adaptive"}`` rather than a budget."""
        generation = cls._model_generation(model)
        if generation is None:
            # Unknown id. Treat as modern, because every model Anthropic
            # currently serves is: the pre-4.6 family is the shrinking set, so
            # defaulting the other way would break new models by default.
            return True
        return generation >= cls._ADAPTIVE_FROM

    @classmethod
    def _normalize_thinking(
        cls, thinking: Any, effort: str | None, model: str | None
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Map a caller's thinking/effort request onto what *model* accepts.

        Returns ``(thinking_block, output_config_effort)``. Either may be
        ``None``.

        The whole point is that these two are not interchangeable across
        generations: 4.6 and later express reasoning depth as
        ``output_config.effort`` with ``thinking: {"type": "adaptive"}``, while
        earlier models express it as ``thinking.budget_tokens``. A caller who
        says ``effort="high"`` means the same thing in both cases and should
        not have to know which.
        """
        adaptive = cls._supports_adaptive_thinking(model)
        block: dict[str, Any] | None = dict(thinking) if isinstance(thinking, dict) else None
        effort_key = str(effort).strip().lower() if effort else None

        if adaptive:
            if block and "budget_tokens" in block:
                dropped = block.pop("budget_tokens")
                logger.warning(
                    "Dropping thinking.budget_tokens=%s: %s rejects it with a 400. "
                    "Claude 4.6 and later take thinking={'type': 'adaptive'} plus "
                    "output_config.effort. Pass effort= instead.",
                    dropped,
                    model,
                )
                block.setdefault("type", "adaptive")
                if block.get("type") == "enabled":
                    block["type"] = "adaptive"
            if effort_key == "none":
                # "none" means do not think, on either generation. Returning
                # adaptive-at-low would quietly spend reasoning tokens the
                # caller explicitly asked not to spend.
                return {"type": "disabled"}, None
            if effort_key:
                if block is None:
                    block = {"type": "adaptive"}
                return block, cls._EFFORT_TO_OUTPUT_CONFIG.get(effort_key, "medium")
            return block, None

        # Pre-4.6: a budget, not an effort level.
        if effort_key:
            budget = cls._EFFORT_TO_BUDGET.get(effort_key, 8192)
            if budget <= 0:
                return {"type": "disabled"}, None
            merged = block or {}
            merged.setdefault("type", "enabled")
            merged.setdefault("budget_tokens", budget)
            return merged, None
        if block and block.get("type") == "adaptive":
            logger.warning(
                "%s does not support adaptive thinking; converting to "
                "{'type': 'enabled', 'budget_tokens': %d}.",
                model,
                cls._EFFORT_TO_BUDGET["medium"],
            )
            block = {"type": "enabled", "budget_tokens": cls._EFFORT_TO_BUDGET["medium"]}
        return block, None

    def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
        """Returns a schema of supported inference parameters for Anthropic models."""
        return {
            "max_tokens": {"type": "integer", "minimum": 1},
            "temperature": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "top_p": {"type": "number"},
            "top_k": {"type": "integer"},
            "stop_sequences": {"type": "array", "items": {"type": "string"}},
            "thinking": {
                "type": "object",
                "description": (
                    "Extended thinking config: "
                    "{type: enabled|adaptive|disabled, budget_tokens: int}. "
                    "budget_tokens is dropped for Claude 4.6+ models, which "
                    "reject it; prefer `effort`."
                ),
            },
            "effort": {
                "type": "string",
                "enum": ["none", "minimal", "low", "medium", "high", "xhigh", "max"],
                "description": (
                    "llmcore's canonical reasoning effort. Mapped to "
                    "output_config.effort on Claude 4.6+ and to "
                    "thinking.budget_tokens on earlier models."
                ),
            },
            "output_config": {
                "type": "object",
                "description": (
                    "Output configuration: "
                    "{effort: low|medium|high|max, "
                    "format: {type: json_schema, schema: {...}}}"
                ),
            },
            "metadata": {
                "type": "object",
                "description": "Request metadata: {user_id: str}",
            },
            "service_tier": {
                "type": "string",
                "enum": ["auto", "standard_only"],
            },
        }

    def get_max_context_length(self, model: str | None = None) -> int:
        """Returns the maximum context length (tokens) for the given model."""
        model_name = model or self.default_model

        # Check dynamic discovery cache first
        if self._discovered_context_lengths:
            limit = self._discovered_context_lengths.get(model_name)
            if limit is not None:
                return limit

        # Check model cards via registry
        try:
            from ..model_cards import get_model_card_registry

            registry = get_model_card_registry()
            ctx = registry.get_context_length("anthropic", model_name)
            if ctx and ctx > 0:
                return ctx
        except Exception:
            pass

        # Fallback to static table
        limit = DEFAULT_ANTHROPIC_TOKEN_LIMITS.get(model_name)
        if limit is not None:
            return limit

        # Final fallback — all modern Claude models are 200K
        logger.warning(
            f"Unknown context length for Anthropic model '{model_name}'. Using fallback: 200000."
        )
        return 200000

    # ------------------------------------------------------------------
    # Message Conversion
    # ------------------------------------------------------------------

    def _build_content_blocks(self, msg: Message) -> list[dict[str, Any]]:
        """Build Anthropic content block list from a Message.

        Handles multimodal content via ``Message.metadata`` conventions:
        - ``metadata["content_parts"]``: pre-constructed Anthropic blocks (passthrough)
        - ``metadata["inline_images"]``: images as base64 or URL dicts
        - ``metadata["inline_documents"]``: PDFs/text documents as base64 or URL dicts
        - ``metadata["cache_control"]``: applied to the **last** block

        Args:
            msg: An llmcore Message object.

        Returns:
            List of Anthropic content block dicts.
        """
        metadata = msg.metadata or {}

        # Direct passthrough of pre-constructed blocks
        if "content_parts" in metadata:
            blocks = list(metadata["content_parts"])
            return blocks

        blocks: list[dict[str, Any]] = []

        # Text block (if message has text content)
        if msg.content:
            blocks.append({"type": "text", "text": msg.content})

        # Inline images → ImageBlockParam
        for img in metadata.get("inline_images", []):
            if isinstance(img, str):
                # Plain URL string
                blocks.append(
                    {
                        "type": "image",
                        "source": {"type": "url", "url": img},
                    }
                )
            elif isinstance(img, dict):
                if "url" in img:
                    blocks.append(
                        {
                            "type": "image",
                            "source": {"type": "url", "url": img["url"]},
                        }
                    )
                elif "data" in img:
                    blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": img.get("media_type", "image/png"),
                                "data": img["data"],
                            },
                        }
                    )

        # Inline documents → DocumentBlockParam
        for doc in metadata.get("inline_documents", []):
            if isinstance(doc, dict):
                doc_block: dict[str, Any] = {"type": "document"}
                if "url" in doc:
                    doc_block["source"] = {
                        "type": "url",
                        "url": doc["url"],
                    }
                elif "data" in doc:
                    mt = doc.get("media_type", "application/pdf")
                    if mt == "text/plain":
                        doc_block["source"] = {
                            "type": "text",
                            "media_type": "text/plain",
                            "data": doc["data"],
                        }
                    else:
                        doc_block["source"] = {
                            "type": "base64",
                            "media_type": mt,
                            "data": doc["data"],
                        }
                if "title" in doc:
                    doc_block["title"] = doc["title"]
                if "context" in doc:
                    doc_block["context"] = doc["context"]
                if doc.get("citations"):
                    doc_block["citations"] = doc["citations"]
                blocks.append(doc_block)

        # Apply cache_control to last block if requested
        cache_ctrl = metadata.get("cache_control")
        if cache_ctrl and blocks:
            blocks[-1]["cache_control"] = cache_ctrl

        return blocks

    def _convert_llmcore_msgs_to_anthropic(
        self, messages: list[Message]
    ) -> tuple[str | list[dict[str, Any]] | None, list[dict[str, Any]]]:
        """Convert llmcore Messages to the Anthropic API message format.

        Handles:
        - System prompt extraction (first message if role==SYSTEM)
        - User / assistant message conversion
        - Tool result messages (role==TOOL → tool_result content blocks in user msg)
        - Assistant messages with tool_calls (first-class field or metadata)
        - Multimodal content (images, documents)
        - Consecutive same-role message merging

        Args:
            messages: List of llmcore Message objects.

        Returns:
            Tuple of (system_prompt, anthropic_messages_list).
            system_prompt may be a string or a list of TextBlockParam dicts
            (for cache_control on system).
        """
        anthropic_messages: list[dict[str, Any]] = []
        system_prompt: str | list[dict[str, Any]] | None = None

        processed = list(messages)

        # Extract system prompt
        if processed and processed[0].role == LLMCoreRole.SYSTEM:
            sys_msg = processed.pop(0)
            sys_cache = (sys_msg.metadata or {}).get("cache_control")
            if sys_cache:
                # System with cache_control must be array of TextBlockParam
                system_prompt = [
                    {
                        "type": "text",
                        "text": sys_msg.content,
                        "cache_control": sys_cache,
                    }
                ]
            else:
                system_prompt = sys_msg.content

        # Accumulate tool results that need to be grouped into user messages
        pending_tool_results: list[dict[str, Any]] = []

        for msg in processed:
            # Tool result messages → collect for grouping into user turn
            if msg.role == LLMCoreRole.TOOL:
                is_error = (msg.metadata or {}).get("is_error", False)
                tool_result_block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id or "unknown",
                    "content": msg.content,
                }
                if is_error:
                    tool_result_block["is_error"] = True
                cache_ctrl = (msg.metadata or {}).get("cache_control")
                if cache_ctrl:
                    tool_result_block["cache_control"] = cache_ctrl
                pending_tool_results.append(tool_result_block)
                continue

            # Flush pending tool results into a user message before any
            # non-tool message.
            if pending_tool_results:
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": pending_tool_results,
                    }
                )
                pending_tool_results = []

            # Map role
            if msg.role == LLMCoreRole.USER:
                role_str = "user"
            elif msg.role == LLMCoreRole.ASSISTANT:
                role_str = "assistant"
            else:
                logger.warning(f"Skipping message with unmappable role for Anthropic: {msg.role}")
                continue

            metadata = msg.metadata or {}

            # First-class Message.tool_calls (R-2) takes precedence over the
            # legacy metadata["tool_calls"] channel.
            tool_calls = getattr(msg, "tool_calls", None) or metadata.get("tool_calls")

            # Build content blocks
            if role_str == "assistant" and tool_calls:
                # Assistant message with tool calls — construct tool_use blocks
                content_blocks: list[dict[str, Any]] = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for tc in tool_calls:
                    # Accept both OpenAI-normalized and Anthropic-native formats
                    if tc.get("type") == "function":
                        # OpenAI-normalized format from extract_tool_calls
                        func = tc.get("function", {})
                        args = func.get("arguments", "{}")
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {"raw": args}
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc.get("id", str(uuid.uuid4())),
                                "name": func.get("name", "unknown"),
                                "input": args,
                            }
                        )
                    elif tc.get("type") == "tool_use":
                        # Already Anthropic-native format
                        content_blocks.append(tc)
                    else:
                        # Legacy/minimal format
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc.get("id", str(uuid.uuid4())),
                                "name": tc.get("name", "unknown"),
                                "input": tc.get("input", tc.get("arguments", {})),
                            }
                        )
                anthropic_messages.append(
                    {
                        "role": "assistant",
                        "content": content_blocks,
                    }
                )
            else:
                content_blocks = self._build_content_blocks(msg)
                if not content_blocks:
                    # Anthropic requires non-empty content
                    content_blocks = [{"type": "text", "text": ""}]

                # Merge consecutive same-role messages
                if anthropic_messages and anthropic_messages[-1]["role"] == role_str:
                    logger.debug(f"Merging consecutive '{role_str}' messages for Anthropic API.")
                    existing = anthropic_messages[-1]["content"]
                    if isinstance(existing, list):
                        existing.extend(content_blocks)
                    else:
                        anthropic_messages[-1]["content"] = content_blocks
                    continue

                anthropic_messages.append(
                    {
                        "role": role_str,
                        "content": content_blocks,
                    }
                )

        # Flush any remaining tool results
        if pending_tool_results:
            anthropic_messages.append(
                {
                    "role": "user",
                    "content": pending_tool_results,
                }
            )

        return system_prompt, anthropic_messages

    @staticmethod
    def _convert_tools_to_anthropic(tools: list[Tool]) -> list[dict[str, Any]]:
        """Convert llmcore Tool objects to Anthropic tool format.

        Anthropic uses ``input_schema`` instead of ``parameters``.

        Args:
            tools: List of llmcore Tool objects.

        Returns:
            List of Anthropic-format tool dicts.
        """
        anthropic_tools: list[dict[str, Any]] = []
        for tool in tools:
            anthropic_tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                }
            )
        return anthropic_tools

    @staticmethod
    def _convert_tool_choice(tool_choice: str) -> dict[str, Any]:
        """Convert a tool_choice string to Anthropic's format.

        Args:
            tool_choice: One of "auto", "any", "none", or a specific tool name.

        Returns:
            Anthropic tool_choice dict.
        """
        if tool_choice == "auto":
            return {"type": "auto"}
        elif tool_choice == "any":
            return {"type": "any"}
        elif tool_choice == "none":
            return {"type": "none"}
        else:
            # Assume it's a specific tool name
            return {"type": "tool", "name": tool_choice}

    # ------------------------------------------------------------------
    # Response Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_tool_calls_from_content(
        content_blocks: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        """Extract tool_use blocks from response content and normalize to
        OpenAI-compatible format.

        Args:
            content_blocks: List of Anthropic content block dicts.

        Returns:
            List of OpenAI-normalized tool call dicts, or None if no tool calls.
        """
        tool_calls: list[dict[str, Any]] = []
        for block in content_blocks:
            if block.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id", str(uuid.uuid4())),
                        "type": "function",
                        "function": {
                            "name": block.get("name", "unknown"),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    }
                )
        return tool_calls if tool_calls else None

    @staticmethod
    def _normalize_stop_reason(stop_reason: str | None) -> str | None:
        """Map Anthropic stop_reason to a normalized value.

        Anthropic values: end_turn, tool_use, max_tokens, stop_sequence,
                          pause_turn, refusal
        Normalized:        stop, tool_calls, length, stop, stop, content_filter
        """
        if stop_reason is None:
            return None
        mapping = {
            "end_turn": "stop",
            "tool_use": "tool_calls",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "pause_turn": "stop",
            "refusal": "content_filter",
        }
        return mapping.get(stop_reason, stop_reason)

    @staticmethod
    def _normalize_usage(usage_dict: dict[str, Any]) -> dict[str, Any]:
        """Normalize Anthropic usage to include OpenAI-compatible keys.

        Anthropic uses ``input_tokens``/``output_tokens``.
        We add ``prompt_tokens``/``completion_tokens``/``total_tokens`` aliases.
        """
        normalized = dict(usage_dict)
        input_t = usage_dict.get("input_tokens", 0)
        output_t = usage_dict.get("output_tokens", 0)
        normalized["prompt_tokens"] = input_t
        normalized["completion_tokens"] = output_t
        normalized["total_tokens"] = input_t + output_t
        return normalized

    # ------------------------------------------------------------------
    # Chat Completion
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
        """Send a chat completion request to the Anthropic API.

        Args:
            context: List of llmcore Message objects.
            model: Model identifier. Defaults to ``self.default_model``.
            stream: If True, returns async generator of streaming chunks.
            tools: Optional list of Tool objects.
            tool_choice: Tool choice strategy
                (``"auto"``, ``"any"``, ``"none"``, or tool name).
            **kwargs: Additional parameters. Supported keys:
                ``max_tokens``, ``temperature``, ``top_p``, ``top_k``,
                ``stop_sequences``, ``thinking``, ``output_config``,
                ``metadata``, ``service_tier``.

        Returns:
            Dict with OpenAI-normalized response, or async generator
            for streaming.
        """
        if not self._client:
            raise ProviderError(self.get_name(), "Anthropic client not initialized.")

        model_name = model or self.default_model

        if not (isinstance(context, list) and all(isinstance(msg, Message) for msg in context)):
            raise ProviderError(self.get_name(), "Unsupported context type.")

        system_prompt, messages_payload = self._convert_llmcore_msgs_to_anthropic(context)
        if not messages_payload:
            raise ProviderError(
                self.get_name(),
                "No valid messages to send after context processing.",
            )

        # Build API kwargs
        api_kwargs: dict[str, Any] = {}

        # Required: max_tokens (Anthropic always requires this)
        api_kwargs["max_tokens"] = kwargs.pop("max_tokens", 4096)

        # Standard generation params
        for key in ("temperature", "top_p", "top_k", "stop_sequences"):
            if key in kwargs:
                api_kwargs[key] = kwargs.pop(key)

        # Extended thinking and effort, mapped to what this model generation
        # actually accepts (see _normalize_thinking).
        thinking, mapped_effort = self._normalize_thinking(
            kwargs.pop("thinking", None), kwargs.pop("effort", None), model_name
        )
        if thinking:
            api_kwargs["thinking"] = thinking

        # Output config (structured output + effort)
        output_config = kwargs.pop("output_config", None)
        if mapped_effort:
            output_config = {**(output_config or {})}
            # An explicit output_config.effort wins: the caller named
            # Anthropic's own vocabulary, so they meant it literally.
            output_config.setdefault("effort", mapped_effort)
        if output_config:
            api_kwargs["output_config"] = output_config

        # Metadata (user_id for abuse detection)
        metadata_param = kwargs.pop("metadata", None)
        if metadata_param:
            api_kwargs["metadata"] = metadata_param

        # Service tier
        service_tier = kwargs.pop("service_tier", None)
        if service_tier:
            api_kwargs["service_tier"] = service_tier

        # Tools
        if tools:
            api_kwargs["tools"] = self._convert_tools_to_anthropic(tools)
        if tool_choice:
            api_kwargs["tool_choice"] = self._convert_tool_choice(tool_choice)

        # Warn about unrecognized kwargs but pass through (forward-compat)
        supported = self.get_supported_parameters()
        for key in list(kwargs.keys()):
            if key not in supported:
                logger.warning(
                    f"Parameter '{key}' not in supported parameters for "
                    f"Anthropic. Passing through anyway."
                )
            api_kwargs[key] = kwargs.pop(key)

        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
            log_data = {
                "model": model_name,
                "messages": messages_payload,
                "system": system_prompt,
                "stream": stream,
                **api_kwargs,
            }
            logger.debug(
                f"RAW LLM REQUEST ({self.get_name()}): "
                f"{json.dumps(log_data, indent=2, default=str)}"
            )

        try:
            if stream:
                return await self._handle_streaming(
                    model_name, messages_payload, system_prompt, api_kwargs
                )
            else:
                return await self._handle_non_streaming(
                    model_name, messages_payload, system_prompt, api_kwargs
                )

        except AnthropicBadRequestError as e:
            err_msg = str(e)
            # Detect context length errors
            if "too long" in err_msg.lower() or "token" in err_msg.lower():
                raise ContextLengthError(
                    model_name=model_name,
                    limit=self.get_max_context_length(model_name),
                    message=f"Anthropic context length exceeded: {e}",
                )
            raise ProviderError(self.get_name(), f"Bad request: {e}")
        except AnthropicAuthError as e:
            raise ProviderError(self.get_name(), f"Authentication failed: {e}")
        except AnthropicRateLimitError as e:
            raise ProviderError(self.get_name(), f"Rate limited: {e}")
        except OverloadedError as e:
            raise ProviderError(self.get_name(), f"API overloaded: {e}")
        except APIStatusError as e:
            logger.error(f"Anthropic API error: {e}", exc_info=True)
            raise ProviderError(
                self.get_name(),
                f"API Error (Status: {getattr(e, 'status_code', 'N/A')}): {e}",
            )
        except (ContextLengthError, ProviderError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error during Anthropic chat: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"An unexpected error occurred: {e}")

    async def _handle_non_streaming(
        self,
        model_name: str,
        messages_payload: list[dict[str, Any]],
        system_prompt: str | list[dict[str, Any]] | None,
        api_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Process a non-streaming messages.create call and normalize response.

        Args:
            model_name: The model identifier string.
            messages_payload: Converted Anthropic messages list.
            system_prompt: System prompt (string or list of text blocks).
            api_kwargs: Additional API keyword arguments.

        Returns:
            OpenAI-normalized response dict.
        """
        if self._backend == "httpx":
            body: dict[str, Any] = {"model": model_name, "messages": messages_payload}
            if system_prompt is not None:
                body["system"] = system_prompt
            body.update(api_kwargs)
            response_dict = await self._direct_messages(body, model_name)
        else:
            response = await self._client.messages.create(  # type: ignore[union-attr]
                model=model_name,
                messages=messages_payload,  # type: ignore[arg-type]
                system=system_prompt,  # type: ignore[arg-type]
                **api_kwargs,
            )
            response_dict = response.model_dump(exclude_none=True)

        if self.log_raw_payloads_enabled:
            logger.debug(
                f"RAW LLM RESPONSE ({self.get_name()}): "
                f"{json.dumps(response_dict, indent=2, default=str)}"
            )

        content_blocks = response_dict.get("content", [])

        # Extract text content (excluding thinking and tool_use blocks)
        text_content = "".join(
            block.get("text", "") for block in content_blocks if block.get("type") == "text"
        )

        # Extract thinking content
        thinking_content = None
        thinking_texts = [
            b.get("thinking", "") for b in content_blocks if b.get("type") == "thinking"
        ]
        if thinking_texts:
            thinking_content = "".join(thinking_texts)

        # Extract and normalize tool calls
        tool_calls = self._normalize_tool_calls_from_content(content_blocks)

        # Normalize finish reason
        finish_reason = self._normalize_stop_reason(response_dict.get("stop_reason"))

        # Build message dict
        message_dict: dict[str, Any] = {
            "role": "assistant",
            "content": (text_content if not tool_calls else (text_content or None)),
        }
        if tool_calls:
            message_dict["tool_calls"] = tool_calls
        if thinking_content:
            message_dict["thinking"] = thinking_content

        # Build usage dict
        usage_dict: dict[str, Any] = {}
        raw_usage = response_dict.get("usage", {})
        if raw_usage:
            usage_dict = self._normalize_usage(raw_usage)

        return {
            "id": response_dict.get("id"),
            "model": response_dict.get("model"),
            "choices": [
                {
                    "index": 0,
                    "message": message_dict,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage_dict,
        }

    async def _handle_streaming(
        self,
        model_name: str,
        messages_payload: list[dict[str, Any]],
        system_prompt: str | list[dict[str, Any]] | None,
        api_kwargs: dict[str, Any],
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Process a streaming messages.create call.

        Uses ``messages.create(stream=True)`` which returns an ``AsyncStream``
        of ``RawMessageStreamEvent`` objects directly (no context manager
        needed).  Yields OpenAI-normalized chunks.

        Handles text deltas, thinking deltas, tool call streaming
        (``input_json_delta``), and message-level events.
        """
        if self._backend == "httpx":
            body: dict[str, Any] = {"model": model_name, "messages": messages_payload}
            if system_prompt is not None:
                body["system"] = system_prompt
            body.update(api_kwargs)
            event_source = self._direct_message_events(body, model_name)
        else:
            raw_stream = await self._client.messages.create(  # type: ignore[union-attr]
                model=model_name,
                messages=messages_payload,  # type: ignore[arg-type]
                system=system_prompt,  # type: ignore[arg-type]
                stream=True,
                **api_kwargs,
            )
            event_source = self._sdk_message_events(raw_stream)

        # Track state for streaming tool calls
        current_tool_id: str | None = None
        current_tool_name: str | None = None

        async def stream_wrapper() -> AsyncGenerator[dict[str, Any], None]:
            nonlocal current_tool_id, current_tool_name

            async for event_dict in event_source:
                event_type = event_dict.get("type")

                if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"RAW LLM STREAM CHUNK ({self.get_name()}): "
                        f"{json.dumps(event_dict, indent=2, default=str)}"
                    )

                if event_type == "content_block_start":
                    block = event_dict.get("content_block", {})
                    block_type = block.get("type")

                    if block_type == "tool_use":
                        current_tool_id = block.get("id")
                        current_tool_name = block.get("name")
                        # Emit tool call start
                        yield {
                            "type": "content_block_start",
                            "content_block": block,
                            "choices": [
                                {
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": event_dict.get("index", 0),
                                                "id": current_tool_id,
                                                "type": "function",
                                                "function": {
                                                    "name": current_tool_name,
                                                    "arguments": "",
                                                },
                                            }
                                        ]
                                    }
                                }
                            ],
                        }
                    elif block_type == "thinking":
                        yield {
                            "type": "content_block_start",
                            "content_block": block,
                            "choices": [{"delta": {"thinking": ""}}],
                        }
                    continue

                elif event_type == "content_block_delta":
                    delta = event_dict.get("delta", {})
                    delta_type = delta.get("type")

                    if delta_type == "text_delta":
                        text = delta.get("text", "")
                        yield {
                            "type": "content_block_delta",
                            "delta": delta,
                            "choices": [{"delta": {"content": text}}],
                        }

                    elif delta_type == "input_json_delta":
                        partial = delta.get("partial_json", "")
                        yield {
                            "type": "content_block_delta",
                            "delta": delta,
                            "choices": [
                                {
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": event_dict.get("index", 0),
                                                "function": {
                                                    "arguments": partial,
                                                },
                                            }
                                        ]
                                    }
                                }
                            ],
                        }

                    elif delta_type == "thinking_delta":
                        thinking_text = delta.get("thinking", "")
                        yield {
                            "type": "content_block_delta",
                            "delta": delta,
                            "choices": [{"delta": {"thinking": thinking_text}}],
                        }

                    elif delta_type in ("citations_delta", "signature_delta"):
                        yield {
                            "type": "content_block_delta",
                            "delta": delta,
                        }

                    continue

                elif event_type == "content_block_stop":
                    if current_tool_id is not None:
                        # Tool call complete — reset state
                        current_tool_id = None
                        current_tool_name = None
                    yield {"type": "content_block_stop", **event_dict}
                    continue

                elif event_type == "message_start":
                    msg = event_dict.get("message", {})
                    usage = msg.get("usage", {})
                    yield {
                        "type": "message_start",
                        "message": msg,
                        "usage": (self._normalize_usage(usage) if usage else {}),
                    }
                    continue

                elif event_type == "message_delta":
                    delta = event_dict.get("delta", {})
                    stop_reason = self._normalize_stop_reason(delta.get("stop_reason"))
                    usage = event_dict.get("usage", {})
                    yield {
                        "type": "message_delta",
                        "delta": delta,
                        "choices": [{"finish_reason": stop_reason}],
                        "usage": (self._normalize_usage(usage) if usage else {}),
                    }
                    continue

                elif event_type == "message_stop":
                    yield {"type": "message_stop"}
                    continue

                # Pass through any unknown events
                yield event_dict

        return stream_wrapper()

    # ------------------------------------------------------------------
    # Response Extraction
    # ------------------------------------------------------------------

    def extract_response_content(self, response: dict[str, Any]) -> str:
        """Extract text content from a non-streaming response.

        Works with the OpenAI-normalized format produced by
        ``chat_completion()``.
        """
        try:
            choices = response.get("choices", [])
            if not choices:
                return ""
            message = choices[0].get("message", {})
            return message.get("content") or ""
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Failed to extract content from Anthropic response: {e}")
            return ""

    def extract_delta_content(self, chunk: dict[str, Any]) -> str:
        """Extract text delta from a streaming chunk.

        Works with the OpenAI-normalized format produced by streaming.
        """
        try:
            choices = chunk.get("choices", [])
            if not choices:
                return ""
            delta = choices[0].get("delta", {})
            return delta.get("content") or ""
        except (KeyError, IndexError, TypeError):
            return ""

    def extract_tool_calls(self, response: dict[str, Any]) -> list[ToolCall]:
        """Extract tool calls from a normalized Anthropic response dict.

        Extracts from the OpenAI-normalized response format produced by
        ``chat_completion()``.

        Args:
            response: The normalized response dict.

        Returns:
            List of ``ToolCall`` objects. Empty list if no tool calls.
        """
        tool_calls_out: list[ToolCall] = []
        try:
            choices = response.get("choices", [])
            if not choices:
                return tool_calls_out
            message = choices[0].get("message", {})
            raw_calls = message.get("tool_calls")
            if not raw_calls:
                return tool_calls_out
            for tc in raw_calls:
                func = tc.get("function", {})
                args_str = func.get("arguments", "{}")
                try:
                    arguments = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    arguments = {"raw": args_str}
                tool_calls_out.append(
                    ToolCall(
                        id=tc.get("id", str(uuid.uuid4())),
                        name=func.get("name", "unknown"),
                        arguments=arguments,
                    )
                )
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Failed to extract tool calls from Anthropic response: {e}")
        return tool_calls_out

    # ------------------------------------------------------------------
    # Token Counting
    # ------------------------------------------------------------------

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """Count tokens for a text string.

        Uses the Anthropic ``messages.count_tokens()`` API for accuracy.
        Falls back to character-based approximation if unavailable.
        """
        if not self._client:
            logger.warning("Anthropic client not available. Approximating token count.")
            return _EstimateCounter().count(text)

        if not text:
            return 0

        model_name = model or self.default_model

        try:
            result = await self._client.messages.count_tokens(
                model=model_name,
                messages=[{"role": "user", "content": text}],
            )
            return result.input_tokens
        except Exception as e:
            logger.warning(
                f"Anthropic count_tokens API failed ({e}). Using character-based approximation."
            )
            return _EstimateCounter().count(text)

    async def count_message_tokens(self, messages: list[Message], model: str | None = None) -> int:
        """Count tokens for a list of messages using the Anthropic API.

        Uses ``messages.count_tokens()`` which accounts for message
        formatting, system prompts, and multimodal content.
        """
        if not self._client:
            logger.warning("Anthropic client not available. Approximating token count.")
            counter = _EstimateCounter()
            return sum(counter.count(m.content) for m in messages)

        model_name = model or self.default_model
        system_prompt, anthropic_msgs = self._convert_llmcore_msgs_to_anthropic(messages)

        if not anthropic_msgs:
            return 0

        try:
            ct_kwargs: dict[str, Any] = {
                "model": model_name,
                "messages": anthropic_msgs,
            }
            if system_prompt:
                ct_kwargs["system"] = system_prompt

            result = await self._client.messages.count_tokens(
                **ct_kwargs  # type: ignore[arg-type]
            )
            return result.input_tokens
        except Exception as e:
            logger.warning(
                f"Anthropic count_message_tokens API failed ({e}). "
                f"Using character-based approximation."
            )
            counter = _EstimateCounter()
            return sum(counter.count(m.content) for m in messages)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying Anthropic client session."""
        if self._http is not None:
            try:
                await self._http.aclose()
            except Exception as e:
                logger.warning("Error closing direct HTTP client: %s", e)
            finally:
                self._http = None
        if self._client:
            try:
                await self._client.close()
            except Exception as e:
                logger.debug(f"Error during Anthropic client cleanup: {e}")
        self._client = None
        logger.debug("AnthropicProvider closed.")
