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

Tested against anthropic SDK v2.8.0 and API version 2026-10-21 (live,
2026-11-16).
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from collections.abc import AsyncGenerator
from typing import Any, ClassVar

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
    # Public exports (the private ``anthropic._exceptions`` module is not API).
    from anthropic import (
        APIConnectionError as AnthropicConnectionError,
    )
    from anthropic import (
        APIStatusError,
        AsyncAnthropic,
        OverloadedError,
    )
    from anthropic import (
        APITimeoutError as AnthropicTimeoutError,
    )
    from anthropic import (
        AuthenticationError as AnthropicAuthError,
    )
    from anthropic import (
        BadRequestError as AnthropicBadRequestError,
    )
    from anthropic import (
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
# What each model accepts, as Anthropic's Models API states it.
# ---------------------------------------------------------------------------
#: Snapshot of ``GET /v1/models`` capabilities (2026-11-16): accepted thinking
#: types, accepted ``output_config.effort`` levels, max input and max output.
#: The provider reads the live list once per process and uses this only when
#: that fails (offline, a gateway without capabilities, a stub in tests). The
#: rules are per model, not per generation: Sonnet 5.5 alone takes
#: ``between_tools``, Opus 5.5 and every Fable model have no off switch at all,
#: Sonnet 4.6 has no ``xhigh``, and Haiku/Sonnet 4.5 take no effort level.
_E5 = frozenset({"low", "medium", "high", "xhigh", "max"})
_MEASURED_CAPABILITIES: dict[str, tuple[frozenset[str], frozenset[str], int, int]] = {
    "claude-fable-5-2": (frozenset({"adaptive"}), _E5, 1_000_000, 128_000),
    "claude-fable-5-1": (frozenset({"adaptive"}), _E5, 1_000_000, 128_000),
    "claude-fable-5": (frozenset({"adaptive"}), _E5, 1_000_000, 128_000),
    "claude-opus-5-5": (frozenset({"adaptive"}), _E5, 1_000_000, 128_000),
    "claude-sonnet-5-5": (frozenset({"adaptive", "between_tools"}), _E5, 1_000_000, 128_000),
    "claude-haiku-5-5": (frozenset({"adaptive", "disabled"}), _E5, 1_000_000, 128_000),
    "claude-opus-5": (frozenset({"adaptive", "disabled"}), _E5, 1_000_000, 128_000),
    "claude-sonnet-5": (frozenset({"adaptive", "disabled"}), _E5, 1_000_000, 128_000),
    "claude-opus-4-8": (frozenset({"adaptive", "disabled"}), _E5, 1_000_000, 128_000),
    "claude-opus-4-7": (frozenset({"adaptive", "disabled"}), _E5, 1_000_000, 128_000),
    "claude-sonnet-4-6": (
        frozenset({"enabled", "adaptive", "disabled"}),
        frozenset({"low", "medium", "high", "max"}),
        1_000_000,
        128_000,
    ),
    "claude-opus-4-6": (
        frozenset({"enabled", "adaptive", "disabled"}),
        frozenset({"low", "medium", "high", "max"}),
        1_000_000,
        128_000,
    ),
    "claude-opus-4-5-20251101": (
        frozenset({"enabled", "disabled"}),
        frozenset({"low", "medium", "high"}),
        200_000,
        64_000,
    ),
    "claude-haiku-4-5-20251001": (frozenset({"enabled", "disabled"}), frozenset(), 200_000, 64_000),
    "claude-sonnet-4-5-20250929": (frozenset({"enabled", "disabled"}), frozenset(), 200_000, 64_000),
}

# Default context lengths, kept for older ids the Models API no longer lists.
DEFAULT_ANTHROPIC_TOKEN_LIMITS: dict[str, int] = {
    **{model_id: caps[2] for model_id, caps in _MEASURED_CAPABILITIES.items()},
    "claude-opus-4-5": 200000,
    "claude-sonnet-4-5": 200000,
    "claude-haiku-4-5": 200000,
}

#: claude-sonnet-4-5-20250929 (the old default) retires 2026-11-30.
DEFAULT_MODEL = "claude-sonnet-5-5"


#: Public API root for the direct transport.
_ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"

#: Version header the Messages API requires on every direct request. The SDK
#: sends this for you; calling the API directly means owning it.
#: 2026-10-21 is what SDK 2.x sends. It removes temperature/top_p/top_k, and
#: rejects documents in assistant turns and citations in user turns.
_ANTHROPIC_VERSION = "2026-10-21"


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
        #: Read per-model capabilities from GET /v1/models once per process.
        #: Off for gateways whose model list carries no capabilities.
        self._capability_lookup = bool(config.get("capability_lookup", True))
        self._capabilities_loaded = False
        #: Server tool used for native_search=True.
        self._native_search_tool = dict(
            config.get("native_search_tool")
            or {"type": "web_search_20250305", "name": "web_search"}
        )

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

    @staticmethod
    def _is_context_overflow(text: str) -> bool:
        """Anthropic's prompt-too-long error ("prompt is too long: N tokens > M")."""
        lowered = text.lower()
        return "prompt is too long" in lowered or "model_context_window_exceeded" in lowered or (
            "context" in lowered and ("too long" in lowered or "exceed" in lowered)
        )

    def _raise_direct_status(self, status: int, body: str, model_name: str) -> None:
        """Map a direct-transport failure onto the same errors the SDK path uses.

        Raises:
            ContextLengthError: If the prompt exceeded the model's window.
            ProviderError: For every other failure.
        """
        if status == 400 and self._is_context_overflow(body):
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
                            # Each type is an always-present object; what matters
                            # is its ``supported`` flag (the old `is not None`
                            # check made every model "support thinking").
                            supports_thinking = any(
                                getattr(getattr(thinking_types, t, None), "supported", False)
                                for t in ("enabled", "adaptive")
                            )
                    self._remember_capabilities(model_info)

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

    # There are **two** boundaries here, not one, and they do not coincide.
    # Measured against the live API on 2026-10-01, one request per model:
    #
    #   generation   thinking.type=enabled (+budget_tokens)   thinking.type=adaptive
    #   <= 4.5       200 (the only option)                    400 "adaptive thinking
    #                                                             is not supported"
    #   4.6          200                                      200   <- both work
    #   >= 4.7       400 "thinking.type.enabled is not         200
    #                     supported for this model"
    #
    # So 4.6 is an overlap where either form is accepted, which is why a
    # single cutoff is wrong: assuming one at 4.6 discards a caller's explicit
    # token budget on a model that would have honoured it.

    #: Earliest generation that accepts ``{"type": "adaptive"}``.
    _ADAPTIVE_FROM = (4, 6)

    #: Earliest generation that **rejects** ``{"type": "enabled"}`` and its
    #: ``budget_tokens``. This is the one that turns a reasonable request into
    #: a 400, and it is the defect recorded in
    #: the provider modernization plan and the routing spec.
    _BUDGET_REJECTED_FROM = (4, 7)

    #: Live per-model capabilities read from ``GET /v1/models`` (shared by
    #: every instance in the process): model id -> (thinking types, effort
    #: levels, max input, max output). Falls back to ``_MEASURED_CAPABILITIES``.
    _LIVE_CAPABILITIES: ClassVar[dict[str, tuple[frozenset[str], frozenset[str], int, int]]] = {}

    #: Models seen to reject a forced tool_choice ("any"/"tool"); see
    #: ``chat_completion``.
    _NO_FORCED_TOOL_CHOICE: ClassVar[set[str]] = set()

    _warned_sampling: ClassVar[bool] = False

    #: Anthropic requires ``max_tokens`` **strictly greater** than
    #: ``thinking.budget_tokens`` -- equal values are rejected -- and a budget
    #: below this floor is rejected outright.
    _MIN_BUDGET_TOKENS = 1024

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

    @classmethod
    def _capabilities(
        cls, model: str | None
    ) -> tuple[frozenset[str], frozenset[str], int, int] | None:
        """What ``model`` accepts: live Models API data first, then the snapshot.

        A dated id (``claude-opus-4-5-20251101``) and its undated alias
        (``claude-opus-4-5``) resolve to each other.
        """
        if not model:
            return None
        for table in (cls._LIVE_CAPABILITIES, _MEASURED_CAPABILITIES):
            if model in table:
                return table[model]
            for known, caps in table.items():
                if known.startswith(model + "-") or model.startswith(known + "-"):
                    return caps
        return None

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
        # A one- or two-digit minor, then a date or the end: "claude-sonnet-4-
        # 20250514" is generation 4.0, not (4, 20250514) as the old pattern read
        # it, and "claude-opus-5" (no minor) is 5.0.
        match = re.search(r"-(\d)-(\d{1,2})(?:-\d{8}|$)", model)
        if match:
            return int(match.group(1)), int(match.group(2))
        match = re.search(r"-(\d)(?:-\d{8}|$)", model)
        if match:
            return int(match.group(1)), 0
        return None

    @classmethod
    def _supports_adaptive_thinking(cls, model: str | None) -> bool:
        """Whether *model* accepts ``{"type": "adaptive"}``."""
        caps = cls._capabilities(model)
        if caps is not None:
            return "adaptive" in caps[0]
        generation = cls._model_generation(model)
        if generation is None:
            # Unknown id: assume the newest behaviour. The older family is the
            # shrinking set, so defaulting the other way would break every new
            # model by default.
            return True
        return generation >= cls._ADAPTIVE_FROM

    @classmethod
    def _rejects_budget_tokens(cls, model: str | None) -> bool:
        """Whether *model* answers 400 to ``{"type": "enabled"}``."""
        caps = cls._capabilities(model)
        if caps is not None:
            return "enabled" not in caps[0]
        generation = cls._model_generation(model)
        if generation is None:
            return True
        return generation >= cls._BUDGET_REJECTED_FROM

    @classmethod
    def _no_thinking_block(cls, model: str | None) -> dict[str, str] | None:
        """The way to say "do not think" that *model* accepts, or ``None``.

        Measured live 2026-11-16: Haiku 5.5 and Sonnet 4.6 take ``disabled``,
        Sonnet 5.5 takes only ``between_tools``, and Opus 5.5 and every Fable
        model take neither ("Use thinking.type.adaptive and
        output_config.effort"). ``None`` means omit ``thinking`` and ask for
        the lowest effort, which is as close to "do not think" as those models
        go. An unknown modern model gets ``None`` too, which no model rejects.
        """
        caps = cls._capabilities(model)
        if caps is not None:
            if "disabled" in caps[0]:
                return {"type": "disabled"}
            if "between_tools" in caps[0]:
                return {"type": "between_tools"}
            return None
        generation = cls._model_generation(model)
        if generation is not None and generation < (4, 7):
            return {"type": "disabled"}
        return None

    @classmethod
    def _fold_effort(cls, effort: str, model: str | None) -> str | None:
        """llmcore's effort level as an ``output_config.effort`` *model* accepts.

        ``None`` means the model takes no effort level at all (Haiku and
        Sonnet 4.5), so none is sent.
        """
        wanted = cls._EFFORT_TO_OUTPUT_CONFIG.get(effort, "medium")
        caps = cls._capabilities(model)
        if caps is None:
            return wanted
        accepted = caps[1]
        if not accepted:
            return None
        if effort in accepted:
            return effort  # xhigh is real on 4.7+; the table folds it for older ones
        if wanted in accepted:
            return wanted
        order = ["low", "medium", "high", "xhigh", "max"]
        target = order.index(wanted) if wanted in order else 1
        return min(accepted, key=lambda level: (abs(order.index(level) - target), -order.index(level)))

    @classmethod
    def _normalize_thinking(
        cls,
        thinking: Any,
        effort: str | None,
        model: str | None,
        max_tokens: int | None = None,
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
        rejects_budget = cls._rejects_budget_tokens(model)
        block: dict[str, Any] | None = dict(thinking) if isinstance(thinking, dict) else None
        effort_key = str(effort).strip().lower() if effort else None

        if adaptive:
            # Only convert when the model would actually refuse the budget.
            # On 4.6 both forms are accepted, so a caller who asked for a
            # precise token budget keeps it -- discarding it there would be
            # overriding an explicit, valid choice.
            if block and "budget_tokens" in block and rejects_budget:
                dropped = block.pop("budget_tokens")
                logger.warning(
                    "Dropping thinking.budget_tokens=%s: %s answers 400 to "
                    "thinking.type=enabled ('not supported for this model'). "
                    "Claude 4.7 and later take thinking={'type': 'adaptive'} plus "
                    "output_config.effort. Pass effort= instead.",
                    dropped,
                    model,
                )
                block["type"] = "adaptive"
            elif block and block.get("type") == "enabled" and rejects_budget:
                block["type"] = "adaptive"
            if block and "budget_tokens" in block and not rejects_budget:
                # Honoured as given; nothing to map.
                return block, None
            if effort_key == "none":
                # "none" means do not think. Where a model has an off switch it
                # is used; where it has none (Opus 5.5, Fable), the lowest
                # effort is the closest thing, and anything else is a 400.
                off = cls._no_thinking_block(model)
                if off is not None:
                    return off, None
                return None, cls._fold_effort("low", model)
            if effort_key:
                if block is None:
                    block = {"type": "adaptive"}
                return block, cls._fold_effort(effort_key, model)
            return block, None

        # Pre-4.6: a budget, not an effort level.
        if effort_key:
            budget = cls._EFFORT_TO_BUDGET.get(effort_key, 8192)
            if budget <= 0:
                return cls._no_thinking_block(model), None
            budget = cls._fit_budget(budget, max_tokens)
            if budget is None:
                logger.warning(
                    "effort=%r maps to a thinking budget that does not fit "
                    "max_tokens=%s on %s (Anthropic needs max_tokens > budget, and a "
                    "budget of at least %d). Disabling thinking for this call; raise "
                    "max_tokens to get reasoning.",
                    effort_key,
                    max_tokens,
                    model,
                    cls._MIN_BUDGET_TOKENS,
                )
                return cls._no_thinking_block(model), None
            merged = block or {}
            merged.setdefault("type", "enabled")
            merged.setdefault("budget_tokens", budget)
            return merged, None
        if block and block.get("type") == "adaptive":
            fitted = cls._fit_budget(cls._EFFORT_TO_BUDGET["medium"], max_tokens)
            if fitted is None:
                logger.warning(
                    "%s does not support adaptive thinking, and max_tokens=%s leaves no "
                    "room for a budget; disabling thinking for this call.",
                    model,
                    max_tokens,
                )
                return cls._no_thinking_block(model), None
            logger.warning(
                "%s does not support adaptive thinking; converting to "
                "{'type': 'enabled', 'budget_tokens': %d}.",
                model,
                fitted,
            )
            block = {"type": "enabled", "budget_tokens": fitted}
        elif block and block.get("type") == "enabled" and "budget_tokens" in block:
            # A caller's own budget still has to fit, and the API rejects
            # equality rather than clamping.
            fitted = cls._fit_budget(int(block["budget_tokens"]), max_tokens)
            if fitted is None:
                logger.warning(
                    "thinking.budget_tokens=%s does not fit max_tokens=%s on %s; "
                    "disabling thinking for this call.",
                    block["budget_tokens"],
                    max_tokens,
                    model,
                )
                return cls._no_thinking_block(model), None
            if fitted != block["budget_tokens"]:
                logger.info(
                    "Reducing thinking.budget_tokens from %s to %d so it fits "
                    "max_tokens=%s.",
                    block["budget_tokens"],
                    fitted,
                    max_tokens,
                )
            block["budget_tokens"] = fitted
        return block, None

    @classmethod
    def _fit_budget(cls, budget: int, max_tokens: int | None) -> int | None:
        """Largest usable budget at or below *budget*, or ``None`` if none fits.

        Anthropic requires ``max_tokens`` **strictly greater** than
        ``budget_tokens`` -- equal values are rejected, which is easy to get
        wrong -- and rejects a budget below
        :attr:`_MIN_BUDGET_TOKENS`. Between those two constraints there is
        simply no valid budget when ``max_tokens`` is small, and the honest
        answer there is to turn thinking off rather than send a request that
        cannot succeed.
        """
        if max_tokens is None:
            return budget
        ceiling = int(max_tokens) - 1
        if ceiling < cls._MIN_BUDGET_TOKENS:
            return None
        return min(budget, ceiling)

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
            "thinking_display": {
                "type": "string",
                "enum": ["summarized", "omitted"],
                "description": (
                    "Whether thinking text comes back. Claude 4.7+ and 5.x omit it "
                    "by default; 'summarized' returns a readable summary."
                ),
            },
            "native_search": {"type": "boolean", "description": "Add the web-search server tool."},
            "extra_tools": {
                "type": "array",
                "description": "Raw tool declarations, e.g. server tools (web_fetch, code_execution).",
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
        caps = self._capabilities(model_name)
        if caps is not None and caps[2]:
            return caps[2]

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

        # Final fallback. Every current Claude model has 1M except the 4.5 line,
        # so an unknown 4.6+ id gets 1M and anything older 200K.
        generation = self._model_generation(model_name)
        fallback = 1_000_000 if generation is None or generation >= (4, 6) else 200_000
        logger.warning(
            "Unknown context length for Anthropic model '%s'. Using fallback: %d.",
            model_name,
            fallback,
        )
        return fallback

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

    @staticmethod
    def _fit_blocks_to_role(blocks: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
        """Apply API 2026-10-21's per-role content rules.

        A ``document`` block in an assistant turn and a ``text`` block carrying
        ``citations`` in a user turn are both 400s from that version on.
        Documents are dropped from assistant turns with a warning, and
        citations are stripped from user text (the text itself is kept).
        """
        if role == "assistant":
            kept = [b for b in blocks if b.get("type") != "document"]
            if len(kept) != len(blocks):
                logger.warning(
                    "Dropping document block(s) from an assistant turn: Anthropic rejects "
                    "documents outside user turns (API 2026-10-21)."
                )
            return kept
        if role == "user":
            out = []
            for block in blocks:
                if block.get("type") == "text" and block.get("citations"):
                    block = {k: v for k, v in block.items() if k != "citations"}
                out.append(block)
            return out
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
            raw_blocks = metadata.get("anthropic_content") if role_str == "assistant" else None
            if raw_blocks:
                # Replayed verbatim: thinking signatures, redacted thinking and
                # server-tool blocks have to come back exactly as received.
                anthropic_messages.append({"role": "assistant", "content": list(raw_blocks)})
                continue
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
                content_blocks = self._fit_blocks_to_role(content_blocks, role_str)
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
            # The prompt plus output reached the model's window: as "length" as
            # a max_tokens stop, and callers check for that, not the raw value.
            "model_context_window_exceeded": "length",
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
        """Send a chat request, shaped to what the model accepts.

        On top of :meth:`_chat_completion_once`:

        - Per-model capabilities are read from the Models API once (thinking
          types, effort levels, windows), so "no thinking" and effort levels
          are sent in the form each model takes.
        - ``temperature``/``top_p``/``top_k`` are dropped with a one-time
          warning. API version 2026-10-21 rejects them on every model, and
          Claude 5.x rejected them before that.
        - A forced ``tool_choice`` (``"any"`` or a tool name) on a model that
          rejects it (Opus 5.5, Sonnet 5.5, Fable) is sent as ``"auto"`` with a
          warning, and the model is remembered.
        - ``native_search=True`` adds Anthropic's web-search server tool, and
          ``extra_tools=[...]`` passes raw tool declarations (server tools).
        """
        model_name = model or self.default_model
        await self._ensure_capabilities()
        dropped = [k for k in ("temperature", "top_p", "top_k") if kwargs.pop(k, None) is not None]
        if dropped and not AnthropicProvider._warned_sampling:
            AnthropicProvider._warned_sampling = True
            logger.warning(
                "Dropping %s for Anthropic: API version 2026-10-21 removed sampling "
                "parameters (Claude 5.x rejected them before that). Use effort= instead.",
                ", ".join(dropped),
            )
        for attempt in (0, 1):
            choice = tool_choice
            if (
                choice not in (None, "auto", "none")
                and model_name in AnthropicProvider._NO_FORCED_TOOL_CHOICE
            ):
                choice = "auto"
            try:
                return await self._chat_completion_once(
                    context, model_name, stream, tools, choice, **dict(kwargs)
                )
            except ProviderError as exc:
                if (
                    attempt == 0
                    and choice not in (None, "auto", "none")
                    and self._rejects_forced_tool_choice(str(exc))
                ):
                    AnthropicProvider._NO_FORCED_TOOL_CHOICE.add(model_name)
                    logger.warning(
                        "%s rejects a forced tool_choice; sending tool_choice='auto' instead "
                        "(remembered for this model).",
                        model_name,
                    )
                    continue
                raise
        raise AssertionError("unreachable")  # pragma: no cover

    @staticmethod
    def _rejects_forced_tool_choice(text: str) -> bool:
        """Anthropic's 'tool_choice: type "tool" and "any" are not supported'.

        Matched without the quotes: on the direct transport they arrive JSON-
        escaped inside the error body.
        """
        return "tool_choice" in text and "are not supported for this model" in text

    def _remember_capabilities(self, model_info: Any) -> None:
        """Record one Models API entry's thinking types, effort levels, limits."""
        info = model_info.model_dump() if hasattr(model_info, "model_dump") else dict(model_info)
        model_id = info.get("id")
        caps = info.get("capabilities") or {}
        if not model_id or not isinstance(caps, dict) or not caps.get("thinking"):
            return  # e.g. a gateway's catalog: nothing to learn
        types = frozenset(
            name
            for name, flag in ((caps.get("thinking") or {}).get("types") or {}).items()
            if isinstance(flag, dict) and flag.get("supported")
        )
        efforts = frozenset(
            name
            for name, flag in (caps.get("effort") or {}).items()
            if isinstance(flag, dict) and flag.get("supported")
        )
        AnthropicProvider._LIVE_CAPABILITIES[model_id] = (
            types,
            efforts,
            int(info.get("max_input_tokens") or 0),
            int(info.get("max_tokens") or 0),
        )

    async def _ensure_capabilities(self) -> None:
        """Read the Models API once; on any failure keep the measured snapshot."""
        if self._capabilities_loaded or not self._capability_lookup:
            return
        self._capabilities_loaded = True
        try:
            if self._backend == "httpx":
                resp = await self._get_http().get("/v1/models", params={"limit": 1000})
                if resp.status_code >= 400:
                    raise RuntimeError(f"HTTP {resp.status_code}")
                for entry in resp.json().get("data") or []:
                    self._remember_capabilities(entry)
            else:
                page = await self._client.models.list(limit=1000)  # type: ignore[union-attr]
                for entry in page.data:
                    self._remember_capabilities(entry)
        except Exception as exc:  # the snapshot is a complete fallback
            logger.debug("Anthropic capability lookup failed (%s); using the snapshot.", exc)

    async def _chat_completion_once(
        self,
        context: ContextPayload,
        model: str | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Send one chat completion request to the Anthropic API.

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
            kwargs.pop("thinking", None),
            kwargs.pop("effort", None),
            model_name,
            max_tokens=api_kwargs.get("max_tokens"),
        )
        display = kwargs.pop("thinking_display", None)
        if display and thinking is None and self._supports_adaptive_thinking(model_name):
            thinking = {"type": "adaptive"}
        if display and thinking and thinking.get("type") in ("adaptive", "enabled"):
            # 4.7+ and every 5.x model omit thinking text by default (only the
            # signature comes back); "summarized" returns a readable summary.
            thinking = {**thinking, "display": display}
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

        # Tools: llmcore tools, then raw server tools.
        native_search = bool(kwargs.pop("native_search", False))
        extra_tools = list(kwargs.pop("extra_tools", None) or [])
        if native_search:
            extra_tools.append(dict(self._native_search_tool))
        if tools or extra_tools:
            api_kwargs["tools"] = self._convert_tools_to_anthropic(tools or []) + [
                dict(t) for t in extra_tools
            ]
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
            # Only the prompt-too-long error: the old `"token" in message` test
            # turned every 400 that mentioned max_tokens or budget_tokens into a
            # ContextLengthError.
            if self._is_context_overflow(err_msg):
                raise ContextLengthError(
                    model_name=model_name,
                    limit=self.get_max_context_length(model_name),
                    message=f"Anthropic context length exceeded: {e}",
                ) from e
            raise ProviderError(
                self.get_name(), f"Bad request: {e}", model_name=model_name,
                status_code=400, retryable=False, original_exception=e,
            ) from e
        except AnthropicAuthError as e:
            raise ProviderError(
                self.get_name(), f"Authentication failed: {e}", model_name=model_name,
                status_code=401, retryable=False, original_exception=e,
            ) from e
        except AnthropicRateLimitError as e:
            raise ProviderError(
                self.get_name(), f"Rate limited: {e}", model_name=model_name,
                status_code=429, retryable=True, original_exception=e,
            ) from e
        except OverloadedError as e:
            raise ProviderError(
                self.get_name(), f"API overloaded: {e}", model_name=model_name,
                status_code=529, retryable=True, original_exception=e,
            ) from e
        except APIStatusError as e:
            status = getattr(e, "status_code", None)
            logger.debug("Anthropic API error (%s): %s", status, e)
            raise ProviderError(
                self.get_name(),
                f"API Error (Status: {status if status is not None else 'N/A'}): {e}",
                model_name=model_name,
                status_code=status,
                original_exception=e,
            ) from e
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
            # `system` is omitted rather than passed as None. The SDK
            # serialises an explicit None into the body as `"system": null`,
            # and the API answers 400 "system: Input should be a valid array"
            # -- so on this transport *every* call without a system message
            # failed. The httpx path already omitted it; the two transports
            # have to agree.
            sdk_kwargs: dict[str, Any] = dict(api_kwargs)
            if system_prompt is not None:
                sdk_kwargs["system"] = system_prompt
            response = await self._client.messages.create(  # type: ignore[union-attr]
                model=model_name,
                messages=messages_payload,  # type: ignore[arg-type]
                **sdk_kwargs,
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
        if any(block.get("type") not in ("text", "tool_use") for block in content_blocks):
            # Thinking blocks (with their signatures), redacted thinking and
            # server-tool blocks must go back complete and in order on the next
            # assistant turn of a tool loop. The OpenAI-shaped message above
            # cannot carry them, so the raw blocks travel alongside; see
            # assistant_replay_metadata().
            message_dict["anthropic_content"] = content_blocks
        if response_dict.get("stop_details"):
            message_dict["stop_details"] = response_dict["stop_details"]
        if response_dict.get("stop_reason") == "pause_turn":
            message_dict["pause_turn"] = True  # resumable server-tool turn

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
            # Omitted when absent, for the same reason as the non-streaming
            # path: an explicit None becomes `"system": null` and is rejected.
            sdk_kwargs: dict[str, Any] = dict(api_kwargs)
            if system_prompt is not None:
                sdk_kwargs["system"] = system_prompt
            raw_stream = await self._client.messages.create(  # type: ignore[union-attr]
                model=model_name,
                messages=messages_payload,  # type: ignore[arg-type]
                stream=True,
                **sdk_kwargs,
            )
            event_source = self._sdk_message_events(raw_stream)

        # Track state for streaming tool calls
        current_tool_id: str | None = None
        current_tool_name: str | None = None
        # The raw content blocks, rebuilt from the events, so the final chunk
        # can carry them for replay (thinking text + signature, server tools).
        blocks: dict[int, dict[str, Any]] = {}
        partial_json: dict[int, str] = {}

        def track(event: dict[str, Any]) -> None:
            kind = event.get("type")
            index = int(event.get("index", 0) or 0)
            if kind == "content_block_start":
                blocks[index] = dict(event.get("content_block") or {})
            elif kind == "content_block_delta" and index in blocks:
                delta = event.get("delta") or {}
                block = blocks[index]
                dtype = delta.get("type")
                if dtype == "text_delta":
                    block["text"] = block.get("text", "") + delta.get("text", "")
                elif dtype == "thinking_delta":
                    block["thinking"] = block.get("thinking", "") + delta.get("thinking", "")
                elif dtype == "signature_delta":
                    block["signature"] = delta.get("signature", "")
                elif dtype == "input_json_delta":
                    partial_json[index] = partial_json.get(index, "") + delta.get("partial_json", "")
            elif kind == "content_block_stop" and index in partial_json:
                try:
                    blocks[index]["input"] = json.loads(partial_json.pop(index) or "{}")
                except ValueError:
                    pass

        def raw_blocks() -> list[dict[str, Any]] | None:
            ordered = [blocks[i] for i in sorted(blocks)]
            if any(b.get("type") not in ("text", "tool_use") for b in ordered):
                return ordered
            return None

        async def stream_wrapper() -> AsyncGenerator[dict[str, Any], None]:
            nonlocal current_tool_id, current_tool_name

            async for event_dict in event_source:
                event_type = event_dict.get("type")
                track(event_dict)

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
                    final: dict[str, Any] = {"type": "message_stop"}
                    replay = raw_blocks()
                    if replay is not None:
                        final["choices"] = [{"delta": {"anthropic_content": replay}}]
                    yield final
                    continue

                # Pass through any unknown events
                yield event_dict

        return stream_wrapper()

    # ------------------------------------------------------------------
    # Response Extraction
    # ------------------------------------------------------------------

    def assistant_replay_metadata(self, response: dict[str, Any]) -> dict[str, Any]:
        """Metadata to attach to the next assistant ``Message`` for a faithful replay.

        Thinking blocks (with signatures), redacted thinking and server-tool
        blocks have to go back complete and in order within a tool loop. Put
        the returned dict in the assistant message's ``metadata`` and the turn
        is replayed verbatim. For a stream, pass the last chunk.
        """
        try:
            choice = (response.get("choices") or [{}])[0]
        except (AttributeError, IndexError, TypeError):
            return {}
        source = choice.get("message") or choice.get("delta") or {}
        blocks = source.get("anthropic_content")
        return {"anthropic_content": list(blocks)} if blocks else {}

    def supports_native_search(self, model: str | None = None) -> bool:
        """Anthropic's web-search server tool is available on current models."""
        return True

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
