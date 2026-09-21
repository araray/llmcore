# src/llmcore/providers/friendli_provider.py
"""
FriendliAI provider implementation for the LLMCore library.

Handles interactions with all three FriendliAI inference surfaces:

- **Model APIs** (``serverless``) — the hosted, pay-per-token catalog
  (``zai-org/GLM-5.3``, ``deepseek-ai/DeepSeek-V3.2``, ``google/gemma-4-31B-it``,
  ``MiniMaxAI/MiniMax-M2.5``, …) at ``https://api.friendli.ai/serverless/v1``.
- **Dedicated Endpoints** (``dedicated``) — your own GPU deployments at
  ``https://api.friendli.ai/dedicated/v1``; the ``model`` field is the
  *endpoint ID* (optionally ``ENDPOINT_ID:ADAPTER_ROUTE`` for Multi-LoRA).
- **Friendli Container** (``container``) — a self-hosted Friendli Engine; set
  ``base_url`` to your container (e.g. ``http://localhost:8000/v1``).

The chat endpoint is OpenAI-compatible (``POST /chat/completions``) with a set
of Friendli-specific extensions handled here:

- **Reasoning controls** — ``reasoning_effort``
  (``minimal|low|medium|high|xhigh|max|ultracode``), ``reasoning_budget``
  (token cap on the chain of thought), ``parse_reasoning`` (split the chain of
  thought out of ``content`` into ``reasoning_content``), and
  ``include_reasoning``.
- **Chat-template kwargs** — ``chat_template_kwargs`` carries the per-model
  template switches documented by FriendliAI, notably ``enable_thinking``
  (controllable reasoning models) and ``clear_thinking``.  Both are also
  accepted as flat kwargs and folded into ``chat_template_kwargs`` for you.
- **Friendli Engine sampling** — ``top_k``, ``min_p``, ``min_tokens``,
  ``repetition_penalty``, ``eos_token``, and XTC sampling
  (``xtc_threshold`` / ``xtc_probability``).
- **Structured output** — ``response_format`` of ``json_schema``,
  ``json_object``, ``regex`` (Friendli-specific), or ``text``.
- **Exact token counting** via the native ``POST /tokenize`` endpoint, with a
  tiktoken/heuristic fallback; ``POST /detokenize`` and ``POST /chat/render``
  are exposed as auxiliary helpers.
- **Cache-aware usage** — ``usage.prompt_tokens_details.cached_tokens``.
- **Team scoping** — every request carries the ``X-Friendli-Team`` header when
  a team ID is configured, and :meth:`get_team_cost` / :meth:`get_team_usage`
  read the Friendli Suite billing APIs for that team.

Transport (selectable via the ``backend`` config key)
-----------------------------------------------------

- ``"openai"`` — the ``openai`` SDK (``AsyncOpenAI``) pointed at the Friendli
  base URL.  Native async, full SSE handling; **the default**.
- ``"httpx"`` — direct async REST calls against the documented endpoints.
- ``"sdk"`` — the official ``friendli`` Python SDK (``AsyncFriendli``).

The backend governs **chat completions**.  Endpoints outside the OpenAI-compatible
surface — the model catalog, ``/tokenize``, ``/detokenize``, ``/chat/render``,
``/completions``, ``/embeddings``, ``/images/generations``,
``/audio/transcriptions`` and the Suite billing reads — always travel over the
provider's own ``httpx`` client (or the vendor SDK when ``backend = "sdk"``),
because the ``openai`` SDK either does not model them or would discard the extra
fields Friendli returns.

When unset, the backend auto-resolves ``openai`` → ``httpx`` → ``sdk`` based on
which libraries are installed.  The vendor SDK is deliberately *last*: its
generated response models are strict (``extra`` is ignored), so fields Friendli
adds outside the published schema — including ``reasoning_content`` and
``reasoning`` on assistant messages — are silently dropped, and it offers no
``extra_body`` escape hatch.  Request ``backend = "sdk"`` explicitly if you want
it anyway; this provider logs a warning when reasoning parsing is combined with
the SDK backend.

References:
  - https://friendli.ai/docs/llms.txt          (documentation index)
  - https://friendli.ai/docs/guides/openai-compatibility
  - https://friendli.ai/docs/openapi/model-apis/chat-completions
  - https://friendli.ai/docs/guides/capabilities/reasoning
  - https://friendli.ai/docs/guides/structured-outputs
  - Friendli Python SDK (``friendli``)
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
from collections.abc import AsyncGenerator
from typing import Any, Literal

# --- Optional official Friendli SDK (``friendli``) ---
# Native async client.  NOTE: its generated response models drop unknown
# fields, so reasoning_content is lost on this backend (see module docstring).
try:
    from friendli import AsyncFriendli
    from friendli.models import FriendliCoreError

    friendli_sdk_available = True
except ImportError:
    friendli_sdk_available = False
    AsyncFriendli = None  # type: ignore
    FriendliCoreError = Exception  # type: ignore

# --- Optional OpenAI SDK (OpenAI-compatibility backend) ---
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
    import httpx

    httpx_available = True
except ImportError:  # pragma: no cover - httpx is a hard dep of openai
    httpx_available = False
    httpx = None  # type: ignore

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
from ..models_multimodal import (
    GeneratedImage,
    ImageGenerationResult,
    TranscriptionResult,
)
from ..tokens import EstimateCounter as _EstimateCounter
from .base import BaseProvider, ContextPayload

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Inference base URLs per endpoint type.  ``container`` has no default — it is
#: self-hosted, so ``base_url`` is mandatory there.
_BASE_URLS: dict[str, str] = {
    "serverless": "https://api.friendli.ai/serverless/v1",
    "dedicated": "https://api.friendli.ai/dedicated/v1",
}

#: Friendli Suite (team billing/usage) API root — distinct from inference.
_SUITE_BASE_URL = "https://api.friendli.ai/v1"

#: Default model when none is configured (Model APIs flagship).
_DEFAULT_MODEL = "zai-org/GLM-5.3"

#: Endpoint types this provider understands.
EndpointType = Literal["serverless", "dedicated", "container"]

#: Transport backends, in auto-resolution preference order.
_BACKEND_ORDER: tuple[str, ...] = ("openai", "httpx", "sdk")

#: Reasoning-effort tiers accepted by the Friendli chat API.  The tiers a given
#: model actually supports are advertised per-model in ``/models``
#: (``reasoning_options``); unsupported tiers are rejected by the server.
_VALID_EFFORTS: frozenset[str] = frozenset(
    {"minimal", "low", "medium", "high", "xhigh", "max", "ultracode"}
)

#: Static context-length fallback for Model APIs models, used only when live
#: discovery and the model-card registry are both unavailable.
_CONTEXT_LENGTHS: dict[str, int] = {
    "zai-org/GLM-5.3": 1_048_576,
    "zai-org/GLM-5.3-Flash": 1_048_576,
    "zai-org/GLM-5.2": 1_048_576,
    "zai-org/GLM-5.1": 202_752,
    "google/gemma-4-31B-it": 262_144,
    "deepseek-ai/DeepSeek-V3.2": 163_840,
    "MiniMaxAI/MiniMax-M2.5": 196_608,
}

#: Conservative context length when nothing else is known.
_FALLBACK_CONTEXT_LENGTH = 131_072

#: Environment variables checked (in order) for the Friendli API key.
#: ``FRIENDLI_TOKEN`` is the official SDK convention; ``FRIENDLIAI_API_KEY`` is
#: the spelling used throughout friendli.ai's own documentation examples.
_API_KEY_ENV_VARS: tuple[str, ...] = (
    "FRIENDLI_TOKEN",
    "FRIENDLIAI_API_KEY",
    "FRIENDLI_API_KEY",
)

#: Environment variables checked (in order) for the Friendli team ID.
_TEAM_ID_ENV_VARS: tuple[str, ...] = ("FRIENDLI_TEAM_ID", "FRIENDLIAI_TEAM_ID")

#: Placeholder key for self-hosted containers launched without auth (the
#: ``openai`` SDK refuses an empty API key).
_CONTAINER_PLACEHOLDER_KEY = "EMPTY"

#: Request-body keys that are Friendli-specific, i.e. not native ``openai``
#: chat-completion parameters.  They travel via ``extra_body`` on the openai
#: backend and as plain body keys everywhere else.
_EXTRA_BODY_KEYS: frozenset[str] = frozenset(
    {
        "chat_template_kwargs",
        "eos_token",
        "include_reasoning",
        "min_p",
        "min_tokens",
        "parse_reasoning",
        "reasoning_budget",
        "reasoning_effort",
        "repetition_penalty",
        "top_k",
        "xtc_probability",
        "xtc_threshold",
    }
)

#: Flat kwargs folded into ``chat_template_kwargs`` (Friendli passes these to
#: the model's chat template rather than treating them as body parameters).
_TEMPLATE_KWARG_KEYS: frozenset[str] = frozenset({"enable_thinking", "clear_thinking"})


def _looks_like_context_overflow(message: str) -> bool:
    """Heuristic: does *message* describe a context/length overflow?"""
    low = message.lower()
    if "max_tokens" in low and "exceed" in low:
        return True
    return ("context" in low or "prompt" in low or "input" in low) and (
        "too long" in low or "length" in low or "exceed" in low
    )


class FriendliProvider(BaseProvider):
    """First-class FriendliAI provider (Model APIs / Dedicated / Container).

    Configuration keys (under ``[providers.friendli]``):

    - ``api_key`` / ``api_key_env_var`` — Friendli Personal API key (``flp_…``).
      Resolved from the config, then ``api_key_env_var``, then
      ``FRIENDLI_TOKEN`` / ``FRIENDLIAI_API_KEY`` / ``FRIENDLI_API_KEY``.
      Optional when ``endpoint_type = "container"``.
    - ``team_id`` / ``team_id_env_var`` — team to run requests as, sent as the
      ``X-Friendli-Team`` header.  Falls back to ``FRIENDLI_TEAM_ID`` /
      ``FRIENDLIAI_TEAM_ID``.
    - ``endpoint_type`` — ``"serverless"`` (default), ``"dedicated"``, or
      ``"container"``.
    - ``base_url`` — override the inference root.  Required for ``container``.
    - ``suite_base_url`` — override the Suite billing root
      (default ``https://api.friendli.ai/v1``).
    - ``backend`` — ``"openai"`` (default), ``"httpx"``, or ``"sdk"``.  Omit or
      use ``"auto"`` to resolve openai → httpx → sdk.
    - ``default_model`` — default model ID (Model APIs) or endpoint ID
      (Dedicated).  Default: ``zai-org/GLM-5.3``.
    - ``timeout`` — HTTP request timeout in seconds (default: 300).
    - ``reasoning_effort`` — default effort tier, or unset to leave it to the
      model's own default.
    - ``reasoning_budget`` — default cap on reasoning tokens.
    - ``parse_reasoning`` — split reasoning into ``reasoning_content``
      (default: ``true``).
    - ``include_reasoning`` — include parsed reasoning in the response.
    - ``enable_thinking`` — default ``chat_template_kwargs.enable_thinking``
      for controllable reasoning models.
    - ``native_token_count`` — count tokens with the model's own tokenizer via
      ``POST /tokenize`` instead of locally (default: ``false``; each count is
      an extra API request).
    - ``fallback_context_length`` — context window used when neither live
      discovery nor a model card knows the model (default: 131072).
    """

    default_model: str
    _backend: str  # "openai" | "httpx" | "sdk"
    _endpoint_type: str
    _api_key: str
    _base_url: str
    _suite_base_url: str
    _team_id: str | None
    _timeout: float
    _client: Any  # AsyncOpenAI | None
    _sdk_client: Any  # AsyncFriendli | None
    _http: Any  # httpx.AsyncClient | None
    _encoding: Any  # tiktoken.Encoding | None
    _catalog: dict[str, dict[str, Any]] | None

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False):
        """Initialize the FriendliAI provider.

        Args:
            config: Provider configuration dict from ``[providers.friendli]``.
            log_raw_payloads: Whether to log raw request/response payloads.

        Raises:
            ConfigError: If no transport library is installed, no API key is
                available for a hosted endpoint type, or ``base_url`` is
                missing for ``endpoint_type = "container"``.
        """
        super().__init__(config, log_raw_payloads)

        if not (openai_available or httpx_available or friendli_sdk_available):
            raise ConfigError(
                "The Friendli provider requires one of: the 'openai' SDK "
                "(compatibility mode, preferred), 'httpx' (direct REST), or the "
                "official 'friendli' SDK. Install with: pip install llmcore[friendli]"
            )

        # --- Endpoint type ---
        endpoint_type = str(config.get("endpoint_type", "serverless")).lower()
        if endpoint_type not in ("serverless", "dedicated", "container"):
            logger.warning(
                "Invalid Friendli endpoint_type '%s'; defaulting to 'serverless'.",
                endpoint_type,
            )
            endpoint_type = "serverless"
        self._endpoint_type = endpoint_type

        # --- Endpoint URL ---
        base_url = config.get("base_url")
        if not base_url:
            if endpoint_type == "container":
                raise ConfigError(
                    "Friendli Container requires an explicit base_url "
                    '(e.g. providers.friendli.base_url = "http://localhost:8000/v1").'
                )
            base_url = _BASE_URLS[endpoint_type]
        self._base_url = str(base_url).rstrip("/")
        self._suite_base_url = str(config.get("suite_base_url", _SUITE_BASE_URL)).rstrip("/")

        # --- API key ---
        api_key = self._resolve_api_key(config)
        if not api_key:
            if endpoint_type == "container":
                # Containers are frequently launched without auth.
                api_key = _CONTAINER_PLACEHOLDER_KEY
            else:
                raise ConfigError(
                    "Friendli API key not found. Set FRIENDLI_TOKEN (or "
                    "FRIENDLIAI_API_KEY) or configure providers.friendli.api_key / "
                    "api_key_env_var. Create a key at "
                    "https://friendli.ai/suite/~/setting/keys."
                )
        self._api_key = api_key

        # --- Team scoping (X-Friendli-Team) ---
        self._team_id = self._resolve_team_id(config)

        # --- Model / timeout ---
        self.default_model = config.get("default_model", _DEFAULT_MODEL)
        self._timeout = float(config.get("timeout", 300))
        self._fallback_context_length = int(
            config.get("fallback_context_length", _FALLBACK_CONTEXT_LENGTH)
        )
        # Exact per-model token counts cost one API round-trip each and consume
        # the Model APIs request budget, so count_tokens()/count_message_tokens()
        # stay local by default.  tokenize()/detokenize() are always available.
        self._native_token_count = bool(config.get("native_token_count", False))

        # --- Reasoning defaults ---
        effort_raw = config.get("reasoning_effort")
        self._default_reasoning_effort: str | None = None
        if effort_raw is not None:
            effort = str(effort_raw).lower()
            if effort in _VALID_EFFORTS:
                self._default_reasoning_effort = effort
            else:
                logger.warning(
                    "Invalid Friendli reasoning_effort '%s'; leaving it to the model "
                    "default. Valid tiers: %s.",
                    effort_raw,
                    ", ".join(sorted(_VALID_EFFORTS)),
                )
        budget_raw = config.get("reasoning_budget")
        self._default_reasoning_budget: int | None = (
            int(budget_raw) if budget_raw is not None else None
        )
        parse_raw = config.get("parse_reasoning", True)
        self._default_parse_reasoning: bool | None = (
            bool(parse_raw) if parse_raw is not None else None
        )
        include_raw = config.get("include_reasoning")
        self._default_include_reasoning: bool | None = (
            bool(include_raw) if include_raw is not None else None
        )
        thinking_raw = config.get("enable_thinking")
        self._default_enable_thinking: bool | None = (
            bool(thinking_raw) if thinking_raw is not None else None
        )

        # --- Transport ---
        self._client = None
        self._sdk_client = None
        self._http = None
        self._catalog = None
        self._backend = self._resolve_backend(config.get("backend"))

        if self._backend == "sdk" and self._default_parse_reasoning:
            logger.warning(
                "Friendli backend 'sdk' is active with parse_reasoning enabled: the "
                "official SDK's response models drop reasoning_content. Use "
                'backend = "openai" or "httpx" to receive parsed reasoning.'
            )

        try:
            if self._backend == "openai":
                self._client = AsyncOpenAI(
                    api_key=self._api_key,
                    base_url=self._base_url,
                    timeout=self._timeout,
                    default_headers=self._team_headers(),
                )
            elif self._backend == "sdk":
                self._sdk_client = AsyncFriendli(
                    token=self._api_key,
                    server_url=self._sdk_server_url(),
                    timeout_ms=int(self._timeout * 1000),
                    x_friendli_team=self._team_id,
                )
            # The "httpx" backend lazily creates its client on first use.
            logger.debug(
                "Friendli client initialized (backend=%s, endpoint_type=%s, "
                "base_url=%s, default_model=%s, team=%s).",
                self._backend,
                self._endpoint_type,
                self._base_url,
                self.default_model,
                self._team_id or "<default>",
            )
        except Exception as e:
            raise ConfigError(f"Friendli client initialization failed: {e}")

        # --- Tokenizer fallback (the native /tokenize endpoint is preferred) ---
        self._encoding = None
        if tiktoken_available:
            try:
                self._encoding = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                logger.warning("Failed to load tiktoken for Friendli: %s", e)

    # =========================================================================
    # Configuration helpers
    # =========================================================================

    @staticmethod
    def _resolve_api_key(config: dict[str, Any]) -> str | None:
        """Resolve the Friendli API key from config or the environment."""
        api_key = config.get("api_key")
        if api_key:
            return str(api_key)
        env_var = config.get("api_key_env_var")
        if env_var:
            value = os.environ.get(str(env_var))
            if value:
                return value
        for name in _API_KEY_ENV_VARS:
            value = os.environ.get(name)
            if value:
                return value
        return None

    @staticmethod
    def _resolve_team_id(config: dict[str, Any]) -> str | None:
        """Resolve the Friendli team ID from config or the environment."""
        team_id = config.get("team_id")
        if team_id:
            return str(team_id)
        env_var = config.get("team_id_env_var")
        if env_var:
            value = os.environ.get(str(env_var))
            if value:
                return value
        for name in _TEAM_ID_ENV_VARS:
            value = os.environ.get(name)
            if value:
                return value
        return None

    @staticmethod
    def _resolve_backend(requested: str | None) -> str:
        """Resolve the transport backend, honoring library availability.

        Preference order when unset/``"auto"``: ``openai`` → ``httpx`` → ``sdk``
        (see the module docstring for why the vendor SDK is last).  An
        explicitly requested backend that is unavailable falls back through the
        same chain with a warning.
        """
        available = {
            "openai": openai_available,
            "httpx": httpx_available,
            "sdk": friendli_sdk_available,
        }

        req = (requested or "auto").lower()
        if req not in ("auto", *_BACKEND_ORDER):
            logger.warning("Unknown Friendli backend '%s'; using auto-detection.", req)
            req = "auto"

        if req != "auto":
            if available.get(req):
                return req
            logger.warning(
                "Requested Friendli backend '%s' is unavailable; falling back. "
                "Install with: pip install llmcore[friendli]",
                req,
            )

        for backend in _BACKEND_ORDER:
            if available[backend]:
                if req != "auto" and backend != req:
                    logger.info("Friendli backend resolved to '%s'.", backend)
                return backend
        # Unreachable given the __init__ guard.
        raise ConfigError("No usable Friendli transport backend is installed.")

    def _team_headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        """Return the default headers, including team scoping when configured."""
        headers: dict[str, str] = {}
        if self._team_id:
            headers["X-Friendli-Team"] = self._team_id
        if extra:
            headers.update(extra)
        return headers

    def _sdk_server_url(self) -> str | None:
        """Return the ``server_url`` the official SDK should target.

        The SDK's generated operations already carry the ``/serverless/v1`` and
        ``/dedicated/v1`` path prefixes, so the hosted types need only the API
        origin.  Containers pass their own root through unchanged.
        """
        if self._endpoint_type == "container":
            return self._base_url
        default_root = _BASE_URLS[self._endpoint_type]
        if self._base_url == default_root:
            return None  # SDK default server (https://api.friendli.ai)
        # A custom hosted root: strip the operation path prefix if present.
        suffix = default_root.rsplit("/", 2)[-2:]  # e.g. ["serverless", "v1"]
        trimmed = self._base_url
        for part in reversed(suffix):
            if trimmed.endswith(f"/{part}"):
                trimmed = trimmed[: -(len(part) + 1)]
        return trimmed or None

    async def _run_sdk(self, fn: Any) -> Any:
        """Await an SDK coroutine factory (kept for symmetry/testability)."""
        return await fn()

    def _sdk_namespace(self, resource: str) -> Any:
        """Return the SDK sub-client for *resource* on the active endpoint type.

        Args:
            resource: ``"chat"``, ``"token"``, ``"completions"``, ``"audio"``,
                ``"image"``, ``"chat_render"``, or ``"embeddings"``.

        Raises:
            ProviderError: If the SDK backend is not active, or the resource is
                not available for this endpoint type.
        """
        if self._sdk_client is None:
            raise ProviderError(self.get_name(), "Friendli SDK client not initialized.")
        root = getattr(self._sdk_client, self._endpoint_type, None)
        if root is None:
            raise ProviderError(
                self.get_name(),
                f"Friendli SDK has no '{self._endpoint_type}' namespace.",
            )
        namespace = getattr(root, resource, None)
        if namespace is None:
            raise ProviderError(
                self.get_name(),
                f"Friendli SDK '{self._endpoint_type}' endpoints do not expose '{resource}'.",
            )
        return namespace

    # =========================================================================
    # BaseProvider interface
    # =========================================================================

    def get_name(self) -> str:
        """Return the configured instance name (default: ``"friendli"``)."""
        return self._provider_instance_name or "friendli"

    async def warm_up(self) -> None:
        """Prime the Model APIs catalog so context lookups are exact.

        Only ``serverless`` exposes a catalog; for ``dedicated`` and
        ``container`` this is a cheap no-op that logs the resolved endpoint.
        """
        if self._endpoint_type != "serverless":
            logger.debug(
                "Friendli provider ready (instance=%s, endpoint_type=%s, base_url=%s, model=%s).",
                self.get_name(),
                self._endpoint_type,
                self._base_url,
                self.default_model,
            )
            return
        try:
            catalog = await self._fetch_catalog()
            logger.debug("Friendli catalog warmed: %d models.", len(catalog))
        except Exception as e:
            logger.warning("Friendli catalog warm-up failed: %s", e)

    async def _fetch_catalog(self, force: bool = False) -> dict[str, dict[str, Any]]:
        """Fetch and cache the Model APIs catalog, keyed by model ID.

        Args:
            force: Re-fetch even when a cached catalog is present.

        Returns:
            Mapping of model ID to the raw catalog entry.  Empty for endpoint
            types that expose no catalog.
        """
        if self._catalog is not None and not force:
            return self._catalog
        if self._endpoint_type != "serverless":
            self._catalog = {}
            return self._catalog

        raw: list[dict[str, Any]] = []
        if self._backend == "sdk":
            resp = await self._run_sdk(lambda: self._sdk_client.serverless.model.models())
            raw = [self._normalize_obj(m) for m in (getattr(resp, "data", None) or [])]
        else:
            # Both the openai and httpx backends read the catalog over raw HTTP:
            # the Friendli listing is a superset of the OpenAI /models shape and
            # the openai SDK's typed Model objects would discard the extras.
            resp = await self._raw_get("/models")
            raw = resp.json().get("data", []) or []

        self._catalog = {m["id"]: m for m in raw if isinstance(m, dict) and m.get("id")}
        return self._catalog

    async def get_models_details(self) -> list[ModelDetails]:
        """Discover available models.

        ``serverless`` uses the rich Friendli catalog (``GET /models``), which
        reports context length, modalities, reasoning options, and per-token
        pricing.  ``dedicated`` and ``container`` have no catalog, so the
        configured default model is described from the model-card registry.
        """
        try:
            registry = get_model_card_registry()
        except Exception:
            registry = None

        if self._endpoint_type == "serverless":
            try:
                catalog = await self._fetch_catalog()
                details = [
                    self._build_model_details(mid, registry, entry)
                    for mid, entry in catalog.items()
                ]
                if details:
                    return details
            except Exception as e:
                logger.warning(
                    "Friendli model listing failed (%s); falling back to the static table.",
                    e,
                )
            return [self._build_model_details(mid, registry, None) for mid in _CONTEXT_LENGTHS]

        # Dedicated endpoints / containers serve exactly one deployment.
        return [self._build_model_details(self.default_model, registry, None)]

    def _build_model_details(
        self,
        model_id: str,
        registry: Any | None,
        entry: dict[str, Any] | None,
    ) -> ModelDetails:
        """Build :class:`ModelDetails` from a catalog entry and/or a model card."""
        provider = self.get_name()
        functionality = (entry or {}).get("functionality") or {}
        input_modalities = (entry or {}).get("input_modalities") or []
        output_modalities = (entry or {}).get("output_modalities") or []

        supports_tools = bool(functionality.get("tool_call", True))
        supports_vision = "image" in input_modalities
        supports_reasoning = bool((entry or {}).get("reasoning", False))
        max_output = (entry or {}).get("max_completion_tokens")

        if entry is None and registry:
            card = registry.get(provider, model_id)
            if card is not None and card.capabilities:
                supports_tools = card.capabilities.tool_use or card.capabilities.function_calling
                supports_vision = card.capabilities.vision
                supports_reasoning = card.capabilities.reasoning

        return ModelDetails(
            id=model_id,
            provider_name=provider,
            display_name=(entry or {}).get("name") or None,
            context_length=self.get_max_context_length(model_id),
            max_output_tokens=max_output,
            supports_streaming=True,
            supports_tools=supports_tools,
            supports_vision=supports_vision,
            supports_reasoning=supports_reasoning,
            model_type=(entry or {}).get("mode") or "chat",
            metadata={
                "endpoint_type": self._endpoint_type,
                "base_model": (entry or {}).get("base_model"),
                "description": (entry or {}).get("description"),
                "input_modalities": input_modalities,
                "output_modalities": output_modalities,
                "reasoning_options": (entry or {}).get("reasoning_options"),
                "interleaved": (entry or {}).get("interleaved"),
                "pricing": (entry or {}).get("pricing"),
                "functionality": functionality or None,
                "default_params": (entry or {}).get("default_params"),
                "deprecation_date": (entry or {}).get("deprecation_date"),
            },
        )

    def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
        """Return the inference parameters accepted by the Friendli chat API."""
        return {
            # --- OpenAI-compatible sampling ---
            "temperature": {"type": "number", "minimum": 0.0},
            "top_p": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "max_tokens": {"type": "integer", "minimum": 1},
            "n": {"type": "integer", "minimum": 1},
            "seed": {"type": ["integer", "array"]},
            "stop": {"type": "array", "items": {"type": "string"}},
            "frequency_penalty": {"type": "number", "minimum": -2.0, "maximum": 2.0},
            "presence_penalty": {"type": "number", "minimum": -2.0, "maximum": 2.0},
            "logit_bias": {"type": "object"},
            "logprobs": {"type": "boolean"},
            "top_logprobs": {"type": "integer", "minimum": 0},
            "parallel_tool_calls": {"type": "boolean"},
            "response_format": {"type": "object"},
            "stream_options": {"type": "object"},
            # --- Friendli Engine sampling ---
            "top_k": {"type": "integer", "minimum": 0},
            "min_p": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "min_tokens": {"type": "integer", "minimum": 0},
            "repetition_penalty": {"type": "number", "exclusiveMinimum": 0.0},
            "eos_token": {"type": "array", "items": {"type": "integer"}},
            "xtc_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "xtc_probability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            # --- Reasoning controls ---
            "reasoning_effort": {"type": "string", "enum": sorted(_VALID_EFFORTS)},
            "reasoning_budget": {"type": "integer"},
            "parse_reasoning": {"type": "boolean"},
            "include_reasoning": {"type": "boolean"},
            # --- Chat-template switches ---
            "chat_template_kwargs": {"type": "object"},
            "enable_thinking": {
                "type": "boolean",
                "description": "Folded into chat_template_kwargs.enable_thinking.",
            },
            "clear_thinking": {
                "type": "boolean",
                "description": "Folded into chat_template_kwargs.clear_thinking.",
            },
        }

    def get_max_context_length(self, model: str | None = None) -> int:
        """Return the maximum context length (tokens) for a Friendli model.

        Resolution order: the live catalog (when warmed), the static table, the
        model-card registry, then the configured ``fallback_context_length``.
        """
        model_name = model or self.default_model

        if self._catalog:
            entry = self._catalog.get(model_name)
            if entry and entry.get("context_length"):
                return int(entry["context_length"])

        limit = _CONTEXT_LENGTHS.get(model_name)
        if limit is not None:
            return limit

        try:
            registry = get_model_card_registry()
            card = registry.get(self.get_name(), model_name)
            if card is not None:
                return card.get_context_length()
        except Exception:
            pass

        logger.warning(
            "Unknown context length for Friendli model '%s'. Falling back to %d.",
            model_name,
            self._fallback_context_length,
        )
        return self._fallback_context_length

    # =========================================================================
    # Request construction
    # =========================================================================

    @staticmethod
    def _normalize_media_part(item: Any, kind: str) -> dict[str, Any] | None:
        """Normalize an inline media entry into a Friendli content part.

        Accepted *item* forms:
            - ``str`` — an HTTPS URL or a base64 data URI.
            - ``dict`` — a ready-made content part (passed through), a
              ``{"<kind>": {...}}`` wrapper, or ``{"url": "..."}``.

        Args:
            item: The raw inline entry.
            kind: ``"image_url"``, ``"audio_url"``, or ``"video_url"``.

        Returns:
            A content-part dict, or ``None`` if the entry is unrecognized.
        """
        if isinstance(item, str):
            return {"type": kind, kind: {"url": item}}
        if isinstance(item, dict):
            if item.get("type") in ("image_url", "audio_url", "video_url"):
                return item
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
        """Build a Friendli-format message dict from an llmcore Message.

        Handles multimodal user content (``inline_images`` / ``inline_audio`` /
        ``inline_videos`` / ``content_parts`` in ``metadata``), assistant
        ``tool_calls`` with ``reasoning_content`` preservation, and tool-result
        messages (``tool_call_id``).
        """
        role_str = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        metadata = msg.metadata or {}

        content: Any = msg.content
        if "content_parts" in metadata:
            content = metadata["content_parts"]
        else:
            inline_images = metadata.get("inline_images") or []
            inline_audio = metadata.get("inline_audio") or []
            inline_videos = metadata.get("inline_videos") or []
            if inline_images or inline_audio or inline_videos:
                parts: list[dict[str, Any]] = []
                for item, kind in (
                    *((i, "image_url") for i in inline_images),
                    *((a, "audio_url") for a in inline_audio),
                    *((v, "video_url") for v in inline_videos),
                ):
                    part = self._normalize_media_part(item, kind)
                    if part:
                        parts.append(part)
                if msg.content:
                    parts.append({"type": "text", "text": msg.content})
                content = parts

        msg_dict: dict[str, Any] = {"role": role_str, "content": content}

        if msg.role == LLMCoreRole.TOOL and msg.tool_call_id:
            msg_dict["tool_call_id"] = msg.tool_call_id

        if role_str == "assistant":
            # First-class Message.tool_calls (R-2) takes precedence over the
            # legacy metadata channel so native tool-role results pair up.
            tool_calls = getattr(msg, "tool_calls", None) or metadata.get("tool_calls")
            if tool_calls:
                msg_dict["tool_calls"] = tool_calls
                if not msg.content:
                    # Friendli's AssistantMessage allows null content when
                    # tool_calls is present.
                    msg_dict["content"] = None
            if "reasoning_content" in metadata:
                msg_dict["reasoning_content"] = metadata["reasoning_content"]

        name = metadata.get("name")
        if name:
            msg_dict["name"] = name

        return msg_dict

    def _resolve_request_params(
        self, kwargs: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Split validated kwargs into native and Friendli-specific body params.

        Configured reasoning defaults are applied for any key the caller did not
        supply, and the flat ``enable_thinking`` / ``clear_thinking`` switches
        are folded into ``chat_template_kwargs``.

        Returns:
            ``(native, extras)`` — ``native`` holds OpenAI-compatible chat
            parameters, ``extras`` holds the Friendli-specific ones.
        """
        native: dict[str, Any] = {}
        extras: dict[str, Any] = {}
        template_kwargs: dict[str, Any] = dict(kwargs.pop("chat_template_kwargs", None) or {})

        for key in _TEMPLATE_KWARG_KEYS:
            if key in kwargs:
                template_kwargs[key] = bool(kwargs.pop(key))
        if "enable_thinking" not in template_kwargs and self._default_enable_thinking is not None:
            template_kwargs["enable_thinking"] = self._default_enable_thinking

        effort = kwargs.pop("reasoning_effort", None)
        if effort is None:
            effort = self._default_reasoning_effort
        elif str(effort).lower() not in _VALID_EFFORTS:
            logger.warning(
                "Ignoring invalid Friendli reasoning_effort '%s'. Valid tiers: %s.",
                effort,
                ", ".join(sorted(_VALID_EFFORTS)),
            )
            effort = self._default_reasoning_effort
        else:
            effort = str(effort).lower()
        if effort is not None:
            extras["reasoning_effort"] = effort

        budget = kwargs.pop("reasoning_budget", self._default_reasoning_budget)
        if budget is not None:
            extras["reasoning_budget"] = int(budget)

        parse_reasoning = kwargs.pop("parse_reasoning", self._default_parse_reasoning)
        if parse_reasoning is not None:
            extras["parse_reasoning"] = bool(parse_reasoning)

        include_reasoning = kwargs.pop("include_reasoning", self._default_include_reasoning)
        if include_reasoning is not None:
            extras["include_reasoning"] = bool(include_reasoning)

        for key, value in kwargs.items():
            if key in _EXTRA_BODY_KEYS:
                extras[key] = value
            else:
                native[key] = value

        if template_kwargs:
            extras["chat_template_kwargs"] = template_kwargs

        # ``seed`` may be a list (one per generation with ``n``), which the
        # openai SDK does not type; route those through the Friendli extras.
        if isinstance(native.get("seed"), (list, tuple)):
            extras["seed"] = list(native.pop("seed"))

        return native, extras

    @staticmethod
    def _apply_mutual_exclusions(
        native: dict[str, Any], extras: dict[str, Any], has_tools: bool
    ) -> None:
        """Drop body fields Friendli rejects in combination, with a warning.

        The API rejects ``min_tokens`` and ``response_format`` when ``tools`` is
        set, and ``min_tokens`` when ``response_format`` is set.
        """
        if has_tools:
            for key, bucket in (("min_tokens", extras), ("response_format", native)):
                if key in bucket:
                    bucket.pop(key)
                    logger.warning("Friendli rejects '%s' together with 'tools'; dropping it.", key)
        elif "response_format" in native and "min_tokens" in extras:
            extras.pop("min_tokens")
            logger.warning(
                "Friendli rejects 'min_tokens' together with 'response_format'; dropping it."
            )

    # =========================================================================
    # Chat completion
    # =========================================================================

    async def chat_completion(
        self,
        context: ContextPayload,
        model: str | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Perform a chat completion against the Friendli API.

        Dispatches to the active backend (``openai`` / ``httpx`` / ``sdk``).
        Across all backends it provides:

        - Reasoning controls (``reasoning_effort``, ``reasoning_budget``,
          ``parse_reasoning``, ``include_reasoning``) with configured defaults.
        - ``chat_template_kwargs`` assembly from ``enable_thinking`` /
          ``clear_thinking``.
        - Friendli Engine sampling parameters (``top_k``, ``min_p``,
          ``repetition_penalty``, ``min_tokens``, XTC, ``eos_token``).
        - Multimodal (image / audio / video) content assembly.
        - ``reasoning_content`` in both streaming and non-streaming responses
          (``openai`` and ``httpx`` backends).

        Args:
            context: The conversation as a list of ``Message`` objects.
            model: Model ID (Model APIs) or endpoint ID (Dedicated Endpoints).
            stream: Return an async generator of raw chunk dicts.
            tools: Tools the model may call.
            tool_choice: ``"none"``, ``"auto"``, ``"required"``, or a
                ``{"type": "function", "function": {"name": ...}}`` dict.
            **kwargs: Any parameter from :meth:`get_supported_parameters`.

        Returns:
            A response dict (``stream=False``) or an async generator of chunk
            dicts (``stream=True``).

        Raises:
            ValueError: On an unsupported parameter name.
            ProviderError: On API, auth, or transport failures.
            ContextLengthError: When the prompt exceeds the model's window.
        """
        model_name = model or self.default_model

        supported = self.get_supported_parameters(model_name)
        for key in kwargs:
            if key not in supported:
                raise ValueError(f"Unsupported parameter '{key}' for Friendli provider.")

        if not (isinstance(context, list) and all(isinstance(m, Message) for m in context)):
            raise ProviderError(self.get_name(), "Context must be list[Message].")

        messages_payload = [self._build_message_payload(m) for m in context]
        if not messages_payload:
            raise ProviderError(self.get_name(), "No valid messages.")

        tools_payload = None
        if tools:
            tools_payload = [{"type": "function", "function": t.model_dump()} for t in tools]

        native, extras = self._resolve_request_params(dict(kwargs))
        self._apply_mutual_exclusions(native, extras, has_tools=bool(tools_payload))

        if tool_choice:
            native["tool_choice"] = tool_choice
        if stream:
            native.setdefault("stream_options", {"include_usage": True})

        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "RAW FRIENDLI REQUEST (backend=%s, endpoint_type=%s): %s",
                self._backend,
                self._endpoint_type,
                json.dumps(
                    {
                        "model": model_name,
                        "messages": messages_payload,
                        "stream": stream,
                        "tools": tools_payload,
                        **native,
                        **extras,
                    },
                    indent=2,
                    default=str,
                ),
            )

        try:
            if self._backend == "openai":
                return await self._chat_via_openai(
                    model_name, messages_payload, stream, tools_payload, native, extras
                )
            if self._backend == "sdk":
                return await self._chat_via_sdk(
                    model_name, messages_payload, stream, tools_payload, native, extras
                )
            return await self._chat_via_httpx(
                model_name, messages_payload, stream, tools_payload, native, extras
            )
        except (ProviderError, ContextLengthError, ValueError):
            raise
        except Exception as e:
            self._raise_error(e, model_name)
            raise  # pragma: no cover - _raise_error always raises

    async def _chat_via_openai(
        self,
        model_name: str,
        messages: list[dict[str, Any]],
        stream: bool,
        tools_payload: list[dict[str, Any]] | None,
        native: dict[str, Any],
        extras: dict[str, Any],
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Chat via ``AsyncOpenAI`` pointed at Friendli (extras via extra_body)."""
        api_kwargs: dict[str, Any] = dict(native)
        if extras:
            api_kwargs["extra_body"] = dict(extras)
        if tools_payload:
            api_kwargs["tools"] = tools_payload

        resp = await self._client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=stream,
            **api_kwargs,
        )  # type: ignore[arg-type]

        if stream:

            async def stream_wrapper() -> AsyncGenerator[dict[str, Any], None]:
                async for chunk in resp:  # type: ignore[union-attr]
                    chunk_dict = self._normalize_obj(chunk)
                    if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "RAW FRIENDLI STREAM CHUNK: %s", json.dumps(chunk_dict, default=str)
                        )
                    yield chunk_dict

            return stream_wrapper()

        response_dict = self._normalize_obj(resp)
        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "RAW FRIENDLI RESPONSE: %s", json.dumps(response_dict, indent=2, default=str)
            )
        return response_dict

    async def _chat_via_httpx(
        self,
        model_name: str,
        messages: list[dict[str, Any]],
        stream: bool,
        tools_payload: list[dict[str, Any]] | None,
        native: dict[str, Any],
        extras: dict[str, Any],
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Chat via direct httpx calls against ``POST /chat/completions``."""
        body: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "stream": stream,
            **native,
            **extras,
        }
        if tools_payload:
            body["tools"] = tools_payload

        if stream:
            return self._httpx_sse_stream(body)
        resp = await self._raw_post("/chat/completions", json_body=body)
        return resp.json()

    async def _httpx_sse_stream(self, body: dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]:
        """Stream ``/chat/completions`` over httpx, parsing SSE ``data:`` lines."""
        client = self._get_http()
        model_name = body.get("model", "")
        try:
            async with client.stream("POST", "/chat/completions", json=body) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    self._raise_status_error(resp.status_code, resp.text, model_name)
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "RAW FRIENDLI STREAM CHUNK: %s", json.dumps(chunk, default=str)
                        )
                    yield chunk
        except (ProviderError, ContextLengthError):
            raise
        except httpx.HTTPError as e:
            raise ProviderError(self.get_name(), f"Streaming error: {e}", model_name=model_name)

    async def _chat_via_sdk(
        self,
        model_name: str,
        messages: list[dict[str, Any]],
        stream: bool,
        tools_payload: list[dict[str, Any]] | None,
        native: dict[str, Any],
        extras: dict[str, Any],
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Chat via the official ``friendli`` SDK.

        The SDK types every parameter explicitly, so there is no ``extra_body``
        escape hatch: any key it does not declare is dropped with a warning
        rather than silently changing the request.
        """
        chat = self._sdk_namespace("chat")
        call_kwargs: dict[str, Any] = {**native, **extras}
        if tools_payload:
            call_kwargs["tools"] = tools_payload
        if self._endpoint_type == "container":
            call_kwargs["server_url"] = self._base_url

        method = chat.stream if stream else chat.complete
        call_kwargs = self._filter_sdk_kwargs(method, call_kwargs)

        if stream:
            event_stream = await self._run_sdk(
                lambda: chat.stream(model=model_name, messages=messages, **call_kwargs)
            )
            return self._bridge_sdk_stream(event_stream)

        resp = await self._run_sdk(
            lambda: chat.complete(model=model_name, messages=messages, **call_kwargs)
        )
        return self._normalize_obj(resp)

    @staticmethod
    def _filter_sdk_kwargs(method: Any, call_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Drop kwargs the SDK method does not declare, with a warning.

        The generated SDK types every parameter explicitly and has no
        ``extra_body`` escape hatch, so an undeclared key would raise an opaque
        ``TypeError`` deep inside the vendor package.  Dropping it here keeps
        the failure legible and the request valid; the ``openai`` and ``httpx``
        backends pass the same key through untouched.

        Args:
            method: The bound SDK operation about to be called.
            call_kwargs: The assembled request parameters.

        Returns:
            ``call_kwargs`` restricted to the parameters *method* accepts.
        """
        try:
            accepted = set(inspect.signature(method).parameters)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return call_kwargs
        kept = {k: v for k, v in call_kwargs.items() if k in accepted}
        dropped = sorted(set(call_kwargs) - set(kept))
        if dropped:
            logger.warning(
                "Friendli SDK backend does not accept %s; dropping. Use "
                'backend = "openai" or "httpx" to send them.',
                ", ".join(dropped),
            )
        return kept

    async def _bridge_sdk_stream(self, event_stream: Any) -> AsyncGenerator[dict[str, Any], None]:
        """Yield normalized chunk dicts from an SDK ``EventStreamAsync``."""
        try:
            async for chunk in event_stream:
                chunk_dict = self._normalize_obj(chunk)
                if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "RAW FRIENDLI SDK STREAM CHUNK: %s", json.dumps(chunk_dict, default=str)
                    )
                yield chunk_dict
        finally:
            close = getattr(event_stream, "close", None)
            if callable(close):
                try:
                    await close()
                except Exception as e:  # pragma: no cover - best effort
                    logger.debug("Error closing Friendli SDK stream: %s", e)

    @staticmethod
    def _normalize_obj(obj: Any) -> dict[str, Any]:
        """Normalize an SDK/OpenAI pydantic response into a plain dict."""
        if hasattr(obj, "model_dump"):
            return obj.model_dump(exclude_none=True)
        if isinstance(obj, dict):
            return obj
        return dict(obj)

    # =========================================================================
    # Error mapping
    # =========================================================================

    def _raise_status_error(self, status: int, body: str, model_name: str) -> None:
        """Map an HTTP status + body to an llmcore exception.

        Raises:
            ContextLengthError: For context/length overflow (HTTP 400/422).
            ProviderError: For every other failure.
        """
        logger.error("Friendli status error (%s): %s", status, body)
        if status in (400, 422) and _looks_like_context_overflow(body):
            raise ContextLengthError(
                model_name=model_name,
                limit=self.get_max_context_length(model_name),
                actual=0,
                message=body,
            )
        if status == 401:
            raise ProviderError(
                self.get_name(),
                f"Friendli authentication failed. Verify FRIENDLI_TOKEN / "
                f"FRIENDLIAI_API_KEY (Personal API keys start with 'flp_'). "
                f"Error: {body}",
                model_name=model_name,
                status_code=status,
            )
        if status == 403:
            raise ProviderError(
                self.get_name(),
                f"Friendli request forbidden. Check the team scope "
                f"(X-Friendli-Team = {self._team_id or '<default>'}) and that the key "
                f"may use this endpoint. Error: {body}",
                model_name=model_name,
                status_code=status,
            )
        if status == 404:
            raise ProviderError(
                self.get_name(),
                f"Friendli returned 404 on the '{self._endpoint_type}' surface "
                f"({self._base_url}) for model/endpoint '{model_name}'. Either the "
                f"model/endpoint ID is wrong (Dedicated Endpoints take the endpoint ID, "
                f"not a model name), or that route is not served on this surface - "
                f"/detokenize and /chat/render are documented but currently 404 on "
                f"Model APIs. Error: {body}",
                model_name=model_name,
                status_code=status,
            )
        if status == 429:
            raise ProviderError(
                self.get_name(),
                f"Friendli rate limit exceeded. Model APIs limits scale with your "
                f"usage tier (https://friendli.ai/docs/guides/model-apis/rate-limits). "
                f"Error: {body}",
                model_name=model_name,
                status_code=status,
            )
        raise ProviderError(
            self.get_name(),
            f"API Error ({status}): {body}",
            model_name=model_name,
            status_code=status,
        )

    def _raise_error(self, e: Exception, model_name: str) -> None:
        """Map a backend exception to an llmcore exception.

        Raises:
            ProviderError or ContextLengthError: Always.
        """
        status = getattr(e, "status_code", None)
        if status is None:
            response = getattr(e, "response", None)
            status = getattr(response, "status_code", None)
        body = getattr(e, "body", None) or str(e)
        if status is not None:
            self._raise_status_error(int(status), str(body), model_name)

        if isinstance(e, OpenAIAPITimeoutError):
            raise ProviderError(
                self.get_name(), f"Timeout: {e}", model_name=model_name, retryable=True
            )
        if isinstance(e, OpenAIAPIConnectionError):
            raise ProviderError(
                self.get_name(), f"Connection error: {e}", model_name=model_name, retryable=True
            )
        if httpx_available and isinstance(e, httpx.TimeoutException):
            raise ProviderError(
                self.get_name(), f"Timeout: {e}", model_name=model_name, retryable=True
            )
        if httpx_available and isinstance(e, httpx.HTTPError):
            raise ProviderError(
                self.get_name(), f"HTTP error: {e}", model_name=model_name, retryable=True
            )
        logger.error("Unexpected Friendli error: %s", e, exc_info=True)
        raise ProviderError(
            self.get_name(), f"Error: {e}", model_name=model_name, original_exception=e
        )

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
            logger.warning("Failed to extract Friendli content: %s", e)
            return ""

    def extract_delta_content(self, chunk: dict[str, Any]) -> str:
        """Extract the text delta from a streaming chunk."""
        try:
            choices = chunk.get("choices", [])
            if not choices:
                return ""
            return choices[0].get("delta", {}).get("content") or ""
        except (KeyError, IndexError, TypeError):
            return ""

    def extract_reasoning_content(self, response: dict[str, Any]) -> str | None:
        """Extract parsed reasoning from a non-streaming response.

        Friendli returns the chain of thought as ``reasoning_content`` (and the
        compatibility alias ``reasoning``) when ``parse_reasoning`` is on and
        the model supports it.
        """
        try:
            choices = response.get("choices", [])
            if not choices:
                return None
            message = choices[0].get("message", {})
            return message.get("reasoning_content") or message.get("reasoning")
        except (KeyError, IndexError, TypeError):
            return None

    def extract_delta_reasoning_content(self, chunk: dict[str, Any]) -> str | None:
        """Extract the reasoning delta from a streaming chunk."""
        try:
            choices = chunk.get("choices", [])
            if not choices:
                return None
            delta = choices[0].get("delta", {})
            return delta.get("reasoning_content") or delta.get("reasoning")
        except (KeyError, IndexError, TypeError):
            return None

    def extract_tool_calls(self, response: dict[str, Any]) -> list[ToolCall]:
        """Extract tool calls from a Friendli response."""
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
            logger.warning("Failed to extract Friendli tool calls: %s", e)
        return out

    def extract_usage_details(self, response: dict[str, Any]) -> dict[str, Any]:
        """Extract usage, including Friendli's cached-prompt accounting."""
        usage = response.get("usage") or {}
        if not usage:
            return {}
        result: dict[str, Any] = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
        details = usage.get("prompt_tokens_details") or {}
        if details.get("cached_tokens") is not None:
            result["cached_tokens"] = details["cached_tokens"]
        return result

    def extract_finish_reason(self, response: dict[str, Any]) -> str | None:
        """Extract the finish reason (``stop`` / ``length`` / ``tool_calls``)."""
        try:
            choices = response.get("choices", [])
            if not choices:
                return None
            return choices[0].get("finish_reason")
        except (KeyError, IndexError, TypeError):
            return None

    # =========================================================================
    # Token counting (native /tokenize, with local fallback)
    # =========================================================================

    async def tokenize(self, text: str, model: str | None = None) -> list[int]:
        """Tokenize *text* with the model's own tokenizer (``POST /tokenize``).

        Args:
            text: The prompt text to tokenize.
            model: Model/endpoint ID; defaults to the configured model.

        Returns:
            The list of token IDs.

        Raises:
            ProviderError: On API or transport failures.
        """
        model_name = model or self.default_model
        body = {"model": model_name, "prompt": text}
        try:
            if self._backend == "sdk":
                resp = await self._run_sdk(
                    lambda: self._sdk_namespace("token").tokenize(
                        model=model_name,
                        prompt=text,
                        **self._sdk_server_kwargs(),
                    )
                )
                return list(self._normalize_obj(resp).get("tokens", []))
            resp = await self._raw_post("/tokenize", json_body=body)
            return list(resp.json().get("tokens", []))
        except (ProviderError, ContextLengthError):
            raise
        except Exception as e:
            self._raise_error(e, model_name)
            raise  # pragma: no cover

    async def detokenize(self, tokens: list[int], model: str | None = None) -> str:
        """Convert token IDs back into text (``POST /detokenize``).

        Args:
            tokens: The token IDs to decode.
            model: Model/endpoint ID; defaults to the configured model.

        Returns:
            The decoded text.
        """
        model_name = model or self.default_model
        body = {"model": model_name, "tokens": list(tokens)}
        try:
            if self._backend == "sdk":
                resp = await self._run_sdk(
                    lambda: self._sdk_namespace("token").detokenize(
                        model=model_name,
                        tokens=list(tokens),
                        **self._sdk_server_kwargs(),
                    )
                )
                return str(self._normalize_obj(resp).get("text", ""))
            resp = await self._raw_post("/detokenize", json_body=body)
            return str(resp.json().get("text", ""))
        except (ProviderError, ContextLengthError):
            raise
        except Exception as e:
            self._raise_error(e, model_name)
            raise  # pragma: no cover

    async def render_chat(
        self,
        messages: list[Message],
        model: str | None = None,
        tools: list[Tool] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Render messages into the exact prompt text sent to the model.

        Useful for debugging chat templates and for computing an exact prompt
        token count (render, then :meth:`tokenize`).

        Args:
            messages: The conversation to render.
            model: Model/endpoint ID; defaults to the configured model.
            tools: Tools to include in the rendered template.
            chat_template_kwargs: Template switches (e.g. ``enable_thinking``).

        Returns:
            The rendered prompt text.
        """
        model_name = model or self.default_model
        payload = [self._build_message_payload(m) for m in messages]
        body: dict[str, Any] = {"model": model_name, "messages": payload}
        if tools:
            body["tools"] = [{"type": "function", "function": t.model_dump()} for t in tools]
        if chat_template_kwargs:
            body["chat_template_kwargs"] = chat_template_kwargs
        try:
            if self._backend == "sdk":
                resp = await self._run_sdk(
                    lambda: self._sdk_namespace("chat_render").render(
                        **body,
                        **self._sdk_server_kwargs(),
                    )
                )
                return str(self._normalize_obj(resp).get("text", ""))
            resp = await self._raw_post("/chat/render", json_body=body)
            return str(resp.json().get("text", ""))
        except (ProviderError, ContextLengthError):
            raise
        except Exception as e:
            self._raise_error(e, model_name)
            raise  # pragma: no cover

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """Count tokens for *text* using the model's own tokenizer.

        Counting is local (tiktoken ``cl100k_base``, then a character-ratio
        estimate) unless ``native_token_count = true`` is configured, in which
        case the model's own tokenizer is used via ``POST /tokenize``.  Each
        native count is an API request against the Model APIs rate-limit
        budget, so it is opt-in; a failed native call falls back locally.
        """
        if not text:
            return 0
        if self._native_token_count:
            try:
                return len(await self.tokenize(text, model))
            except Exception as e:
                logger.debug("Friendli /tokenize failed (%s); estimating locally.", e)
        return self._count_tokens_locally(text)

    def _count_tokens_locally(self, text: str) -> int:
        """Count tokens with tiktoken, or a character-ratio estimate."""
        if not text:
            return 0
        if self._encoding is None:
            return _EstimateCounter().count(text)
        try:
            return len(self._encoding.encode(text))
        except Exception:
            return _EstimateCounter().count(text)

    async def count_message_tokens(self, messages: list[Message], model: str | None = None) -> int:
        """Estimate the total prompt tokens for *messages*.

        The role-tagged conversation is counted as one block plus a small
        per-message overhead for the chat template's delimiters.  With
        ``native_token_count = true`` the block is tokenized by the model's own
        tokenizer in a single ``/tokenize`` call; otherwise it is counted
        locally.  For an exact, template-accurate count use ``render_chat()``
        followed by ``tokenize()``.
        """
        if not messages:
            return 0
        joined = "\n".join(
            f"{m.role.value if hasattr(m.role, 'value') else m.role}: {m.content or ''}"
            for m in messages
        )
        overhead = 4 * len(messages) + 3
        if self._native_token_count:
            try:
                return len(await self.tokenize(joined, model)) + overhead
            except Exception as e:
                logger.debug("Friendli /tokenize failed for message counting (%s); estimating.", e)
        return self._count_tokens_locally(joined) + overhead

    # =========================================================================
    # Auxiliary inference endpoints
    # =========================================================================

    def _sdk_server_kwargs(self) -> dict[str, Any]:
        """Per-call SDK kwargs that pin a self-hosted container's URL."""
        if self._endpoint_type == "container":
            return {"server_url": self._base_url}
        return {}

    async def text_completion(
        self,
        prompt: str,
        *,
        model: str | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Generate a raw text completion (``POST /completions``).

        This is the prompt-based (non-chat) surface; the chat template is *not*
        applied, so the prompt is sent to the model verbatim.  It always goes
        over the raw HTTP client, independent of the configured ``backend``.

        Args:
            prompt: The raw prompt text.
            model: Model/endpoint ID; defaults to the configured model.
            stream: Return an async generator of raw chunk dicts.
            **kwargs: Additional generation parameters (``max_tokens``,
                ``temperature``, ``top_k``, ``min_tokens``, ``stop``, …).

        Returns:
            A response dict, or an async generator of chunk dicts when
            ``stream=True``.
        """
        model_name = model or self.default_model
        body: dict[str, Any] = {"model": model_name, "prompt": prompt, "stream": stream, **kwargs}
        try:
            if stream:
                return self._httpx_sse_stream_path("/completions", body)
            resp = await self._raw_post("/completions", json_body=body)
            return resp.json()
        except (ProviderError, ContextLengthError):
            raise
        except Exception as e:
            self._raise_error(e, model_name)
            raise  # pragma: no cover

    async def create_embeddings(
        self,
        input_texts: str | list[str],
        *,
        model: str | None = None,
        encoding_format: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create text embeddings (``POST /embeddings``).

        Embeddings are served by Dedicated Endpoints and Friendli Container
        only — the hosted Model APIs catalog has no embedding surface.

        Args:
            input_texts: A string or list of strings to embed.
            model: Endpoint ID; defaults to the configured model.
            encoding_format: ``"float"`` (default) or ``"base64"``.
            **kwargs: Additional body parameters.

        Returns:
            The raw API response dict with ``data``, ``model``, and ``usage``.

        Raises:
            ProviderError: On the ``serverless`` endpoint type, or on API errors.
        """
        if self._endpoint_type == "serverless":
            raise ProviderError(
                self.get_name(),
                "Friendli Model APIs do not expose an embeddings endpoint. Deploy an "
                "embedding model on a Dedicated Endpoint (or Container) and set "
                "providers.friendli.endpoint_type accordingly.",
            )
        model_name = model or self.default_model
        body: dict[str, Any] = {"model": model_name, "input": input_texts, **kwargs}
        if encoding_format is not None:
            body["encoding_format"] = encoding_format
        try:
            resp = await self._raw_post("/embeddings", json_body=body)
            return resp.json()
        except (ProviderError, ContextLengthError):
            raise
        except Exception as e:
            self._raise_error(e, model_name)
            raise  # pragma: no cover

    async def generate_image(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
        size: str | None = None,
        quality: str | None = None,
        response_format: str = "url",
        style: str | None = None,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        """Generate images (``POST /images/generations``).

        Image generation is served by Dedicated Endpoints and Friendli Container
        only.  ``n``, ``size``, ``quality``, and ``style`` are part of the
        llmcore signature but have no Friendli equivalent and are ignored;
        use ``num_inference_steps``, ``guidance_scale``, ``seed``, and
        ``control_images`` instead.

        Args:
            prompt: A text description of the desired image.
            model: Endpoint ID; defaults to the configured model.
            n: Ignored (Friendli returns a single image per request).
            size: Ignored.
            quality: Ignored.
            response_format: ``"url"`` (default), ``"raw"``, ``"png"``,
                ``"jpeg"``, or ``"jpg"``.
            style: Ignored.
            **kwargs: Friendli parameters (``num_inference_steps``,
                ``guidance_scale``, ``seed``, ``control_images``,
                ``controlnet_weights``).

        Returns:
            An :class:`ImageGenerationResult`.

        Raises:
            ProviderError: On the ``serverless`` endpoint type, or on API errors.
        """
        if self._endpoint_type == "serverless":
            raise ProviderError(
                self.get_name(),
                "Friendli Model APIs do not expose an image-generation endpoint. "
                "Deploy an image model on a Dedicated Endpoint (or Container) and set "
                "providers.friendli.endpoint_type accordingly.",
            )
        for ignored, value in (
            ("n", n if n != 1 else None),
            ("size", size),
            ("quality", quality),
            ("style", style),
        ):
            if value is not None:
                logger.debug("Friendli image generation ignores '%s'.", ignored)

        model_name = model or self.default_model
        body: dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "response_format": response_format,
            **kwargs,
        }
        try:
            resp = await self._raw_post("/images/generations", json_body=body)
            payload = resp.json()
        except (ProviderError, ContextLengthError):
            raise
        except Exception as e:
            self._raise_error(e, model_name)
            raise  # pragma: no cover

        images: list[GeneratedImage] = []
        for item in payload.get("data", []) or []:
            fmt = item.get("response_format") or response_format
            images.append(
                GeneratedImage(
                    url=item.get("url"),
                    data=item.get("b64_json") or item.get("image"),
                    format="png" if fmt in ("url", "raw") else str(fmt),
                )
            )
        return ImageGenerationResult(
            images=images,
            model=model_name,
            metadata={"raw": payload, "endpoint_type": self._endpoint_type},
        )

    async def transcribe_audio(
        self,
        audio_data: bytes | str,
        *,
        model: str | None = None,
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "json",
        temperature: float | None = None,
        timestamp_granularities: list[str] | None = None,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """Transcribe audio to text (``POST /audio/transcriptions``).

        Args:
            audio_data: Raw audio bytes or a path to an audio file.
            model: Transcription model (e.g. ``openai/whisper-large-v3``) or
                endpoint ID; defaults to the configured model.
            language: ISO-639-1 hint (e.g. ``"en"``) to improve accuracy.
            prompt: Unsupported by Friendli; ignored.
            response_format: Unsupported by Friendli; ignored (JSON is always
                returned).
            temperature: Sampling temperature between 0 and 1.
            timestamp_granularities: Unsupported by Friendli; ignored.
            **kwargs: Additional form fields (e.g. ``chunking_strategy``).

        Returns:
            A :class:`TranscriptionResult`.
        """
        for name, value in (
            ("prompt", prompt),
            ("timestamp_granularities", timestamp_granularities),
        ):
            if value is not None:
                logger.debug("Friendli audio transcription ignores '%s'.", name)
        if response_format not in ("json", "verbose_json"):
            logger.debug(
                "Friendli audio transcription always returns JSON; ignoring response_format='%s'.",
                response_format,
            )

        model_name = model or self.default_model
        if isinstance(audio_data, str):
            with open(audio_data, "rb") as fh:
                payload_bytes = fh.read()
            filename = os.path.basename(audio_data)
        else:
            payload_bytes = audio_data
            filename = "audio.wav"

        data: dict[str, Any] = {"model": model_name}
        if language:
            data["language"] = language
        if temperature is not None:
            data["temperature"] = str(temperature)
        for key, value in kwargs.items():
            data[key] = value if isinstance(value, str) else json.dumps(value)

        try:
            resp = await self._raw_post(
                "/audio/transcriptions",
                data=data,
                files={"file": (filename, payload_bytes)},
            )
            payload = resp.json()
        except (ProviderError, ContextLengthError):
            raise
        except Exception as e:
            self._raise_error(e, model_name)
            raise  # pragma: no cover

        usage = payload.get("usage") or {}
        duration_ms = usage.get("input_audio_length_ms")
        return TranscriptionResult(
            text=payload.get("text", ""),
            language=language,
            duration_seconds=(duration_ms / 1000.0) if duration_ms else None,
            model=model_name,
            metadata={"usage": usage, "raw": payload},
        )

    # =========================================================================
    # Friendli Suite (team billing / usage)
    # =========================================================================

    async def get_team_cost(
        self,
        start_time: str,
        end_time: str,
        *,
        bucket_width: str | None = None,
        limit: int | None = None,
        page: str | None = None,
        group_by: str | None = None,
    ) -> dict[str, Any]:
        """Read team cost buckets from the Friendli Suite API.

        Args:
            start_time: RFC 3339 UTC timestamp with a zeroed time portion
                (e.g. ``"2026-09-01T00:00:00Z"``); no earlier than one year ago.
            end_time: RFC 3339 UTC timestamp with a zeroed time portion.
            bucket_width: Bucket size; only ``"1d"`` is currently supported.
            limit: Number of buckets to return (1-35, default 7).
            page: Pagination cursor from a previous ``next_page``.
            group_by: Currently only ``"line_item"``.

        Returns:
            The raw API response dict.

        Note:
            Friendli asks callers to wait at least five minutes between repeated
            calls; usage takes a short while to appear in cost.
        """
        params: dict[str, Any] = {"start_time": start_time, "end_time": end_time}
        for key, value in (
            ("bucket_width", bucket_width),
            ("limit", limit),
            ("page", page),
            ("group_by", group_by),
        ):
            if value is not None:
                params[key] = value
        return await self._suite_get("/team/cost", params)

    async def get_team_usage(
        self,
        start_time: str,
        end_time: str,
        *,
        bucket_width: str | None = None,
        limit: int | None = None,
        page: str | None = None,
        group_by: str | None = None,
    ) -> dict[str, Any]:
        """Read team usage buckets from the Friendli Suite API.

        Args:
            start_time: RFC 3339 UTC timestamp with a zeroed time portion.
            end_time: RFC 3339 UTC timestamp with a zeroed time portion.
            bucket_width: Bucket size (e.g. ``"1d"``).
            limit: Number of buckets to return.
            page: Pagination cursor from a previous ``next_page``.
            group_by: Grouping dimension supported by the API.

        Returns:
            The raw API response dict.
        """
        params: dict[str, Any] = {"start_time": start_time, "end_time": end_time}
        for key, value in (
            ("bucket_width", bucket_width),
            ("limit", limit),
            ("page", page),
            ("group_by", group_by),
        ):
            if value is not None:
                params[key] = value
        return await self._suite_get("/team/usage", params)

    async def _suite_get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        """GET a Friendli Suite endpoint (separate root from inference)."""
        if not httpx_available:
            raise ProviderError(
                self.get_name(), "The 'httpx' package is required for the Friendli Suite API."
            )
        url = f"{self._suite_base_url}{path}"
        headers = self._team_headers({"Authorization": f"Bearer {self._api_key}"})
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, params=params, headers=headers)
                if resp.status_code >= 400:
                    self._raise_status_error(resp.status_code, resp.text, "")
                return resp.json()
        except (ProviderError, ContextLengthError):
            raise
        except httpx.HTTPError as e:
            raise ProviderError(self.get_name(), f"HTTP error: {e}", retryable=True)

    # =========================================================================
    # HTTP plumbing (httpx backend + endpoints with no SDK/OpenAI surface)
    # =========================================================================

    def _get_http(self) -> Any:
        """Return (lazily creating) the raw httpx client for Friendli REST calls."""
        if not httpx_available:
            raise ProviderError(
                self.get_name(),
                "The 'httpx' package is required for the Friendli REST endpoints. "
                "Install with: pip install llmcore[friendli]",
            )
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self._base_url,
                headers=self._team_headers({"Authorization": f"Bearer {self._api_key}"}),
                timeout=self._timeout,
            )
        return self._http

    async def _raw_post(
        self,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: Any | None = None,
    ) -> Any:
        """POST to a Friendli endpoint and return the httpx response."""
        client = self._get_http()
        try:
            resp = await client.post(path, json=json_body, data=data, files=files)
        except httpx.HTTPError as e:
            raise ProviderError(self.get_name(), f"HTTP error: {e}", retryable=True)
        if resp.status_code >= 400:
            model_name = (json_body or {}).get("model") or (data or {}).get("model") or ""
            self._raise_status_error(resp.status_code, resp.text, str(model_name))
        return resp

    async def _raw_get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET a Friendli endpoint and return the httpx response."""
        client = self._get_http()
        try:
            resp = await client.get(path, params=params)
        except httpx.HTTPError as e:
            raise ProviderError(self.get_name(), f"HTTP error: {e}", retryable=True)
        if resp.status_code >= 400:
            self._raise_status_error(resp.status_code, resp.text, "")
        return resp

    async def _httpx_sse_stream_path(
        self, path: str, body: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream any SSE endpoint over httpx, parsing ``data:`` lines."""
        client = self._get_http()
        model_name = body.get("model", "")
        try:
            async with client.stream("POST", path, json=body) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    self._raise_status_error(resp.status_code, resp.text, model_name)
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        yield json.loads(payload)
                    except json.JSONDecodeError:
                        continue
        except (ProviderError, ContextLengthError):
            raise
        except httpx.HTTPError as e:
            raise ProviderError(self.get_name(), f"Streaming error: {e}", model_name=model_name)

    # =========================================================================
    # Resource cleanup
    # =========================================================================

    async def close(self) -> None:
        """Close the OpenAI / Friendli SDK / raw HTTP clients (best effort)."""
        if self._client is not None:
            try:
                await self._client.close()
            except Exception as e:
                logger.error("Error closing Friendli OpenAI client: %s", e)
            self._client = None
        if self._sdk_client is not None:
            try:
                closer = getattr(self._sdk_client, "close", None)
                if callable(closer):
                    result = closer()
                    if asyncio.iscoroutine(result):
                        await result
            except Exception as e:
                logger.error("Error closing Friendli SDK client: %s", e)
            self._sdk_client = None
        if self._http is not None:
            try:
                await self._http.aclose()
            except Exception as e:
                logger.error("Error closing Friendli HTTP client: %s", e)
            self._http = None
        logger.info("FriendliProvider closed.")
