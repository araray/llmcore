# src/llmcore/providers/openai_provider.py
"""
OpenAI API provider implementation for the LLMCore library.

Handles interactions with the OpenAI API (GPT models).
Accepts context as List[Message] and supports standardized tool-calling,
multimodal content (vision, audio input), structured output, and
extended token usage extraction.

Tested against openai Python SDK v3.51.0 (live, 2026-11-16).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from types import MappingProxyType

try:
    import httpx

    httpx_available = True
except ImportError:  # pragma: no cover
    httpx_available = False
    httpx = None  # type: ignore
from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

try:
    import openai
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
        AuthenticationError as OpenAIAuthError,
    )
    from openai._exceptions import (
        BadRequestError as OpenAIBadRequestError,
    )
    from openai._exceptions import (
        OpenAIError,
    )
    from openai._exceptions import (
        RateLimitError as OpenAIRateLimitError,
    )
    from openai.types.chat import ChatCompletionChunk

    openai_available = True
except ImportError:
    openai_available = False
    AsyncOpenAI = None  # type: ignore
    OpenAIError = Exception  # type: ignore
    OpenAIAPIError = Exception  # type: ignore
    OpenAIAPIStatusError = Exception  # type: ignore
    OpenAIAPIConnectionError = Exception  # type: ignore
    OpenAIAPITimeoutError = Exception  # type: ignore
    OpenAIAuthError = Exception  # type: ignore
    OpenAIBadRequestError = Exception  # type: ignore
    OpenAIRateLimitError = Exception  # type: ignore
    ChatCompletionChunk = None  # type: ignore

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
from ..tokens import count_tokens as _count_tokens
from ..tokens import get_counter as _get_token_counter
from . import openai_responses as rsp
from .base import BaseProvider, ContextPayload

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import AsyncIterator

    from ..media.models import MediaCapability, MediaExecution, MediaRef, MediaResult

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_TOKEN_LIMITS = {
    "gpt-4o": 128000,
    "gpt-4o-2024-05-13": 128000,
    "gpt-4o-2024-08-06": 128000,
    "gpt-4o-2024-11-20": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4o-mini-2024-07-18": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4-turbo-preview": 128000,
    "gpt-4-turbo-2024-04-09": 128000,
    "gpt-4-0125-preview": 128000,
    "gpt-4-1106-preview": 128000,
    "gpt-4-vision-preview": 128000,
    "gpt-4": 8000,
    "gpt-4-0613": 8000,
    "gpt-4-32k": 32000,
    "gpt-4-32k-0613": 32000,
    "gpt-3.5-turbo-0125": 16000,
    "gpt-3.5-turbo": 16000,
    "gpt-3.5-turbo-1106": 16000,
    "gpt-3.5-turbo-0613": 4000,
    "gpt-3.5-turbo-16k": 16000,
    "gpt-3.5-turbo-16k-0613": 16000,
}

#: Maximum INPUT tokens per model family, measured live on 2026-11-16 from
#: OpenAI's own overflow errors ("Input tokens exceed the configured limit of
#: N tokens" / "maximum context length is N tokens"). This is the number a
#: prompt must fit in, which is below the advertised total window: the GPT-5
#: line's "400K" is 272K in + 128K out, and the "1.05M" models take 922K in.
#: First match wins, so specific prefixes come before general ones.
_OPENAI_PREFIX_CONTEXT_HEURISTICS: list[tuple[str, int]] = [
    ("gpt-6", 922000),
    ("gpt-5.6", 922000),
    ("gpt-5.5", 922000),
    ("gpt-5.4-mini", 272000),
    ("gpt-5.4-nano", 272000),
    ("gpt-5.4", 922000),
    ("gpt-5.3", 272000),
    ("gpt-5.2", 272000),
    ("gpt-5.1", 272000),
    ("gpt-5", 272000),
    ("o4-mini", 200000),
    ("o3", 200000),
    ("o1-mini", 128000),
    ("o1", 200000),
    ("gpt-4.1", 1047576),
    ("codex-mini", 200000),
    ("gpt-4o-mini", 128000),
    ("gpt-4o", 128000),
    ("chatgpt-4o", 128000),
]

_warned_unknown_models: set[str] = set()
DEFAULT_MODEL = "gpt-6-luna"

#: gpt-image-1 and the DALL-E models are gone (live 2026-11-16: "The model
#: 'gpt-image-1' does not exist").
DEFAULT_IMAGE_MODEL = "gpt-image-2"

def _is_reasoning_model(model: str) -> bool:
    """Whether ``model`` is an OpenAI reasoning model.

    The o-series and every GPT-5+ model reason (they reject ``max_tokens`` and,
    most of them, custom sampling). Chat-tuned and search variants do not.
    """
    if re.search(r"(chat-latest|search-api|search-preview|audio|realtime)", model):
        return False
    return bool(re.match(r"(o\d|gpt-[5-9])", model))


def _needs_developer_role(model: str) -> bool:
    """Check if model requires developer role instead of system."""
    return _is_reasoning_model(model)


# ---------------------------------------------------------------------------
# Responses routing and per-model parameter rules (native OpenAI only)
# ---------------------------------------------------------------------------

#: ``responses_api`` values. ``auto`` routes OpenAI's reasoning families to
#: ``/responses`` and keeps the rest on Chat Completions. ``always`` and
#: ``never`` do what they say. Native OpenAI defaults to ``auto``; an
#: OpenAI-compatible endpoint (deepinfra, vllm, groq, …) defaults to ``never``,
#: since most of them have no ``/responses``.
RESPONSES_POLICIES: tuple[str, ...] = ("auto", "always", "never")

#: Models ``auto`` sends to ``/responses``: the o-series and the GPT-5+
#: reasoning families, plus pro/codex/computer-use. Several of those are
#: Responses-only, and every reasoning model keeps its chain across tool turns
#: only there.
_RESPONSES_PREFERRED = re.compile(r"^(o\d|gpt-[5-9]|codex|computer-use)", re.IGNORECASE)

#: Model names that stay on Chat Completions under ``auto``. Search-API and
#: audio models are Chat-only (live 2026-11-16: ``gpt-5-search-api`` answers
#: 400 on ``/responses``).
_CHAT_PREFERRED = re.compile(r"(search-api|search-preview|audio|realtime|chat-latest)", re.IGNORECASE)

#: Chat Completions parameters with no Responses equivalent. Asking for one
#: keeps the call on Chat Completions, whatever the policy.
CHAT_ONLY_PARAMS = frozenset(
    {
        "n", "logprobs", "top_logprobs", "seed", "stop", "logit_bias",
        "frequency_penalty", "presence_penalty", "modalities", "audio",
        "prediction", "web_search_options", "stream_options", "functions",
        "function_call",
    }
)

#: Responses parameters with no Chat Completions equivalent. Asking for one
#: sends the call to ``/responses``.
RESPONSES_ONLY_PARAMS = frozenset(
    {
        "reasoning", "reasoning_summary", "text", "include", "previous_response_id",
        "max_output_tokens", "truncation", "instructions", "extra_tools",
        "max_tool_calls", "safety_identifier", "background", "conversation",
    }
)

#: Effort levels in increasing order. A rejected level moves to the nearest one
#: the model accepts, preferring the higher on a tie.
_EFFORT_ORDER: tuple[str, ...] = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

#: Wire field named in an OpenAI error -> the call keywords that produce it.
_WIRE_KWARGS: dict[str, tuple[str, ...]] = {
    "reasoning_effort": ("reasoning_effort", "effort"),
    "reasoning.effort": ("reasoning_effort", "effort"),
    "max_tokens": ("max_tokens",),
    "max_output_tokens": ("max_tokens", "max_completion_tokens", "max_output_tokens"),
    "text.verbosity": ("verbosity",),
    "logprobs": ("logprobs", "top_logprobs"),
}

# The exact phrasings OpenAI uses, observed live 2026-11-16 across gpt-4.1,
# gpt-5, gpt-5.1, gpt-5.4, gpt-6 and o3.
_RX_USE_INSTEAD = re.compile(
    r"Unsupported parameter: '([\w.]+)' is not supported with this model\. Use '([\w.]+)' instead"
)
_RX_UNSUPPORTED_PARAM = re.compile(r"Unsupported parameter: '([\w.]+)' is not supported with this model")
_RX_DEFAULT_ONLY = re.compile(r"Unsupported value: '([\w.]+)' does not support .*?Only the default")
_RX_SUPPORTED_VALUES = re.compile(
    r"Unsupported value: '([\w.]+)' does not support '([^']+)' with this model\. Supported values are: ([^\n\"}]+)"
)
#: The Responses route words an effort rejection without naming the field:
#: "Unsupported value: 'none' is not supported with the 'gpt-5-nano' model.
#: Supported values are: 'minimal', 'low', 'medium', and 'high'."
_RX_SUPPORTED_VALUES_NAMELESS = re.compile(
    r"Unsupported value: '([^']+)' is not supported with the '[^']+' model\. Supported values are: ([^\n\"}]+)"
)
_RX_UNRECOGNIZED = re.compile(r"Unrecognized request argument supplied: ([\w.]+)")
_RX_NO_LOGPROBS = re.compile(r"not allowed to request logprobs")
_RX_RESPONSES_ONLY = re.compile(
    r"(only supported in v1/responses|Use the v1/responses endpoint|not a chat model)", re.IGNORECASE
)
_RX_NOT_ON_RESPONSES = re.compile(r"not supported with the Responses API", re.IGNORECASE)


#: Parameters only the Responses route takes, advertised where it is enabled.
_RESPONSES_EXTRA_PARAMS: dict[str, Any] = {
    "reasoning": {"type": "object", "description": "Responses {effort, summary}."},
    "reasoning_summary": {"type": "string", "enum": ["auto", "concise", "detailed"]},
    "text": {"type": "object", "description": "Responses {format, verbosity}."},
    "include": {"type": "array", "items": {"type": "string"}},
    "previous_response_id": {"type": "string"},
    "max_output_tokens": {"type": "integer", "minimum": 1},
    "truncation": {"type": "string", "enum": ["auto", "disabled"]},
    "instructions": {"type": "string"},
    "extra_tools": {
        "type": "array",
        "description": "Raw Responses tools (web_search, file_search, code_interpreter, mcp, …).",
    },
    "max_tool_calls": {"type": "integer", "minimum": 1},
    "safety_identifier": {"type": "string"},
}


@dataclass(frozen=True)
class _VendorProfile:
    """How one OpenAI-compatible vendor differs from OpenAI itself.

    xAI, Groq and Together are instances of this class, not subclasses, so
    without a profile they inherited OpenAI's media endpoints, default media
    models and parameter allowlist, none of which they share.
    """

    hosts: tuple[str, ...]
    #: Vendor chat parameters accepted on top of OpenAI's (sent in extra_body
    #: on the SDK transport, merged into the body on the direct one).
    extra_params: frozenset[str] = frozenset()
    #: Media capabilities this vendor serves in OpenAI's shape.
    media: frozenset[str] = frozenset()
    default_stt_model: str | None = None
    default_image_model: str | None = None
    #: Native search: the tools to add, and the route that carries them.
    search_tools: tuple[dict[str, Any], ...] = ()
    search_route: str | None = None  # "chat" | "responses"
    #: When set, native search exists only for models matching this pattern.
    search_models: str | None = None
    #: Body fields that turn search on (merged; ``plugins`` lists extend).
    search_body: Mapping[str, Any] = field(default_factory=dict)
    #: The assistant-message field that carries replayed reasoning text.
    replay_field: str = "reasoning_content"
    #: The vendor reports some failures inside an HTTP 200 (OpenRouter).
    in_band_errors: bool = False


#: Vendor profiles, from each vendor's API docs (2026-11-16). No keys were
#: available for these three, so the profiles follow documentation only.
_VENDOR_PROFILES: dict[str, _VendorProfile] = {
    # xAI: Live Search (search_parameters) answers HTTP 410 since 2026-01; search
    # is now a server-side tool on /responses only. Chat Completions still
    # serves plain chat. Images use the OpenAI images shape; xAI's /tts and /stt
    # do not, so they are not declared.
    "xai": _VendorProfile(
        hosts=("api.x.ai",),
        media=frozenset({"image_generate"}),
        default_image_model="grok-imagine-image",
        search_tools=({"type": "web_search"}, {"type": "x_search"}),
        search_route="responses",
    ),
    # Groq: reasoning_format / include_reasoning control reasoning output;
    # browser_search is a built-in tool on the gpt-oss models only. Whisper ASR
    # is OpenAI-compatible.
    "groq": _VendorProfile(
        hosts=("api.groq.com",),
        extra_params=frozenset(
            {"reasoning_format", "include_reasoning", "search_settings", "service_tier",
             "documents", "citation_options"}
        ),
        media=frozenset({"asr"}),
        default_stt_model="whisper-large-v3-turbo",
        search_tools=({"type": "browser_search"},),
        search_route="chat",
        search_models=r"gpt-oss",
    ),
    # OpenRouter: routing, plugins and caching extensions; reasoning_details
    # must be replayed in order (Gemini 3 / Claude send encrypted items only);
    # failures can arrive inside a 200 (stream chunk or body with "error").
    # Search is the "web" plugin. Verified live 2026-11-16.
    "openrouter": _VendorProfile(
        hosts=("openrouter.ai",),
        extra_params=frozenset(
            {"provider", "models", "reasoning", "plugins", "transforms", "session_id",
             "cache_control", "top_k", "min_p", "top_a", "repetition_penalty",
             "prompt_cache_options", "trace", "verbosity", "stop_server_tools_when",
             "debug", "image_config", "modalities", "deferred_tools", "usage"}
        ),
        search_body={"plugins": [{"id": "web"}]},
        search_route="chat",
        replay_field="reasoning",
        in_band_errors=True,
    ),
    # Together: hybrid-reasoning and sampling extensions. Its media endpoints
    # differ from OpenAI's, so none are declared until verified.
    "together": _VendorProfile(
        hosts=("api.together.ai", "api.together.xyz"),
        extra_params=frozenset(
            {"reasoning", "chat_template_kwargs", "top_k", "min_p", "repetition_penalty",
             "safety_model", "context_length_exceeded_behavior"}
        ),
    ),
}


def _vendor_profile(name: str, base_url: str) -> _VendorProfile | None:
    """The profile for an instance, by name first and then by host."""
    if name in _VENDOR_PROFILES:
        return _VENDOR_PROFILES[name]
    for profile in _VENDOR_PROFILES.values():
        if any(host in base_url for host in profile.hosts):
            return profile
    return None


def _merge_reasoning_detail(details: list[dict[str, Any]], item: Any) -> None:
    """Fold one streamed ``reasoning_details`` item into the turn's list.

    Items with the same ``index`` and ``type`` are pieces of one item: their
    text-like fields concatenate and the rest (id, format, signature) update.
    """
    if not isinstance(item, Mapping):
        return
    index = item.get("index")
    for existing in details:
        if index is not None and existing.get("index") == index and existing.get("type") == item.get("type"):
            for key, value in item.items():
                if key in ("text", "summary", "data") and isinstance(value, str):
                    existing[key] = (existing.get(key) or "") + value
                elif value is not None:
                    existing[key] = value
            return
    details.append(dict(item))


def nearest_effort(requested: str, supported: Sequence[str]) -> str | None:
    """The supported effort level closest to ``requested`` (higher on a tie)."""
    order = {name: i for i, name in enumerate(_EFFORT_ORDER)}
    known = [s for s in supported if s in order]
    if not known:
        return None
    if requested not in order:
        return known[len(known) // 2]
    target = order[requested]
    return min(known, key=lambda s: (abs(order[s] - target), -order[s]))


@dataclass
class _ModelRules:
    """What one model has been seen to reject, learned from its 400s."""

    drop: set[str] = field(default_factory=set)
    rename: dict[str, str] = field(default_factory=dict)
    effort: dict[str, str] = field(default_factory=dict)
    responses_only: bool = False
    chat_only: bool = False


class _ModelList:
    """Adapt a raw ``/models`` JSON payload to the SDK's listing shape.

    The loop that consumes this reads ``.data`` and then ``m.id``, so the direct
    transport returns objects with the same two accessors rather than branching
    the caller.
    """

    __slots__ = ("data",)

    def __init__(self, entries: list[dict[str, Any]]) -> None:
        self.data = [_ModelEntry(e) for e in entries]


class _ModelEntry:
    """One model record from a raw ``/models`` payload."""

    __slots__ = ("_raw", "id")

    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw
        self.id = str(raw.get("id", ""))

    def __getattr__(self, item: str) -> Any:
        return self._raw.get(item)


#: Public API root, used by the direct transport when no ``base_url`` is set.
_OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIProvider(BaseProvider):
    """LLMCore provider for the OpenAI API.

    Multimodal via Message.metadata conventions:
    - metadata["content_parts"]: raw content part list (overrides content)
    - metadata["inline_images"]: list of image dicts -> image_url parts
    - metadata["inline_audio"]: list of audio dicts -> input_audio parts
    - metadata["tool_calls"]: assistant tool calls for multi-turn
    - metadata["name"]: participant name
    """

    _client: AsyncOpenAI | None = None
    _encoding: Any | None = None
    _api_key_env_var: str | None = None

    #: Transport for chat and model listing: ``"sdk"`` (default) or ``"httpx"``.
    #: Declared at class level so instances built without ``__init__`` — which
    #: tests do with ``object.__new__`` to stub the client — still resolve it.
    _transport: str = "sdk"

    #: Extra headers the direct transport sends, populated by subclasses.
    #: The class-level default is immutable on purpose: a shared mutable dict
    #: would let one instance's headers leak into every other, and an
    #: accidental mutation of the default now raises instead of doing that
    #: silently. ``__init__`` replaces it with a per-instance dict.
    _direct_headers: ClassVar[Mapping[str, str]] = MappingProxyType({})

    #: Lazily-built direct HTTP client.
    _http: Any = None

    #: Responses routing and parameter adaptation; see ``__init__``. Class-level
    #: defaults keep instances built with ``object.__new__`` (as tests do) on
    #: the old behaviour: Chat Completions only, nothing adapted.
    _native_openai: bool = False
    _vendor: _VendorProfile | None = None
    _responses_policy: str = "never"
    _adaptive_params: bool = False
    _model_rules: dict[str, _ModelRules] | None = None

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False):
        super().__init__(config, log_raw_payloads)
        if not openai_available:
            raise ImportError("OpenAI library not installed.")
        if not tiktoken_available:
            raise ImportError("tiktoken library not installed.")

        # Kept for the direct transport's organization/project headers, which
        # read it; without it those headers were silently never sent.
        self.config = dict(config)
        self._api_key_env_var = config.get("api_key_env_var")
        api_key = config.get("api_key")
        if not api_key and self._api_key_env_var:
            api_key = os.environ.get(self._api_key_env_var)
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")

        self.api_key = api_key
        self.base_url = config.get("base_url")
        self.default_model = config.get("default_model", DEFAULT_MODEL)
        self.timeout = float(config.get("timeout", 60.0))
        self.max_retries = int(config.get("max_retries", 2))

        if not self.api_key:
            raise ConfigError("OpenAI API key not found. Set OPENAI_API_KEY or configure api_key.")

        try:
            client_kwargs: dict[str, Any] = {
                "api_key": self.api_key,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            org = config.get("organization")
            if org:
                client_kwargs["organization"] = org
            project = config.get("project")
            if project:
                client_kwargs["project"] = project
            # The SDK path needs these too. Without this the docstring below
            # ("so both transports send the same request") was false for
            # configured headers: ``default_headers`` reached only the direct
            # transport, and ``sdk`` is the default -- so a config that relied
            # on a header had it silently dropped. That matters for an endpoint
            # whose authentication *is* a header, such as a runtime served
            # behind an HTTP Basic tunnel.
            headers = config.get("default_headers")
            if headers:
                client_kwargs["default_headers"] = dict(headers)
            self._client = AsyncOpenAI(**client_kwargs)
            logger.debug("AsyncOpenAI client initialized.")
        except Exception as e:
            raise ConfigError(f"OpenAI client initialization failed: {e}")

        # --- Direct (httpx) transport ---------------------------------------
        #
        # The key name is ``transport``, not ``backend``: OpenRouter and Poe
        # already use ``backend`` on their own configs to mean "native vendor
        # SDK vs OpenAI-compatible", and silently overloading it would make one
        # of the two settings unreachable.
        #
        # The default stays ``"sdk"`` deliberately. This class is the base for
        # deepinfra, vllm, poe and openrouter, and their suites mock
        # ``AsyncOpenAI``; flipping the default would route five providers past
        # their own tests, which is exactly how the Z.ai SDK backend broke 21
        # tests when it was introduced. Direct transport is opt-in per instance.
        self._transport = str(config.get("transport") or "sdk").lower()
        if self._transport not in ("sdk", "httpx"):
            logger.warning(
                "Unknown transport '%s' for %s; using the SDK.",
                self._transport,
                self.get_name(),
            )
            self._transport = "sdk"
        if self._transport == "httpx" and not httpx_available:
            logger.warning(
                "Transport 'httpx' requested for %s but httpx is not installed; "
                "using the SDK.",
                self.get_name(),
            )
            self._transport = "sdk"

        #: Extra headers the direct transport sends. Subclasses that customize
        #: the SDK client's headers (OpenRouter's HTTP-Referer / X-Title) add
        #: them here too, so both transports send the same request.
        # A fresh dict per instance: mutating the class-level default would
        # leak one provider's headers into every other instance.
        self._direct_headers = dict(config.get("default_headers") or {})
        self._http = None

        # --- Responses API and per-model parameter rules -------------------
        #
        # Both are on by default only against OpenAI itself. This class is the
        # base for a dozen OpenAI-compatible servers, most of which have no
        # /responses and word their errors differently, so for them both stay
        # off unless configured.
        base = str(self.base_url or "")
        self._native_openai = not base or "api.openai.com" in base
        self._vendor = None if self._native_openai else _vendor_profile(self.get_name(), base)
        if self._vendor is not None:
            # Instance-level, so OpenAI's own declaration stays on the class.
            self._MEDIA_CAPABILITIES = self._vendor.media
        policy = str(
            config.get("responses_api") or ("auto" if self._native_openai else "never")
        ).lower()
        if policy not in RESPONSES_POLICIES:
            logger.warning(
                "Unknown responses_api '%s' for %s; using 'auto'.", policy, self.get_name()
            )
            policy = "auto"
        self._responses_policy = policy
        param_policy = str(
            config.get("param_policy") or ("adapt" if self._native_openai else "strict")
        ).lower()
        self._adaptive_params = param_policy == "adapt"
        self._model_rules = {}

        self._load_tokenizer(self.default_model)

    # ------------------------------------------------------------------
    # Direct transport
    # ------------------------------------------------------------------

    def _direct_base_url(self) -> str:
        """Return the REST root for the direct transport."""
        return str(self.base_url or _OPENAI_DEFAULT_BASE_URL).rstrip("/")

    def _get_http(self) -> Any:
        """Return the lazily-built direct HTTP client."""
        if self._http is None:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                **self._direct_headers,
            }
            org = self.config.get("organization") if hasattr(self, "config") else None
            if org:
                headers["OpenAI-Organization"] = str(org)
            project = self.config.get("project") if hasattr(self, "config") else None
            if project:
                headers["OpenAI-Project"] = str(project)
            self._http = httpx.AsyncClient(
                base_url=self._direct_base_url(),
                headers=headers,
                timeout=self.timeout,
            )
        return self._http

    def _is_context_overflow(self, status: int | None, text: str, code: str | None) -> bool:
        """OpenAI's prompt-too-long error, by code first and wording second.

        From SDK 3.38 ``str(APIStatusError)`` carries only the message, not the
        body, so the code (``context_length_exceeded``) has to be read from
        the exception. Chat words it "maximum context length is N tokens";
        Responses says "exceeds the context window".
        """
        if code == "context_length_exceeded":
            return True
        lowered = text.lower()
        return status in (400, None) and (
            "context_length" in lowered
            or "maximum context length" in lowered
            or "exceeds the context window" in lowered
            or "exceed the configured limit" in lowered
        )

    def _raise_direct_status(self, status: int, body: str, model_name: str) -> None:
        """Map a direct-transport HTTP failure onto llmcore's exceptions.

        Deliberately mirrors the SDK branch below, including the
        :class:`ContextLengthError` and model-not-found cases, so a caller sees
        the same exception for the same condition whichever transport ran. A
        dual transport that reports failures differently is not really dual.

        Raises:
            ContextLengthError: If the prompt exceeded the model's window.
            ProviderError: For every other failure.
        """
        lowered = body.lower()
        if self._is_context_overflow(status, body, None):
            raise ContextLengthError(
                model_name=model_name,
                limit=self.get_max_context_length(model_name),
                actual=0,
                message=body,
            )
        if status in (400, 404) and any(
            phrase in lowered
            for phrase in ("model not exist", "does not exist", "model_not_found", "invalid model",
                           "not a valid model id")
        ):
            raise ProviderError(
                self.get_name(),
                f"Model '{model_name}' not found on provider '{self.get_name()}'. "
                f"The provider's default model is '{self.default_model}'. Please verify "
                f"the model name is correct for this provider's API. "
                f"Original error: {body}",
            )
        raise ProviderError(
            self.get_name(),
            f"API Error ({status}): {body}",
            model_name=model_name,
            status_code=status,
            retryable=status == 429 or status >= 500,
        )

    async def _chat_via_httpx(
        self,
        model_name: str,
        messages_payload: list[dict[str, Any]],
        stream: bool,
        api_kwargs: dict[str, Any],
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Run a chat completion over the direct transport.

        Returns exactly the shapes the SDK path returns — a plain response dict,
        or an async generator of chunk dicts — because every ``extract_*``
        method downstream parses those dicts. The transports have to be
        interchangeable at that boundary or the choice leaks into callers.
        """
        api_kwargs = dict(api_kwargs)
        extra_body = api_kwargs.pop("extra_body", None) or {}
        api_kwargs.pop("extra_headers", None)
        body: dict[str, Any] = {
            "model": model_name,
            "messages": messages_payload,
            "stream": stream,
            **api_kwargs,
            # The SDK merges extra_body into the request body; sending it as a
            # literal "extra_body" field (as this path used to) is a request the
            # server does not understand.
            **dict(extra_body),
        }
        client = self._get_http()

        if not stream:
            try:
                resp = await client.post("/chat/completions", json=body)
            except httpx.TimeoutException as e:
                raise ProviderError(self.get_name(), f"Timeout: {e}") from e
            except httpx.HTTPError as e:
                raise ProviderError(self.get_name(), f"Connection error: {e}") from e
            if resp.status_code >= 400:
                self._raise_direct_status(resp.status_code, resp.text, model_name)
            response_dict = resp.json()
            if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "RAW LLM RESPONSE (%s, direct): %s",
                    self.get_name(),
                    json.dumps(response_dict, indent=2, default=str),
                )
            return response_dict

        return self._stream_via_httpx(client, body, model_name)

    async def _stream_via_httpx(
        self, client: Any, body: dict[str, Any], model_name: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Yield chunk dicts from a server-sent-event stream."""
        try:
            async with client.stream("POST", "/chat/completions", json=body) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    self._raise_direct_status(resp.status_code, resp.text, model_name)
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    # The terminator is a sentinel string, not JSON; parsing it
                    # would raise on the last chunk of every successful stream.
                    if payload == "[DONE]":
                        return
                    try:
                        chunk = json.loads(payload)
                    except ValueError:
                        logger.warning("Skipping unparseable SSE chunk from %s.", self.get_name())
                        continue
                    if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "RAW STREAM CHUNK (%s, direct): %s", self.get_name(), payload
                        )
                    yield chunk
        except ProviderError:
            raise
        except httpx.TimeoutException as e:
            raise ProviderError(self.get_name(), f"Timeout: {e}") from e
        except httpx.HTTPError as e:
            raise ProviderError(self.get_name(), f"Connection error: {e}") from e

    def _load_tokenizer(self, model_name: str):
        if not tiktoken:
            self._encoding = None
            return
        # Gateways prefix the vendor ("openai/gpt-5.4-mini"); tiktoken knows
        # the bare name.
        model_name = model_name.rsplit("/", 1)[-1]
        try:
            self._encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            fallback = (
                "o200k_base"
                if any(s in model_name for s in ("4o", "gpt-5", "gpt-6", "gpt-7", "gpt-4.1", "o1", "o3", "o4"))
                else "cl100k_base"
            )
            logger.warning(f"No tiktoken encoding for '{model_name}'. Using '{fallback}'.")
            try:
                self._encoding = tiktoken.get_encoding(fallback)
            except Exception:
                self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.error(f"Failed to load tiktoken: {e}", exc_info=True)
            self._encoding = None

    def get_name(self) -> str:
        return self._provider_instance_name or "openai"

    def _is_openai_itself(self) -> bool:
        """True for OpenAI's own API (also for test stubs built without __init__)."""
        if self._native_openai:
            return True
        return getattr(self, "base_url", None) is None and self.get_name() == "openai"

    def _profile(self) -> _VendorProfile | None:
        """This instance's vendor profile (by name when built without __init__)."""
        if self._vendor is not None or self._is_openai_itself():
            return self._vendor
        return _VENDOR_PROFILES.get(self.get_name())

    def supports_native_search(self, model: str | None = None) -> bool:
        """Whether this instance has a native web-search surface for ``model``.

        OpenAI: the Responses ``web_search`` tool (every model), or Chat
        Completions search on ``*-search-api`` models. xAI: ``web_search`` and
        ``x_search`` on its Responses API (Live Search was retired). Groq:
        ``browser_search`` on the gpt-oss models. Everything else reports
        ``False``, so ``native_search`` stays a clean no-op.
        """
        if self._is_openai_itself():
            return True
        vendor = self._profile()
        if vendor is None or not (vendor.search_tools or vendor.search_body):
            return False
        if vendor.search_models and model and not re.search(vendor.search_models, model):
            return False
        return True

    def _apply_native_search(self, api_kwargs: dict[str, Any], model: str | None = None) -> None:
        """Attach the instance's Chat Completions search to ``api_kwargs``.

        Additive: caller-supplied tools and options are preserved. Vendors whose
        search lives on /responses are routed there by ``chat_completion``
        instead, so nothing is added here for them.
        """
        vendor = self._profile()
        if vendor is not None and vendor.search_route == "chat":
            if vendor.search_models and model and not re.search(vendor.search_models, model):
                return  # e.g. Groq's browser_search exists only on gpt-oss
            if vendor.search_tools:
                tools = list(api_kwargs.get("tools") or [])
                tools.extend(dict(t) for t in vendor.search_tools)
                api_kwargs["tools"] = tools
            for key, value in vendor.search_body.items():
                if key == "plugins":
                    present = list(api_kwargs.get("plugins") or [])
                    ids = {p.get("id") for p in present if isinstance(p, Mapping)}
                    present.extend(dict(p) for p in value if p.get("id") not in ids)
                    api_kwargs["plugins"] = present
                else:
                    api_kwargs.setdefault(key, value)
        elif self._is_openai_itself():
            # Reached for *-search-api models (others go to /responses) and
            # when responses_api = "never".
            api_kwargs.setdefault("web_search_options", {})
        else:  # pragma: no cover - guarded by supports_native_search
            logger.debug("native_search ignored for '%s': no search surface.", self.get_name())

    async def get_models_details(self) -> list[ModelDetails]:
        if not self._client:
            raise ProviderError(self.get_name(), "Client not initialized.")
        try:
            if self._transport == "httpx":
                resp = await self._get_http().get("/models")
                if resp.status_code >= 400:
                    self._raise_direct_status(resp.status_code, resp.text, "models")
                models_response = _ModelList(resp.json().get("data") or [])
            else:
                models_response = await self._client.models.list()
            result = []
            provider_name = self.get_name()
            try:
                registry = get_model_card_registry()
            except Exception:
                registry = None

            for m in models_response.data:
                mid = m.id
                ctx = self.get_max_context_length(mid)
                supports_tools = False
                supports_vision = False
                if registry:
                    card = registry.get(provider_name, mid)
                    if card and card.capabilities:
                        supports_tools = (
                            card.capabilities.tool_use or card.capabilities.function_calling
                        )
                        supports_vision = card.capabilities.vision
                if not supports_tools:
                    supports_tools = bool(
                        re.match(r"(gpt-3\.5-turbo|gpt-[4-9]|o\d|chatgpt|codex)", mid)
                    ) and not re.search(r"(tts|transcribe|whisper|image|embedding|moderation|realtime|audio)", mid)
                result.append(
                    ModelDetails(
                        id=mid,
                        context_length=ctx,
                        supports_streaming=True,
                        supports_tools=supports_tools,
                        supports_vision=supports_vision,
                        provider_name=provider_name,
                        metadata={"owned_by": m.owned_by, "created": m.created},
                    )
                )
            return result
        except OpenAIError as e:
            raise ProviderError(self.get_name(), f"Failed to fetch models: {e}")

    def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
        model_name = model or self.default_model
        is_reasoning = _is_reasoning_model(model_name)
        params: dict[str, Any] = {
            "temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0},
            "top_p": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "max_completion_tokens": {"type": "integer", "minimum": 1},
            "presence_penalty": {"type": "number", "minimum": -2.0, "maximum": 2.0},
            "frequency_penalty": {"type": "number", "minimum": -2.0, "maximum": 2.0},
            "stop": {"type": "array", "items": {"type": "string"}},
            "seed": {"type": "integer"},
            "n": {"type": "integer", "minimum": 1},
            "response_format": {"type": "object"},
            "logprobs": {"type": "boolean"},
            "top_logprobs": {"type": "integer", "minimum": 0, "maximum": 20},
            "logit_bias": {"type": "object"},
            "parallel_tool_calls": {"type": "boolean"},
            "stream_options": {"type": "object"},
            "modalities": {"type": "array", "items": {"type": "string"}},
            "audio": {"type": "object"},
            "reasoning_effort": {
                "type": "string",
                "enum": ["none", "minimal", "low", "medium", "high", "xhigh", "max"],
                "description": "Accepted levels vary by model; see param_policy.",
            },
            "effort": {
                "type": "string",
                "enum": ["none", "minimal", "low", "medium", "high", "xhigh", "max"],
                "description": "llmcore's canonical reasoning effort.",
            },
            "responses": {
                "type": "boolean",
                "description": "True: use /responses; False: use /chat/completions.",
            },
            "prediction": {"type": "object"},
            "service_tier": {"type": "string"},
            "store": {"type": "boolean"},
            "metadata": {"type": "object"},
            "user": {"type": "string"},
            "prompt_cache_key": {"type": "string"},
            "prompt_cache_retention": {"type": "string"},
            "web_search_options": {"type": "object"},
            "verbosity": {"type": "string", "enum": ["low", "medium", "high"]},
        }
        if not is_reasoning or self._native_openai:
            # Renamed to max_completion_tokens on the way out for OpenAI itself.
            params["max_tokens"] = {"type": "integer", "minimum": 1}
        if self._responses_policy != "never":
            params.update(_RESPONSES_EXTRA_PARAMS)
        params["extra_body"] = {"type": "object", "description": "Extra request-body fields."}
        if self._vendor is not None:
            for name in self._vendor.extra_params:
                params.setdefault(name, {"description": f"{self.get_name()} extension"})
        return params

    def get_max_context_length(self, model: str | None = None) -> int:
        model_name = model or self.default_model
        limit = DEFAULT_OPENAI_TOKEN_LIMITS.get(model_name)
        if limit is None:
            # Known prefixes first: the substring checks below once matched
            # "gpt-4" inside "gpt-4.1" and reported 8,192 for a 1M model, and
            # several generated cards still carry a placeholder 128,000.
            for prefix, ctx in _OPENAI_PREFIX_CONTEXT_HEURISTICS:
                if model_name.startswith(prefix):
                    return ctx
        if limit is None:
            if "gpt-4o" in model_name:
                limit = 128000
            elif "gpt-4-turbo" in model_name:
                limit = 128000
            elif "gpt-4-32k" in model_name:
                limit = 32768
            elif "gpt-4" in model_name:
                limit = 8192
            elif "gpt-3.5-turbo" in model_name:
                limit = 16385
            else:
                provider_name = self.get_name()
                try:
                    registry = get_model_card_registry()
                    card = registry.get(provider_name, model_name)
                    if card is not None:
                        limit = card.get_context_length()
                    else:
                        for prefix, ctx in _OPENAI_PREFIX_CONTEXT_HEURISTICS:
                            if model_name.startswith(prefix):
                                limit = ctx
                                break
                        if limit is None:
                            limit = 4096
                            if model_name not in _warned_unknown_models:
                                _warned_unknown_models.add(model_name)
                                logger.warning(
                                    "Unknown context for '%s'. Fallback: %d.", model_name, limit
                                )
                except Exception as e:
                    limit = 4096
                    if model_name not in _warned_unknown_models:
                        _warned_unknown_models.add(model_name)
                        logger.warning(
                            "Context lookup failed for '%s' (%s). Fallback: %d.",
                            model_name,
                            e,
                            limit,
                        )
        return limit

    def _build_message_payload(self, msg: Message, model_name: str) -> dict[str, Any]:
        """Build an OpenAI-format message dict from an llmcore Message."""
        role_str = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        metadata = msg.metadata or {}

        if role_str == "system" and _needs_developer_role(model_name):
            role_str = "developer"

        content: Any = msg.content
        if "content_parts" in metadata:
            content = metadata["content_parts"]
        elif "inline_images" in metadata or "inline_audio" in metadata:
            parts: list[dict[str, Any]] = []
            if msg.content:
                parts.append({"type": "text", "text": msg.content})
            for img in metadata.get("inline_images", []):
                if isinstance(img, str):
                    parts.append({"type": "image_url", "image_url": {"url": img}})
                elif isinstance(img, dict):
                    img_p: dict[str, Any] = {"url": img.get("url", "")}
                    if "detail" in img:
                        img_p["detail"] = img["detail"]
                    parts.append({"type": "image_url", "image_url": img_p})
            for aud in metadata.get("inline_audio", []):
                if isinstance(aud, dict):
                    parts.append(
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": aud.get("data", ""),
                                "format": aud.get("format", "wav"),
                            },
                        }
                    )
            content = parts

        msg_dict: dict[str, Any] = {"role": role_str, "content": content}

        if msg.role == LLMCoreRole.TOOL and msg.tool_call_id:
            msg_dict["tool_call_id"] = msg.tool_call_id

        # First-class Message.tool_calls (R-2) takes precedence over the
        # legacy metadata["tool_calls"] channel.
        tool_calls = getattr(msg, "tool_calls", None) or metadata.get("tool_calls")
        if role_str == "assistant" and tool_calls:
            msg_dict["tool_calls"] = tool_calls
            if not msg.content:
                msg_dict["content"] = None

        if role_str == "assistant" and "audio" in metadata:
            msg_dict["audio"] = metadata["audio"]

        # OpenAI-compatible vendors carry reasoning across turns in the
        # assistant message (DeepInfra/Groq: reasoning_content, vLLM and
        # OpenRouter: reasoning, OpenRouter: reasoning_details, which must go
        # back unmodified and in order). OpenAI itself uses reasoning items on
        # /responses instead and rejects unknown message fields.
        if role_str == "assistant" and not self._is_openai_itself():
            if metadata.get("reasoning_details"):
                msg_dict["reasoning_details"] = list(metadata["reasoning_details"])
            text = metadata.get("reasoning_content") or metadata.get("reasoning")
            if text:
                profile = self._profile()
                msg_dict[profile.replay_field if profile else "reasoning_content"] = text

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
        native_search: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Send a chat request over Chat Completions or the Responses API.

        ``native_search=True`` against OpenAI uses the Responses ``web_search``
        tool, which every current model has; Chat Completions offers search
        only on the ``*-search-api`` models, so those stay there.

        Which one is used follows ``responses_api`` (see
        :data:`RESPONSES_POLICIES`) and the call itself: ``responses=True`` or
        ``False`` forces a route, a Chat-only parameter (``n``, ``logprobs``,
        ``seed``, ``stop``, …) keeps the call on Chat Completions, and a
        Responses-only one (``reasoning``, ``text``, ``include``,
        ``previous_response_id``, …) sends it to ``/responses``. Both routes
        return the same Chat Completions-shaped dict.

        Against OpenAI itself, a model's parameter rules are learned from its
        own errors (``param_policy = "adapt"``, the default there). When a
        model rejects a parameter or value, the call is adjusted and retried
        once, and the rule is remembered for that model: ``max_tokens`` is
        renamed, an unsupported ``temperature``/``top_p`` is dropped, an effort
        level moves to the nearest supported one, and a Responses-only model
        moves to ``/responses``. Each adjustment is logged once.
        ``param_policy = "strict"`` turns this off.
        """
        if not self._client:
            raise ProviderError(self.get_name(), "Client not initialized.")
        model_name = model or self.default_model
        prefer = kwargs.pop("responses", None)
        if native_search and prefer is None:
            if (
                self._native_openai
                and self._responses_policy != "never"
                and not _CHAT_PREFERRED.search(model_name)
            ):
                prefer = True
            elif (profile := self._profile()) is not None and profile.search_route == "responses":
                prefer = True  # xAI: search exists only on /responses
        supported = self.get_supported_parameters(model_name)
        unknown = [key for key in kwargs if key not in supported]
        if unknown:
            # Checked before routing, so neither route sends anything for a
            # parameter this provider does not know.
            raise ValueError(f"Unsupported parameter '{unknown[0]}' for OpenAI provider.")
        attempts = 3 if self._adaptive_params else 1
        last_error: ProviderError | None = None
        for _attempt in range(attempts):
            call_kwargs = self._apply_model_rules(model_name, dict(kwargs))
            use_responses = self._use_responses(model_name, call_kwargs, context, prefer)
            try:
                if use_responses:
                    result = await self._responses_completion(
                        context, model_name, stream, tools, tool_choice, native_search, **call_kwargs
                    )
                else:
                    result = await self._chat_completions_once(
                        context, model_name, stream, tools, tool_choice, native_search, **call_kwargs
                    )
                if not use_responses and not self._is_openai_itself():
                    if stream:
                        result = self._compat_stream(result, model_name)  # type: ignore[arg-type]
                    else:
                        self._raise_in_band(result, model_name)  # type: ignore[arg-type]
                if stream and self._transport == "httpx":
                    # The direct stream is lazy: pull its first chunk now so a
                    # rejected request fails here, where it can be adapted.
                    result = await self._primed(result)  # type: ignore[arg-type]
                return result
            except ContextLengthError:
                raise
            except ProviderError as exc:
                last_error = exc
                if not self._learn_from_error(model_name, exc, call_kwargs, use_responses):
                    raise
        assert last_error is not None
        raise last_error

    # ------------------------------------------------------------------
    # Routing and learned parameter rules
    # ------------------------------------------------------------------

    def _use_responses(
        self, model: str, kwargs: Mapping[str, Any], context: Any, prefer: bool | None
    ) -> bool:
        """Whether this call goes to ``/responses``."""
        rules = (self._model_rules or {}).get(model)
        if rules is not None and rules.responses_only:
            return True  # the model serves nothing else
        if rules is not None and rules.chat_only:
            return False
        if prefer is not None:
            return bool(prefer)
        if self._responses_policy == "never":
            return False
        if CHAT_ONLY_PARAMS & set(kwargs):
            return False
        if RESPONSES_ONLY_PARAMS & set(kwargs):
            return True
        if self._responses_policy == "always":
            return True
        # auto
        if isinstance(context, list) and any(
            "inline_audio" in (m.metadata or {}) for m in context if isinstance(m, Message)
        ):
            return False  # audio input is a Chat Completions feature
        if _CHAT_PREFERRED.search(model):
            return False
        return bool(_RESPONSES_PREFERRED.search(model))

    def _apply_model_rules(self, model: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Rewrite ``kwargs`` with what ``model`` is known to reject."""
        rules = (self._model_rules or {}).get(model)
        if rules is None:
            return kwargs
        for key in rules.drop:
            kwargs.pop(key, None)
        for old, new in rules.rename.items():
            if old in kwargs and new not in kwargs:
                kwargs[new] = kwargs.pop(old)
        for key in ("effort", "reasoning_effort"):
            value = kwargs.get(key)
            if isinstance(value, str) and value in rules.effort:
                kwargs[key] = rules.effort[value]
        reasoning = kwargs.get("reasoning")
        if isinstance(reasoning, dict) and reasoning.get("effort") in rules.effort:
            kwargs["reasoning"] = {**reasoning, "effort": rules.effort[reasoning["effort"]]}
        return kwargs

    def _learn_from_error(
        self, model: str, exc: ProviderError, kwargs: Mapping[str, Any], responses: bool
    ) -> bool:
        """Learn a rule from OpenAI's rejection of this call.

        Returns:
            Whether a rule changed this call, so a retry may succeed.
        """
        if not self._adaptive_params:
            return False
        text = str(exc)
        if self._model_rules is None:
            self._model_rules = {}
        rules = self._model_rules.setdefault(model, _ModelRules())
        note: str | None = None

        def present(wire: str) -> list[str]:
            names = _WIRE_KWARGS.get(wire, (wire,))
            hits = [n for n in names if n in kwargs]
            if wire.startswith("reasoning") and isinstance(kwargs.get("reasoning"), dict):
                hits.append("reasoning")
            return hits

        if (m := _RX_USE_INSTEAD.search(text)) and present(m.group(1)):
            for name in present(m.group(1)):
                rules.rename[name] = m.group(2)
            note = f"renamed {m.group(1)} to {m.group(2)}"
        elif _RX_RESPONSES_ONLY.search(text) and not responses:
            rules.responses_only = True
            note = "served only on /responses; routing there"
        elif _RX_NOT_ON_RESPONSES.search(text) and responses:
            rules.chat_only = True
            note = "not served on /responses; routing to Chat Completions"
        elif (m := _RX_SUPPORTED_VALUES.search(text)) and present(m.group(1)):
            requested = m.group(2)
            supported = re.findall(r"'([^']+)'", m.group(3))
            chosen = nearest_effort(requested, supported)
            if chosen is None or chosen == requested:
                return False
            rules.effort[requested] = chosen
            note = f"{m.group(1)} '{requested}' is unsupported; using '{chosen}'"
        elif (m := _RX_SUPPORTED_VALUES_NAMELESS.search(text)) and self._effort_requested(
            kwargs, m.group(1)
        ):
            requested = m.group(1)
            chosen = nearest_effort(requested, re.findall(r"'([^']+)'", m.group(2)))
            if chosen is None or chosen == requested:
                return False
            rules.effort[requested] = chosen
            note = f"effort '{requested}' is unsupported; using '{chosen}'"
        else:
            wire = None
            for rx in (_RX_DEFAULT_ONLY, _RX_UNSUPPORTED_PARAM, _RX_UNRECOGNIZED):
                if (m := rx.search(text)) is not None:
                    wire = m.group(1)
                    break
            if wire is None and _RX_NO_LOGPROBS.search(text):
                wire = "logprobs"
            hits = present(wire) if wire else []
            if not hits:
                return False
            rules.drop.update(hits)
            note = f"dropped {', '.join(sorted(hits))} (unsupported)"

        logger.warning(
            "%s: model '%s' rejected the request; %s, and retrying. This is remembered "
            "for the model. Set param_policy = \"strict\" to fail instead.",
            self.get_name(),
            model,
            note,
        )
        return True

    @staticmethod
    def _effort_requested(kwargs: Mapping[str, Any], value: str) -> bool:
        """Whether this call asks for effort ``value`` under any spelling."""
        if value in (kwargs.get("effort"), kwargs.get("reasoning_effort")):
            return True
        reasoning = kwargs.get("reasoning")
        return isinstance(reasoning, Mapping) and reasoning.get("effort") == value

    @staticmethod
    async def _primed(
        stream: AsyncGenerator[dict[str, Any], None],
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Pull the first chunk now; replay it, then the rest."""
        try:
            first = await stream.__anext__()
        except StopAsyncIteration:
            first = None

        async def replay() -> AsyncGenerator[dict[str, Any], None]:
            if first is not None:
                yield first
            async for chunk in stream:
                yield chunk

        return replay()

    def _in_band_error(self, error: Any, model_name: str) -> ProviderError:
        body = error if isinstance(error, Mapping) else {"message": str(error)}
        metadata = body.get("metadata") if isinstance(body.get("metadata"), Mapping) else {}
        code = body.get("code")
        status = code if isinstance(code, int) else None
        detail = metadata.get("error_type") or metadata.get("provider_name") or ""
        return ProviderError(
            self.get_name(),
            f"Upstream error{f' ({detail})' if detail else ''}: {body.get('message') or body}",
            model_name=model_name,
            status_code=status,
            retryable=None if status is None else status in (408, 429, 500, 502, 503),
        )

    def _raise_in_band(self, response: Mapping[str, Any], model_name: str) -> None:
        """Raise for a failure reported inside an HTTP 200 body."""
        profile = self._profile()
        if profile is None or not profile.in_band_errors or not isinstance(response, Mapping):
            return
        if response.get("error") and not response.get("choices"):
            raise self._in_band_error(response["error"], model_name)
        for choice in response.get("choices") or []:
            if choice.get("finish_reason") == "error":
                raise self._in_band_error(choice.get("error") or response.get("error") or {}, model_name)

    async def _compat_stream(
        self, stream: AsyncGenerator[dict[str, Any], None], model_name: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        """An OpenAI-compatible vendor's stream, made replayable and honest.

        The reasoning of the turn arrives in pieces (``reasoning`` /
        ``reasoning_content`` text, ``reasoning_details`` items); the chunk with
        the finish reason also carries the whole of it as
        ``replay_reasoning_content`` / ``replay_reasoning_details`` for
        :meth:`assistant_replay_metadata`. In-band errors raise.
        """
        profile = self._profile()
        check_errors = bool(profile and profile.in_band_errors)
        text: list[str] = []
        details: list[dict[str, Any]] = []
        async for chunk in stream:
            if check_errors and chunk.get("error"):
                raise self._in_band_error(chunk["error"], model_name)
            for choice in chunk.get("choices") or []:
                if choice.get("index", 0) != 0:
                    continue
                delta = choice.get("delta") or {}
                piece = delta.get("reasoning_content") or delta.get("reasoning")
                if isinstance(piece, str) and piece:
                    text.append(piece)
                for item in delta.get("reasoning_details") or []:
                    _merge_reasoning_detail(details, item)
                if check_errors and choice.get("finish_reason") == "error":
                    raise self._in_band_error(choice.get("error") or {}, model_name)
                if choice.get("finish_reason") and (text or details):
                    delta = choice.setdefault("delta", delta)
                    if text:
                        delta["replay_reasoning_content"] = "".join(text)
                    if details:
                        delta["replay_reasoning_details"] = [dict(d) for d in details]
            yield chunk

    # ------------------------------------------------------------------
    # Responses API
    # ------------------------------------------------------------------

    def _on_http_response(self, response: Any) -> None:
        """Hook: called with every direct-transport response (headers etc.)."""

    def _raise_http_error(self, response: Any, model_name: str) -> None:
        """Hook: raise the error for a failed direct-transport response."""
        self._raise_direct_status(response.status_code, response.text, model_name)

    def _responses_failed(self, exc: rsp.ResponsesFailed, model_name: str) -> ProviderError:
        """Hook: the error for a response whose ``status`` is ``failed``."""
        code = str(exc.error.get("code") or "")
        if code == "context_length_exceeded":
            return ContextLengthError(  # type: ignore[return-value]
                model_name=model_name,
                limit=self.get_max_context_length(model_name),
                actual=0,
                message=str(exc.error.get("message") or code),
            )
        retryable = not code.startswith("invalid")
        err = ProviderError(
            self.get_name(),
            f"Response failed ({code or 'unknown'}): {exc.error.get('message') or 'no detail'}",
            model_name=model_name,
            status_code=429 if "rate_limit" in code else 400 if not retryable else 502,
            retryable=retryable,
        )
        err.partial_content = exc.partial  # type: ignore[attr-defined]
        return err

    async def _responses_completion(
        self,
        context: ContextPayload,
        model_name: str,
        stream: bool,
        tools: list[Tool] | None,
        tool_choice: str | None,
        native_search: bool,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """One call to ``POST /responses``, normalized to the Chat shape."""
        if not (isinstance(context, list) and all(isinstance(m, Message) for m in context)):
            raise ProviderError(self.get_name(), "Unsupported context type.")
        extra_tools = list(kwargs.pop("extra_tools", None) or [])
        if native_search and self._native_openai:
            extra_tools.append({"type": "web_search"})
        elif native_search and (profile := self._profile()) is not None and profile.search_route == "responses":
            extra_tools.extend(dict(t) for t in profile.search_tools)
        kwargs.pop("stream_options", None)
        body = rsp.build_responses_request(
            model_name,
            context,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
            extra_tools=extra_tools,
            **kwargs,
        )
        if not body["input"]:
            raise ProviderError(self.get_name(), "No valid messages.")
        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "RAW LLM REQUEST (%s, responses): %s",
                self.get_name(),
                json.dumps(body, indent=2, default=str),
            )
        if self._transport == "httpx":
            if stream:
                return self._responses_stream_httpx(body, model_name)
            return await self._responses_httpx(body, model_name)
        return await self._responses_sdk(body, model_name, stream)

    async def _responses_httpx(self, body: dict[str, Any], model_name: str) -> dict[str, Any]:
        try:
            resp = await self._get_http().post("/responses", json=body)
        except httpx.TimeoutException as e:
            raise ProviderError(self.get_name(), f"Timeout: {e}", model_name=model_name) from e
        except httpx.HTTPError as e:
            raise ProviderError(self.get_name(), f"Connection error: {e}", model_name=model_name) from e
        self._on_http_response(resp)
        if resp.status_code >= 400:
            self._raise_http_error(resp, model_name)
        data = resp.json()
        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "RAW LLM RESPONSE (%s, responses): %s",
                self.get_name(),
                json.dumps(data, indent=2, default=str),
            )
        try:
            return rsp.normalize_response(data)
        except rsp.ResponsesFailed as exc:
            raise self._responses_failed(exc, model_name) from None

    async def _responses_stream_httpx(
        self, body: dict[str, Any], model_name: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        normalizer = rsp.ResponsesStreamNormalizer()
        try:
            async with self._get_http().stream("POST", "/responses", json=body) as resp:
                self._on_http_response(resp)
                if resp.status_code >= 400:
                    await resp.aread()
                    self._raise_http_error(resp, model_name)
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue  # `event:` lines repeat the payload's own `type`
                    payload = line[5:].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        event = json.loads(payload)
                    except ValueError:
                        logger.warning("Skipping unparseable Responses event from %s.", self.get_name())
                        continue
                    try:
                        chunks = normalizer.feed(event)
                    except rsp.ResponsesFailed as exc:
                        raise self._responses_failed(exc, model_name) from None
                    for chunk in chunks:
                        yield chunk
        except ProviderError:
            raise
        except httpx.TimeoutException as e:
            raise ProviderError(self.get_name(), f"Timeout: {e}", model_name=model_name) from e
        except httpx.HTTPError as e:
            raise ProviderError(self.get_name(), f"Connection error: {e}", model_name=model_name) from e

    #: ``responses.create`` keywords the OpenAI SDK knows (shared table).
    _SDK_RESPONSES_KWARGS: ClassVar[frozenset[str]] = rsp.SDK_RESPONSES_KWARGS

    async def _responses_sdk(
        self, body: dict[str, Any], model_name: str, stream: bool
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        sdk_kwargs = {k: v for k, v in body.items() if k in self._SDK_RESPONSES_KWARGS}
        extra = {k: v for k, v in body.items() if k not in self._SDK_RESPONSES_KWARGS}
        if extra:
            sdk_kwargs["extra_body"] = extra
        try:
            result = await self._client.responses.create(**sdk_kwargs)  # type: ignore[union-attr]
        except OpenAIAPIStatusError as e:
            if self._is_context_overflow(e.status_code, str(e), getattr(e, "code", None)):
                raise ContextLengthError(
                    model_name=model_name,
                    limit=self.get_max_context_length(model_name),
                    actual=0,
                    message=str(e),
                ) from e
            raise ProviderError(
                self.get_name(),
                f"API Error ({e.status_code}): {e}",
                model_name=model_name,
                status_code=e.status_code,
                original_exception=e,
            ) from e
        except OpenAIAPITimeoutError as e:
            raise ProviderError(self.get_name(), f"Timeout: {e}", model_name=model_name) from e
        except OpenAIAPIConnectionError as e:
            raise ProviderError(self.get_name(), f"Connection error: {e}", model_name=model_name) from e
        except OpenAIError as e:
            raise ProviderError(self.get_name(), f"Error: {e}", model_name=model_name) from e
        if not stream:
            try:
                return rsp.normalize_response(result.model_dump(exclude_none=True))
            except rsp.ResponsesFailed as exc:
                raise self._responses_failed(exc, model_name) from None
        return self._responses_stream_sdk(result, model_name)

    async def _responses_stream_sdk(
        self, events: Any, model_name: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        normalizer = rsp.ResponsesStreamNormalizer()
        async for event in events:
            try:
                chunks = normalizer.feed(event.model_dump(exclude_none=True))
            except rsp.ResponsesFailed as exc:
                raise self._responses_failed(exc, model_name) from None
            for chunk in chunks:
                yield chunk

    # ------------------------------------------------------------------
    # Chat Completions
    # ------------------------------------------------------------------

    async def _chat_completions_once(
        self,
        context: ContextPayload,
        model: str | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        native_search: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """One Chat Completions call (no routing, no adaptation)."""
        if not self._client:
            raise ProviderError(self.get_name(), "Client not initialized.")

        model_name = model or self.default_model
        supported_params = self.get_supported_parameters(model_name)
        for key in kwargs:
            if key not in supported_params:
                raise ValueError(f"Unsupported parameter '{key}' for OpenAI provider.")
        effort = kwargs.pop("effort", None)
        if effort and "reasoning_effort" not in kwargs:
            kwargs["reasoning_effort"] = str(effort).strip().lower()
        if self._native_openai and "max_tokens" in kwargs:
            # OpenAI deprecated max_tokens; every current model takes
            # max_completion_tokens, and the GPT-5+/o-series reject max_tokens.
            if "max_completion_tokens" not in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
            else:
                kwargs.pop("max_tokens")

        if not (isinstance(context, list) and all(isinstance(msg, Message) for msg in context)):
            raise ProviderError(self.get_name(), "Unsupported context type.")

        messages_payload = [self._build_message_payload(msg, model_name) for msg in context]
        if not messages_payload:
            raise ProviderError(self.get_name(), "No valid messages.")

        tools_payload_api = None
        if tools:
            tools_payload_api = [{"type": "function", "function": t.model_dump()} for t in tools]

        api_kwargs = kwargs.copy()
        if tools_payload_api:
            api_kwargs["tools"] = tools_payload_api
        if tool_choice:
            api_kwargs["tool_choice"] = tool_choice

        if native_search:
            self._apply_native_search(api_kwargs, model_name)
        if stream and self._native_openai and "stream_options" not in api_kwargs:
            # Without this OpenAI sends no usage on a stream, so streamed calls
            # had no token counts and no cost.
            api_kwargs["stream_options"] = {"include_usage": True}

        if _is_reasoning_model(model_name) and "max_tokens" in api_kwargs:
            if "max_completion_tokens" not in api_kwargs:
                api_kwargs["max_completion_tokens"] = api_kwargs.pop("max_tokens")
            else:
                api_kwargs.pop("max_tokens")

        if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "RAW LLM REQUEST (%s): %s",
                self.get_name(),
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

        if self._transport == "httpx":
            # Every request-shaping decision above is transport-independent, so
            # the direct path picks up native search, reasoning-model parameter
            # renaming and tool payloads unchanged.
            return await self._chat_via_httpx(model_name, messages_payload, stream, api_kwargs)

        if self._vendor is not None and self._vendor.extra_params:
            # The OpenAI SDK has no keyword for these; it sends extra_body.
            moved = {k: api_kwargs.pop(k) for k in list(api_kwargs) if k in self._vendor.extra_params}
            if moved:
                api_kwargs["extra_body"] = {**(api_kwargs.get("extra_body") or {}), **moved}

        try:
            resp = await self._client.chat.completions.create(
                model=model_name, messages=messages_payload, stream=stream, **api_kwargs
            )  # type: ignore
            if stream:

                async def stream_wrapper():
                    try:
                        async for chunk in resp:  # type: ignore
                            if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                                logger.debug(
                                    "RAW STREAM CHUNK (%s): %s",
                                    self.get_name(),
                                    chunk.model_dump_json(),
                                )
                            yield chunk.model_dump(exclude_none=True)
                    except OpenAIError as e:
                        # The SDK raises from inside iteration; without this a
                        # raw openai exception reached the caller mid-stream.
                        raise ProviderError(
                            self.get_name(),
                            f"Stream error: {e}",
                            model_name=model_name,
                            status_code=getattr(e, "status_code", None),
                            original_exception=e,
                        ) from e

                return stream_wrapper()
            else:
                response_dict = resp.model_dump(exclude_none=True)  # type: ignore
                if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "RAW LLM RESPONSE (%s): %s",
                        self.get_name(),
                        json.dumps(response_dict, indent=2, default=str),
                    )
                return response_dict

        except OpenAIAPIStatusError as e:
            status = e.status_code
            msg = str(e)
            code = getattr(e, "code", None)
            # Debug, not error: the caller gets the exception, and a rejected
            # parameter is often adapted and retried (param_policy).
            logger.debug("%s status error (%s): %s", self.get_name(), status, msg)
            if self._is_context_overflow(status, msg, code):
                raise ContextLengthError(
                    model_name=model_name,
                    limit=self.get_max_context_length(model_name),
                    actual=0,
                    message=msg,
                ) from e
            # Detect model-not-found errors and provide actionable info.
            # Different providers use different phrasing:
            #   DeepSeek: "Model Not Exist"
            #   OpenAI:   "does not exist" / "model_not_found"
            if code == "model_not_found" or (
                status in (400, 404)
                and any(
                    phrase in msg.lower()
                    for phrase in ("model not exist", "does not exist", "model_not_found", "invalid model",
                                   "not a valid model id")
                )
            ):
                raise ProviderError(
                    self.get_name(),
                    f"Model '{model_name}' not found on provider "
                    f"'{self.get_name()}'. The provider's default model "
                    f"is '{self.default_model}'. Please verify the model "
                    f"name is correct for this provider's API. "
                    f"Original error: {msg}",
                    model_name=model_name,
                    status_code=status,
                    retryable=False,
                    original_exception=e,
                ) from e
            raise ProviderError(
                self.get_name(),
                f"API Error ({status}): {msg}",
                model_name=model_name,
                status_code=status,
                retryable=None if code != "insufficient_quota" else False,
                original_exception=e,
            ) from e
        except OpenAIAPITimeoutError as e:
            logger.error(f"OpenAI timeout: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"Timeout: {e}")
        except OpenAIAPIConnectionError as e:
            logger.error(f"OpenAI connection error: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"Connection error: {e}")
        except OpenAIAPIError as e:
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"API Error: {e}")
        except OpenAIError as e:
            logger.error(f"OpenAI error: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"Error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"Unexpected error: {e}")

    def extract_response_content(self, response: dict[str, Any]) -> str:
        try:
            choices = response.get("choices", [])
            if not choices:
                return ""
            return choices[0].get("message", {}).get("content") or ""
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Failed to extract content: {e}")
            return ""

    def extract_delta_content(self, chunk: dict[str, Any]) -> str:
        try:
            choices = chunk.get("choices", [])
            if not choices:
                return ""
            return choices[0].get("delta", {}).get("content") or ""
        except (KeyError, IndexError, TypeError):
            return ""

    def extract_tool_calls(self, response: dict[str, Any]) -> list[ToolCall]:
        """Extract tool calls from an OpenAI response, normalized to ToolCall objects."""
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
                            id=tc.get("id", ""), name=func.get("name", ""), arguments=args_dict
                        )
                    )
                else:
                    out.append(
                        ToolCall(
                            id=tc.get("id", ""),
                            name=tc.get("custom", {}).get("name", tc_type),
                            arguments=tc,
                        )
                    )
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Failed to extract tool calls: {e}")
        return out

    def extract_audio_response(self, response: dict[str, Any]) -> dict[str, Any] | None:
        """Extract audio data from response (when modalities includes audio)."""
        try:
            choices = response.get("choices", [])
            if not choices:
                return None
            return choices[0].get("message", {}).get("audio")
        except (KeyError, IndexError, TypeError):
            return None

    def extract_annotations(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract web search annotations."""
        try:
            choices = response.get("choices", [])
            if not choices:
                return []
            return choices[0].get("message", {}).get("annotations") or []
        except (KeyError, IndexError, TypeError):
            return []

    def extract_refusal(self, response: dict[str, Any]) -> str | None:
        """Extract safety refusal message."""
        try:
            choices = response.get("choices", [])
            if not choices:
                return None
            return choices[0].get("message", {}).get("refusal")
        except (KeyError, IndexError, TypeError):
            return None

    def extract_usage_details(self, response: dict[str, Any]) -> dict[str, Any]:
        """Extract extended usage details including reasoning/audio/cached tokens."""
        usage = response.get("usage", {})
        if not usage:
            return {}
        result: dict[str, Any] = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
        cd = usage.get("completion_tokens_details", {})
        if cd:
            result["reasoning_tokens"] = cd.get("reasoning_tokens")
            result["audio_output_tokens"] = cd.get("audio_tokens")
            result["accepted_prediction_tokens"] = cd.get("accepted_prediction_tokens")
            result["rejected_prediction_tokens"] = cd.get("rejected_prediction_tokens")
        pd = usage.get("prompt_tokens_details", {})
        if pd:
            result["cached_tokens"] = pd.get("cached_tokens")
            result["audio_input_tokens"] = pd.get("audio_tokens")
            if pd.get("cache_write_tokens") is not None:
                result["cache_write_tokens"] = pd.get("cache_write_tokens")
        elif usage.get("cached_tokens") is not None:
            # Some OpenAI-compatible servers (Together) report it top-level.
            result["cached_tokens"] = usage.get("cached_tokens")
        # Vendor-reported cost (OpenRouter: always; others: some).
        for key in ("cost", "cost_details", "is_byok", "estimated_cost"):
            if usage.get(key) is not None:
                result[key] = usage[key]
        return result

    def extract_reasoning_content(self, response: dict[str, Any]) -> str | None:
        """The model's reasoning text, under whichever key the server uses.

        Responses-routed calls carry the summary as ``reasoning``; OpenAI-
        compatible servers use ``reasoning`` (Groq, Together, gpt-oss) or
        ``reasoning_content`` (xAI, DeepSeek-style servers).
        """
        try:
            message = (response.get("choices") or [{}])[0].get("message") or {}
        except (AttributeError, IndexError, TypeError):
            return None
        return message.get("reasoning") or message.get("reasoning_content") or None

    def extract_delta_reasoning_content(self, chunk: dict[str, Any]) -> str | None:
        """Streaming counterpart of :meth:`extract_reasoning_content`."""
        try:
            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
        except (AttributeError, IndexError, TypeError):
            return None
        return delta.get("reasoning") or delta.get("reasoning_content") or None

    def extract_reasoning_items(self, response: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Encrypted reasoning items from a Responses-routed answer.

        Put them in the next assistant message's ``metadata["reasoning_items"]``
        to carry the model's chain of thought across a tool loop.
        """
        try:
            choice = (response.get("choices") or [{}])[0]
            # A stream carries them on the final chunk's delta.
            message = choice.get("message") or choice.get("delta") or {}
        except (AttributeError, IndexError, TypeError):
            return None
        items = message.get("reasoning_items")
        return list(items) if items else None

    def assistant_replay_metadata(self, response: dict[str, Any]) -> dict[str, Any]:
        """The reasoning to send back with the next assistant turn.

        Responses route (``store: false``): the encrypted reasoning items.
        OpenAI-compatible vendors on Chat Completions: ``reasoning_details``
        (OpenRouter; replayed unmodified, in order) and/or the reasoning text
        (``reasoning_content`` / ``reasoning``). For a stream, pass the chunk
        with the finish reason.
        """
        items = self.extract_reasoning_items(response)
        if items:
            return {"reasoning_items": items}
        if self._is_openai_itself():
            return {}
        try:
            choice = (response.get("choices") or [{}])[0]
        except (AttributeError, IndexError, TypeError):
            return {}
        message = choice.get("message") or {}
        delta = choice.get("delta") or {}
        out: dict[str, Any] = {}
        details = message.get("reasoning_details") or delta.get("replay_reasoning_details")
        if details:
            out["reasoning_details"] = list(details)
        text = (
            message.get("reasoning_content")
            or message.get("reasoning")
            or delta.get("replay_reasoning_content")
        )
        if isinstance(text, str) and text:
            out["reasoning_content"] = text
        return out

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        if not text:
            return 0
        model_name = model or self.default_model
        return await asyncio.to_thread(lambda: _count_tokens(text, model_name))

    async def count_message_tokens(self, messages: list[Message], model: str | None = None) -> int:
        model_name = model or self.default_model
        counter = _get_token_counter(model_name)
        tpm = 4 if "gpt-3.5-turbo-0301" in model_name else 3
        n = 0
        for m in messages:
            n += tpm
            try:
                rv = m.role.value if hasattr(m.role, "value") else str(m.role)
                n += counter.count(rv)
                n += counter.count(m.content)
                if m.role == LLMCoreRole.TOOL:
                    n += 5
            except Exception:
                rv = m.role.value if hasattr(m.role, "value") else str(m.role)
                n += (len(rv) + len(m.content)) // 4
        return n + 3

    async def close(self) -> None:
        if self._http is not None:
            try:
                await self._http.aclose()
            except Exception as e:  # closing must not mask the caller's work
                logger.warning("Error closing direct HTTP client: %s", e)
            finally:
                self._http = None
        if self._client:
            try:
                await self._client.close()
                logger.info("OpenAIProvider closed.")
            except Exception as e:
                logger.error(f"Error closing: {e}", exc_info=True)
            finally:
                self._client = None

    # ------------------------------------------------------------------
    # Multimodal: Text-to-Speech (TTS)
    # ------------------------------------------------------------------

    async def generate_speech(
        self,
        text: str,
        *,
        voice: str = "alloy",
        model: str | None = None,
        response_format: str = "mp3",
        speed: float = 1.0,
        instructions: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate speech audio from text using OpenAI TTS.

        Models: ``tts-1``, ``tts-1-hd``, ``gpt-4o-mini-tts``.
        Voices: alloy, ash, ballad, coral, echo, fable, onyx, nova,
                sage, shimmer, verse, marin, cedar.

        Args:
            text: Text to convert (max 4096 characters).
            voice: Voice identifier.
            model: TTS model.  Defaults to ``tts-1``.
            response_format: mp3, opus, aac, flac, wav, pcm.
            speed: 0.25–4.0 (default 1.0).
            instructions: Voice control instructions
                (gpt-4o-mini-tts only).
            **kwargs: Extra API parameters.

        Returns:
            ``SpeechResult`` with raw audio bytes.
        """
        from ..models_multimodal import SpeechResult

        if not self._client:
            raise ProviderError(self.get_name(), "Client not initialized.")

        tts_model = model or "tts-1"

        api_kwargs: dict[str, Any] = {
            "input": text,
            "model": tts_model,
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
        }
        if instructions is not None:
            api_kwargs["instructions"] = instructions
        api_kwargs.update(kwargs)

        logger.debug(
            "TTS request: model=%s, voice=%s, format=%s, len=%d",
            tts_model,
            voice,
            response_format,
            len(text),
        )

        try:
            response = await self._client.audio.speech.create(**api_kwargs)
            audio_bytes = response.content

            return SpeechResult(
                audio_data=audio_bytes,
                format=response_format,
                model=tts_model,
                voice=voice,
                metadata={},
            )

        except OpenAIAPIStatusError as e:
            raise ProviderError(self.get_name(), f"TTS Error ({e.status_code}): {e}")
        except OpenAIAPITimeoutError as e:
            raise ProviderError(self.get_name(), f"TTS timeout: {e}")
        except OpenAIAPIConnectionError as e:
            raise ProviderError(self.get_name(), f"TTS connection error: {e}")
        except OpenAIError as e:
            raise ProviderError(self.get_name(), f"TTS error: {e}")
        except Exception as e:
            logger.error(f"Unexpected TTS error: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"TTS unexpected error: {e}")

    # ------------------------------------------------------------------
    # Multimodal: Speech-to-Text (STT)
    # ------------------------------------------------------------------

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
        filename: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Transcribe audio to text using OpenAI Whisper / GPT-4o-transcribe.

        Models: ``whisper-1``, ``gpt-4o-transcribe``,
                ``gpt-4o-mini-transcribe``, ``gpt-4o-transcribe-diarize``.

        Args:
            audio_data: Raw audio bytes or file path string.
            model: STT model.  Defaults to ``whisper-1``.
            language: Input language hint (ISO-639-1).
            prompt: Optional transcription style guide.
            response_format: json, text, srt, verbose_json, vtt.
            temperature: Sampling temperature.
            timestamp_granularities: ``["word"]`` and/or ``["segment"]``.
                Whisper models default to both (with ``verbose_json``) when
                neither is given, so segments carry word timings.
            **kwargs: Extra API parameters (e.g. ``chunking_strategy``).

        Returns:
            ``TranscriptionResult`` with transcribed text and segments (with
            their words when word timestamps were returned).
        """
        from ..models_multimodal import (
            TranscriptionResult,
            TranscriptionSegment,
            TranscriptionWord,
            attach_words_to_segments,
        )

        if not self._client:
            raise ProviderError(self.get_name(), "Client not initialized.")

        stt_model = model or (self._vendor.default_stt_model if self._vendor else None) or "whisper-1"

        # Build file parameter
        if isinstance(audio_data, str):
            # File path
            import pathlib

            file_path = pathlib.Path(audio_data)
            file_obj: Any = open(file_path, "rb")
            file_name = file_path.name
        else:
            # Raw bytes — wrap in a tuple for the SDK. The *name* is what tells
            # OpenAI the container format: sending mp3 bytes labelled
            # "audio.wav" is rejected with
            # "This model does not support the format you provided".
            import io
            import mimetypes

            name = filename or "audio.wav"
            mime = mimetypes.guess_type(name)[0] or "audio/wav"
            file_obj = (name, io.BytesIO(audio_data), mime)
            file_name = name

        api_kwargs: dict[str, Any] = {
            "file": file_obj,
            "model": stt_model,
        }
        if language is not None:
            api_kwargs["language"] = language
        if prompt is not None:
            api_kwargs["prompt"] = prompt
        # Whisper models return word and segment timings at no extra cost when
        # asked for verbose_json, so a plain request gets them by default; the
        # gpt-4o-*-transcribe models accept only json (diarize: diarized_json).
        if (
            response_format == "json"
            and timestamp_granularities is None
            and "whisper" in stt_model.lower()
        ):
            timestamp_granularities = ["word", "segment"]
        if timestamp_granularities and response_format == "json":
            # verbose_json needed for timestamps
            response_format = "verbose_json"
        if response_format != "json":
            api_kwargs["response_format"] = response_format
        if temperature is not None:
            api_kwargs["temperature"] = temperature
        if timestamp_granularities:
            api_kwargs["timestamp_granularities"] = timestamp_granularities
        api_kwargs.update(kwargs)

        logger.debug(
            "STT request: model=%s, language=%s, format=%s",
            stt_model,
            language,
            response_format,
        )

        try:
            response = await self._client.audio.transcriptions.create(**api_kwargs)

            # Close file handle if we opened one
            if isinstance(audio_data, str) and hasattr(file_obj, "close"):
                file_obj.close()

            # Parse response — structure depends on format
            if hasattr(response, "text"):
                text = response.text
            elif isinstance(response, str):
                text = response
            else:
                text = str(response)

            def _field(obj: Any, key: str, default: Any = None) -> Any:
                if isinstance(obj, dict):
                    return obj.get(key, default)
                return getattr(obj, key, default)

            # Extract segments if available (verbose_json / diarized_json).
            segments: list[TranscriptionSegment] = []
            for seg in getattr(response, "segments", None) or []:
                speaker = _field(seg, "speaker")
                segments.append(
                    TranscriptionSegment(
                        text=_field(seg, "text", "") or "",
                        start=float(_field(seg, "start", 0.0) or 0.0),
                        end=float(_field(seg, "end", 0.0) or 0.0),
                        speaker=str(speaker) if speaker is not None else None,
                    )
                )

            # Word timings (verbose_json with the "word" granularity) come as a
            # separate flat list; place each word in its segment.
            words = [
                TranscriptionWord(
                    word=_field(w, "word", "") or "",
                    start=float(_field(w, "start", 0.0) or 0.0),
                    end=float(_field(w, "end", 0.0) or 0.0),
                )
                for w in getattr(response, "words", None) or []
            ]
            segments = attach_words_to_segments(segments, words, text=text)

            duration = None
            if hasattr(response, "duration"):
                duration = response.duration

            # verbose_json reports a full language name ("portuguese"), not an
            # ISO-639-1 code. A caller's hint forces the decoding language, so
            # it wins; otherwise the provider's value is kept as given.
            detected_lang = getattr(response, "language", None)

            return TranscriptionResult(
                text=text,
                language=language or detected_lang,
                duration_seconds=duration,
                segments=segments,
                model=stt_model,
                metadata={},
            )

        except OpenAIAPIStatusError as e:
            raise ProviderError(self.get_name(), f"STT Error ({e.status_code}): {e}")
        except OpenAIAPITimeoutError as e:
            raise ProviderError(self.get_name(), f"STT timeout: {e}")
        except OpenAIAPIConnectionError as e:
            raise ProviderError(self.get_name(), f"STT connection error: {e}")
        except OpenAIError as e:
            raise ProviderError(self.get_name(), f"STT error: {e}")
        except ProviderError:
            raise
        except Exception as e:
            logger.error(f"Unexpected STT error: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"STT unexpected error: {e}")

    # ------------------------------------------------------------------
    # Multimodal: Image Generation
    # ------------------------------------------------------------------


    # ==================================================================
    # Media subsystem adapter (llmcore.media protocols)
    # ==================================================================
    #
    # Phase M3 of the media subsystem design spec. The image/speech methods above
    # are unchanged; these adapters translate MediaRef in and MediaArtifact out
    # and delegate, so there is one code path per operation.
    #
    # NOTE FOR SUBCLASSES: DeepInfra, vLLM, Poe and OpenRouter all extend this
    # class and therefore INHERIT these methods — but not the endpoints behind
    # them. vLLM has no image generation; OpenRouter is a chat gateway. Each
    # subclass must set ``_MEDIA_CAPABILITIES`` to what IT can actually serve,
    # and a test asserts every subclass does so explicitly. Inheriting the
    # declaration would make the router confidently call an endpoint that
    # returns 404.

    #: Media capabilities this provider class serves. Subclasses MUST override.
    #: Deliberately NOT inherited meaningfully: see the note above.
    #: Sora video is excluded — openai 3.1 deprecated the video APIs.
    _MEDIA_CAPABILITIES: frozenset[Any] = frozenset(
        {
            "image_generate",
            "image_edit",
            "tts",
            "tts_stream",
            "asr",
        }
    )

    def media_capabilities(self) -> "frozenset[MediaCapability]":
        """Capabilities this provider serves, declared per class."""
        from ..media.models import MediaCapability

        return frozenset(MediaCapability(c) for c in self._MEDIA_CAPABILITIES)

    def media_execution(
        self, capability: "MediaCapability", model: str | None = None
    ) -> "MediaExecution":
        """Return how *capability* completes.

        Every OpenAI media surface is synchronous; only TTS has a stream.
        """
        from ..media.models import MediaCapability, MediaExecution

        if capability is MediaCapability.TTS_STREAM:
            return MediaExecution.STREAM
        return MediaExecution.REQUEST_RESPONSE

    # --- image ---

    async def generate_image_media(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
        size: str | None = None,
        seed: int | None = None,
        negative_prompt: str | None = None,
        reference_images: "Sequence[MediaRef] | None" = None,
        **kwargs: Any,
    ) -> "MediaResult":
        """Generate images, returning a normalized media result.

        ``seed`` and ``negative_prompt`` have no OpenAI equivalent and are
        ignored with a debug log rather than silently dropped. Supplying
        ``reference_images`` routes to the *edit* endpoint, which is how
        OpenAI expresses reference-conditioned generation.
        """
        from ..media.models import MediaCapability, MediaResult, MediaUsage

        if reference_images:
            return await self.edit_image_media(
                prompt, image=reference_images[0], model=model, n=n, size=size, **kwargs
            )
        for unsupported, value in (("seed", seed), ("negative_prompt", negative_prompt)):
            if value is not None:
                logger.debug("OpenAI image generation ignores '%s'.", unsupported)

        result = await self.generate_image(prompt, model=model, n=n, size=size, **kwargs)
        return MediaResult(
            capability=MediaCapability.IMAGE_GENERATE,
            provider=self.get_name(),
            model=result.model,
            artifacts=tuple(result.to_artifacts()),
            usage=MediaUsage(
                provider=self.get_name(),
                model=result.model,
                basis="per_image",
                images=len(result.images),
            ),
            raw=dict(result.metadata),
        )

    async def edit_image_media(
        self,
        prompt: str,
        *,
        image: "MediaRef",
        mask: "MediaRef | None" = None,
        model: str | None = None,
        n: int = 1,
        size: str | None = None,
        **kwargs: Any,
    ) -> "MediaResult":
        """Edit an image via ``POST /v1/images/edits``.

        Args:
            prompt: What the edit should produce.
            image: The image to edit. A remote ref is fetched first, because the
                edits endpoint takes an upload rather than a URL.
            mask: Optional transparency mask restricting the edited region.
            model: Image model; defaults to ``gpt-image-1``.
            n: Number of variants.
            size: Output dimensions.
            **kwargs: Extra OpenAI parameters (``quality``, ``background``,
                ``input_fidelity``, ``output_format`` …).

        Returns:
            A :class:`~llmcore.media.MediaResult` with the edited images.

        Raises:
            ProviderError: If the client is missing or the API call fails.
        """
        from ..media.models import MediaCapability, MediaResult, MediaUsage
        from ..models_multimodal import GeneratedImage, ImageGenerationResult

        if not self._client:
            raise ProviderError(self.get_name(), "Client not initialized.")

        img_model = model or (self._vendor.default_image_model if self._vendor else None) or DEFAULT_IMAGE_MODEL
        api_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "model": img_model,
            "n": n,
            "image": await self._media_upload_tuple(image, "image.png"),
        }
        if mask is not None:
            api_kwargs["mask"] = await self._media_upload_tuple(mask, "mask.png")
        if size is not None:
            api_kwargs["size"] = size
        api_kwargs.update(kwargs)

        try:
            response = await self._client.images.edit(**api_kwargs)
        except OpenAIAPIStatusError as e:
            raise ProviderError(self.get_name(), f"Image edit error ({e.status_code}): {e}")
        except OpenAIError as e:
            raise ProviderError(self.get_name(), f"Image edit error: {e}")

        images = [
            GeneratedImage(
                data=getattr(item, "b64_json", None),
                url=getattr(item, "url", None),
                revised_prompt=getattr(item, "revised_prompt", None),
                format=kwargs.get("output_format", "png"),
            )
            for item in response.data
        ]
        result = ImageGenerationResult(
            images=images,
            model=img_model,
            metadata={"created": getattr(response, "created", None)},
        )
        return MediaResult(
            capability=MediaCapability.IMAGE_EDIT,
            provider=self.get_name(),
            model=img_model,
            artifacts=tuple(result.to_artifacts()),
            usage=MediaUsage(
                provider=self.get_name(),
                model=img_model,
                basis="per_image",
                images=len(images),
            ),
            raw=dict(result.metadata),
        )

    async def _media_upload_tuple(self, ref: "MediaRef", fallback_name: str) -> Any:
        """Turn a :class:`MediaRef` into an OpenAI file-upload tuple.

        The images endpoints take an upload, not a URL, so a remote ref is
        fetched here rather than handed through.
        """
        if ref.is_remote:
            from ..media.artifacts import default_fetcher

            data = await default_fetcher()(ref.url or "")
        else:
            data = ref.read_bytes()
        return (ref.filename or fallback_name, data, ref.mime_type or "image/png")

    # --- audio ---

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
    ) -> "MediaResult":
        """Synthesize speech, returning a normalized media result."""
        from ..media.models import MediaCapability, MediaResult, MediaUsage

        if sample_rate_hz is not None:
            logger.debug("OpenAI TTS has no sample-rate parameter; ignoring it.")
        speech_kwargs: dict[str, Any] = {}
        if voice is not None:
            speech_kwargs["voice"] = voice
        if audio_format is not None:
            speech_kwargs["response_format"] = audio_format
        if speed is not None:
            speech_kwargs["speed"] = speed

        speech = await self.generate_speech(text, model=model, **speech_kwargs, **kwargs)
        return MediaResult(
            capability=MediaCapability.TTS,
            provider=self.get_name(),
            model=speech.model,
            artifacts=(speech.to_artifact(),),
            usage=MediaUsage(
                provider=self.get_name(),
                model=speech.model,
                basis="per_character",
                characters=len(text),
            ),
            raw=dict(speech.metadata),
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
    ) -> "AsyncIterator[bytes]":
        """Stream synthesized speech as it is produced.

        Uses the SDK's streaming response so the first bytes arrive before
        synthesis completes. Returns the iterator directly, not a coroutine.
        """
        client = self._client
        provider_name = self.get_name()

        async def _stream() -> "AsyncIterator[bytes]":
            if not client:
                raise ProviderError(provider_name, "Client not initialized.")
            api_kwargs: dict[str, Any] = {
                "model": model or "gpt-4o-mini-tts",
                "voice": voice or "alloy",
                "input": text,
                "response_format": audio_format or "mp3",
                **kwargs,
            }
            try:
                async with client.audio.speech.with_streaming_response.create(
                    **api_kwargs
                ) as response:
                    async for chunk in response.iter_bytes():
                        yield chunk
            except OpenAIAPIStatusError as e:
                raise ProviderError(provider_name, f"TTS stream error ({e.status_code}): {e}")
            except OpenAIError as e:
                raise ProviderError(provider_name, f"TTS stream error: {e}")

        return _stream()

    async def transcribe_media(
        self,
        *,
        audio: "MediaRef",
        model: str | None = None,
        language: str | None = None,
        diarize: bool | None = None,
        timestamps: bool | None = None,
        **kwargs: Any,
    ) -> "MediaResult":
        """Transcribe audio, returning a normalized media result.

        ``timestamps`` maps onto ``timestamp_granularities=["segment"]``.
        ``diarize`` is only meaningful on the diarizing models, so it is passed
        through rather than emulated.
        """
        from ..media.models import MediaCapability, MediaResult, MediaUsage

        if audio.is_remote:
            from ..media.artifacts import default_fetcher

            payload: bytes = await default_fetcher()(audio.url or "")
        else:
            payload = audio.read_bytes()

        extra: dict[str, Any] = dict(kwargs)
        if timestamps:
            extra.setdefault("timestamp_granularities", ["segment"])
        if diarize is not None:
            extra.setdefault("diarize", diarize)

        transcript = await self.transcribe_audio(
            payload,
            model=model,
            language=language,
            filename=_audio_filename_for(audio),
            **extra,
        )
        return MediaResult(
            capability=MediaCapability.ASR,
            provider=self.get_name(),
            model=transcript.model,
            artifacts=(transcript.to_artifact(),),
            usage=MediaUsage(
                provider=self.get_name(),
                model=transcript.model,
                basis="per_audio_minute",
                audio_minutes=(transcript.duration_seconds / 60.0)
                if transcript.duration_seconds
                else None,
                seconds=transcript.duration_seconds,
            ),
            raw=dict(transcript.metadata),
        )

    # --- embeddings ---

    async def create_embeddings(
        self,
        input_texts: str | list[str],
        *,
        model: str | None = None,
        dimensions: int | None = None,
        encoding_format: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create text embeddings via ``POST /v1/embeddings``.

        Closes a long-standing gap: OpenAI embeddings were reachable only
        through the separate ``[embedding.openai]`` subsystem, so callers
        holding a provider could not embed with it.

        Args:
            input_texts: A string or list of strings to embed.
            model: Embedding model; defaults to ``text-embedding-3-small``.
            dimensions: Output dimensionality, where the model supports it.
            encoding_format: ``"float"`` or ``"base64"``.
            **kwargs: Extra API parameters.

        Returns:
            The raw API response dict with ``data``, ``model`` and ``usage``.

        Raises:
            ProviderError: If the client is missing or the API call fails.
        """
        if not self._client:
            raise ProviderError(self.get_name(), "Client not initialized.")

        api_kwargs: dict[str, Any] = {
            "model": model or "text-embedding-3-small",
            "input": input_texts,
            **kwargs,
        }
        if dimensions is not None:
            api_kwargs["dimensions"] = dimensions
        if encoding_format is not None:
            api_kwargs["encoding_format"] = encoding_format

        try:
            response = await self._client.embeddings.create(**api_kwargs)
        except OpenAIAPIStatusError as e:
            raise ProviderError(self.get_name(), f"Embeddings error ({e.status_code}): {e}")
        except OpenAIError as e:
            raise ProviderError(self.get_name(), f"Embeddings error: {e}")
        return response.model_dump(exclude_none=True)

    async def generate_image(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
        size: str | None = None,
        quality: str | None = None,
        response_format: str = "b64_json",
        style: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate images from a text prompt using OpenAI.

        Models: ``dall-e-2``, ``dall-e-3``, ``gpt-image-1``,
                ``gpt-image-1-mini``, ``gpt-image-1.5``.

        Args:
            prompt: Text description of desired image(s).
            model: Image model.  Defaults to ``gpt-image-1``.
            n: Number of images (1–10; dall-e-3 only supports 1).
            size: Dimensions (e.g. ``"1024x1024"``).
            quality: Quality level (standard, hd, low, medium, high, auto).
            response_format: ``"url"`` or ``"b64_json"``.
            style: ``"vivid"`` or ``"natural"`` (dall-e-3).
            **kwargs: Extra params (background, output_format, etc.).

        Returns:
            ``ImageGenerationResult`` with generated images.
        """
        from ..models_multimodal import GeneratedImage, ImageGenerationResult

        if not self._client:
            raise ProviderError(self.get_name(), "Client not initialized.")

        img_model = model or (self._vendor.default_image_model if self._vendor else None) or DEFAULT_IMAGE_MODEL

        api_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "model": img_model,
            "n": n,
        }
        if img_model.startswith("dall-e"):
            # GPT image models always return base64 and reject the field
            # (live 2026-11-16: "Unknown parameter: 'response_format'").
            api_kwargs["response_format"] = response_format
        if size is not None:
            api_kwargs["size"] = size
        if quality is not None:
            api_kwargs["quality"] = quality
        if style is not None:
            api_kwargs["style"] = style
        api_kwargs.update(kwargs)

        logger.debug(
            "Image gen: model=%s, n=%d, size=%s, quality=%s",
            img_model,
            n,
            size,
            quality,
        )

        try:
            response = await self._client.images.generate(**api_kwargs)

            images: list[GeneratedImage] = []
            for img_data in response.data:
                images.append(
                    GeneratedImage(
                        data=getattr(img_data, "b64_json", None),
                        url=getattr(img_data, "url", None),
                        revised_prompt=getattr(img_data, "revised_prompt", None),
                        format=kwargs.get("output_format", "png"),
                    )
                )

            return ImageGenerationResult(
                images=images,
                model=img_model,
                metadata={
                    "created": getattr(response, "created", None),
                },
            )

        except OpenAIAPIStatusError as e:
            raise ProviderError(self.get_name(), f"Image Error ({e.status_code}): {e}")
        except OpenAIAPITimeoutError as e:
            raise ProviderError(self.get_name(), f"Image timeout: {e}")
        except OpenAIAPIConnectionError as e:
            raise ProviderError(self.get_name(), f"Image connection error: {e}")
        except OpenAIError as e:
            raise ProviderError(self.get_name(), f"Image error: {e}")
        except ProviderError:
            raise
        except Exception as e:
            logger.error(f"Unexpected image error: {e}", exc_info=True)
            raise ProviderError(self.get_name(), f"Image unexpected error: {e}")


def _audio_filename_for(ref: "MediaRef") -> str:
    """Pick an upload filename whose extension names the real audio format.

    OpenAI infers the container from the filename, so an artifact chained in
    from TTS (mp3 bytes) must not be uploaded as ``audio.wav``.
    """
    import mimetypes

    if ref.filename:
        return ref.filename
    if ref.mime_type:
        ext = mimetypes.guess_extension(ref.mime_type)
        if ext:
            return f"audio{ext}"
    if ref.url:
        from pathlib import Path as _Path

        suffix = _Path(ref.url.split("?", 1)[0]).suffix
        if suffix:
            return f"audio{suffix}"
    return "audio.wav"
