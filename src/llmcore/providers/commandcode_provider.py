# src/llmcore/providers/commandcode_provider.py
"""
LLMCore provider for Command Code's Provider API, one key for 98+ models.

Command Code (https://commandcode.ai) is a coding-agent CLI and a paid model
gateway at ``https://api.commandcode.ai/provider/v1``. The gateway serves
Claude, GPT, Gemini, Grok, DeepSeek, Kimi, GLM, Qwen, MiniMax and more behind
one key. It does not speak one protocol. It speaks three, plus one bespoke
endpoint, and each model answers on only some of them:

=========================  ===========================  =======================
route                      wire shape                   served models
=========================  ===========================  =======================
``POST /messages``         Anthropic Messages           Claude only
``POST /chat/completions`` OpenAI Chat Completions      everything but Claude
``POST /responses``        OpenAI Responses             most non-Claude models
``POST /systemone``        TypeSafe System One          ``typesafe/jev`` only
``GET  /models``           OpenAI list + routes         public, no key needed
=========================  ===========================  =======================

A model sent to the wrong route gets ``400 unsupported_model``. So this
provider routes **per model**: it reads ``supported_endpoints`` from the live
catalog and hands each call to a *lane*. A lane is a small subclass of an
existing llmcore wire implementation:

- ``messages``: :class:`AnthropicProvider` pointed at the gateway. Claude's
  per-generation thinking rules, effort mapping, prompt caching, vision and
  tools are inherited rather than re-derived.
- ``chat``: :class:`OpenAIProvider` pointed at the gateway, plus the gateway's
  quirks (see :class:`_ChatLane`).
- ``responses``: OpenAI's Responses API (:mod:`.openai_responses`), used for
  GPT models by default (``responses_api = "openai"``, as Command Code's own
  pi provider does), for every model that offers it with ``"all"``, or never.
  Results are normalized to the Chat Completions shape like the other lanes.
- ``system_one``: :class:`TypeSafeProvider` pointed at the gateway, exposed as
  :meth:`CommandCodeProvider.system_one`.

Lanes are built lazily and are never registered as provider instances of their
own. They report the outer instance name, so spans, usage records, cost
estimates and routing events all say ``commandcode``. They never say
``anthropic`` or ``openai``. Command Code prices models itself, so cost
estimates must come from ``commandcode`` cards.

Gateway behaviour verified live (2026-11-15) that the docs do not state:

- An upstream failure can arrive as **HTTP 200** with
  ``choices[0].finish_reason == "error"``, a ``choices[0].error`` object and
  *partial* content. A plain OpenAI-compatible wrapper would return that
  fragment as the answer. This provider raises instead.
- Some 400s carry an ``error.message`` that is itself JSON: the upstream
  error plus the gateway's routing attempts. It is unwrapped here.
- Every response carries ``x-trace-id``. No rate-limit headers are sent.
- Chat Completions responses carry the upstream's name (``provider``) and
  OpenRouter-style ``reasoning`` / ``reasoning_details``.

Transport: Command Code publishes no SDK of its own. Its docs use the OpenAI
and Anthropic SDKs as clients. So, following llmcore's rule, the default is
direct HTTP (``transport = "httpx"``), with the official SDKs selectable through
``transport = "sdk"``.

Configuration (``[providers.commandcode]``)::

    [providers.commandcode]
    type = "commandcode"
    default_model = "claude-sonnet-5-5"
    # api_key / api_key_env_var. Default lookup: CMD_API_KEY,
    # COMMAND_CODE_API_KEY, COMMANDCODE_API_KEY.
    transport = "httpx"     # or "sdk"
    zdr = false             # send x-cmd-zdr: 1 on every request (env CMD_ZDR=1)
    responses_api = "openai"  # GPT -> /responses; "all" | "never"
    catalog_ttl = 3600      # seconds the live /models catalog is trusted
    timeout = 120
    max_retries = 2
    # base_url = "https://api.commandcode.ai/provider"   (env CMD_BASE_URL)
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import re
import time
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, field
from typing import Any

try:
    import httpx

    httpx_available = True
except ImportError:  # pragma: no cover - httpx is a dependency of the extra
    httpx = None  # type: ignore[assignment]
    httpx_available = False

try:
    import tiktoken

    tiktoken_available = True
except ImportError:  # pragma: no cover
    tiktoken = None  # type: ignore[assignment]
    tiktoken_available = False

from ..exceptions import ConfigError, ContextLengthError, ProviderError
from ..models import Message, ModelDetails, Tool, ToolCall
from . import openai_responses as rsp
from .base import BaseProvider, ContextPayload

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROVIDER_NAME = "commandcode"

#: Gateway root. The Anthropic wire appends ``/v1/messages`` itself, and the
#: OpenAI wire uses ``<root>/v1``. This is the same split the official pi
#: provider makes.
COMMANDCODE_BASE_URL = "https://api.commandcode.ai/provider"

DEFAULT_MODEL = "claude-sonnet-5-5"

#: The decision model served on ``/systemone``.
SYSTEM_ONE_MODEL = "typesafe/jev"

#: Key spellings, in lookup order. The first two are the ones Command Code's
#: own CLI and pi provider read.
API_KEY_ENV_VARS: tuple[str, ...] = ("CMD_API_KEY", "COMMAND_CODE_API_KEY", "COMMANDCODE_API_KEY")
ENV_ZDR = "CMD_ZDR"
ENV_BASE_URL = "CMD_BASE_URL"

ZDR_HEADER = "x-cmd-zdr"
TRACE_HEADER = "x-trace-id"

ROUTE_MESSAGES = "messages"
ROUTE_CHAT = "chat"
ROUTE_RESPONSES = "responses"
ROUTE_SYSTEM_ONE = "system_one"

#: Which models use ``/responses`` when they also answer on Chat Completions.
#: ``openai`` (default) is what Command Code's official pi provider does: GPT
#: models get OpenAI's native API, everything else stays on Chat Completions.
RESPONSES_POLICIES: tuple[str, ...] = ("openai", "all", "never")
DEFAULT_RESPONSES_POLICY = "openai"

_OPENAI_FAMILY = re.compile(r"^(gpt-|openai/|o\d)", re.IGNORECASE)

_DEFAULT_CATALOG_TTL = 3600.0
#: How long a failed catalog fetch is remembered before trying again, so an
#: outage does not add a timeout to every request.
_CATALOG_RETRY_AFTER = 60.0
_CATALOG_TIMEOUT = 15.0

#: Request body ceiling stated by the docs (413 ``request_too_large``).
MAX_REQUEST_BYTES = 33_554_432

#: Fallback windows when neither the catalog nor a card knows the model. These
#: are the same figures the official pi provider falls back to.
_FALLBACK_CONTEXT = 131_072
_FALLBACK_CLAUDE_CONTEXT = 200_000

#: ``error.metadata.error_type`` values on in-band failures that are worth
#: retrying.
_RETRYABLE_INBAND_TYPES = frozenset(
    {"provider_unavailable", "upstream_unavailable", "timeout", "rate_limit", "server_error"}
)

_ROUTE_HINT = re.compile(r"/provider/v1/(messages|chat/completions|responses)")

#: Details of the most recent wire call made by a lane in this task: trace id
#: and the upstream's name. A ContextVar, not instance state, so concurrent
#: requests on one provider never see each other's ids.
_LAST_CALL: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "commandcode_last_call", default=None
)


def _note_call(headers: Any, **extra: Any) -> None:
    """Record the trace id (and anything else) of the call just made."""
    info = dict(_LAST_CALL.get() or {})
    trace = None
    try:
        trace = headers.get(TRACE_HEADER) if headers is not None else None
    except Exception:  # pragma: no cover - defensive against odd header types
        trace = None
    if trace:
        info["trace_id"] = trace
    info.update({k: v for k, v in extra.items() if v is not None})
    _LAST_CALL.set(info)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommandCodeModel:
    """One entry of Command Code's live catalog (``GET /models``)."""

    id: str
    name: str | None = None
    context_length: int | None = None
    endpoints: tuple[str, ...] = ()
    created: int | None = None

    @property
    def route(self) -> str | None:
        """The lane that serves this model under the default policy."""
        return route_for_endpoints(self.id, self.endpoints)

    @classmethod
    def from_wire(cls, row: Mapping[str, Any]) -> CommandCodeModel | None:
        model_id = row.get("id")
        if not model_id:
            return None
        ctx = row.get("context_length")
        return cls(
            id=str(model_id),
            name=str(row["name"]) if row.get("name") else None,
            context_length=int(ctx) if isinstance(ctx, int) and ctx > 0 else None,
            endpoints=tuple(str(e) for e in (row.get("supported_endpoints") or [])),
            created=row.get("created") if isinstance(row.get("created"), int) else None,
        )


def route_for_endpoints(
    model_id: str,
    endpoints: tuple[str, ...] | list[str],
    policy: str = DEFAULT_RESPONSES_POLICY,
) -> str | None:
    """Pick the lane for a model from its ``supported_endpoints``.

    ``/messages`` and ``/systemone`` are exclusive. A model on both
    ``/responses`` and ``/chat/completions`` goes to Responses when ``policy``
    says so (``openai``: GPT models only; ``all``; ``never``). A model listed
    only on one of the two goes there whatever the policy. ``None`` means no
    endpoint llmcore can drive.
    """
    eps = set(endpoints)
    if "/messages" in eps:
        return ROUTE_MESSAGES
    if "/systemone" in eps:
        return ROUTE_SYSTEM_ONE
    has_responses = "/responses" in eps
    has_chat = "/chat/completions" in eps
    if has_responses and has_chat:
        if policy == "all" or (policy == "openai" and _OPENAI_FAMILY.search(model_id)):
            return ROUTE_RESPONSES
        return ROUTE_CHAT
    if has_chat:
        return ROUTE_CHAT
    if has_responses:
        return ROUTE_RESPONSES
    if not eps and model_id.startswith("typesafe/"):
        return ROUTE_SYSTEM_ONE
    return None


def guess_route(model_id: str) -> str:
    """Route a model nothing else knows about, by family name.

    Used only when the catalog cannot be read and no card covers the model.
    Claude answers only on ``/messages``, Jev only on ``/systemone``, and
    everything else answers on Chat Completions.
    """
    lowered = model_id.lower()
    if lowered.startswith("claude-") or lowered.startswith("anthropic/"):
        return ROUTE_MESSAGES
    if lowered.startswith("typesafe/") or lowered.startswith("jev"):
        return ROUTE_SYSTEM_ONE
    return ROUTE_CHAT


@dataclass
class CommandCodeCatalog:
    """Live, cached view of ``GET /models``, plus model-name resolution.

    Names resolve the way Command Code's CLI resolves ``--model``. An exact id
    wins, then a case-insensitive id, then the part after ``/``. So ``kimi-k3``
    is ``moonshotai/Kimi-K3``. A short name that matches several models raises
    rather than picking one.
    """

    base_url: str
    ttl: float = _DEFAULT_CATALOG_TTL
    timeout: float = _CATALOG_TIMEOUT
    headers: dict[str, str] = field(default_factory=dict)
    transport: Any = None  # httpx transport override, for tests

    _models: dict[str, CommandCodeModel] = field(default_factory=dict, init=False)
    _fetched_at: float | None = field(default=None, init=False)
    _failed_at: float | None = field(default=None, init=False)
    _last_error: str | None = field(default=None, init=False)
    _lock: asyncio.Lock | None = field(default=None, init=False)

    # -- loading ---------------------------------------------------------

    @property
    def models_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/models"

    @property
    def loaded(self) -> bool:
        return self._fetched_at is not None

    @property
    def fresh(self) -> bool:
        return self._fetched_at is not None and (time.monotonic() - self._fetched_at) < self.ttl

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def load_rows(self, rows: list[Any]) -> None:
        """Replace the catalog with ``rows`` (the ``data`` list of ``/models``)."""
        models: dict[str, CommandCodeModel] = {}
        for row in rows:
            if isinstance(row, Mapping):
                entry = CommandCodeModel.from_wire(row)
                if entry is not None:
                    models[entry.id] = entry
        self._models = models
        self._fetched_at = time.monotonic()
        self._failed_at = None
        self._last_error = None

    async def refresh(self) -> None:
        """Fetch the catalog now.

        Raises:
            ProviderError: If the catalog cannot be read or has an unknown shape.
        """
        if not httpx_available:  # pragma: no cover
            raise ProviderError(PROVIDER_NAME, "httpx is required to read the catalog.")
        kwargs: dict[str, Any] = {"timeout": self.timeout, "headers": dict(self.headers)}
        if self.transport is not None:
            kwargs["transport"] = self.transport
        try:
            async with httpx.AsyncClient(**kwargs) as client:
                resp = await client.get(self.models_url)
        except httpx.HTTPError as exc:
            self._mark_failed(f"connection error: {exc}")
            raise ProviderError(
                PROVIDER_NAME, f"Could not read the Command Code catalog: {exc}"
            ) from exc
        if resp.status_code >= 400:
            self._mark_failed(f"HTTP {resp.status_code}")
            raise ProviderError(
                PROVIDER_NAME,
                f"Command Code catalog returned HTTP {resp.status_code}: {resp.text[:300]}",
                status_code=resp.status_code,
            )
        try:
            payload = resp.json()
        except ValueError as exc:
            self._mark_failed("unparseable JSON")
            raise ProviderError(PROVIDER_NAME, "Command Code catalog is not JSON.") from exc
        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            self._mark_failed("no data list")
            raise ProviderError(
                PROVIDER_NAME, "Command Code catalog has no `data` list; refusing to guess."
            )
        self.load_rows(rows)
        logger.debug("Command Code catalog loaded: %d models.", len(self._models))

    def _mark_failed(self, reason: str) -> None:
        self._failed_at = time.monotonic()
        self._last_error = reason

    async def ensure(self) -> bool:
        """Load or refresh the catalog when stale. Never raises.

        Returns:
            Whether a catalog (possibly stale) is available.
        """
        if self.fresh:
            return True
        if self._failed_at is not None and (
            time.monotonic() - self._failed_at
        ) < _CATALOG_RETRY_AFTER:
            return self.loaded
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self.fresh:
                return True
            try:
                await self.refresh()
            except ProviderError as exc:
                level = logging.WARNING if not self.loaded else logging.INFO
                logger.log(
                    level,
                    "Command Code catalog unavailable (%s); routing from %s.",
                    exc,
                    "the stale catalog" if self.loaded else "model cards and family names",
                )
        return self.loaded

    # -- lookup ----------------------------------------------------------

    def __contains__(self, model_id: object) -> bool:
        return model_id in self._models

    def __len__(self) -> int:
        return len(self._models)

    def get(self, model_id: str) -> CommandCodeModel | None:
        return self._models.get(model_id)

    def entries(self) -> list[CommandCodeModel]:
        return list(self._models.values())

    def resolve(self, name: str) -> str:
        """Resolve a friendly name to a catalog id.

        Returns ``name`` unchanged when the catalog does not know it. The
        gateway then answers with its own ``unsupported_model``, which is a
        better message than a guess.

        Raises:
            ConfigError: If a short name matches more than one model.
        """
        if not self._models or name in self._models:
            return name
        lowered = name.lower()
        exact = [mid for mid in self._models if mid.lower() == lowered]
        if len(exact) == 1:
            return exact[0]
        tail = [mid for mid in self._models if mid.split("/", 1)[-1].lower() == lowered]
        if len(tail) == 1:
            return tail[0]
        if len(exact) > 1 or len(tail) > 1:
            candidates = sorted(set(exact) | set(tail))
            raise ConfigError(
                f"Command Code model name '{name}' is ambiguous; it matches "
                f"{', '.join(candidates)}. Use the full id."
            )
        return name


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


def _parse_error_body(body: Any) -> dict[str, Any]:
    """Pull ``message``/``type``/``code``/``param`` out of any gateway error.

    Handles the three shapes the gateway uses: OpenAI's
    ``{"error": {...}}``, Anthropic's ``{"type": "error", "error": {...}}``
    and the router's own ``{"success": false, "message", "cause"}``. It also
    handles the double-encoded case, where ``error.message`` is a JSON string
    carrying the upstream error and the gateway's routing attempts.
    """
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8", "replace")
    if isinstance(body, str):
        text = body.strip()
        try:
            body = json.loads(text) if text else {}
        except ValueError:
            return {"message": text[:500]}
    if not isinstance(body, Mapping):
        return {"message": str(body)[:500]}

    out: dict[str, Any] = {}
    err = body.get("error")
    if err is None and "message" in body and body.get("success") is not False:
        # The OpenAI SDK hands over the error object already unwrapped.
        err = body
    if isinstance(err, Mapping):
        out.update(
            {k: err.get(k) for k in ("message", "type", "code", "param") if err.get(k) is not None}
        )
    elif isinstance(err, str):
        out["message"] = err
    elif body.get("success") is False:
        out["message"] = body.get("message")
        if body.get("cause"):
            out["cause"] = body.get("cause")

    msg = out.get("message")
    if isinstance(msg, str) and msg.lstrip().startswith("{"):
        try:
            inner = json.loads(msg)
        except ValueError:
            inner = None
        if isinstance(inner, Mapping):
            routing = (
                ((inner.get("providerMetadata") or {}).get("gateway") or {}).get("routing")
                if isinstance(inner.get("providerMetadata"), Mapping)
                else None
            )
            inner_parsed = _parse_error_body({"error": inner.get("error")})
            if inner_parsed.get("message"):
                out["message"] = inner_parsed["message"]
            for key in ("type", "code"):
                if key not in out and inner_parsed.get(key):
                    out[key] = inner_parsed[key]
            if isinstance(routing, Mapping):
                out["routing"] = dict(routing)
    return out


def _routing_summary(routing: Mapping[str, Any] | None) -> str | None:
    """One line describing which upstreams the gateway tried."""
    if not routing:
        return None
    attempts: list[str] = []
    for model_attempt in routing.get("modelAttempts") or []:
        for pa in (model_attempt or {}).get("providerAttempts") or []:
            status = pa.get("statusCode")
            attempts.append(f"{pa.get('provider')}={status}" if status else str(pa.get("provider")))
    if not attempts:
        final = routing.get("finalProvider")
        return f"upstream {final}" if final else None
    return "upstream attempts: " + ", ".join(attempts)


def build_commandcode_error(
    provider_name: str,
    status: int,
    body: Any,
    *,
    model: str | None,
    headers: Any = None,
    route: str | None = None,
    original: Exception | None = None,
) -> ProviderError:
    """Map a Command Code HTTP failure onto llmcore's exceptions.

    Every error carries ``failure_kind`` (a :class:`~llmcore.routing.models.FailureKind`
    value) so routing fails over for the right reason. Without it, a 403 that
    only means "this model is not served in your region" would bench the whole
    provider as an auth failure. Errors also carry ``error_code``, ``trace_id``
    and, when the gateway sent them, ``gateway_routing`` attempts.
    """
    parsed = _parse_error_body(body)
    message = str(parsed.get("message") or "").strip() or f"HTTP {status}"
    code = str(parsed.get("code") or "")
    etype = str(parsed.get("type") or "")
    trace = None
    try:
        trace = headers.get(TRACE_HEADER) if headers is not None else None
    except Exception:  # pragma: no cover
        trace = None

    retryable: bool | None = None
    kind = "unknown"
    lowered = f"{message} {code} {etype}".lower()

    if status == 400 and code == "unsupported_model":
        hint = _ROUTE_HINT.search(message)
        if hint:
            text = (
                f"Model '{model}' is served on /{hint.group(1)}, not on the route "
                f"llmcore chose ({route}). The catalog may be stale; "
                f"`await provider.refresh_catalog()` fixes routing. Gateway said: {message}"
            )
            kind = "bad_request"
        else:
            text = (
                f"Command Code does not serve model '{model}'. "
                f"`await provider.get_models_details()` lists what it does. Gateway said: {message}"
            )
            kind = "model_not_found"
        retryable = False
    elif status == 400 and ("context" in lowered and ("length" in lowered or "too long" in lowered)):
        err = ContextLengthError(model_name=model or "", limit=0, actual=0, message=message)
        err.failure_kind = "context_length"  # type: ignore[attr-defined]
        err.trace_id = trace  # type: ignore[attr-defined]
        return err  # type: ignore[return-value]
    elif status == 401:
        text = (
            "Command Code rejected the API key (401). Set CMD_API_KEY (or "
            "COMMAND_CODE_API_KEY / COMMANDCODE_API_KEY); keys come from "
            f"https://commandcode.ai/studio. Gateway said: {message}"
        )
        kind, retryable = "auth", False
    elif status == 403 and code == "upgrade_required":
        text = (
            "This Command Code plan has no API access (403 upgrade_required). The $1 "
            f"Go plan is CLI-only; GOAT or higher includes the API. Gateway said: {message}"
        )
        kind, retryable = "auth", False
    elif status == 403 and code == "model_region_unavailable":
        text = (
            f"Model '{model}' is not served in your region (not charged). "
            f"Gateway said: {message}"
        )
        kind, retryable = "model_not_found", False
    elif status == 403 and code in ("content_policy_refused", "upstream_refused"):
        text = f"Request refused ({code}, not charged). Gateway said: {message}"
        kind, retryable = "refusal", False
    elif status == 413 or code == "request_too_large":
        text = (
            "Request body exceeds Command Code's 32 MiB limit (413). Do not retry the "
            f"same body; shrink images or drop history. Gateway said: {message}"
        )
        kind, retryable = "bad_request", False
    elif status == 422 and code == "cmd_zdr_no_providers":
        text = (
            f"Zero data retention was required (x-cmd-zdr: 1) but model '{model}' "
            f"has no ZDR upstream. Pick another model or drop zdr. Gateway said: {message}"
        )
        kind, retryable = "model_not_found", False
    elif status == 402 or "usage limit" in lowered or "usage_exceeded" in lowered or (
        "spending limit" in lowered
    ):
        text = f"Command Code usage or credit limit reached ({status}). Gateway said: {message}"
        kind, retryable = "insufficient_credit", False
    elif status == 429:
        text = f"Command Code rate limit (429); retry with backoff. Gateway said: {message}"
        kind, retryable = "rate_limit", True
    elif status == 404:
        text = f"Command Code route not found (404). Gateway said: {message}"
        if parsed.get("cause"):
            text += f" ({parsed['cause']})"
        kind, retryable = "bad_request", False
    elif status >= 500:
        text = f"Command Code upstream unavailable ({status}{', ' + code if code else ''}; not charged). Gateway said: {message}"
        kind, retryable = "server", True
    elif status == 403:
        text = f"Command Code permission denied (403{', ' + code if code else ''}). Gateway said: {message}"
        kind, retryable = "auth", False
    elif status == 400:
        text = f"Command Code rejected the request (400). Gateway said: {message}"
        kind, retryable = "bad_request", False
    else:
        text = f"Command Code API error ({status}). Gateway said: {message}"

    summary = _routing_summary(parsed.get("routing"))
    if summary:
        text = f"{text} [{summary}]"
    if trace:
        text = f"{text} (trace_id={trace})"

    error = ProviderError(
        provider_name,
        text,
        model_name=model,
        status_code=status,
        retryable=retryable,
        original_exception=original,
        headers=dict(headers) if headers is not None else None,
    )
    error.failure_kind = kind  # type: ignore[attr-defined]
    error.error_code = code or None  # type: ignore[attr-defined]
    error.trace_id = trace  # type: ignore[attr-defined]
    error.gateway_routing = parsed.get("routing")  # type: ignore[attr-defined]
    return error


def _from_sdk_exception(
    provider_name: str, exc: BaseException, *, model: str | None, route: str
) -> ProviderError | None:
    """Rebuild a gateway-aware error from the SDK exception behind ``exc``.

    The OpenAI and Anthropic providers wrap SDK status errors in a plain
    :class:`ProviderError` inside their ``except`` blocks, so the SDK exception
    survives as ``__context__``. It carries the status, the headers and the
    body this module needs.
    """
    inner: BaseException | None = exc
    seen = 0
    while inner is not None and seen < 4:
        status = getattr(inner, "status_code", None)
        if isinstance(status, int) and not isinstance(inner, ProviderError):
            response = getattr(inner, "response", None)
            body = getattr(inner, "body", None)
            if body is None and response is not None:
                body = getattr(response, "text", None)
            headers = getattr(response, "headers", None) if response is not None else None
            return build_commandcode_error(
                provider_name,
                status,
                body if body is not None else str(inner),
                model=model,
                headers=headers,
                route=route,
                original=inner if isinstance(inner, Exception) else None,
            )
        inner = inner.__cause__ or inner.__context__
        seen += 1
    return None


def _inband_error(response: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the in-band error of a 200 response or chunk, if it has one."""
    for choice in response.get("choices") or []:
        if not isinstance(choice, Mapping):
            continue
        err = choice.get("error")
        if choice.get("finish_reason") == "error" or isinstance(err, Mapping):
            return dict(err) if isinstance(err, Mapping) else {"message": "unknown upstream error"}
    return None


def build_inband_error(
    provider_name: str,
    err: Mapping[str, Any],
    *,
    model: str | None,
    upstream: str | None,
    partial: str | None,
) -> ProviderError:
    """Turn an HTTP-200 ``finish_reason: "error"`` into a raised error."""
    code = err.get("code")
    status = int(code) if isinstance(code, int) or str(code).isdigit() else 502
    meta = err.get("metadata") if isinstance(err.get("metadata"), Mapping) else {}
    etype = str((meta or {}).get("error_type") or "")
    retryable = status >= 500 or status == 429 or etype in _RETRYABLE_INBAND_TYPES
    trace = (_LAST_CALL.get() or {}).get("trace_id")
    text = f"Command Code upstream failed mid-response ({status}"
    text += f", {etype})" if etype else ")"
    if upstream:
        text += f" from {upstream}"
    text += f": {err.get('message') or 'no detail'}"
    if partial:
        text += f" [discarded {len(partial)} chars of partial output]"
    if trace:
        text += f" (trace_id={trace})"
    error = ProviderError(
        provider_name, text, model_name=model, status_code=status, retryable=retryable
    )
    error.failure_kind = "rate_limit" if status == 429 else "server" if retryable else "unknown"  # type: ignore[attr-defined]
    error.error_code = etype or None  # type: ignore[attr-defined]
    error.trace_id = trace  # type: ignore[attr-defined]
    error.partial_content = partial  # type: ignore[attr-defined]
    return error


# ---------------------------------------------------------------------------
# Lanes
# ---------------------------------------------------------------------------

try:
    from .openai_provider import OpenAIProvider

    _openai_lane_available = True
except Exception:  # pragma: no cover - openai extra not installed
    OpenAIProvider = None  # type: ignore[assignment,misc]
    _openai_lane_available = False

try:
    from .anthropic_provider import AnthropicProvider

    _anthropic_lane_available = True
except Exception:  # pragma: no cover - anthropic extra not installed
    AnthropicProvider = None  # type: ignore[assignment,misc]
    _anthropic_lane_available = False

from .typesafe_provider import TypeSafeProvider  # noqa: E402 - no optional deps

#: Chat Completions parameters the gateway accepts beyond OpenAI's own: the
#: open-model upstreams read these, and pass-through is how the gateway works.
_GATEWAY_CHAT_PARAMS: dict[str, Any] = {
    "max_tokens": {"type": "integer", "minimum": 1},
    "effort": {
        "type": "string",
        "enum": ["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        "description": "llmcore's canonical reasoning effort; sent as reasoning_effort.",
    },
    "reasoning_effort": {
        "type": "string",
        "enum": ["none", "minimal", "low", "medium", "high", "xhigh", "max"],
    },
    "reasoning": {"type": "object", "description": "OpenRouter-style {effort, max_tokens, enabled}."},
    "thinking": {"type": "object", "description": "Upstream thinking toggle, e.g. {type: enabled}."},
    "top_k": {"type": "integer", "minimum": 0},
}

#: Parameters the ``/responses`` lane accepts: Chat Completions spellings it
#: translates, plus the Responses fields themselves.
_RESPONSES_PARAMS: dict[str, Any] = {
    "max_tokens": {"type": "integer", "minimum": 1},
    "max_output_tokens": {"type": "integer", "minimum": 1},
    "temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0},
    "top_p": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    "effort": _GATEWAY_CHAT_PARAMS["effort"],
    "reasoning_effort": _GATEWAY_CHAT_PARAMS["reasoning_effort"],
    "reasoning": {"type": "object", "description": "Responses {effort, summary}."},
    "reasoning_summary": {"type": "string", "enum": ["auto", "concise", "detailed"]},
    "response_format": {"type": "object", "description": "Chat-style; sent as text.format."},
    "text": {"type": "object", "description": "Responses {format, verbosity}."},
    "verbosity": {"type": "string", "enum": ["low", "medium", "high"]},
    "parallel_tool_calls": {"type": "boolean"},
    "max_tool_calls": {"type": "integer", "minimum": 1},
    "store": {"type": "boolean", "description": "Default false: nothing kept server-side."},
    "include": {"type": "array", "items": {"type": "string"}},
    "previous_response_id": {"type": "string"},
    "instructions": {"type": "string"},
    "metadata": {"type": "object"},
    "user": {"type": "string"},
    "safety_identifier": {"type": "string"},
    "service_tier": {"type": "string"},
    "truncation": {"type": "string", "enum": ["auto", "disabled"]},
    "prompt_cache_key": {"type": "string"},
    "prompt_cache_retention": {"type": "string"},
    "top_logprobs": {"type": "integer", "minimum": 0, "maximum": 20},
    "extra_tools": {
        "type": "array",
        "description": "Raw Responses tool declarations (custom, local_shell, web_search).",
    },
}

#: Keys the OpenAI SDK has no keyword for, which must travel in ``extra_body``.
_SDK_EXTRA_BODY_KEYS = ("reasoning", "thinking", "top_k")


if _openai_lane_available:

    class _ChatLane(OpenAIProvider):  # type: ignore[misc,valid-type]
        """``/chat/completions`` lane, with the gateway's quirks handled.

        Differences from a plain OpenAI-compatible client:

        - The system prompt always goes out as ``system``. Qwen, GLM and Kimi
          upstreams reject ``developer``, and every upstream accepts ``system``.
        - ``effort`` maps to ``reasoning_effort``.
        - ``reasoning_details`` on an assistant message's metadata is sent back,
          so multi-turn tool use keeps the upstream's reasoning state.
        - HTTP-200 ``finish_reason: "error"`` responses and stream chunks raise.
        - Errors are mapped by :func:`build_commandcode_error`, on both transports.
        """

        #: The gateway serves chat only. Without this the lane would inherit
        #: OpenAI's image and audio capabilities and claim endpoints Command
        #: Code does not have.
        _MEDIA_CAPABILITIES: frozenset[Any] = frozenset()

        def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False) -> None:
            self._owner_context: Any = config.pop("_context_lookup", None)
            super().__init__(config, log_raw_payloads)

        def _load_tokenizer(self, model_name: str) -> None:
            # Token counts come from the outer provider; a lane never counts, so
            # loading (and warning about) a tokenizer for a gateway id is waste.
            self._encoding = None

        # -- request shaping --------------------------------------------

        def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
            params = dict(super().get_supported_parameters(model))
            params.update(_GATEWAY_CHAT_PARAMS)
            if self._transport == "sdk":
                params["extra_body"] = {"type": "object"}
            return params

        def get_max_context_length(self, model: str | None = None) -> int:
            if self._owner_context is not None:
                return int(self._owner_context(model or self.default_model))
            return super().get_max_context_length(model)

        def _build_message_payload(self, msg: Message, model_name: str) -> dict[str, Any]:
            payload = super()._build_message_payload(msg, model_name)
            if payload.get("role") == "developer":
                payload["role"] = "system"
            if payload.get("role") == "assistant":
                details = (msg.metadata or {}).get("reasoning_details")
                if details:
                    payload["reasoning_details"] = details
            return payload

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
            model_name = model or self.default_model
            effort = kwargs.pop("effort", None)
            if effort and "reasoning_effort" not in kwargs:
                kwargs["reasoning_effort"] = str(effort).strip().lower()
            if stream and "stream_options" not in kwargs:
                kwargs["stream_options"] = {"include_usage": True}
            if self._transport == "sdk":
                extra = dict(kwargs.pop("extra_body", None) or {})
                for key in _SDK_EXTRA_BODY_KEYS:
                    if key in kwargs:
                        extra[key] = kwargs.pop(key)
                if extra:
                    kwargs["extra_body"] = extra
            try:
                result = await super().chat_completion(
                    context, model_name, stream, tools, tool_choice, False, **kwargs
                )
            except ContextLengthError:
                raise
            except ProviderError as exc:
                mapped = _from_sdk_exception(self.get_name(), exc, model=model_name, route=ROUTE_CHAT)
                if mapped is not None:
                    raise mapped from exc
                raise
            if isinstance(result, dict):
                self._check_inband(result, model_name)
                return result
            return self._guard_stream(result, model_name)

        # -- in-band failures -------------------------------------------

        def _check_inband(self, response: dict[str, Any], model_name: str) -> None:
            err = _inband_error(response)
            if err is None:
                return
            partial = self.extract_response_content(response) or None
            raise build_inband_error(
                self.get_name(),
                err,
                model=model_name,
                upstream=response.get("provider"),
                partial=partial,
            )

        async def _guard_stream(
            self, stream: AsyncGenerator[dict[str, Any], None], model_name: str
        ) -> AsyncGenerator[dict[str, Any], None]:
            emitted: list[str] = []
            async for chunk in stream:
                err = _inband_error(chunk)
                if err is not None:
                    raise build_inband_error(
                        self.get_name(),
                        err,
                        model=model_name,
                        upstream=chunk.get("provider"),
                        partial="".join(emitted) or None,
                    )
                if chunk.get("provider"):
                    _note_call(None, upstream=chunk.get("provider"))
                emitted.append(self.extract_delta_content(chunk))
                yield chunk

        # -- direct transport, with headers kept for the trace id ----------

        def _raise_direct_status(self, status: int, body: str, model_name: str) -> None:
            raise build_commandcode_error(
                self.get_name(), status, body, model=model_name, route=ROUTE_CHAT
            )

        async def _chat_via_httpx(
            self,
            model_name: str,
            messages_payload: list[dict[str, Any]],
            stream: bool,
            api_kwargs: dict[str, Any],
        ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
            body: dict[str, Any] = {
                "model": model_name,
                "messages": messages_payload,
                "stream": stream,
                **api_kwargs,
            }
            client = self._get_http()
            if stream:
                return self._stream_via_httpx(client, body, model_name)
            try:
                resp = await client.post("/chat/completions", json=body)
            except httpx.TimeoutException as e:
                raise ProviderError(self.get_name(), f"Timeout: {e}", model_name=model_name) from e
            except httpx.HTTPError as e:
                raise ProviderError(
                    self.get_name(), f"Connection error: {e}", model_name=model_name
                ) from e
            _note_call(resp.headers)
            if resp.status_code >= 400:
                raise build_commandcode_error(
                    self.get_name(),
                    resp.status_code,
                    resp.text,
                    model=model_name,
                    headers=resp.headers,
                    route=ROUTE_CHAT,
                )
            data = resp.json()
            if isinstance(data, dict) and data.get("provider"):
                _note_call(None, upstream=data.get("provider"))
            if self.log_raw_payloads_enabled and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "RAW LLM RESPONSE (%s, direct): %s",
                    self.get_name(),
                    json.dumps(data, indent=2, default=str),
                )
            return data

        async def _stream_via_httpx(
            self, client: Any, body: dict[str, Any], model_name: str
        ) -> AsyncGenerator[dict[str, Any], None]:
            try:
                async with client.stream("POST", "/chat/completions", json=body) as resp:
                    _note_call(resp.headers)
                    if resp.status_code >= 400:
                        await resp.aread()
                        raise build_commandcode_error(
                            self.get_name(),
                            resp.status_code,
                            resp.text,
                            model=model_name,
                            headers=resp.headers,
                            route=ROUTE_CHAT,
                        )
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            return
                        try:
                            chunk = json.loads(payload)
                        except ValueError:
                            logger.warning("Skipping unparseable SSE chunk from %s.", self.get_name())
                            continue
                        yield chunk
            except ProviderError:
                raise
            except httpx.TimeoutException as e:
                raise ProviderError(self.get_name(), f"Timeout: {e}", model_name=model_name) from e
            except httpx.HTTPError as e:
                raise ProviderError(
                    self.get_name(), f"Connection error: {e}", model_name=model_name
                ) from e

    class _ResponsesLane(_ChatLane):
        """``/responses`` lane: OpenAI's Responses API through the gateway.

        The request building, both transports and the normalization to the
        Chat Completions shape are :class:`OpenAIProvider`'s own Responses
        support. This lane only supplies the gateway's error mapping, its
        trace-id capture and its in-band failure shape, through the three
        hooks OpenAIProvider exposes for exactly that.

        - ``store`` defaults to false and encrypted reasoning is requested, so
          a reasoning model's chain can be carried across turns without
          server-side state (``message["reasoning_items"]`` out,
          ``metadata["reasoning_items"]`` back in).
        - ``extra_tools`` passes raw Responses tool declarations (``custom``,
          ``local_shell``) next to llmcore ``Tool`` objects.
        """

        #: Chat only, like the Chat Completions lane.
        _MEDIA_CAPABILITIES: frozenset[Any] = frozenset()

        def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
            return dict(_RESPONSES_PARAMS)

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
            model_name = model or self.default_model
            try:
                return await self._responses_completion(
                    context, model_name, stream, tools, tool_choice, False, **kwargs
                )
            except ContextLengthError:
                raise
            except ProviderError as exc:
                if getattr(exc, "failure_kind", None):
                    raise
                mapped = _from_sdk_exception(
                    self.get_name(), exc, model=model_name, route=ROUTE_RESPONSES
                )
                if mapped is not None:
                    raise mapped from exc
                raise

        # -- OpenAIProvider hooks -----------------------------------------

        def _on_http_response(self, response: Any) -> None:
            _note_call(response.headers)

        def _raise_http_error(self, response: Any, model_name: str) -> None:
            raise build_commandcode_error(
                self.get_name(),
                response.status_code,
                response.text,
                model=model_name,
                headers=response.headers,
                route=ROUTE_RESPONSES,
            )

        def _responses_failed(self, exc: rsp.ResponsesFailed, model_name: str) -> ProviderError:
            code = str(exc.error.get("code") or "")
            status = 429 if "rate_limit" in code else 400 if code.startswith("invalid") else 502
            return build_inband_error(
                self.get_name(),
                {
                    "code": status,
                    "message": exc.error.get("message") or code or "response failed",
                    "metadata": {"error_type": code or "server_error"},
                },
                model=model_name,
                upstream=None,
                partial=exc.partial,
            )

else:  # pragma: no cover
    _ChatLane = None  # type: ignore[assignment,misc]
    _ResponsesLane = None  # type: ignore[assignment,misc]


if _anthropic_lane_available:

    class _MessagesLane(AnthropicProvider):  # type: ignore[misc,valid-type]
        """``/messages`` lane: Claude through the gateway.

        Thinking, effort, caching, vision and tool use are all inherited. This
        class adds the gateway's headers, its error mapping and context windows
        from the catalog.
        """

        def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False) -> None:
            self._owner_context: Any = config.pop("_context_lookup", None)
            self._extra_headers: dict[str, str] = dict(config.pop("default_headers", None) or {})
            super().__init__(config, log_raw_payloads)
            if self._extra_headers:
                from anthropic import AsyncAnthropic

                self._client = AsyncAnthropic(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout,
                    default_headers=self._extra_headers,
                )

        def get_max_context_length(self, model: str | None = None) -> int:
            if self._owner_context is not None:
                return int(self._owner_context(model or self.default_model))
            return super().get_max_context_length(model)

        def _get_http(self) -> Any:
            client = super()._get_http()
            for key, value in self._extra_headers.items():
                client.headers[key] = value
            return client

        def _raise_direct_status(self, status: int, body: str, model_name: str) -> None:
            raise build_commandcode_error(
                self.get_name(), status, body, model=model_name, route=ROUTE_MESSAGES
            )

        async def _direct_messages(self, body: dict[str, Any], model_name: str) -> dict[str, Any]:
            try:
                resp = await self._get_http().post("/v1/messages", json=body)
            except httpx.TimeoutException as e:
                raise ProviderError(self.get_name(), f"Timeout: {e}", model_name=model_name) from e
            except httpx.HTTPError as e:
                raise ProviderError(
                    self.get_name(), f"Connection error: {e}", model_name=model_name
                ) from e
            _note_call(resp.headers)
            if resp.status_code >= 400:
                raise build_commandcode_error(
                    self.get_name(),
                    resp.status_code,
                    resp.text,
                    model=model_name,
                    headers=resp.headers,
                    route=ROUTE_MESSAGES,
                )
            return self._drop_none(resp.json())

        async def chat_completion(
            self,
            context: ContextPayload,
            model: str | None = None,
            stream: bool = False,
            tools: list[Tool] | None = None,
            tool_choice: str | None = None,
            **kwargs: Any,
        ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
            model_name = model or self.default_model
            try:
                return await super().chat_completion(
                    context, model_name, stream, tools, tool_choice, **kwargs
                )
            except ContextLengthError:
                raise
            except ProviderError as exc:
                if getattr(exc, "failure_kind", None):
                    raise
                mapped = _from_sdk_exception(
                    self.get_name(), exc, model=model_name, route=ROUTE_MESSAGES
                )
                if mapped is not None:
                    raise mapped from exc
                raise

else:  # pragma: no cover
    _MessagesLane = None  # type: ignore[assignment,misc]


class _SystemOneLane(TypeSafeProvider):
    """``/systemone`` lane: TypeSafe's typed-judgment API through the gateway."""

    def _map_error(self, response: Any, *, model: str | None, method: str, path: str) -> ProviderError:
        try:
            body: Any = response.json()
        except Exception:
            body = getattr(response, "text", "")
        return build_commandcode_error(
            self.get_name(),
            int(response.status_code),
            body,
            model=model,
            headers=getattr(response, "headers", None),
            route=ROUTE_SYSTEM_ONE,
        )


# ---------------------------------------------------------------------------
# The provider
# ---------------------------------------------------------------------------


class CommandCodeProvider(BaseProvider):
    """Command Code's Provider API: one key, every model, routed per model.

    See the module docstring for the routing model and configuration. Beyond
    the :class:`BaseProvider` surface, it offers:

    - :meth:`resolve_model`: friendly name to catalog id (``kimi-k3`` to
      ``moonshotai/Kimi-K3``).
    - :meth:`route_for`: which lane serves a model.
    - :meth:`refresh_catalog`: re-read ``/models`` now.
    - :meth:`system_one`: typed judgments from ``typesafe/jev``.
    - :meth:`last_call_info`: trace id and upstream of this task's last call.
    """

    #: Command Code serves chat only; media stays with the media providers.
    _MEDIA_CAPABILITIES: frozenset[Any] = frozenset()

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False):
        super().__init__(config, log_raw_payloads)
        self.config = dict(config)

        env_var = config.get("api_key_env_var")
        api_key = config.get("api_key") or (os.environ.get(env_var) if env_var else None)
        if not api_key:
            for name in API_KEY_ENV_VARS:
                api_key = (os.environ.get(name) or "").strip() or None
                if api_key:
                    break
        if not api_key:
            tried = ([env_var] if env_var else []) + list(API_KEY_ENV_VARS)
            raise ConfigError(
                "Command Code API key not found. Set one of "
                f"{', '.join(tried)}, or configure api_key in [providers.commandcode]. "
                "Keys come from https://commandcode.ai/studio (GOAT plan or higher)."
            )
        self.api_key: str = str(api_key)

        base_url = config.get("base_url") or (os.environ.get(ENV_BASE_URL) or "").strip()
        self.base_url: str = str(base_url or COMMANDCODE_BASE_URL).rstrip("/")
        if self.base_url.endswith("/v1"):
            # Accept the URL the docs print (".../provider/v1") as well as the root.
            self.base_url = self.base_url[: -len("/v1")]

        self.default_model: str = str(config.get("default_model") or DEFAULT_MODEL)
        self.timeout: float = float(config.get("timeout", 120.0))
        self.max_retries: int = int(config.get("max_retries", 2))
        transport = str(config.get("transport") or config.get("backend") or "httpx").lower()
        if transport not in ("httpx", "sdk"):
            logger.warning("Unknown transport '%s' for Command Code; using httpx.", transport)
            transport = "httpx"
        self.transport: str = transport
        zdr_cfg = config.get("zdr")
        self.zdr: bool = _truthy(zdr_cfg) if zdr_cfg is not None else _truthy(os.environ.get(ENV_ZDR))
        self._extra_headers: dict[str, str] = {
            str(k): str(v) for k, v in (config.get("default_headers") or {}).items()
        }

        policy = str(config.get("responses_api") or DEFAULT_RESPONSES_POLICY).lower()
        if policy not in RESPONSES_POLICIES:
            logger.warning(
                "Unknown responses_api '%s' for Command Code; using '%s'.",
                policy,
                DEFAULT_RESPONSES_POLICY,
            )
            policy = DEFAULT_RESPONSES_POLICY
        self.responses_policy: str = policy

        self.catalog = CommandCodeCatalog(
            base_url=self.base_url,
            ttl=float(config.get("catalog_ttl", _DEFAULT_CATALOG_TTL)),
            transport=config.get("_catalog_transport"),
        )
        self._lanes: dict[tuple[str, bool], BaseProvider] = {}
        self._encoding: Any = None
        if tiktoken_available:
            try:
                self._encoding = tiktoken.get_encoding("o200k_base")
            except Exception:  # pragma: no cover - offline tiktoken cache
                self._encoding = None

    # -- identity --------------------------------------------------------

    def get_name(self) -> str:
        return self._provider_instance_name or PROVIDER_NAME

    async def warm_up(self) -> None:
        """Read the catalog ahead of the first request."""
        await self.catalog.ensure()

    # -- catalog & routing -------------------------------------------------

    async def refresh_catalog(self) -> int:
        """Re-read ``/models`` now. Returns the number of models listed.

        Raises:
            ProviderError: If the catalog cannot be read.
        """
        await self.catalog.refresh()
        return len(self.catalog)

    def resolve_model(self, name: str | None) -> str:
        """Resolve a friendly model name to its catalog id.

        Falls back to card ids when the catalog has not been read.

        Raises:
            ConfigError: If a short name is ambiguous.
        """
        model = name or self.default_model
        if self.catalog.loaded:
            return self.catalog.resolve(model)
        card = self._card(model)
        return card.model_id if card is not None else model

    def route_for(self, model: str | None, *, responses: bool | None = None) -> str:
        """Which lane serves ``model``: ``messages``, ``chat``, ``responses`` or ``system_one``.

        The live catalog is consulted first, then bundled cards, then family
        names. ``responses`` overrides the instance's ``responses_api`` policy
        for this lookup: ``True`` prefers ``/responses``, ``False`` avoids it,
        wherever the model offers the choice.

        Raises:
            ProviderError: If the catalog lists the model on no route llmcore
                can drive.
        """
        policy = (
            self.responses_policy if responses is None else ("all" if responses else "never")
        )
        model_id = self.resolve_model(model)
        entry = self.catalog.get(model_id)
        if entry is not None:
            route = route_for_endpoints(model_id, entry.endpoints, policy)
            if route is None:
                raise ProviderError(
                    self.get_name(),
                    f"Model '{model_id}' is served only on {list(entry.endpoints)}, none of "
                    "which llmcore's Command Code provider drives.",
                    model_name=model_id,
                )
            return route
        card = self._card(model_id)
        if card is not None:
            routes = (card.provider_extension or {}).get("routes") or []
            route = route_for_endpoints(model_id, tuple(routes), policy)
            if route is not None:
                return route
        return guess_route(model_id)

    def _card(self, model: str) -> Any:
        try:
            from ..model_cards import get_model_card_registry

            return get_model_card_registry().get(PROVIDER_NAME, model)
        except Exception:
            return None

    def _context_for(self, model: str | None) -> int:
        model_id = self.resolve_model(model)
        entry = self.catalog.get(model_id)
        if entry is not None and entry.context_length:
            return entry.context_length
        card = self._card(model_id)
        if card is not None:
            try:
                ctx = card.get_context_length()
                if ctx:
                    return int(ctx)
            except Exception:
                pass
        return _FALLBACK_CLAUDE_CONTEXT if guess_route(model_id) == ROUTE_MESSAGES else _FALLBACK_CONTEXT

    # -- lanes -------------------------------------------------------------

    def _headers(self, zdr: bool) -> dict[str, str]:
        headers = dict(self._extra_headers)
        if zdr:
            headers[ZDR_HEADER] = "1"
        return headers

    def _lane(self, route: str, zdr: bool) -> BaseProvider:
        key = (route, zdr)
        lane = self._lanes.get(key)
        if lane is not None:
            return lane
        common: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.timeout,
            "_instance_name": self.get_name(),
        }
        headers = self._headers(zdr)
        if route == ROUTE_CHAT:
            if _ChatLane is None:  # pragma: no cover
                raise ConfigError("Command Code chat models need `pip install llmcore[commandcode]`.")
            chat_default = (
                self.default_model if self.route_for(self.default_model) == ROUTE_CHAT else "gpt-5.4"
            )
            lane = _ChatLane(
                {
                    **common,
                    "base_url": f"{self.base_url}/v1",
                    "default_model": chat_default,
                    "transport": self.transport,
                    "max_retries": self.max_retries,
                    "default_headers": headers,
                    "_context_lookup": self._context_for,
                },
                self.log_raw_payloads_enabled,
            )
        elif route == ROUTE_RESPONSES:
            if _ResponsesLane is None:  # pragma: no cover
                raise ConfigError("Command Code GPT models need `pip install llmcore[commandcode]`.")
            lane = _ResponsesLane(
                {
                    **common,
                    "base_url": f"{self.base_url}/v1",
                    "default_model": "gpt-6-sol",
                    "transport": self.transport,
                    "max_retries": self.max_retries,
                    "default_headers": headers,
                    "_context_lookup": self._context_for,
                },
                self.log_raw_payloads_enabled,
            )
        elif route == ROUTE_MESSAGES:
            if _MessagesLane is None:  # pragma: no cover
                raise ConfigError("Command Code Claude models need `pip install llmcore[commandcode]`.")
            msg_default = self.default_model if self.route_for(self.default_model) == ROUTE_MESSAGES else "claude-sonnet-5-5"
            lane = _MessagesLane(
                {
                    **common,
                    "base_url": self.base_url,
                    "default_model": msg_default,
                    "backend": self.transport,
                    "default_headers": headers,
                    "_context_lookup": self._context_for,
                    # The gateway's /models carries no capabilities; the
                    # Anthropic provider's measured snapshot covers Claude.
                    "capability_lookup": False,
                },
                self.log_raw_payloads_enabled,
            )
        elif route == ROUTE_SYSTEM_ONE:
            lane = _SystemOneLane(
                {
                    **common,
                    "base_url": self.base_url,
                    "default_model": SYSTEM_ONE_MODEL,
                    "backend": "httpx",
                    "max_retries": self.max_retries,
                    "headers": headers,
                    "fallback_context_length": 32768,
                },
                self.log_raw_payloads_enabled,
            )
        else:  # pragma: no cover - route_for only returns the three above
            raise ProviderError(self.get_name(), f"Unknown Command Code route '{route}'.")
        self._lanes[key] = lane
        return lane

    # -- chat --------------------------------------------------------------

    async def chat_completion(
        self,
        context: ContextPayload,
        model: str | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        """Send a chat request to whichever route serves ``model``.

        Extra keyword arguments beyond the lane's own parameters:

        - ``zdr``: ``True``/``False`` overrides the instance's zero-data-
          retention setting for this call.
        - ``responses``: ``True`` sends the call over ``/responses`` and
          ``False`` over ``/chat/completions``, for models served on both.
          Overrides ``responses_api`` for this call.

        A stale catalog heals itself. If the gateway answers that the model is
        served on another route, the catalog is re-read and the call is retried
        once on the right lane.
        """
        kwargs.pop("native_search", None)  # no native search surface here
        prefer_responses = kwargs.pop("responses", None)
        zdr_arg = kwargs.pop("zdr", None)
        zdr = self.zdr if zdr_arg is None else _truthy(zdr_arg)
        await self.catalog.ensure()
        model_id = self.resolve_model(model)
        _LAST_CALL.set({})

        for attempt in (0, 1):
            route = self.route_for(model_id, responses=prefer_responses)
            if route == ROUTE_SYSTEM_ONE:
                raise ProviderError(
                    self.get_name(),
                    f"'{model_id}' is a decision model, not a chat model; call "
                    "`await provider.system_one(state=..., questions=...)`.",
                    model_name=model_id,
                )
            lane = self._lane(route, zdr)
            try:
                result = await lane.chat_completion(
                    context, model_id, stream, tools, tool_choice, **dict(kwargs)
                )
                if stream:
                    # Pull the first chunk now, so a failure before any byte is
                    # emitted raises here, where routing can still fail over or
                    # this method can still re-route.
                    return await self._primed(result, model_id, route)  # type: ignore[arg-type]
            except ProviderError as exc:
                if attempt == 0 and self._wrong_route(exc):
                    logger.info(
                        "Command Code says '%s' is not served on /%s; refreshing the catalog.",
                        model_id,
                        route,
                    )
                    try:
                        await self.catalog.refresh()
                    except ProviderError:
                        raise exc from None
                    if self.route_for(model_id, responses=prefer_responses) != route:
                        continue
                raise
            self._annotate(result, model_id, route)  # type: ignore[arg-type]
            return result
        raise AssertionError("unreachable")  # pragma: no cover

    @staticmethod
    def _wrong_route(exc: ProviderError) -> bool:
        return getattr(exc, "error_code", None) == "unsupported_model" and bool(
            _ROUTE_HINT.search(str(exc))
        )

    def _remap(self, exc: ProviderError, model_id: str, route: str) -> ProviderError:
        """Gateway-aware version of a lane error that lacks one."""
        if getattr(exc, "failure_kind", None):
            return exc
        return _from_sdk_exception(self.get_name(), exc, model=model_id, route=route) or exc

    async def _primed(
        self, stream: AsyncGenerator[dict[str, Any], None], model_id: str, route: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        try:
            first = await stream.__anext__()
        except StopAsyncIteration:
            first = None
        except ContextLengthError:
            raise
        except ProviderError as exc:
            mapped = self._remap(exc, model_id, route)
            if mapped is exc:
                raise
            raise mapped from exc

        async def replay() -> AsyncGenerator[dict[str, Any], None]:
            if first is not None:
                yield first
            try:
                async for chunk in stream:
                    yield chunk
            except ContextLengthError:
                raise
            except ProviderError as exc:
                mapped = self._remap(exc, model_id, route)
                if mapped is exc:
                    raise
                raise mapped from exc

        return replay()

    def _annotate(self, response: dict[str, Any], model_id: str, route: str) -> None:
        """Attach where the answer came from to a non-streaming response."""
        if not isinstance(response, dict):
            return
        info = dict(_LAST_CALL.get() or {})
        info["route"] = route
        info["model"] = model_id
        if response.get("provider") and "upstream" not in info:
            info["upstream"] = response.get("provider")
        response.setdefault("commandcode", info)

    def last_call_info(self) -> dict[str, Any]:
        """Trace id, upstream and route of the last call made in this task.

        Quote the ``trace_id`` when reporting a bad answer to Command Code.
        """
        return dict(_LAST_CALL.get() or {})

    # -- System One --------------------------------------------------------

    async def system_one(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        model: str | None = None,
        zdr: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        """Typed judgments from TypeSafe's Jev via ``POST /systemone``.

        Takes the same arguments as
        :meth:`llmcore.providers.typesafe_provider.TypeSafeProvider.system_one`
        and returns its ``SystemOneResult``. Questions are
        :class:`~llmcore.providers.typesafe_provider.Noul`, ``Choice`` or
        ``Score`` (or the equivalent dicts). Jev has no ZDR upstream, so
        ``zdr=True`` makes the gateway answer 422.
        """
        lane = self._lane(ROUTE_SYSTEM_ONE, self.zdr if zdr is None else bool(zdr))
        return await lane.system_one(  # type: ignore[attr-defined]
            state, questions, model=model or SYSTEM_ONE_MODEL, **kwargs
        )

    # -- discovery -----------------------------------------------------------

    async def get_models_details(self) -> list[ModelDetails]:
        """Every model the gateway serves, with its route and context window.

        Capability flags come from ``commandcode`` model cards when one exists,
        and are left unknown (``None``) otherwise.

        Raises:
            ProviderError: If the catalog cannot be read.
        """
        if not self.catalog.fresh:
            await self.catalog.refresh()
        out: list[ModelDetails] = []
        for entry in self.catalog.entries():
            card = self._card(entry.id)
            caps = getattr(card, "capabilities", None) if card is not None else None
            ctx_block = getattr(card, "context", None) if card is not None else None
            out.append(
                ModelDetails(
                    id=entry.id,
                    provider_name=self.get_name(),
                    display_name=entry.name,
                    context_length=entry.context_length,
                    max_output_tokens=getattr(ctx_block, "max_output_tokens", None),
                    supports_streaming=entry.route != ROUTE_SYSTEM_ONE,
                    supports_tools=(bool(caps.tool_use or caps.function_calling) if caps else None),
                    supports_vision=(bool(caps.vision) if caps else None),
                    supports_reasoning=(bool(caps.reasoning) if caps else None),
                    model_type="decision" if entry.route == ROUTE_SYSTEM_ONE else "chat",
                    metadata={
                        "route": entry.route,
                        "endpoints": list(entry.endpoints),
                        "owned_by": "command-code",
                        "created": entry.created,
                    },
                )
            )
        return out

    def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
        """Parameters accepted for ``model``, which depend on its route."""
        try:
            route = self.route_for(model)
        except ProviderError:
            route = ROUTE_CHAT
        params: dict[str, Any]
        if route == ROUTE_MESSAGES and _anthropic_lane_available:
            params = dict(AnthropicProvider.get_supported_parameters(self, model))  # type: ignore[arg-type]
        elif route == ROUTE_CHAT and _openai_lane_available:
            params = {
                "temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0},
                "top_p": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "max_completion_tokens": {"type": "integer", "minimum": 1},
                "presence_penalty": {"type": "number"},
                "frequency_penalty": {"type": "number"},
                "stop": {"type": "array", "items": {"type": "string"}},
                "seed": {"type": "integer"},
                "response_format": {"type": "object"},
                "logprobs": {"type": "boolean"},
                "top_logprobs": {"type": "integer"},
                "parallel_tool_calls": {"type": "boolean"},
                "stream_options": {"type": "object"},
                "user": {"type": "string"},
                "verbosity": {"type": "string"},
                **_GATEWAY_CHAT_PARAMS,
            }
            if self.transport == "sdk":
                params["extra_body"] = {"type": "object"}
        elif route == ROUTE_RESPONSES:
            params = dict(_RESPONSES_PARAMS)
        else:
            params = {}
        params["responses"] = {
            "type": "boolean",
            "description": "True: use /responses; False: use /chat/completions (models on both).",
        }
        params["zdr"] = {
            "type": "boolean",
            "description": "Require zero-data-retention upstreams for this call (x-cmd-zdr: 1).",
        }
        return params

    def get_max_context_length(self, model: str | None = None) -> int:
        return self._context_for(model)

    # -- response parsing (shared by both lanes; both are OpenAI-shaped) --------

    def extract_response_content(self, response: dict[str, Any]) -> str:
        try:
            choices = response.get("choices") or []
            return (choices[0].get("message") or {}).get("content") or "" if choices else ""
        except (AttributeError, IndexError, TypeError):
            return ""

    def extract_delta_content(self, chunk: dict[str, Any]) -> str:
        try:
            choices = chunk.get("choices") or []
            return (choices[0].get("delta") or {}).get("content") or "" if choices else ""
        except (AttributeError, IndexError, TypeError):
            return ""

    def extract_reasoning_content(self, response: dict[str, Any]) -> str | None:
        """The model's reasoning: ``reasoning`` (open models) or ``thinking`` (Claude)."""
        try:
            message = (response.get("choices") or [{}])[0].get("message") or {}
        except (AttributeError, IndexError, TypeError):
            return None
        return (
            message.get("reasoning")
            or message.get("reasoning_content")
            or message.get("thinking")
            or None
        )

    def extract_delta_reasoning_content(self, chunk: dict[str, Any]) -> str | None:
        try:
            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
        except (AttributeError, IndexError, TypeError):
            return None
        return delta.get("reasoning") or delta.get("reasoning_content") or delta.get("thinking") or None

    def extract_reasoning_details(self, response: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Structured reasoning to echo back on the next assistant turn.

        Put it in that message's ``metadata["reasoning_details"]`` and the chat
        lane sends it back, which is what keeps tool loops on reasoning models
        coherent.
        """
        try:
            message = (response.get("choices") or [{}])[0].get("message") or {}
        except (AttributeError, IndexError, TypeError):
            return None
        details = message.get("reasoning_details")
        return list(details) if details else None

    def extract_reasoning_items(self, response: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Encrypted reasoning items from a ``/responses`` answer.

        Put them in the next assistant message's ``metadata["reasoning_items"]``
        and they are sent back ahead of that turn, so a GPT reasoning model
        keeps its chain across a tool loop with nothing stored server-side.
        """
        try:
            message = (response.get("choices") or [{}])[0].get("message") or {}
        except (AttributeError, IndexError, TypeError):
            return None
        items = message.get("reasoning_items")
        return list(items) if items else None

    def extract_tool_calls(self, response: dict[str, Any]) -> list[ToolCall]:
        out: list[ToolCall] = []
        try:
            message = (response.get("choices") or [{}])[0].get("message") or {}
        except (AttributeError, IndexError, TypeError):
            return out
        for tc in message.get("tool_calls") or []:
            func = tc.get("function") or {}
            raw = func.get("arguments", "{}")
            if isinstance(raw, dict):
                args = raw
            else:
                try:
                    args = json.loads(raw or "{}")
                except (TypeError, ValueError):
                    args = {"_raw": raw}
            out.append(ToolCall(id=tc.get("id", ""), name=func.get("name", ""), arguments=args))
        return out

    def extract_finish_reason(self, response: dict[str, Any]) -> str | None:
        try:
            return (response.get("choices") or [{}])[0].get("finish_reason")
        except (AttributeError, IndexError, TypeError):
            return None

    def extract_usage_details(self, response: dict[str, Any]) -> dict[str, Any]:
        """Token usage from either route, with cache reads and writes.

        Chat Completions reports ``prompt_tokens_details.cached_tokens`` and
        ``cache_write_tokens``. Messages reports ``cache_read_input_tokens`` and
        ``cache_creation_input_tokens``. Both map onto llmcore's
        ``cached_tokens`` and ``cache_write_tokens``, which cost estimates price.
        """
        usage = response.get("usage") or {}
        if not usage:
            return {}
        out: dict[str, Any] = {
            "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
            "completion_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
        }
        if usage.get("total_tokens") is not None:
            out["total_tokens"] = usage["total_tokens"]
        elif out["prompt_tokens"] is not None and out["completion_tokens"] is not None:
            out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
        pd = usage.get("prompt_tokens_details") or {}
        cd = usage.get("completion_tokens_details") or {}
        cached = pd.get("cached_tokens", usage.get("cache_read_input_tokens"))
        written = pd.get("cache_write_tokens", usage.get("cache_creation_input_tokens"))
        reasoning = cd.get("reasoning_tokens")
        if reasoning is None:
            reasoning = (usage.get("output_tokens_details") or {}).get("thinking_tokens")
        if cached is not None:
            out["cached_tokens"] = cached
        if written is not None:
            out["cache_write_tokens"] = written
        if reasoning is not None:
            out["reasoning_tokens"] = reasoning
        info = response.get("commandcode")
        if isinstance(info, Mapping):
            out.update({k: info[k] for k in ("trace_id", "upstream", "route") if info.get(k)})
        return out

    # -- tokens --------------------------------------------------------------

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        """Estimate tokens with ``o200k_base``.

        Exact counts would need each upstream's own tokenizer. Command Code
        serves a dozen families and exposes no count endpoint.
        """
        if not text:
            return 0
        if self._encoding is None:
            return max(1, len(text) // 4)
        return await asyncio.to_thread(lambda: len(self._encoding.encode(text, disallowed_special=())))

    async def count_message_tokens(self, messages: list[Message], model: str | None = None) -> int:
        total = 3
        for message in messages:
            role = message.role.value if hasattr(message.role, "value") else str(message.role)
            total += 4 + await self.count_tokens(f"{role}\n{message.content or ''}", model)
        return total

    # -- lifecycle -----------------------------------------------------------

    async def close(self) -> None:
        lanes, self._lanes = list(self._lanes.values()), {}
        for lane in lanes:
            try:
                await lane.close()
            except Exception as exc:  # closing must not mask the caller's work
                logger.warning("Error closing a Command Code lane: %s", exc)
