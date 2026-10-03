# src/llmcore/bridge/proxy_app.py
"""OpenAI-compatible proxy, so an unmodified agent harness routes through llmcore.

The request this exists for was specific: *"add skills and proper ways to
configure llmcore as the 'proxy' for calls for agent harnesses"*, in order to
save money when llmcore sits between a harness and a vendor.

The design follows from one observation. A harness owns its own API call — you
cannot add a `lane=` argument to it — but it almost always lets you set a base
URL and a model name. So those two strings are the entire interface:

    base_url = "http://127.0.0.1:8900/v1"
    model    = "lane:standard"

and the harness gets llmcore's pools, failover, classifiers, cascades,
transforms and cost accounting without knowing llmcore exists.

Endpoints:

``POST /v1/chat/completions``
    The one that matters. ``model`` may be ``lane:<name>``, ``pool:<name>``, a
    target spec (``openai:gpt-5.4?effort=high``), a bare model name, or
    ``auto`` to let the classifier chain decide. Streaming maps to SSE.
``GET /v1/models``
    Lists lanes and pools alongside reachable models, so a harness's model
    picker becomes a way to choose routing policy.
``GET /v1/routing/health`` and ``GET /v1/routing/explain?prompt=...``
    llmcore extensions, for debugging what the proxy did.

Two things are deliberately not hidden:

* **Usage reports the target that actually served the request**, in an
  ``llmcore`` extension block alongside the standard ``usage`` numbers. A
  harness that logs cost per model should not be lied to about which model
  answered — and under a pool, the answer is genuinely not the one that was
  asked for.
* **Auth.** This process holds every provider credential in the config. It
  binds to loopback unless told otherwise, and refuses any other bind without
  a bearer token. That is not configurable politeness; an open routing proxy on
  0.0.0.0 is a credential giveaway.
"""

from __future__ import annotations

import json
import logging
import secrets
import time
import uuid
from typing import Any, AsyncIterator

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from ..routing.budget import BudgetPolicy, TurnBudget

logger = logging.getLogger(__name__)

__all__ = ["ProxyConfig", "create_proxy_app"]

#: Conversations whose budgets are tracked at once. A harness that never
#: reuses a session id would otherwise grow this without limit; the oldest
#: entry is dropped, which loses a budget rather than leaking memory.
MAX_TRACKED_TURNS = 512


class ProxyConfig:
    """Resolved proxy settings.

    Args:
        host: Bind address. Anything other than a loopback address requires
            ``api_key``.
        port: Bind port.
        api_key: Bearer token required on every request. Mandatory for a
            non-loopback bind.
        advertise_lanes: Include lanes in ``GET /v1/models``.
        advertise_pools: Include pools in ``GET /v1/models``.
        default_model: What ``model`` means when a client omits it.
        budget: Per-turn step and spend policy. Unbounded unless configured.

    Raises:
        ValueError: If a non-loopback bind has no token.
    """

    LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost", "127.0.0.0/8"})

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8900,
        api_key: str | None = None,
        advertise_lanes: bool = True,
        advertise_pools: bool = True,
        default_model: str = "auto",
        budget: BudgetPolicy | None = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.api_key = api_key or None
        self.advertise_lanes = advertise_lanes
        self.advertise_pools = advertise_pools
        self.default_model = default_model
        self.budget = budget or BudgetPolicy()

        if not self.is_loopback and not self.api_key:
            raise ValueError(
                f"Refusing to bind the routing proxy to {host!r} without an API key. This "
                f"process holds every provider credential in your config, so an open bind "
                f"hands them to anyone who can reach the port. Set "
                f"routing.proxy.api_key (or LLMCORE_ROUTING__PROXY__API_KEY), or bind to "
                f"127.0.0.1."
            )

    @property
    def is_loopback(self) -> bool:
        return self.host in self.LOOPBACK or self.host.startswith("127.")

    @classmethod
    def from_config(cls, get: Any) -> ProxyConfig:
        """Build from ``[routing.proxy]``."""
        read = get or (lambda _k, d=None: d)
        return cls(
            host=str(read("routing.proxy.host", "127.0.0.1")),
            port=int(read("routing.proxy.port", 8900)),
            api_key=read("routing.proxy.api_key", "") or None,
            advertise_lanes=bool(read("routing.proxy.advertise_lanes", True)),
            advertise_pools=bool(read("routing.proxy.advertise_pools", True)),
            default_model=str(read("routing.proxy.default_model", "auto")),
            budget=BudgetPolicy.from_config(read),
        )

    def harness_env(self) -> dict[str, str]:
        """Environment a harness needs, ready to paste.

        The ``llmcore-proxy`` skill prints this. ``OPENAI_API_KEY`` is set to
        the proxy's own token (or a placeholder on loopback) because most
        harnesses insist on *some* key being present.
        """
        return {
            "OPENAI_BASE_URL": f"http://{self.host}:{self.port}/v1",
            "OPENAI_API_KEY": self.api_key or "llmcore-local",
            "OPENAI_MODEL": self.default_model,
        }


# ---------------------------------------------------------------------------
# Model-name parsing
# ---------------------------------------------------------------------------


def parse_model(model: str | None, *, default: str = "auto") -> dict[str, Any]:
    """Turn a wire ``model`` string into routing arguments.

    This is the whole trick of proxy mode: a harness can only send a model
    name, so the model name has to be able to carry routing intent.

    ============================  ==========================================
    ``model``                     meaning
    ============================  ==========================================
    ``lane:deep``                 route to the ``deep`` lane
    ``pool:main``                 route through the ``main`` pool
    ``auto``                      let the classifier chain decide
    ``openai:gpt-5.4?effort=max`` an explicit target, parameters and all
    ``gpt-5.4``                   a bare model name on the default provider
    ============================  ==========================================

    A bare name is sent as ``model_name`` rather than as a target, so a harness
    configured for a plain OpenAI model keeps working unchanged — which is the
    difference between a drop-in proxy and a migration.
    """
    text = (model or default or "auto").strip()
    lowered = text.lower()

    if lowered in ("auto", "", "llmcore", "default"):
        return {}
    if lowered.startswith("lane:"):
        return {"lane": text[5:].strip()}
    if lowered.startswith("pool:"):
        return {"pool": text[5:].strip()}
    if lowered.startswith("profile:"):
        return {"profile": text[8:].strip()}
    if ":" in text:
        return {"target": text}
    return {"model_name": text}


def _messages_to_chat(messages: list[dict[str, Any]]) -> tuple[str, str | None, list[Any]]:
    """Split an OpenAI message array into llmcore's ``chat()`` shape.

    Returns ``(final_user_message, system_message, prior_messages)``. The last
    user message becomes the prompt and everything before it becomes history,
    which is how ``chat()`` expects a turn to arrive.
    """
    from llmcore.models import Message, Role

    system: str | None = None
    prior: list[Any] = []
    prompt = ""

    for index, raw in enumerate(messages):
        role = str(raw.get("role", "user")).lower()
        content = raw.get("content")
        if isinstance(content, list):
            # Multimodal content parts: keep the text, which is all a routing
            # proxy can honestly forward through a text chat() call.
            content = "\n".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") in (None, "text")
            )
        content = "" if content is None else str(content)

        if role == "system":
            system = content if system is None else f"{system}\n\n{content}"
            continue
        if index == len(messages) - 1 and role == "user":
            prompt = content
            continue
        prior.append(
            Message(
                role=Role.ASSISTANT if role == "assistant" else Role.USER,
                content=content,
                session_id="",
            )
        )

    if not prompt:
        # No trailing user turn (a harness replaying an assistant prefix, say).
        # Use the last message we have rather than failing the request.
        for raw in reversed(messages):
            if raw.get("content"):
                prompt = str(raw["content"])
                break
        if prior and prompt:
            prior = prior[:-1]
    return prompt, system, prior


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def create_proxy_app(llm: Any, config: ProxyConfig | None = None) -> Starlette:
    """Build the OpenAI-compatible ASGI app.

    Args:
        llm: A live :class:`~llmcore.api.LLMCore`.
        config: Proxy settings; read from ``[routing.proxy]`` when omitted.

    Returns:
        A Starlette app with the ``/v1`` surface mounted at the root.
    """
    cfg = config or ProxyConfig.from_config(getattr(llm, "config", None) and llm.config.get)

    #: Live turn budgets, by conversation. Only populated for conversations
    #: the caller identified -- see ``_budget_for``.
    turns: dict[str, TurnBudget] = {}

    def _budget_for(caller_session: str | None) -> TurnBudget | None:
        """The budget for this conversation, or ``None`` if it cannot be kept.

        A step budget needs a stable key across the calls of one turn, and the
        proxy only has one when the harness identifies the conversation --
        through ``user`` or ``llmcore.session_id``. Without that every request
        gets a fresh synthetic id, so "steps so far" would always be zero and
        enforcement would be theatre. Rather than counting something
        meaningless, this declines to track the turn at all.
        """
        if not cfg.budget.is_bounded or not caller_session:
            return None
        budget = turns.get(caller_session)
        if budget is None:
            if len(turns) >= MAX_TRACKED_TURNS:
                # Oldest first: a dropped budget under-counts, which is the
                # safe direction to fail compared with unbounded growth in a
                # process holding every provider credential.
                turns.pop(next(iter(turns)), None)
            budget = TurnBudget(cfg.budget)
            turns[caller_session] = budget
        return budget

    def budget_exceeded(verdict: Any) -> JSONResponse:
        """Refuse a step the budget will not pay for."""
        logger.warning("Proxy refused a step: %s", verdict.reason)
        return JSONResponse(
            {
                "error": {
                    "message": (
                        f"llmcore refused this request: {verdict.reason}. The "
                        f"turn's budget is set by [routing.budget]."
                    ),
                    "type": "budget_exceeded",
                    "code": "budget_exceeded",
                },
                "llmcore": {"budget": verdict.as_dict()},
            },
            # 429 rather than 400: the request is well-formed and the same
            # request in a new conversation would succeed, which is what a
            # harness's retry logic reads this status as.
            status_code=429,
        )

    def authorised(request: Request) -> bool:
        if not cfg.api_key:
            return True
        header = request.headers.get("authorization", "")
        token = header[7:].strip() if header.lower().startswith("bearer ") else ""
        # Constant-time compare: this token guards every provider credential
        # in the process, so it should not be discoverable by timing.
        return bool(token) and secrets.compare_digest(token, cfg.api_key)

    def unauthorised() -> JSONResponse:
        return JSONResponse(
            {
                "error": {
                    "message": "Missing or invalid bearer token.",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
            status_code=401,
        )

    def error_response(exc: BaseException) -> JSONResponse:
        """Map an llmcore exception onto an OpenAI-shaped error.

        Harnesses branch on these: a 429 triggers their own backoff, a 400
        usually surfaces to the user. Getting the status right is what makes
        the proxy behave like the thing it is impersonating.
        """
        from llmcore.exceptions import (
            ConfigError,
            ContextLengthError,
            NoTargetAvailableError,
            PromptBlockedError,
            ProviderError,
        )
        from llmcore.routing import FailureKind, classify_failure

        if isinstance(exc, PromptBlockedError):
            return JSONResponse(
                {
                    "error": {
                        "message": str(exc),
                        "type": "invalid_request_error",
                        "code": "blocked_by_policy",
                    }
                },
                status_code=403,
            )
        if isinstance(exc, NoTargetAvailableError):
            return JSONResponse(
                {
                    "error": {
                        "message": str(exc),
                        "type": "server_error",
                        "code": "no_target_available",
                    }
                },
                status_code=503,
            )
        if isinstance(exc, ContextLengthError):
            return JSONResponse(
                {
                    "error": {
                        "message": str(exc),
                        "type": "invalid_request_error",
                        "code": "context_length_exceeded",
                    }
                },
                status_code=400,
            )
        if isinstance(exc, ConfigError):
            return JSONResponse(
                {"error": {"message": str(exc), "type": "invalid_request_error"}},
                status_code=400,
            )
        # Everything else goes through the routing classifier rather than only
        # ProviderError: a bare TimeoutError from a transport is a 504, not a
        # 500, and a harness treats those very differently. An exception the
        # classifier cannot place is a genuine 500 and is logged with its
        # traceback, since it is a bug rather than a vendor condition.
        kind, retry_after = classify_failure(exc)
        if kind is FailureKind.UNKNOWN and not isinstance(exc, ProviderError):
            logger.exception("Unhandled error in the routing proxy")
            return JSONResponse(
                {"error": {"message": str(exc), "type": "server_error"}}, status_code=500
            )
        status = {
            FailureKind.RATE_LIMIT: 429,
            FailureKind.INSUFFICIENT_CREDIT: 402,
            FailureKind.AUTH: 401,
            FailureKind.BAD_REQUEST: 400,
            FailureKind.MODEL_NOT_FOUND: 404,
            FailureKind.CONTEXT_LENGTH: 400,
            FailureKind.TIMEOUT: 504,
            FailureKind.SERVER: 502,
        }.get(kind, 502)
        headers = {"retry-after": str(int(retry_after))} if retry_after else None
        return JSONResponse(
            {"error": {"message": str(exc), "type": "api_error", "code": str(kind)}},
            status_code=status,
            headers=headers,
        )

    # -- endpoints --------------------------------------------------------

    async def chat_completions(request: Request) -> Response:
        if not authorised(request):
            return unauthorised()
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                {"error": {"message": "Request body is not valid JSON.", "type": "invalid_request_error"}},
                status_code=400,
            )

        messages = body.get("messages") or []
        if not isinstance(messages, list) or not messages:
            return JSONResponse(
                {"error": {"message": "'messages' must be a non-empty array.", "type": "invalid_request_error"}},
                status_code=400,
            )

        prompt, system, prior = _messages_to_chat(messages)
        routing_kwargs = parse_model(body.get("model"), default=cfg.default_model)
        stream = bool(body.get("stream"))

        # Standard sampling parameters pass straight through. Anything the
        # chosen target does not support is dropped rather than erroring,
        # because a harness sends the same body to every model it talks to.
        passthrough = {
            key: body[key]
            for key in ("temperature", "top_p", "max_tokens", "stop", "seed", "presence_penalty",
                        "frequency_penalty", "reasoning_effort")
            if body.get(key) is not None
        }
        if "reasoning_effort" in passthrough:
            routing_kwargs.setdefault("effort", passthrough.pop("reasoning_effort"))

        # Routing policy may also arrive in extra_body, which is how the
        # OpenAI SDKs let a caller send non-standard fields.
        extra = body.get("llmcore") or body.get("extra_body", {}).get("llmcore") or {}
        if isinstance(extra, dict):
            for key in ("lane", "pool", "profile", "target", "complexity", "effort"):
                if extra.get(key) is not None:
                    routing_kwargs[key] = extra[key]
            if isinstance(extra.get("routing"), dict):
                routing_kwargs["routing"] = extra["routing"]

        caller_session = (
            extra.get("session_id") if isinstance(extra, dict) else None
        ) or body.get("user")
        budget = _budget_for(caller_session)
        if budget is not None:
            verdict = budget.check()
            if verdict.should_stop:
                return budget_exceeded(verdict)

        created = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        # A synthetic session id when the caller gave none. llmcore keys its
        # per-turn introspection by session, and that introspection is where
        # the real token counts and the target that answered come from -- so
        # without an id the usage block would be empty, which is the one thing
        # this endpoint promised not to do. `save_session` stays off, so
        # nothing is persisted, and the cached state is dropped after reading
        # it (see below) rather than accumulating one entry per request.
        session_id = caller_session or f"proxy-{uuid.uuid4().hex[:16]}"

        if stream:
            return StreamingResponse(
                _stream(
                    llm, prompt, system, prior, routing_kwargs, passthrough,
                    session_id=session_id, caller_session=caller_session,
                    completion_id=completion_id, created=created,
                    model_label=str(body.get("model") or cfg.default_model),
                    budget=budget,
                ),
                media_type="text/event-stream",
                headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
            )

        try:
            answer = await llm.chat(
                prompt,
                system_message=system,
                extra_messages=prior or None,
                session_id=session_id,
                save_session=bool(caller_session),
                stream=False,
                **routing_kwargs,
                **passthrough,
            )
        except BaseException as exc:
            if not caller_session:
                llm.discard_transient_state(session_id)
            return error_response(exc)

        info = llm.get_last_interaction_context_info(session_id)
        served = _served(llm, info)
        if budget is not None:
            _record_step(budget, served)
            served["budget"] = budget.check().as_dict()
        if not caller_session:
            llm.discard_transient_state(session_id)
        return JSONResponse(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": served.get("model") or str(body.get("model") or ""),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": served.get("prompt_tokens", 0),
                    "completion_tokens": served.get("completion_tokens", 0),
                    "total_tokens": served.get("total_tokens", 0),
                },
                # Not decoration: under a pool the model that answered is
                # genuinely not the one that was requested, and a harness
                # logging spend per model needs to know which.
                "llmcore": served,
            }
        )

    async def list_models(request: Request) -> Response:
        if not authorised(request):
            return unauthorised()
        created = int(time.time())
        entries: list[dict[str, Any]] = [
            {
                "id": "auto",
                "object": "model",
                "created": created,
                "owned_by": "llmcore",
                "llmcore": {"kind": "auto", "description": "Let the classifier chain decide"},
            }
        ]
        try:
            if cfg.advertise_lanes:
                for name, destination in llm.routing.lanes().items():
                    entries.append(
                        {
                            "id": f"lane:{name}",
                            "object": "model",
                            "created": created,
                            "owned_by": "llmcore",
                            "llmcore": {"kind": "lane", "destination": destination},
                        }
                    )
            if cfg.advertise_pools:
                for name, members in llm.routing.pools().items():
                    entries.append(
                        {
                            "id": f"pool:{name}",
                            "object": "model",
                            "created": created,
                            "owned_by": "llmcore",
                            "llmcore": {"kind": "pool", "members": members},
                        }
                    )
        except Exception:
            logger.debug("routing is unavailable; listing providers only", exc_info=True)

        for instance in llm.get_available_providers():
            entries.append(
                {
                    "id": instance,
                    "object": "model",
                    "created": created,
                    "owned_by": "llmcore",
                    "llmcore": {"kind": "provider"},
                }
            )
        return JSONResponse({"object": "list", "data": entries})

    async def routing_health(request: Request) -> Response:
        if not authorised(request):
            return unauthorised()
        return JSONResponse({"targets": llm.routing.health()})

    async def routing_explain(request: Request) -> Response:
        if not authorised(request):
            return unauthorised()
        prompt = request.query_params.get("prompt") or ""
        if not prompt:
            return JSONResponse(
                {"error": {"message": "Pass ?prompt=...", "type": "invalid_request_error"}},
                status_code=400,
            )
        kwargs = parse_model(request.query_params.get("model"), default=cfg.default_model)
        kwargs.pop("model_name", None)
        try:
            plan = await llm.routing.explain(prompt, **kwargs)
        except BaseException as exc:
            return error_response(exc)
        return JSONResponse(
            {
                "summary": plan.summary(),
                "lane": plan.lane,
                "pool": plan.pool,
                "chosen": plan.chosen.spec() if plan.chosen else None,
                "estimated_cost_usd": plan.estimated_cost_usd,
                "classifier": plan.classification.source if plan.classification else None,
                "confidence": plan.classification.confidence if plan.classification else None,
                "candidates": [
                    {
                        "target": candidate.target.spec(),
                        "eligible": candidate.eligible,
                        "reason": candidate.reason,
                        "score": candidate.score,
                    }
                    for candidate in plan.candidates
                ],
                "notes": list(plan.notes),
            }
        )

    async def healthz(request: Request) -> Response:
        return JSONResponse({"status": "ok", "proxy": "llmcore"})

    return Starlette(
        routes=[
            Route("/v1/chat/completions", chat_completions, methods=["POST"]),
            Route("/v1/models", list_models, methods=["GET"]),
            Route("/v1/routing/health", routing_health, methods=["GET"]),
            Route("/v1/routing/explain", routing_explain, methods=["GET"]),
            Route("/healthz", healthz, methods=["GET"]),
        ]
    )


def _record_step(budget: TurnBudget, served: dict[str, Any]) -> None:
    """Add one served request to *budget*, pricing it from its model card.

    The cost is priced here rather than taken from the response because
    OpenAI's wire format has no cost field -- and the target that answered may
    not be the one that was asked for, so the harness could not price it
    either.

    An unpriceable target records ``None``, which the budget counts as a step
    whose cost is unknown. It must not arrive as ``0.0``: a turn full of
    unpriced calls would then look free and never trip a spend ceiling, which
    is the failure this subsystem's cost model has had repeatedly.
    """
    cost: float | None = None
    provider = served.get("provider")
    model = served.get("model")
    prompt_tokens = int(served.get("prompt_tokens") or 0)
    completion_tokens = int(served.get("completion_tokens") or 0)
    if provider and model:
        try:
            from ..routing.cards import estimate_cost_usd
            from ..routing.models import Target

            cost = estimate_cost_usd(
                Target.parse(f"{provider}:{model}"),
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
            )
        except Exception as exc:  # pragma: no cover - pricing is best-effort
            logger.debug("Proxy could not price %s:%s: %s", provider, model, exc)
    budget.record(
        cost_usd=cost,
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
    )


def _served(llm: Any, info: Any) -> dict[str, Any]:
    """Describe who actually answered, for the ``llmcore`` usage block."""
    served: dict[str, Any] = {}
    if info is not None:
        for field in ("provider", "model", "prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(info, field, None)
            if value is not None:
                served[field] = value
    if "provider" in served and "model" in served:
        served["target"] = f"{served['provider']}:{served['model']}"
    return served


async def _stream(
    llm: Any,
    prompt: str,
    system: str | None,
    prior: list[Any],
    routing_kwargs: dict[str, Any],
    passthrough: dict[str, Any],
    *,
    session_id: str | None,
    caller_session: str | None,
    completion_id: str,
    created: int,
    model_label: str,
    budget: TurnBudget | None = None,
) -> AsyncIterator[bytes]:
    """Emit an OpenAI-shaped SSE stream.

    An error *before* the first chunk is emitted as an error event the client
    can act on. An error *after* is emitted too, but the caller has already
    received partial content — which is exactly why routing does not fail over
    a stream (see ``chat()``): the bytes cannot be taken back.
    """
    from llmcore.exceptions import LLMCoreError

    def chunk(delta: dict[str, Any], finish: str | None = None) -> bytes:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_label,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload)}\n\n".encode()

    try:
        generator = await llm.chat(
            prompt,
            system_message=system,
            extra_messages=prior or None,
            session_id=session_id,
            save_session=bool(caller_session),
            stream=True,
            **routing_kwargs,
            **passthrough,
        )
        yield chunk({"role": "assistant", "content": ""})
        async for piece in generator:
            if piece:
                yield chunk({"content": piece})
        yield chunk({}, finish="stop")
        yield b"data: [DONE]\n\n"
        # Recorded after the stream completes, not before it starts: the step
        # has happened either way, and leaving it uncounted would make
        # streaming a way to spend without a budget noticing.
        if budget is not None:
            _record_step(budget, _served(llm, llm.get_last_interaction_context_info(session_id)))
        if not caller_session:
            llm.discard_transient_state(session_id)
    except LLMCoreError as exc:
        if budget is not None:
            # A stream that died partway still burned tokens upstream.
            budget.record(cost_usd=None)
        yield f"data: {json.dumps({'error': {'message': str(exc), 'type': 'api_error'}})}\n\n".encode()
        yield b"data: [DONE]\n\n"
    except Exception as exc:
        logger.exception("Routing proxy stream failed")
        yield f"data: {json.dumps({'error': {'message': str(exc), 'type': 'server_error'}})}\n\n".encode()
        yield b"data: [DONE]\n\n"
