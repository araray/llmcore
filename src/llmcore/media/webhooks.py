# src/llmcore/media/webhooks.py
"""Generic webhook receiver for long-running media jobs (spec §2.8).

Async media jobs need a callback path, and every vendor spells it differently:
fal takes a ``?fal_webhook=`` query parameter, ElevenLabs registers one per
speech-to-text request, Replicate posts prediction events. What they share is
the shape — *"here is a URL; I will POST to it when the work is done"* — so
llmcore issues the URL and owns the receiving end, rather than growing a
per-provider callback handler.

Three properties matter, and they are the reason this is not just a dict:

1. **Polling is always the fallback.** A callback is an optimization, never a
   requirement. llmcore must stay fully usable with no public ingress, which is
   the normal case in local development, so :meth:`MediaJobManager.wait` races
   the callback against its existing poll loop and takes whichever arrives
   first. That is also what makes webhook and poll delivery *equivalent* rather
   than two code paths with two sets of bugs.

2. **The URL is a credential.** It goes to a third party and travels over the
   public internet, so the token is HMAC-signed, verified in constant time, and
   **single-use** — a replayed delivery is rejected. The callback URL has to be
   handed to the vendor *at submission*, before the job it will report on
   exists, so a token is **reserved** first and **bound** to the job the
   submission returns. The binding lives server-side, which is what stops a
   token being transplanted onto another job.

3. **A callback is untrusted input.** Anyone who learns a URL can POST to it,
   so a delivery may only *report* a job's outcome; it can never create a job,
   redirect one to another provider, or hand back an artifact for work llmcore
   never submitted. Parsing the body stays with the provider adapter, which
   already knows that vendor's result shape.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from ..exceptions import MediaError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .models import MediaJob

logger = logging.getLogger(__name__)

__all__ = [
    "WebhookDelivery",
    "WebhookRegistry",
    "create_webhook_app",
]

#: Path the receiver serves, relative to the configured base URL.
WEBHOOK_PATH = "/media/jobs"


@dataclass(frozen=True, slots=True)
class WebhookDelivery:
    """The outcome of handling one inbound callback.

    Attributes:
        accepted: Whether the delivery was applied to a job.
        job_id: The job it addressed, when the token was valid.
        reason: Why it was rejected, when it was.
        status_code: The HTTP status a server should return.
    """

    accepted: bool
    job_id: str | None = None
    reason: str | None = None
    status_code: int = 200


@dataclass
class _Ticket:
    """Server-side state for one issued callback token.

    ``job_id`` is ``None`` between :meth:`WebhookRegistry.reserve` and
    :meth:`WebhookRegistry.bind` — the window during which the submission is
    in flight and the job does not yet exist. A delivery arriving in that
    window is refused, because there is nothing it could correctly report on.
    """

    nonce: str
    job_id: str | None = None
    issued_at: float = field(default_factory=time.monotonic)
    used: bool = False
    event: asyncio.Event = field(default_factory=asyncio.Event)


class WebhookRegistry:
    """Issues and verifies single-use callback tokens for media jobs.

    Args:
        base_url: Public base URL callbacks are delivered to. When empty or
            ``None``, the registry is **disabled** and :meth:`issue` returns
            ``None`` — llmcore then polls, which is the documented default
            rather than an error state.
        secret: HMAC key. Generated per process when omitted, which means
            tokens do not survive a restart; set it explicitly if jobs must
            outlive the process that submitted them.
        path: Receiver path appended to *base_url*.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        secret: str | None = None,
        path: str = WEBHOOK_PATH,
    ) -> None:
        self._base_url = (base_url or "").rstrip("/")
        self._secret = (secret or secrets.token_urlsafe(32)).encode()
        self._path = "/" + path.strip("/")
        self._tickets: dict[str, _Ticket] = {}

    @property
    def enabled(self) -> bool:
        """Whether callbacks are configured. ``False`` means poll-only."""
        return bool(self._base_url)

    def _sign(self, nonce: str) -> str:
        """Return the HMAC tag proving this registry issued *nonce*."""
        mac = hmac.new(self._secret, nonce.encode(), hashlib.sha256)
        return mac.hexdigest()[:32]

    def reserve(self) -> tuple[str, str] | None:
        """Reserve an unbound callback token, or ``None`` when disabled.

        Returns:
            ``(token, url)``. The token is not yet attached to a job — call
            :meth:`bind` once the submission returns one, or :meth:`release`
            if it did not produce a job at all.
        """
        if not self.enabled:
            return None
        nonce = secrets.token_urlsafe(16)
        token = f"{nonce}.{self._sign(nonce)}"
        self._tickets[token] = _Ticket(nonce=nonce)
        return token, f"{self._base_url}{self._path}/{quote(token, safe='')}"

    def bind(self, token: str, job_id: str) -> None:
        """Attach a reserved *token* to the job the submission produced."""
        ticket = self._tickets.get(token)
        if ticket is not None and ticket.job_id is None:
            ticket.job_id = job_id
            logger.debug("Bound webhook token to media job %s", job_id)

    def release(self, token: str) -> None:
        """Discard a reserved token that never became a job."""
        self._tickets.pop(token, None)

    def issue(self, job: MediaJob) -> str | None:
        """Reserve and immediately bind a callback URL for an existing *job*."""
        reserved = self.reserve()
        if reserved is None:
            return None
        token, url = reserved
        self.bind(token, job.id)
        return url

    def revoke(self, job_id: str) -> None:
        """Drop any tokens issued for *job_id*."""
        for token, ticket in list(self._tickets.items()):
            if ticket.job_id == job_id:
                ticket.event.set()
                del self._tickets[token]

    def verify(self, token: str) -> str | None:
        """Return the job id *token* addresses, or ``None`` if it is not valid.

        Rejects anything malformed, unknown, already used, still unbound, or
        whose signature does not match. The signature comparison is constant
        time, so it leaks nothing about the expected value.
        """
        ticket = self._tickets.get(token)
        if ticket is None or ticket.used or ticket.job_id is None:
            return None
        try:
            nonce, tag = token.split(".")
        except ValueError:
            return None
        if nonce != ticket.nonce:
            return None
        if not hmac.compare_digest(tag, self._sign(nonce)):
            return None
        return ticket.job_id

    def consume(self, token: str) -> str | None:
        """Verify *token* and burn it, so a replayed delivery is rejected."""
        job_id = self.verify(token)
        if job_id is None:
            return None
        ticket = self._tickets[token]
        ticket.used = True
        ticket.event.set()
        return job_id

    def event_for(self, job_id: str) -> asyncio.Event | None:
        """Return the event signalled when *job_id* receives a callback."""
        for ticket in self._tickets.values():
            if ticket.job_id == job_id and not ticket.used:
                return ticket.event
        return None

    def pending(self) -> int:
        """Number of tokens still awaiting delivery."""
        return sum(1 for t in self._tickets.values() if not t.used)


def create_webhook_app(manager: Any, registry: WebhookRegistry) -> Any:
    """Build a minimal ASGI app that receives callbacks for *manager*.

    Deliberately dependency-free: it speaks raw ASGI rather than importing a web
    framework, so mounting a receiver never forces a framework choice on a
    library consumer. Mount it under an existing server (llmcore's bridge, or
    any ASGI host) at the path the registry was configured with.

    Args:
        manager: The :class:`~llmcore.media.jobs.MediaJobManager` to notify.
        registry: The registry that issued the tokens.

    Returns:
        An ASGI application callable.
    """

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":  # pragma: no cover - lifespan/ws ignored
            return
        status, body = await _dispatch(manager, registry, scope, receive)
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body})

    return app


async def _dispatch(
    manager: Any, registry: WebhookRegistry, scope: dict[str, Any], receive: Any
) -> tuple[int, bytes]:
    """Route one ASGI request to the job manager."""
    import json

    if scope.get("method") != "POST":
        return 405, b'{"error":"method not allowed"}'

    token = scope.get("path", "").rstrip("/").rsplit("/", 1)[-1]
    raw = b""
    while True:
        message = await receive()
        raw += message.get("body", b"")
        if not message.get("more_body"):
            break

    try:
        payload = json.loads(raw or b"{}")
    except ValueError:
        return 400, b'{"error":"invalid json"}'

    delivery = await manager.handle_webhook(token, payload)
    if delivery.accepted:
        return delivery.status_code, b'{"ok":true}'
    # A rejected token gets 404 rather than 401: confirming a token exists but
    # is spent tells an unauthenticated caller something they should not learn.
    return delivery.status_code, json.dumps({"error": delivery.reason}).encode()


async def apply_webhook_payload(job: MediaJob, payload: dict[str, Any], adapter: Any) -> MediaJob:
    """Fold a vendor callback *payload* into *job* using its *adapter*.

    Providers that can parse their own callback implement
    ``apply_webhook_payload(job, payload)``. Those that cannot fall back to a
    poll, because the callback still tells us *when* to look even if we cannot
    read *what* it says — which is most of the latency win, at no correctness
    cost.
    """
    handler = getattr(adapter, "apply_webhook_payload", None)
    if callable(handler):
        return await handler(job, payload)

    poller = getattr(adapter, "poll_media_job", None)
    if not callable(poller):
        raise MediaError(
            f"Provider '{job.provider}' can neither parse a webhook payload nor be polled."
        )
    logger.debug(
        "Provider %s has no webhook parser; polling job %s after callback.",
        job.provider,
        job.id,
    )
    return await poller(job)
