# tests/providers/httpx2_wire.py
"""A recording httpx2 transport for providers built on the openai 3.x SDK.

openai 3.x (like google-genai 3.x) speaks httpx2, which respx cannot
intercept. Give the provider an ``AsyncOpenAI`` on this transport and the
real SDK runs end to end while the test sees the exact wire body.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

httpx2 = pytest.importorskip("httpx2")


class Wire:
    """Queued responses in (per path), recorded requests out."""

    def __init__(self) -> None:
        self.responses: dict[str, list[Any]] = {}
        self.requests: list[Any] = []

    def on(self, path: str, *responses: Any) -> Wire:
        self.responses.setdefault(path, []).extend(responses)
        return self

    def handler(self, request: Any) -> Any:
        self.requests.append(request)
        queue = self.responses[request.url.path]
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def sent(self, i: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[i].content)

    def client(self, base_url: str) -> Any:
        from openai import AsyncOpenAI

        return AsyncOpenAI(
            api_key="k",
            base_url=base_url,
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self.handler)),
        )


def R(status: int, **kw: Any) -> Any:  # noqa: N802
    """An httpx2 response."""
    return httpx2.Response(status, **kw)


def sse(*events: dict[str, Any]) -> bytes:
    """A Chat Completions SSE body."""
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode() + b"data: [DONE]\n\n"
