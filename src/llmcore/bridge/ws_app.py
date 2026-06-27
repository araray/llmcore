"""WebSocket transport placeholder for Tier-2 live duplex audio (phase B3).

The HTTP/SSE app (``http_app.py``) covers all Tier-0 RPCs including server
streaming (``ChatStream`` over SSE). Bidirectional audio (``TranscribeStream``,
``SynthesizeStream``, ``VoiceAgent``) needs a *client->server* byte/event channel
that SSE cannot provide; on the HTTP side that maps to WebSocket.

These routes are intentionally **not** mounted by :func:`create_http_app` in v1.
They are defined here so the wire layout is documented and so B3 can enable them
by adding ``create_ws_routes(core)`` to the app's route table. Each currently
accepts the socket and closes immediately with policy code ``1011`` and a
machine-readable reason, mirroring the gRPC ``UNIMPLEMENTED`` behaviour.
"""

from __future__ import annotations

from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocket

from .core import BridgeCore

__all__ = ["create_ws_routes"]

_REASON = (
    "AudioService live duplex (TranscribeStream/SynthesizeStream/VoiceAgent) is "
    "part of the llmcore.v1 contract but is not implemented in this release "
    "(Tier 2 / phase B3)."
)


def _make_unimplemented_ws(_core: BridgeCore):
    async def endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        # 1011 = internal error / unsupported per RFC 6455 application convention.
        await websocket.close(code=1011, reason=_REASON)

    return endpoint


def create_ws_routes(core: BridgeCore) -> list[WebSocketRoute]:
    """Return the (currently UNIMPLEMENTED) Tier-2 WebSocket routes.

    Not mounted in v1; B3 will register these on the Starlette app.
    """
    base = "/llmcore.v1/AudioService"
    handler = _make_unimplemented_ws(core)
    return [
        WebSocketRoute(f"{base}/TranscribeStream", handler),
        WebSocketRoute(f"{base}/SynthesizeStream", handler),
        WebSocketRoute(f"{base}/VoiceAgent", handler),
    ]
