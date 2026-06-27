"""WebSocket transport for Tier-2 live duplex audio (phase B3).

The HTTP/SSE app (``http_app.py``) covers all Tier-0 RPCs including server
streaming (``ChatStream`` over SSE). Bidirectional audio (``TranscribeStream``,
``SynthesizeStream``, ``VoiceAgent``) needs a *client -> server* channel that
SSE cannot provide; on the HTTP side that maps to WebSocket.

Wire format mirrors the HTTP transport: each WebSocket **text** frame is one
proto rendered as snake_case JSON (``json_format``). The client sends one
request proto per frame and signals end-of-input with an empty object ``{}``
(WebSocket has no half-close, so this in-band sentinel lets the server keep
streaming responses -- e.g. the voice-agent ``CLOSE`` event);
``TranscribeStream`` / ``SynthesizeStream`` additionally honour their in-band
``*_CONTROL_CLOSE`` frames. The server streams response protos as JSON text
frames, and on failure sends one ``{"error": {...}}`` frame then closes with
code ``1011``; on success it closes with ``1000`` when the response stream
ends. When audio is disabled the socket is accepted then closed ``1011``
(mirrors the gRPC ``UNIMPLEMENTED`` / HTTP ``501`` behaviour).

These routes back onto the identical :class:`BridgeCore` used by gRPC and HTTP,
so :meth:`BridgeCore.transcribe_stream` / :meth:`~BridgeCore.synthesize_stream`
/ :meth:`~BridgeCore.voice_agent` are reused verbatim. Mounted by
:func:`llmcore.bridge.http_app.create_http_app`.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

from google.protobuf import json_format
from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from ._generated.llmcore.v1 import audio_pb2
from .core import BridgeCore
from .errors import BridgeError, invalid_argument, map_exception

__all__ = ["create_ws_routes"]

_DISABLED_REASON = (
    "AudioService live duplex (TranscribeStream/SynthesizeStream/VoiceAgent) is "
    "part of the llmcore.v1 contract but is disabled in this deployment "
    "(Tier 2 / audio off)."
)


def _to_dict(msg: Any) -> dict[str, Any]:
    return json_format.MessageToDict(msg, preserving_proto_field_name=True)


async def _ws_request_iter(
    websocket: WebSocket, req_type: type
) -> AsyncIterator[Any]:
    """Yield decoded request protos from inbound WebSocket text frames.

    Ends (``StopAsyncIteration``) on client disconnect or on an empty ``{}``
    frame (the in-band end-of-input sentinel, since WebSocket has no
    half-close). Raises :class:`BridgeError` on a malformed frame so the
    endpoint can surface a structured error.

    Args:
        websocket: The accepted Starlette WebSocket.
        req_type: The protobuf request message class to parse each frame into.

    Yields:
        Parsed ``req_type`` instances, one per inbound JSON frame.
    """
    while True:
        try:
            text = await websocket.receive_text()
        except WebSocketDisconnect:
            return
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise invalid_argument(
                f"WebSocket frame is not valid JSON: {exc}"
            ) from exc
        if not isinstance(obj, dict):
            raise invalid_argument("WebSocket frame must be a JSON object")
        if not obj:
            # {} -> explicit end-of-input sentinel.
            return
        msg = req_type()
        try:
            json_format.ParseDict(obj, msg, ignore_unknown_fields=True)
        except json_format.ParseError as exc:
            raise invalid_argument(
                f"invalid WebSocket request payload: {exc}"
            ) from exc
        yield msg


async def _try_send_json(websocket: WebSocket, payload: dict[str, Any]) -> None:
    """Best-effort JSON send; swallow errors if the socket is already gone."""
    try:
        await websocket.send_text(json.dumps(payload))
    except (WebSocketDisconnect, RuntimeError):  # pragma: no cover - client gone
        pass


async def _try_close(websocket: WebSocket, *, code: int, reason: str) -> None:
    """Best-effort close; swallow errors if the socket is already closing."""
    try:
        await websocket.close(code=code, reason=reason)
    except (WebSocketDisconnect, RuntimeError):  # pragma: no cover - already closed
        pass


def _make_audio_ws(core: BridgeCore, method_name: str, req_type: type):
    """Build a duplex WebSocket endpoint backed by a ``BridgeCore`` streamer.

    Args:
        core: The shared :class:`BridgeCore`.
        method_name: Name of the streaming method on ``core`` to drive
            (``transcribe_stream`` / ``synthesize_stream`` / ``voice_agent``).
        req_type: The protobuf request message class for inbound frames.

    Returns:
        An ``async`` Starlette WebSocket endpoint callable.
    """

    async def endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        if not core.audio_enabled:
            await _try_close(websocket, code=1011, reason=_DISABLED_REASON)
            return

        method = getattr(core, method_name)
        request_iter = _ws_request_iter(websocket, req_type)
        try:
            async for resp in method(request_iter):
                await websocket.send_text(json.dumps(_to_dict(resp)))
        except WebSocketDisconnect:  # pragma: no cover - client closed mid-stream
            return
        except asyncio.CancelledError:  # pragma: no cover - server shutdown
            raise
        except BridgeError as be:
            await _try_send_json(websocket, {"error": _to_dict(be.proto)})
            await _try_close(websocket, code=1011, reason="bridge-error")
            return
        except Exception as exc:
            be = map_exception(exc)
            await _try_send_json(websocket, {"error": _to_dict(be.proto)})
            await _try_close(websocket, code=1011, reason="internal-error")
            return
        finally:
            await request_iter.aclose()
        await _try_close(websocket, code=1000, reason="done")

    return endpoint


def create_ws_routes(core: BridgeCore) -> list[WebSocketRoute]:
    """Return the Tier-2 live-audio WebSocket routes, backed by ``core``.

    Mounted by :func:`llmcore.bridge.http_app.create_http_app`. Each route is
    gated at runtime on ``core.audio_enabled`` (the socket is accepted then
    closed ``1011`` when audio is off), mirroring the gRPC ``UNIMPLEMENTED``
    and HTTP ``501`` behaviour for the one-shot audio RPCs.

    Args:
        core: The shared :class:`BridgeCore`.

    Returns:
        The three ``starlette.routing.WebSocketRoute`` objects for the live
        AudioService RPCs.
    """
    base = "/llmcore.v1/AudioService"
    return [
        WebSocketRoute(
            f"{base}/TranscribeStream",
            _make_audio_ws(core, "transcribe_stream", audio_pb2.AudioIn),
        ),
        WebSocketRoute(
            f"{base}/SynthesizeStream",
            _make_audio_ws(core, "synthesize_stream", audio_pb2.SynthControl),
        ),
        WebSocketRoute(
            f"{base}/VoiceAgent",
            _make_audio_ws(core, "voice_agent", audio_pb2.VoiceAgentClientEvent),
        ),
    ]
