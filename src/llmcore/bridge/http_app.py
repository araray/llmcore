"""HTTP + SSE transport facade (spec §8) — the *secondary* transport.

Maps the same ``llmcore.v1`` protos to JSON (snake_case, via ``json_format``):
unary RPCs are ``POST /llmcore.v1/<Service>/<Method>``; ``ChatStream`` is the
same POST returning ``text/event-stream`` (one ``data:`` JSON chunk per token,
terminated by ``event: done`` or ``event: error``). Errors render as
``{"error": {...}}`` with the mapped HTTP status. Live duplex audio is a
WebSocket concern (phase B3); the one-shot Audio RPCs return 501 here to mirror
the gRPC UNIMPLEMENTED behaviour.

This module backs onto the identical :class:`BridgeCore` used by gRPC, which is
what the transport-parity test asserts.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable

from google.protobuf import json_format
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from ._generated.llmcore.v1 import (
    catalog_pb2,
    common_pb2,
    control_pb2,
    inference_pb2,
)
from .core import BridgeCore
from .errors import BridgeError, http_status_for, invalid_argument, map_exception

__all__ = ["create_http_app"]

_AUDIO_UNIMPLEMENTED = (
    "AudioService is part of the llmcore.v1 contract but is not implemented in "
    "this release (Tier 2 / phase B3). Live duplex audio uses WebSocket; one-shot "
    "audio is enabled in B3."
)


def _to_dict(msg: Any) -> dict[str, Any]:
    return json_format.MessageToDict(msg, preserving_proto_field_name=True)


def _error_response(err: BridgeError) -> JSONResponse:
    return JSONResponse({"error": _to_dict(err.proto)}, status_code=http_status_for(err.proto))


async def _parse(request: Request, req_type: type) -> Any:
    if req_type is common_pb2.Empty:
        return common_pb2.Empty()
    raw = await request.body()
    if not raw:
        return req_type()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise invalid_argument(f"request body is not valid JSON: {exc}") from exc
    msg = req_type()
    try:
        json_format.ParseDict(data, msg, ignore_unknown_fields=True)
    except json_format.ParseError as exc:
        raise invalid_argument(f"invalid request payload: {exc}") from exc
    return msg


def _make_unary(
    method: Callable[[Any], Awaitable[Any]], req_type: type
) -> Callable[[Request], Awaitable[Response]]:
    async def handler(request: Request) -> Response:
        try:
            req = await _parse(request, req_type)
            resp = await method(req)
        except BridgeError as be:
            return _error_response(be)
        except asyncio.CancelledError:  # pragma: no cover - client disconnect
            raise
        except Exception as exc:
            return _error_response(map_exception(exc))
        return JSONResponse(_to_dict(resp))

    return handler


def _make_chat_stream(core: BridgeCore) -> Callable[[Request], Awaitable[Response]]:
    async def handler(request: Request) -> Response:
        try:
            req = await _parse(request, inference_pb2.ChatRequest)
        except BridgeError as be:
            return _error_response(be)

        async def events():
            try:
                async for chunk in core.chat_stream(req):
                    payload = json.dumps(_to_dict(chunk))
                    event = "done" if chunk.done else "message"
                    if event == "done":
                        yield f"event: done\ndata: {payload}\n\n"
                    else:
                        yield f"data: {payload}\n\n"
            except asyncio.CancelledError:  # pragma: no cover - client disconnect
                raise
            except BridgeError as be:
                yield f"event: error\ndata: {json.dumps(_to_dict(be.proto))}\n\n"
            except Exception as exc:
                be = map_exception(exc)
                yield f"event: error\ndata: {json.dumps(_to_dict(be.proto))}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return handler


def _audio_501(request: Request) -> Response:
    err = invalid_argument(_AUDIO_UNIMPLEMENTED)
    # Force 501 to mirror gRPC UNIMPLEMENTED rather than 400.
    err.proto.http_status = 501
    err.proto.code = "unsupported.capability"
    from ._generated.llmcore.v1 import errors_pb2

    err.proto.category = errors_pb2.ErrorCategory.ERROR_CATEGORY_UNSUPPORTED
    return _error_response(err)


def create_http_app(core: BridgeCore) -> Starlette:
    """Build the Starlette ASGI app exposing the bridge over HTTP/SSE.

    Args:
        core: The shared :class:`BridgeCore`.

    Returns:
        A configured ``starlette.applications.Starlette`` instance.
    """
    P = "/llmcore.v1"
    routes = [
        Route(f"{P}/InferenceService/Chat", _make_unary(core.chat, inference_pb2.ChatRequest), methods=["POST"]),
        Route(f"{P}/InferenceService/ChatStream", _make_chat_stream(core), methods=["POST"]),
        Route(f"{P}/InferenceService/Embed", _make_unary(core.embed, inference_pb2.EmbedRequest), methods=["POST"]),
        Route(f"{P}/InferenceService/CountTokens", _make_unary(core.count_tokens, inference_pb2.CountTokensRequest), methods=["POST"]),
        Route(f"{P}/InferenceService/EstimateCost", _make_unary(core.estimate_cost, inference_pb2.EstimateCostRequest), methods=["POST"]),
        Route(f"{P}/CatalogService/ListProviders", _make_unary(core.list_providers, common_pb2.Empty), methods=["POST"]),
        Route(f"{P}/CatalogService/ListModels", _make_unary(core.list_models, catalog_pb2.ListModelsRequest), methods=["POST"]),
        Route(f"{P}/CatalogService/GetProviderDetails", _make_unary(core.get_provider_details, catalog_pb2.GetProviderRequest), methods=["POST"]),
        Route(f"{P}/ControlService/GetInfo", _make_unary(core.get_info, common_pb2.Empty), methods=["POST"]),
        Route(f"{P}/ControlService/Health", _make_unary(core.health, common_pb2.Empty), methods=["POST"]),
        Route(f"{P}/ControlService/ReloadConfig", _make_unary(core.reload_config, control_pb2.ReloadConfigRequest), methods=["POST"]),
        Route("/healthz", _make_unary(core.health, common_pb2.Empty), methods=["GET", "POST"]),
    ]
    # One-shot AudioService paths mirror gRPC UNIMPLEMENTED (501) until B3.
    for m in ("Synthesize", "Transcribe", "GenerateImage", "Ocr", "AnalyzeText"):
        routes.append(Route(f"{P}/AudioService/{m}", _audio_501, methods=["POST"]))

    return Starlette(routes=routes)
