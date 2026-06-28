"""gRPC transport facade (spec §7) — the *primary*, recommended transport.

Servicers are thin: each delegates to :class:`BridgeCore` and renders a
``BridgeError`` as ``context.abort(status, message, trailing_metadata)`` where
the serialized :class:`LlmcoreError` proto is attached under the binary metadata
key ``llmcore-error-bin``. This avoids a hard dependency on ``grpcio-status``
while still giving clients the full structured error. ``CancelledError`` is
propagated untouched so client cancellation maps to gRPC ``CANCELLED``.

AudioService (Tier 2) is registered but returns ``UNIMPLEMENTED`` until B3.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

import grpc

from ._generated.llmcore.v1 import (
    audio_pb2_grpc,
    catalog_pb2_grpc,
    control_pb2_grpc,
    inference_pb2_grpc,
    sessions_pb2_grpc,
    vector_pb2_grpc,
)
from .core import BridgeCore
from .errors import BridgeError, grpc_status_for, map_exception

__all__ = ["ERROR_METADATA_KEY", "create_grpc_server"]

#: Binary trailing-metadata key carrying the serialized ``LlmcoreError``.
ERROR_METADATA_KEY = "llmcore-error-bin"

_AUDIO_UNIMPLEMENTED = (
    "AudioService is part of the llmcore.v1 contract but is not implemented in "
    "this release (Tier 2 / phase B3). Use gRPC once enabled; the capability "
    "flags tier2.* are absent from ServerInfo until then."
)


async def _abort(context: grpc.aio.ServicerContext, err: BridgeError) -> None:
    """Abort the RPC, attaching the structured error as trailing metadata."""
    await context.abort(
        grpc_status_for(err.proto),
        err.proto.message or err.proto.code,
        trailing_metadata=((ERROR_METADATA_KEY, err.proto.SerializeToString()),),
    )


async def _unary(
    call: Callable[[], Awaitable[Any]], context: grpc.aio.ServicerContext
) -> Any:
    """Run a unary BridgeCore coroutine, mapping failures to ``abort``."""
    try:
        return await call()
    except BridgeError as be:
        await _abort(context, be)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        await _abort(context, map_exception(exc))


class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    def __init__(self, core: BridgeCore) -> None:
        self._core = core

    async def Chat(self, request, context):
        return await _unary(lambda: self._core.chat(request), context)

    async def ChatStream(self, request, context):
        try:
            async for chunk in self._core.chat_stream(request):
                yield chunk
        except BridgeError as be:
            await _abort(context, be)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await _abort(context, map_exception(exc))

    async def Embed(self, request, context):
        return await _unary(lambda: self._core.embed(request), context)

    async def CountTokens(self, request, context):
        return await _unary(lambda: self._core.count_tokens(request), context)

    async def EstimateCost(self, request, context):
        return await _unary(lambda: self._core.estimate_cost(request), context)


class CatalogServicer(catalog_pb2_grpc.CatalogServiceServicer):
    def __init__(self, core: BridgeCore) -> None:
        self._core = core

    async def ListProviders(self, request, context):
        return await _unary(lambda: self._core.list_providers(request), context)

    async def ListModels(self, request, context):
        return await _unary(lambda: self._core.list_models(request), context)

    async def GetProviderDetails(self, request, context):
        return await _unary(lambda: self._core.get_provider_details(request), context)


class SessionServicer(sessions_pb2_grpc.SessionServiceServicer):
    """Tier-1 sessions & context items (phase B4)."""

    def __init__(self, core: BridgeCore) -> None:
        self._core = core

    async def CreateSession(self, request, context):
        return await _unary(lambda: self._core.create_session(request), context)

    async def GetSession(self, request, context):
        return await _unary(lambda: self._core.get_session(request), context)

    async def ListSessions(self, request, context):
        return await _unary(lambda: self._core.list_sessions(request), context)

    async def DeleteSession(self, request, context):
        return await _unary(lambda: self._core.delete_session(request), context)

    async def UpdateSessionName(self, request, context):
        return await _unary(lambda: self._core.update_session_name(request), context)

    async def ForkSession(self, request, context):
        return await _unary(lambda: self._core.fork_session(request), context)

    async def CloneSession(self, request, context):
        return await _unary(lambda: self._core.clone_session(request), context)

    async def DeleteMessages(self, request, context):
        return await _unary(lambda: self._core.delete_messages(request), context)

    async def GetMessagesByRange(self, request, context):
        return await _unary(lambda: self._core.get_messages_by_range(request), context)

    async def AddContextItem(self, request, context):
        return await _unary(lambda: self._core.add_context_item(request), context)

    async def GetContextItem(self, request, context):
        return await _unary(lambda: self._core.get_context_item(request), context)

    async def RemoveContextItem(self, request, context):
        return await _unary(lambda: self._core.remove_context_item(request), context)


class VectorServicer(vector_pb2_grpc.VectorServiceServicer):
    """Tier-1 vector store & RAG collections (phase B4)."""

    def __init__(self, core: BridgeCore) -> None:
        self._core = core

    async def AddDocuments(self, request, context):
        return await _unary(lambda: self._core.add_documents(request), context)

    async def SearchVectorStore(self, request, context):
        return await _unary(lambda: self._core.search_vector_store(request), context)

    async def ListVectorCollections(self, request, context):
        return await _unary(lambda: self._core.list_vector_collections(request), context)

    async def ListRagCollections(self, request, context):
        return await _unary(lambda: self._core.list_rag_collections(request), context)

    async def GetRagCollectionInfo(self, request, context):
        return await _unary(lambda: self._core.get_rag_collection_info(request), context)

    async def DeleteRagCollection(self, request, context):
        return await _unary(lambda: self._core.delete_rag_collection(request), context)


class ControlServicer(control_pb2_grpc.ControlServiceServicer):
    def __init__(self, core: BridgeCore) -> None:
        self._core = core

    async def GetInfo(self, request, context):
        return await _unary(lambda: self._core.get_info(request), context)

    async def Health(self, request, context):
        return await _unary(lambda: self._core.health(request), context)

    async def ReloadConfig(self, request, context):
        return await _unary(lambda: self._core.reload_config(request), context)


class AudioServicer(audio_pb2_grpc.AudioServiceServicer):
    """Tier-2 surface — registered for contract completeness; UNIMPLEMENTED (B3)."""

    def __init__(self, core: BridgeCore) -> None:
        self._core = core

    async def Synthesize(self, request, context):
        return await _unary(lambda: self._core.synthesize(request), context)

    async def Transcribe(self, request, context):
        return await _unary(lambda: self._core.transcribe(request), context)

    async def GenerateImage(self, request, context):
        return await _unary(lambda: self._core.generate_image(request), context)

    async def Ocr(self, request, context):
        return await _unary(lambda: self._core.ocr(request), context)

    async def AnalyzeText(self, request, context):
        return await _unary(lambda: self._core.analyze_text(request), context)

    async def TranscribeStream(self, request_iterator, context):
        if not self._core.audio_enabled:
            # Drain inbound frames so the client's sends complete before we
            # close with a terminal status. Otherwise grpc.aio surfaces the
            # client-side send race ("Failed execute_batch: SendMessageOperation")
            # as INTERNAL, masking the real UNIMPLEMENTED. A response-streaming
            # handler that yields nothing also cannot use context.abort() cleanly,
            # so we set the status and end the stream.
            try:
                async for _frame in request_iterator:
                    pass
            except Exception:
                pass
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details(_AUDIO_UNIMPLEMENTED)
            return
        try:
            async for ev in self._core.transcribe_stream(request_iterator):
                yield ev
        except BridgeError as be:
            await _abort(context, be)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await _abort(context, map_exception(exc))

    async def SynthesizeStream(self, request_iterator, context):
        if not self._core.audio_enabled:
            try:
                async for _frame in request_iterator:
                    pass
            except Exception:
                pass
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details(_AUDIO_UNIMPLEMENTED)
            return
        try:
            async for out in self._core.synthesize_stream(request_iterator):
                yield out
        except BridgeError as be:
            await _abort(context, be)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await _abort(context, map_exception(exc))

    async def VoiceAgent(self, request_iterator, context):
        if not self._core.audio_enabled:
            try:
                async for _frame in request_iterator:
                    pass
            except Exception:
                pass
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details(_AUDIO_UNIMPLEMENTED)
            return
        try:
            async for ev in self._core.voice_agent(request_iterator):
                yield ev
        except BridgeError as be:
            await _abort(context, be)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await _abort(context, map_exception(exc))


def create_grpc_server(
    core: BridgeCore,
    *,
    options: list[tuple[str, Any]] | None = None,
    interceptors: list[grpc.aio.ServerInterceptor] | None = None,
) -> grpc.aio.Server:
    """Build a ``grpc.aio.Server`` with all bridge servicers registered.

    The caller binds a port (``add_insecure_port`` / ``add_secure_port``) and
    starts/stops the server. Registering AudioService keeps the served reflection
    surface aligned with the published contract even while it is UNIMPLEMENTED.

    Args:
        core: The shared :class:`BridgeCore`.
        options: Optional gRPC channel/server options.
        interceptors: Optional server interceptors (e.g. an auth gate).

    Returns:
        An unstarted ``grpc.aio.Server``.
    """
    server = grpc.aio.server(options=options or [], interceptors=interceptors or [])
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceServicer(core), server)
    catalog_pb2_grpc.add_CatalogServiceServicer_to_server(CatalogServicer(core), server)
    control_pb2_grpc.add_ControlServiceServicer_to_server(ControlServicer(core), server)
    audio_pb2_grpc.add_AudioServiceServicer_to_server(AudioServicer(core), server)
    sessions_pb2_grpc.add_SessionServiceServicer_to_server(SessionServicer(core), server)
    vector_pb2_grpc.add_VectorServiceServicer_to_server(VectorServicer(core), server)
    return server
