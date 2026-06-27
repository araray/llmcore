"""``BridgeCore`` — the transport-agnostic adapter (spec §4.2, §17.3).

It owns a single :class:`LLMCoreFacade`, accepts ``llmcore.v1`` request protos,
and returns response protos (or async-iterates ``ChatChunk``). Both transports
(gRPC servicer, HTTP/SSE app) call only into this object, which is why the two
transports are guaranteed to produce identical canonical results.

Hard rule: this is *pure translation* — decode -> call -> encode. No
retrieval/RAG/routing logic. ``enable_rag=False`` is passed on every chat call
to preserve the ecosystem's External-RAG invariant.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Iterable

from google.protobuf import struct_pb2

from ._generated.llmcore.v1 import (
    audio_pb2,
    catalog_pb2,
    common_pb2,
    control_pb2,
    inference_pb2,
)
from .errors import BridgeError, map_exception, unsupported
from .facade import LLMCoreFacade
from .info import audio_capable, build_server_info

__all__ = ["BridgeCore"]


# --------------------------------------------------------------------------- #
# proto <-> python helpers
# --------------------------------------------------------------------------- #
def _opt(msg: Any, field: str) -> Any | None:
    """Return an ``optional`` proto field value, or ``None`` if unset."""
    try:
        return getattr(msg, field) if msg.HasField(field) else None
    except ValueError:
        # Non-optional field: fall back to truthiness of the scalar.
        val = getattr(msg, field)
        return val if val != "" else None


def _struct_to_dict(struct: struct_pb2.Struct) -> dict[str, Any]:
    return {k: _value_to_py(v) for k, v in struct.fields.items()}


def _value_to_py(value: struct_pb2.Value) -> Any:
    kind = value.WhichOneof("kind")
    if kind == "null_value":
        return None
    if kind == "number_value":
        return value.number_value
    if kind == "string_value":
        return value.string_value
    if kind == "bool_value":
        return value.bool_value
    if kind == "struct_value":
        return _struct_to_dict(value.struct_value)
    if kind == "list_value":
        return [_value_to_py(v) for v in value.list_value.values]
    return None


def _usage_to_proto(usage: Any) -> common_pb2.Usage:
    out = common_pb2.Usage()
    if usage is None:
        return out
    pt = getattr(usage, "prompt_tokens", None)
    ct = getattr(usage, "completion_tokens", None)
    tt = getattr(usage, "total_tokens", None)
    provider = getattr(usage, "provider", None)
    model = getattr(usage, "model", None)
    if pt is not None:
        out.prompt_tokens = int(pt)
    if ct is not None:
        out.completion_tokens = int(ct)
    if tt is not None:
        out.total_tokens = int(tt)
    if provider:
        out.provider = str(provider)
    if model:
        out.model = str(model)
    return out


def _cost_to_proto(c: Any) -> inference_pb2.CostEstimate:
    p = inference_pb2.CostEstimate(
        input_cost=c.input_cost,
        output_cost=c.output_cost,
        cached_discount=c.cached_discount,
        reasoning_cost=c.reasoning_cost,
        total_cost=c.total_cost,
        currency=c.currency,
        pricing_source=c.pricing_source,
        prompt_tokens=c.prompt_tokens,
        completion_tokens=c.completion_tokens,
        cached_tokens=c.cached_tokens,
        reasoning_tokens=c.reasoning_tokens,
    )
    if c.input_price_per_million is not None:
        p.input_price_per_million = c.input_price_per_million
    if c.output_price_per_million is not None:
        p.output_price_per_million = c.output_price_per_million
    if c.cached_price_per_million is not None:
        p.cached_price_per_million = c.cached_price_per_million
    if c.model_id is not None:
        p.model_id = c.model_id
    if c.provider is not None:
        p.provider = c.provider
    return p


def _model_details_to_proto(m: Any) -> catalog_pb2.ModelDetails:
    p = catalog_pb2.ModelDetails(
        id=m.id,
        provider_name=m.provider_name,
        context_length=m.context_length,
        supports_streaming=m.supports_streaming,
        supports_tools=m.supports_tools,
        supports_vision=m.supports_vision,
        supports_reasoning=m.supports_reasoning,
    )
    if m.display_name is not None:
        p.display_name = m.display_name
    if m.max_output_tokens is not None:
        p.max_output_tokens = m.max_output_tokens
    if m.family is not None:
        p.family = m.family
    if m.parameter_count is not None:
        p.parameter_count = m.parameter_count
    if m.quantization_level is not None:
        p.quantization_level = m.quantization_level
    if m.file_size_bytes is not None:
        p.file_size_bytes = m.file_size_bytes
    if m.model_type is not None:
        p.model_type = m.model_type
    if getattr(m, "metadata", None):
        try:
            p.metadata.update(m.metadata)
        except (TypeError, ValueError):
            # Metadata held a non-JSON value; skip rather than fail the call.
            pass
    return p


# -- AudioService (Tier 2) helpers ------------------------------------------ #
# models_multimodal.StreamEventType (str-enum) -> audio.proto StreamEventType.
_STREAM_EVENT_TYPE_TO_PROTO: dict[str, int] = {
    "interim": audio_pb2.STREAM_EVENT_TYPE_INTERIM,
    "final": audio_pb2.STREAM_EVENT_TYPE_FINAL,
    "utterance_end": audio_pb2.STREAM_EVENT_TYPE_UTTERANCE_END,
    "speech_started": audio_pb2.STREAM_EVENT_TYPE_SPEECH_STARTED,
    "metadata": audio_pb2.STREAM_EVENT_TYPE_METADATA,
    "start_of_turn": audio_pb2.STREAM_EVENT_TYPE_START_OF_TURN,
    "eager_end_of_turn": audio_pb2.STREAM_EVENT_TYPE_EAGER_END_OF_TURN,
    "turn_resumed": audio_pb2.STREAM_EVENT_TYPE_TURN_RESUMED,
    "end_of_turn": audio_pb2.STREAM_EVENT_TYPE_END_OF_TURN,
    "update": audio_pb2.STREAM_EVENT_TYPE_UPDATE,
    "open": audio_pb2.STREAM_EVENT_TYPE_OPEN,
    "close": audio_pb2.STREAM_EVENT_TYPE_CLOSE,
    "error": audio_pb2.STREAM_EVENT_TYPE_ERROR,
    "other": audio_pb2.STREAM_EVENT_TYPE_OTHER,
}


def _json_safe(value: Any) -> Any:
    """Coerce a value to something ``google.protobuf.Struct.update`` accepts."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _transcription_event_to_proto(ev: Any) -> audio_pb2.TranscriptionStreamEvent:
    """Map a ``models_multimodal.TranscriptionStreamEvent`` to its proto."""
    et = getattr(ev, "type", None)
    et_str = getattr(et, "value", et)  # str-enum -> its string value
    out = audio_pb2.TranscriptionStreamEvent(
        type=_STREAM_EVENT_TYPE_TO_PROTO.get(str(et_str), audio_pb2.STREAM_EVENT_TYPE_OTHER),
        text=getattr(ev, "text", "") or "",
        provider=getattr(ev, "provider", "") or "",
    )
    is_final = getattr(ev, "is_final", None)
    if is_final is not None:
        out.is_final = bool(is_final)
    speech_final = getattr(ev, "speech_final", None)
    if speech_final is not None:
        out.speech_final = bool(speech_final)
    start = getattr(ev, "start", None)
    if start is not None:
        out.start = float(start)
    end = getattr(ev, "end", None)
    if end is not None:
        out.end = float(end)
    confidence = getattr(ev, "confidence", None)
    if confidence is not None:
        out.confidence = float(confidence)
    speaker = getattr(ev, "speaker", None)
    if speaker is not None:
        out.speaker = str(speaker)
    channel_index = getattr(ev, "channel_index", None)
    if channel_index:
        out.channel_index.extend(int(c) for c in channel_index)
    words = getattr(ev, "words", None)
    if words:
        for word in words:
            if isinstance(word, dict):
                out.words.add().update(_json_safe(word))
    raw = getattr(ev, "raw", None)
    if raw:
        try:
            out.raw.update(_json_safe(raw))
        except (TypeError, ValueError):
            pass
    return out


class BridgeCore:
    """Adapter from ``llmcore.v1`` protos to the ``LLMCore`` facade.

    Args:
        facade: Anything satisfying :class:`LLMCoreFacade` (real or fake).
        count_tokens: Optional ``callable(text, model) -> int``. Defaults to
            ``llmcore.count_tokens`` (resolved lazily on first use).
        transports: Enabled transports, advertised in ``ServerInfo``.
    """

    def __init__(
        self,
        facade: LLMCoreFacade,
        *,
        count_tokens: Any | None = None,
        transports: Iterable[str] = ("grpc",),
    ) -> None:
        self._facade = facade
        self._count_tokens = count_tokens
        self._transports = tuple(transports)

    # -- helpers ---------------------------------------------------------- #
    def _tokens(self, text: str, model: str | None) -> int:
        fn = self._count_tokens
        if fn is None:
            from llmcore import count_tokens as fn  # lazy import by design
        return int(fn(text, model))

    def _tools(self, proto_tools: Any) -> list[Any]:
        if not proto_tools:
            return []
        from llmcore.models import Tool  # lazy import by design

        return [
            Tool(
                name=t.name,
                description=t.description,
                parameters=_struct_to_dict(t.parameters),
            )
            for t in proto_tools
        ]

    @staticmethod
    def _kwargs(struct: struct_pb2.Struct) -> dict[str, Any]:
        return _struct_to_dict(struct) if len(struct.fields) else {}

    # -- InferenceService ------------------------------------------------- #
    async def chat(self, req: inference_pb2.ChatRequest) -> inference_pb2.ChatResponse:
        try:
            text, usage = await self._facade.chat_with_usage(
                req.message,
                session_id=_opt(req, "session_id"),
                system_message=_opt(req, "system_message"),
                provider_name=_opt(req, "provider_name"),
                model_name=_opt(req, "model_name"),
                save_session=req.save_session,
                enable_rag=False,
                tools=self._tools(req.tools) or None,
                tool_choice=_opt(req, "tool_choice"),
                **self._kwargs(req.provider_kwargs),
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        resp = inference_pb2.ChatResponse(text=text or "")
        resp.usage.CopyFrom(_usage_to_proto(usage))
        # tool_calls / finish_reason are PROVISIONAL in v1 and left unset.
        return resp

    async def chat_stream(
        self, req: inference_pb2.ChatRequest
    ) -> AsyncIterator[inference_pb2.ChatChunk]:
        try:
            gen = await self._facade.chat(
                req.message,
                session_id=_opt(req, "session_id"),
                system_message=_opt(req, "system_message"),
                provider_name=_opt(req, "provider_name"),
                model_name=_opt(req, "model_name"),
                stream=True,
                save_session=req.save_session,
                enable_rag=False,
                tools=self._tools(req.tools) or None,
                tool_choice=_opt(req, "tool_choice"),
                **self._kwargs(req.provider_kwargs),
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        try:
            async for piece in gen:
                yield inference_pb2.ChatChunk(text=piece or "")
            yield inference_pb2.ChatChunk(done=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise map_exception(exc) from exc

    async def embed(self, req: inference_pb2.EmbedRequest) -> inference_pb2.EmbedResponse:
        raise unsupported(
            "Embed is not available in llmcore.v1: LLMCore exposes no public "
            "embeddings method (provider-level create_embeddings sits behind the "
            "private provider manager). Tracked for a follow-up phase; see "
            "CONTINUATION_GUIDE_B1.md."
        )

    async def count_tokens(
        self, req: inference_pb2.CountTokensRequest
    ) -> inference_pb2.CountTokensResponse:
        try:
            n = self._tokens(req.text, _opt(req, "model_name"))
        except Exception as exc:
            raise map_exception(exc) from exc
        return inference_pb2.CountTokensResponse(tokens=n)

    async def estimate_cost(
        self, req: inference_pb2.EstimateCostRequest
    ) -> inference_pb2.CostEstimate:
        try:
            ce = self._facade.estimate_cost(
                req.provider_name,
                req.model_name,
                req.prompt_tokens,
                req.completion_tokens,
                cached_tokens=req.cached_tokens,
                reasoning_tokens=req.reasoning_tokens,
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        return _cost_to_proto(ce)

    # -- CatalogService --------------------------------------------------- #
    async def list_providers(self, req: common_pb2.Empty) -> catalog_pb2.ListProvidersResponse:
        try:
            providers = self._facade.get_available_providers()
        except Exception as exc:
            raise map_exception(exc) from exc
        return catalog_pb2.ListProvidersResponse(providers=[str(p) for p in providers])

    async def list_models(
        self, req: catalog_pb2.ListModelsRequest
    ) -> catalog_pb2.ListModelsResponse:
        try:
            models = self._facade.get_models_for_provider(req.provider_name)
        except Exception as exc:
            raise map_exception(exc) from exc
        return catalog_pb2.ListModelsResponse(models=[str(m) for m in models])

    async def get_provider_details(
        self, req: catalog_pb2.GetProviderRequest
    ) -> catalog_pb2.ModelDetails:
        try:
            md = self._facade.get_provider_details(_opt(req, "provider_name"))
        except Exception as exc:
            raise map_exception(exc) from exc
        return _model_details_to_proto(md)

    # -- ControlService --------------------------------------------------- #
    async def get_info(self, req: common_pb2.Empty) -> control_pb2.ServerInfo:
        return build_server_info(self._facade, self._transports)

    async def health(self, req: common_pb2.Empty) -> control_pb2.HealthStatus:
        return control_pb2.HealthStatus(ok=True, detail="ok")

    async def reload_config(
        self, req: control_pb2.ReloadConfigRequest
    ) -> control_pb2.ReloadConfigResponse:
        try:
            path = _opt(req, "path")
            if path and hasattr(self._facade, "reload_config_from_file"):
                await self._facade.reload_config_from_file(path)  # type: ignore[attr-defined]
            else:
                await self._facade.reload_config()
        except Exception as exc:
            raise map_exception(exc) from exc
        return control_pb2.ReloadConfigResponse(ok=True)

    # -- AudioService (Tier 2) -------------------------------------------- #
    @property
    def audio_enabled(self) -> bool:
        """Whether Tier-2 audio is enabled for the underlying facade."""
        return audio_capable(self._facade)

    async def transcribe_stream(
        self, request_iter: Any
    ) -> AsyncIterator[audio_pb2.TranscriptionStreamEvent]:
        """Bidi: consume ``AudioIn`` frames, yield ``TranscriptionStreamEvent``.

        The leading ``open`` frame (``OpenStt``) carries model/language and may
        carry ``provider_name`` in its options; subsequent frames are audio
        bytes or control (``CLOSE`` ends the inbound audio). The resolved
        provider's public ``transcribe_stream`` is driven and each event mapped
        to proto. Providers lacking live transcription yield ``UNSUPPORTED``.
        """
        agen = request_iter.__aiter__()

        async def _next() -> Any:
            try:
                return await agen.__anext__()
            except StopAsyncIteration:
                return None

        first = await _next()
        model = language = provider_name = None
        if first is not None and first.WhichOneof("frame") == "open":
            op = first.open
            if op.HasField("model"):
                model = op.model
            if op.HasField("language"):
                language = op.language
            if len(op.options.fields):
                pn = _struct_to_dict(op.options).get("provider_name")
                provider_name = str(pn) if pn else None
            pending = None
        else:
            pending = first  # no leading open frame; first frame was audio/control

        try:
            provider = self._facade.get_provider(provider_name)
        except Exception as exc:
            raise map_exception(exc) from exc
        if not hasattr(provider, "transcribe_stream"):
            raise unsupported(
                "provider '%s' does not support live transcription "
                "(transcribe_stream)" % (provider_name or "default")
            )

        async def _audio() -> AsyncIterator[bytes]:
            frame = pending if pending is not None else await _next()
            while frame is not None:
                which = frame.WhichOneof("frame")
                if which == "audio":
                    if frame.audio:
                        yield frame.audio
                elif which == "control":
                    if frame.control == audio_pb2.STT_CONTROL_CLOSE:
                        return
                    # FINALIZE / KEEPALIVE: no-op for this passthrough adapter.
                # a repeated 'open' frame is ignored
                frame = await _next()

        kwargs: dict[str, Any] = {}
        if model is not None:
            kwargs["model"] = model
        if language is not None:
            kwargs["language"] = language

        try:
            async for ev in provider.transcribe_stream(audio=_audio(), **kwargs):
                yield _transcription_event_to_proto(ev)
        except asyncio.CancelledError:
            raise
        except BridgeError:
            raise
        except Exception as exc:
            raise map_exception(exc) from exc

    async def close(self) -> None:
        """Close the underlying facade (best-effort)."""
        close = getattr(self._facade, "close", None)
        if close is not None:
            try:
                await close()
            except Exception:
                pass
