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
    sessions_pb2,
    vector_pb2,
)
from .errors import (
    BridgeError,
    invalid_argument,
    map_exception,
    not_found,
    unsupported,
)
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


# -- SessionService (Tier 1) helpers ---------------------------------------- #
_ROLE_TO_PROTO: dict[str, int] = {
    "system": common_pb2.ROLE_SYSTEM,
    "user": common_pb2.ROLE_USER,
    "assistant": common_pb2.ROLE_ASSISTANT,
    "tool": common_pb2.ROLE_TOOL,
}


def _iso(ts: Any) -> str:
    """Serialize a datetime (or pass through a string) to ISO-8601 UTC."""
    if ts is None:
        return ""
    if isinstance(ts, str):
        return ts
    iso = getattr(ts, "isoformat", None)
    if callable(iso):
        return iso().replace("+00:00", "Z")
    return str(ts)


def _set_struct(target: struct_pb2.Struct, data: Any) -> None:
    """Populate a Struct field from a dict, skipping non-JSON values."""
    if data:
        try:
            target.update(_json_safe(data))
        except (TypeError, ValueError):
            pass


def _message_to_proto(m: Any) -> common_pb2.Message:
    role = getattr(m, "role", None)
    role_str = getattr(role, "value", role)
    out = common_pb2.Message(
        id=getattr(m, "id", "") or "",
        session_id=getattr(m, "session_id", "") or "",
        role=_ROLE_TO_PROTO.get(str(role_str), common_pb2.ROLE_UNSPECIFIED),
        content=getattr(m, "content", "") or "",
        timestamp=_iso(getattr(m, "timestamp", None)),
        tool_call_id=getattr(m, "tool_call_id", "") or "",
        tokens=int(getattr(m, "tokens", 0) or 0),
    )
    _set_struct(out.metadata, getattr(m, "metadata", None))
    return out


def _context_item_to_proto(ci: Any) -> sessions_pb2.ContextItem:
    t = getattr(ci, "type", None)
    t_str = getattr(t, "value", t)
    out = sessions_pb2.ContextItem(
        id=getattr(ci, "id", "") or "",
        type=str(t_str) if t_str is not None else "",
        content=getattr(ci, "content", "") or "",
        is_truncated=bool(getattr(ci, "is_truncated", False)),
        timestamp=_iso(getattr(ci, "timestamp", None)),
    )
    if getattr(ci, "source_id", None) is not None:
        out.source_id = ci.source_id
    if getattr(ci, "tokens", None) is not None:
        out.tokens = ci.tokens
    if getattr(ci, "original_tokens", None) is not None:
        out.original_tokens = ci.original_tokens
    _set_struct(out.metadata, getattr(ci, "metadata", None))
    return out


def _chat_session_to_proto(s: Any) -> sessions_pb2.ChatSession:
    out = sessions_pb2.ChatSession(
        id=getattr(s, "id", "") or "",
        created_at=_iso(getattr(s, "created_at", None)),
        updated_at=_iso(getattr(s, "updated_at", None)),
    )
    if getattr(s, "name", None) is not None:
        out.name = s.name
    for m in getattr(s, "messages", None) or []:
        out.messages.append(_message_to_proto(m))
    for ci in getattr(s, "context_items", None) or []:
        out.context_items.append(_context_item_to_proto(ci))
    _set_struct(out.metadata, getattr(s, "metadata", None))
    return out


def _context_document_to_proto(d: Any) -> vector_pb2.ContextDocument:
    out = vector_pb2.ContextDocument(
        id=getattr(d, "id", "") or "",
        content=getattr(d, "content", "") or "",
    )
    embedding = getattr(d, "embedding", None)
    if embedding:
        out.embedding.extend(float(x) for x in embedding)
    if getattr(d, "score", None) is not None:
        out.score = d.score
    _set_struct(out.metadata, getattr(d, "metadata", None))
    return out


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


# models_multimodal.VoiceAgentEventType -> audio.proto VoiceAgentEventType.
_VOICE_AGENT_EVENT_TYPE_TO_PROTO: dict[str, int] = {
    "welcome": audio_pb2.VOICE_AGENT_EVENT_TYPE_WELCOME,
    "settings_applied": audio_pb2.VOICE_AGENT_EVENT_TYPE_SETTINGS_APPLIED,
    "conversation_text": audio_pb2.VOICE_AGENT_EVENT_TYPE_CONVERSATION_TEXT,
    "user_started_speaking": audio_pb2.VOICE_AGENT_EVENT_TYPE_USER_STARTED_SPEAKING,
    "agent_thinking": audio_pb2.VOICE_AGENT_EVENT_TYPE_AGENT_THINKING,
    "agent_started_speaking": audio_pb2.VOICE_AGENT_EVENT_TYPE_AGENT_STARTED_SPEAKING,
    "agent_audio_done": audio_pb2.VOICE_AGENT_EVENT_TYPE_AGENT_AUDIO_DONE,
    "audio": audio_pb2.VOICE_AGENT_EVENT_TYPE_AUDIO,
    "function_call_request": audio_pb2.VOICE_AGENT_EVENT_TYPE_FUNCTION_CALL_REQUEST,
    "prompt_updated": audio_pb2.VOICE_AGENT_EVENT_TYPE_PROMPT_UPDATED,
    "think_updated": audio_pb2.VOICE_AGENT_EVENT_TYPE_THINK_UPDATED,
    "speak_updated": audio_pb2.VOICE_AGENT_EVENT_TYPE_SPEAK_UPDATED,
    "injection_refused": audio_pb2.VOICE_AGENT_EVENT_TYPE_INJECTION_REFUSED,
    "error": audio_pb2.VOICE_AGENT_EVENT_TYPE_ERROR,
    "warning": audio_pb2.VOICE_AGENT_EVENT_TYPE_WARNING,
    "open": audio_pb2.VOICE_AGENT_EVENT_TYPE_OPEN,
    "close": audio_pb2.VOICE_AGENT_EVENT_TYPE_CLOSE,
    "other": audio_pb2.VOICE_AGENT_EVENT_TYPE_OTHER,
}


def _voice_agent_function_call_to_proto(fc: Any) -> audio_pb2.VoiceAgentFunctionCall:
    """Map a ``models_multimodal.VoiceAgentFunctionCall`` to its proto."""
    out = audio_pb2.VoiceAgentFunctionCall(
        id=getattr(fc, "id", "") or "",
        name=getattr(fc, "name", "") or "",
        client_side=bool(getattr(fc, "client_side", True)),
    )
    arguments = getattr(fc, "arguments", None)
    if arguments:
        try:
            out.arguments.update(_json_safe(arguments))
        except (TypeError, ValueError):
            pass
    raw = getattr(fc, "raw", None)
    if raw:
        try:
            out.raw.update(_json_safe(raw))
        except (TypeError, ValueError):
            pass
    return out


def _voice_agent_event_to_proto(ev: Any) -> audio_pb2.VoiceAgentEvent:
    """Map a ``models_multimodal.VoiceAgentEvent`` to its proto."""
    et = getattr(ev, "type", None)
    et_str = getattr(et, "value", et)
    out = audio_pb2.VoiceAgentEvent(
        type=_VOICE_AGENT_EVENT_TYPE_TO_PROTO.get(
            str(et_str), audio_pb2.VOICE_AGENT_EVENT_TYPE_OTHER
        ),
        provider=getattr(ev, "provider", "") or "",
    )
    role = getattr(ev, "role", None)
    if role is not None:
        out.role = role
    content = getattr(ev, "content", None)
    if content is not None:
        out.content = content
    audio = getattr(ev, "audio", None)
    if audio is not None:
        out.audio = bytes(audio)
    fc = getattr(ev, "function_call", None)
    if fc is not None:
        out.function_call.CopyFrom(_voice_agent_function_call_to_proto(fc))
    raw = getattr(ev, "raw", None)
    if raw:
        try:
            out.raw.update(_json_safe(raw))
        except (TypeError, ValueError):
            pass
    return out


def _struct_update(target: struct_pb2.Struct, value: Any) -> None:
    """Best-effort populate a ``Struct`` field from a Python mapping."""
    if value:
        try:
            target.update(_json_safe(value))
        except (TypeError, ValueError):
            pass


def _repeated_struct(target: Any, items: Any) -> None:
    """Append each mapping in ``items`` to a ``repeated Struct`` proto field."""
    for item in items or []:
        s = target.add()
        if item:
            try:
                s.update(_json_safe(item))
            except (TypeError, ValueError):
                pass


def _value_from_py(value: Any) -> struct_pb2.Value:
    """Convert an arbitrary JSON-able Python value to a protobuf ``Value``."""
    holder = struct_pb2.Struct()
    holder.update({"v": _json_safe(value)})
    return holder.fields["v"]


def _speech_result_to_proto(result: Any) -> audio_pb2.SpeechResult:
    """Map a ``models_multimodal.SpeechResult`` to its proto."""
    out = audio_pb2.SpeechResult(
        audio_data=bytes(getattr(result, "audio_data", b"") or b""),
        format=getattr(result, "format", "") or "",
        model=getattr(result, "model", "") or "",
        voice=getattr(result, "voice", "") or "",
    )
    dur = getattr(result, "duration_seconds", None)
    if dur is not None:
        out.duration_seconds = dur
    _struct_update(out.metadata, getattr(result, "metadata", None))
    return out


def _transcription_segment_to_proto(seg: Any) -> audio_pb2.TranscriptionSegment:
    """Map a ``models_multimodal.TranscriptionSegment`` to its proto."""
    out = audio_pb2.TranscriptionSegment(
        text=getattr(seg, "text", "") or "",
        start=float(getattr(seg, "start", 0.0) or 0.0),
        end=float(getattr(seg, "end", 0.0) or 0.0),
    )
    speaker = getattr(seg, "speaker", None)
    if speaker is not None:
        out.speaker = speaker
    return out


def _transcription_result_to_proto(result: Any) -> audio_pb2.TranscriptionResult:
    """Map a ``models_multimodal.TranscriptionResult`` to its proto."""
    out = audio_pb2.TranscriptionResult(
        text=getattr(result, "text", "") or "",
        model=getattr(result, "model", "") or "",
    )
    language = getattr(result, "language", None)
    if language is not None:
        out.language = language
    dur = getattr(result, "duration_seconds", None)
    if dur is not None:
        out.duration_seconds = dur
    for seg in getattr(result, "segments", None) or []:
        out.segments.append(_transcription_segment_to_proto(seg))
    _struct_update(out.metadata, getattr(result, "metadata", None))
    return out


def _generated_image_to_proto(img: Any) -> audio_pb2.GeneratedImage:
    """Map a ``models_multimodal.GeneratedImage`` to its proto."""
    out = audio_pb2.GeneratedImage(format=getattr(img, "format", "") or "")
    data = getattr(img, "data", None)
    if data is not None:
        out.data = data
    url = getattr(img, "url", None)
    if url is not None:
        out.url = url
    revised = getattr(img, "revised_prompt", None)
    if revised is not None:
        out.revised_prompt = revised
    return out


def _image_result_to_proto(result: Any) -> audio_pb2.ImageGenerationResult:
    """Map a ``models_multimodal.ImageGenerationResult`` to its proto."""
    out = audio_pb2.ImageGenerationResult(model=getattr(result, "model", "") or "")
    for img in getattr(result, "images", None) or []:
        out.images.append(_generated_image_to_proto(img))
    _struct_update(out.metadata, getattr(result, "metadata", None))
    return out


def _ocr_result_to_proto(result: Any) -> audio_pb2.OCRResult:
    """Map a ``models_multimodal.OCRResult`` to its proto."""
    out = audio_pb2.OCRResult(
        model=getattr(result, "model", "") or "",
        pages_processed=int(getattr(result, "pages_processed", 0) or 0),
    )
    _repeated_struct(out.pages, getattr(result, "pages", None))
    annotation = getattr(result, "document_annotation", None)
    if annotation is not None:
        out.document_annotation.CopyFrom(_value_from_py(annotation))
    size = getattr(result, "doc_size_bytes", None)
    if size is not None:
        out.doc_size_bytes = int(size)
    _struct_update(out.metadata, getattr(result, "metadata", None))
    return out


def _text_analysis_result_to_proto(result: Any) -> audio_pb2.TextAnalysisResult:
    """Map a ``models_multimodal.TextAnalysisResult`` to its proto."""
    out = audio_pb2.TextAnalysisResult()
    summary = getattr(result, "summary", None)
    if summary is not None:
        out.summary = summary
    _repeated_struct(out.topics, getattr(result, "topics", None))
    _repeated_struct(out.intents, getattr(result, "intents", None))
    _struct_update(out.sentiments, getattr(result, "sentiments", None))
    language = getattr(result, "language", None)
    if language is not None:
        out.language = language
    model = getattr(result, "model", None)
    if model is not None:
        out.model = model
    request_id = getattr(result, "request_id", None)
    if request_id is not None:
        out.request_id = request_id
    _struct_update(out.metadata, getattr(result, "metadata", None))
    _struct_update(out.raw, getattr(result, "raw", None))
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

    # -- SessionService (Tier 1) ------------------------------------------ #
    async def create_session(
        self, req: sessions_pb2.CreateSessionRequest
    ) -> sessions_pb2.ChatSession:
        try:
            s = await self._facade.create_session(
                session_id=_opt(req, "session_id"),
                name=_opt(req, "name"),
                system_message=_opt(req, "system_message"),
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        return _chat_session_to_proto(s)

    async def get_session(
        self, req: sessions_pb2.GetSessionRequest
    ) -> sessions_pb2.ChatSession:
        try:
            s = await self._facade.get_session(req.session_id)
        except Exception as exc:
            raise map_exception(exc) from exc
        return _chat_session_to_proto(s)

    async def list_sessions(
        self, req: sessions_pb2.ListSessionsRequest
    ) -> sessions_pb2.ListSessionsResponse:
        try:
            sessions = await self._facade.list_sessions(limit=_opt(req, "limit"))
        except Exception as exc:
            raise map_exception(exc) from exc
        out = sessions_pb2.ListSessionsResponse()
        for s in sessions or []:
            out.sessions.append(_chat_session_to_proto(s))
        return out

    async def delete_session(
        self, req: sessions_pb2.DeleteSessionRequest
    ) -> common_pb2.Empty:
        try:
            await self._facade.delete_session(req.session_id)
        except Exception as exc:
            raise map_exception(exc) from exc
        return common_pb2.Empty()

    async def update_session_name(
        self, req: sessions_pb2.UpdateSessionNameRequest
    ) -> common_pb2.Empty:
        try:
            await self._facade.update_session_name(req.session_id, req.new_name)
        except Exception as exc:
            raise map_exception(exc) from exc
        return common_pb2.Empty()

    async def fork_session(
        self, req: sessions_pb2.ForkSessionRequest
    ) -> sessions_pb2.ForkSessionResponse:
        message_range = None
        if req.HasField("message_range"):
            message_range = (req.message_range.start, req.message_range.end)
        try:
            new_id = await self._facade.fork_session(
                req.session_id,
                new_name=_opt(req, "new_name"),
                from_message_id=_opt(req, "from_message_id"),
                message_ids=list(req.message_ids) or None,
                message_range=message_range,
                include_context_items=req.include_context_items,
                metadata=self._kwargs(req.metadata) or None,
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        return sessions_pb2.ForkSessionResponse(session_id=new_id)

    async def clone_session(
        self, req: sessions_pb2.CloneSessionRequest
    ) -> sessions_pb2.CloneSessionResponse:
        try:
            new_id = await self._facade.clone_session(
                req.session_id,
                new_name=_opt(req, "new_name"),
                include_messages=req.include_messages,
                include_context_items=req.include_context_items,
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        return sessions_pb2.CloneSessionResponse(session_id=new_id)

    async def delete_messages(
        self, req: sessions_pb2.DeleteMessagesRequest
    ) -> sessions_pb2.DeleteMessagesResponse:
        try:
            n = await self._facade.delete_messages(req.session_id, list(req.message_ids))
        except Exception as exc:
            raise map_exception(exc) from exc
        return sessions_pb2.DeleteMessagesResponse(deleted_count=int(n))

    async def get_messages_by_range(
        self, req: sessions_pb2.GetMessagesByRangeRequest
    ) -> sessions_pb2.GetMessagesByRangeResponse:
        try:
            msgs = await self._facade.get_messages_by_range(
                req.session_id, req.start_index, req.end_index
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        out = sessions_pb2.GetMessagesByRangeResponse()
        for m in msgs or []:
            out.messages.append(_message_to_proto(m))
        return out

    async def add_context_item(
        self, req: sessions_pb2.AddContextItemRequest
    ) -> sessions_pb2.AddContextItemResponse:
        from llmcore.models import ContextItemType  # lazy import by design

        type_value = _opt(req, "type") or "user_text"
        try:
            item_type = ContextItemType(type_value)
        except ValueError as exc:
            raise invalid_argument(f"unknown context item type: {type_value!r}") from exc
        try:
            item_id = await self._facade.add_context_item(
                req.session_id,
                req.content,
                item_type=item_type,
                source_id=_opt(req, "source_id"),
                metadata=self._kwargs(req.metadata) or None,
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        return sessions_pb2.AddContextItemResponse(item_id=item_id)

    async def get_context_item(
        self, req: sessions_pb2.GetContextItemRequest
    ) -> sessions_pb2.ContextItem:
        try:
            ci = await self._facade.get_context_item(req.session_id, req.item_id)
        except Exception as exc:
            raise map_exception(exc) from exc
        if ci is None:
            raise not_found(
                f"context item {req.item_id!r} not found in session {req.session_id!r}",
                code="not_found.context_item",
            )
        return _context_item_to_proto(ci)

    async def remove_context_item(
        self, req: sessions_pb2.RemoveContextItemRequest
    ) -> sessions_pb2.RemoveContextItemResponse:
        try:
            removed = await self._facade.remove_context_item(req.session_id, req.item_id)
        except Exception as exc:
            raise map_exception(exc) from exc
        return sessions_pb2.RemoveContextItemResponse(removed=bool(removed))

    # -- VectorService (Tier 1) ------------------------------------------ #
    async def add_documents(
        self, req: vector_pb2.AddDocumentsRequest
    ) -> vector_pb2.AddDocumentsResponse:
        documents = [_struct_to_dict(d) for d in req.documents]
        try:
            ids = await self._facade.add_documents_to_vector_store(
                documents, collection_name=_opt(req, "collection_name")
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        return vector_pb2.AddDocumentsResponse(ids=[str(i) for i in ids or []])

    async def search_vector_store(
        self, req: vector_pb2.SearchVectorStoreRequest
    ) -> vector_pb2.SearchVectorStoreResponse:
        try:
            docs = await self._facade.search_vector_store(
                req.query,
                k=req.k or 5,
                collection_name=_opt(req, "collection_name"),
                metadata_filter=self._kwargs(req.metadata_filter) or None,
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        out = vector_pb2.SearchVectorStoreResponse()
        for d in docs or []:
            out.documents.append(_context_document_to_proto(d))
        return out

    async def list_vector_collections(
        self, req: common_pb2.Empty
    ) -> vector_pb2.ListCollectionsResponse:
        try:
            collections = await self._facade.list_vector_collections()
        except Exception as exc:
            raise map_exception(exc) from exc
        return vector_pb2.ListCollectionsResponse(collections=[str(c) for c in collections or []])

    async def list_rag_collections(
        self, req: common_pb2.Empty
    ) -> vector_pb2.ListCollectionsResponse:
        try:
            collections = await self._facade.list_rag_collections()
        except Exception as exc:
            raise map_exception(exc) from exc
        return vector_pb2.ListCollectionsResponse(collections=[str(c) for c in collections or []])

    async def get_rag_collection_info(
        self, req: vector_pb2.GetRagCollectionInfoRequest
    ) -> vector_pb2.RagCollectionInfo:
        try:
            info = await self._facade.get_rag_collection_info(req.collection_name)
        except Exception as exc:
            raise map_exception(exc) from exc
        if info is None:
            raise not_found(
                f"rag collection {req.collection_name!r} not found",
                code="not_found.rag_collection",
            )
        out = vector_pb2.RagCollectionInfo(collection_name=req.collection_name)
        _set_struct(out.info, info)
        return out

    async def delete_rag_collection(
        self, req: vector_pb2.DeleteRagCollectionRequest
    ) -> vector_pb2.DeleteRagCollectionResponse:
        try:
            deleted = await self._facade.delete_rag_collection(
                req.collection_name, force=req.force
            )
        except Exception as exc:
            raise map_exception(exc) from exc
        return vector_pb2.DeleteRagCollectionResponse(deleted=bool(deleted))

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

    async def synthesize_stream(
        self, request_iter: Any
    ) -> AsyncIterator[audio_pb2.AudioOut]:
        """Bidi: consume ``SynthControl`` frames, yield ``AudioOut`` chunks.

        The leading ``open`` frame (``OpenTts``) carries voice/model/format;
        subsequent ``text`` frames are streamed into the provider's
        ``stream_speech`` (WebSocket incremental mode) and each produced audio
        chunk is emitted as ``AudioOut(audio=..., seq=i)``. ``CLOSE`` ends the
        inbound text. Providers lacking streaming TTS yield ``UNSUPPORTED``.
        """
        agen = request_iter.__aiter__()

        async def _next() -> Any:
            try:
                return await agen.__anext__()
            except StopAsyncIteration:
                return None

        first = await _next()
        model = response_format = provider_name = None
        if first is not None and first.WhichOneof("frame") == "open":
            op = first.open
            if op.HasField("model"):
                model = op.model
            elif op.HasField("voice"):
                model = op.voice
            if op.HasField("format"):
                response_format = op.format
            pending = None
        else:
            pending = first  # no leading open frame; first frame was text/control

        try:
            provider = self._facade.get_provider(provider_name)
        except Exception as exc:
            raise map_exception(exc) from exc
        if not hasattr(provider, "stream_speech"):
            raise unsupported(
                "provider '%s' does not support streaming synthesis "
                "(stream_speech)" % (provider_name or "default")
            )

        async def _text() -> AsyncIterator[str]:
            frame = pending if pending is not None else await _next()
            while frame is not None:
                which = frame.WhichOneof("frame")
                if which == "text":
                    if frame.text:
                        yield frame.text
                elif which == "control":
                    if frame.control == audio_pb2.TTS_CONTROL_CLOSE:
                        return
                    # FLUSH / CLEAR: no-op for this passthrough adapter.
                frame = await _next()

        kwargs: dict[str, Any] = {}
        if model is not None:
            kwargs["model"] = model
        if response_format is not None:
            kwargs["response_format"] = response_format

        seq = 0
        try:
            async for chunk in provider.stream_speech(_text(), **kwargs):
                if not chunk:
                    continue
                out = audio_pb2.AudioOut(audio=bytes(chunk), seq=seq)
                seq += 1
                yield out
        except asyncio.CancelledError:
            raise
        except BridgeError:
            raise
        except Exception as exc:
            raise map_exception(exc) from exc

    async def voice_agent(
        self, request_iter: Any
    ) -> AsyncIterator[audio_pb2.VoiceAgentEvent]:
        """Duplex: bridge ``VoiceAgentClientEvent`` frames to a provider voice
        agent session and stream ``VoiceAgentEvent`` back.

        A leading ``settings`` frame (if present) opens the session; its optional
        ``provider_name`` selects the provider. A background pump dispatches
        inbound frames to the session (audio / inject / update / respond /
        keepalive) while the session's events are mapped to proto and yielded.
        Providers without ``open_voice_agent`` yield ``UNSUPPORTED``.
        """
        agen = request_iter.__aiter__()

        async def _next() -> Any:
            try:
                return await agen.__anext__()
            except StopAsyncIteration:
                return None

        first = await _next()
        open_settings: dict[str, Any] | None = None
        provider_name = None
        pending = first
        if first is not None and first.WhichOneof("event") == "settings":
            open_settings = _struct_to_dict(first.settings)
            pn = open_settings.pop("provider_name", None)
            provider_name = str(pn) if pn else None
            pending = None

        try:
            provider = self._facade.get_provider(provider_name)
        except Exception as exc:
            raise map_exception(exc) from exc
        if not hasattr(provider, "open_voice_agent"):
            raise unsupported(
                "provider '%s' does not support voice agent "
                "(open_voice_agent)" % (provider_name or "default")
            )

        cm = provider.open_voice_agent(settings=open_settings or None)
        session = await cm.__aenter__()

        async def _pump() -> None:
            try:
                frame = pending if pending is not None else await _next()
                while frame is not None:
                    which = frame.WhichOneof("event")
                    if which == "audio":
                        if frame.audio:
                            await session.send_audio(frame.audio)
                    elif which == "inject_user_message":
                        await session.inject_user_message(frame.inject_user_message)
                    elif which == "inject_agent_message":
                        await session.inject_agent_message(frame.inject_agent_message)
                    elif which == "update_prompt":
                        await session.update_prompt(frame.update_prompt)
                    elif which == "update_think":
                        if hasattr(session, "update_think"):
                            await session.update_think(_struct_to_dict(frame.update_think))
                    elif which == "update_speak":
                        if hasattr(session, "update_speak"):
                            await session.update_speak(_struct_to_dict(frame.update_speak))
                    elif which == "respond_to_function_call":
                        fcr = frame.respond_to_function_call
                        await session.respond_to_function_call(fcr.id, "", fcr.output)
                    elif which == "keepalive":
                        if frame.keepalive and hasattr(session, "keepalive"):
                            await session.keepalive()
                    # a mid-stream 'settings' frame has no session method; ignore.
                    frame = await _next()
            finally:
                # Deterministic fakes expose close() to end their event stream;
                # real provider sessions have no close frame (teardown on exit).
                close = getattr(session, "close", None)
                if close is not None:
                    try:
                        close()
                    except Exception:
                        pass

        pump_task = asyncio.ensure_future(_pump())
        try:
            async for ev in session:
                yield _voice_agent_event_to_proto(ev)
        except asyncio.CancelledError:
            raise
        except BridgeError:
            raise
        except Exception as exc:
            raise map_exception(exc) from exc
        finally:
            if not pump_task.done():
                pump_task.cancel()
                try:
                    await pump_task
                except asyncio.CancelledError:
                    pass
            try:
                await cm.__aexit__(None, None, None)
            except Exception:
                pass

    # -- AudioService: one-shot (unary) RPCs ------------------------------ #
    def _audio_provider(
        self, provider_name: str | None, method: str, label: str
    ) -> Any:
        """Resolve the provider for an audio RPC.

        Gates on ``audio_enabled`` (raising ``UNSUPPORTED`` -> gRPC
        UNIMPLEMENTED / HTTP 501 when audio is off) and feature-detects the
        required ``BaseProvider`` method.
        """
        if not self.audio_enabled:
            raise unsupported(
                "AudioService is disabled in this deployment (Tier 2 / audio "
                "off); enable audio support to use audio RPCs."
            )
        try:
            provider = self._facade.get_provider(provider_name)
        except Exception as exc:
            raise map_exception(exc) from exc
        if not hasattr(provider, method):
            raise unsupported(
                "provider '%s' does not support %s (%s)"
                % (provider_name or "default", label, method)
            )
        return provider

    async def synthesize(self, request: Any) -> audio_pb2.SpeechResult:
        """Unary text-to-speech: ``generate_speech`` -> ``SpeechResult``."""
        provider = self._audio_provider(
            _opt(request, "provider_name"), "generate_speech", "synthesis"
        )
        kwargs: dict[str, Any] = {}
        for fld in ("voice", "model", "response_format", "speed"):
            val = _opt(request, fld)
            if val is not None:
                kwargs[fld] = val
        try:
            result = await provider.generate_speech(request.text, **kwargs)
        except BridgeError:
            raise
        except Exception as exc:
            raise map_exception(exc) from exc
        return _speech_result_to_proto(result)

    async def transcribe(self, request: Any) -> audio_pb2.TranscriptionResult:
        """Unary speech-to-text: ``transcribe_audio`` -> ``TranscriptionResult``."""
        provider = self._audio_provider(
            _opt(request, "provider_name"), "transcribe_audio", "transcription"
        )
        kwargs: dict[str, Any] = {}
        for fld in ("model", "language", "response_format"):
            val = _opt(request, fld)
            if val is not None:
                kwargs[fld] = val
        try:
            result = await provider.transcribe_audio(request.audio_data, **kwargs)
        except BridgeError:
            raise
        except Exception as exc:
            raise map_exception(exc) from exc
        return _transcription_result_to_proto(result)

    async def generate_image(self, request: Any) -> audio_pb2.ImageGenerationResult:
        """Unary image generation: ``generate_image`` -> ``ImageGenerationResult``."""
        provider = self._audio_provider(
            _opt(request, "provider_name"), "generate_image", "image generation"
        )
        kwargs: dict[str, Any] = {"n": request.n or 1}
        for fld in ("model", "size", "quality"):
            val = _opt(request, fld)
            if val is not None:
                kwargs[fld] = val
        try:
            result = await provider.generate_image(request.prompt, **kwargs)
        except BridgeError:
            raise
        except Exception as exc:
            raise map_exception(exc) from exc
        return _image_result_to_proto(result)

    async def ocr(self, request: Any) -> audio_pb2.OCRResult:
        """Unary OCR: ``ocr`` -> ``OCRResult``."""
        provider = self._audio_provider(_opt(request, "provider_name"), "ocr", "OCR")
        which = request.WhichOneof("source")
        if which == "url":
            document: Any = request.url
        elif which == "data":
            document = request.data
        else:
            raise invalid_argument("OcrRequest requires a 'url' or 'data' source")
        kwargs: dict[str, Any] = {}
        model = _opt(request, "model")
        if model is not None:
            kwargs["model"] = model
        try:
            result = await provider.ocr(document, **kwargs)
        except BridgeError:
            raise
        except Exception as exc:
            raise map_exception(exc) from exc
        return _ocr_result_to_proto(result)

    async def analyze_text(self, request: Any) -> audio_pb2.TextAnalysisResult:
        """Unary text analysis: ``analyze_text`` -> ``TextAnalysisResult``.

        The proto ``features`` Struct is unpacked into ``analyze_text``'s flags
        (``summarize`` / ``topics`` / ``sentiment`` / ``intents`` /
        ``language``); other keys pass through as kwargs. ``model`` is accepted
        in the proto for forward-compatibility but not forwarded (the Deepgram
        ``read`` endpoint takes no model).
        """
        provider = self._audio_provider(
            _opt(request, "provider_name"), "analyze_text", "text analysis"
        )
        kwargs: dict[str, Any] = {}
        if len(request.features.fields):
            kwargs.update(_struct_to_dict(request.features))
        try:
            result = await provider.analyze_text(request.text, **kwargs)
        except BridgeError:
            raise
        except Exception as exc:
            raise map_exception(exc) from exc
        return _text_analysis_result_to_proto(result)

    async def close(self) -> None:
        """Close the underlying facade (best-effort)."""
        close = getattr(self._facade, "close", None)
        if close is not None:
            try:
                await close()
            except Exception:
                pass
