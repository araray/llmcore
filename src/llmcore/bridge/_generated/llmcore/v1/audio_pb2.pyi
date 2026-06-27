from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class StreamEventType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    STREAM_EVENT_TYPE_UNSPECIFIED: _ClassVar[StreamEventType]
    STREAM_EVENT_TYPE_INTERIM: _ClassVar[StreamEventType]
    STREAM_EVENT_TYPE_FINAL: _ClassVar[StreamEventType]
    STREAM_EVENT_TYPE_UTTERANCE_END: _ClassVar[StreamEventType]
    STREAM_EVENT_TYPE_SPEECH_STARTED: _ClassVar[StreamEventType]
    STREAM_EVENT_TYPE_METADATA: _ClassVar[StreamEventType]
    STREAM_EVENT_TYPE_START_OF_TURN: _ClassVar[StreamEventType]
    STREAM_EVENT_TYPE_EAGER_END_OF_TURN: _ClassVar[StreamEventType]
    STREAM_EVENT_TYPE_TURN_RESUMED: _ClassVar[StreamEventType]
    STREAM_EVENT_TYPE_END_OF_TURN: _ClassVar[StreamEventType]
    STREAM_EVENT_TYPE_UPDATE: _ClassVar[StreamEventType]
    STREAM_EVENT_TYPE_OPEN: _ClassVar[StreamEventType]
    STREAM_EVENT_TYPE_CLOSE: _ClassVar[StreamEventType]
    STREAM_EVENT_TYPE_ERROR: _ClassVar[StreamEventType]
    STREAM_EVENT_TYPE_OTHER: _ClassVar[StreamEventType]

class VoiceAgentEventType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    VOICE_AGENT_EVENT_TYPE_UNSPECIFIED: _ClassVar[VoiceAgentEventType]
    VOICE_AGENT_EVENT_TYPE_WELCOME: _ClassVar[VoiceAgentEventType]
    VOICE_AGENT_EVENT_TYPE_SETTINGS_APPLIED: _ClassVar[VoiceAgentEventType]
    VOICE_AGENT_EVENT_TYPE_CONVERSATION_TEXT: _ClassVar[VoiceAgentEventType]
    VOICE_AGENT_EVENT_TYPE_USER_STARTED_SPEAKING: _ClassVar[VoiceAgentEventType]
    VOICE_AGENT_EVENT_TYPE_AGENT_THINKING: _ClassVar[VoiceAgentEventType]
    VOICE_AGENT_EVENT_TYPE_AGENT_STARTED_SPEAKING: _ClassVar[VoiceAgentEventType]
    VOICE_AGENT_EVENT_TYPE_AGENT_AUDIO_DONE: _ClassVar[VoiceAgentEventType]
    VOICE_AGENT_EVENT_TYPE_AUDIO: _ClassVar[VoiceAgentEventType]
    VOICE_AGENT_EVENT_TYPE_FUNCTION_CALL_REQUEST: _ClassVar[VoiceAgentEventType]
    VOICE_AGENT_EVENT_TYPE_PROMPT_UPDATED: _ClassVar[VoiceAgentEventType]
    VOICE_AGENT_EVENT_TYPE_THINK_UPDATED: _ClassVar[VoiceAgentEventType]
    VOICE_AGENT_EVENT_TYPE_SPEAK_UPDATED: _ClassVar[VoiceAgentEventType]
    VOICE_AGENT_EVENT_TYPE_INJECTION_REFUSED: _ClassVar[VoiceAgentEventType]
    VOICE_AGENT_EVENT_TYPE_ERROR: _ClassVar[VoiceAgentEventType]
    VOICE_AGENT_EVENT_TYPE_WARNING: _ClassVar[VoiceAgentEventType]
    VOICE_AGENT_EVENT_TYPE_OPEN: _ClassVar[VoiceAgentEventType]
    VOICE_AGENT_EVENT_TYPE_CLOSE: _ClassVar[VoiceAgentEventType]
    VOICE_AGENT_EVENT_TYPE_OTHER: _ClassVar[VoiceAgentEventType]

class SttControl(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    STT_CONTROL_UNSPECIFIED: _ClassVar[SttControl]
    STT_CONTROL_FINALIZE: _ClassVar[SttControl]
    STT_CONTROL_KEEPALIVE: _ClassVar[SttControl]
    STT_CONTROL_CLOSE: _ClassVar[SttControl]

class TtsControl(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TTS_CONTROL_UNSPECIFIED: _ClassVar[TtsControl]
    TTS_CONTROL_FLUSH: _ClassVar[TtsControl]
    TTS_CONTROL_CLEAR: _ClassVar[TtsControl]
    TTS_CONTROL_CLOSE: _ClassVar[TtsControl]
STREAM_EVENT_TYPE_UNSPECIFIED: StreamEventType
STREAM_EVENT_TYPE_INTERIM: StreamEventType
STREAM_EVENT_TYPE_FINAL: StreamEventType
STREAM_EVENT_TYPE_UTTERANCE_END: StreamEventType
STREAM_EVENT_TYPE_SPEECH_STARTED: StreamEventType
STREAM_EVENT_TYPE_METADATA: StreamEventType
STREAM_EVENT_TYPE_START_OF_TURN: StreamEventType
STREAM_EVENT_TYPE_EAGER_END_OF_TURN: StreamEventType
STREAM_EVENT_TYPE_TURN_RESUMED: StreamEventType
STREAM_EVENT_TYPE_END_OF_TURN: StreamEventType
STREAM_EVENT_TYPE_UPDATE: StreamEventType
STREAM_EVENT_TYPE_OPEN: StreamEventType
STREAM_EVENT_TYPE_CLOSE: StreamEventType
STREAM_EVENT_TYPE_ERROR: StreamEventType
STREAM_EVENT_TYPE_OTHER: StreamEventType
VOICE_AGENT_EVENT_TYPE_UNSPECIFIED: VoiceAgentEventType
VOICE_AGENT_EVENT_TYPE_WELCOME: VoiceAgentEventType
VOICE_AGENT_EVENT_TYPE_SETTINGS_APPLIED: VoiceAgentEventType
VOICE_AGENT_EVENT_TYPE_CONVERSATION_TEXT: VoiceAgentEventType
VOICE_AGENT_EVENT_TYPE_USER_STARTED_SPEAKING: VoiceAgentEventType
VOICE_AGENT_EVENT_TYPE_AGENT_THINKING: VoiceAgentEventType
VOICE_AGENT_EVENT_TYPE_AGENT_STARTED_SPEAKING: VoiceAgentEventType
VOICE_AGENT_EVENT_TYPE_AGENT_AUDIO_DONE: VoiceAgentEventType
VOICE_AGENT_EVENT_TYPE_AUDIO: VoiceAgentEventType
VOICE_AGENT_EVENT_TYPE_FUNCTION_CALL_REQUEST: VoiceAgentEventType
VOICE_AGENT_EVENT_TYPE_PROMPT_UPDATED: VoiceAgentEventType
VOICE_AGENT_EVENT_TYPE_THINK_UPDATED: VoiceAgentEventType
VOICE_AGENT_EVENT_TYPE_SPEAK_UPDATED: VoiceAgentEventType
VOICE_AGENT_EVENT_TYPE_INJECTION_REFUSED: VoiceAgentEventType
VOICE_AGENT_EVENT_TYPE_ERROR: VoiceAgentEventType
VOICE_AGENT_EVENT_TYPE_WARNING: VoiceAgentEventType
VOICE_AGENT_EVENT_TYPE_OPEN: VoiceAgentEventType
VOICE_AGENT_EVENT_TYPE_CLOSE: VoiceAgentEventType
VOICE_AGENT_EVENT_TYPE_OTHER: VoiceAgentEventType
STT_CONTROL_UNSPECIFIED: SttControl
STT_CONTROL_FINALIZE: SttControl
STT_CONTROL_KEEPALIVE: SttControl
STT_CONTROL_CLOSE: SttControl
TTS_CONTROL_UNSPECIFIED: TtsControl
TTS_CONTROL_FLUSH: TtsControl
TTS_CONTROL_CLEAR: TtsControl
TTS_CONTROL_CLOSE: TtsControl

class SpeechResult(_message.Message):
    __slots__ = ("audio_data", "format", "model", "voice", "duration_seconds", "metadata")
    AUDIO_DATA_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    VOICE_FIELD_NUMBER: _ClassVar[int]
    DURATION_SECONDS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    audio_data: bytes
    format: str
    model: str
    voice: str
    duration_seconds: float
    metadata: _struct_pb2.Struct
    def __init__(self, audio_data: _Optional[bytes] = ..., format: _Optional[str] = ..., model: _Optional[str] = ..., voice: _Optional[str] = ..., duration_seconds: _Optional[float] = ..., metadata: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class TranscriptionSegment(_message.Message):
    __slots__ = ("text", "start", "end", "speaker")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    START_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    SPEAKER_FIELD_NUMBER: _ClassVar[int]
    text: str
    start: float
    end: float
    speaker: str
    def __init__(self, text: _Optional[str] = ..., start: _Optional[float] = ..., end: _Optional[float] = ..., speaker: _Optional[str] = ...) -> None: ...

class TranscriptionResult(_message.Message):
    __slots__ = ("text", "language", "duration_seconds", "segments", "model", "metadata")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    DURATION_SECONDS_FIELD_NUMBER: _ClassVar[int]
    SEGMENTS_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    text: str
    language: str
    duration_seconds: float
    segments: _containers.RepeatedCompositeFieldContainer[TranscriptionSegment]
    model: str
    metadata: _struct_pb2.Struct
    def __init__(self, text: _Optional[str] = ..., language: _Optional[str] = ..., duration_seconds: _Optional[float] = ..., segments: _Optional[_Iterable[_Union[TranscriptionSegment, _Mapping]]] = ..., model: _Optional[str] = ..., metadata: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class GeneratedImage(_message.Message):
    __slots__ = ("data", "url", "revised_prompt", "format")
    DATA_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    REVISED_PROMPT_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    data: str
    url: str
    revised_prompt: str
    format: str
    def __init__(self, data: _Optional[str] = ..., url: _Optional[str] = ..., revised_prompt: _Optional[str] = ..., format: _Optional[str] = ...) -> None: ...

class ImageGenerationResult(_message.Message):
    __slots__ = ("images", "model", "metadata")
    IMAGES_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    images: _containers.RepeatedCompositeFieldContainer[GeneratedImage]
    model: str
    metadata: _struct_pb2.Struct
    def __init__(self, images: _Optional[_Iterable[_Union[GeneratedImage, _Mapping]]] = ..., model: _Optional[str] = ..., metadata: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class OCRResult(_message.Message):
    __slots__ = ("pages", "model", "document_annotation", "pages_processed", "doc_size_bytes", "metadata")
    PAGES_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    DOCUMENT_ANNOTATION_FIELD_NUMBER: _ClassVar[int]
    PAGES_PROCESSED_FIELD_NUMBER: _ClassVar[int]
    DOC_SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    pages: _containers.RepeatedCompositeFieldContainer[_struct_pb2.Struct]
    model: str
    document_annotation: _struct_pb2.Value
    pages_processed: int
    doc_size_bytes: int
    metadata: _struct_pb2.Struct
    def __init__(self, pages: _Optional[_Iterable[_Union[_struct_pb2.Struct, _Mapping]]] = ..., model: _Optional[str] = ..., document_annotation: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ..., pages_processed: _Optional[int] = ..., doc_size_bytes: _Optional[int] = ..., metadata: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class TextAnalysisResult(_message.Message):
    __slots__ = ("summary", "topics", "intents", "sentiments", "language", "model", "request_id", "metadata", "raw")
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    TOPICS_FIELD_NUMBER: _ClassVar[int]
    INTENTS_FIELD_NUMBER: _ClassVar[int]
    SENTIMENTS_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    RAW_FIELD_NUMBER: _ClassVar[int]
    summary: str
    topics: _containers.RepeatedCompositeFieldContainer[_struct_pb2.Struct]
    intents: _containers.RepeatedCompositeFieldContainer[_struct_pb2.Struct]
    sentiments: _struct_pb2.Struct
    language: str
    model: str
    request_id: str
    metadata: _struct_pb2.Struct
    raw: _struct_pb2.Struct
    def __init__(self, summary: _Optional[str] = ..., topics: _Optional[_Iterable[_Union[_struct_pb2.Struct, _Mapping]]] = ..., intents: _Optional[_Iterable[_Union[_struct_pb2.Struct, _Mapping]]] = ..., sentiments: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., language: _Optional[str] = ..., model: _Optional[str] = ..., request_id: _Optional[str] = ..., metadata: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., raw: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class TranscriptionStreamEvent(_message.Message):
    __slots__ = ("type", "text", "is_final", "speech_final", "start", "end", "confidence", "words", "speaker", "channel_index", "provider", "raw")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    IS_FINAL_FIELD_NUMBER: _ClassVar[int]
    SPEECH_FINAL_FIELD_NUMBER: _ClassVar[int]
    START_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    WORDS_FIELD_NUMBER: _ClassVar[int]
    SPEAKER_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_INDEX_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_FIELD_NUMBER: _ClassVar[int]
    RAW_FIELD_NUMBER: _ClassVar[int]
    type: StreamEventType
    text: str
    is_final: bool
    speech_final: bool
    start: float
    end: float
    confidence: float
    words: _containers.RepeatedCompositeFieldContainer[_struct_pb2.Struct]
    speaker: str
    channel_index: _containers.RepeatedScalarFieldContainer[int]
    provider: str
    raw: _struct_pb2.Struct
    def __init__(self, type: _Optional[_Union[StreamEventType, str]] = ..., text: _Optional[str] = ..., is_final: _Optional[bool] = ..., speech_final: _Optional[bool] = ..., start: _Optional[float] = ..., end: _Optional[float] = ..., confidence: _Optional[float] = ..., words: _Optional[_Iterable[_Union[_struct_pb2.Struct, _Mapping]]] = ..., speaker: _Optional[str] = ..., channel_index: _Optional[_Iterable[int]] = ..., provider: _Optional[str] = ..., raw: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class VoiceAgentFunctionCall(_message.Message):
    __slots__ = ("id", "name", "arguments", "client_side", "raw")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ARGUMENTS_FIELD_NUMBER: _ClassVar[int]
    CLIENT_SIDE_FIELD_NUMBER: _ClassVar[int]
    RAW_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    arguments: _struct_pb2.Struct
    client_side: bool
    raw: _struct_pb2.Struct
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., arguments: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., client_side: _Optional[bool] = ..., raw: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class VoiceAgentEvent(_message.Message):
    __slots__ = ("type", "role", "content", "audio", "function_call", "provider", "raw")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    FUNCTION_CALL_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_FIELD_NUMBER: _ClassVar[int]
    RAW_FIELD_NUMBER: _ClassVar[int]
    type: VoiceAgentEventType
    role: str
    content: str
    audio: bytes
    function_call: VoiceAgentFunctionCall
    provider: str
    raw: _struct_pb2.Struct
    def __init__(self, type: _Optional[_Union[VoiceAgentEventType, str]] = ..., role: _Optional[str] = ..., content: _Optional[str] = ..., audio: _Optional[bytes] = ..., function_call: _Optional[_Union[VoiceAgentFunctionCall, _Mapping]] = ..., provider: _Optional[str] = ..., raw: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class OpenStt(_message.Message):
    __slots__ = ("model", "language", "options")
    MODEL_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    model: str
    language: str
    options: _struct_pb2.Struct
    def __init__(self, model: _Optional[str] = ..., language: _Optional[str] = ..., options: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class AudioIn(_message.Message):
    __slots__ = ("open", "audio", "control")
    OPEN_FIELD_NUMBER: _ClassVar[int]
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    CONTROL_FIELD_NUMBER: _ClassVar[int]
    open: OpenStt
    audio: bytes
    control: SttControl
    def __init__(self, open: _Optional[_Union[OpenStt, _Mapping]] = ..., audio: _Optional[bytes] = ..., control: _Optional[_Union[SttControl, str]] = ...) -> None: ...

class OpenTts(_message.Message):
    __slots__ = ("voice", "model", "format")
    VOICE_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    voice: str
    model: str
    format: str
    def __init__(self, voice: _Optional[str] = ..., model: _Optional[str] = ..., format: _Optional[str] = ...) -> None: ...

class SynthControl(_message.Message):
    __slots__ = ("open", "text", "control")
    OPEN_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    CONTROL_FIELD_NUMBER: _ClassVar[int]
    open: OpenTts
    text: str
    control: TtsControl
    def __init__(self, open: _Optional[_Union[OpenTts, _Mapping]] = ..., text: _Optional[str] = ..., control: _Optional[_Union[TtsControl, str]] = ...) -> None: ...

class AudioOut(_message.Message):
    __slots__ = ("audio", "seq")
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    SEQ_FIELD_NUMBER: _ClassVar[int]
    audio: bytes
    seq: int
    def __init__(self, audio: _Optional[bytes] = ..., seq: _Optional[int] = ...) -> None: ...

class FunctionCallResponse(_message.Message):
    __slots__ = ("id", "output")
    ID_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    id: str
    output: str
    def __init__(self, id: _Optional[str] = ..., output: _Optional[str] = ...) -> None: ...

class VoiceAgentClientEvent(_message.Message):
    __slots__ = ("settings", "audio", "inject_user_message", "inject_agent_message", "update_prompt", "update_think", "update_speak", "respond_to_function_call", "keepalive")
    SETTINGS_FIELD_NUMBER: _ClassVar[int]
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    INJECT_USER_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    INJECT_AGENT_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    UPDATE_PROMPT_FIELD_NUMBER: _ClassVar[int]
    UPDATE_THINK_FIELD_NUMBER: _ClassVar[int]
    UPDATE_SPEAK_FIELD_NUMBER: _ClassVar[int]
    RESPOND_TO_FUNCTION_CALL_FIELD_NUMBER: _ClassVar[int]
    KEEPALIVE_FIELD_NUMBER: _ClassVar[int]
    settings: _struct_pb2.Struct
    audio: bytes
    inject_user_message: str
    inject_agent_message: str
    update_prompt: str
    update_think: _struct_pb2.Struct
    update_speak: _struct_pb2.Struct
    respond_to_function_call: FunctionCallResponse
    keepalive: bool
    def __init__(self, settings: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., audio: _Optional[bytes] = ..., inject_user_message: _Optional[str] = ..., inject_agent_message: _Optional[str] = ..., update_prompt: _Optional[str] = ..., update_think: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., update_speak: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., respond_to_function_call: _Optional[_Union[FunctionCallResponse, _Mapping]] = ..., keepalive: _Optional[bool] = ...) -> None: ...

class SynthesizeRequest(_message.Message):
    __slots__ = ("text", "voice", "model", "response_format", "speed", "provider_name")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    VOICE_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FORMAT_FIELD_NUMBER: _ClassVar[int]
    SPEED_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_NAME_FIELD_NUMBER: _ClassVar[int]
    text: str
    voice: str
    model: str
    response_format: str
    speed: float
    provider_name: str
    def __init__(self, text: _Optional[str] = ..., voice: _Optional[str] = ..., model: _Optional[str] = ..., response_format: _Optional[str] = ..., speed: _Optional[float] = ..., provider_name: _Optional[str] = ...) -> None: ...

class TranscribeRequest(_message.Message):
    __slots__ = ("audio_data", "model", "language", "response_format", "provider_name")
    AUDIO_DATA_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FORMAT_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_NAME_FIELD_NUMBER: _ClassVar[int]
    audio_data: bytes
    model: str
    language: str
    response_format: str
    provider_name: str
    def __init__(self, audio_data: _Optional[bytes] = ..., model: _Optional[str] = ..., language: _Optional[str] = ..., response_format: _Optional[str] = ..., provider_name: _Optional[str] = ...) -> None: ...

class GenerateImageRequest(_message.Message):
    __slots__ = ("prompt", "model", "n", "size", "quality", "provider_name")
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    N_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    QUALITY_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_NAME_FIELD_NUMBER: _ClassVar[int]
    prompt: str
    model: str
    n: int
    size: str
    quality: str
    provider_name: str
    def __init__(self, prompt: _Optional[str] = ..., model: _Optional[str] = ..., n: _Optional[int] = ..., size: _Optional[str] = ..., quality: _Optional[str] = ..., provider_name: _Optional[str] = ...) -> None: ...

class OcrRequest(_message.Message):
    __slots__ = ("url", "data", "model", "provider_name")
    URL_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_NAME_FIELD_NUMBER: _ClassVar[int]
    url: str
    data: bytes
    model: str
    provider_name: str
    def __init__(self, url: _Optional[str] = ..., data: _Optional[bytes] = ..., model: _Optional[str] = ..., provider_name: _Optional[str] = ...) -> None: ...

class AnalyzeTextRequest(_message.Message):
    __slots__ = ("text", "model", "features", "provider_name")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_NAME_FIELD_NUMBER: _ClassVar[int]
    text: str
    model: str
    features: _struct_pb2.Struct
    provider_name: str
    def __init__(self, text: _Optional[str] = ..., model: _Optional[str] = ..., features: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., provider_name: _Optional[str] = ...) -> None: ...
