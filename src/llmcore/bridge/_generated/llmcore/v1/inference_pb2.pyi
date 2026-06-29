from google.protobuf import struct_pb2 as _struct_pb2
from llmcore.v1 import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CostEstimate(_message.Message):
    __slots__ = ("input_cost", "output_cost", "cached_discount", "reasoning_cost", "total_cost", "currency", "pricing_source", "prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens", "input_price_per_million", "output_price_per_million", "cached_price_per_million", "model_id", "provider")
    INPUT_COST_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_COST_FIELD_NUMBER: _ClassVar[int]
    CACHED_DISCOUNT_FIELD_NUMBER: _ClassVar[int]
    REASONING_COST_FIELD_NUMBER: _ClassVar[int]
    TOTAL_COST_FIELD_NUMBER: _ClassVar[int]
    CURRENCY_FIELD_NUMBER: _ClassVar[int]
    PRICING_SOURCE_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TOKENS_FIELD_NUMBER: _ClassVar[int]
    CACHED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    REASONING_TOKENS_FIELD_NUMBER: _ClassVar[int]
    INPUT_PRICE_PER_MILLION_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_PRICE_PER_MILLION_FIELD_NUMBER: _ClassVar[int]
    CACHED_PRICE_PER_MILLION_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_FIELD_NUMBER: _ClassVar[int]
    input_cost: float
    output_cost: float
    cached_discount: float
    reasoning_cost: float
    total_cost: float
    currency: str
    pricing_source: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    input_price_per_million: float
    output_price_per_million: float
    cached_price_per_million: float
    model_id: str
    provider: str
    def __init__(self, input_cost: _Optional[float] = ..., output_cost: _Optional[float] = ..., cached_discount: _Optional[float] = ..., reasoning_cost: _Optional[float] = ..., total_cost: _Optional[float] = ..., currency: _Optional[str] = ..., pricing_source: _Optional[str] = ..., prompt_tokens: _Optional[int] = ..., completion_tokens: _Optional[int] = ..., cached_tokens: _Optional[int] = ..., reasoning_tokens: _Optional[int] = ..., input_price_per_million: _Optional[float] = ..., output_price_per_million: _Optional[float] = ..., cached_price_per_million: _Optional[float] = ..., model_id: _Optional[str] = ..., provider: _Optional[str] = ...) -> None: ...

class ChatRequest(_message.Message):
    __slots__ = ("message", "session_id", "system_message", "provider_name", "model_name", "save_session", "tools", "tool_choice", "provider_kwargs")
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    SYSTEM_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    SAVE_SESSION_FIELD_NUMBER: _ClassVar[int]
    TOOLS_FIELD_NUMBER: _ClassVar[int]
    TOOL_CHOICE_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_KWARGS_FIELD_NUMBER: _ClassVar[int]
    message: str
    session_id: str
    system_message: str
    provider_name: str
    model_name: str
    save_session: bool
    tools: _containers.RepeatedCompositeFieldContainer[_common_pb2.Tool]
    tool_choice: str
    provider_kwargs: _struct_pb2.Struct
    def __init__(self, message: _Optional[str] = ..., session_id: _Optional[str] = ..., system_message: _Optional[str] = ..., provider_name: _Optional[str] = ..., model_name: _Optional[str] = ..., save_session: _Optional[bool] = ..., tools: _Optional[_Iterable[_Union[_common_pb2.Tool, _Mapping]]] = ..., tool_choice: _Optional[str] = ..., provider_kwargs: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class ChatResponse(_message.Message):
    __slots__ = ("text", "usage", "tool_calls", "finish_reason")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    USAGE_FIELD_NUMBER: _ClassVar[int]
    TOOL_CALLS_FIELD_NUMBER: _ClassVar[int]
    FINISH_REASON_FIELD_NUMBER: _ClassVar[int]
    text: str
    usage: _common_pb2.Usage
    tool_calls: _containers.RepeatedCompositeFieldContainer[_common_pb2.ToolCall]
    finish_reason: str
    def __init__(self, text: _Optional[str] = ..., usage: _Optional[_Union[_common_pb2.Usage, _Mapping]] = ..., tool_calls: _Optional[_Iterable[_Union[_common_pb2.ToolCall, _Mapping]]] = ..., finish_reason: _Optional[str] = ...) -> None: ...

class ChatChunk(_message.Message):
    __slots__ = ("text", "done")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    DONE_FIELD_NUMBER: _ClassVar[int]
    text: str
    done: bool
    def __init__(self, text: _Optional[str] = ..., done: _Optional[bool] = ...) -> None: ...

class EmbedRequest(_message.Message):
    __slots__ = ("input", "provider_name", "model_name")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    input: _containers.RepeatedScalarFieldContainer[str]
    provider_name: str
    model_name: str
    def __init__(self, input: _Optional[_Iterable[str]] = ..., provider_name: _Optional[str] = ..., model_name: _Optional[str] = ...) -> None: ...

class EmbedResponse(_message.Message):
    __slots__ = ("vectors", "model", "usage")
    VECTORS_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    USAGE_FIELD_NUMBER: _ClassVar[int]
    vectors: _containers.RepeatedCompositeFieldContainer[_common_pb2.FloatVector]
    model: str
    usage: _common_pb2.Usage
    def __init__(self, vectors: _Optional[_Iterable[_Union[_common_pb2.FloatVector, _Mapping]]] = ..., model: _Optional[str] = ..., usage: _Optional[_Union[_common_pb2.Usage, _Mapping]] = ...) -> None: ...

class CountTokensRequest(_message.Message):
    __slots__ = ("text", "provider_name", "model_name")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    text: str
    provider_name: str
    model_name: str
    def __init__(self, text: _Optional[str] = ..., provider_name: _Optional[str] = ..., model_name: _Optional[str] = ...) -> None: ...

class CountTokensResponse(_message.Message):
    __slots__ = ("tokens",)
    TOKENS_FIELD_NUMBER: _ClassVar[int]
    tokens: int
    def __init__(self, tokens: _Optional[int] = ...) -> None: ...

class EstimateCostRequest(_message.Message):
    __slots__ = ("provider_name", "model_name", "prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens")
    PROVIDER_NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TOKENS_FIELD_NUMBER: _ClassVar[int]
    CACHED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    REASONING_TOKENS_FIELD_NUMBER: _ClassVar[int]
    provider_name: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    def __init__(self, provider_name: _Optional[str] = ..., model_name: _Optional[str] = ..., prompt_tokens: _Optional[int] = ..., completion_tokens: _Optional[int] = ..., cached_tokens: _Optional[int] = ..., reasoning_tokens: _Optional[int] = ...) -> None: ...
