from google.protobuf import struct_pb2 as _struct_pb2
from llmcore.v1 import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ModelDetails(_message.Message):
    __slots__ = ("id", "provider_name", "display_name", "context_length", "max_output_tokens", "supports_streaming", "supports_tools", "supports_vision", "supports_reasoning", "family", "parameter_count", "quantization_level", "file_size_bytes", "model_type", "metadata")
    ID_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_NAME_FIELD_NUMBER: _ClassVar[int]
    DISPLAY_NAME_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_LENGTH_FIELD_NUMBER: _ClassVar[int]
    MAX_OUTPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_STREAMING_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_TOOLS_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_VISION_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_REASONING_FIELD_NUMBER: _ClassVar[int]
    FAMILY_FIELD_NUMBER: _ClassVar[int]
    PARAMETER_COUNT_FIELD_NUMBER: _ClassVar[int]
    QUANTIZATION_LEVEL_FIELD_NUMBER: _ClassVar[int]
    FILE_SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    MODEL_TYPE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    id: str
    provider_name: str
    display_name: str
    context_length: int
    max_output_tokens: int
    supports_streaming: bool
    supports_tools: bool
    supports_vision: bool
    supports_reasoning: bool
    family: str
    parameter_count: str
    quantization_level: str
    file_size_bytes: int
    model_type: str
    metadata: _struct_pb2.Struct
    def __init__(self, id: _Optional[str] = ..., provider_name: _Optional[str] = ..., display_name: _Optional[str] = ..., context_length: _Optional[int] = ..., max_output_tokens: _Optional[int] = ..., supports_streaming: _Optional[bool] = ..., supports_tools: _Optional[bool] = ..., supports_vision: _Optional[bool] = ..., supports_reasoning: _Optional[bool] = ..., family: _Optional[str] = ..., parameter_count: _Optional[str] = ..., quantization_level: _Optional[str] = ..., file_size_bytes: _Optional[int] = ..., model_type: _Optional[str] = ..., metadata: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class ListProvidersResponse(_message.Message):
    __slots__ = ("providers",)
    PROVIDERS_FIELD_NUMBER: _ClassVar[int]
    providers: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, providers: _Optional[_Iterable[str]] = ...) -> None: ...

class ListModelsRequest(_message.Message):
    __slots__ = ("provider_name",)
    PROVIDER_NAME_FIELD_NUMBER: _ClassVar[int]
    provider_name: str
    def __init__(self, provider_name: _Optional[str] = ...) -> None: ...

class ListModelsResponse(_message.Message):
    __slots__ = ("models",)
    MODELS_FIELD_NUMBER: _ClassVar[int]
    models: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, models: _Optional[_Iterable[str]] = ...) -> None: ...

class GetProviderRequest(_message.Message):
    __slots__ = ("provider_name",)
    PROVIDER_NAME_FIELD_NUMBER: _ClassVar[int]
    provider_name: str
    def __init__(self, provider_name: _Optional[str] = ...) -> None: ...
