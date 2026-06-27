from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Role(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ROLE_UNSPECIFIED: _ClassVar[Role]
    ROLE_SYSTEM: _ClassVar[Role]
    ROLE_USER: _ClassVar[Role]
    ROLE_ASSISTANT: _ClassVar[Role]
    ROLE_TOOL: _ClassVar[Role]
ROLE_UNSPECIFIED: Role
ROLE_SYSTEM: Role
ROLE_USER: Role
ROLE_ASSISTANT: Role
ROLE_TOOL: Role

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class Message(_message.Message):
    __slots__ = ("id", "session_id", "role", "content", "timestamp", "tool_call_id", "tokens", "metadata")
    ID_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    TOOL_CALL_ID_FIELD_NUMBER: _ClassVar[int]
    TOKENS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    id: str
    session_id: str
    role: Role
    content: str
    timestamp: str
    tool_call_id: str
    tokens: int
    metadata: _struct_pb2.Struct
    def __init__(self, id: _Optional[str] = ..., session_id: _Optional[str] = ..., role: _Optional[_Union[Role, str]] = ..., content: _Optional[str] = ..., timestamp: _Optional[str] = ..., tool_call_id: _Optional[str] = ..., tokens: _Optional[int] = ..., metadata: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class Tool(_message.Message):
    __slots__ = ("name", "description", "parameters")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    name: str
    description: str
    parameters: _struct_pb2.Struct
    def __init__(self, name: _Optional[str] = ..., description: _Optional[str] = ..., parameters: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class ToolCall(_message.Message):
    __slots__ = ("id", "name", "arguments")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ARGUMENTS_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    arguments: _struct_pb2.Struct
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., arguments: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class ToolResult(_message.Message):
    __slots__ = ("tool_call_id", "content", "is_error")
    TOOL_CALL_ID_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    IS_ERROR_FIELD_NUMBER: _ClassVar[int]
    tool_call_id: str
    content: str
    is_error: bool
    def __init__(self, tool_call_id: _Optional[str] = ..., content: _Optional[str] = ..., is_error: _Optional[bool] = ...) -> None: ...

class Usage(_message.Message):
    __slots__ = ("prompt_tokens", "completion_tokens", "total_tokens", "provider", "model")
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_TOKENS_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    provider: str
    model: str
    def __init__(self, prompt_tokens: _Optional[int] = ..., completion_tokens: _Optional[int] = ..., total_tokens: _Optional[int] = ..., provider: _Optional[str] = ..., model: _Optional[str] = ...) -> None: ...

class FloatVector(_message.Message):
    __slots__ = ("values", "index")
    VALUES_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[float]
    index: int
    def __init__(self, values: _Optional[_Iterable[float]] = ..., index: _Optional[int] = ...) -> None: ...
