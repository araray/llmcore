from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ErrorCategory(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ERROR_CATEGORY_UNSPECIFIED: _ClassVar[ErrorCategory]
    ERROR_CATEGORY_PROVIDER: _ClassVar[ErrorCategory]
    ERROR_CATEGORY_CONFIG: _ClassVar[ErrorCategory]
    ERROR_CATEGORY_CONTEXT_LENGTH: _ClassVar[ErrorCategory]
    ERROR_CATEGORY_CONTEXT: _ClassVar[ErrorCategory]
    ERROR_CATEGORY_EMBEDDING: _ClassVar[ErrorCategory]
    ERROR_CATEGORY_STORAGE: _ClassVar[ErrorCategory]
    ERROR_CATEGORY_NOT_FOUND: _ClassVar[ErrorCategory]
    ERROR_CATEGORY_SEARCH: _ClassVar[ErrorCategory]
    ERROR_CATEGORY_UNSUPPORTED: _ClassVar[ErrorCategory]
    ERROR_CATEGORY_CANCELLED: _ClassVar[ErrorCategory]
    ERROR_CATEGORY_INTERNAL: _ClassVar[ErrorCategory]
    ERROR_CATEGORY_INVALID_ARGUMENT: _ClassVar[ErrorCategory]
ERROR_CATEGORY_UNSPECIFIED: ErrorCategory
ERROR_CATEGORY_PROVIDER: ErrorCategory
ERROR_CATEGORY_CONFIG: ErrorCategory
ERROR_CATEGORY_CONTEXT_LENGTH: ErrorCategory
ERROR_CATEGORY_CONTEXT: ErrorCategory
ERROR_CATEGORY_EMBEDDING: ErrorCategory
ERROR_CATEGORY_STORAGE: ErrorCategory
ERROR_CATEGORY_NOT_FOUND: ErrorCategory
ERROR_CATEGORY_SEARCH: ErrorCategory
ERROR_CATEGORY_UNSUPPORTED: ErrorCategory
ERROR_CATEGORY_CANCELLED: ErrorCategory
ERROR_CATEGORY_INTERNAL: ErrorCategory
ERROR_CATEGORY_INVALID_ARGUMENT: ErrorCategory

class LlmcoreError(_message.Message):
    __slots__ = ("category", "code", "message", "provider", "model", "http_status", "retryable", "retry_after_ms", "details")
    CATEGORY_FIELD_NUMBER: _ClassVar[int]
    CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    HTTP_STATUS_FIELD_NUMBER: _ClassVar[int]
    RETRYABLE_FIELD_NUMBER: _ClassVar[int]
    RETRY_AFTER_MS_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    category: ErrorCategory
    code: str
    message: str
    provider: str
    model: str
    http_status: int
    retryable: bool
    retry_after_ms: float
    details: _struct_pb2.Struct
    def __init__(self, category: _Optional[_Union[ErrorCategory, str]] = ..., code: _Optional[str] = ..., message: _Optional[str] = ..., provider: _Optional[str] = ..., model: _Optional[str] = ..., http_status: _Optional[int] = ..., retryable: _Optional[bool] = ..., retry_after_ms: _Optional[float] = ..., details: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...
