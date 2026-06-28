from google.protobuf import struct_pb2 as _struct_pb2
from llmcore.v1 import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ContextItem(_message.Message):
    __slots__ = ("id", "type", "source_id", "content", "tokens", "original_tokens", "is_truncated", "metadata", "timestamp")
    ID_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    TOKENS_FIELD_NUMBER: _ClassVar[int]
    ORIGINAL_TOKENS_FIELD_NUMBER: _ClassVar[int]
    IS_TRUNCATED_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    id: str
    type: str
    source_id: str
    content: str
    tokens: int
    original_tokens: int
    is_truncated: bool
    metadata: _struct_pb2.Struct
    timestamp: str
    def __init__(self, id: _Optional[str] = ..., type: _Optional[str] = ..., source_id: _Optional[str] = ..., content: _Optional[str] = ..., tokens: _Optional[int] = ..., original_tokens: _Optional[int] = ..., is_truncated: _Optional[bool] = ..., metadata: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., timestamp: _Optional[str] = ...) -> None: ...

class ChatSession(_message.Message):
    __slots__ = ("id", "name", "messages", "context_items", "created_at", "updated_at", "metadata")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_ITEMS_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    messages: _containers.RepeatedCompositeFieldContainer[_common_pb2.Message]
    context_items: _containers.RepeatedCompositeFieldContainer[ContextItem]
    created_at: str
    updated_at: str
    metadata: _struct_pb2.Struct
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., messages: _Optional[_Iterable[_Union[_common_pb2.Message, _Mapping]]] = ..., context_items: _Optional[_Iterable[_Union[ContextItem, _Mapping]]] = ..., created_at: _Optional[str] = ..., updated_at: _Optional[str] = ..., metadata: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class Int32Range(_message.Message):
    __slots__ = ("start", "end")
    START_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    start: int
    end: int
    def __init__(self, start: _Optional[int] = ..., end: _Optional[int] = ...) -> None: ...

class CreateSessionRequest(_message.Message):
    __slots__ = ("session_id", "name", "system_message")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    SYSTEM_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    name: str
    system_message: str
    def __init__(self, session_id: _Optional[str] = ..., name: _Optional[str] = ..., system_message: _Optional[str] = ...) -> None: ...

class GetSessionRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class ListSessionsRequest(_message.Message):
    __slots__ = ("limit",)
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    limit: int
    def __init__(self, limit: _Optional[int] = ...) -> None: ...

class ListSessionsResponse(_message.Message):
    __slots__ = ("sessions",)
    SESSIONS_FIELD_NUMBER: _ClassVar[int]
    sessions: _containers.RepeatedCompositeFieldContainer[ChatSession]
    def __init__(self, sessions: _Optional[_Iterable[_Union[ChatSession, _Mapping]]] = ...) -> None: ...

class DeleteSessionRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class UpdateSessionNameRequest(_message.Message):
    __slots__ = ("session_id", "new_name")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    NEW_NAME_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    new_name: str
    def __init__(self, session_id: _Optional[str] = ..., new_name: _Optional[str] = ...) -> None: ...

class ForkSessionRequest(_message.Message):
    __slots__ = ("session_id", "new_name", "from_message_id", "message_ids", "message_range", "include_context_items", "metadata")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    NEW_NAME_FIELD_NUMBER: _ClassVar[int]
    FROM_MESSAGE_ID_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_IDS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_RANGE_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_CONTEXT_ITEMS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    new_name: str
    from_message_id: str
    message_ids: _containers.RepeatedScalarFieldContainer[str]
    message_range: Int32Range
    include_context_items: bool
    metadata: _struct_pb2.Struct
    def __init__(self, session_id: _Optional[str] = ..., new_name: _Optional[str] = ..., from_message_id: _Optional[str] = ..., message_ids: _Optional[_Iterable[str]] = ..., message_range: _Optional[_Union[Int32Range, _Mapping]] = ..., include_context_items: _Optional[bool] = ..., metadata: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class ForkSessionResponse(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class CloneSessionRequest(_message.Message):
    __slots__ = ("session_id", "new_name", "include_messages", "include_context_items")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    NEW_NAME_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_MESSAGES_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_CONTEXT_ITEMS_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    new_name: str
    include_messages: bool
    include_context_items: bool
    def __init__(self, session_id: _Optional[str] = ..., new_name: _Optional[str] = ..., include_messages: _Optional[bool] = ..., include_context_items: _Optional[bool] = ...) -> None: ...

class CloneSessionResponse(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class DeleteMessagesRequest(_message.Message):
    __slots__ = ("session_id", "message_ids")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_IDS_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    message_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, session_id: _Optional[str] = ..., message_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class DeleteMessagesResponse(_message.Message):
    __slots__ = ("deleted_count",)
    DELETED_COUNT_FIELD_NUMBER: _ClassVar[int]
    deleted_count: int
    def __init__(self, deleted_count: _Optional[int] = ...) -> None: ...

class GetMessagesByRangeRequest(_message.Message):
    __slots__ = ("session_id", "start_index", "end_index")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    START_INDEX_FIELD_NUMBER: _ClassVar[int]
    END_INDEX_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    start_index: int
    end_index: int
    def __init__(self, session_id: _Optional[str] = ..., start_index: _Optional[int] = ..., end_index: _Optional[int] = ...) -> None: ...

class GetMessagesByRangeResponse(_message.Message):
    __slots__ = ("messages",)
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    messages: _containers.RepeatedCompositeFieldContainer[_common_pb2.Message]
    def __init__(self, messages: _Optional[_Iterable[_Union[_common_pb2.Message, _Mapping]]] = ...) -> None: ...

class AddContextItemRequest(_message.Message):
    __slots__ = ("session_id", "content", "type", "source_id", "metadata")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    content: str
    type: str
    source_id: str
    metadata: _struct_pb2.Struct
    def __init__(self, session_id: _Optional[str] = ..., content: _Optional[str] = ..., type: _Optional[str] = ..., source_id: _Optional[str] = ..., metadata: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class AddContextItemResponse(_message.Message):
    __slots__ = ("item_id",)
    ITEM_ID_FIELD_NUMBER: _ClassVar[int]
    item_id: str
    def __init__(self, item_id: _Optional[str] = ...) -> None: ...

class GetContextItemRequest(_message.Message):
    __slots__ = ("session_id", "item_id")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    ITEM_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    item_id: str
    def __init__(self, session_id: _Optional[str] = ..., item_id: _Optional[str] = ...) -> None: ...

class RemoveContextItemRequest(_message.Message):
    __slots__ = ("session_id", "item_id")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    ITEM_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    item_id: str
    def __init__(self, session_id: _Optional[str] = ..., item_id: _Optional[str] = ...) -> None: ...

class RemoveContextItemResponse(_message.Message):
    __slots__ = ("removed",)
    REMOVED_FIELD_NUMBER: _ClassVar[int]
    removed: bool
    def __init__(self, removed: _Optional[bool] = ...) -> None: ...
