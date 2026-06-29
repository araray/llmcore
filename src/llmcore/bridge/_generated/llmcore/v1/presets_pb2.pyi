from google.protobuf import struct_pb2 as _struct_pb2
from llmcore.v1 import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ContextPresetItem(_message.Message):
    __slots__ = ("item_id", "type", "content", "source_identifier", "metadata")
    ITEM_ID_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    SOURCE_IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    item_id: str
    type: str
    content: str
    source_identifier: str
    metadata: _struct_pb2.Struct
    def __init__(self, item_id: _Optional[str] = ..., type: _Optional[str] = ..., content: _Optional[str] = ..., source_identifier: _Optional[str] = ..., metadata: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class ContextPreset(_message.Message):
    __slots__ = ("name", "description", "items", "created_at", "updated_at", "metadata")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    name: str
    description: str
    items: _containers.RepeatedCompositeFieldContainer[ContextPresetItem]
    created_at: str
    updated_at: str
    metadata: _struct_pb2.Struct
    def __init__(self, name: _Optional[str] = ..., description: _Optional[str] = ..., items: _Optional[_Iterable[_Union[ContextPresetItem, _Mapping]]] = ..., created_at: _Optional[str] = ..., updated_at: _Optional[str] = ..., metadata: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class SaveContextPresetRequest(_message.Message):
    __slots__ = ("preset",)
    PRESET_FIELD_NUMBER: _ClassVar[int]
    preset: ContextPreset
    def __init__(self, preset: _Optional[_Union[ContextPreset, _Mapping]] = ...) -> None: ...

class GetContextPresetRequest(_message.Message):
    __slots__ = ("preset_name",)
    PRESET_NAME_FIELD_NUMBER: _ClassVar[int]
    preset_name: str
    def __init__(self, preset_name: _Optional[str] = ...) -> None: ...

class ListContextPresetsResponse(_message.Message):
    __slots__ = ("presets",)
    PRESETS_FIELD_NUMBER: _ClassVar[int]
    presets: _containers.RepeatedCompositeFieldContainer[_struct_pb2.Struct]
    def __init__(self, presets: _Optional[_Iterable[_Union[_struct_pb2.Struct, _Mapping]]] = ...) -> None: ...

class DeleteContextPresetRequest(_message.Message):
    __slots__ = ("preset_name",)
    PRESET_NAME_FIELD_NUMBER: _ClassVar[int]
    preset_name: str
    def __init__(self, preset_name: _Optional[str] = ...) -> None: ...

class DeleteContextPresetResponse(_message.Message):
    __slots__ = ("deleted",)
    DELETED_FIELD_NUMBER: _ClassVar[int]
    deleted: bool
    def __init__(self, deleted: _Optional[bool] = ...) -> None: ...
