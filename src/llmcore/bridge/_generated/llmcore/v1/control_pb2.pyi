from llmcore.v1 import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ProviderInfo(_message.Message):
    __slots__ = ("name", "available")
    NAME_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    name: str
    available: bool
    def __init__(self, name: _Optional[str] = ..., available: _Optional[bool] = ...) -> None: ...

class ServerInfo(_message.Message):
    __slots__ = ("llmcore_version", "bridge_version", "contract_version", "transports", "providers", "capabilities", "tiers")
    LLMCORE_VERSION_FIELD_NUMBER: _ClassVar[int]
    BRIDGE_VERSION_FIELD_NUMBER: _ClassVar[int]
    CONTRACT_VERSION_FIELD_NUMBER: _ClassVar[int]
    TRANSPORTS_FIELD_NUMBER: _ClassVar[int]
    PROVIDERS_FIELD_NUMBER: _ClassVar[int]
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    TIERS_FIELD_NUMBER: _ClassVar[int]
    llmcore_version: str
    bridge_version: str
    contract_version: str
    transports: _containers.RepeatedScalarFieldContainer[str]
    providers: _containers.RepeatedCompositeFieldContainer[ProviderInfo]
    capabilities: _containers.RepeatedScalarFieldContainer[str]
    tiers: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, llmcore_version: _Optional[str] = ..., bridge_version: _Optional[str] = ..., contract_version: _Optional[str] = ..., transports: _Optional[_Iterable[str]] = ..., providers: _Optional[_Iterable[_Union[ProviderInfo, _Mapping]]] = ..., capabilities: _Optional[_Iterable[str]] = ..., tiers: _Optional[_Iterable[str]] = ...) -> None: ...

class HealthStatus(_message.Message):
    __slots__ = ("ok", "detail")
    OK_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    detail: str
    def __init__(self, ok: _Optional[bool] = ..., detail: _Optional[str] = ...) -> None: ...

class ReloadConfigRequest(_message.Message):
    __slots__ = ("path",)
    PATH_FIELD_NUMBER: _ClassVar[int]
    path: str
    def __init__(self, path: _Optional[str] = ...) -> None: ...

class ReloadConfigResponse(_message.Message):
    __slots__ = ("ok",)
    OK_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    def __init__(self, ok: _Optional[bool] = ...) -> None: ...
