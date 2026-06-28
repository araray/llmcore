from google.protobuf import struct_pb2 as _struct_pb2
from llmcore.v1 import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ContextDocument(_message.Message):
    __slots__ = ("id", "content", "embedding", "metadata", "score")
    ID_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    id: str
    content: str
    embedding: _containers.RepeatedScalarFieldContainer[float]
    metadata: _struct_pb2.Struct
    score: float
    def __init__(self, id: _Optional[str] = ..., content: _Optional[str] = ..., embedding: _Optional[_Iterable[float]] = ..., metadata: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., score: _Optional[float] = ...) -> None: ...

class AddDocumentsRequest(_message.Message):
    __slots__ = ("documents", "collection_name")
    DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    documents: _containers.RepeatedCompositeFieldContainer[_struct_pb2.Struct]
    collection_name: str
    def __init__(self, documents: _Optional[_Iterable[_Union[_struct_pb2.Struct, _Mapping]]] = ..., collection_name: _Optional[str] = ...) -> None: ...

class AddDocumentsResponse(_message.Message):
    __slots__ = ("ids",)
    IDS_FIELD_NUMBER: _ClassVar[int]
    ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, ids: _Optional[_Iterable[str]] = ...) -> None: ...

class SearchVectorStoreRequest(_message.Message):
    __slots__ = ("query", "k", "collection_name", "metadata_filter")
    QUERY_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    METADATA_FILTER_FIELD_NUMBER: _ClassVar[int]
    query: str
    k: int
    collection_name: str
    metadata_filter: _struct_pb2.Struct
    def __init__(self, query: _Optional[str] = ..., k: _Optional[int] = ..., collection_name: _Optional[str] = ..., metadata_filter: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class SearchVectorStoreResponse(_message.Message):
    __slots__ = ("documents",)
    DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    documents: _containers.RepeatedCompositeFieldContainer[ContextDocument]
    def __init__(self, documents: _Optional[_Iterable[_Union[ContextDocument, _Mapping]]] = ...) -> None: ...

class ListCollectionsResponse(_message.Message):
    __slots__ = ("collections",)
    COLLECTIONS_FIELD_NUMBER: _ClassVar[int]
    collections: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, collections: _Optional[_Iterable[str]] = ...) -> None: ...

class GetRagCollectionInfoRequest(_message.Message):
    __slots__ = ("collection_name",)
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    def __init__(self, collection_name: _Optional[str] = ...) -> None: ...

class RagCollectionInfo(_message.Message):
    __slots__ = ("collection_name", "info")
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    INFO_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    info: _struct_pb2.Struct
    def __init__(self, collection_name: _Optional[str] = ..., info: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class DeleteRagCollectionRequest(_message.Message):
    __slots__ = ("collection_name", "force")
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    FORCE_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    force: bool
    def __init__(self, collection_name: _Optional[str] = ..., force: _Optional[bool] = ...) -> None: ...

class DeleteRagCollectionResponse(_message.Message):
    __slots__ = ("deleted",)
    DELETED_FIELD_NUMBER: _ClassVar[int]
    deleted: bool
    def __init__(self, deleted: _Optional[bool] = ...) -> None: ...
