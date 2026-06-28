"""End-to-end tests for the Tier-1 VectorService RPCs (phase B4).

Exercised over **both** transports — gRPC (``grpc_channel``) and HTTP/JSON
(``http_client``) — against the deterministic in-memory vector store in
``FakeFacade``. As with SessionService the handlers are not capability-gated;
``LLMCORE_BRIDGE_FAKE_VECTOR=1`` only toggles the ``tier1.vector`` capability
advertisement in ``ServerInfo``.
"""

from __future__ import annotations

import grpc
import pytest
from google.protobuf import struct_pb2
from google.protobuf.json_format import MessageToDict

from llmcore.bridge._generated.llmcore.v1 import (
    common_pb2,
    control_pb2_grpc,
    vector_pb2,
    vector_pb2_grpc,
)

_P = "/llmcore.v1/VectorService"


def _doc(content: str, **metadata) -> struct_pb2.Struct:
    s = struct_pb2.Struct()
    s.update({"content": content, "metadata": metadata})
    return s


# --------------------------------------------------------------------------- #
# gRPC
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_grpc_add_and_search(grpc_channel):
    stub = vector_pb2_grpc.VectorServiceStub(grpc_channel)
    added = await stub.AddDocuments(
        vector_pb2.AddDocumentsRequest(
            documents=[_doc("the cat sat", topic="a"), _doc("the dog ran", topic="b")]
        )
    )
    assert len(added.ids) == 2
    res = await stub.SearchVectorStore(vector_pb2.SearchVectorStoreRequest(query="cat"))
    assert len(res.documents) == 1
    assert res.documents[0].content == "the cat sat"
    assert res.documents[0].score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_grpc_search_with_metadata_filter(grpc_channel):
    stub = vector_pb2_grpc.VectorServiceStub(grpc_channel)
    await stub.AddDocuments(
        vector_pb2.AddDocumentsRequest(
            documents=[_doc("alpha doc", topic="a"), _doc("beta doc", topic="b")]
        )
    )
    mf = struct_pb2.Struct()
    mf.update({"topic": "b"})
    res = await stub.SearchVectorStore(
        vector_pb2.SearchVectorStoreRequest(query="doc", metadata_filter=mf)
    )
    assert [d.content for d in res.documents] == ["beta doc"]


@pytest.mark.asyncio
async def test_grpc_search_respects_k(grpc_channel):
    stub = vector_pb2_grpc.VectorServiceStub(grpc_channel)
    await stub.AddDocuments(
        vector_pb2.AddDocumentsRequest(documents=[_doc(f"doc {i}") for i in range(5)])
    )
    res = await stub.SearchVectorStore(vector_pb2.SearchVectorStoreRequest(query="doc", k=2))
    assert len(res.documents) == 2


@pytest.mark.asyncio
async def test_grpc_collections_and_info(grpc_channel):
    stub = vector_pb2_grpc.VectorServiceStub(grpc_channel)
    await stub.AddDocuments(
        vector_pb2.AddDocumentsRequest(documents=[_doc("x")], collection_name="kb")
    )
    cols = await stub.ListVectorCollections(common_pb2.Empty())
    assert "kb" in cols.collections
    rag = await stub.ListRagCollections(common_pb2.Empty())
    assert "kb" in rag.collections
    info = await stub.GetRagCollectionInfo(
        vector_pb2.GetRagCollectionInfoRequest(collection_name="kb")
    )
    assert info.collection_name == "kb"
    assert MessageToDict(info.info, preserving_proto_field_name=True)["document_count"] == 1


@pytest.mark.asyncio
async def test_grpc_delete_collection(grpc_channel):
    stub = vector_pb2_grpc.VectorServiceStub(grpc_channel)
    await stub.AddDocuments(
        vector_pb2.AddDocumentsRequest(documents=[_doc("x")], collection_name="tmp")
    )
    res = await stub.DeleteRagCollection(
        vector_pb2.DeleteRagCollectionRequest(collection_name="tmp", force=True)
    )
    assert res.deleted is True
    again = await stub.DeleteRagCollection(
        vector_pb2.DeleteRagCollectionRequest(collection_name="tmp")
    )
    assert again.deleted is False


@pytest.mark.asyncio
async def test_grpc_get_missing_collection_is_not_found(grpc_channel):
    stub = vector_pb2_grpc.VectorServiceStub(grpc_channel)
    with pytest.raises(grpc.aio.AioRpcError) as ei:
        await stub.GetRagCollectionInfo(
            vector_pb2.GetRagCollectionInfoRequest(collection_name="ghost")
        )
    assert ei.value.code() == grpc.StatusCode.NOT_FOUND


@pytest.mark.asyncio
async def test_grpc_getinfo_advertises_tier1_vector_when_enabled(grpc_channel, monkeypatch):
    monkeypatch.setenv("LLMCORE_BRIDGE_FAKE_VECTOR", "1")
    stub = control_pb2_grpc.ControlServiceStub(grpc_channel)
    info = await stub.GetInfo(common_pb2.Empty())
    assert "tier1.vector" in info.capabilities
    assert "vector.search" in info.capabilities
    assert "T1" in info.tiers


# --------------------------------------------------------------------------- #
# HTTP / JSON
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_http_add_and_search(http_client):
    r = await http_client.post(
        f"{_P}/AddDocuments",
        json={"documents": [{"content": "hello world", "metadata": {"topic": "greet"}}]},
    )
    assert r.status_code == 200
    assert len(r.json()["ids"]) == 1
    r2 = await http_client.post(f"{_P}/SearchVectorStore", json={"query": "hello", "k": 3})
    assert r2.status_code == 200
    docs = r2.json()["documents"]
    assert docs[0]["content"] == "hello world"
    assert docs[0]["metadata"]["topic"] == "greet"


@pytest.mark.asyncio
async def test_http_missing_collection_is_404(http_client):
    r = await http_client.post(
        f"{_P}/GetRagCollectionInfo", json={"collection_name": "ghost"}
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found.rag_collection"


@pytest.mark.asyncio
async def test_http_list_collections_empty(http_client):
    r = await http_client.post(f"{_P}/ListVectorCollections", json={})
    assert r.status_code == 200
    assert r.json().get("collections", []) == []
