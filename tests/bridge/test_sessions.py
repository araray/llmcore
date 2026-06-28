"""End-to-end tests for the Tier-1 SessionService RPCs (phase B4).

Each operation is exercised over **both** transports — gRPC (the ``grpc_channel``
fixture: real ``grpc.aio`` server on a Unix socket) and HTTP/JSON (the
``http_client`` fixture: Starlette app via ``httpx.ASGITransport``) — against the
deterministic in-memory store in ``FakeFacade``. Unlike Tier-2 audio, the session
handlers are not capability-gated (a real ``LLMCore`` always supports sessions);
``LLMCORE_BRIDGE_FAKE_SESSIONS=1`` only toggles the ``tier1.*`` capability
advertisement in ``ServerInfo``, which the GetInfo tests below verify.
"""

from __future__ import annotations

import grpc
import pytest

from llmcore.bridge._generated.llmcore.v1 import (
    common_pb2,
    control_pb2_grpc,
    sessions_pb2,
    sessions_pb2_grpc,
)

_P = "/llmcore.v1/SessionService"


# --------------------------------------------------------------------------- #
# gRPC
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_grpc_create_get_roundtrip(grpc_channel):
    stub = sessions_pb2_grpc.SessionServiceStub(grpc_channel)
    created = await stub.CreateSession(
        sessions_pb2.CreateSessionRequest(name="chat-a", system_message="be brief")
    )
    assert created.id
    assert created.name == "chat-a"
    assert created.created_at  # ISO timestamp populated
    assert len(created.messages) == 1
    assert created.messages[0].role == common_pb2.ROLE_SYSTEM
    assert created.messages[0].content == "be brief"

    fetched = await stub.GetSession(sessions_pb2.GetSessionRequest(session_id=created.id))
    assert fetched.id == created.id
    assert fetched.name == "chat-a"


@pytest.mark.asyncio
async def test_grpc_list_with_limit(grpc_channel):
    stub = sessions_pb2_grpc.SessionServiceStub(grpc_channel)
    for _ in range(3):
        await stub.CreateSession(sessions_pb2.CreateSessionRequest())
    full = await stub.ListSessions(sessions_pb2.ListSessionsRequest())
    assert len(full.sessions) == 3
    limited = await stub.ListSessions(sessions_pb2.ListSessionsRequest(limit=2))
    assert len(limited.sessions) == 2


@pytest.mark.asyncio
async def test_grpc_context_item_lifecycle(grpc_channel):
    stub = sessions_pb2_grpc.SessionServiceStub(grpc_channel)
    sess = await stub.CreateSession(sessions_pb2.CreateSessionRequest())
    added = await stub.AddContextItem(
        sessions_pb2.AddContextItemRequest(
            session_id=sess.id, content="a fact", type="rag_snippet", source_id="doc-1"
        )
    )
    assert added.item_id
    item = await stub.GetContextItem(
        sessions_pb2.GetContextItemRequest(session_id=sess.id, item_id=added.item_id)
    )
    assert item.type == "rag_snippet"
    assert item.content == "a fact"
    assert item.source_id == "doc-1"
    assert item.tokens == 2  # whitespace token count of "a fact"

    removed = await stub.RemoveContextItem(
        sessions_pb2.RemoveContextItemRequest(session_id=sess.id, item_id=added.item_id)
    )
    assert removed.removed is True


@pytest.mark.asyncio
async def test_grpc_fork_includes_context_items(grpc_channel):
    stub = sessions_pb2_grpc.SessionServiceStub(grpc_channel)
    sess = await stub.CreateSession(sessions_pb2.CreateSessionRequest(name="src"))
    await stub.AddContextItem(
        sessions_pb2.AddContextItemRequest(session_id=sess.id, content="keep me")
    )
    forked = await stub.ForkSession(
        sessions_pb2.ForkSessionRequest(session_id=sess.id, include_context_items=True)
    )
    assert forked.session_id and forked.session_id != sess.id
    got = await stub.GetSession(sessions_pb2.GetSessionRequest(session_id=forked.session_id))
    assert len(got.context_items) == 1
    assert got.context_items[0].content == "keep me"


@pytest.mark.asyncio
async def test_grpc_clone_without_messages(grpc_channel):
    stub = sessions_pb2_grpc.SessionServiceStub(grpc_channel)
    sess = await stub.CreateSession(
        sessions_pb2.CreateSessionRequest(system_message="sys")
    )
    cloned = await stub.CloneSession(
        sessions_pb2.CloneSessionRequest(
            session_id=sess.id, include_messages=False, include_context_items=True
        )
    )
    got = await stub.GetSession(sessions_pb2.GetSessionRequest(session_id=cloned.session_id))
    assert len(got.messages) == 0


@pytest.mark.asyncio
async def test_grpc_update_name_and_delete(grpc_channel):
    stub = sessions_pb2_grpc.SessionServiceStub(grpc_channel)
    sess = await stub.CreateSession(sessions_pb2.CreateSessionRequest(name="old"))
    await stub.UpdateSessionName(
        sessions_pb2.UpdateSessionNameRequest(session_id=sess.id, new_name="new")
    )
    got = await stub.GetSession(sessions_pb2.GetSessionRequest(session_id=sess.id))
    assert got.name == "new"
    await stub.DeleteSession(sessions_pb2.DeleteSessionRequest(session_id=sess.id))
    with pytest.raises(grpc.aio.AioRpcError) as ei:
        await stub.GetSession(sessions_pb2.GetSessionRequest(session_id=sess.id))
    assert ei.value.code() == grpc.StatusCode.NOT_FOUND


@pytest.mark.asyncio
async def test_grpc_get_missing_session_is_not_found(grpc_channel):
    stub = sessions_pb2_grpc.SessionServiceStub(grpc_channel)
    with pytest.raises(grpc.aio.AioRpcError) as ei:
        await stub.GetSession(sessions_pb2.GetSessionRequest(session_id="ghost"))
    assert ei.value.code() == grpc.StatusCode.NOT_FOUND


@pytest.mark.asyncio
async def test_grpc_get_missing_context_item_is_not_found(grpc_channel):
    stub = sessions_pb2_grpc.SessionServiceStub(grpc_channel)
    sess = await stub.CreateSession(sessions_pb2.CreateSessionRequest())
    with pytest.raises(grpc.aio.AioRpcError) as ei:
        await stub.GetContextItem(
            sessions_pb2.GetContextItemRequest(session_id=sess.id, item_id="nope")
        )
    assert ei.value.code() == grpc.StatusCode.NOT_FOUND


@pytest.mark.asyncio
async def test_grpc_add_context_item_bad_type_is_invalid_argument(grpc_channel):
    stub = sessions_pb2_grpc.SessionServiceStub(grpc_channel)
    sess = await stub.CreateSession(sessions_pb2.CreateSessionRequest())
    with pytest.raises(grpc.aio.AioRpcError) as ei:
        await stub.AddContextItem(
            sessions_pb2.AddContextItemRequest(
                session_id=sess.id, content="x", type="not_a_real_type"
            )
        )
    assert ei.value.code() == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.asyncio
async def test_grpc_getinfo_advertises_tier1_when_enabled(grpc_channel, monkeypatch):
    monkeypatch.setenv("LLMCORE_BRIDGE_FAKE_SESSIONS", "1")
    stub = control_pb2_grpc.ControlServiceStub(grpc_channel)
    info = await stub.GetInfo(common_pb2.Empty())
    assert "tier1.sessions" in info.capabilities
    assert "sessions.create" in info.capabilities
    assert "T1" in info.tiers


@pytest.mark.asyncio
async def test_grpc_getinfo_omits_tier1_when_disabled(grpc_channel, monkeypatch):
    monkeypatch.delenv("LLMCORE_BRIDGE_FAKE_SESSIONS", raising=False)
    stub = control_pb2_grpc.ControlServiceStub(grpc_channel)
    info = await stub.GetInfo(common_pb2.Empty())
    assert not any(c.startswith("tier1.") for c in info.capabilities)
    assert "T1" not in info.tiers


# --------------------------------------------------------------------------- #
# HTTP / JSON
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_http_create_get_roundtrip(http_client):
    r = await http_client.post(f"{_P}/CreateSession", json={"name": "via-http"})
    assert r.status_code == 200
    created = r.json()
    assert created["name"] == "via-http"
    sid = created["id"]

    r2 = await http_client.post(f"{_P}/GetSession", json={"session_id": sid})
    assert r2.status_code == 200
    assert r2.json()["id"] == sid


@pytest.mark.asyncio
async def test_http_context_item_roundtrip(http_client):
    sid = (await http_client.post(f"{_P}/CreateSession", json={})).json()["id"]
    r = await http_client.post(
        f"{_P}/AddContextItem",
        json={"session_id": sid, "content": "hello world", "type": "user_text"},
    )
    assert r.status_code == 200
    item_id = r.json()["item_id"]
    r2 = await http_client.post(
        f"{_P}/GetContextItem", json={"session_id": sid, "item_id": item_id}
    )
    assert r2.status_code == 200
    body = r2.json()
    assert body["content"] == "hello world"
    assert body["type"] == "user_text"
    assert body["tokens"] == 2


@pytest.mark.asyncio
async def test_http_missing_session_is_404(http_client):
    r = await http_client.post(f"{_P}/GetSession", json={"session_id": "ghost"})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found.session"


@pytest.mark.asyncio
async def test_http_missing_context_item_is_404(http_client):
    sid = (await http_client.post(f"{_P}/CreateSession", json={})).json()["id"]
    r = await http_client.post(
        f"{_P}/GetContextItem", json={"session_id": sid, "item_id": "nope"}
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found.context_item"


@pytest.mark.asyncio
async def test_http_getinfo_advertises_tier1_when_enabled(http_client, monkeypatch):
    monkeypatch.setenv("LLMCORE_BRIDGE_FAKE_SESSIONS", "1")
    r = await http_client.post("/llmcore.v1/ControlService/GetInfo", json={})
    assert r.status_code == 200
    caps = r.json()["capabilities"]
    assert "tier1.sessions" in caps
    assert "T1" in r.json()["tiers"]
