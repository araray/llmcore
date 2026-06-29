"""End-to-end tests for the Tier-1 PresetService RPCs (phase B4).

Exercised over **both** transports — gRPC (``grpc_channel``) and HTTP/JSON
(``http_client``) — against the deterministic in-memory preset store in
``FakeFacade``. SaveContextPreset is the one inbound model conversion in T1
(proto ContextPreset -> llmcore.models.ContextPreset), so the round-trip
verifies both directions.
"""

from __future__ import annotations

import grpc
import pytest

from llmcore.bridge._generated.llmcore.v1 import (
    common_pb2,
    control_pb2_grpc,
    presets_pb2,
    presets_pb2_grpc,
)

_P = "/llmcore.v1/PresetService"


def _preset(name: str) -> presets_pb2.ContextPreset:
    p = presets_pb2.ContextPreset(name=name, description="a preset")
    item = p.items.add()
    item.type = "preset_text_content"
    item.content = "boilerplate"
    item.source_identifier = "tpl-1"
    p.metadata.update({"team": "core"})
    return p


# --------------------------------------------------------------------------- #
# gRPC
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_grpc_save_get_roundtrip(grpc_channel):
    stub = presets_pb2_grpc.PresetServiceStub(grpc_channel)
    await stub.SaveContextPreset(
        presets_pb2.SaveContextPresetRequest(preset=_preset("p1"))
    )
    got = await stub.GetContextPreset(
        presets_pb2.GetContextPresetRequest(preset_name="p1")
    )
    assert got.name == "p1"
    assert got.description == "a preset"
    assert got.metadata["team"] == "core"
    assert len(got.items) == 1
    assert got.items[0].type == "preset_text_content"
    assert got.items[0].content == "boilerplate"
    assert got.items[0].source_identifier == "tpl-1"


@pytest.mark.asyncio
async def test_grpc_list_and_delete(grpc_channel):
    stub = presets_pb2_grpc.PresetServiceStub(grpc_channel)
    await stub.SaveContextPreset(presets_pb2.SaveContextPresetRequest(preset=_preset("p1")))
    await stub.SaveContextPreset(presets_pb2.SaveContextPresetRequest(preset=_preset("p2")))
    listed = await stub.ListContextPresets(common_pb2.Empty())
    names = {dict(s.fields)["name"].string_value for s in listed.presets}
    assert names == {"p1", "p2"}
    res = await stub.DeleteContextPreset(
        presets_pb2.DeleteContextPresetRequest(preset_name="p1")
    )
    assert res.deleted is True
    again = await stub.DeleteContextPreset(
        presets_pb2.DeleteContextPresetRequest(preset_name="p1")
    )
    assert again.deleted is False


@pytest.mark.asyncio
async def test_grpc_get_missing_preset_is_not_found(grpc_channel):
    stub = presets_pb2_grpc.PresetServiceStub(grpc_channel)
    with pytest.raises(grpc.aio.AioRpcError) as ei:
        await stub.GetContextPreset(
            presets_pb2.GetContextPresetRequest(preset_name="ghost")
        )
    assert ei.value.code() == grpc.StatusCode.NOT_FOUND


@pytest.mark.asyncio
async def test_grpc_save_requires_name(grpc_channel):
    stub = presets_pb2_grpc.PresetServiceStub(grpc_channel)
    with pytest.raises(grpc.aio.AioRpcError) as ei:
        await stub.SaveContextPreset(
            presets_pb2.SaveContextPresetRequest(preset=presets_pb2.ContextPreset())
        )
    assert ei.value.code() == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.asyncio
async def test_grpc_save_bad_item_type_is_invalid_argument(grpc_channel):
    stub = presets_pb2_grpc.PresetServiceStub(grpc_channel)
    p = presets_pb2.ContextPreset(name="p1")
    p.items.add().type = "not_a_real_type"
    with pytest.raises(grpc.aio.AioRpcError) as ei:
        await stub.SaveContextPreset(presets_pb2.SaveContextPresetRequest(preset=p))
    assert ei.value.code() == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.asyncio
async def test_grpc_getinfo_advertises_presets_when_sessions_enabled(grpc_channel, monkeypatch):
    monkeypatch.setenv("LLMCORE_BRIDGE_FAKE_SESSIONS", "1")
    stub = control_pb2_grpc.ControlServiceStub(grpc_channel)
    info = await stub.GetInfo(common_pb2.Empty())
    assert "presets.save" in info.capabilities


# --------------------------------------------------------------------------- #
# HTTP / JSON
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_http_save_get_roundtrip(http_client):
    body = {
        "preset": {
            "name": "http-preset",
            "description": "via http",
            "items": [
                {
                    "type": "preset_text_content",
                    "content": "hello",
                    "source_identifier": "s1",
                }
            ],
            "metadata": {"k": "v"},
        }
    }
    r = await http_client.post(f"{_P}/SaveContextPreset", json=body)
    assert r.status_code == 200
    r2 = await http_client.post(
        f"{_P}/GetContextPreset", json={"preset_name": "http-preset"}
    )
    assert r2.status_code == 200
    got = r2.json()
    assert got["name"] == "http-preset"
    assert got["items"][0]["content"] == "hello"
    assert got["metadata"]["k"] == "v"


@pytest.mark.asyncio
async def test_http_missing_preset_is_404(http_client):
    r = await http_client.post(f"{_P}/GetContextPreset", json={"preset_name": "ghost"})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found.context_preset"
