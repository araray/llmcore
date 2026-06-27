"""Shared fixtures for the bridge conformance suite.

The gRPC fixture binds a real ``grpc.aio`` server to a Unix-domain socket and
connects a real channel (full HTTP/2 + protobuf framing). The HTTP fixture
drives the Starlette app through ``httpx.ASGITransport`` (full ASGI path). Both
are backed by the identical :class:`BridgeCore`/:class:`FakeFacade`, which is
what lets ``test_transport_parity`` compare them.
"""

from __future__ import annotations

import grpc
import httpx
import pytest_asyncio

from llmcore.bridge._testing import FakeFacade, fake_count_tokens
from llmcore.bridge.core import BridgeCore
from llmcore.bridge.grpc_server import create_grpc_server
from llmcore.bridge.http_app import create_http_app


def make_core() -> BridgeCore:
    return BridgeCore(
        FakeFacade(), count_tokens=fake_count_tokens, transports=("grpc", "http")
    )


@pytest_asyncio.fixture
async def grpc_channel(tmp_path):
    core = make_core()
    server = create_grpc_server(core)
    address = f"unix:{tmp_path}/bridge.sock"
    server.add_insecure_port(address)
    await server.start()
    try:
        async with grpc.aio.insecure_channel(address) as channel:
            yield channel
    finally:
        await server.stop(grace=None)


@pytest_asyncio.fixture
async def http_client():
    core = make_core()
    app = create_http_app(core)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://bridge") as client:
        yield client
