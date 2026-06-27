"""Real-``LLMCore`` integration (skip-guarded).

This drives the *actual* ``LLMCore.create()`` construction path — config load,
provider wiring via ``PROVIDER_MAP`` — with an offline :class:`FakeProvider`,
then exercises :class:`BridgeCore` against it. It is the end-to-end proof that
the bridge's thin-adapter calls line up with the real facade.

It is **skipped** wherever the sovereign ``confy>=0.4.2`` dependency is absent
(``LLMCore.create()`` needs confy at runtime). To run it in the llmcore dev
environment::

    pip install -e ".[bridge]"        # confy>=0.4.2 resolvable there
    pytest tests/bridge/test_integration_real_llmcore.py -q

The conformance suites (``test_conformance_*.py``) + the contract guard are the
authoritative, dependency-free B1 proof; this test is the additional real-path
confirmation.
"""

from __future__ import annotations

import pytest

try:
    import confy  # noqa: F401  (sovereign; only present in the llmcore dev env)

    _CONFY = True
except Exception:
    _CONFY = False

pytestmark = pytest.mark.skipif(
    not _CONFY,
    reason="requires confy>=0.4.2 (sovereign, tarball-only); run in the llmcore dev env",
)


@pytest.mark.asyncio
async def test_bridge_over_real_llmcore():
    from llmcore import LLMCore
    from llmcore.bridge._generated.llmcore.v1 import common_pb2, inference_pb2
    from llmcore.bridge._testing import register_fake_provider
    from llmcore.bridge.core import BridgeCore

    register_fake_provider()
    core_llm = await LLMCore.create(
        config_overrides={
            "providers": {"fake": {"type": "fake", "default_model": "fake-1"}},
            "llmcore": {"default_provider": "fake", "default_model": "fake-1"},
        }
    )
    try:
        bridge = BridgeCore(core_llm, transports=("grpc",))

        # Unary chat through the real facade.
        resp = await bridge.chat(inference_pb2.ChatRequest(message="hello real", save_session=False))
        assert "echo: hello real" in resp.text

        # Streaming concatenates to the same content.
        chunks = []
        async for chunk in bridge.chat_stream(
            inference_pb2.ChatRequest(message="stream real", save_session=False)
        ):
            if not chunk.done:
                chunks.append(chunk.text)
        assert "echo: stream real" in "".join(chunks)

        # Catalog reflects the configured provider.
        providers = await bridge.list_providers(common_pb2.Empty())
        assert "fake" in list(providers.providers)
    finally:
        await core_llm.close()
