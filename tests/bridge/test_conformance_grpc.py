"""gRPC conformance — exercises the full primary transport over a real channel."""

from __future__ import annotations

import asyncio

import grpc
import pytest

from llmcore.bridge._generated.llmcore.v1 import (
    catalog_pb2,
    catalog_pb2_grpc,
    common_pb2,
    control_pb2_grpc,
    errors_pb2,
    inference_pb2,
    inference_pb2_grpc,
)
from llmcore.bridge.grpc_server import ERROR_METADATA_KEY


def _error_from_trailing(rpc_error: grpc.aio.AioRpcError) -> errors_pb2.LlmcoreError:
    for key, value in rpc_error.trailing_metadata() or ():
        if key == ERROR_METADATA_KEY:
            return errors_pb2.LlmcoreError.FromString(value)
    raise AssertionError("no llmcore-error-bin trailing metadata on error")


@pytest.mark.asyncio
async def test_chat(grpc_channel):
    stub = inference_pb2_grpc.InferenceServiceStub(grpc_channel)
    resp = await stub.Chat(inference_pb2.ChatRequest(message="hello world"))
    assert resp.text == "echo: hello world"
    assert resp.usage.prompt_tokens == 2
    assert resp.usage.total_tokens == resp.usage.prompt_tokens + resp.usage.completion_tokens
    assert resp.usage.provider == "fake"


@pytest.mark.asyncio
async def test_chat_stream_concatenates_to_unary(grpc_channel):
    stub = inference_pb2_grpc.InferenceServiceStub(grpc_channel)
    chunks, done = [], False
    async for chunk in stub.ChatStream(inference_pb2.ChatRequest(message="streaming please")):
        if chunk.done:
            done = True
        else:
            chunks.append(chunk.text)
    assert done is True
    assert "".join(chunks) == "echo: streaming please"


@pytest.mark.asyncio
async def test_count_tokens(grpc_channel):
    stub = inference_pb2_grpc.InferenceServiceStub(grpc_channel)
    resp = await stub.CountTokens(inference_pb2.CountTokensRequest(text="one two three"))
    assert resp.tokens == 3


@pytest.mark.asyncio
async def test_estimate_cost(grpc_channel):
    stub = inference_pb2_grpc.InferenceServiceStub(grpc_channel)
    resp = await stub.EstimateCost(
        inference_pb2.EstimateCostRequest(
            provider_name="fake", model_name="fake-1",
            prompt_tokens=1_000_000, completion_tokens=1_000_000,
        )
    )
    assert resp.currency == "USD"
    assert resp.input_cost == pytest.approx(1.0)
    assert resp.output_cost == pytest.approx(2.0)
    assert resp.total_cost == pytest.approx(3.0)
    assert resp.model_id == "fake-1"


@pytest.mark.asyncio
async def test_catalog(grpc_channel):
    stub = catalog_pb2_grpc.CatalogServiceStub(grpc_channel)
    providers = await stub.ListProviders(common_pb2.Empty())
    assert list(providers.providers) == ["fake"]
    models = await stub.ListModels(catalog_pb2.ListModelsRequest(provider_name="fake"))
    assert list(models.models) == ["fake-1", "fake-2"]
    details = await stub.GetProviderDetails(catalog_pb2.GetProviderRequest(provider_name="fake"))
    assert details.id == "fake-1"
    assert details.context_length == 8192
    assert details.supports_tools is True
    assert details.metadata["vendor"] == "fake"


@pytest.mark.asyncio
async def test_control_info_and_health(grpc_channel):
    stub = control_pb2_grpc.ControlServiceStub(grpc_channel)
    info = await stub.GetInfo(common_pb2.Empty())
    assert info.contract_version == "llmcore.v1"
    assert "tier0" in info.capabilities
    assert "transport.grpc" in info.capabilities
    # Tier-2 must NOT be advertised in v1.
    assert not any(c.startswith("tier2.") for c in info.capabilities)
    assert any(p.name == "fake" and p.available for p in info.providers)
    health = await stub.Health(common_pb2.Empty())
    assert health.ok is True


@pytest.mark.asyncio
async def test_embed_unimplemented(grpc_channel):
    stub = inference_pb2_grpc.InferenceServiceStub(grpc_channel)
    with pytest.raises(grpc.aio.AioRpcError) as ei:
        await stub.Embed(inference_pb2.EmbedRequest(input=["x"]))
    assert ei.value.code() == grpc.StatusCode.UNIMPLEMENTED
    err = _error_from_trailing(ei.value)
    assert err.category == errors_pb2.ErrorCategory.ERROR_CATEGORY_UNSUPPORTED
    assert err.code == "unsupported.capability"


@pytest.mark.asyncio
async def test_error_provider_rate_limited(grpc_channel):
    stub = inference_pb2_grpc.InferenceServiceStub(grpc_channel)
    with pytest.raises(grpc.aio.AioRpcError) as ei:
        await stub.Chat(inference_pb2.ChatRequest(message="__error__:provider_rate_limited"))
    assert ei.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
    err = _error_from_trailing(ei.value)
    assert err.category == errors_pb2.ErrorCategory.ERROR_CATEGORY_PROVIDER
    assert err.code == "provider.rate_limited"
    assert err.http_status == 429
    assert err.retryable is True
    assert err.retry_after_ms == pytest.approx(2000.0)
    assert err.provider == "fake"


@pytest.mark.asyncio
async def test_error_unauthorized_maps_to_unauthenticated(grpc_channel):
    stub = inference_pb2_grpc.InferenceServiceStub(grpc_channel)
    with pytest.raises(grpc.aio.AioRpcError) as ei:
        await stub.Chat(inference_pb2.ChatRequest(message="__error__:provider_unauthorized"))
    assert ei.value.code() == grpc.StatusCode.UNAUTHENTICATED
    err = _error_from_trailing(ei.value)
    assert err.http_status == 401
    assert err.retryable is False


@pytest.mark.asyncio
async def test_error_mid_stream_terminates(grpc_channel):
    stub = inference_pb2_grpc.InferenceServiceStub(grpc_channel)
    received = []
    with pytest.raises(grpc.aio.AioRpcError) as ei:
        async for chunk in stub.ChatStream(
            inference_pb2.ChatRequest(message="__error_mid__:internal")
        ):
            received.append(chunk.text)
    assert received  # at least one chunk before the failure
    assert ei.value.code() == grpc.StatusCode.INTERNAL


@pytest.mark.asyncio
async def test_cancellation(grpc_channel):
    stub = inference_pb2_grpc.InferenceServiceStub(grpc_channel)
    call = stub.ChatStream(inference_pb2.ChatRequest(message="__cancel__"))
    it = call.__aiter__()
    first = await asyncio.wait_for(it.__anext__(), timeout=5)
    assert first.text  # streaming started
    call.cancel()
    # grpc.aio surfaces a locally-applied cancellation as CancelledError; the
    # RPC's reported status is CANCELLED. Both are asserted.
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(it.__anext__(), timeout=5)
    assert call.cancelled()
    assert await call.code() == grpc.StatusCode.CANCELLED
