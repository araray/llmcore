"""The gpu.ai provider: catalogue filtering, context, embeddings.

gpu.ai sells text inference, media generation and GPU rental. This provider
covers the text surface only; the rental surface has no backend because
`runtimes/` has only ever had one (Colab), which is the same gap DeepInfra
has.

Its `/v1/models` is the richest catalogue of any provider integrated here —
`context_length`, `pricing`, `supported_parameters` and `aliases` inline —
which is what these tests mostly pin: that the richness is used, and that
two thirds of the catalogue (image and video) is filtered out rather than
offered to a router that cannot call it.

Offline by design. Two live behaviours are asserted only in comments
because reproducing them needs the API: `max_tokens=8` returns empty
content with 8 completion tokens consumed (reasoning eats the budget), and
`max_tokens=256` returns "OK" with 45 — both observed.
"""

from __future__ import annotations

import json

import pytest

from llmcore.providers.gpuai_provider import (
    CHAT_MODALITIES,
    DEFAULT_GPUAI_BASE_URL,
    DEFAULT_GPUAI_MODEL,
    MIN_REASONING_MAX_TOKENS,
    GpuAiProvider,
)

CATALOGUE = {"data": [
    {"id": "gpuai/qwen3.8-flash", "object": "model", "owned_by": "gpuai",
     "name": "Qwen3.8 Flash", "author": "Qwen", "modality": "chat",
     "category": "chat", "tier": "serverless", "status": "active",
     "context_length": 262144, "aliases": ["qwen-flash"],
     "supported_parameters": ["messages", "temperature", "max_tokens", "stream"],
     "pricing": {"currency": "usd", "input_per_1m_tokens_cents": 15,
                 "output_per_1m_tokens_cents": 47}},
    {"id": "gpuai/no-stream-model", "object": "model", "owned_by": "gpuai",
     "name": "No Stream", "modality": "chat", "status": "active",
     "context_length": 8192,
     "supported_parameters": ["messages", "temperature"],
     "pricing": {"currency": "usd", "input_per_1m_tokens_cents": 10}},
    {"id": "gpuai/qwen3-embedding-8b", "object": "model", "owned_by": "gpuai",
     "name": "Qwen3 Embedding 8B", "modality": "embedding", "status": "active",
     "context_length": 40960,
     "pricing": {"currency": "usd", "input_per_1m_tokens_cents": 10}},
    # Two thirds of the real catalogue looks like this.
    {"id": "gpuai/happyhorse-1.0-t2v", "object": "model", "modality": "video",
     "name": "Happyhorse T2V", "status": "active",
     "supported_parameters": ["model", "prompt", "seconds"],
     "pricing": {"currency": "usd", "per_video_second_cents": 24}},
    {"id": "gpuai/some-image-model", "object": "model", "modality": "image",
     "name": "Some Image Model", "status": "active",
     "pricing": {"currency": "usd", "per_image_cents": 3}},
]}


class _Response:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _Http:
    """Stands in for the direct httpx client."""

    def __init__(self, get_payload=None, post_payload=None, status=200):
        self._get = get_payload
        self._post = post_payload
        self._status = status
        self.posted: list[tuple[str, dict]] = []

    async def get(self, path):
        return _Response(self._get, self._status)

    async def post(self, path, json=None):
        self.posted.append((path, json or {}))
        return _Response(self._post, self._status)


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv("GPUAI_API_KEY", "test-key")
    p = GpuAiProvider({})
    p._transport = "httpx"
    return p


class TestIdentityAndDefaults:
    def test_name_and_base_url(self, provider):
        assert provider.get_name() == "gpuai"
        assert provider.base_url == DEFAULT_GPUAI_BASE_URL
        assert provider.default_model == DEFAULT_GPUAI_MODEL

    def test_the_key_is_read_from_the_environment(self, monkeypatch):
        monkeypatch.delenv("GPUAI_API_KEY", raising=False)
        monkeypatch.setenv("GPU_AI_API_KEY", "alternate")
        assert GpuAiProvider({}).api_key == "alternate"

    def test_no_media_capabilities_are_claimed(self, provider):
        # gpu.ai *has* image and video models; this provider does not serve
        # them, and claiming otherwise would make them routable.
        assert provider._MEDIA_CAPABILITIES == frozenset()


class TestCatalogue:
    async def _models(self, provider, monkeypatch, payload=CATALOGUE):
        monkeypatch.setattr(provider, "_get_http", lambda: _Http(get_payload=payload))
        return await provider.get_models_details()

    @pytest.mark.asyncio
    async def test_image_and_video_models_are_filtered_out(self, provider, monkeypatch):
        models = await self._models(provider, monkeypatch)
        ids = {m.id for m in models}
        assert "gpuai/happyhorse-1.0-t2v" not in ids
        assert "gpuai/some-image-model" not in ids
        assert len(models) == 3

    @pytest.mark.asyncio
    async def test_chat_and_embedding_are_distinguished(self, provider, monkeypatch):
        models = await self._models(provider, monkeypatch)
        kinds = {m.id: m.model_type for m in models}
        assert kinds["gpuai/qwen3.8-flash"] == "chat"
        assert kinds["gpuai/qwen3-embedding-8b"] == "embedding"

    @pytest.mark.asyncio
    async def test_the_context_window_comes_from_the_catalogue(self, provider, monkeypatch):
        models = await self._models(provider, monkeypatch)
        flash = next(m for m in models if m.id == "gpuai/qwen3.8-flash")
        assert flash.context_length == 262144

    @pytest.mark.asyncio
    async def test_streaming_absence_is_a_real_negative(self, provider, monkeypatch):
        # The catalogue enumerates accepted parameters, so a missing
        # `stream` means streaming is not accepted -- unlike a capability
        # stub on a generated card, where false means "unfilled".
        models = await self._models(provider, monkeypatch)
        by_id = {m.id: m for m in models}
        assert by_id["gpuai/qwen3.8-flash"].supports_streaming is True
        assert by_id["gpuai/no-stream-model"].supports_streaming is False

    @pytest.mark.asyncio
    async def test_the_rich_fields_are_kept_for_card_generation(self, provider, monkeypatch):
        models = await self._models(provider, monkeypatch)
        meta = next(m for m in models if m.id == "gpuai/qwen3.8-flash").metadata
        assert meta["aliases"] == ["qwen-flash"]
        assert meta["author"] == "Qwen"
        assert meta["pricing_raw"]["input_per_1m_tokens_cents"] == 15
        assert "stream" in meta["supported_parameters"]

    @pytest.mark.asyncio
    async def test_an_entry_without_an_id_is_skipped(self, provider, monkeypatch):
        models = await self._models(provider, monkeypatch,
                                    {"data": [{"modality": "chat"}]})
        assert models == []

    @pytest.mark.asyncio
    async def test_an_http_failure_raises_rather_than_returning_empty(
        self, provider, monkeypatch
    ):
        from llmcore.exceptions import LLMCoreError

        monkeypatch.setattr(
            provider, "_get_http",
            lambda: _Http(get_payload={"error": "nope"}, status=500))
        with pytest.raises(LLMCoreError):
            await provider.get_models_details()


class TestContextLookup:
    @pytest.mark.asyncio
    async def test_discovery_populates_the_lookup_including_aliases(
        self, provider, monkeypatch
    ):
        monkeypatch.setattr(provider, "_get_http", lambda: _Http(get_payload=CATALOGUE))
        await provider.get_models_details()
        assert provider.get_max_context_length("gpuai/qwen3.8-flash") == 262144
        assert provider.get_max_context_length("qwen-flash") == 262144

    def test_an_unknown_model_falls_back_to_the_configured_default(self, monkeypatch):
        monkeypatch.setenv("GPUAI_API_KEY", "k")
        p = GpuAiProvider({"default_context_length": 4096})
        assert p.get_max_context_length("gpuai/never-seen") == 4096

    def test_a_bad_configured_default_does_not_raise(self, monkeypatch):
        monkeypatch.setenv("GPUAI_API_KEY", "k")
        p = GpuAiProvider({"default_context_length": "not a number"})
        assert p.get_max_context_length("x") > 0


class TestEmbeddings:
    @pytest.mark.asyncio
    async def test_embeddings_are_returned_as_plain_lists(self, provider, monkeypatch):
        http = _Http(post_payload={"data": [{"embedding": [0.1, 0.2]},
                                            {"embedding": [0.3, 0.4]}]})
        monkeypatch.setattr(provider, "_get_http", lambda: http)
        assert await provider.create_embeddings(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]

    @pytest.mark.asyncio
    async def test_encoding_format_is_never_sent(self, provider, monkeypatch):
        # gpu.ai rejects the *parameter*, not a value:
        #   Parameter "encoding_format" is not supported in v1.1
        # The OpenAI SDK injects it, which is why this path builds the
        # payload itself.
        http = _Http(post_payload={"data": [{"embedding": [0.0]}]})
        monkeypatch.setattr(provider, "_get_http", lambda: http)
        await provider.create_embeddings(["a"])
        _path, body = http.posted[0]
        assert "encoding_format" not in body
        assert set(body) == {"model", "input"}

    @pytest.mark.asyncio
    async def test_an_empty_input_makes_no_request(self, provider, monkeypatch):
        http = _Http(post_payload={"data": []})
        monkeypatch.setattr(provider, "_get_http", lambda: http)
        assert await provider.create_embeddings([]) == []
        assert http.posted == []

    @pytest.mark.asyncio
    async def test_a_failure_raises(self, provider, monkeypatch):
        from llmcore.exceptions import LLMCoreError

        monkeypatch.setattr(
            provider, "_get_http",
            lambda: _Http(post_payload={"error": "no"}, status=400))
        with pytest.raises(LLMCoreError):
            await provider.create_embeddings(["a"])


class TestReasoningBudget:
    def test_a_floor_is_documented(self):
        # Observed live: max_tokens=8 returned content='' with
        # completion_tokens=8, because the reasoning pass consumed the
        # whole budget. max_tokens=256 returned 'OK' with 45. A layer that
        # trims max_tokens to save money must stay above this.
        assert MIN_REASONING_MAX_TOKENS >= 64

    def test_chat_modality_set_is_narrow(self):
        assert CHAT_MODALITIES == frozenset({"chat"})
