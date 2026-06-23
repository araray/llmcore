from typing import Any, ClassVar

import pytest

from llmcore.embedding.base import BaseEmbeddingModel
from llmcore.embedding.manager import EMBEDDING_PROVIDER_CLASS_MAP, EmbeddingManager
from llmcore.exceptions import EmbeddingError


class SimpleConfig:
    def __init__(self, values: dict[str, Any]) -> None:
        self._values = values

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)


class WarmEmbedding(BaseEmbeddingModel):
    initialize_calls = 0
    warm_up_calls = 0
    initialized_model_names: ClassVar[list[str]] = []
    warmed_model_names: ClassVar[list[str]] = []
    fail_initialize = False
    fail_warm_up = False

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model_name = str(config.get("default_model", "unknown"))

    @classmethod
    def reset(cls) -> None:
        cls.initialize_calls = 0
        cls.warm_up_calls = 0
        cls.initialized_model_names = []
        cls.warmed_model_names = []
        cls.fail_initialize = False
        cls.fail_warm_up = False

    async def initialize(self) -> None:
        type(self).initialize_calls += 1
        type(self).initialized_model_names.append(self.model_name)
        if type(self).fail_initialize:
            raise RuntimeError("embedding init failed")

    async def warm_up(self) -> None:
        type(self).warm_up_calls += 1
        type(self).warmed_model_names.append(self.model_name)
        if type(self).fail_warm_up:
            raise RuntimeError("embedding warm-up failed")

    async def generate_embedding(self, text: str) -> list[float]:
        return [float(len(text))]

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]


@pytest.fixture(autouse=True)
def reset_warm_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    WarmEmbedding.reset()
    monkeypatch.setitem(EMBEDDING_PROVIDER_CLASS_MAP, "test", WarmEmbedding)


def _manager(values: dict[str, Any]) -> EmbeddingManager:
    return EmbeddingManager(SimpleConfig(values))


@pytest.mark.asyncio
async def test_base_embedding_warm_up_noop_is_safe() -> None:
    class NoopEmbedding(BaseEmbeddingModel):
        def __init__(self, config: dict[str, Any]) -> None:
            self.config = config

        async def initialize(self) -> None:
            return None

        async def generate_embedding(self, text: str) -> list[float]:
            return [1.0]

        async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
            return [[1.0] for _ in texts]

    model = NoopEmbedding({})

    assert await model.warm_up() is None


@pytest.mark.asyncio
async def test_embedding_warm_up_disabled_by_default() -> None:
    manager = _manager(
        {
            "llmcore.default_embedding_model": "test:alpha",
        }
    )

    await manager.initialize()
    assert WarmEmbedding.initialize_calls == 0
    assert WarmEmbedding.warm_up_calls == 0

    await manager.get_model("test:alpha")
    assert WarmEmbedding.initialize_calls == 1
    assert WarmEmbedding.warm_up_calls == 0


@pytest.mark.asyncio
async def test_embedding_warm_up_enabled_warms_default_model_once() -> None:
    manager = _manager(
        {
            "llmcore.default_embedding_model": "test:alpha",
            "llmcore.embedding_warm_up_enabled": True,
        }
    )

    await manager.initialize()
    await manager.initialize()
    await manager.get_model("test:alpha")

    assert WarmEmbedding.initialize_calls == 1
    assert WarmEmbedding.warm_up_calls == 1
    assert WarmEmbedding.initialized_model_names == ["alpha"]
    assert WarmEmbedding.warmed_model_names == ["alpha"]


@pytest.mark.asyncio
async def test_embedding_warm_up_uses_configured_model_list() -> None:
    manager = _manager(
        {
            "llmcore.default_embedding_model": "test:default",
            "llmcore.embedding_warm_up_enabled": True,
            "llmcore.embedding_warm_up_models": ["test:alpha", "test:beta"],
        }
    )

    await manager.initialize()

    assert WarmEmbedding.initialize_calls == 2
    assert WarmEmbedding.warm_up_calls == 2
    assert WarmEmbedding.initialized_model_names == ["alpha", "beta"]
    assert WarmEmbedding.warmed_model_names == ["alpha", "beta"]
    assert manager.get_cached_models() == ["test:alpha", "test:beta"]


@pytest.mark.asyncio
async def test_embedding_warm_up_failure_non_strict_logs_and_continues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    WarmEmbedding.fail_warm_up = True
    manager = _manager(
        {
            "llmcore.default_embedding_model": "test:alpha",
            "llmcore.embedding_warm_up_enabled": True,
            "llmcore.embedding_warm_up_strict": False,
        }
    )

    await manager.initialize()

    assert WarmEmbedding.initialize_calls == 1
    assert WarmEmbedding.warm_up_calls == 1
    assert "Embedding warm-up failed for model 'test:alpha'" in caplog.text
    assert manager.get_cached_models() == ["test:alpha"]


@pytest.mark.asyncio
async def test_embedding_warm_up_initialization_failure_non_strict_preserves_lazy_first_use(
    caplog: pytest.LogCaptureFixture,
) -> None:
    WarmEmbedding.fail_initialize = True
    manager = _manager(
        {
            "llmcore.default_embedding_model": "test:alpha",
            "llmcore.embedding_warm_up_enabled": True,
            "llmcore.embedding_warm_up_strict": False,
        }
    )

    await manager.initialize()

    assert WarmEmbedding.initialize_calls == 1
    assert WarmEmbedding.warm_up_calls == 0
    assert manager.get_cached_models() == []
    assert "Embedding warm-up failed for model 'test:alpha'" in caplog.text

    WarmEmbedding.fail_initialize = False
    model = await manager.get_model("test:alpha")

    assert isinstance(model, WarmEmbedding)
    assert WarmEmbedding.initialize_calls == 2
    assert WarmEmbedding.warm_up_calls == 1


@pytest.mark.asyncio
async def test_embedding_warm_up_failure_strict_raises() -> None:
    WarmEmbedding.fail_warm_up = True
    manager = _manager(
        {
            "llmcore.default_embedding_model": "test:alpha",
            "llmcore.embedding_warm_up_enabled": True,
            "llmcore.embedding_warm_up_strict": True,
        }
    )

    with pytest.raises(EmbeddingError, match="Embedding warm-up failed"):
        await manager.initialize()

    assert WarmEmbedding.initialize_calls == 1
    assert WarmEmbedding.warm_up_calls == 1
