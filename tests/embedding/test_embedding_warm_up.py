from typing import Any, ClassVar

import pytest

from llmcore.embedding.base import BaseEmbeddingModel
from llmcore.embedding.manager import EMBEDDING_PROVIDER_CLASS_MAP, EmbeddingManager
from llmcore.exceptions import EmbeddingError
from llmcore.observability.events import Event
from llmcore.observability.federation import federate_llmcore_event


class SimpleConfig:
    def __init__(self, values: dict[str, Any]) -> None:
        self._values = values

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)


class CapturingEventLogger:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def log_event(
        self,
        *,
        category: str,
        event_type: str,
        data: dict[str, Any],
        severity: str = "info",
        source: str | None = None,
        tags: list[str] | None = None,
    ) -> Event:
        event = Event(
            category=category,
            event_type=event_type,
            data=data,
            severity=severity,
            source=source,
            tags=tags or [],
        )
        self.events.append(event)
        return event


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


def _manager(
    values: dict[str, Any],
    *,
    event_logger: CapturingEventLogger | None = None,
) -> EmbeddingManager:
    return EmbeddingManager(SimpleConfig(values), event_logger=event_logger)


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
    event_logger = CapturingEventLogger()
    manager = _manager(
        {
            "llmcore.default_embedding_model": "test:alpha",
        },
        event_logger=event_logger,
    )

    await manager.initialize()
    assert WarmEmbedding.initialize_calls == 0
    assert WarmEmbedding.warm_up_calls == 0
    assert [event.event_type for event in event_logger.events] == [
        "embedding_warm_up_skipped"
    ]
    skipped = event_logger.events[0]
    assert skipped.category == "lifecycle"
    assert skipped.severity == "debug"
    assert skipped.source == "embedding.manager"
    assert skipped.tags == ["embedding", "warm_up"]
    assert skipped.data["component"] == "embedding.manager"
    assert skipped.data["enabled"] is False

    await manager.get_model("test:alpha")
    assert WarmEmbedding.initialize_calls == 1
    assert WarmEmbedding.warm_up_calls == 0


@pytest.mark.asyncio
async def test_embedding_warm_up_enabled_warms_default_model_once() -> None:
    event_logger = CapturingEventLogger()
    manager = _manager(
        {
            "llmcore.default_embedding_model": "test:alpha",
            "llmcore.embedding_warm_up_enabled": True,
        },
        event_logger=event_logger,
    )

    await manager.initialize()
    await manager.initialize()
    await manager.get_model("test:alpha")

    assert WarmEmbedding.initialize_calls == 1
    assert WarmEmbedding.warm_up_calls == 1
    assert WarmEmbedding.initialized_model_names == ["alpha"]
    assert WarmEmbedding.warmed_model_names == ["alpha"]
    assert [event.event_type for event in event_logger.events] == [
        "embedding_warm_up_started",
        "embedding_warm_up_completed",
    ]
    completed = event_logger.events[-1]
    assert completed.category == "lifecycle"
    assert completed.severity == "info"
    assert completed.data["model_identifier"] == "test:alpha"
    assert completed.data["component"] == "embedding.test"
    assert completed.data["success"] is True
    assert completed.data["duration_ms"] >= 0

    federated = federate_llmcore_event(completed)
    assert federated.source_component == "embedding.manager"
    assert federated.event_type == "embedding_warm_up_completed"
    assert federated.duration_ms is not None
    assert federated.payload["component"] == "embedding.test"


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
    event_logger = CapturingEventLogger()
    manager = _manager(
        {
            "llmcore.default_embedding_model": "test:alpha",
            "llmcore.embedding_warm_up_enabled": True,
            "llmcore.embedding_warm_up_strict": False,
        },
        event_logger=event_logger,
    )

    await manager.initialize()

    assert WarmEmbedding.initialize_calls == 1
    assert WarmEmbedding.warm_up_calls == 1
    assert "Embedding warm-up failed for model 'test:alpha'" in caplog.text
    assert manager.get_cached_models() == ["test:alpha"]
    assert [event.event_type for event in event_logger.events] == [
        "embedding_warm_up_started",
        "embedding_warm_up_failed",
    ]
    failed = event_logger.events[-1]
    assert failed.severity == "warning"
    assert failed.data["model_identifier"] == "test:alpha"
    assert failed.data["component"] == "embedding.test"
    assert failed.data["stage"] == "warm_up"
    assert failed.data["strict"] is False
    assert failed.data["success"] is False
    assert failed.data["error_type"] == "RuntimeError"
    assert failed.data["error_message"] == "embedding warm-up failed"


@pytest.mark.asyncio
async def test_embedding_warm_up_initialization_failure_non_strict_preserves_lazy_first_use(
    caplog: pytest.LogCaptureFixture,
) -> None:
    WarmEmbedding.fail_initialize = True
    event_logger = CapturingEventLogger()
    manager = _manager(
        {
            "llmcore.default_embedding_model": "test:alpha",
            "llmcore.embedding_warm_up_enabled": True,
            "llmcore.embedding_warm_up_strict": False,
        },
        event_logger=event_logger,
    )

    await manager.initialize()

    assert WarmEmbedding.initialize_calls == 1
    assert WarmEmbedding.warm_up_calls == 0
    assert manager.get_cached_models() == []
    assert "Embedding warm-up failed for model 'test:alpha'" in caplog.text
    assert [event.event_type for event in event_logger.events] == [
        "embedding_warm_up_failed"
    ]
    failed = event_logger.events[-1]
    assert failed.severity == "warning"
    assert failed.data["stage"] == "initialize"
    assert failed.data["strict"] is False
    assert failed.data["success"] is False
    assert failed.data["error_type"] == "EmbeddingError"

    WarmEmbedding.fail_initialize = False
    model = await manager.get_model("test:alpha")

    assert isinstance(model, WarmEmbedding)
    assert WarmEmbedding.initialize_calls == 2
    assert WarmEmbedding.warm_up_calls == 1
    assert [event.event_type for event in event_logger.events] == [
        "embedding_warm_up_failed",
        "embedding_warm_up_started",
        "embedding_warm_up_completed",
    ]


@pytest.mark.asyncio
async def test_embedding_warm_up_failure_strict_raises() -> None:
    WarmEmbedding.fail_warm_up = True
    event_logger = CapturingEventLogger()
    manager = _manager(
        {
            "llmcore.default_embedding_model": "test:alpha",
            "llmcore.embedding_warm_up_enabled": True,
            "llmcore.embedding_warm_up_strict": True,
        },
        event_logger=event_logger,
    )

    with pytest.raises(EmbeddingError, match="Embedding warm-up failed"):
        await manager.initialize()

    assert WarmEmbedding.initialize_calls == 1
    assert WarmEmbedding.warm_up_calls == 1
    assert [event.event_type for event in event_logger.events] == [
        "embedding_warm_up_started",
        "embedding_warm_up_failed",
    ]
    assert event_logger.events[-1].severity == "error"
    assert event_logger.events[-1].data["strict"] is True
