from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest

from llmcore.api import LLMCore
from llmcore.embedding.base import BaseEmbeddingModel
from llmcore.embedding.manager import EMBEDDING_PROVIDER_CLASS_MAP
from llmcore.models import Message, ModelDetails, Tool
from llmcore.observability.events import Event
from llmcore.observability.federation import federate_llmcore_event
from llmcore.providers.base import BaseProvider


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


class ObservabilityContainer:
    def __init__(self, logger: CapturingEventLogger) -> None:
        self.logger = logger


class FakeProvider(BaseProvider):
    warm_up_calls = 0

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False):
        super().__init__(config, log_raw_payloads=log_raw_payloads)
        self.default_model = config.get("default_model", "fake-model")

    def get_name(self) -> str:
        return "fake"

    async def warm_up(self) -> None:
        type(self).warm_up_calls += 1

    async def get_models_details(self) -> list[ModelDetails]:
        return []

    def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
        return {}

    def get_max_context_length(self, model: str | None = None) -> int:
        return 4096

    async def chat_completion(
        self,
        context: list[Message],
        model: str | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        return {"choices": [{"message": {"content": "ok"}}]}

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        return len(text.split())

    async def count_message_tokens(self, messages: list[Message], model: str | None = None) -> int:
        return len(messages)

    def extract_response_content(self, response: dict[str, Any]) -> str:
        return response["choices"][0]["message"]["content"]

    def extract_delta_content(self, chunk: dict[str, Any]) -> str:
        return ""


class FakeEmbedding(BaseEmbeddingModel):
    warm_up_calls = 0

    def __init__(self, config: dict[str, Any]) -> None:
        self.model_name = str(config.get("default_model", "fake-embedding"))

    async def initialize(self) -> None:
        return None

    async def warm_up(self) -> None:
        type(self).warm_up_calls += 1

    async def generate_embedding(self, text: str) -> list[float]:
        return [float(len(text))]

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]


@pytest.fixture(autouse=True)
def register_fake_models(monkeypatch: pytest.MonkeyPatch) -> None:
    from llmcore.providers import manager as provider_manager

    FakeProvider.warm_up_calls = 0
    FakeEmbedding.warm_up_calls = 0
    monkeypatch.setitem(provider_manager.PROVIDER_MAP, "fake", FakeProvider)
    monkeypatch.setitem(EMBEDDING_PROVIDER_CLASS_MAP, "test", FakeEmbedding)


@pytest.mark.asyncio
async def test_create_wires_observability_to_provider_and_embedding_warm_up(
    tmp_path,
) -> None:
    event_logger = CapturingEventLogger()
    observability = ObservabilityContainer(event_logger)

    llm = await LLMCore.create(
        config_overrides={
            "llmcore": {
                "default_provider": "fake",
                "provider_warm_up_enabled": True,
                "embedding_warm_up_enabled": True,
                "default_embedding_model": "test:alpha",
            },
            "providers": {
                "fake": {"type": "fake", "default_model": "fake-model"},
            },
            "storage": {
                "vector": {"type": ""},
                "session": {"type": "json", "path": str(tmp_path / "sessions")},
            },
        },
        observability=observability,
    )

    try:
        event_types = [event.event_type for event in event_logger.events]

        assert FakeProvider.warm_up_calls == 1
        assert FakeEmbedding.warm_up_calls == 1
        assert "provider_warm_up_completed" in event_types
        assert "embedding_warm_up_completed" in event_types

        provider_event = next(
            event
            for event in event_logger.events
            if event.event_type == "provider_warm_up_completed"
            and event.data.get("component") == "providers.fake"
        )
        embedding_event = next(
            event
            for event in event_logger.events
            if event.event_type == "embedding_warm_up_completed"
        )

        assert provider_event.source == "providers.manager"
        assert provider_event.data["component"] == "providers.fake"
        assert embedding_event.source == "embedding.manager"
        assert embedding_event.data["component"] == "embedding.test"

        federated = federate_llmcore_event(provider_event)
        assert federated.source_component == "providers.manager"
        assert federated.event_type == "provider_warm_up_completed"
        assert federated.payload["component"] == "providers.fake"
    finally:
        await llm.close()


def test_resolve_observability_event_logger_accepts_direct_and_wrapped_loggers() -> None:
    direct = CapturingEventLogger()
    wrapped = ObservabilityContainer(direct)

    assert LLMCore._resolve_observability_event_logger(direct) is direct
    assert LLMCore._resolve_observability_event_logger(wrapped) is direct
    assert LLMCore._resolve_observability_event_logger(object()) is None
