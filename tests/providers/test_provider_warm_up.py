from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest

from llmcore.exceptions import ProviderError
from llmcore.models import Message, ModelDetails, Tool
from llmcore.observability.events import Event
from llmcore.observability.federation import federate_llmcore_event
from llmcore.providers.base import BaseProvider
from llmcore.providers.manager import ProviderManager


class SimpleConfig:
    def __init__(self, data: dict[str, Any]):
        self.data = data

    def get(self, key: str, default: Any = None) -> Any:
        if key in self.data:
            return self.data[key]
        current: Any = self.data
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current


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


class WarmProvider(BaseProvider):
    warm_up_calls = 0
    fail_warm_up = False

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False):
        super().__init__(config, log_raw_payloads=log_raw_payloads)

    def get_name(self) -> str:
        return "warm"

    async def warm_up(self) -> None:
        type(self).warm_up_calls += 1
        if type(self).fail_warm_up:
            raise RuntimeError("boom")

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
        return {}

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        return len(text.split())

    async def count_message_tokens(self, messages: list[Message], model: str | None = None) -> int:
        return len(messages)

    def extract_response_content(self, response: dict[str, Any]) -> str:
        return ""

    def extract_delta_content(self, chunk: dict[str, Any]) -> str:
        return ""


def _config(*, warm_up: bool = False, strict: bool = False) -> SimpleConfig:
    return SimpleConfig(
        {
            "llmcore": {
                "default_provider": "warm",
                "provider_warm_up_enabled": warm_up,
                "provider_warm_up_strict": strict,
            },
            "providers": {"warm": {"type": "warm"}},
        }
    )


@pytest.fixture(autouse=True)
def reset_provider(monkeypatch):
    from llmcore.providers import manager as provider_manager

    WarmProvider.warm_up_calls = 0
    WarmProvider.fail_warm_up = False
    monkeypatch.setitem(provider_manager.PROVIDER_MAP, "warm", WarmProvider)


@pytest.mark.asyncio
async def test_provider_warm_up_disabled_by_default() -> None:
    event_logger = CapturingEventLogger()
    manager = ProviderManager(_config(), event_logger=event_logger)

    await manager.initialize()

    assert WarmProvider.warm_up_calls == 0
    assert [event.event_type for event in event_logger.events] == [
        "provider_warm_up_skipped"
    ]
    skipped = event_logger.events[0]
    assert skipped.category == "lifecycle"
    assert skipped.severity == "debug"
    assert skipped.data["component"] == "providers.manager"
    assert skipped.data["enabled"] is False


@pytest.mark.asyncio
async def test_provider_warm_up_enabled_is_idempotent() -> None:
    event_logger = CapturingEventLogger()
    manager = ProviderManager(_config(warm_up=True), event_logger=event_logger)

    await manager.initialize()
    await manager.initialize()

    assert WarmProvider.warm_up_calls == 1
    assert [event.event_type for event in event_logger.events] == [
        "provider_warm_up_started",
        "provider_warm_up_completed",
    ]
    completed = event_logger.events[-1]
    assert completed.category == "lifecycle"
    assert completed.severity == "info"
    assert completed.source == "providers.manager"
    assert completed.tags == ["provider", "warm_up"]
    assert completed.data["provider_name"] == "warm"
    assert completed.data["component"] == "providers.warm"
    assert completed.data["success"] is True
    assert completed.data["duration_ms"] >= 0

    federated = federate_llmcore_event(completed)
    assert federated.source_component == "providers.manager"
    assert federated.category == "lifecycle"
    assert federated.event_type == "provider_warm_up_completed"
    assert federated.duration_ms is not None
    assert federated.payload["component"] == "providers.warm"


@pytest.mark.asyncio
async def test_provider_warm_up_failure_non_strict_logs_and_continues() -> None:
    WarmProvider.fail_warm_up = True
    event_logger = CapturingEventLogger()
    manager = ProviderManager(
        _config(warm_up=True, strict=False),
        event_logger=event_logger,
    )

    await manager.initialize()

    assert WarmProvider.warm_up_calls == 1
    assert [event.event_type for event in event_logger.events] == [
        "provider_warm_up_started",
        "provider_warm_up_failed",
    ]
    failed = event_logger.events[-1]
    assert failed.severity == "warning"
    assert failed.data["provider_name"] == "warm"
    assert failed.data["component"] == "providers.warm"
    assert failed.data["strict"] is False
    assert failed.data["success"] is False
    assert failed.data["error_type"] == "RuntimeError"
    assert failed.data["error_message"] == "boom"
    assert failed.data["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_provider_warm_up_failure_strict_raises() -> None:
    WarmProvider.fail_warm_up = True
    event_logger = CapturingEventLogger()
    manager = ProviderManager(
        _config(warm_up=True, strict=True),
        event_logger=event_logger,
    )

    with pytest.raises(ProviderError, match="warm-up failed"):
        await manager.initialize()

    assert [event.event_type for event in event_logger.events] == [
        "provider_warm_up_started",
        "provider_warm_up_failed",
    ]
    assert event_logger.events[-1].severity == "error"
    assert event_logger.events[-1].data["strict"] is True
