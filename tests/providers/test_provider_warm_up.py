from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest

from llmcore.exceptions import ProviderError
from llmcore.models import Message, ModelDetails, Tool
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
    manager = ProviderManager(_config())

    await manager.initialize()

    assert WarmProvider.warm_up_calls == 0


@pytest.mark.asyncio
async def test_provider_warm_up_enabled_is_idempotent() -> None:
    manager = ProviderManager(_config(warm_up=True))

    await manager.initialize()
    await manager.initialize()

    assert WarmProvider.warm_up_calls == 1


@pytest.mark.asyncio
async def test_provider_warm_up_failure_non_strict_logs_and_continues() -> None:
    WarmProvider.fail_warm_up = True
    manager = ProviderManager(_config(warm_up=True, strict=False))

    await manager.initialize()

    assert WarmProvider.warm_up_calls == 1


@pytest.mark.asyncio
async def test_provider_warm_up_failure_strict_raises() -> None:
    WarmProvider.fail_warm_up = True
    manager = ProviderManager(_config(warm_up=True, strict=True))

    with pytest.raises(ProviderError, match="warm-up failed"):
        await manager.initialize()
