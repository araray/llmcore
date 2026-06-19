from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, ClassVar
from unittest.mock import AsyncMock

import pytest

from llmcore.exceptions import ProviderError
from llmcore.models import Message, ModelDetails, Role, Tool
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


class FlakyProvider(BaseProvider):
    errors: ClassVar[list[ProviderError]] = []
    calls: ClassVar[int] = 0

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False):
        super().__init__(config, log_raw_payloads=log_raw_payloads)
        self.default_model = config.get("default_model", "flaky-model")

    def get_name(self) -> str:
        return "flaky"

    async def get_models_details(self) -> list[ModelDetails]:
        return []

    def get_supported_parameters(self, model: str | None = None) -> dict[str, Any]:
        return {"temperature": {"type": "number"}}

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
        type(self).calls += 1
        if type(self).errors:
            raise type(self).errors.pop(0)
        return {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 3}}

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        return len(text.split())

    async def count_message_tokens(self, messages: list[Message], model: str | None = None) -> int:
        return len(messages)

    def extract_response_content(self, response: dict[str, Any]) -> str:
        return response["choices"][0]["message"]["content"]

    def extract_delta_content(self, chunk: dict[str, Any]) -> str:
        return ""


def _config(**llmcore_overrides: Any) -> SimpleConfig:
    return SimpleConfig(
        {
            "llmcore": {
                "default_provider": "flaky",
                "provider_retry_base_delay_seconds": 0.0,
                "provider_retry_max_delay_seconds": 10.0,
                **llmcore_overrides,
            },
            "providers": {"flaky": {"type": "flaky", "default_model": "flaky-model"}},
        }
    )


@pytest.fixture(autouse=True)
def register_flaky(monkeypatch):
    from llmcore.providers import manager as provider_manager

    FlakyProvider.calls = 0
    FlakyProvider.errors = []
    monkeypatch.setitem(provider_manager.PROVIDER_MAP, "flaky", FlakyProvider)


def test_provider_error_infers_retry_metadata_from_status_and_headers() -> None:
    error = ProviderError(
        "openai",
        "API Error (429): slow down",
        model_name="gpt-test",
        headers={"Retry-After": "2.5"},
    )

    assert error.provider_name == "openai"
    assert error.provider == "openai"
    assert error.model_name == "gpt-test"
    assert error.status_code == 429
    assert error.retryable is True
    assert error.retry_after_seconds == 2.5
    assert error.to_dict()["retryable"] is True


@pytest.mark.asyncio
async def test_chat_completion_with_retry_retries_retryable_provider_error() -> None:
    FlakyProvider.errors = [ProviderError("flaky", "API Error (500): temporary outage")]
    manager = ProviderManager(_config(provider_retry_max_attempts=2))
    provider = manager.get_provider()

    response = await manager.chat_completion_with_retry(
        provider,
        context=[Message(role=Role.USER, content="hello")],
    )

    assert response["choices"][0]["message"]["content"] == "ok"
    assert FlakyProvider.calls == 2


@pytest.mark.asyncio
async def test_chat_completion_with_retry_fails_fast_for_non_retryable_error() -> None:
    FlakyProvider.errors = [
        ProviderError("flaky", "Authentication failed", status_code=401)
    ]
    manager = ProviderManager(_config(provider_retry_max_attempts=3))
    provider = manager.get_provider()

    with pytest.raises(ProviderError) as exc_info:
        await manager.chat_completion_with_retry(
            provider,
            context=[Message(role=Role.USER, content="hello")],
        )

    assert exc_info.value.retryable is False
    assert FlakyProvider.calls == 1


@pytest.mark.asyncio
async def test_chat_completion_with_retry_uses_retry_after(monkeypatch) -> None:
    sleep = AsyncMock()
    monkeypatch.setattr("llmcore.providers.manager.asyncio.sleep", sleep)
    FlakyProvider.errors = [
        ProviderError(
            "flaky",
            "API Error (429): rate limit",
            headers={"retry-after": "4"},
        )
    ]
    manager = ProviderManager(
        _config(provider_retry_base_delay_seconds=0.1, provider_retry_max_attempts=2)
    )
    provider = manager.get_provider()

    await manager.chat_completion_with_retry(
        provider,
        context=[Message(role=Role.USER, content="hello")],
    )

    sleep.assert_awaited_once_with(4.0)
    assert FlakyProvider.calls == 2
