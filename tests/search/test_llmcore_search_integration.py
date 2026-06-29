# tests/search/test_llmcore_search_integration.py
"""Integration tests: the search subsystem wired through the public ``LLMCore`` API.

These build a real :class:`LLMCore` via ``LLMCore.create()`` and verify that:

* a configured search provider is discoverable and that ``LLMCore.web_search``
  routes to it (HTTP intercepted with ``respx`` — no network), and
* when no search provider is configured, the search methods raise a clear
  ``ConfigError`` (the subsystem is optional and must not break existing setups).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import httpx
import pytest
import respx

from llmcore import LLMCore
from llmcore.exceptions import ConfigError
from llmcore.models import Message, ModelDetails, Tool
from llmcore.providers.base import BaseProvider
from llmcore.search.models import WebSearchResult
from llmcore.search.providers.brightdata_provider import DEFAULT_BASE_URL

BASE = DEFAULT_BASE_URL


class _FakeProvider(BaseProvider):
    """Minimal in-process LLM provider so ``LLMCore.create()`` initializes without
    any external library, API key, or network (e.g. no Ollama daemon)."""

    def __init__(self, config: dict[str, Any], log_raw_payloads: bool = False) -> None:
        super().__init__(config, log_raw_payloads=log_raw_payloads)
        self.default_model = config.get("default_model", "fake-model")

    def get_name(self) -> str:
        return "fake"

    async def warm_up(self) -> None:
        return None

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

    async def count_message_tokens(
        self, messages: list[Message], model: str | None = None
    ) -> int:
        return len(messages)

    def extract_response_content(self, response: dict[str, Any]) -> str:
        return response["choices"][0]["message"]["content"]

    def extract_delta_content(self, chunk: dict[str, Any]) -> str:
        return ""


@pytest.fixture(autouse=True)
def _register_fake_provider(monkeypatch):
    """Register the ``fake`` LLM provider type used by ``_base_overrides``."""
    from llmcore.providers import manager as provider_manager

    monkeypatch.setitem(provider_manager.PROVIDER_MAP, "fake", _FakeProvider)
    yield


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Make any polling/retry sleeps instant."""

    async def _instant(_seconds):
        return None

    monkeypatch.setattr("asyncio.sleep", _instant)
    yield


@pytest.fixture
def _no_bd_token(monkeypatch):
    """Force a genuinely empty search subsystem.

    Clearing only the Bright Data token is insufficient because the packaged
    default config also declares *key-optional* providers (e.g. Semantic Scholar,
    which loads against the public pool with no key). To make availability
    deterministic regardless of the host's env vars, also blank the
    ``SEARCH_PROVIDER_MAP`` so no ``[search_providers.*]`` section resolves to a
    provider class — every section is skipped and the subsystem is empty.
    """
    monkeypatch.delenv("BRIGHTDATA_API_TOKEN", raising=False)
    monkeypatch.setattr("llmcore.search.manager.SEARCH_PROVIDER_MAP", {})
    yield


def _base_overrides(tmp_path) -> dict:
    """Lightweight LLMCore config: no vector store, temp sqlite session DB.

    Keeps ``LLMCore.create()`` self-contained, fast and deterministic for CI (no
    ChromaDB / Postgres / network and no dependency on a reachable Ollama daemon)
    by using the in-process ``fake`` LLM provider as the default. The search
    subsystem under test is configured separately per-test.
    """
    return {
        "llmcore": {"default_provider": "fake"},
        "providers": {"fake": {"type": "fake", "default_model": "fake-model"}},
        "storage": {
            "vector": {"type": ""},  # disable vector store (RAG not needed here)
            "session": {"type": "sqlite", "path": str(tmp_path / "sessions.db")},
        },
    }


async def test_web_search_routes_through_llmcore(tmp_path):
    """A configured provider is adopted and web_search() returns normalized results."""
    overrides = _base_overrides(tmp_path)
    overrides["search_providers"] = {
        "brightdata": {
            "api_key": "tok_integration_123",
            "serp_zone": "z_serp",
            "unlocker_zone": "z_unlocker",
        }
    }
    llm = await LLMCore.create(config_overrides=overrides)
    try:
        # Provider discovered + auto-adopted as default (single instance).
        assert "brightdata" in llm.get_available_search_providers()
        assert llm.get_search_provider().get_name() == "brightdata"

        with respx.mock:
            respx.post(f"{BASE}/request").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "organic": [
                            {
                                "rank": 1,
                                "title": "Hit",
                                "link": "https://hit.example",
                                "description": "d",
                            }
                        ],
                        "total_results": 7,
                    },
                )
            )
            result = await llm.web_search("llmcore wiring test", count=1, country="US")

        assert isinstance(result, WebSearchResult)
        assert result.success is True
        assert result.provider == "brightdata"
        assert result.total_results == 7
        assert result.items[0].title == "Hit"
    finally:
        await llm.close()


async def test_search_methods_raise_when_unconfigured(_no_bd_token, tmp_path):
    """With no usable search provider, search methods raise ConfigError (not at init)."""
    # The packaged default config declares [search_providers.brightdata] with an
    # env-var key; with the token absent the provider is skipped, leaving the
    # subsystem empty. Construction must still succeed.
    llm = await LLMCore.create(config_overrides=_base_overrides(tmp_path))
    try:
        assert llm.get_available_search_providers() == []
        with pytest.raises(ConfigError):
            await llm.web_search("anything")
        with pytest.raises(ConfigError):
            await llm.scrape_url("https://example.com")
        with pytest.raises(ConfigError):
            await llm.discover("anything")
    finally:
        await llm.close()
