# tests/search/test_llmcore_search_integration.py
"""Integration tests: the search subsystem wired through the public ``LLMCore`` API.

These build a real :class:`LLMCore` via ``LLMCore.create()`` and verify that:

* a configured search provider is discoverable and that ``LLMCore.web_search``
  routes to it (HTTP intercepted with ``respx`` — no network), and
* when no search provider is configured, the search methods raise a clear
  ``ConfigError`` (the subsystem is optional and must not break existing setups).
"""

from __future__ import annotations

import httpx
import pytest
import respx

from llmcore import LLMCore
from llmcore.exceptions import ConfigError
from llmcore.search.models import WebSearchResult
from llmcore.search.providers.brightdata_provider import DEFAULT_BASE_URL

BASE = DEFAULT_BASE_URL


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Make any polling/retry sleeps instant."""

    async def _instant(_seconds):
        return None

    monkeypatch.setattr("asyncio.sleep", _instant)
    yield


@pytest.fixture
def _no_bd_token(monkeypatch):
    """Ensure the conventional Bright Data token is absent for this test."""
    monkeypatch.delenv("BRIGHTDATA_API_TOKEN", raising=False)
    yield


def _base_overrides(tmp_path) -> dict:
    """Lightweight LLMCore config: no vector store, temp sqlite session DB.

    Keeps ``LLMCore.create()`` self-contained and fast for CI (no ChromaDB /
    Postgres / network), while leaving the default LLM provider (ollama, lazy)
    in place so the LLM ProviderManager initializes successfully.
    """
    return {
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
