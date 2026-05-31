# tests/search/test_search_manager.py
"""Tests for :mod:`llmcore.search.manager` (SearchProviderManager).

These exercise configuration-driven loading semantics: env-var key resolution,
graceful skipping of misconfigured providers, default resolution, type aliases,
and the *optional* nature of the whole subsystem (zero providers is valid).
"""

from __future__ import annotations

import pytest
from confy.loader import Config

from llmcore.exceptions import ConfigError
from llmcore.search.manager import SearchProviderManager


def _mk_config(defaults: dict) -> Config:
    """Build a confy Config from a plain dict (no files, no env prefix)."""
    return Config(defaults=defaults)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def test_loads_provider_with_inline_api_key():
    cfg = _mk_config(
        {
            "llmcore": {"default_search_provider": "brightdata"},
            "search_providers": {
                "brightdata": {"api_key": "tok_inline_123", "serp_zone": "z"},
            },
        }
    )
    m = SearchProviderManager(cfg)
    assert m.get_available_search_providers() == ["brightdata"]
    assert m.has_search_providers() is True
    assert m.default_search_provider_name == "brightdata"
    assert m.get_search_provider().get_name() == "brightdata"


def test_loads_api_key_from_env_var(monkeypatch):
    monkeypatch.setenv("MY_BD_TOKEN", "tok_env_abc")
    cfg = _mk_config(
        {
            "search_providers": {
                "brightdata": {"api_key_env_var": "MY_BD_TOKEN"},
            },
        }
    )
    m = SearchProviderManager(cfg)
    # Single provider → auto-adopted as default.
    assert m.default_search_provider_name == "brightdata"
    assert m.get_default_search_provider().get_name() == "brightdata"


def test_loads_api_key_from_conventional_env(monkeypatch):
    monkeypatch.setenv("BRIGHTDATA_API_TOKEN", "tok_conv_xyz")
    cfg = _mk_config({"search_providers": {"brightdata": {}}})
    m = SearchProviderManager(cfg)
    assert m.has_search_providers() is True


def test_missing_token_skips_provider_gracefully(monkeypatch):
    # No inline key, no env var set → ConfigError inside provider is caught and
    # the provider is simply not loaded (no exception bubbles up).
    monkeypatch.delenv("BRIGHTDATA_API_TOKEN", raising=False)
    cfg = _mk_config({"search_providers": {"brightdata": {"serp_zone": "z"}}})
    m = SearchProviderManager(cfg)
    assert m.get_available_search_providers() == []
    assert m.has_search_providers() is False


# ---------------------------------------------------------------------------
# Optional subsystem (zero providers is valid)
# ---------------------------------------------------------------------------
def test_no_section_yields_empty_manager():
    cfg = _mk_config({"llmcore": {"default_provider": "ollama"}})
    m = SearchProviderManager(cfg)
    assert m.has_search_providers() is False
    assert m.get_available_search_providers() == []


def test_get_search_provider_without_any_raises():
    cfg = _mk_config({})
    m = SearchProviderManager(cfg)
    with pytest.raises(ConfigError, match="no default configured"):
        m.get_search_provider()


def test_get_unknown_named_provider_raises(monkeypatch):
    monkeypatch.setenv("BRIGHTDATA_API_TOKEN", "tok")
    cfg = _mk_config({"search_providers": {"brightdata": {}}})
    m = SearchProviderManager(cfg)
    with pytest.raises(ConfigError, match="not configured or failed to load"):
        m.get_search_provider("does_not_exist")


# ---------------------------------------------------------------------------
# Defaults / aliases / unknown types
# ---------------------------------------------------------------------------
def test_invalid_default_is_not_fatal(monkeypatch):
    monkeypatch.setenv("BRIGHTDATA_API_TOKEN", "tok")
    cfg = _mk_config(
        {
            "llmcore": {"default_search_provider": "nonexistent"},
            "search_providers": {"brightdata": {}},
        }
    )
    m = SearchProviderManager(cfg)
    # The bad default is dropped; with a single provider it auto-adopts that one.
    assert m.default_search_provider_name == "brightdata"


def test_type_alias_with_distinct_section_name(monkeypatch):
    monkeypatch.setenv("BRIGHTDATA_API_TOKEN", "tok")
    cfg = _mk_config(
        {
            "search_providers": {
                # Section name != type; type uses the hyphenated alias.
                "bd_primary": {"type": "bright-data"},
            },
        }
    )
    m = SearchProviderManager(cfg)
    assert m.get_available_search_providers() == ["bd_primary"]
    assert m.get_search_provider("bd_primary").get_name() == "brightdata"


def test_unknown_type_is_skipped():
    cfg = _mk_config({"search_providers": {"weirdsearch": {"type": "weird", "api_key": "x"}}})
    m = SearchProviderManager(cfg)
    assert m.get_available_search_providers() == []


def test_multiple_providers_no_default_does_not_autoadopt(monkeypatch):
    monkeypatch.setenv("BRIGHTDATA_API_TOKEN", "tok")
    cfg = _mk_config(
        {
            "search_providers": {
                "bd_a": {"type": "brightdata"},
                "bd_b": {"type": "brightdata"},
            },
        }
    )
    m = SearchProviderManager(cfg)
    assert set(m.get_available_search_providers()) == {"bd_a", "bd_b"}
    # Two providers, no explicit default → no auto-adoption.
    assert m.default_search_provider_name is None
    with pytest.raises(ConfigError):
        m.get_search_provider()  # ambiguous → must name one
    assert m.get_search_provider("bd_a").get_name() == "brightdata"


# ---------------------------------------------------------------------------
# Runtime controls / lifecycle
# ---------------------------------------------------------------------------
def test_update_log_raw_payloads_propagates(monkeypatch):
    monkeypatch.setenv("BRIGHTDATA_API_TOKEN", "tok")
    cfg = _mk_config({"search_providers": {"brightdata": {}}})
    m = SearchProviderManager(cfg)
    provider = m.get_default_search_provider()
    assert provider.log_raw_payloads_enabled is False
    m.update_log_raw_payloads_setting(True)
    assert provider.log_raw_payloads_enabled is True


async def test_close_all_is_safe_when_empty():
    cfg = _mk_config({})
    m = SearchProviderManager(cfg)
    await m.close_all()  # no providers, no error


async def test_close_all_closes_providers(monkeypatch):
    monkeypatch.setenv("BRIGHTDATA_API_TOKEN", "tok")
    cfg = _mk_config({"search_providers": {"brightdata": {}}})
    m = SearchProviderManager(cfg)
    provider = m.get_default_search_provider()
    # Force client creation by poking the lazy accessor.
    provider._get_client()
    assert provider._client is not None
    await m.close_all()
    assert provider._client is None
