# tests/providers/test_openai_compatible_providers.py
"""
Tests for OpenAI-compatible provider auto-detection in ProviderManager.

Covers:
- API key resolution from conventional env vars (DEEPSEEK_API_KEY, etc.)
- base_url auto-injection for known providers
- Backward compatibility: explicit api_key/base_url in config still works
- Backward compatibility: api_key_env_var mechanism still works
- Unknown provider sections are not affected
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from llmcore.providers.manager import (
    _OPENAI_COMPATIBLE_DEFAULTS,
    PROVIDER_MAP,
    ProviderManager,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(providers_dict: dict, default_provider: str = "deepseek"):
    """Build a mock ConfyConfig that responds to .get() calls."""
    cfg = MagicMock()

    store = {
        "llmcore.default_provider": default_provider,
        "llmcore.log_raw_payloads": False,
        "providers": providers_dict,
    }

    def _get(key, default=None):
        return store.get(key, default)

    cfg.get = _get
    return cfg


def _mock_provider_cls():
    """Create a mock provider class whose instances pass isinstance checks."""
    cls = MagicMock()
    cls.return_value = MagicMock()
    return cls


def _patch_provider_map(overrides: dict[str, MagicMock]):
    """Context manager that temporarily replaces PROVIDER_MAP entries.

    ProviderManager resolves provider classes from PROVIDER_MAP (built at
    import time), not from module-level names.  Patching the module-level
    ``OpenAIProvider`` has no effect on the map, so we must patch the dict
    entries directly.
    """
    return patch.dict(
        "llmcore.providers.manager.PROVIDER_MAP",
        overrides,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOpenAICompatibleDefaults:
    """Verify the _OPENAI_COMPATIBLE_DEFAULTS table is consistent."""

    def test_all_compat_providers_in_provider_map(self):
        """Every entry in _OPENAI_COMPATIBLE_DEFAULTS must appear in PROVIDER_MAP."""
        for name in _OPENAI_COMPATIBLE_DEFAULTS:
            assert name in PROVIDER_MAP, f"{name} missing from PROVIDER_MAP"

    def test_all_compat_providers_have_required_keys(self):
        for name, defaults in _OPENAI_COMPATIBLE_DEFAULTS.items():
            assert "env_var" in defaults, f"{name} missing 'env_var'"
            assert "base_url" in defaults, f"{name} missing 'base_url'"
            assert defaults["env_var"].endswith("_API_KEY")
            assert defaults["base_url"].startswith("https://")


class TestApiKeyAutoDetection:
    """ProviderManager should resolve API keys from conventional env vars."""

    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-deep-test-123"}, clear=False)
    def test_deepseek_api_key_from_env(self):
        """[providers.deepseek] picks up DEEPSEEK_API_KEY automatically."""
        mock_cls = _mock_provider_cls()
        with _patch_provider_map({"deepseek": mock_cls}):
            cfg = _make_config(
                {"deepseek": {"default_model": "deepseek-chat"}},
                default_provider="deepseek",
            )
            mgr = ProviderManager(cfg)

        assert "deepseek" in mgr.get_available_providers()
        config_passed = mock_cls.call_args[0][0]
        assert config_passed["api_key"] == "sk-deep-test-123"

    @patch.dict(os.environ, {"MISTRAL_API_KEY": "sk-mistral-test"}, clear=False)
    def test_mistral_api_key_from_env(self):
        """[providers.mistral] picks up MISTRAL_API_KEY automatically."""
        mock_cls = _mock_provider_cls()
        with _patch_provider_map({"mistral": mock_cls}):
            cfg = _make_config(
                {"mistral": {"default_model": "mistral-large-latest"}},
                default_provider="mistral",
            )
            mgr = ProviderManager(cfg)

        assert "mistral" in mgr.get_available_providers()
        config_passed = mock_cls.call_args[0][0]
        assert config_passed["api_key"] == "sk-mistral-test"

    @patch.dict(os.environ, {"XAI_API_KEY": "sk-xai-test"}, clear=False)
    def test_xai_api_key_from_env(self):
        """[providers.xai] picks up XAI_API_KEY automatically."""
        mock_cls = _mock_provider_cls()
        with _patch_provider_map({"xai": mock_cls}):
            cfg = _make_config(
                {"xai": {"default_model": "grok-2"}},
                default_provider="xai",
            )
            mgr = ProviderManager(cfg)

        assert "xai" in mgr.get_available_providers()
        config_passed = mock_cls.call_args[0][0]
        assert config_passed["api_key"] == "sk-xai-test"


class TestBaseUrlAutoInjection:
    """ProviderManager should inject base_url for known compatible providers."""

    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-deep-test"}, clear=False)
    def test_deepseek_base_url_injected(self):
        """base_url is auto-injected for deepseek when not in config."""
        mock_cls = _mock_provider_cls()
        with _patch_provider_map({"deepseek": mock_cls}):
            cfg = _make_config(
                {"deepseek": {"default_model": "deepseek-chat"}},
                default_provider="deepseek",
            )
            ProviderManager(cfg)

        config_passed = mock_cls.call_args[0][0]
        assert config_passed["base_url"] == "https://api.deepseek.com"

    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-deep-test"}, clear=False)
    def test_explicit_base_url_not_overridden(self):
        """Explicit base_url in config is NOT overridden by defaults."""
        mock_cls = _mock_provider_cls()
        with _patch_provider_map({"deepseek": mock_cls}):
            cfg = _make_config(
                {
                    "deepseek": {
                        "default_model": "deepseek-chat",
                        "base_url": "https://custom.proxy.example/v1",
                    }
                },
                default_provider="deepseek",
            )
            ProviderManager(cfg)

        config_passed = mock_cls.call_args[0][0]
        assert config_passed["base_url"] == "https://custom.proxy.example/v1"


class TestBackwardCompatibility:
    """Existing config patterns must continue to work."""

    def test_explicit_api_key_in_config(self):
        """Explicit api_key in config is used as-is, no env lookup."""
        mock_cls = _mock_provider_cls()
        with _patch_provider_map({"deepseek": mock_cls}):
            cfg = _make_config(
                {"deepseek": {"api_key": "sk-explicit", "default_model": "deepseek-chat"}},
                default_provider="deepseek",
            )
            ProviderManager(cfg)

        config_passed = mock_cls.call_args[0][0]
        assert config_passed["api_key"] == "sk-explicit"

    @patch.dict(os.environ, {"MY_CUSTOM_KEY": "sk-custom"}, clear=False)
    def test_api_key_env_var_still_works(self):
        """api_key_env_var mechanism is still honoured."""
        mock_cls = _mock_provider_cls()
        with _patch_provider_map({"deepseek": mock_cls}):
            cfg = _make_config(
                {
                    "deepseek": {
                        "api_key_env_var": "MY_CUSTOM_KEY",
                        "default_model": "deepseek-chat",
                    }
                },
                default_provider="deepseek",
            )
            ProviderManager(cfg)

        config_passed = mock_cls.call_args[0][0]
        assert config_passed["api_key"] == "sk-custom"

    def test_ollama_unaffected(self):
        """Ollama (no API key needed) continues to work."""
        mock_cls = _mock_provider_cls()
        with _patch_provider_map({"ollama": mock_cls}):
            cfg = _make_config(
                {"ollama": {"api_url": "http://localhost:11434"}},
                default_provider="ollama",
            )
            mgr = ProviderManager(cfg)

        assert "ollama" in mgr.get_available_providers()
