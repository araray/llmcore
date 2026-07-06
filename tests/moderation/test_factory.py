# tests/moderation/test_factory.py
"""
Tests for build_moderation_gateway / build_moderation_policy config gating (SF-1).

Covers:
- Disabled-by-default: None/absent/enabled=false all return None (degrade-to-None)
- The packaged default_config.toml ships moderation disabled
- Enabled paths: openai gateway, noop gateway, unknown provider
- Enabled-but-broken is LOUD (missing key / missing SDK raise, never None)
- Section extraction: full config mapping vs the section itself vs .get objects
- Policy construction from config with conservative fallbacks
"""

from __future__ import annotations

import importlib.resources
import tomllib

import pytest

from llmcore.exceptions import ConfigError
from llmcore.moderation import (
    ModerationAction,
    NoopGateway,
    OpenAIModerationGateway,
    build_moderation_gateway,
    build_moderation_policy,
)
from llmcore.moderation import openai_gateway as openai_gateway_module

# =============================================================================
# GATEWAY GATING (degrade-to-None only when disabled)
# =============================================================================


def test_none_config_returns_none():
    assert build_moderation_gateway(None) is None


def test_missing_section_returns_none():
    assert build_moderation_gateway({"llmcore": {"default_provider": "openai"}}) is None


def test_enabled_false_returns_none():
    assert build_moderation_gateway({"moderation": {"enabled": False}}) is None


def test_default_config_toml_ships_disabled():
    """Acceptance: disabled-by-default, verified against the packaged defaults."""
    ref = importlib.resources.files("llmcore.config") / "default_config.toml"
    config = tomllib.loads(ref.read_text(encoding="utf-8"))
    assert config["moderation"]["enabled"] is False
    assert config["moderation"]["fail_safe"] is True
    assert build_moderation_gateway(config) is None


def test_enabled_openai_builds_gateway(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    gateway = build_moderation_gateway({"moderation": {"enabled": True}})
    assert isinstance(gateway, OpenAIModerationGateway)


def test_section_itself_is_accepted(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    gateway = build_moderation_gateway({"enabled": True, "model": "text-moderation-stable"})
    assert isinstance(gateway, OpenAIModerationGateway)
    assert gateway.model == "text-moderation-stable"


def test_config_object_with_get_is_accepted(monkeypatch):
    """Objects exposing .get('moderation') (confy-style) are supported."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    class FakeConfig:
        def get(self, key, default=None):
            return {"enabled": True} if key == "moderation" else default

    gateway = build_moderation_gateway(FakeConfig())
    assert isinstance(gateway, OpenAIModerationGateway)


def test_noop_provider_builds_noop_gateway():
    gateway = build_moderation_gateway({"moderation": {"enabled": True, "provider": "noop"}})
    assert isinstance(gateway, NoopGateway)


def test_unknown_provider_raises():
    with pytest.raises(ConfigError, match="Unknown moderation provider"):
        build_moderation_gateway({"moderation": {"enabled": True, "provider": "acme"}})


def test_enabled_without_api_key_raises_not_none(monkeypatch):
    """Enabled-but-broken must be loud — never a silent None."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="API key"):
        build_moderation_gateway({"moderation": {"enabled": True}})


def test_enabled_without_sdk_raises_not_none(monkeypatch):
    monkeypatch.setattr(openai_gateway_module, "openai_available", False)
    with pytest.raises(ImportError):
        build_moderation_gateway({"moderation": {"enabled": True}})


# =============================================================================
# POLICY CONSTRUCTION
# =============================================================================


def test_policy_defaults():
    policy = build_moderation_policy(None)
    assert policy.thresholds == {}
    assert policy.default_threshold is None
    assert policy.default_action is ModerationAction.BLOCK
    assert policy.fail_safe is True


def test_policy_from_full_config():
    policy = build_moderation_policy(
        {
            "moderation": {
                "thresholds": {"violence": 0.8, "self_harm": "0.5"},
                "default_threshold": 0.95,
                "default_action": "warn",
                "fail_safe": False,
            }
        }
    )
    assert policy.thresholds == {"violence": 0.8, "self_harm": 0.5}
    assert policy.default_threshold == 0.95
    assert policy.default_action is ModerationAction.WARN
    assert policy.fail_safe is False


def test_policy_drops_non_numeric_threshold():
    policy = build_moderation_policy({"moderation": {"thresholds": {"violence": "high"}}})
    assert policy.thresholds == {}


def test_policy_unknown_default_action_falls_back_to_block():
    policy = build_moderation_policy({"moderation": {"default_action": "shrug"}})
    assert policy.default_action is ModerationAction.BLOCK


def test_policy_allow_default_action_coerced_to_block():
    """'allow' as the on-trigger action would disable enforcement; block instead."""
    policy = build_moderation_policy({"moderation": {"default_action": "allow"}})
    assert policy.default_action is ModerationAction.BLOCK


def test_policy_non_numeric_default_threshold_dropped():
    policy = build_moderation_policy({"moderation": {"default_threshold": "very high"}})
    assert policy.default_threshold is None
