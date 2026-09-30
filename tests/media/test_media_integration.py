# tests/media/test_media_integration.py
"""Integration of the media subsystem with the facade and the provider manager.

Covers ``llm.media`` wiring, the ``[media]`` config section, and the dynamic
provider registration that both the media and remote-runtime programs need.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import llmcore
from llmcore.exceptions import ConfigError
from llmcore.media import MediaCapability, MediaManager
from llmcore.media.protocols import MediaCapableProvider
from llmcore.media.testing import FakeMediaProvider
from llmcore.providers.manager import ProviderManager

DEFAULT_CONFIG = Path(llmcore.__file__).parent / "config" / "default_config.toml"


def _config(providers: dict, default: str = "vllm"):
    """A mock ConfyConfig exposing ``.get(key, default)``."""
    store = {
        "llmcore.default_provider": default,
        "llmcore.log_raw_payloads": False,
        "providers": providers,
    }
    cfg = MagicMock()
    cfg.get = lambda key, d=None: store.get(key, d)
    return cfg


VLLM = {"base_url": "http://localhost:8000/v1", "default_model": "m"}


# ---------------------------------------------------------------------------
# Config section
# ---------------------------------------------------------------------------


class TestMediaConfigSection:
    @pytest.fixture(scope="class")
    def cfg(self) -> dict:
        with open(DEFAULT_CONFIG, "rb") as fh:
            return tomllib.load(fh)

    def test_media_section_exists(self, cfg):
        assert "media" in cfg

    def test_artifact_defaults(self, cfg):
        assert cfg["media"]["artifact_materialize"] == "on_expiry"
        assert cfg["media"]["artifact_path"]

    def test_job_defaults(self, cfg):
        jobs = cfg["media"]["jobs"]
        assert jobs["poll_initial_seconds"] < jobs["poll_max_seconds"]
        assert jobs["job_timeout_seconds"] >= jobs["poll_max_seconds"]

    def test_routing_table_present_and_commented_out(self, cfg):
        # Present so the table exists, empty so built-in defaults apply.
        assert cfg["media"]["routing"] == {}

    def test_no_media_provider_credentials_section(self, cfg):
        """Media adapters are the chat providers; credentials are not duplicated."""
        assert "providers" not in cfg["media"]


# ---------------------------------------------------------------------------
# Facade wiring
# ---------------------------------------------------------------------------


class TestFacadeWiring:
    def test_media_property_exists(self):
        assert isinstance(getattr(llmcore.LLMCore, "media", None), property)

    def test_media_before_create_raises(self):
        # __init__ is private; a bare instance has no initialized subsystems.
        bare = llmcore.LLMCore()
        with pytest.raises(ConfigError, match="not initialized"):
            _ = bare.media

    def test_media_manager_is_exported(self):
        from llmcore.media import MediaManager as Exported

        assert Exported is MediaManager


# ---------------------------------------------------------------------------
# Dynamic provider registration (shared prerequisite)
# ---------------------------------------------------------------------------


class TestDynamicProviderRegistration:
    @pytest.fixture
    def pm(self) -> ProviderManager:
        return ProviderManager(_config({"vllm": dict(VLLM)}))

    def test_baseline(self, pm):
        assert pm.get_available_providers() == ["vllm"]
        assert pm.ephemeral_instances == []

    def test_register_instance(self, pm):
        provider = pm.register_instance(
            "colab-qwen", "vllm", {**VLLM, "base_url": "http://127.0.0.1:19001/v1"}
        )
        assert provider.get_name() == "colab-qwen"
        assert pm.get_provider("colab-qwen") is provider
        assert "colab-qwen" in pm.get_available_providers()

    def test_register_marks_ephemeral(self, pm):
        pm.register_instance("rt", "vllm", dict(VLLM), ephemeral=True)
        assert pm.is_ephemeral("rt") is True
        assert pm.ephemeral_instances == ["rt"]
        assert pm.is_ephemeral("vllm") is False

    def test_name_is_case_insensitive(self, pm):
        pm.register_instance("MixedCase", "vllm", dict(VLLM))
        assert pm.get_provider("mixedcase").get_name() == "mixedcase"

    def test_collision_raises_without_replace(self, pm):
        pm.register_instance("rt", "vllm", dict(VLLM))
        with pytest.raises(ConfigError, match="already exists"):
            pm.register_instance("rt", "vllm", dict(VLLM))

    def test_replace_swaps_the_instance(self, pm):
        first = pm.register_instance("rt", "vllm", dict(VLLM))
        second = pm.register_instance("rt", "vllm", dict(VLLM), replace=True)
        assert first is not second
        assert pm.get_provider("rt") is second

    def test_replace_clears_ephemeral_when_not_requested(self, pm):
        pm.register_instance("rt", "vllm", dict(VLLM), ephemeral=True)
        pm.register_instance("rt", "vllm", dict(VLLM), replace=True)
        assert pm.is_ephemeral("rt") is False

    def test_unknown_type_raises(self, pm):
        with pytest.raises(ConfigError, match="not supported"):
            pm.register_instance("x", "no-such-provider", {})

    def test_construction_failure_raises_config_error(self, pm):
        # vllm requires a base_url; omitting it fails construction.
        with pytest.raises(ConfigError, match="Failed to register"):
            pm.register_instance("broken", "vllm", {})

    async def test_unregister(self, pm):
        pm.register_instance("rt", "vllm", dict(VLLM), ephemeral=True)
        assert await pm.unregister_instance("rt") is True
        assert "rt" not in pm.get_available_providers()
        assert pm.ephemeral_instances == []

    async def test_unregister_missing_returns_false(self, pm):
        assert await pm.unregister_instance("ghost") is False

    async def test_unregister_refuses_the_default(self, pm):
        with pytest.raises(ConfigError, match="default provider"):
            await pm.unregister_instance("vllm")

    async def test_unregister_closes_by_default(self, pm):
        provider = pm.register_instance("rt", "vllm", dict(VLLM))
        closed = False

        async def _close():
            nonlocal closed
            closed = True

        provider.close = _close
        await pm.unregister_instance("rt")
        assert closed is True

    async def test_unregister_can_skip_close(self, pm):
        provider = pm.register_instance("rt", "vllm", dict(VLLM))
        closed = False

        async def _close():
            nonlocal closed
            closed = True

        provider.close = _close
        await pm.unregister_instance("rt", close=False)
        assert closed is False

    async def test_close_failure_does_not_block_unregister(self, pm):
        provider = pm.register_instance("rt", "vllm", dict(VLLM))

        async def _boom():
            raise RuntimeError("vendor down")

        provider.close = _boom
        assert await pm.unregister_instance("rt") is True
        assert "rt" not in pm.get_available_providers()


# ---------------------------------------------------------------------------
# Manager built from a real ProviderManager
# ---------------------------------------------------------------------------


class TestManagerFromRealProviderManager:
    def test_capability_less_providers_are_not_adapters(self):
        """Implementing the protocols is not the same as serving anything.

        vLLM subclasses OpenAIProvider, so since M3 it *inherits* the media
        protocol methods — but not the endpoints behind them, so it declares an
        empty capability set. It must not appear as an adapter that can route
        nothing.
        """
        pm = ProviderManager(_config({"vllm": dict(VLLM)}))
        provider = pm.get_provider("vllm")
        assert isinstance(provider, MediaCapableProvider)  # has the methods
        assert provider.media_capabilities() == frozenset()  # serves nothing

        media = MediaManager.from_provider_manager(pm, lambda k, d=None: d)
        assert media.adapter_names == []
        assert media.has_adapters() is False

    def test_a_registered_media_adapter_is_routable(self):
        pm = ProviderManager(_config({"vllm": dict(VLLM)}))
        media = MediaManager.from_provider_manager(pm, lambda k, d=None: d)
        media.register_adapter("fake", FakeMediaProvider("fake"))
        assert media.who_can(MediaCapability.IMAGE_GENERATE) == ["fake"]

    async def test_end_to_end_through_the_manager(self):
        pm = ProviderManager(_config({"vllm": dict(VLLM)}))
        media = MediaManager.from_provider_manager(pm, lambda k, d=None: d)
        media.register_adapter("fake", FakeMediaProvider("fake", poll_count=0))
        result = await media.images.generate("a tabby", n=2)
        assert len(result.artifacts) == 2
        job = await media.video.generate("dunes")
        assert (await media.wait(job)).usage.seconds == 4.0
