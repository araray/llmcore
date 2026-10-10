# tests/runtimes/test_cli_degraded.py
"""The commands that stop you spending money must work when the library cannot.

``runtimes/cli.py`` says why it exists: *"the commands that matter most here
are the ones you need when something has gone wrong... the thing you are trying
to do is stop paying for a VM."* It then built a full ``LLMCore`` for every
command — pulling in every provider SDK, the prompt pack, storage and
embeddings — so one missing vendor library, or a ``default_provider`` that no
longer loads, took ``status`` and ``down`` down with it.

Reproduced with a config whose default provider does not exist:
``estimate`` and ``status`` both failed outright, while the sizing they needed
touches no provider at all.
"""

from __future__ import annotations

import argparse

import pytest

from llmcore.exceptions import ConfigError
from llmcore.runtimes.cli import _BareRuntimes, _llm, _standalone_config


def args(config: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(config=config)


class TestStandaloneConfig:
    def test_it_reads_runtimes_config_without_building_llmcore(self, tmp_path):
        path = tmp_path / "c.toml"
        path.write_text(
            "[llmcore]\n"
            'default_provider = "a-provider-that-does-not-exist"\n'
            "\n[runtimes]\n"
            "enabled = true\n"
            'default_backend = "gpuai"\n'
        )
        config = _standalone_config(args(str(path)))
        assert config.get("runtimes.enabled", False) is True
        assert config.get("runtimes.default_backend", None) == "gpuai"

    def test_package_defaults_are_present(self):
        """Without them a bare manager would see an empty config and disagree
        with the same command run normally."""
        config = _standalone_config(args())
        assert config.get("runtimes.defaults.idle_minutes", None) is not None


class TestDegradation:
    @pytest.fixture
    def broken(self, monkeypatch):
        async def explode(*a, **k):
            raise ConfigError("Default provider 'nope' is not configured")

        monkeypatch.setattr("llmcore.api.LLMCore.create", explode)

    @pytest.mark.asyncio
    async def test_a_degradable_command_gets_a_runtimes_only_stand_in(
        self, broken, capsys
    ):
        llm = await _llm(args(), degrade=True)
        assert isinstance(llm, _BareRuntimes)
        assert llm.runtimes.backends  # the backends still registered
        warning = capsys.readouterr().err
        assert "could not fully initialise" in warning
        assert "provisioning and attaching are unavailable" in warning

    @pytest.mark.asyncio
    async def test_the_stand_in_closes_without_complaint(self, broken):
        llm = await _llm(args(), degrade=True)
        await llm.close()

    @pytest.mark.asyncio
    async def test_a_spending_command_refuses_to_degrade(self, broken):
        """``up`` and ``adopt`` attach the runtime as a provider instance, so
        for them a failure to initialise is a real failure. Silently serving a
        crippled object would mean provisioning compute and then being unable
        to route to it."""
        with pytest.raises(ConfigError):
            await _llm(args(), degrade=False)

    @pytest.mark.asyncio
    async def test_nothing_degrades_when_the_library_is_healthy(self, monkeypatch):
        sentinel = object()

        async def ok(*a, **k):
            return sentinel

        monkeypatch.setattr("llmcore.api.LLMCore.create", ok)
        assert await _llm(args(), degrade=True) is sentinel

    def test_the_commands_that_degrade_are_the_read_only_and_teardown_ones(self):
        """Pinned deliberately: if a future command is added to the spending
        set, this is the test that should be revisited."""
        import inspect

        from llmcore.runtimes import cli

        degrades, strict = set(), set()
        for name, function in vars(cli).items():
            if not name.startswith("_cmd_"):
                continue
            source = inspect.getsource(function)
            (degrades if "degrade=True" in source else strict).add(
                name.removeprefix("_cmd_")
            )
        # `doctor` degrades above all: diagnosing a broken environment is the
        # one job that must survive the environment being broken.
        assert degrades == {
            "estimate", "status", "down", "logs", "bake", "cache", "doctor",
            "spend",
        }
        assert strict == {"up", "adopt"}
