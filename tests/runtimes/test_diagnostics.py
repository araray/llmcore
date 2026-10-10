# tests/runtimes/test_diagnostics.py
"""Preflight checks — ``llmcore-runtimes doctor``.

llmcore had no answer to ``agent-lens colab doctor``, so it discovered each
problem one at a time and only when it was in the way. The first live run of
this found a real one immediately: llmcore's default Colab ``cli_path`` is a
bare ``colab``, while the CLI on this machine lives in a venv — so llmcore
could not have driven Colab at all, and would only have said so at provision
time, after the decision to spend.

Severity is graded by *consequence* rather than by state, because that is what
decides whether to act. ``MONEY`` outranks everything: a broken backend costs
nothing, and a forgotten instance costs money every hour.
"""

from __future__ import annotations

import pytest

from llmcore.runtimes import CostUnit, RuntimePhase, RuntimeStatus
from llmcore.runtimes.diagnostics import (
    Check,
    Severity,
    binary_check,
    key_check,
    reachable_check,
    run_diagnostics,
)

pytestmark = pytest.mark.asyncio


class FakeState:
    def __init__(self, directory) -> None:
        self.directory = directory


class FakeManager:
    """Only what ``run_diagnostics`` touches."""

    def __init__(self, directory, *, enabled=True, backends=None, statuses=None):
        self.state = FakeState(directory)
        self.enabled = enabled
        self._registry = backends or {}
        self.backends = sorted(self._registry)
        self._statuses = statuses if statuses is not None else []

    def _backend_for(self, name):
        if name not in self._registry:
            raise KeyError(f"no backend {name!r}")
        return self._registry[name]

    async def status(self, name=None):
        return self._statuses


def by_name(checks):
    return {c.name: c for c in checks}


class TestGenericChecks:
    async def test_a_writable_state_directory_passes(self, tmp_path):
        checks = by_name(await run_diagnostics(FakeManager(tmp_path)))
        assert checks["state: directory"].ok

    async def test_an_unwritable_state_directory_blocks(self, tmp_path):
        """Graded BLOCKS deliberately. llmcore writes state *before*
        provisioning so a crash cannot lose a VM; if that write cannot happen,
        provisioning must not start."""
        locked = tmp_path / "locked"
        locked.mkdir()
        locked.chmod(0o500)
        try:
            checks = by_name(await run_diagnostics(FakeManager(locked / "state")))
            assert checks["state: directory"].severity == Severity.BLOCKS
            assert "runtimes.state_dir" in checks["state: directory"].fix
        finally:
            locked.chmod(0o700)

    async def test_a_disabled_subsystem_degrades_rather_than_blocks(self, tmp_path):
        """Sizing and status work regardless, so this is not fatal."""
        checks = by_name(await run_diagnostics(FakeManager(tmp_path, enabled=False)))
        assert checks["subsystem"].severity == Severity.DEGRADES

    async def test_nothing_running_is_reported_as_such(self, tmp_path):
        checks = by_name(await run_diagnostics(FakeManager(tmp_path)))
        assert checks["billing"].ok


class TestBillingOutranksEverything:
    async def test_a_running_runtime_is_flagged_as_money(self, tmp_path):
        running = RuntimeStatus(
            name="qwen", runtime="gpuai", phase=RuntimePhase.READY,
            spend_used=4.2, spend_unit=CostUnit.USD,
        )
        checks = by_name(await run_diagnostics(FakeManager(tmp_path, statuses=[running])))
        found = checks["billing: running"]
        assert found.severity == Severity.MONEY
        assert "qwen" in found.detail

    async def test_an_orphan_says_how_to_make_it_killable(self, tmp_path):
        orphan = RuntimeStatus(
            name="stray", runtime="gpuai", phase=RuntimePhase.DEGRADED,
            error="orphan: gpu.ai instance inst-9 is running",
        )
        checks = by_name(await run_diagnostics(FakeManager(tmp_path, statuses=[orphan])))
        found = checks["billing: orphans"]
        assert found.severity == Severity.MONEY
        assert "adopt" in found.fix

    async def test_a_status_failure_degrades_instead_of_crashing(self, tmp_path):
        manager = FakeManager(tmp_path)

        async def explode(name=None):
            raise RuntimeError("backend unreachable")

        manager.status = explode
        checks = by_name(await run_diagnostics(manager))
        assert checks["billing"].severity == Severity.DEGRADES


class TestBackendChecks:
    async def test_a_backend_is_asked_for_its_own_checks(self, tmp_path):
        class Backend:
            async def diagnose(self):
                return [Check("fake: thing", Severity.OK, "fine")]

        checks = by_name(
            await run_diagnostics(FakeManager(tmp_path, backends={"fake": Backend()}))
        )
        assert checks["fake: thing"].ok

    async def test_a_backend_without_diagnose_is_not_an_error(self, tmp_path):
        """Discovery is structural, like the backend protocol itself."""
        checks = by_name(
            await run_diagnostics(FakeManager(tmp_path, backends={"bare": object()}))
        )
        assert checks["bare: checks"].ok
        assert "no checks of its own" in checks["bare: checks"].detail

    async def test_a_backend_whose_checks_explode_degrades(self, tmp_path):
        class Hostile:
            async def diagnose(self):
                raise RuntimeError("boom")

        checks = by_name(
            await run_diagnostics(FakeManager(tmp_path, backends={"bad": Hostile()}))
        )
        assert checks["bad: checks"].severity == Severity.DEGRADES

    async def test_one_backend_can_be_selected(self, tmp_path):
        class Backend:
            def __init__(self, label):
                self.label = label

            async def diagnose(self):
                return [Check(f"{self.label}: x", Severity.OK, "fine")]

        manager = FakeManager(
            tmp_path, backends={"a": Backend("a"), "b": Backend("b")}
        )
        checks = by_name(await run_diagnostics(manager, backend="a"))
        assert "a: x" in checks
        assert "b: x" not in checks


class TestHelpers:
    async def test_a_credential_check_never_prints_the_value(self):
        check = key_check("gpuai", "GPUAI_API_KEY", value="sk-super-secret-123")
        assert check.ok
        assert "sk-super-secret-123" not in check.detail
        assert "GPUAI_API_KEY" in check.detail

    async def test_a_missing_credential_blocks_and_names_the_variable(self):
        check = key_check("gpuai", "GPUAI_API_KEY", value=None)
        assert check.severity == Severity.BLOCKS
        assert "GPUAI_API_KEY" in check.fix

    async def test_a_missing_binary_blocks(self):
        check = binary_check("colab", "a-binary-that-does-not-exist-anywhere")
        assert check.severity == Severity.BLOCKS

    async def test_a_present_binary_passes(self):
        check = binary_check("sh", "sh")
        assert check.ok

    async def test_an_unreachable_probe_degrades_rather_than_raising(self):
        async def explode():
            raise OSError("connection refused")

        check = await reachable_check("x", "thing", explode)
        assert check.severity == Severity.DEGRADES
        assert "connection refused" in check.detail

    async def test_a_slow_probe_times_out_rather_than_hanging(self):
        import asyncio

        async def slow():
            await asyncio.sleep(10)

        check = await reachable_check("x", "thing", slow, timeout=0.05)
        assert check.severity == Severity.DEGRADES
        assert "timed out" in check.detail


class TestShippedBackendsPublishChecks:
    async def test_all_three_have_diagnose(self):
        """A new backend is free not to, but the shipped ones should —
        otherwise `doctor` is silent about the thing you are about to use."""
        from llmcore.runtimes.colab import ColabRuntime
        from llmcore.runtimes.deepinfra import DeepInfraRuntime
        from llmcore.runtimes.gpuai import GpuAiRuntime

        for backend in (ColabRuntime, GpuAiRuntime, DeepInfraRuntime):
            assert callable(getattr(backend, "diagnose", None)), backend.__name__

    async def test_a_keyless_rental_backend_stops_after_the_credential(self, tmp_path):
        """Four copies of "no key" is noise; say it once."""
        from llmcore.runtimes.deepinfra import DeepInfraRuntime
        from llmcore.runtimes.state import RuntimeStateStore

        backend = DeepInfraRuntime(
            config=type("C", (), {"get": staticmethod(lambda k, d=None: d)})(),
            state_store=RuntimeStateStore(tmp_path),
            api_key=None,
        )
        import os

        saved = {k: os.environ.pop(k, None) for k in ("DEEPINFRA_API_KEY", "DEEPINFRA_TOKEN")}
        try:
            checks = await backend.diagnose()
        finally:
            for key, value in saved.items():
                if value is not None:
                    os.environ[key] = value
        assert len(checks) == 1
        assert checks[0].severity == Severity.BLOCKS
