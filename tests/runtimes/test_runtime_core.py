# tests/runtimes/test_runtime_core.py
"""The runtime subsystem core (spec phase R1).

Most of this file tests a **safety model** rather than a feature. Every other
llmcore provider is stateless and bills per request; a runtime bills per minute
from the moment it is assigned, whether or not anyone calls it. The spec calls
its five rules "non-negotiable design constraints rather than polish", so each
one gets explicit coverage here:

1. no implicit spend,
2. no implicit persistence of spend,
3. bounded by default,
4. fail closed,
5. no implicit secrets (backend-owned; the manager's part is tested).

The gate for this phase is *"dynamic provider registration works and is
covered"*, which :class:`TestProviderAttachment` does.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from llmcore.exceptions import ConfigError
from llmcore.runtimes import (
    ComputeRuntime,
    ModelSpec,
    Plan,
    Quantization,
    RuntimeHandle,
    RuntimeManager,
    RuntimePhase,
    RuntimeStateStore,
    RuntimeStatus,
    SpendNotConfirmedError,
)
from llmcore.runtimes.manager import RuntimeError_
from llmcore.runtimes.testing import FakeRuntime

REPO = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def _config(**overrides: Any):
    values: dict[str, Any] = {
        "runtimes.enabled": True,
        "runtimes.default_backend": "fake",
        "runtimes.defaults.confirm_spend": True,
        "runtimes.defaults.idle_minutes": 45,
        "runtimes.defaults.max_lifetime_minutes": 240,
    }
    values.update(overrides)
    return lambda key, default=None: values.get(key, default)


@pytest.fixture
def store(tmp_path: Path) -> RuntimeStateStore:
    return RuntimeStateStore(tmp_path / "runtimes")


@pytest.fixture
def manager(tmp_path: Path):
    backend = FakeRuntime()
    mgr = RuntimeManager(
        {"fake": backend},
        config_get=_config(**{"runtimes.state_dir": str(tmp_path / "state")}),
    )
    return mgr, backend


def _handle(name: str = "r1", **kw: Any) -> RuntimeHandle:
    return RuntimeHandle(
        name=name,
        runtime=kw.pop("runtime", "fake"),
        external_id=kw.pop("external_id", "ext-1"),
        base_url=kw.pop("base_url", "http://127.0.0.1:8111/v1"),
        served_model=kw.pop("served_model", REPO),
        **kw,
    )


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class TestModelSpec:
    def test_requires_an_owner_qualified_repo(self):
        with pytest.raises(ValueError, match="owner/name"):
            ModelSpec("justaname")

    def test_rejects_a_nonpositive_context(self):
        with pytest.raises(ValueError, match="context_length"):
            ModelSpec(REPO, context_length=0)

    def test_remote_code_is_off_by_default(self):
        """Running arbitrary code from a model repo should be a visible choice."""
        assert ModelSpec(REPO).trust_remote_code is False


class TestPlan:
    def test_headroom_is_derived(self):
        plan = Plan(spec=ModelSpec(REPO), sku="L4",
                    vram_required_gb=20.0, vram_available_gb=22.5)
        assert plan.headroom_gb == pytest.approx(2.5)

    def test_negative_headroom_signals_a_bad_fit(self):
        plan = Plan(spec=ModelSpec(REPO), sku="T4",
                    vram_required_gb=40.0, vram_available_gb=16.0, fits=False)
        assert plan.headroom_gb < 0

    def test_notes_accumulate(self):
        plan = Plan(spec=ModelSpec(REPO), sku="L4", notes=("a",)).with_notes("b", "c")
        assert plan.notes == ("a", "b", "c")


class TestRuntimePhase:
    @pytest.mark.parametrize(
        ("phase", "billing"),
        [
            (RuntimePhase.PLANNED, False),
            (RuntimePhase.STARTING, True),
            (RuntimePhase.READY, True),
            (RuntimePhase.DEGRADED, True),
            (RuntimePhase.STOPPING, True),
            (RuntimePhase.STOPPED, False),
            (RuntimePhase.FAILED, False),
        ],
    )
    def test_which_phases_cost_money(self, phase, billing):
        """DEGRADED counts: a broken runtime is still an assigned one."""
        assert phase.is_billing is billing

    def test_terminal_phases(self):
        assert RuntimePhase.STOPPED.is_terminal and RuntimePhase.FAILED.is_terminal
        assert not RuntimePhase.READY.is_terminal


class TestSpendCeilings:
    """Rule 3: bounded by default — and bounded against the *busy* case too."""

    def test_a_fresh_runtime_is_not_expired(self):
        assert _handle().expired_reason() is None

    def test_hard_deadline_expires(self):
        past = datetime.now(timezone.utc) - timedelta(minutes=1)
        reason = _handle(hard_deadline=past).expired_reason()
        assert "hard lifetime deadline" in reason

    def test_idle_deadline_expires(self):
        past = datetime.now(timezone.utc) - timedelta(minutes=1)
        assert "idle since" in _handle(idle_deadline=past).expired_reason()

    def test_compute_ceiling_takes_precedence(self):
        """An idle reaper does not protect against a runtime busy in a loop,
        which is the expensive failure mode."""
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        handle = _handle(
            idle_deadline=future, hard_deadline=future,
            max_compute_units=10.0, compute_units_used=10.0,
        )
        assert "compute ceiling" in handle.expired_reason()

    def test_compute_ceiling_is_ignored_when_unreported(self):
        handle = _handle(max_compute_units=10.0, compute_units_used=None)
        assert handle.expired_reason() is None

    def test_touch_pushes_the_idle_deadline_out(self):
        now = datetime.now(timezone.utc)
        handle = _handle(idle_deadline=now + timedelta(minutes=1))
        handle.metadata["idle_minutes"] = 45
        handle.touch(now)
        assert handle.idle_deadline > now + timedelta(minutes=40)
        assert handle.last_activity_at == now

    def test_touch_does_not_create_a_deadline(self):
        """A runtime with reaping disabled must not gain one by being used."""
        handle = _handle(idle_deadline=None)
        handle.touch()
        assert handle.idle_deadline is None


class TestSerialization:
    def test_roundtrip_preserves_the_fields_that_matter(self):
        now = datetime.now(timezone.utc)
        original = _handle(
            sku="A100-40", phase=RuntimePhase.READY, started_at=now,
            hard_deadline=now + timedelta(hours=2), max_compute_units=50.0,
        )
        restored = RuntimeHandle.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.phase is RuntimePhase.READY
        assert restored.sku == "A100-40"
        assert restored.max_compute_units == 50.0
        assert restored.hard_deadline == original.hard_deadline

    def test_json_is_human_readable(self):
        """Someone who suspects they are being billed must be able to `cat` it."""
        payload = _handle().to_json()
        assert "\n" in payload and '  "name"' in payload
        assert json.loads(payload)["name"] == "r1"

    def test_an_unknown_phase_degrades_rather_than_raising(self):
        """A state file from a newer llmcore still describes a VM burning money;
        refusing to parse it would hide that."""
        handle = RuntimeHandle.from_dict({"name": "x", "phase": "teleporting"})
        assert handle.phase is RuntimePhase.DEGRADED

    def test_garbage_timestamps_do_not_raise(self):
        handle = RuntimeHandle.from_dict({"name": "x", "started_at": "not-a-date"})
        assert handle.started_at is not None


class TestRuntimeStatus:
    def test_derives_uptime_and_expiry(self):
        now = datetime.now(timezone.utc)
        handle = _handle(
            started_at=now - timedelta(minutes=10),
            idle_deadline=now + timedelta(minutes=5),
            hard_deadline=now + timedelta(minutes=50),
        )
        status = RuntimeStatus.from_handle(handle, now=now, attached=True)
        assert status.uptime_seconds == pytest.approx(600, abs=2)
        # The *nearest* deadline is what matters to a caller.
        assert status.expires_in_seconds == pytest.approx(300, abs=2)
        assert status.attached is True

    def test_no_deadlines_means_no_expiry(self):
        assert RuntimeStatus.from_handle(_handle()).expires_in_seconds is None


# ---------------------------------------------------------------------------
# State store — rule 2
# ---------------------------------------------------------------------------


class TestStateStore:
    def test_save_then_load(self, store):
        store.save(_handle("alpha", sku="L4"))
        loaded = store.load("alpha")
        assert loaded.sku == "L4"

    def test_missing_record_is_none_not_an_error(self, store):
        assert store.load("nope") is None

    def test_save_records_the_path_on_the_handle(self, store):
        handle = _handle("alpha")
        path = store.save(handle)
        assert handle.state_path == path
        assert path.is_file()

    def test_names_are_sanitized_into_filenames(self, store):
        store.save(_handle("weird/name with spaces"))
        assert any(store.directory.iterdir())

    def test_load_all_is_newest_first(self, store):
        now = datetime.now(timezone.utc)
        store.save(_handle("old", started_at=now - timedelta(hours=1)))
        store.save(_handle("new", started_at=now))
        assert [h.name for h in store.load_all()] == ["new", "old"]

    def test_a_corrupt_record_does_not_hide_the_others(self, store):
        """One bad file must not conceal runtimes that are still billing."""
        store.save(_handle("good"))
        store.directory.joinpath("broken.json").write_text("{not json")
        names = [h.name for h in store.load_all()]
        assert names == ["good"]

    def test_a_non_object_record_is_skipped(self, store):
        store.save(_handle("good"))
        store.directory.joinpath("list.json").write_text("[1, 2, 3]")
        assert [h.name for h in store.load_all()] == ["good"]

    def test_delete_reports_whether_anything_existed(self, store):
        store.save(_handle("alpha"))
        assert store.delete("alpha") is True
        assert store.delete("alpha") is False

    def test_no_temp_files_are_left_behind(self, store):
        store.save(_handle("alpha"))
        assert not list(store.directory.glob(".tmp-*"))

    def test_load_all_on_a_missing_directory_is_empty(self, tmp_path):
        assert RuntimeStateStore(tmp_path / "absent").load_all() == []


# ---------------------------------------------------------------------------
# Rule 1: no implicit spend
# ---------------------------------------------------------------------------


class TestNoImplicitSpend:
    def test_the_subsystem_is_disabled_by_default(self):
        assert RuntimeManager({"fake": FakeRuntime()}).enabled is False

    def test_construction_provisions_nothing(self):
        backend = FakeRuntime()
        RuntimeManager({"fake": backend}, config_get=_config())
        assert backend.calls == []

    async def test_up_refuses_while_disabled(self):
        backend = FakeRuntime()
        mgr = RuntimeManager({"fake": backend}, config_get=_config(**{"runtimes.enabled": False}))
        with pytest.raises(RuntimeError_, match="subsystem is disabled"):
            await mgr.up(REPO, name="r", confirm_spend=True)
        assert backend.called("up") == 0

    async def test_up_refuses_without_spend_confirmation(self, manager):
        mgr, backend = manager
        with pytest.raises(SpendNotConfirmedError, match="costs money per minute"):
            await mgr.up(REPO, name="r")
        assert backend.called("up") == 0

    async def test_confirmation_can_be_waived_by_config(self, tmp_path):
        backend = FakeRuntime()
        mgr = RuntimeManager(
            {"fake": backend},
            config_get=_config(**{
                "runtimes.defaults.confirm_spend": False,
                "runtimes.state_dir": str(tmp_path / "s"),
            }),
        )
        await mgr.up(REPO, name="r", attach=False)
        assert backend.called("up") == 1

    async def test_estimate_is_free_and_works_while_disabled(self):
        """Deciding whether to spend should not require enabling spend."""
        backend = FakeRuntime()
        mgr = RuntimeManager({"fake": backend}, config_get=_config(**{"runtimes.enabled": False}))
        plan = await mgr.estimate(REPO, context_length=32768)
        assert plan.sku == "L4"
        assert backend.called("up") == 0

    async def test_a_plan_that_does_not_fit_is_refused(self, tmp_path):
        backend = FakeRuntime(fits=False)
        mgr = RuntimeManager(
            {"fake": backend},
            config_get=_config(**{"runtimes.state_dir": str(tmp_path / "s")}),
        )
        with pytest.raises(RuntimeError_, match="does not fit"):
            await mgr.up(REPO, name="r", confirm_spend=True)
        assert backend.called("up") == 0

    async def test_a_duplicate_name_is_refused(self, manager):
        mgr, _ = manager
        await mgr.up(REPO, name="r", confirm_spend=True, attach=False)
        with pytest.raises(RuntimeError_, match="already running"):
            await mgr.up(REPO, name="r", confirm_spend=True, attach=False)


# ---------------------------------------------------------------------------
# Rules 2 and 3
# ---------------------------------------------------------------------------


class TestPersistenceAndBounds:
    async def test_up_persists_state(self, manager):
        mgr, _ = manager
        handle = await mgr.up(REPO, name="r", confirm_spend=True, attach=False)
        assert handle.state_path.is_file()
        assert mgr.state.load("r").served_model == REPO

    async def test_deadlines_come_from_config_defaults(self, manager):
        """Bounded by default, not by the caller remembering."""
        mgr, _ = manager
        handle = await mgr.up(REPO, name="r", confirm_spend=True, attach=False)
        assert handle.idle_deadline is not None
        assert handle.hard_deadline is not None
        assert handle.metadata["idle_minutes"] == 45
        assert handle.metadata["max_lifetime_minutes"] == 240

    async def test_zero_disables_a_reaper(self, manager):
        mgr, _ = manager
        handle = await mgr.up(
            REPO, name="r", confirm_spend=True, attach=False,
            idle_minutes=0, max_lifetime_minutes=0,
        )
        assert handle.idle_deadline is None and handle.hard_deadline is None

    async def test_a_compute_ceiling_can_be_set(self, manager):
        mgr, _ = manager
        handle = await mgr.up(
            REPO, name="r", confirm_spend=True, attach=False, max_compute_units=25.0
        )
        assert handle.max_compute_units == 25.0

    async def test_status_includes_runtimes_from_other_processes(self, manager, tmp_path):
        """A runtime started by a previous session is still costing money."""
        mgr, _ = manager
        mgr.state.save(_handle("from-elsewhere"))
        names = {s.name for s in await mgr.status()}
        assert "from-elsewhere" in names

    async def test_reap_stops_expired_runtimes(self, manager):
        mgr, backend = manager
        handle = await mgr.up(REPO, name="r", confirm_spend=True, attach=False)
        handle.hard_deadline = datetime.now(timezone.utc) - timedelta(seconds=1)
        reaped = await mgr.reap()
        assert [n for n, _ in reaped] == ["r"]
        assert "r" in backend.released

    async def test_reap_leaves_healthy_runtimes_alone(self, manager):
        mgr, backend = manager
        await mgr.up(REPO, name="r", confirm_spend=True, attach=False)
        assert await mgr.reap() == []
        assert backend.released == []


# ---------------------------------------------------------------------------
# Rule 4: fail closed
# ---------------------------------------------------------------------------


class TestFailClosed:
    async def test_a_bootstrap_failure_propagates_without_leaving_state(self, tmp_path):
        backend = FakeRuntime(fail_on_up=True)
        mgr = RuntimeManager(
            {"fake": backend},
            config_get=_config(**{"runtimes.state_dir": str(tmp_path / "s")}),
        )
        with pytest.raises(RuntimeError, match="fake bootstrap failure"):
            await mgr.up(REPO, name="r", confirm_spend=True)
        assert mgr.state.load("r") is None

    async def test_an_attach_failure_releases_the_runtime(self, tmp_path):
        """A runtime we cannot reach is still billing, so release it rather
        than leaving it for the reaper."""

        class _BadProviders:
            def register_instance(self, *a, **k):
                raise RuntimeError("registry is full")

            async def unregister_instance(self, *a, **k):
                return False

        backend = FakeRuntime()
        mgr = RuntimeManager(
            {"fake": backend},
            provider_manager=_BadProviders(),
            config_get=_config(**{"runtimes.state_dir": str(tmp_path / "s")}),
        )
        with pytest.raises(RuntimeError_, match="was released rather than left running"):
            await mgr.up(REPO, name="r", confirm_spend=True)
        assert "r" in backend.released

    async def test_down_is_idempotent(self, manager):
        mgr, _ = manager
        await mgr.up(REPO, name="r", confirm_spend=True, attach=False)
        assert await mgr.down("r") is True
        assert await mgr.down("r") is False

    async def test_down_on_an_unknown_name_does_not_raise(self, manager):
        """This is the function someone reaches for when things already broke."""
        mgr, _ = manager
        assert await mgr.down("never-existed") is False

    async def test_a_release_failure_keeps_the_state_record(self, manager):
        """If we could not stop it, we must not forget it."""
        mgr, backend = manager
        await mgr.up(REPO, name="r", confirm_spend=True, attach=False)

        async def _boom(name, *, release=True):
            raise RuntimeError("backend unreachable")

        backend.down = _boom
        assert await mgr.down("r") is False
        assert mgr.state.load("r") is not None

    async def test_an_unknown_backend_in_state_is_reported_not_fatal(self, manager):
        mgr, _ = manager
        mgr.state.save(_handle("orphan", runtime="runpod"))
        assert await mgr.down("orphan") is False
        assert mgr.state.load("orphan") is not None


# ---------------------------------------------------------------------------
# The phase gate: dynamic provider registration
# ---------------------------------------------------------------------------


class _RecordingProviders:
    """Minimal stand-in for ProviderManager's dynamic-registration surface."""

    def __init__(self) -> None:
        self.registered: dict[str, tuple[str, dict[str, Any]]] = {}
        self.unregistered: list[str] = []
        self.flags: dict[str, dict[str, Any]] = {}

    def register_instance(
        self,
        name: str,
        provider_type: str,
        config: dict[str, Any],
        *,
        ephemeral: bool = False,
        replace: bool = False,
    ) -> None:
        self.registered[name] = (provider_type, config)
        self.flags[name] = {"ephemeral": ephemeral, "replace": replace}

    async def unregister_instance(self, name: str, *, close: bool = True) -> bool:
        self.unregistered.append(name)
        return self.registered.pop(name, None) is not None


class TestProviderAttachment:
    """The R1 gate: a runtime's endpoint becomes a usable provider instance."""

    @pytest.fixture
    def wired(self, tmp_path):
        backend = FakeRuntime()
        providers = _RecordingProviders()
        mgr = RuntimeManager(
            {"fake": backend},
            provider_manager=providers,
            config_get=_config(**{"runtimes.state_dir": str(tmp_path / "s")}),
        )
        return mgr, backend, providers

    async def test_up_registers_a_provider_instance(self, wired):
        mgr, _, providers = wired
        await mgr.up(REPO, name="qwen30", confirm_spend=True)
        assert "qwen30" in providers.registered
        provider_type, config = providers.registered["qwen30"]
        assert provider_type == "vllm", "an OpenAI-compatible endpoint needs no new class"
        assert config["base_url"] == "http://127.0.0.1:8111/v1"
        assert config["default_model"] == REPO
        assert mgr.attached == ["qwen30"]

    async def test_the_instance_is_marked_ephemeral(self, wired):
        """Without the flag the instance would outlive its runtime and hand
        callers a dead endpoint on close_all()."""
        mgr, _, providers = wired
        await mgr.up(REPO, name="qwen30", confirm_spend=True)
        assert providers.flags["qwen30"]["ephemeral"] is True

    async def test_reattaching_replaces_rather_than_failing(self, wired):
        """A runtime may legitimately be re-attached after a reconnect."""
        mgr, _, providers = wired
        handle = await mgr.up(REPO, name="qwen30", confirm_spend=True)
        mgr.attach(handle)
        assert providers.flags["qwen30"]["replace"] is True

    async def test_api_style_selects_the_provider_type(self, wired):
        """A future recipe speaking another protocol attaches a different type
        without touching the runtime layer."""
        mgr, _, providers = wired
        handle = _handle("custom", api_style="vllm")
        mgr.attach(handle)
        assert providers.registered["custom"][0] == "vllm"

    async def test_an_unknown_api_style_is_refused(self, wired):
        mgr, _, _ = wired
        with pytest.raises(RuntimeError_, match="maps to no provider type"):
            mgr.attach(_handle("weird", api_style="grpc-telepathy"))

    async def test_down_unregisters_then_releases(self, wired):
        mgr, backend, providers = wired
        await mgr.up(REPO, name="qwen30", confirm_spend=True)
        await mgr.down("qwen30")
        assert providers.unregistered == ["qwen30"]
        assert "qwen30" not in providers.registered
        assert "qwen30" in backend.released
        assert mgr.attached == []

    async def test_attach_without_a_provider_manager_is_refused(self, manager):
        mgr, _ = manager
        with pytest.raises(RuntimeError_, match="no ProviderManager"):
            mgr.attach(_handle())

    async def test_attach_can_be_skipped(self, wired):
        mgr, _, providers = wired
        await mgr.up(REPO, name="bare", confirm_spend=True, attach=False)
        assert providers.registered == {}

    async def test_close_detaches_without_stopping_anything(self, wired):
        """A process exiting is not a reason to destroy paid compute."""
        mgr, backend, providers = wired
        await mgr.up(REPO, name="qwen30", confirm_spend=True)
        await mgr.close()
        assert providers.unregistered == ["qwen30"]
        assert backend.released == [], "close() must not release compute"
        assert mgr.state.load("qwen30") is not None, "state must survive for recovery"

    async def test_down_all_stops_everything_including_disk_records(self, wired):
        mgr, backend, _ = wired
        await mgr.up(REPO, name="a", confirm_spend=True)
        await mgr.up(REPO, name="b", confirm_spend=True)
        stopped = await mgr.down_all()
        assert sorted(stopped) == ["a", "b"]
        assert sorted(backend.released) == ["a", "b"]

    async def test_adopt_registers_a_leaked_runtime(self, wired):
        """The recovery path: something is billing and nothing tracks it."""
        mgr, _backend, providers = wired
        handle = await mgr.adopt("colab-session-xyz", name="rescued")
        assert handle.external_id == "colab-session-xyz"
        assert "rescued" in providers.registered
        assert mgr.state.load("rescued") is not None

    async def test_adopt_refuses_while_disabled(self):
        backend = FakeRuntime()
        mgr = RuntimeManager({"fake": backend}, config_get=_config(**{"runtimes.enabled": False}))
        with pytest.raises(RuntimeError_, match="subsystem is disabled"):
            await mgr.adopt("x", name="y")


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class TestBackendRegistry:
    def test_fake_runtime_satisfies_the_protocol(self):
        assert isinstance(FakeRuntime(), ComputeRuntime)

    def test_a_non_conforming_backend_is_refused(self):
        class NotARuntime:
            name = "nope"

        with pytest.raises(ConfigError, match="ComputeRuntime protocol"):
            RuntimeManager().register_backend(NotARuntime())

    def test_register_backend_uses_its_own_name(self):
        mgr = RuntimeManager()
        mgr.register_backend(FakeRuntime("colab"))
        assert mgr.backends == ["colab"]

    async def test_an_unknown_backend_is_an_actionable_error(self):
        """The error has to name what *is* available. `colab` is registered by
        default now, so this asserts the behaviour rather than a fixed string.
        """
        mgr = RuntimeManager({"fake": FakeRuntime()}, config_get=_config())
        with pytest.raises(ConfigError) as excinfo:
            await mgr.estimate(REPO, backend="runpod")
        message = str(excinfo.value)
        assert "runpod" in message and "fake" in message
        assert "Available:" in message

    async def test_logs_stream(self):
        backend = FakeRuntime()
        lines = [line async for line in backend.logs("r", component="server")]
        assert len(lines) == 2 and "server" in lines[0]

    async def test_quantization_strings_are_coerced(self):
        mgr = RuntimeManager({"fake": FakeRuntime()}, config_get=_config())
        plan = await mgr.estimate(REPO, quantization="awq")
        assert plan.quantization is Quantization.AWQ
