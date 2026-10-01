# tests/runtimes/test_colab.py
"""The Colab backend (spec phase R3-R5).

This backend spends money, so the tests concentrate on the four rules the
module exists to enforce, in the order the spec states them:

1. state is written **before** compute can be assigned;
2. any bootstrap failure **releases** the VM;
3. nothing is connected to until the session actually exists;
4. deadlines are set at creation.

Every external command goes through one seam (``ColabRuntime._run``), so the
tests replace that and assert on the argv the backend would have run. That is
the only honest way to test this without a real VM: a mock further in would
test the mock, and a real VM costs money per minute.
"""

from __future__ import annotations

import json

import pytest

from llmcore.runtimes.colab import (
    LLAMACPP_RECIPE,
    VLLM_RECIPE,
    ColabRuntime,
    CommandResult,
    RuntimeError_,
    _bake_script,
    _bootstrap_script,
    _free_port,
    _parse_sessions,
)
from llmcore.runtimes.models import ModelSpec, Plan, Quantization, RuntimePhase
from llmcore.runtimes.state import RuntimeStateStore


def result(argv, rc=0, stdout="", stderr="") -> CommandResult:
    return CommandResult(tuple(argv), rc, stdout, stderr, 0.01)


class FakeCli:
    """A stand-in for the Colab CLI, with a little state.

    It tracks which sessions have been created and synthesises the
    ``colab sessions`` table from them. That matters: the real sequence is
    ``new -s NAME`` and *then* the session exists. A fixed table would either
    let the assignment guard pass for a session that was never created, or make
    it poll the full timeout for one that was.

    Keys in ``answers`` match the first one or two arguments after the binary,
    so ``("new",)`` covers ``colab new -s x --gpu L4``.
    """

    def __init__(
        self,
        answers: dict[tuple[str, ...], CommandResult] | None = None,
        *,
        preexisting: tuple[str, ...] = (),
        gpu: str = "L4",
    ) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.stdins: list[str | None] = []
        self.answers = answers or {}
        self.sessions: dict[str, str] = dict.fromkeys(preexisting, gpu)

    def _table(self) -> str:
        if not self.sessions:
            return "(no sessions)\n"
        rows = "\n".join(
            f"| {name:12} | {gpu:6} | running |" for name, gpu in self.sessions.items()
        )
        return f"| Name | GPU | Status |\n|------|-----|--------|\n{rows}\n"

    @staticmethod
    def _session_of(argv: tuple[str, ...]) -> str | None:
        return argv[argv.index("-s") + 1] if "-s" in argv else None

    @staticmethod
    def _gpu_of(argv: tuple[str, ...]) -> str:
        return argv[argv.index("--gpu") + 1] if "--gpu" in argv else "CPU"

    async def __call__(
        self,
        *argv: str,
        timeout=120.0,  # noqa: ASYNC109 - matches the seam it replaces
        stdin=None,
        check=False,
    ):
        self.calls.append(tuple(argv))
        self.stdins.append(stdin)

        verb = argv[1] if len(argv) > 1 else ""
        session = self._session_of(argv)

        for width in (2, 1):
            key = tuple(argv[1 : 1 + width])
            if key in self.answers:
                answer = self.answers[key]
                if answer.ok and verb == "new" and session:
                    self.sessions[session] = self._gpu_of(argv)
                if check and not answer.ok:
                    raise RuntimeError_(answer.brief())
                return answer

        if verb == "new" and session:
            self.sessions[session] = self._gpu_of(argv)
        elif verb == "stop" and session:
            self.sessions.pop(session, None)
        elif verb == "sessions":
            return result(argv, stdout=self._table())
        return result(argv)

    def argv_for(self, verb: str) -> tuple[str, ...] | None:
        return next((call for call in self.calls if len(call) > 1 and call[1] == verb), None)

    def count(self, verb: str) -> int:
        return sum(1 for call in self.calls if len(call) > 1 and call[1] == verb)


#: A table in the shape the real CLI prints, for the parser tests.
SESSIONS_TABLE = """\
┌──────────────┬────────┬─────────┐
│ Name         │ GPU    │ Status  │
├──────────────┼────────┼─────────┤
│ qwen30       │ L4     │ running │
│ other        │ A100   │ running │
└──────────────┴────────┴─────────┘
"""


@pytest.fixture
def store(tmp_path) -> RuntimeStateStore:
    return RuntimeStateStore(tmp_path / "runtimes")


async def _no_sessions() -> list[dict]:
    """`colab sessions` returning nothing, for the assignment-guard test."""
    return []


@pytest.fixture(autouse=True)
def fast_assignment(monkeypatch):
    """Shrink the assignment guard's timeout for the whole module.

    180 seconds is right in production -- Colab really does take that long to
    assign a GPU -- and ruinous in a test suite, where the negative case has to
    actually wait it out.
    """
    monkeypatch.setattr("llmcore.runtimes.colab.ASSIGNMENT_TIMEOUT", 1.0)
    monkeypatch.setattr("llmcore.runtimes.colab.READY_POLL", 0.01)


@pytest.fixture
def runtime(store, monkeypatch) -> ColabRuntime:
    rt = ColabRuntime(state_store=store)
    monkeypatch.setattr(rt, "_cli_binary", lambda: "colab")
    return rt


def plan_for(repo="Qwen/Qwen2.5-7B-Instruct", *, sku="L4", fits=True, recipe="vllm") -> Plan:
    return Plan(
        spec=ModelSpec(repo_id=repo, context_length=16384),
        sku=sku,
        recipe=recipe,
        quantization=Quantization.NONE,
        vram_required_gb=17.0,
        vram_available_gb=18.4,
        context_length=16384,
        fits=fits,
    )


def wire(runtime, monkeypatch, cli: FakeCli) -> FakeCli:
    """Replace every external interaction with fakes.

    No static sessions table: the fake synthesises one from the sessions it has
    been asked to create, so `new -s NAME` and then "NAME exists" stay in step.
    """
    monkeypatch.setattr(runtime, "_run", cli)

    async def no_tunnel(handle):
        handle.metadata["tunnel_pid"] = 4242

    async def ready(handle, recipe):
        return None

    monkeypatch.setattr(runtime, "_open_tunnel", no_tunnel)
    monkeypatch.setattr(runtime, "_await_ready", ready)
    monkeypatch.setattr(runtime, "_start_keepalive", lambda handle: None)
    return cli


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_constructing_contacts_nothing(self, monkeypatch):
        """LLMCore.create() builds a RuntimeManager, which now builds this. If
        the constructor touched a backend, importing llmcore could provision."""

        def explode(*a, **k):
            raise AssertionError("no subprocess may run during construction")

        monkeypatch.setattr("asyncio.create_subprocess_exec", explode)
        monkeypatch.setattr("shutil.which", explode)
        ColabRuntime()

    def test_it_satisfies_the_runtime_protocol(self, runtime):
        from llmcore.runtimes.protocols import ComputeRuntime

        assert isinstance(runtime, ComputeRuntime)

    def test_config_drives_the_sku_ladder(self):
        class Cfg:
            def get(self, key, default=None):
                return {"runtimes.colab.sku_ladder": ["T4", "L4"]}.get(key, default)

        assert ColabRuntime(config=Cfg())._sizer.ladder == ("T4", "L4")

    def test_the_cli_name_comes_from_config(self):
        class Cfg:
            def get(self, key, default=None):
                return {"runtimes.colab.cli_path": "/opt/colab"}.get(key, default)

        assert ColabRuntime(config=Cfg())._cli_path == "/opt/colab"


# ---------------------------------------------------------------------------
# estimate
# ---------------------------------------------------------------------------


class TestEstimate:
    @pytest.mark.asyncio
    async def test_it_adds_the_exact_command_and_the_burn_rate(self, runtime, monkeypatch):
        """Someone about to approve spend should see what will run and what it
        costs per hour of *being assigned*, not per request."""

        async def sized(spec):
            return plan_for()

        monkeypatch.setattr(runtime._sizer, "estimate", sized)
        plan = await runtime.estimate(ModelSpec(repo_id="x/y", context_length=4096))
        joined = " ".join(plan.notes)
        assert "colab new --gpu L4" in joined
        assert "compute units/hour while assigned" in joined

    @pytest.mark.asyncio
    async def test_estimating_runs_no_commands(self, runtime, monkeypatch):
        cli = wire(runtime, monkeypatch, FakeCli())
        await runtime.estimate(ModelSpec(repo_id="Qwen/Qwen2.5-7B-Instruct", context_length=4096))
        assert cli.calls == []


# ---------------------------------------------------------------------------
# up — the four rules
# ---------------------------------------------------------------------------


class TestUpRefusals:
    @pytest.mark.asyncio
    async def test_a_plan_that_does_not_fit_is_refused_before_anything_runs(
        self, runtime, monkeypatch
    ):
        cli = wire(runtime, monkeypatch, FakeCli())
        with pytest.raises(RuntimeError_, match="does not fit"):
            await runtime.up(plan_for(fits=False), name="x")
        assert cli.calls == []

    @pytest.mark.asyncio
    async def test_an_unknown_sku_is_refused(self, runtime, monkeypatch):
        wire(runtime, monkeypatch, FakeCli())
        with pytest.raises(RuntimeError_, match="Unknown SKU"):
            await runtime.up(plan_for(sku="B200"), name="x")

    @pytest.mark.asyncio
    async def test_an_unknown_recipe_is_refused(self, runtime, monkeypatch):
        wire(runtime, monkeypatch, FakeCli())
        with pytest.raises(RuntimeError_, match="Unknown recipe"):
            await runtime.up(plan_for(recipe="tgi"), name="x")

    @pytest.mark.asyncio
    async def test_the_draft_sku_spellings_still_provision(self, runtime, monkeypatch):
        """A config written against the draft spec says A100-40; the CLI only
        accepts A100."""
        cli = wire(runtime, monkeypatch, FakeCli())
        await runtime.up(plan_for(sku="A100-40"), name="x")
        assert "--gpu" in cli.argv_for("new")
        argv = cli.argv_for("new")
        assert argv[argv.index("--gpu") + 1] == "A100"


class TestUpSafetyRules:
    @pytest.mark.asyncio
    async def test_state_is_written_before_the_session_is_created(
        self, runtime, store, monkeypatch
    ):
        """Rule 1. The dangerous window is a crash between assignment and
        bookkeeping, so the record must already exist when `colab new` runs."""
        seen: list[bool] = []

        class Watching(FakeCli):
            async def __call__(self, *argv, **kwargs):
                if len(argv) > 1 and argv[1] == "new":
                    seen.append(store.load("qwen30") is not None)
                return await super().__call__(*argv, **kwargs)

        wire(runtime, monkeypatch, Watching())
        await runtime.up(plan_for(), name="qwen30")
        assert seen == [True], "the handle must be on disk before `colab new`"

    @pytest.mark.asyncio
    async def test_nothing_is_connected_to_before_the_session_exists(
        self, runtime, monkeypatch
    ):
        """Rule 3: never ssh into the void. Without the guard a failed
        assignment becomes a confusing SSH timeout."""
        # `colab new` reports success but the session never materialises --
        # exactly the case the guard exists for.
        wire(runtime, monkeypatch, FakeCli())
        monkeypatch.setattr(runtime, "sessions", _no_sessions)

        opened: list[str] = []

        async def record_tunnel(handle):
            opened.append(handle.name)

        monkeypatch.setattr(runtime, "_open_tunnel", record_tunnel)
        with pytest.raises(RuntimeError_, match="never appeared in"):
            await runtime.up(plan_for(), name="ghost")
        assert opened == [], "a tunnel must not be opened to a session that does not exist"

    @pytest.mark.asyncio
    async def test_a_failed_bootstrap_releases_the_vm(self, runtime, monkeypatch):
        """Rule 2. A half-started runtime is the expensive failure mode."""
        cli = wire(
            runtime,
            monkeypatch,
            FakeCli({("exec",): result(("colab", "exec"), rc=1, stderr="CUDA OOM")}),
        )
        with pytest.raises(RuntimeError_, match="bootstrap failed"):
            await runtime.up(plan_for(), name="qwen30")
        assert cli.argv_for("stop") is not None, "the VM was not released"

    @pytest.mark.asyncio
    async def test_a_failed_readiness_check_releases_the_vm(self, runtime, monkeypatch):
        cli = wire(runtime, monkeypatch, FakeCli())

        async def never_ready(handle, recipe):
            raise RuntimeError_("never answered")

        monkeypatch.setattr(runtime, "_await_ready", never_ready)
        with pytest.raises(RuntimeError_):
            await runtime.up(plan_for(), name="qwen30")
        assert cli.argv_for("stop") is not None

    @pytest.mark.asyncio
    async def test_a_release_that_also_fails_is_shouted_about_not_swallowed(
        self, runtime, monkeypatch, caplog
    ):
        """The worst case: bootstrap failed *and* the stop failed, so a VM is
        billing with nothing watching it. The user has to be told."""
        wire(
            runtime,
            monkeypatch,
            FakeCli(
                {
                    ("exec",): result(("colab", "exec"), rc=1, stderr="boom"),
                    ("stop",): result(("colab", "stop"), rc=1, stderr="network down"),
                }
            ),
        )
        with pytest.raises(RuntimeError_):
            await runtime.up(plan_for(), name="qwen30")
        assert "MAY STILL BE BILLING" in caplog.text

    @pytest.mark.asyncio
    async def test_deadlines_are_set_at_creation(self, runtime, monkeypatch):
        """Rule 4. An idle reaper does not stop a runtime busy in a loop, so
        the hard deadline matters as much as the idle one."""

        class Cfg:
            def get(self, key, default=None):
                return {
                    "runtimes.defaults.idle_minutes": 30,
                    "runtimes.defaults.max_lifetime_minutes": 120,
                }.get(key, default)

        rt = ColabRuntime(config=Cfg(), state_store=runtime._state)
        monkeypatch.setattr(rt, "_cli_binary", lambda: "colab")
        wire(rt, monkeypatch, FakeCli())
        handle = await rt.up(plan_for(), name="qwen30")
        assert handle.idle_deadline is not None and handle.hard_deadline is not None
        assert handle.hard_deadline > handle.idle_deadline

    @pytest.mark.asyncio
    async def test_deadlines_can_be_disabled_explicitly(self, runtime, monkeypatch):
        class Cfg:
            def get(self, key, default=None):
                return {
                    "runtimes.defaults.idle_minutes": 0,
                    "runtimes.defaults.max_lifetime_minutes": 0,
                }.get(key, default)

        rt = ColabRuntime(config=Cfg(), state_store=runtime._state)
        monkeypatch.setattr(rt, "_cli_binary", lambda: "colab")
        wire(rt, monkeypatch, FakeCli())
        handle = await rt.up(plan_for(), name="x")
        assert handle.idle_deadline is None and handle.hard_deadline is None


class TestUpSucceeds:
    @pytest.mark.asyncio
    async def test_the_handle_describes_a_reachable_endpoint(self, runtime, monkeypatch):
        wire(runtime, monkeypatch, FakeCli())
        handle = await runtime.up(plan_for(), name="qwen30")
        assert handle.phase is RuntimePhase.READY
        assert handle.base_url.startswith("http://127.0.0.1:")
        assert handle.base_url.endswith("/v1")
        assert handle.api_style == "openai"
        assert handle.served_model == "Qwen/Qwen2.5-7B-Instruct"

    @pytest.mark.asyncio
    async def test_each_runtime_gets_its_own_local_port(self, runtime, monkeypatch):
        """Two runtimes is the normal case once this works, and a fixed port
        would make the second one fail."""
        wire(runtime, monkeypatch, FakeCli())
        first = await runtime.up(plan_for(), name="a")
        second = await runtime.up(plan_for(), name="b")
        assert first.metadata["local_port"] != second.metadata["local_port"]

    @pytest.mark.asyncio
    async def test_drive_is_mounted_before_the_bootstrap_runs(self, runtime, monkeypatch):
        cli = wire(runtime, monkeypatch, FakeCli())
        await runtime.up(plan_for(), name="qwen30")
        verbs = [call[1] for call in cli.calls if len(call) > 1]
        assert verbs.index("drivemount") < verbs.index("exec")

    @pytest.mark.asyncio
    async def test_the_handle_is_persisted(self, runtime, store, monkeypatch):
        wire(runtime, monkeypatch, FakeCli())
        await runtime.up(plan_for(), name="qwen30")
        loaded = store.load("qwen30")
        assert loaded is not None and loaded.phase is RuntimePhase.READY


class TestSecrets:
    @pytest.mark.asyncio
    async def test_the_hf_token_never_appears_in_argv(self, runtime, monkeypatch):
        """argv is visible in the VM's process list and in llmcore's own debug
        logs; an env assignment through the CLI is not."""
        monkeypatch.setenv("HF_TOKEN", "hf_secret_value_123")
        cli = wire(runtime, monkeypatch, FakeCli())
        await runtime.up(plan_for(), name="qwen30")
        flat = " ".join(" ".join(call) for call in cli.calls)
        # It does travel as --env KEY=VALUE, which the CLI sets inside the
        # kernel rather than putting on the remote process line.
        assert flat.count("hf_secret_value_123") == 1
        assert "--env" in flat

    @pytest.mark.asyncio
    async def test_the_bootstrap_script_does_not_embed_the_token(self, runtime, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_secret_value_123")
        cli = wire(runtime, monkeypatch, FakeCli())
        await runtime.up(plan_for(), name="qwen30")
        scripts = [text for text in cli.stdins if text and "llmcore" in text]
        assert scripts
        assert all("hf_secret_value_123" not in script for script in scripts)


# ---------------------------------------------------------------------------
# status, orphans
# ---------------------------------------------------------------------------


class TestStatus:
    @pytest.mark.asyncio
    async def test_a_session_colab_no_longer_has_is_marked_stale(
        self, runtime, monkeypatch
    ):
        wire(runtime, monkeypatch, FakeCli())
        await runtime.up(plan_for(), name="qwen30")
        wire(runtime, monkeypatch, FakeCli())
        statuses = await runtime.status("qwen30")
        assert statuses[0].phase is RuntimePhase.STOPPED

    @pytest.mark.asyncio
    async def test_a_session_llmcore_does_not_know_is_reported_as_an_orphan(
        self, runtime, monkeypatch
    ):
        """Unmonitored spend. This is a safety feature, not a nicety."""
        wire(runtime, monkeypatch, FakeCli(preexisting=("qwen30", "other")))
        statuses = await runtime.status()
        orphans = [s for s in statuses if "orphan" in (s.error or "")]
        assert {s.name for s in orphans} == {"qwen30", "other"}
        assert "adopt" in orphans[0].error

    @pytest.mark.asyncio
    async def test_a_known_session_is_not_an_orphan(self, runtime, monkeypatch):
        wire(runtime, monkeypatch, FakeCli())
        await runtime.up(plan_for(), name="qwen30")
        statuses = await runtime.status()
        assert not any("orphan" in (s.error or "") and s.name == "qwen30" for s in statuses)

    @pytest.mark.asyncio
    async def test_status_survives_a_cli_that_will_not_list(self, runtime, monkeypatch):
        """If `colab sessions` breaks, status must still report what llmcore
        knows -- that is the list of things that might be billing."""
        wire(runtime, monkeypatch, FakeCli())
        await runtime.up(plan_for(), name="qwen30")
        wire(
            runtime,
            monkeypatch,
            FakeCli({("sessions",): result(("colab", "sessions"), rc=1, stderr="auth expired")}),
        )
        statuses = await runtime.status()
        assert [s.name for s in statuses] == ["qwen30"]


class TestAdopt:
    @pytest.mark.asyncio
    async def test_adopting_makes_an_orphan_killable(self, runtime, store, monkeypatch):
        wire(runtime, monkeypatch, FakeCli(preexisting=("other",)))
        handle = await runtime.adopt("other", name="rescued")
        assert store.load("rescued") is not None
        assert handle.phase is RuntimePhase.DEGRADED

    @pytest.mark.asyncio
    async def test_an_adopted_runtime_admits_it_is_unknown(self, runtime, monkeypatch):
        """Claiming READY would be worse than admitting the gap: llmcore has no
        idea what is running on it."""
        wire(runtime, monkeypatch, FakeCli(preexisting=("other",)))
        handle = await runtime.adopt("other", name="rescued")
        assert "does not know what it is running" in (handle.error or "")

    @pytest.mark.asyncio
    async def test_adopting_something_that_does_not_exist_names_what_does(
        self, runtime, monkeypatch
    ):
        wire(runtime, monkeypatch, FakeCli(preexisting=("qwen30",)))
        with pytest.raises(RuntimeError_, match="qwen30"):
            await runtime.adopt("nonexistent", name="x")


# ---------------------------------------------------------------------------
# down
# ---------------------------------------------------------------------------


class TestDown:
    @pytest.mark.asyncio
    async def test_down_releases_and_forgets(self, runtime, store, monkeypatch):
        cli = wire(runtime, monkeypatch, FakeCli())
        await runtime.up(plan_for(), name="qwen30")
        await runtime.down("qwen30")
        assert cli.argv_for("stop") is not None
        assert store.load("qwen30") is None

    @pytest.mark.asyncio
    async def test_down_without_release_keeps_the_record(self, runtime, store, monkeypatch):
        cli = wire(runtime, monkeypatch, FakeCli())
        await runtime.up(plan_for(), name="qwen30")
        await runtime.down("qwen30", release=False)
        assert cli.argv_for("stop") is None
        loaded = store.load("qwen30")
        assert loaded is not None and loaded.phase is RuntimePhase.DETACHED

    @pytest.mark.asyncio
    async def test_down_is_idempotent(self, runtime, monkeypatch):
        """Recovery from a half-failed start depends on this, so it must not
        raise on something already gone."""
        wire(runtime, monkeypatch, FakeCli())
        await runtime.down("never-existed")
        await runtime.down("never-existed")

    @pytest.mark.asyncio
    async def test_an_already_gone_session_is_not_an_error(self, runtime, monkeypatch, caplog):
        wire(
            runtime,
            monkeypatch,
            FakeCli({("stop",): result(("colab", "stop"), rc=1, stderr="session not found")}),
        )
        await runtime.down("qwen30")
        assert "MAY STILL BE BILLING" not in caplog.text

    @pytest.mark.asyncio
    async def test_a_stop_that_really_fails_is_loud(self, runtime, monkeypatch, caplog):
        wire(
            runtime,
            monkeypatch,
            FakeCli({("stop",): result(("colab", "stop"), rc=1, stderr="500 server error")}),
        )
        await runtime.down("qwen30")
        assert "MAY STILL BE BILLING" in caplog.text


# ---------------------------------------------------------------------------
# bake and the cache (R5)
# ---------------------------------------------------------------------------


class TestBake:
    @pytest.mark.asyncio
    async def test_bake_uses_a_cpu_runtime(self, runtime, monkeypatch):
        """The whole point: never spend GPU minutes on pip install."""
        cli = wire(runtime, monkeypatch, FakeCli())
        await runtime.bake("vllm")
        argv = cli.argv_for("new")
        assert "--gpu" not in argv

    @pytest.mark.asyncio
    async def test_bake_always_releases_the_vm(self, runtime, monkeypatch):
        cli = wire(
            runtime,
            monkeypatch,
            FakeCli({("exec",): result(("colab", "exec"), rc=1, stderr="pip failed")}),
        )
        with pytest.raises(RuntimeError_):
            await runtime.bake("vllm")
        assert cli.argv_for("stop") is not None

    @pytest.mark.asyncio
    async def test_an_unknown_recipe_is_refused_before_any_vm(self, runtime, monkeypatch):
        cli = wire(runtime, monkeypatch, FakeCli())
        with pytest.raises(RuntimeError_, match="Unknown recipe"):
            await runtime.bake("tgi")
        assert cli.calls == []

    @pytest.mark.asyncio
    async def test_cache_inventory_parses_json(self, runtime, monkeypatch):
        payload = {"env": [{"name": "vllm.tar.gz", "bytes": 10}], "models": [], "bytes": 10}
        wire(
            runtime,
            monkeypatch,
            FakeCli(
                {("exec",): result(("colab", "exec"), stdout="noise\n" + json.dumps(payload))}
            ),
        )
        assert (await runtime.cache_inventory())["bytes"] == 10


# ---------------------------------------------------------------------------
# Session parsing
# ---------------------------------------------------------------------------


#: The format `colab sessions` ACTUALLY prints, copied verbatim from a live
#: run against google-colab-cli 0.7.4. This string is the regression test: the
#: generic table scraper appeared to handle it -- it returned a row rather than
#: raising -- but with "[llmcore-e2e] gpu-t4-..." as the session name, so the
#: assignment guard never recognised its own session and released a healthy
#: T4. Being forgiving is not the same as being right.
REAL_SESSIONS_OUTPUT = (
    "[llmcore-e2e] gpu-t4-s-kkb-usw4b1-2vstf6ryp4yd8 "
    "| Hardware: T4 | Shape: Standard | Variant: GPU"
)

REAL_EMPTY_OUTPUT = "[colab] No active sessions found on server."


class TestSessionParsing:
    def test_the_real_cli_format_is_parsed(self):
        """The bug that released a live VM."""
        rows = _parse_sessions(REAL_SESSIONS_OUTPUT)
        assert len(rows) == 1
        assert rows[0]["name"] == "llmcore-e2e"
        assert rows[0]["id"] == "gpu-t4-s-kkb-usw4b1-2vstf6ryp4yd8"
        assert rows[0]["gpu"] == "T4"

    def test_the_real_empty_output_is_no_sessions(self):
        """`[colab] No active sessions...` is bracketed like a session row and
        must not be read as a session named 'colab'."""
        assert _parse_sessions(REAL_EMPTY_OUTPUT) == []

    def test_several_real_rows(self):
        text = (
            REAL_SESSIONS_OUTPUT
            + "\n[other] gpu-l4-abc | Hardware: L4 | Shape: Standard | Variant: GPU"
        )
        assert [row["name"] for row in _parse_sessions(text)] == ["llmcore-e2e", "other"]

    def test_a_table_is_parsed(self):
        rows = _parse_sessions(SESSIONS_TABLE)
        assert [row["name"] for row in rows] == ["qwen30", "other"]
        assert rows[0]["gpu"] == "L4"

    def test_json_is_preferred_if_a_future_cli_emits_it(self):
        rows = _parse_sessions('[{"name": "a", "id": "abc", "gpu": "L4"}]')
        assert rows == [{"name": "a", "id": "abc", "gpu": "L4"}]

    def test_a_json_envelope_is_unwrapped(self):
        assert _parse_sessions('{"sessions": [{"name": "a"}]}') == [{"name": "a"}]

    def test_an_unexpected_format_degrades_to_names(self):
        """This function is how llmcore finds things that cost money, so it
        must never raise on format drift."""
        rows = _parse_sessions("weird-name   something-else\nanother  thing")
        assert {row["name"] for row in rows} == {"weird-name", "another"}

    def test_empty_output_is_no_sessions(self):
        assert _parse_sessions("") == []
        assert _parse_sessions("(no sessions)\n")[0]["name"] == "(no sessions)"

    def test_headers_are_skipped(self):
        assert all(row["name"].lower() != "name" for row in _parse_sessions(SESSIONS_TABLE))


# ---------------------------------------------------------------------------
# Remote scripts
# ---------------------------------------------------------------------------


class TestRemoteScripts:
    def _script(self, **kwargs):
        defaults = {
            "repo_id": "Qwen/Qwen2.5-7B-Instruct",
            "revision": None,
            "recipe": VLLM_RECIPE,
            "drive_cache": "/content/drive/MyDrive/.llmcore-cache",
            "context_length": 16384,
            "quantization": Quantization.NONE,
            "gpu_memory_utilization": 0.9,
            "trust_remote_code": False,
            "remote_port": 8000,
        }
        defaults.update(kwargs)
        return _bootstrap_script(**defaults)

    def test_the_bootstrap_script_is_valid_python(self):
        compile(self._script(), "<bootstrap>", "exec")

    def test_the_bake_script_is_valid_python(self):
        compile(_bake_script(VLLM_RECIPE, "/cache"), "<bake>", "exec")

    def test_the_server_is_detached_from_the_kernel(self):
        """A kernel restart would otherwise kill the server while the VM
        carried on billing."""
        script = self._script()
        assert "setsid" in script and "start_new_session=True" in script

    def test_quantization_is_passed_to_vllm(self):
        assert "--quantization" in self._script(quantization=Quantization.AWQ)

    def test_an_unquantized_model_passes_no_quantization_flag(self):
        script = self._script(quantization=Quantization.NONE)
        assert 'QUANT = \'none\'' in script

    def test_trust_remote_code_is_opt_in(self):
        assert "TRUST = False" in self._script()
        assert "TRUST = True" in self._script(trust_remote_code=True)

    def test_extra_arguments_are_forwarded(self):
        assert "max_num_seqs" in self._script(extra={"max_num_seqs": 16})

    def test_a_gguf_recipe_serves_with_llamacpp(self):
        script = self._script(recipe=LLAMACPP_RECIPE)
        assert "llama_cpp.server" in script and "*.gguf" in script

    def test_cached_artefacts_are_gated_on_a_sentinel(self):
        """A miss costs time; a corrupt hit costs a debugging session on a
        billing VM."""
        script = self._script()
        assert "sentinel.is_file()" in script
        assert script.index("sentinel.write_text") > script.index("tarball)")

    def test_weight_download_is_restricted_to_what_a_server_loads(self):
        assert "allow_patterns" in self._script()

    def test_the_script_waits_on_the_vm_side_too(self):
        """So a startup failure is reported with the log that explains it,
        rather than as a local connection timeout."""
        script = self._script()
        assert "/v1/models" in script and "Traceback" in script


class TestFreePort:
    def test_it_returns_a_usable_port(self):
        assert 1024 < _free_port() < 65536

    def test_successive_calls_differ(self):
        """A fixed port would collide with a second runtime."""
        assert len({_free_port() for _ in range(5)}) > 1
