# src/llmcore/runtimes/colab.py
"""The Colab backend (spec phase R3-R5).

Provisions a Colab GPU VM, serves an open-weights model on it, and tunnels the
endpoint to localhost so llmcore can attach it as an ordinary provider.

**This is the module that spends money**, and its shape is dictated by that
rather than by convenience. Four rules are enforced here rather than left to
the caller:

1. **State before compute.** The record is written when provisioning *begins*,
   not when it succeeds. The dangerous window is a crash between assignment and
   bookkeeping — the case where a VM is billing and nothing knows about it.
2. **Fail closed.** Any failure during bootstrap releases the VM before
   raising. A half-started runtime is the expensive failure mode, so there is
   exactly one error path and it ends in ``down``.
3. **Never SSH into the void.** The session must appear in ``colab sessions``
   before any connection is attempted. Without the guard, a failed assignment
   turns into a confusing SSH timeout instead of a clear "no GPU available".
4. **Bounded by default.** Idle and hard deadlines are set at creation, because
   an idle reaper does not stop a runtime that is busy in a loop.

Everything that leaves the process goes through :meth:`ColabRuntime._run`, a
single seam. That is partly for testability and partly so there is one place
where a command is logged, timed out and have its output captured — a backend
that shells out from fifteen places cannot be reasoned about.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import socket
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..exceptions import ConfigError
from .models import (
    ModelSpec,
    Plan,
    Quantization,
    RuntimeHandle,
    RuntimePhase,
    RuntimeStatus,
)
from .sizing import DEFAULT_SKU_LADDER, GPU_SKUS, Sizer, resolve_sku
from .state import RuntimeStateStore

logger = logging.getLogger(__name__)

__all__ = ["VLLM_RECIPE", "ColabRuntime", "CommandResult", "RuntimeError_"]

#: The remote port the model server listens on. Fixed, because it only has to
#: be unique inside one VM and a fixed value keeps the log messages readable.
REMOTE_PORT = 8000

#: How long to wait for the session to appear in `colab sessions` (rule 3).
ASSIGNMENT_TIMEOUT = 180.0

#: The bootstrap prints this only after the server answers on the VM. It is
#: the success contract, because `colab exec` returns 0 regardless of whether
#: the code it ran succeeded.
BOOTSTRAP_READY_MARKER = "[llmcore] READY"

#: How long to wait for the server to answer /v1/models after it is started.
#: Generous because a cold model download is included in it.
SERVE_TIMEOUT = 2700.0

#: Poll interval while waiting for readiness.
READY_POLL = 10.0


class RuntimeError_(ConfigError):
    """A runtime operation failed.

    Subclasses :class:`~llmcore.exceptions.ConfigError` so existing callers
    that catch configuration problems keep working; the distinct type exists so
    a caller can tell "your config is wrong" from "the VM did not come up".
    """


@dataclass(frozen=True, slots=True)
class CommandResult:
    """The result of one external command."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    seconds: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def brief(self, limit: int = 300) -> str:
        """One line naming the command and the most useful output.

        Prefers stderr, because that is where a CLI puts the reason.
        """
        text = " ".join((self.stderr or self.stdout or "").split())
        if len(text) > limit:
            text = text[: limit - 3] + "..."
        return f"{self.argv[0]} {' '.join(self.argv[1:3])} -> rc={self.returncode}: {text}"


#: The vLLM recipe. A dict rather than a class because a recipe is data: the
#: command to run, how to tell whether it is up, and which llmcore provider
#: speaks to it. Adding TGI or llama.cpp is a new entry, not new machinery.
VLLM_RECIPE: dict[str, Any] = {
    "name": "vllm",
    "api_style": "openai",
    "provider_type": "vllm",
    "pip": ["vllm"],
    "health_path": "/v1/models",
}

LLAMACPP_RECIPE: dict[str, Any] = {
    "name": "llamacpp",
    "api_style": "openai",
    "provider_type": "vllm",  # llama.cpp's server is OpenAI-compatible too
    "pip": ["llama-cpp-python[server]"],
    "health_path": "/v1/models",
}

RECIPES: dict[str, dict[str, Any]] = {
    "vllm": VLLM_RECIPE,
    "llamacpp": LLAMACPP_RECIPE,
}


class ColabRuntime:
    """Serves an open-weights model on a Colab GPU VM.

    Args:
        config: The ``[runtimes]`` config section as a flat accessor dict, or
            anything with ``get(key, default)``.
        state_store: Where handles are recorded. Shared with the manager, so a
            runtime started in one process is visible to another.
        sizer: Sizing engine. Built from config when omitted.
        cli_path: The Colab CLI. Discovered on ``PATH`` when the configured
            value is not executable, because llmcore is often installed in a
            venv whose ``bin`` is not the user's shell ``PATH``.
        ssh_path: The ssh binary.

    The constructor **contacts nothing**. Discovery happens on first use, so
    constructing a manager — which ``LLMCore.create()`` always does — can never
    touch a backend.
    """

    name = "colab"

    def __init__(
        self,
        *,
        config: Mapping[str, Any] | Any = None,
        state_store: RuntimeStateStore | None = None,
        sizer: Sizer | None = None,
        cli_path: str | None = None,
        ssh_path: str = "ssh",
    ) -> None:
        get = getattr(config, "get", None) if config is not None else None
        self._get = get or (lambda _k, d=None: d)
        self._state = state_store or RuntimeStateStore()
        self._cli_path = cli_path or str(self._get("runtimes.colab.cli_path", "colab") or "colab")
        self._ssh_path = ssh_path
        self._sizer = sizer or self._build_sizer()
        self._tunnels: dict[str, asyncio.subprocess.Process] = {}
        self._keepalives: dict[str, asyncio.Task[None]] = {}
        self._resolved_cli: str | None = None

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _build_sizer(self) -> Sizer:
        ladder = self._get("runtimes.colab.sku_ladder", None) or DEFAULT_SKU_LADDER
        return Sizer(
            ladder=tuple(ladder),
            headroom_fraction=float(self._get("runtimes.defaults.headroom_fraction", 0.15)),
            gpu_memory_utilization=float(
                self._get("runtimes.defaults.gpu_memory_utilization", 0.90)
            ),
            hf_token=self._hf_token(),
        )

    def _hf_token(self) -> str | None:
        env_var = str(self._get("runtimes.colab.hf_token_env_var", "HF_TOKEN") or "HF_TOKEN")
        return os.environ.get(env_var) or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    @property
    def _drive_cache(self) -> str:
        return str(
            self._get("runtimes.colab.drive_cache_dir", "/content/drive/MyDrive/.llmcore-cache")
            or "/content/drive/MyDrive/.llmcore-cache"
        )

    def _deadlines(self, now: datetime) -> tuple[datetime | None, datetime | None]:
        idle = int(self._get("runtimes.defaults.idle_minutes", 45) or 0)
        hard = int(self._get("runtimes.defaults.max_lifetime_minutes", 240) or 0)
        return (
            now + timedelta(minutes=idle) if idle > 0 else None,
            now + timedelta(minutes=hard) if hard > 0 else None,
        )

    # ------------------------------------------------------------------
    # The single external seam
    # ------------------------------------------------------------------

    async def _run(
        self,
        *argv: str,
        timeout: float = 120.0,  # noqa: ASYNC109 - mirrors subprocess semantics
        stdin: str | None = None,
        check: bool = False,
    ) -> CommandResult:
        """Run one external command.

        Args:
            argv: The command.
            timeout: Seconds before the process is killed.
            stdin: Text to feed on stdin. **The only channel used for secrets**
                — a Hugging Face token must never appear in argv, where it
                would be visible in the VM's process list and in llmcore's own
                debug logs.
            check: Raise on a non-zero exit.

        Returns:
            A :class:`CommandResult`.

        Raises:
            RuntimeError_: On a non-zero exit when *check*, or on timeout.
        """
        started = time.perf_counter()
        logger.debug("runtimes.colab: running %s", " ".join(argv[:4]))
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError_(
                f"{argv[0]!r} is not installed or not on PATH. The Colab backend needs the "
                f"official CLI: pip install google-colab-cli"
            ) from exc

        try:
            out, err = await asyncio.wait_for(
                process.communicate(stdin.encode() if stdin is not None else None),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise RuntimeError_(
                f"{argv[0]} {' '.join(argv[1:3])} timed out after {timeout:.0f}s"
            ) from None

        result = CommandResult(
            argv=tuple(argv),
            returncode=process.returncode or 0,
            stdout=out.decode(errors="replace"),
            stderr=err.decode(errors="replace"),
            seconds=time.perf_counter() - started,
        )
        if check and not result.ok:
            raise RuntimeError_(result.brief())
        return result

    async def _cli(self, *args: str, **kwargs: Any) -> CommandResult:
        """Run the Colab CLI, with the configured auth strategy.

        ``--auth`` is a *global* flag and must precede the subcommand, which is
        why it is spliced in here rather than by each caller.
        """
        auth = str(self._get("runtimes.colab.auth", "") or "").strip()
        prefix = ("--auth", auth) if auth else ()
        return await self._run(self._cli_binary(), *prefix, *args, **kwargs)

    def _cli_binary(self) -> str:
        """Locate the Colab CLI, preferring llmcore's own environment.

        Checked lazily and cached. llmcore is frequently installed in a venv
        whose ``bin`` is not on the user's shell ``PATH``, so looking next to
        the running interpreter first finds the CLI that was installed as part
        of the ``runtimes`` extra.
        """
        if self._resolved_cli:
            return self._resolved_cli
        import sys

        candidates = [
            self._cli_path,
            str(Path(sys.executable).parent / self._cli_path),
            shutil.which(self._cli_path) or "",
        ]
        for candidate in candidates:
            if candidate and (Path(candidate).is_file() or shutil.which(candidate)):
                self._resolved_cli = candidate
                return candidate
        # Fall through with the configured name so _run raises the actionable
        # "not installed" error rather than this function inventing one.
        self._resolved_cli = self._cli_path
        return self._cli_path

    # ------------------------------------------------------------------
    # estimate
    # ------------------------------------------------------------------

    async def estimate(self, spec: ModelSpec) -> Plan:
        """Size *spec*. Free, read-only, provisions nothing."""
        plan = await self._sizer.estimate(spec)
        sku = GPU_SKUS.get(plan.sku)
        if sku is not None:
            notes = [f"colab new --gpu {sku.gpu_flag}" + (" --high-mem" if sku.high_mem else "")]
            if sku.compute_units_per_hour:
                notes.append(
                    f"burn rate ~{sku.compute_units_per_hour:g} compute units/hour while assigned, "
                    f"whether or not anything calls it"
                )
            return plan.with_notes(*notes)
        return plan

    # ------------------------------------------------------------------
    # up
    # ------------------------------------------------------------------

    async def up(self, plan: Plan, *, name: str) -> RuntimeHandle:
        """Provision a VM and serve ``plan.spec``. **This spends money.**

        The sequence follows the spec's §3.4, and every step after session
        creation is inside one try/except whose handler releases the VM.

        Raises:
            RuntimeError_: On any bootstrap failure, *after* releasing.
        """
        if not plan.fits:
            raise RuntimeError_(
                f"Refusing to provision: the plan says {plan.spec.repo_id} does not fit "
                f"{plan.sku} ({plan.vram_required_gb:.1f} GB needed, "
                f"{plan.vram_available_gb:.1f} GB usable). "
                + (plan.notes[-1] if plan.notes else "")
            )

        sku = GPU_SKUS.get(resolve_sku(plan.sku) or plan.sku)
        if sku is None:
            raise RuntimeError_(
                f"Unknown SKU {plan.sku!r}. Known: {', '.join(GPU_SKUS)}."
            )
        recipe = RECIPES.get(plan.recipe)
        if recipe is None:
            raise RuntimeError_(
                f"Unknown recipe {plan.recipe!r}. Known: {', '.join(RECIPES)}."
            )

        now = datetime.now(timezone.utc)
        idle_deadline, hard_deadline = self._deadlines(now)
        local_port = _free_port()

        handle = RuntimeHandle(
            name=name,
            runtime=self.name,
            external_id="",                     # filled in once the session exists
            base_url=f"http://127.0.0.1:{local_port}/v1",
            served_model=plan.spec.repo_id,
            api_style=recipe["api_style"],
            recipe=recipe["name"],
            sku=sku.name,
            phase=RuntimePhase.STARTING,
            started_at=now,
            idle_deadline=idle_deadline,
            hard_deadline=hard_deadline,
            metadata={
                "local_port": local_port,
                "remote_port": REMOTE_PORT,
                # The plan is kept in metadata rather than as a field: the
                # handle is the thing that must stay small and readable on
                # disk, and sizing is already reproducible from the spec.
                "plan": {
                    "sku": plan.sku,
                    "recipe": plan.recipe,
                    "quantization": str(plan.quantization),
                    "context_length": plan.context_length,
                    "vram_required_gb": plan.vram_required_gb,
                    "vram_available_gb": plan.vram_available_gb,
                },
            },
        )
        # Rule 1: the record exists before anything can be assigned. If the
        # process dies in the next few seconds, `status` still shows that a
        # session by this name may be billing.
        self._state.save(handle)

        try:
            await self._create_session(name, sku)
            handle.external_id = await self._await_assignment(name)
            self._state.save(handle)

            await self._bootstrap(handle, plan, recipe)
            await self._open_tunnel(handle)
            await self._await_ready(handle, recipe)

            handle.phase = RuntimePhase.READY
            handle.touch()
            self._state.save(handle)
            self._start_keepalive(handle)
            logger.info(
                "runtimes.colab: %s is serving %s on %s at %s",
                name,
                plan.spec.repo_id,
                sku.name,
                handle.base_url,
            )
            return handle
        except BaseException as exc:
            # Rule 2. One error path, and it releases.
            logger.error(
                "runtimes.colab: bootstrap of %s failed (%s); releasing the VM", name, exc
            )
            handle.phase = RuntimePhase.FAILED
            handle.error = str(exc)[:500]
            self._state.save(handle)
            try:
                await self.down(name, release=True)
            except Exception:
                logger.exception(
                    "runtimes.colab: could not release %s after a failed start. CHECK "
                    "`colab sessions` AND STOP IT MANUALLY -- it may still be billing.",
                    name,
                )
            raise

    async def _create_session(self, name: str, sku: Any) -> None:
        argv = ["new", "-s", name, "--gpu", sku.gpu_flag]
        if sku.high_mem:
            argv.append("--high-mem")
        result = await self._cli(*argv, timeout=300.0)
        if not result.ok:
            text = (result.stderr or result.stdout).lower()
            if "quota" in text or "unavailable" in text or "not available" in text:
                raise RuntimeError_(
                    f"Colab would not give us a {sku.gpu_flag}: {result.brief()}. Try a cheaper "
                    f"SKU, or set runtimes.colab.sku_ladder to what this account can get."
                )
            raise RuntimeError_(f"colab new failed: {result.brief()}")

    async def _await_assignment(self, name: str) -> str:
        """Rule 3: wait for the session to appear before connecting to it.

        Returns the session's external id.
        """
        deadline = time.monotonic() + ASSIGNMENT_TIMEOUT
        last = ""
        while time.monotonic() < deadline:
            sessions = await self.sessions()
            for session in sessions:
                if session.get("name") == name:
                    external = str(session.get("id") or session.get("name") or name)
                    logger.info("runtimes.colab: %s is assigned (%s)", name, external)
                    return external
            last = ", ".join(str(s.get("name")) for s in sessions) or "(none)"
            await asyncio.sleep(5.0)
        raise RuntimeError_(
            f"Session {name!r} never appeared in `colab sessions` within "
            f"{ASSIGNMENT_TIMEOUT:.0f}s. Sessions seen: {last}. Nothing was connected to, so "
            f"nothing should be billing -- but check `colab sessions` to be sure."
        )

    async def sessions(self) -> list[dict[str, Any]]:
        """Parse ``colab sessions`` into dicts.

        The CLI prints a human table, so this is best-effort by design: a
        parser that raises on an unexpected column would turn a cosmetic CLI
        change into an inability to *find and kill* runtimes, which is exactly
        when parsing has to keep working.
        """
        result = await self._cli("sessions", timeout=60.0)
        if not result.ok:
            logger.warning("runtimes.colab: `colab sessions` failed: %s", result.brief())
            return []
        return _parse_sessions(result.stdout)

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    async def _bootstrap(self, handle: RuntimeHandle, plan: Plan, recipe: dict[str, Any]) -> None:
        """Mount Drive, restore or build the env, fetch weights, start serving.

        Runs as one generated script rather than a sequence of ``colab exec``
        calls. Two reasons: each ``exec`` is a round trip through the kernel
        protocol and the slow steps here take minutes, and a script is a single
        artifact that can be read, diffed and reproduced by hand when something
        goes wrong on the VM.
        """
        sku = GPU_SKUS.get(resolve_sku(plan.sku) or plan.sku)
        script = _bootstrap_script(
            repo_id=plan.spec.repo_id,
            revision=plan.spec.revision,
            recipe=recipe,
            drive_cache=self._drive_cache,
            context_length=plan.context_length,
            quantization=plan.quantization,
            dtype="half" if (sku and sku.needs_fp16) else None,
            gpu_memory_utilization=float(
                self._get("runtimes.defaults.gpu_memory_utilization", 0.90)
            ),
            trust_remote_code=plan.spec.trust_remote_code,
            remote_port=REMOTE_PORT,
            extra=plan.spec.extra,
        )
        handle.metadata["bootstrap_bytes"] = len(script)

        await self._cli("drivemount", "-s", handle.name, timeout=300.0)

        # The token goes on stdin, never in argv: argv is visible in the VM's
        # process list and in llmcore's own debug logging.
        token = self._hf_token()
        env_args = ["--env", f"LLMCORE_DRIVE_CACHE={self._drive_cache}"]
        if token:
            env_args += ["--env", f"HF_TOKEN={token}"]

        result = await self._cli(
            "exec",
            "-s",
            handle.name,
            "--timeout",
            str(SERVE_TIMEOUT),
            *env_args,
            stdin=script,
            timeout=SERVE_TIMEOUT + 120.0,
        )
        log = (result.stdout or "") + (result.stderr or "")
        handle.metadata["bootstrap_log"] = log[-8000:]

        # `colab exec` reports rc=0 even when the code it ran raised, so the
        # exit status says nothing about whether the bootstrap worked. Observed
        # directly: a SystemExit inside the script came back as a success, and
        # the caller went on to open a tunnel to a server that had never
        # started and wait 45 minutes for it.
        #
        # So the contract is the marker, not the exit code: the script prints
        # READY only after the server answers on the VM.
        if not result.ok or BOOTSTRAP_READY_MARKER not in log:
            raise RuntimeError_(
                f"bootstrap did not reach the ready marker on the VM. "
                f"Last output: ...{_last_meaningful(log)}"
            )

    async def _open_tunnel(self, handle: RuntimeHandle) -> None:
        """Forward the remote server port to localhost over the Colab SSH proxy."""
        local_port = int(handle.metadata["local_port"])
        proxy = f"{self._cli_binary()} ssh --proxy-mode -s {handle.name}"
        argv = [
            self._ssh_path,
            "-N",
            "-o", f"ProxyCommand={proxy}",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30",
            "-L", f"127.0.0.1:{local_port}:127.0.0.1:{REMOTE_PORT}",
            "root@colab",
        ]
        logger.info(
            "runtimes.colab: forwarding 127.0.0.1:%d -> the VM's :%d", local_port, REMOTE_PORT
        )
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError_(
                "ssh is not installed. The Colab backend tunnels the model endpoint over SSH."
            ) from exc
        self._tunnels[handle.name] = process
        handle.metadata["tunnel_pid"] = process.pid
        await asyncio.sleep(2.0)
        if process.returncode is not None:
            _, err = await process.communicate()
            raise RuntimeError_(
                f"the SSH tunnel exited immediately (rc={process.returncode}): "
                f"{err.decode(errors='replace')[:300]}"
            )

    async def _await_ready(self, handle: RuntimeHandle, recipe: dict[str, Any]) -> None:
        """Poll the tunnelled endpoint until the server answers."""
        import httpx

        url = handle.base_url.rstrip("/").removesuffix("/v1") + recipe["health_path"]
        deadline = time.monotonic() + SERVE_TIMEOUT
        last = ""
        async with httpx.AsyncClient(timeout=10.0) as client:
            while time.monotonic() < deadline:
                try:
                    response = await client.get(url)
                    if response.status_code == 200:
                        logger.info("runtimes.colab: %s answered %s", handle.name, url)
                        return
                    last = f"HTTP {response.status_code}"
                except Exception as exc:
                    last = type(exc).__name__
                await asyncio.sleep(READY_POLL)
        raise RuntimeError_(
            f"the model server never answered {url} within {SERVE_TIMEOUT / 60:.0f} minutes "
            f"(last: {last}). Check `llm.runtimes.logs({handle.name!r})`."
        )

    # ------------------------------------------------------------------
    # Keepalive
    # ------------------------------------------------------------------

    def _start_keepalive(self, handle: RuntimeHandle) -> None:
        """Hold the kernel active so Colab does not reclaim the VM.

        A background task rather than a remote daemon, so that killing the
        llmcore process stops the keepalive. That is the safer default: a
        keepalive that outlives its owner is a leaked VM with a heartbeat.
        """
        interval = float(self._get("runtimes.colab.keepalive_seconds", 60.0) or 60.0)
        if interval <= 0:
            return

        async def loop() -> None:
            while True:
                await asyncio.sleep(interval)
                try:
                    await self._cli(
                        "exec", "-s", handle.name, "--timeout", "20", stdin="1\n", timeout=40.0
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.debug("runtimes.colab: keepalive for %s failed: %s", handle.name, exc)

        task = asyncio.create_task(loop(), name=f"colab-keepalive-{handle.name}")
        self._keepalives[handle.name] = task

    def _stop_keepalive(self, name: str) -> None:
        task = self._keepalives.pop(name, None)
        if task is not None and not task.done():
            task.cancel()

    # ------------------------------------------------------------------
    # status / logs / down / adopt
    # ------------------------------------------------------------------

    async def status(self, name: str | None = None) -> list[RuntimeStatus]:
        """Report on runtimes, reconciled against what Colab says exists.

        The reconciliation is a safety feature, not a nicety. Two
        disagreements matter: a session llmcore recorded that Colab no longer
        has (stale state), and a session Colab has that llmcore does not
        (**an orphan** — unmonitored spend).
        """
        handles = [self._state.load(name)] if name else self._state.load_all()
        handles = [handle for handle in handles if handle is not None]

        live: set[str] = set()
        try:
            live = {str(session.get("name")) for session in await self.sessions()}
        except Exception as exc:
            logger.debug("runtimes.colab: could not list sessions: %s", exc)
            live = set()

        out: list[RuntimeStatus] = []
        for handle in handles:
            if handle.name not in live and handle.phase.is_billing:
                handle.phase = RuntimePhase.STOPPED
                handle.error = "no matching Colab session; marked stale"
                self._state.save(handle)
            expired = handle.expired_reason()
            if expired and handle.phase.is_billing:
                handle.error = expired
            out.append(RuntimeStatus.from_handle(handle))

        if name is None:
            known = {handle.name for handle in handles}
            for orphan in sorted(live - known):
                out.append(
                    RuntimeStatus(
                        name=orphan,
                        runtime=self.name,
                        # DEGRADED, not a notional "unknown": it exists, it is
                        # billing, and llmcore cannot serve through it. That is
                        # what degraded means.
                        phase=RuntimePhase.DEGRADED,
                        error=(
                            f"orphan: a Colab session named {orphan!r} is running that llmcore "
                            f"has no record of. It may be billing. Adopt it to make it killable: "
                            f"llm.runtimes.adopt({orphan!r}, name={orphan!r})"
                        ),
                    )
                )
        return out

    async def logs(
        self, name: str, *, component: str = "server", tail: int = 100
    ) -> AsyncIterator[str]:
        """Stream log lines for *name*.

        ``component`` selects which log: ``server`` is the model server's own
        output on the VM, ``bootstrap`` is what the bootstrap script printed
        (kept locally, so it survives the VM).
        """
        handle = self._state.load(name)
        if handle is None:
            yield f"no runtime named {name!r}"
            return

        if component == "bootstrap":
            for line in (handle.metadata.get("bootstrap_log") or "").splitlines()[-tail:]:
                yield line
            return

        result = await self._cli(
            "exec",
            "-s",
            name,
            "--timeout",
            "60",
            stdin="print(open('/content/llmcore-server.log').read()[-20000:])\n",
            timeout=90.0,
        )
        if not result.ok:
            yield f"could not read the server log: {result.brief()}"
            return
        for line in result.stdout.splitlines()[-tail:]:
            yield line

    async def down(self, name: str, *, release: bool = True) -> None:
        """Stop *name*. Idempotent, because recovery depends on it.

        Tearing down something already gone must not raise: that is how
        recovery from a half-failed start works, and a `down` that throws
        leaves the caller unable to clean up.
        """
        self._stop_keepalive(name)

        tunnel = self._tunnels.pop(name, None)
        if tunnel is not None and tunnel.returncode is None:
            tunnel.terminate()
            try:
                await asyncio.wait_for(tunnel.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                tunnel.kill()

        handle = self._state.load(name)
        if release:
            result = await self._cli("stop", "-s", name, timeout=120.0)
            if not result.ok:
                text = (result.stderr or result.stdout).lower()
                if "not found" in text or "no such" in text or "unknown session" in text:
                    logger.info("runtimes.colab: %s was already gone", name)
                else:
                    # Loud, because the consequence is money. Still not raised:
                    # the caller is cleaning up and needs the rest to happen.
                    logger.error(
                        "runtimes.colab: `colab stop -s %s` failed: %s. THE VM MAY STILL BE "
                        "BILLING -- check `colab sessions`.",
                        name,
                        result.brief(),
                    )

        if handle is not None:
            handle.phase = RuntimePhase.STOPPED if release else RuntimePhase.DETACHED
            self._state.save(handle)
            if release:
                self._state.delete(name)

    async def adopt(self, external_id: str, *, name: str) -> RuntimeHandle:
        """Take ownership of a session llmcore did not start.

        The recovery path for an orphan: something is billing and llmcore has
        no record of it, so adopting it is what makes it killable. The handle
        is deliberately marked ``DEGRADED`` rather than ``READY`` — llmcore has
        no idea what is running on it, and claiming otherwise would be worse
        than admitting the gap.
        """
        sessions = await self.sessions()
        match = next(
            (s for s in sessions if external_id in (str(s.get("id")), str(s.get("name")))), None
        )
        if match is None:
            raise RuntimeError_(
                f"No Colab session matching {external_id!r}. Sessions: "
                + (", ".join(str(s.get("name")) for s in sessions) or "(none)")
            )

        now = datetime.now(timezone.utc)
        idle_deadline, hard_deadline = self._deadlines(now)
        handle = RuntimeHandle(
            name=name,
            runtime=self.name,
            external_id=str(match.get("id") or match.get("name") or external_id),
            base_url="",
            served_model="",
            api_style="openai",
            recipe="unknown",
            sku=str(match.get("gpu") or "unknown"),
            phase=RuntimePhase.DEGRADED,
            started_at=now,
            idle_deadline=idle_deadline,
            hard_deadline=hard_deadline,
            error=(
                "adopted; llmcore did not start this session and does not know what it is "
                "running. It is now killable with down()."
            ),
        )
        self._state.save(handle)
        logger.warning(
            "runtimes.colab: adopted session %s as %r. It is now tracked and killable, but "
            "llmcore cannot serve through it -- nothing is known about what it runs.",
            external_id,
            name,
        )
        return handle

    # ------------------------------------------------------------------
    # bake (R5)
    # ------------------------------------------------------------------

    async def bake(self, recipe: str = "vllm", *, session: str = "llmcore-bake") -> str:
        """Pre-build an environment tarball on a **CPU** VM.

        Worth its own command because the alternative is spending GPU minutes
        on ``pip install``. A CPU runtime is created (no ``--gpu``), the
        environment is built and tarred into the Drive cache, and the VM is
        released — so the next GPU launch restores a tar instead of resolving
        dependencies.

        Returns:
            The Drive path of the tarball.
        """
        spec = RECIPES.get(recipe)
        if spec is None:
            raise RuntimeError_(f"Unknown recipe {recipe!r}. Known: {', '.join(RECIPES)}.")

        result = await self._cli("new", "-s", session, timeout=300.0)
        if not result.ok:
            raise RuntimeError_(f"could not create a CPU session to bake in: {result.brief()}")
        try:
            await self._await_assignment(session)
            await self._cli("drivemount", "-s", session, timeout=300.0)
            script = _bake_script(spec, self._drive_cache)
            baked = await self._cli(
                "exec", "-s", session, "--timeout", "3600", stdin=script, timeout=3700.0
            )
            if not baked.ok:
                raise RuntimeError_(f"bake failed: {baked.brief(600)}")
            path = f"{self._drive_cache}/env/{spec['name']}.tar.gz"
            logger.info("runtimes.colab: baked %s into %s", spec["name"], path)
            return path
        finally:
            # Always release: a bake VM left running is the same leak as any
            # other, and it is not even serving anything.
            await self._cli("stop", "-s", session, timeout=120.0)

    async def cache_inventory(self, *, session: str = "llmcore-cache") -> dict[str, Any]:
        """List what is in the Drive cache, on a CPU VM.

        Needs a VM because the cache lives in the user's Drive, not locally.
        Uses a CPU runtime, so inspecting the cache never costs GPU minutes.
        """
        result = await self._cli("new", "-s", session, timeout=300.0)
        if not result.ok:
            raise RuntimeError_(f"could not create a CPU session: {result.brief()}")
        try:
            await self._await_assignment(session)
            await self._cli("drivemount", "-s", session, timeout=300.0)
            listed = await self._cli(
                "exec", "-s", session, "--timeout", "120",
                stdin=_INVENTORY_SCRIPT.format(cache=self._drive_cache),
                timeout=180.0,
            )
            if not listed.ok:
                raise RuntimeError_(f"cache inventory failed: {listed.brief()}")
            return _parse_inventory(listed.stdout)
        finally:
            await self._cli("stop", "-s", session, timeout=120.0)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_sessions(text: str) -> list[dict[str, Any]]:
    """Parse ``colab sessions`` output, tolerating format drift.

    Tries JSON first in case a future CLI version offers it, then falls back to
    scraping a table. Deliberately forgiving: this function is how llmcore
    *finds runtimes that are costing money*, so it must degrade to "I found
    these names" rather than fail.
    """
    stripped = text.strip()
    if stripped.startswith(("[", "{")):
        try:
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                payload = payload.get("sessions") or []
            return [row for row in payload if isinstance(row, dict)]
        except ValueError:
            pass

    rows: list[dict[str, Any]] = []
    for line in stripped.splitlines():
        line = line.strip()

        # The format the CLI actually prints (verified against
        # google-colab-cli 0.7.4):
        #
        #   [llmcore-e2e] gpu-t4-s-kkb-usw4b1-2vstf6ryp4yd8 | Hardware: T4 | ...
        #
        # This is matched first and explicitly. The generic table scraper below
        # *appeared* to work on it -- it produced a row rather than raising --
        # but with the whole `[name] id` chunk as the name, so the assignment
        # guard never matched its own session and released a healthy VM. A
        # parser being forgiving is not the same as a parser being right.
        bracketed = re.match(r"^\[([^\]]+)\]\s*(\S+)?(.*)$", line)
        if bracketed:
            name = bracketed.group(1).strip()
            if name.lower() == "colab":
                continue  # "[colab] No active sessions found on server."
            row: dict[str, Any] = {"name": name}
            if bracketed.group(2):
                row["id"] = bracketed.group(2).strip()
            for field in bracketed.group(3).split("|"):
                key, _, value = field.partition(":")
                key, value = key.strip().lower(), value.strip()
                if not value:
                    continue
                if key == "hardware":
                    row["gpu"] = value.upper()
                elif key:
                    row[key] = value
            rows.append(row)
            continue

        line = line.strip("│|").strip()
        if not line or set(line) <= set("\u2500\u250c\u252c\u2510\u251c\u253c\u2524\u2514\u2534\u2518\u2502-+=| "):
            continue
        cells = [cell.strip() for cell in re.split(r"\s*[│|]\s*|\s{2,}", line) if cell.strip()]
        if not cells:
            continue
        lowered = cells[0].lower()
        if lowered in ("name", "session", "session name", "id"):
            continue   # header
        row: dict[str, Any] = {"name": cells[0]}
        for cell in cells[1:]:
            if re.fullmatch(r"(?i)(t4|l4|g4|a100|h100|v5e1|v6e1|cpu)", cell):
                row["gpu"] = cell.upper()
            elif re.fullmatch(r"[0-9a-f-]{8,}", cell):
                row["id"] = cell
            elif "status" not in row:
                row["status"] = cell
        rows.append(row)
    return rows


def _parse_inventory(text: str) -> dict[str, Any]:
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return {"env": [], "models": [], "bytes": 0}


def _last_meaningful(log: str, limit: int = 700) -> str:
    """The tail of a bootstrap log, preferring the part that explains a failure.

    A traceback's useful line is its last one, and a VM log is mostly progress
    chatter, so the naive tail is usually right -- but when the script named a
    specific failure, lead with that instead.
    """
    lines = [line for line in log.splitlines() if line.strip()]
    for marker in ("SystemExit", "Error", "Traceback"):
        hits = [i for i, line in enumerate(lines) if marker in line]
        if hits:
            return " | ".join(lines[hits[0] :][-6:])[-limit:]
    return " | ".join(lines[-6:])[-limit:]


def _free_port() -> int:
    """Ask the OS for a free local port.

    Binding to port 0 and reading the assignment is the only way to avoid a
    race with everything else on the machine. A fixed port would collide with a
    second runtime, and two runtimes is the normal case once this works.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# ---------------------------------------------------------------------------
# Remote-side scripts
# ---------------------------------------------------------------------------
#
# These run *on the VM*, inside Colab's Python kernel. Three constraints shape
# them, and all three are unusual:
#
# * **Every second is billed.** So the order is: cheapest check first, and
#   anything that can be cached in Drive is cached. A cold start resolves
#   dependencies and downloads weights; a warm start restores a tar and copies
#   from Drive, which is the difference between ten minutes and under one.
# * **The kernel can churn.** The model server is started with `setsid` so it
#   survives the kernel that launched it. If it were a child of the kernel, a
#   kernel restart would silently kill the server while the VM kept billing.
# * **Secrets arrive in the environment, not in the source.** The token is set
#   by `colab exec --env`, which does not put it in the VM's process list.

_BOOTSTRAP_TEMPLATE = r'''
import json, os, shlex, subprocess, sys, time
from pathlib import Path

CACHE = Path(os.environ.get("LLMCORE_DRIVE_CACHE", {drive_cache!r}))
REPO = {repo_id!r}
REVISION = {revision!r}
RECIPE = {recipe_name!r}
PIP = {pip!r}
PORT = {remote_port}
CTX = {context_length}
QUANT = {quantization!r}
GPU_UTIL = {gpu_memory_utilization}
TRUST = {trust_remote_code}
DTYPE = {dtype!r}
EXTRA = {extra!r}
LOG = Path("/content/llmcore-server.log")

def say(*a):
    print("[llmcore]", *a, flush=True)

def run(cmd, **kw):
    say("$", cmd if isinstance(cmd, str) else " ".join(cmd))
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=False,
                          capture_output=True, text=True, **kw)

# --- 1. Drive cache layout -------------------------------------------------
# A miss is normal and costs time; a *corrupt* hit costs a debugging session on
# a billing VM, so every cached artefact is gated on a sentinel written only
# after the artefact is complete.
env_dir = CACHE / "env"
model_dir = CACHE / "models" / REPO.replace("/", "--")
try:
    env_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    drive_ok = True
except Exception as exc:
    say("Drive cache unavailable (%s); continuing without it" % exc)
    drive_ok = False

# --- 2. Environment --------------------------------------------------------
tarball = env_dir / (RECIPE + ".tar.gz")
sentinel = env_dir / (RECIPE + ".ok")

# Where pip ACTUALLY installs for this interpreter.
#
# Two earlier attempts got this wrong, and both failed silently. Extracting to
# /content/llmcore-env meant a restore "succeeded" and then nothing could be
# imported, because that path was never on sys.path. Using
# site.getsitepackages()[-1] resolved to /usr/lib/python3/dist-packages on a
# real Colab VM while pip was installing into
# /usr/local/lib/python3.13/dist-packages -- so the cached tar would have held
# the wrong tree, and a "cache hit" would have produced an environment with no
# recipe in it.
#
# sysconfig's purelib is the path pip itself resolves, which makes it the one
# answer that cannot disagree with the installer.
import sysconfig
target = Path(sysconfig.get_paths()["purelib"])
say("site-packages (pip purelib):", target)

restored = False
if drive_ok and tarball.is_file() and sentinel.is_file():
    say("restoring the %s environment from the Drive cache" % RECIPE)
    result = run(["tar", "-xzf", str(tarball), "-C", str(target)])
    restored = result.returncode == 0
    if restored:
        import importlib
        importlib.invalidate_caches()
    else:
        say("restore failed; falling back to pip:", result.stderr[-400:])

if not restored:
    say("installing %s (cold start)" % ", ".join(PIP))
    # Local disk, never Drive. Pointing PIP_CACHE_DIR at the Drive mount makes
    # every wheel download write through FUSE to Google Drive, which on a cold
    # cache is gigabytes of network round trips -- turning a slow step into an
    # unbounded one, on a VM that bills by the minute. The artefact worth
    # persisting is the finished environment tarball, written once at the end;
    # the pip cache is scratch.
    os.environ.setdefault("PIP_CACHE_DIR", "/content/pipcache")
    result = run([sys.executable, "-m", "pip", "install", "--quiet", *PIP])
    if result.returncode != 0:
        say("pip install failed:", result.stdout[-2000:], result.stderr[-2000:])
        raise SystemExit("pip install failed")

# --- 2b. Reconcile the torch ecosystem -------------------------------------
# Installing a serving stack upgrades torch, and the host image's *other*
# torch packages stay on the CUDA build they shipped with. transformers
# imports torchaudio unconditionally, and torchaudio refuses to load against a
# different CUDA than torch -- so the server dies on import with a message
# about CUDA versions that has nothing to do with the model.
#
# Observed on Colab: torch 2.13.0+cu130 against a preinstalled
# torchaudio 2.11.0+cu128.
#
# Removing the mismatched companion is the right fix here rather than pinning:
# nothing in a text-serving recipe needs audio or vision, and chasing a
# matching build costs a second multi-gigabyte download on a billing VM.
try:
    import torch as _torch
    want = getattr(_torch.version, "cuda", None)
    if want:
        want_tag = "cu" + want.replace(".", "")
        result = run([sys.executable, "-m", "pip", "list", "--format=freeze"])
        for line in result.stdout.splitlines():
            name, _, version = line.partition("==")
            if name.strip() in ("torchaudio", "torchvision") and "+cu" in version:
                have_tag = version.split("+", 1)[1].strip()
                if have_tag != want_tag:
                    say("removing %s %s: built for %s, torch is %s"
                        % (name, version.strip(), have_tag, want_tag))
                    run([sys.executable, "-m", "pip", "uninstall", "-y", name.strip()])
except Exception as exc:
    say("torch ecosystem check skipped (%s)" % exc)

# --- 3. Weights ------------------------------------------------------------
# allow_patterns keeps this to what a server actually loads: pulling the whole
# repo often means downloading a second copy of the weights in another format.
local_weights = None
weights_sentinel = model_dir / ".complete"
if drive_ok and weights_sentinel.is_file():
    say("weights already in the Drive cache")
    local_weights = str(model_dir)
else:
    say("downloading weights for", REPO)
    from huggingface_hub import snapshot_download
    patterns = ["*.safetensors", "*.json", "*.model", "*.txt", "tokenizer*"]
    if RECIPE == "llamacpp":
        patterns = ["*.gguf", "*.json", "tokenizer*"]
    try:
        path = snapshot_download(
            REPO, revision=REVISION, allow_patterns=patterns,
            token=os.environ.get("HF_TOKEN") or None,
            local_dir=str(model_dir) if drive_ok else None,
        )
        local_weights = path
        if drive_ok:
            weights_sentinel.write_text("ok")
    except Exception as exc:
        say("weight download failed:", repr(exc))
        raise

# --- 4. Serve --------------------------------------------------------------
# setsid, so the server outlives the kernel that started it. Without this a
# kernel restart kills the server while the VM carries on billing.
if RECIPE == "llamacpp":
    gguf = sorted(Path(local_weights).rglob("*.gguf"))
    if not gguf:
        raise SystemExit("no .gguf file found in " + str(local_weights))
    argv = [sys.executable, "-m", "llama_cpp.server",
            "--model", str(gguf[0]), "--host", "127.0.0.1", "--port", str(PORT),
            "--n_ctx", str(CTX), "--n_gpu_layers", "-1"]
else:
    argv = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", local_weights, "--served-model-name", REPO,
            "--host", "127.0.0.1", "--port", str(PORT),
            "--max-model-len", str(CTX),
            "--gpu-memory-utilization", str(GPU_UTIL)]
    if QUANT and QUANT not in ("none", "gguf"):
        argv += ["--quantization", QUANT]
    if DTYPE:
        # Pre-Ampere cards have no bfloat16 and vLLM refuses rather than
        # downcasting, so a bf16 checkpoint on a T4 needs this or it will not
        # start at all -- after the VM is already billing.
        argv += ["--dtype", DTYPE]
    if TRUST:
        argv += ["--trust-remote-code"]
for key, value in (EXTRA or {{}}).items():
    argv += ["--" + str(key).replace("_", "-"), str(value)]

say("starting:", " ".join(shlex.quote(a) for a in argv))
LOG.write_text("")
subprocess.Popen(
    ["setsid", "nohup", *argv],
    stdout=open(LOG, "ab"), stderr=subprocess.STDOUT,
    stdin=subprocess.DEVNULL, start_new_session=True,
)

# --- 5. Wait for the server, on the VM side --------------------------------
# Waiting here as well as locally gives a *useful* error: the log is on this
# machine, so a startup failure is reported with its reason instead of as a
# local connection timeout.
import urllib.request
deadline = time.time() + 2400
url = "http://127.0.0.1:%d/v1/models" % PORT
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status == 200:
                say("server is up on :%d" % PORT)
                break
    except Exception:
        pass
    if LOG.is_file():
        tail = LOG.read_text(errors="replace")[-600:]
        if "Error" in tail or "Traceback" in tail:
            say("server log shows an error:\n" + tail)
            raise SystemExit("the model server failed to start")
    time.sleep(10)
else:
    say("server did not come up; last log:\n" + LOG.read_text(errors="replace")[-2000:])
    raise SystemExit("timed out waiting for the model server")

# --- 6. Cache the environment for next time --------------------------------
if drive_ok and not restored:
    # Tar the directory pip actually installed into. Hard-coding
    # /usr/lib/python3/dist-packages produced a tarball that did not contain
    # the recipe at all on Colab, which installs into /usr/local.
    say("tarring", target, "into the Drive cache for next time")
    result = run(["tar", "-czf", str(tarball) + ".tmp", "-C", str(target), "."])
    if result.returncode == 0:
        Path(str(tarball) + ".tmp").replace(tarball)
        sentinel.write_text("ok")   # written last: a sentinel means complete
    else:
        say("env tar failed (not fatal):", result.stderr[-300:])

say("READY")
'''


def _bootstrap_script(
    *,
    repo_id: str,
    revision: str | None,
    recipe: dict[str, Any],
    drive_cache: str,
    context_length: int,
    quantization: Quantization,
    gpu_memory_utilization: float,
    trust_remote_code: bool,
    remote_port: int,
    dtype: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Render the VM-side bootstrap script."""
    return _BOOTSTRAP_TEMPLATE.format(
        dtype=dtype,
        drive_cache=drive_cache,
        repo_id=repo_id,
        revision=revision,
        recipe_name=recipe["name"],
        pip=list(recipe["pip"]),
        remote_port=remote_port,
        context_length=int(context_length),
        quantization=str(quantization),
        gpu_memory_utilization=float(gpu_memory_utilization),
        trust_remote_code=bool(trust_remote_code),
        extra=dict(extra or {}),
    )


_BAKE_TEMPLATE = r'''
import os, subprocess, sys
from pathlib import Path

CACHE = Path({drive_cache!r})
RECIPE = {recipe_name!r}
PIP = {pip!r}
env_dir = CACHE / "env"
env_dir.mkdir(parents=True, exist_ok=True)
tarball = env_dir / (RECIPE + ".tar.gz")
sentinel = env_dir / (RECIPE + ".ok")

print("[llmcore] baking", RECIPE, "on a CPU runtime", flush=True)
# Local, for the same reason as the bootstrap: Drive is for the finished tar.
os.environ.setdefault("PIP_CACHE_DIR", "/content/pipcache")
result = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *PIP],
                        capture_output=True, text=True)
if result.returncode != 0:
    print(result.stdout[-2000:], result.stderr[-2000:], flush=True)
    raise SystemExit("pip install failed")

import sysconfig
target = sysconfig.get_paths()["purelib"]
print("[llmcore] tarring", target, "into", tarball, flush=True)
tmp = str(tarball) + ".tmp"
result = subprocess.run(["tar", "-czf", tmp, "-C", "/usr/lib/python3/dist-packages", "."],
                        capture_output=True, text=True)
if result.returncode != 0:
    raise SystemExit("tar failed: " + result.stderr[-500:])
Path(tmp).replace(tarball)
sentinel.write_text("ok")
print("[llmcore] baked", tarball, flush=True)
'''


def _bake_script(recipe: dict[str, Any], drive_cache: str) -> str:
    """Render the CPU-side bake script."""
    return _BAKE_TEMPLATE.format(
        drive_cache=drive_cache, recipe_name=recipe["name"], pip=list(recipe["pip"])
    )


_INVENTORY_SCRIPT = r'''
import json
from pathlib import Path
cache = Path({cache!r})
out = {{"env": [], "models": [], "bytes": 0}}
for kind, folder in (("env", cache / "env"), ("models", cache / "models")):
    if not folder.is_dir():
        continue
    for entry in sorted(folder.iterdir()):
        size = 0
        if entry.is_file():
            size = entry.stat().st_size
        elif entry.is_dir():
            size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
        out[kind].append({{"name": entry.name, "bytes": size,
                          "complete": (entry / ".complete").is_file() if entry.is_dir() else True}})
        out["bytes"] += size
print(json.dumps(out))
'''
