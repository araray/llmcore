# src/llmcore/runtimes/cli.py
"""``llmcore-runtimes`` — size, start, watch and stop remote GPU runtimes.

A CLI rather than only a Python API, for a specific reason: **the commands
that matter most here are the ones you need when something has gone wrong.**
`status` and `down` have to work from a shell, in a hurry, possibly in a
different process from the one that started the runtime — because the thing
you are trying to do is stop paying for a VM.

That shapes which commands exist:

``estimate``
    Free. Sizes a model and prints the arithmetic. Works while the subsystem
    is disabled, because deciding whether to spend should not require enabling
    spend.
``up``
    Spends money. Requires ``--yes``, and prints the burn rate first.
``status``
    What is running, what it is costing, and **what is running that llmcore
    did not start** — orphans are listed with the command that adopts them.
``down`` / ``down-all``
    Stop paying.
``doctor``
    What would stop a provision, and what is costing money right now. Free,
    read-only, and exits non-zero on anything that blocks — so ``doctor && up``
    does not provision into a broken state.
``spend``
    What provisioned compute has cost, per day and per unit — across every
    runtime, including released ones. ``status`` can only speak for what is
    still running.
``logs``, ``bake``, ``cache``
    The rest.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Any

logger = logging.getLogger("llmcore.runtimes")

__all__ = ["main"]


def _fmt_minutes(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 0:
        return "expired"
    minutes = seconds / 60
    return f"{minutes:.0f}m" if minutes < 120 else f"{minutes / 60:.1f}h"


def _standalone_config(args: argparse.Namespace) -> Any:
    """Load configuration without constructing an :class:`LLMCore`.

    Mirrors what ``LLMCore.create()`` does for config alone, so the commands
    below can read ``[runtimes]`` without initialising storage, embeddings,
    prompt packs or a single provider.
    """
    import pathlib as _pathlib

    from confy.loader import Config

    from llmcore.api import LLMCore as _LLMCore

    paths: list[str] = []
    user_config = _pathlib.Path("~/.config/llmcore/config.toml").expanduser()
    if user_config.exists():
        paths.append(str(user_config))
    if args.config:
        paths.append(str(args.config))
    return Config(
        defaults=_LLMCore._load_package_defaults(),
        file_paths=paths or None,
        prefix="LLMCORE",
    )


class _BareRuntimes:
    """Just enough of ``LLMCore`` for the commands that must always work.

    Carries a :class:`RuntimeManager` built straight from config, with **no
    provider manager** -- so a runtime cannot be attached through it, which is
    why ``up`` and ``adopt`` never get one of these.
    """

    def __init__(self, manager: Any) -> None:
        self.runtimes = manager

    async def close(self) -> None:
        """Nothing was opened."""
        return None


async def _llm(args: argparse.Namespace, *, degrade: bool = False) -> Any:
    """Build an ``LLMCore``, or a bare stand-in when that is impossible.

    The module docstring explains why this matters: the commands here are the
    ones you need *when something has gone wrong*, and the thing you are trying
    to do is stop paying for a VM. A full ``LLMCore`` pulls in every provider,
    the prompt pack, storage and embeddings -- so a missing vendor SDK or a
    default provider that no longer loads used to make ``status`` and ``down``
    fail, exactly when they were needed.

    Args:
        degrade: Whether this command can run without providers. ``up`` and
            ``adopt`` cannot: they attach the runtime as a provider instance,
            so for them a failure to initialise is a real failure.
    """
    from llmcore import LLMCore

    try:
        return await LLMCore.create(config_file_path=args.config)
    except Exception as exc:
        if not degrade:
            raise
        from .manager import RuntimeManager

        config = _standalone_config(args)
        print(
            f"warning: llmcore could not fully initialise ({type(exc).__name__}: "
            f"{exc}).\n"
            f"         Continuing with runtimes only -- provisioning and "
            f"attaching are unavailable,\n"
            f"         but sizing, status and teardown work.",
            file=sys.stderr,
        )
        return _BareRuntimes(RuntimeManager(config_get=config.get))


async def _cmd_estimate(args: argparse.Namespace) -> int:
    llm = await _llm(args, degrade=True)
    try:
        plan = await llm.runtimes.estimate(
            args.repo_id,
            context_length=args.context,
            quantization=args.quantization,
            backend=args.backend,
        )
    finally:
        await llm.close()

    print(f"model      {plan.spec.repo_id}")
    print(f"recipe     {plan.recipe}")
    print(f"quant      {plan.quantization}")
    print(f"context    {plan.context_length:,}")
    print(f"needs      {plan.vram_required_gb:.1f} GB")
    print(f"sku        {plan.sku} ({plan.vram_available_gb:.1f} GB usable)")
    print(f"fits       {'yes' if plan.fits else 'NO'}")
    print(f"shape      {plan.shape}"
          + (f" in {plan.region}" if plan.region else ""))
    if plan.burn_rate:
        print(f"burn rate  ~{plan.burn_rate} while assigned")
    print("\nworking:")
    for note in plan.notes:
        print(f"  - {note}")
    return 0 if plan.fits else 1


async def _cmd_up(args: argparse.Namespace) -> int:
    llm = await _llm(args)
    try:
        plan = await llm.runtimes.estimate(
            args.repo_id,
            context_length=args.context,
            quantization=args.quantization,
            backend=args.backend,
        )
        if not plan.fits:
            print(f"Refusing: {args.repo_id} does not fit {plan.sku}.", file=sys.stderr)
            for note in plan.notes[-3:]:
                print(f"  {note}", file=sys.stderr)
            return 1

        rate = plan.burn_rate
        print(
            f"About to start {plan.shape} serving {plan.spec.repo_id} at "
            f"{plan.context_length:,} context"
            + (f" in {plan.region}" if plan.region else "")
            + (f", burning ~{rate} from the moment it is assigned."
               if rate else ".")
        )
        if not args.yes:
            # Not a confirm_spend bypass: that is enforced in the manager. This
            # is the CLI refusing to make the decision on the user's behalf.
            print(
                "Pass --yes to confirm. Nothing has been provisioned.", file=sys.stderr
            )
            return 2

        handle = await llm.runtimes.up(
            args.repo_id,
            name=args.name,
            context_length=args.context,
            quantization=args.quantization,
            backend=args.backend,
            confirm_spend=True,
        )
        print(f"\n{handle.name} is serving {handle.served_model} at {handle.base_url}")
        print(f"  use it:  await llm.chat('hi', provider_name={handle.name!r})")
        print(f"  stop it: llmcore-runtimes down {handle.name}")
        if handle.hard_deadline:
            print(f"  it stops itself at {handle.hard_deadline:%Y-%m-%d %H:%M} UTC")
        return 0
    finally:
        await llm.close()


async def _cmd_status(args: argparse.Namespace) -> int:
    llm = await _llm(args, degrade=True)
    try:
        statuses = await llm.runtimes.status(args.name)
    finally:
        await llm.close()

    if not statuses:
        print("No runtimes. Nothing is billing, as far as llmcore can see.")
        return 0

    print(f"{'NAME':16} {'PHASE':10} {'SKU':8} {'MODEL':28} {'EXPIRES':8} {'SPENT':10}")
    orphans = []
    for status in statuses:
        if status.error and "orphan" in status.error:
            orphans.append(status)
            continue
        # "-" rather than a zero: a backend that does not report consumption
        # has not told us the runtime is free.
        spent = status.spend_so_far or "-"
        print(
            f"{status.name:16} {status.phase!s:10} {status.sku:8} "
            f"{status.served_model[:28]:28} {_fmt_minutes(status.expires_in_seconds):8} "
            f"{spent:10}"
        )
    for status in orphans:
        print(f"\n! {status.name}: {status.error}")
    return 0


async def _cmd_down(args: argparse.Namespace) -> int:
    llm = await _llm(args, degrade=True)
    try:
        if args.all:
            stopped = await llm.runtimes.down_all()
            print(f"Stopped: {', '.join(stopped) or '(nothing was running)'}")
        else:
            ok = await llm.runtimes.down(args.name, release=not args.detach)
            verb = "detached from" if args.detach else "stopped"
            print(f"{args.name}: {verb}" if ok else f"{args.name}: nothing to stop")
            if args.detach:
                print("  NOTE: the compute was NOT released and is still billing.")
    finally:
        await llm.close()
    return 0


async def _cmd_logs(args: argparse.Namespace) -> int:
    llm = await _llm(args, degrade=True)
    try:
        async for line in llm.runtimes.logs(
            args.name, component=args.component, tail=args.tail
        ):
            print(line)
    finally:
        await llm.close()
    return 0


async def _cmd_adopt(args: argparse.Namespace) -> int:
    llm = await _llm(args)
    try:
        handle = await llm.runtimes.adopt(
            args.external_id, name=args.name, backend=args.backend, attach=False
        )
        print(f"Adopted {args.external_id} as {handle.name!r}. It is now killable:")
        print(f"  llmcore-runtimes down {handle.name}")
    finally:
        await llm.close()
    return 0


async def _cmd_bake(args: argparse.Namespace) -> int:
    llm = await _llm(args, degrade=True)
    try:
        backend = llm.runtimes._backend_for(args.backend)
        if not hasattr(backend, "bake"):
            print(f"The {backend.name} backend has no bake step.", file=sys.stderr)
            return 1
        path = await backend.bake(args.recipe)
        print(f"Baked {args.recipe} into {path}. Future launches restore it instead of "
              f"resolving dependencies on a GPU.")
    finally:
        await llm.close()
    return 0


async def _cmd_spend(args: argparse.Namespace) -> int:
    """Per-day totals from the spend ledger, per unit.

    The figure no other command can give: ``status`` shows what one runtime
    has cost, and this shows what all of them have, including the ones that
    have already been released.
    """
    llm = await _llm(args, degrade=True)
    try:
        ledger = llm.runtimes.ledger
        days = ledger.days(limit=args.days)
    finally:
        await llm.close()

    if not days:
        print(
            "No spend recorded. The ledger fills as runtimes are released, so "
            "this stays empty until the first one is stopped."
        )
        return 0

    print(f"{'DAY':12} {'UNIT':14} {'RUNS':>5} {'SPENT':>18}")
    for day in sorted(days, reverse=True):
        for unit, total in sorted(days[day].items(), key=lambda kv: str(kv[0])):
            print(
                f"{day.isoformat():12} {unit!s:14} {total.runs:>5} "
                f"{total.rendered():>18}"
            )
    return 0


async def _cmd_doctor(args: argparse.Namespace) -> int:
    """Report what would stop a provision, and what is costing money now.

    Exit code is the worst finding: ``2`` if anything blocks, ``1`` if
    something is billing or degraded, ``0`` if clean. That ordering is so a
    script can gate on it -- ``doctor && up`` should not provision into a
    broken state.
    """
    from .diagnostics import Severity, run_diagnostics

    llm = await _llm(args, degrade=True)
    try:
        checks = await run_diagnostics(llm.runtimes, backend=args.backend)
    finally:
        await llm.close()

    marks = {
        Severity.OK: "ok  ",
        Severity.DEGRADES: "warn",
        Severity.BLOCKS: "FAIL",
        Severity.MONEY: "$$$ ",
    }
    for check in checks:
        print(f"[{marks.get(check.severity, '?')}] {check.name:26} {check.detail}")
        if check.fix and not check.ok:
            print(f"                                  -> {check.fix}")

    if any(c.severity == Severity.BLOCKS for c in checks):
        return 2
    if any(c.severity in (Severity.DEGRADES, Severity.MONEY) for c in checks):
        return 1
    return 0


async def _cmd_cache(args: argparse.Namespace) -> int:
    llm = await _llm(args, degrade=True)
    try:
        backend = llm.runtimes._backend_for(args.backend)
        if not hasattr(backend, "cache_inventory"):
            print(f"The {backend.name} backend has no cache.", file=sys.stderr)
            return 1
        inventory = await backend.cache_inventory()
    finally:
        await llm.close()

    total = inventory.get("bytes", 0) / 1024**3
    for kind in ("env", "models"):
        entries = inventory.get(kind) or []
        if entries:
            print(f"{kind}:")
            for entry in entries:
                flag = "" if entry.get("complete", True) else "  (INCOMPLETE)"
                print(f"  {entry['name']:46} {entry['bytes'] / 1024**3:6.1f} GB{flag}")
    print(f"\ntotal {total:.1f} GB in the Drive cache")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmcore-runtimes",
        description="Size, start, watch and stop remote GPU runtimes.",
        epilog=(
            "A runtime bills per minute from the moment it is assigned, whether or not "
            "anything calls it. `status` tells you what is running; `down` stops paying."
        ),
    )
    parser.add_argument("--config", default=None, help="path to a TOML config")
    parser.add_argument(
        "--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_backend(p: argparse.ArgumentParser) -> None:
        p.add_argument("--backend", default=None, help="runtime backend (default: colab)")

    estimate = sub.add_parser("estimate", help="size a model (free; provisions nothing)")
    estimate.add_argument("repo_id", help="Hugging Face repo id, e.g. Qwen/Qwen2.5-7B-Instruct")
    estimate.add_argument("--context", type=int, default=8192)
    estimate.add_argument("--quantization", default=None, help="awq, gptq, fp8, int8, ...")
    add_backend(estimate)
    estimate.set_defaults(func=_cmd_estimate)

    up = sub.add_parser("up", help="provision compute and serve a model (SPENDS MONEY)")
    up.add_argument("repo_id")
    up.add_argument("--name", required=True, help="the provider instance name to attach it as")
    up.add_argument("--context", type=int, default=8192)
    up.add_argument("--quantization", default=None)
    up.add_argument("--yes", action="store_true", help="confirm the spend; required")
    add_backend(up)
    up.set_defaults(func=_cmd_up)

    status = sub.add_parser("status", help="what is running, and what it costs")
    status.add_argument("name", nargs="?", default=None)
    status.set_defaults(func=_cmd_status)

    down = sub.add_parser("down", help="stop a runtime and stop paying for it")
    down.add_argument("name", nargs="?", default=None)
    down.add_argument("--all", action="store_true", help="stop every runtime")
    down.add_argument(
        "--detach",
        action="store_true",
        help="stop tracking it WITHOUT releasing the compute (it keeps billing)",
    )
    down.set_defaults(func=_cmd_down)

    logs = sub.add_parser("logs", help="read a runtime's logs")
    logs.add_argument("name")
    logs.add_argument("--component", default="server", choices=["server", "bootstrap"])
    logs.add_argument("--tail", type=int, default=100)
    logs.set_defaults(func=_cmd_logs)

    adopt = sub.add_parser("adopt", help="take ownership of a runtime llmcore did not start")
    adopt.add_argument("external_id")
    adopt.add_argument("--name", required=True)
    add_backend(adopt)
    adopt.set_defaults(func=_cmd_adopt)

    bake = sub.add_parser("bake", help="pre-build an environment on a CPU VM")
    bake.add_argument("--recipe", default="vllm")
    add_backend(bake)
    bake.set_defaults(func=_cmd_bake)

    spend = sub.add_parser(
        "spend", help="what provisioned compute has cost, per day and per unit"
    )
    spend.add_argument("--days", type=int, default=7)
    spend.set_defaults(func=_cmd_spend)

    doctor = sub.add_parser(
        "doctor", help="check what would stop a provision, and what is billing now"
    )
    doctor.add_argument(
        "--backend", default=None, help="only check this backend (default: all)"
    )
    doctor.set_defaults(func=_cmd_doctor)

    cache = sub.add_parser("cache", help="list what is in the Drive cache")
    add_backend(cache)
    cache.set_defaults(func=_cmd_cache)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``llmcore-runtimes``."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level, logging.WARNING),
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return asyncio.run(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
