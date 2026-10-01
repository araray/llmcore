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


async def _llm(args: argparse.Namespace) -> Any:
    from llmcore import LLMCore

    return await LLMCore.create(config_file_path=args.config)


async def _cmd_estimate(args: argparse.Namespace) -> int:
    llm = await _llm(args)
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
    if plan.estimated_cost_per_hour:
        print(f"burn rate  ~{plan.estimated_cost_per_hour:g} compute units/hour while assigned")
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

        rate = plan.estimated_cost_per_hour
        print(
            f"About to start {plan.sku} serving {plan.spec.repo_id} at "
            f"{plan.context_length:,} context"
            + (f", burning ~{rate:g} compute units/hour from the moment it is assigned."
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
    llm = await _llm(args)
    try:
        statuses = await llm.runtimes.status(args.name)
    finally:
        await llm.close()

    if not statuses:
        print("No runtimes. Nothing is billing, as far as llmcore can see.")
        return 0

    print(f"{'NAME':16} {'PHASE':10} {'SKU':8} {'MODEL':34} {'EXPIRES':8}")
    orphans = []
    for status in statuses:
        if status.error and "orphan" in status.error:
            orphans.append(status)
            continue
        print(
            f"{status.name:16} {str(status.phase):10} {status.sku:8} "
            f"{status.served_model[:34]:34} {_fmt_minutes(status.expires_in_seconds):8}"
        )
    for status in orphans:
        print(f"\n! {status.name}: {status.error}")
    return 0


async def _cmd_down(args: argparse.Namespace) -> int:
    llm = await _llm(args)
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
    llm = await _llm(args)
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
    llm = await _llm(args)
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


async def _cmd_cache(args: argparse.Namespace) -> int:
    llm = await _llm(args)
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
