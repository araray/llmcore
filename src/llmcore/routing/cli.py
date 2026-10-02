# src/llmcore/routing/cli.py
"""``llmcore-routing`` — explain, inspect and *measure* routing.

The last of those is the one that matters and the one most routers skip.
``llmcore-routing eval`` runs your labelled prompts through one or more
classifiers and reports, separately, how often each was **too cheap** (which
produces bad answers) and how often it was **too expensive** (which only costs
money). A single accuracy figure hides that distinction, and the distinction is
the whole decision.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("llmcore.routing")

__all__ = ["main"]


async def _llm(args: argparse.Namespace) -> Any:
    from llmcore import LLMCore

    return await LLMCore.create(config_file_path=args.config)


async def _cmd_why(args: argparse.Namespace) -> int:
    llm = await _llm(args)
    try:
        print(await llm.routing.why(args.prompt, lane=args.lane, pool=args.pool))
    finally:
        await llm.close()
    return 0


async def _cmd_health(args: argparse.Namespace) -> int:
    llm = await _llm(args)
    try:
        health = llm.routing.health()
    finally:
        await llm.close()
    if not health:
        print("No target health recorded yet. Nothing has been called through routing.")
        return 0
    print(f"{'TARGET':34} {'OK':5} {'COOLDOWN':9} {'OK/FAIL':9} {'LATENCY':8} LAST ERROR")
    for key, row in sorted(health.items()):
        latency = f"{row['ewma_latency_seconds']:.2f}s" if row["ewma_latency_seconds"] else "-"
        print(
            f"{key:34} {row['available']!s:5} "
            f"{row['cooldown_remaining']:>7.0f}s  "
            f"{row['successes']}/{row['failures']:<7} {latency:8} "
            f"{(row['last_error'] or '')[:40]}"
        )
    return 0


async def _cmd_show(args: argparse.Namespace) -> int:
    llm = await _llm(args)
    try:
        print("pools:")
        for name, members in (llm.routing.pools() or {}).items():
            print(f"  {name}: {', '.join(members)}")
        print("lanes:")
        for name, destination in (llm.routing.lanes() or {}).items():
            print(f"  {name} -> {destination}")
        print(f"classifier chain (in run order): {', '.join(llm.routing.classifiers()) or '(none)'}")
        print(f"transforms: {', '.join(llm.routing.transforms()) or '(none)'}")
    finally:
        await llm.close()
    return 0


async def _cmd_eval(args: argparse.Namespace) -> int:
    from .classifiers import build_classifier
    from .evaluation import compare, load_cases

    llm = await _llm(args)
    try:
        cases = load_cases(args.cases)
        lanes = llm.routing.lanes()
        order = tuple(args.lane_order.split(",")) if args.lane_order else tuple(lanes)
        if not order:
            print(
                "No lanes configured and no --lane-order given, so the report cannot say "
                "whether a misroute was too cheap or too expensive -- which is the number "
                "worth having. Configure [routing.lanes] or pass --lane-order.",
                file=sys.stderr,
            )

        manager = llm._routing_manager
        config = dict(manager._classifier_config)
        config.setdefault("provider_getter", manager._typesafe_provider)
        config.setdefault("ask", llm._routing_ask)

        names = args.classifiers.split(",") if args.classifiers else list(
            llm.routing.classifiers()
        )
        built: dict[str, Any] = {}
        for name in names:
            classifier = build_classifier(name.strip(), config=config, lanes=manager._lanes)
            if classifier is not None:
                built[name.strip()] = classifier
        if args.chain:
            built["<chain>"] = manager.classifier_chain()
        if not built:
            print("No classifiers could be built.", file=sys.stderr)
            return 1

        labelled = sum(1 for c in cases if c.expected is not None)
        print(
            f"{len(cases)} cases ({labelled} labelled), lanes cheapest-first: "
            f"{' < '.join(order) or '(unknown)'}\n"
        )
        reports = await compare(built, cases, lane_order=order)
    finally:
        await llm.close()

    for report in reports.values():
        print(report.summary())
        if args.show_misroutes:
            for row in report.misroutes(direction="cheap", limit=args.show_misroutes):
                print(
                    f"    too cheap: {row.case.prompt[:58]!r}\n"
                    f"      wanted {row.case.expected}, got {row.predicted}"
                    + (f" ({row.rationale})" if row.rationale else "")
                )
        print()

    if args.json:
        # Off the event loop. Harmless in a one-shot CLI, and the kind of
        # blocking call that becomes a real bug the moment this function is
        # lifted into a service.
        payload = json.dumps({k: v.to_dict() for k, v in reports.items()}, indent=2)
        await asyncio.to_thread(Path(args.json).write_text, payload)
        print(f"wrote {args.json}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmcore-routing",
        description="Explain, inspect and measure llmcore routing.",
    )
    parser.add_argument("--config", default=None, help="path to a TOML config")
    parser.add_argument(
        "--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    sub = parser.add_subparsers(dest="command", required=True)

    why = sub.add_parser("why", help="where would this prompt go, and why? (no call made)")
    why.add_argument("prompt")
    why.add_argument("--lane", default=None)
    why.add_argument("--pool", default=None)
    why.set_defaults(func=_cmd_why)

    health = sub.add_parser("health", help="per-target cooldowns, latency and failures")
    health.set_defaults(func=_cmd_health)

    show = sub.add_parser("show", help="configured pools, lanes, classifiers and transforms")
    show.set_defaults(func=_cmd_show)

    ev = sub.add_parser(
        "eval",
        help="measure classifiers against YOUR labelled prompts",
        description=(
            "Reports too-cheap and too-expensive separately, because the two error "
            "directions are not interchangeable: routing too cheap produces a bad answer, "
            "routing too expensive only costs money."
        ),
    )
    ev.add_argument("cases", help="JSONL of {\"prompt\": ..., \"expected\": \"<lane>\"}")
    ev.add_argument(
        "--classifiers",
        default=None,
        help="comma list to compare (default: the configured chain, individually)",
    )
    ev.add_argument("--chain", action="store_true", help="also evaluate the chain as a whole")
    ev.add_argument(
        "--lane-order",
        default=None,
        help="lanes cheapest-first, comma separated; needed to judge misroute direction",
    )
    ev.add_argument("--show-misroutes", type=int, default=5, metavar="N")
    ev.add_argument("--json", default=None, help="write the report to this path")
    ev.set_defaults(func=_cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``llmcore-routing``."""
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
