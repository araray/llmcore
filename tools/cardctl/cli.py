# tools/cardctl/cli.py
"""Unified CLI for model card management.

Usage::

    python -m tools.cardctl generate openai
    python -m tools.cardctl validate
    python -m tools.cardctl diff mistral
    python -m tools.cardctl cleanup mistral --write
    python -m tools.cardctl stats
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from .adapters import get_adapter, list_providers
from .core.builder import CardBuilder
from .core.common import DEFAULT_CARDS_ROOT, cards_dir_for_provider
from .core.differ import diff_provider
from .core.enrichment import EnrichmentStore
from .core.validator import validate_all_cards, validate_card_dict, validate_provider_cards
from .core.writer import WriteResult, cleanup_unlisted, write_cards

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns exit code."""
    parser = argparse.ArgumentParser(
        prog="cardctl",
        description="Model card lifecycle management for llmcore.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    parser.add_argument(
        "--cards-root",
        type=Path,
        default=None,
        help=f"Root directory for model cards (default: {DEFAULT_CARDS_ROOT}).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- generate ---
    gen = sub.add_parser("generate", help="Fetch models from API and write/update cards.")
    gen.add_argument("provider", nargs="?", help="Provider name (or --all).")
    gen.add_argument("--all", action="store_true", help="Generate for all providers.")
    gen.add_argument("--dry-run", action="store_true", help="Preview without writing.")
    gen.add_argument("--force", action="store_true", help="Overwrite manual (builtin) cards.")
    gen.add_argument("--include-deprecated", action="store_true", help="Include deprecated models.")
    gen.add_argument("--api-key", default=None, help="API key (overrides env var).")

    # --- validate ---
    val = sub.add_parser("validate", help="Validate cards against ModelCard schema.")
    val.add_argument("provider", nargs="?", help="Provider name (or all if omitted).")

    # --- diff ---
    dif = sub.add_parser("diff", help="Compare local cards vs live API (read-only).")
    dif.add_argument("provider", help="Provider name.")
    dif.add_argument("--api-key", default=None, help="API key.")
    dif.add_argument("--format", choices=["text", "json"], default="text")

    # --- cleanup ---
    cln = sub.add_parser("cleanup", help="Remove stale/invalid cards.")
    cln.add_argument("provider", help="Provider name.")
    cln.add_argument("--write", action="store_true", help="Actually delete files (default: dry-run).")
    cln.add_argument("--remove-unlisted", action="store_true", help="Remove cards not in API.")
    cln.add_argument("--remove-invalid", action="store_true", help="Remove cards failing validation.")
    cln.add_argument("--api-key", default=None, help="API key (needed for --remove-unlisted).")

    # --- stats ---
    sub.add_parser("stats", help="Dashboard of card coverage and health.")

    args = parser.parse_args(argv)

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)-5s %(message)s",
        stream=sys.stderr,
    )

    cards_root = args.cards_root

    try:
        if args.command == "generate":
            return _cmd_generate(args, cards_root)
        elif args.command == "validate":
            return _cmd_validate(args, cards_root)
        elif args.command == "diff":
            return _cmd_diff(args, cards_root)
        elif args.command == "cleanup":
            return _cmd_cleanup(args, cards_root)
        elif args.command == "stats":
            return _cmd_stats(cards_root)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as e:
        logger.error("Fatal: %s", e, exc_info=args.verbose)
        return 1

    return 0


# ======================================================================
# Subcommand implementations
# ======================================================================


def _cmd_generate(args: argparse.Namespace, cards_root: Path | None) -> int:
    """Generate model cards from provider APIs."""
    if args.all:
        providers = list_providers()
    elif args.provider:
        providers = [args.provider]
    else:
        print("ERROR: specify a provider name or --all", file=sys.stderr)
        return 1

    total_created = total_updated = total_skipped = total_failed = 0

    for provider in providers:
        print(f"\n{'='*60}")
        print(f"  Generating: {provider}")
        print(f"{'='*60}")

        try:
            adapter = get_adapter(provider, api_key=args.api_key)
        except ValueError as e:
            print(f"  SKIP: {e}", file=sys.stderr)
            continue

        # Check API key availability
        try:
            adapter.check_api_key()
        except RuntimeError as e:
            print(f"  SKIP: {e}", file=sys.stderr)
            continue

        # Fetch models
        try:
            models = asyncio.run(adapter.fetch_models())
        except Exception as e:
            print(f"  ERROR fetching models: {e}", file=sys.stderr)
            continue

        print(f"  Fetched {len(models)} models from API")

        # Filter deprecated
        if not args.include_deprecated:
            before = len(models)
            models = [m for m in models if not m.is_deprecated]
            skipped_dep = before - len(models)
            if skipped_dep:
                print(f"  Skipped {skipped_dep} deprecated models")

        # Load enrichments and build cards
        enrichments = EnrichmentStore.load(provider)
        builder = CardBuilder(provider, enrichments)

        cards: list[dict[str, Any]] = []
        validation_failures = 0

        for model in models:
            if enrichments.is_excluded(model.model_id):
                continue

            card = builder.build(model)

            # Validate before writing
            result = validate_card_dict(card)
            if not result.valid:
                print(f"  VALIDATION FAIL: {model.model_id}: {result.error}")
                validation_failures += 1
                continue

            cards.append(card)

        # Write cards
        results = write_cards(
            provider, cards, cards_root,
            dry_run=args.dry_run, force=args.force,
        )

        # Summary
        created = sum(1 for r in results if r.action == "created")
        updated = sum(1 for r in results if r.action == "updated")
        skipped = sum(1 for r in results if r.action == "skipped")

        prefix = "[DRY-RUN] " if args.dry_run else ""
        print(f"  {prefix}{created} created, {updated} updated, "
              f"{skipped} skipped, {validation_failures} validation failures")

        total_created += created
        total_updated += updated
        total_skipped += skipped
        total_failed += validation_failures

    print(f"\nTotal: {total_created} created, {total_updated} updated, "
          f"{total_skipped} skipped, {total_failed} failed")

    return 1 if total_failed > 0 else 0


def _cmd_validate(args: argparse.Namespace, cards_root: Path | None) -> int:
    """Validate model cards against schema."""
    if args.provider:
        results = {args.provider: validate_provider_cards(args.provider, cards_root)}
    else:
        results = validate_all_cards(cards_root)

    total_pass = total_fail = 0
    for provider, vresults in sorted(results.items()):
        passed = sum(1 for r in vresults if r.valid)
        failed = sum(1 for r in vresults if not r.valid)
        total_pass += passed
        total_fail += failed

        status = "OK" if failed == 0 else "FAIL"
        print(f"  {provider:15s}: {passed:4d} pass, {failed:3d} fail  [{status}]")

        for r in vresults:
            if not r.valid:
                print(f"    FAIL: {r.path}: {r.error}")

    print(f"\nTotal: {total_pass} passed, {total_fail} failed")
    return 1 if total_fail > 0 else 0


def _cmd_diff(args: argparse.Namespace, cards_root: Path | None) -> int:
    """Diff local cards vs API."""
    adapter = get_adapter(args.provider, api_key=args.api_key)
    adapter.check_api_key()

    models = asyncio.run(adapter.fetch_models())
    report = diff_provider(args.provider, models, cards_root)

    if args.format == "json":
        out = {
            "provider": report.provider,
            "local_count": report.local_count,
            "api_count": report.api_count,
            "entries": [
                {"model_id": e.model_id, "kind": e.kind, "details": e.details}
                for e in report.entries
            ],
        }
        print(json.dumps(out, indent=2))
    else:
        print(report.summary())

    return 1 if report.has_differences else 0


def _cmd_cleanup(args: argparse.Namespace, cards_root: Path | None) -> int:
    """Clean up stale/invalid cards."""
    results: list[WriteResult] = []
    dry_run = not args.write

    if args.remove_invalid:
        vresults = validate_provider_cards(args.provider, cards_root)
        card_dir = cards_dir_for_provider(args.provider, cards_root)
        for vr in vresults:
            if not vr.valid:
                fp = Path(vr.path)
                if fp.exists():
                    # Only remove generated cards
                    try:
                        with open(fp) as f:
                            data = json.load(f)
                        if data.get("source") == "generated":
                            if not dry_run:
                                fp.unlink()
                            results.append(WriteResult(
                                model_id=data.get("model_id", fp.stem),
                                action="removed" if not dry_run else "would_remove",
                                path=str(fp),
                                reason=f"invalid: {vr.error}",
                            ))
                    except Exception:
                        pass

    if args.remove_unlisted:
        adapter = get_adapter(args.provider, api_key=args.api_key)
        adapter.check_api_key()
        models = asyncio.run(adapter.fetch_models())
        listed_ids = {m.model_id for m in models}
        results.extend(cleanup_unlisted(
            args.provider, listed_ids, cards_root, dry_run=dry_run,
        ))

    prefix = "[DRY-RUN] " if dry_run else ""
    for r in results:
        print(f"  {prefix}{r.action}: {r.model_id} ({r.reason})")

    if not results:
        print("  Nothing to clean up.")
    else:
        print(f"\n{prefix}{len(results)} files {'would be ' if dry_run else ''}affected.")
        if dry_run:
            print("  Use --write to actually delete files.")

    return 0


def _cmd_stats(cards_root: Path | None) -> int:
    """Print dashboard of card coverage and health."""
    root = cards_root or DEFAULT_CARDS_ROOT
    available = list_providers()

    header = (
        f"{'Provider':15s} {'Cards':>5s} {'Gen':>5s} {'Manual':>6s} "
        f"{'Valid':>5s} {'Invalid':>7s} {'Pricing':>7s} {'Arch':>5s} {'Adapter':>7s}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    totals = [0] * 7

    for provider_dir in sorted(root.iterdir()):
        if not provider_dir.is_dir() or provider_dir.name.startswith("_"):
            continue
        provider = provider_dir.name
        cards = list(provider_dir.glob("*.json"))
        total = len(cards)

        generated = manual = valid = invalid = has_pricing = has_arch = 0
        for fp in cards:
            try:
                with open(fp) as f:
                    data = json.load(f)
                if data.get("source") == "generated":
                    generated += 1
                else:
                    manual += 1
                if data.get("pricing"):
                    has_pricing += 1
                if data.get("architecture"):
                    has_arch += 1
                # Quick validation (just check required fields exist)
                if data.get("model_id") and data.get("context"):
                    valid += 1
                else:
                    invalid += 1
            except Exception:
                invalid += 1

        has_adapter = "yes" if provider in available else " - "

        print(
            f"{provider:15s} {total:5d} {generated:5d} {manual:6d} "
            f"{valid:5d} {invalid:7d} {has_pricing:7d} {has_arch:5d} {has_adapter:>7s}"
        )
        totals[0] += total
        totals[1] += generated
        totals[2] += manual
        totals[3] += valid
        totals[4] += invalid
        totals[5] += has_pricing
        totals[6] += has_arch

    print(sep)
    print(
        f"{'TOTAL':15s} {totals[0]:5d} {totals[1]:5d} {totals[2]:6d} "
        f"{totals[3]:5d} {totals[4]:7d} {totals[5]:7d} {totals[6]:5d}"
    )

    return 0
