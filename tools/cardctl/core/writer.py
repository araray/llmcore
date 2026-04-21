# tools/cardctl/core/writer.py
"""File I/O for model cards: create, update, cleanup."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import cards_dir_for_provider, ensure_init_py, sanitize_filename

logger = logging.getLogger(__name__)


@dataclass
class WriteResult:
    """Outcome of a single card write operation."""

    model_id: str
    action: str  # "created", "updated", "skipped", "failed"
    path: str = ""
    reason: str = ""


def write_cards(
    provider: str,
    cards: list[dict[str, Any]],
    cards_root: Path | None = None,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> list[WriteResult]:
    """Write a batch of card dicts to the provider's card directory.

    Args:
        provider: Provider name.
        cards: List of card dicts (already validated).
        cards_root: Root of card directories.
        dry_run: If True, report what would happen without writing.
        force: If True, overwrite even ``source: builtin`` cards.

    Returns:
        List of WriteResult describing each action.
    """
    card_dir = cards_dir_for_provider(provider, cards_root)
    results: list[WriteResult] = []

    if not dry_run:
        card_dir.mkdir(parents=True, exist_ok=True)
        ensure_init_py(card_dir)

    for card in cards:
        model_id = card.get("model_id", "")
        filename = sanitize_filename(model_id)
        filepath = card_dir / filename

        # Check existing card
        if filepath.exists():
            try:
                with open(filepath, encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = {}

            existing_source = existing.get("source", "builtin")

            if existing_source == "builtin" and not force:
                results.append(WriteResult(
                    model_id=model_id,
                    action="skipped",
                    path=str(filepath),
                    reason="manual card (source=builtin); use --force to overwrite",
                ))
                continue

            # Update: write the new card
            if not dry_run:
                _write_json(filepath, card)
            results.append(WriteResult(
                model_id=model_id,
                action="updated",
                path=str(filepath),
            ))
        else:
            # Create new card
            if not dry_run:
                _write_json(filepath, card)
            results.append(WriteResult(
                model_id=model_id,
                action="created",
                path=str(filepath),
            ))

    return results


def cleanup_unlisted(
    provider: str,
    listed_model_ids: set[str],
    cards_root: Path | None = None,
    *,
    dry_run: bool = True,
) -> list[WriteResult]:
    """Remove card files for models no longer listed by the API.

    Only removes cards with ``source: generated``.  Manual (builtin) cards
    are never touched.

    Args:
        provider: Provider name.
        listed_model_ids: Set of model IDs currently returned by the API.
        cards_root: Root of card directories.
        dry_run: If True (default), report without deleting.

    Returns:
        List of WriteResult for each removed/would-remove file.
    """
    card_dir = cards_dir_for_provider(provider, cards_root)
    results: list[WriteResult] = []

    if not card_dir.exists():
        return results

    for json_file in sorted(card_dir.glob("*.json")):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        model_id = data.get("model_id", "")
        source = data.get("source", "builtin")

        if model_id not in listed_model_ids and source == "generated":
            if not dry_run:
                json_file.unlink()
            results.append(WriteResult(
                model_id=model_id,
                action="removed" if not dry_run else "would_remove",
                path=str(json_file),
                reason="not listed by API",
            ))

    return results


def _write_json(filepath: Path, data: dict[str, Any]) -> None:
    """Atomically write a JSON file (write to temp, then rename)."""
    tmp = filepath.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.rename(filepath)
