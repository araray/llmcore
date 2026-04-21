# tools/cardctl/core/differ.py
"""Diff local model cards against API-discovered models."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..adapters.base import NormalizedModel
from .common import cards_dir_for_provider, sanitize_filename

logger = logging.getLogger(__name__)


@dataclass
class DiffEntry:
    """A single difference between local and API state."""

    model_id: str
    kind: str  # "new", "removed", "changed", "missing_pricing", "missing_arch"
    details: str = ""


@dataclass
class DiffReport:
    """Full diff report for a provider."""

    provider: str
    entries: list[DiffEntry] = field(default_factory=list)
    local_count: int = 0
    api_count: int = 0

    @property
    def new_models(self) -> list[DiffEntry]:
        return [e for e in self.entries if e.kind == "new"]

    @property
    def removed_models(self) -> list[DiffEntry]:
        return [e for e in self.entries if e.kind == "removed"]

    @property
    def changed_models(self) -> list[DiffEntry]:
        return [e for e in self.entries if e.kind == "changed"]

    @property
    def has_differences(self) -> bool:
        return len(self.entries) > 0

    def summary(self) -> str:
        lines = [f"Provider: {self.provider}  (local={self.local_count}, api={self.api_count})"]
        new = self.new_models
        removed = self.removed_models
        changed = self.changed_models
        if new:
            lines.append(f"  NEW ({len(new)}):")
            for e in new:
                lines.append(f"    + {e.model_id}")
        if removed:
            lines.append(f"  REMOVED ({len(removed)}):")
            for e in removed:
                lines.append(f"    - {e.model_id}")
        if changed:
            lines.append(f"  CHANGED ({len(changed)}):")
            for e in changed:
                lines.append(f"    ~ {e.model_id}: {e.details}")
        if not self.has_differences:
            lines.append("  (no differences)")
        return "\n".join(lines)


def diff_provider(
    provider: str,
    api_models: list[NormalizedModel],
    cards_root: Path | None = None,
) -> DiffReport:
    """Compare local cards against API-discovered models.

    Args:
        provider: Provider name.
        api_models: Models returned by the adapter.
        cards_root: Root of card directories.

    Returns:
        DiffReport with all differences.
    """
    card_dir = cards_dir_for_provider(provider, cards_root)
    report = DiffReport(provider=provider, api_count=len(api_models))

    # Load local cards
    local_cards: dict[str, dict[str, Any]] = {}
    if card_dir.exists():
        for json_file in card_dir.glob("*.json"):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)
                mid = data.get("model_id", json_file.stem)
                local_cards[mid] = data
            except Exception:
                pass
    report.local_count = len(local_cards)

    api_ids = {m.model_id for m in api_models}
    local_ids = set(local_cards.keys())

    # New models (in API, not local)
    for m in api_models:
        if m.model_id not in local_ids:
            report.entries.append(DiffEntry(
                model_id=m.model_id,
                kind="new",
                details=f"context={m.context_length}",
            ))

    # Removed models (in local as generated, not in API)
    for mid, card in local_cards.items():
        if mid not in api_ids and card.get("source") == "generated":
            report.entries.append(DiffEntry(
                model_id=mid,
                kind="removed",
                details="generated card no longer in API",
            ))

    # Changed models (both exist, key fields differ)
    api_by_id = {m.model_id: m for m in api_models}
    for mid in api_ids & local_ids:
        model = api_by_id[mid]
        card = local_cards[mid]
        changes: list[str] = []

        # Context length changed
        local_ctx = card.get("context", {}).get("max_input_tokens")
        if model.context_length and local_ctx and model.context_length != local_ctx:
            changes.append(f"context: {local_ctx}→{model.context_length}")

        # Deprecation status changed
        local_status = card.get("lifecycle", {}).get("status", "active")
        if model.is_deprecated and local_status == "active":
            changes.append("now deprecated")
        elif not model.is_deprecated and local_status == "deprecated":
            changes.append("un-deprecated")

        if changes:
            report.entries.append(DiffEntry(
                model_id=mid,
                kind="changed",
                details="; ".join(changes),
            ))

    return report
