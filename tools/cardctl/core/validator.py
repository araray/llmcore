# tools/cardctl/core/validator.py
"""Validate model card dicts against the ModelCard Pydantic schema."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a single card."""

    path: str  # filepath or model_id
    valid: bool
    error: str | None = None


def _get_model_card_class() -> type:
    """Import ModelCard, handling the heavy dependency chain gracefully.

    Strategy: import the schema module directly via its package path,
    bypassing ``llmcore/__init__.py`` which pulls in heavy deps
    (sqlalchemy, chromadb, etc.) that may not be installed in the
    tooling environment.
    """
    src_root = str(Path(__file__).resolve().parents[3] / "src")
    if src_root not in sys.path:
        sys.path.insert(0, src_root)

    # Direct subpackage import — avoids llmcore/__init__.py entirely.
    import importlib

    try:
        schema_mod = importlib.import_module("llmcore.model_cards.schema")
        ModelCard = schema_mod.ModelCard
        try:
            ModelCard.model_rebuild()
        except Exception:
            pass  # Already built if loaded via package __init__
        return ModelCard
    except ImportError:
        pass

    # Ultimate fallback: exec the file in a prepared namespace.
    from pydantic import BaseModel, ConfigDict, Field
    from enum import Enum
    from datetime import datetime

    schema_path = (
        Path(__file__).resolve().parents[3]
        / "src" / "llmcore" / "model_cards" / "schema.py"
    )
    ns: dict[str, Any] = {
        "__builtins__": __builtins__,
        "BaseModel": BaseModel,
        "Field": Field,
        "ConfigDict": ConfigDict,
        "Enum": Enum,
        "Any": Any,
        "Literal": __import__("typing").Literal,
        "datetime": datetime,
    }
    exec(schema_path.read_text(), ns)
    ModelCard = ns["ModelCard"]
    ModelCard.model_rebuild(_types_namespace=ns)
    return ModelCard


# Lazy singleton
_ModelCard: type | None = None


def get_model_card_class() -> type:
    """Get the ModelCard class (lazy loaded, cached)."""
    global _ModelCard
    if _ModelCard is None:
        _ModelCard = _get_model_card_class()
    return _ModelCard


def validate_card_dict(card_data: dict[str, Any]) -> ValidationResult:
    """Validate a single card dict against the ModelCard schema.

    Args:
        card_data: Card dictionary (must include ``source`` field).

    Returns:
        ValidationResult with ``valid=True`` or error message.
    """
    ModelCard = get_model_card_class()
    model_id = card_data.get("model_id", "<unknown>")

    # Ensure source field
    if "source" not in card_data:
        card_data = {**card_data, "source": "generated"}

    try:
        ModelCard.model_validate(card_data)
        return ValidationResult(path=model_id, valid=True)
    except Exception as e:
        # Extract first line of error for brevity
        err_msg = str(e).split("\n")[0]
        return ValidationResult(path=model_id, valid=False, error=err_msg)


def validate_card_file(filepath: Path) -> ValidationResult:
    """Validate a single ``.json`` card file.

    Args:
        filepath: Path to the JSON card file.

    Returns:
        ValidationResult.
    """
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return ValidationResult(path=str(filepath), valid=False, error=f"Invalid JSON: {e}")
    except Exception as e:
        return ValidationResult(path=str(filepath), valid=False, error=str(e))

    if "source" not in data:
        data["source"] = "builtin"

    result = validate_card_dict(data)
    result.path = str(filepath)
    return result


def validate_provider_cards(
    provider: str, cards_root: Path | None = None
) -> list[ValidationResult]:
    """Validate all card files for a provider.

    Args:
        provider: Provider name (directory under cards_root).
        cards_root: Root of card directories (default: ``src/llmcore/model_cards/default_cards``).

    Returns:
        List of ValidationResult for each card file.
    """
    from .common import cards_dir_for_provider

    card_dir = cards_dir_for_provider(provider, cards_root)
    if not card_dir.exists():
        logger.warning("Card directory does not exist: %s", card_dir)
        return []

    results: list[ValidationResult] = []
    for json_file in sorted(card_dir.glob("*.json")):
        results.append(validate_card_file(json_file))
    return results


def validate_all_cards(cards_root: Path | None = None) -> dict[str, list[ValidationResult]]:
    """Validate cards across all providers.

    Returns:
        Dict mapping provider name to list of ValidationResult.
    """
    from .common import DEFAULT_CARDS_ROOT

    root = cards_root or DEFAULT_CARDS_ROOT
    if not root.exists():
        logger.warning("Cards root does not exist: %s", root)
        return {}

    results: dict[str, list[ValidationResult]] = {}
    for provider_dir in sorted(root.iterdir()):
        if provider_dir.is_dir() and not provider_dir.name.startswith("_"):
            provider = provider_dir.name
            results[provider] = validate_provider_cards(provider, root)
    return results
