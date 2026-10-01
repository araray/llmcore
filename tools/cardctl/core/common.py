# tools/cardctl/core/common.py
"""Shared utilities for the model card management system."""

from __future__ import annotations

import re
from pathlib import Path

# Root of the model cards directory tree.
DEFAULT_CARDS_ROOT = Path("src/llmcore/model_cards/default_cards")

# Enrichment files directory.
ENRICHMENTS_DIR = Path(__file__).parent.parent / "enrichments"


def sanitize_filename(model_id: str) -> str:
    """Convert a model ID to a safe filesystem name.

    Rules:
    - ``/`` → ``--``  (e.g. ``openai/gpt-4o`` → ``openai--gpt-4o``)
    - ``:`` → ``-``   (e.g. ``model:tag`` → ``model-tag``)
    - spaces → ``_``
    - Append ``.json``

    Returns:
        Safe filename string ending in ``.json``.
    """
    safe = model_id.replace("/", "--").replace(":", "-").replace(" ", "_")
    return f"{safe}.json"


def model_id_to_display_name(model_id: str) -> str:
    """Heuristic conversion of a model ID to a human-readable display name.

    Examples:
        ``gpt-4o-mini`` → ``GPT 4o Mini``
        ``claude-sonnet-4-20250514`` → ``Claude Sonnet 4 20250514``
        ``mistral-large-2512`` → ``Mistral Large 2512``
    """
    # Strip provider prefix if present (e.g. "openai/gpt-4o" → "gpt-4o")
    if "/" in model_id:
        model_id = model_id.rsplit("/", 1)[-1]

    parts = re.split(r"[-_]+", model_id)
    result: list[str] = []
    for p in parts:
        if not p:
            continue
        # All-digit tokens (dates, version numbers) stay as-is
        if p.isdigit():
            result.append(p)
        # Short uppercase tokens (2-3 chars) stay uppercase
        elif len(p) <= 3 and p.isalpha():
            result.append(p.upper())
        # Mixed alphanumeric (e.g. "4o", "3b", "2.5") stay as-is
        elif re.match(r"^\d", p):
            result.append(p)
        else:
            result.append(p.capitalize())
    return " ".join(result)


#: Adapter aliases that must share another provider's card directory.
#:
#: Registering an alias in the adapter registry is not enough on its own: the
#: card directory is named after whatever string the caller passed, so
#: ``cardctl generate gemini`` would build a second ``gemini/`` tree duplicating
#: ``google/``. Aliases therefore canonicalize here, so there is exactly one
#: directory per provider no matter which spelling is used.
_CARD_DIR_ALIASES: dict[str, str] = {
    # llmcore's provider registry says "gemini"; the card tree has always been
    # "google", and the cards are the same models either way.
    "gemini": "google",
    "moonshot": "kimi",
    "friendliai": "friendli",
    "friendli_ai": "friendli",
    "glm": "zai",
    "zhipu": "zai",
    "zhipuai": "zai",
    "bigmodel": "zai",
    "jev": "typesafe",
    "fal_ai": "fal",
    "eleven_labs": "elevenlabs",
}


def canonical_provider_dir_name(provider: str) -> str:
    """Return the card-directory name for *provider*, resolving aliases."""
    return _CARD_DIR_ALIASES.get(provider.strip().lower(), provider.strip().lower())


def cards_dir_for_provider(provider: str, root: Path | None = None) -> Path:
    """Return the card directory for a given provider name.

    Aliases resolve to the canonical provider's directory, so a provider never
    ends up with two card trees under different spellings.
    """
    base = root or DEFAULT_CARDS_ROOT
    return base / canonical_provider_dir_name(provider)


def ensure_init_py(directory: Path) -> None:
    """Create ``__init__.py`` in *directory* if it does not exist."""
    init_path = directory / "__init__.py"
    if not init_path.exists():
        init_path.write_text(
            f'"""Auto-generated {directory.name} model cards."""\n'
        )
