# tests/model_cards/test_no_duplicate_cards.py
"""One card file per (provider, model_id) in the packaged tree.

Two files claiming one id (cardctl's ``a--b-c.json`` next to an older
``a__b_c.json``) made the registry pick a winner by origin; for Ollama that
meant qwen3:32b reported a 262144-token window instead of the server's 40960.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import llmcore.model_cards as mc


def test_each_model_id_has_one_card_file():
    root = Path(mc.__file__).parent / "default_cards"
    seen: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for path in root.rglob("*.json"):
        card = json.loads(path.read_text())
        seen[(str(card.get("provider")), str(card.get("model_id")))].append(path.name)
    duplicates = {f"{p}/{m}": names for (p, m), names in seen.items() if len(names) > 1}
    assert duplicates == {}
