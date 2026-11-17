# tests/tools/test_google_adapter_classification.py
"""cardctl Google adapter against the 2026-11-16 listing.

28 live models had no card: Gemini 4, Gemma 4, Lyria, Live, decision and
agent models were filtered out by a gemini-only prefix and a
generateContent/embedContent-only method check; streaming read a method the
listing no longer names; reasoning ignored the entry's ``thinking`` flag.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.cardctl.adapters.google_adapter import GoogleAdapter  # noqa: E402
from tools.cardctl.core.enrichment import EnrichmentStore  # noqa: E402


def _parse(name: str, methods: list[str], **extra):
    return GoogleAdapter(api_key="k")._parse_model(
        {"name": f"models/{name}", "supportedGenerationMethods": methods, "inputTokenLimit": 1000, **extra})


def test_families_and_types():
    carbon = _parse("gemini-4-carbon", ["generateContent", "countTokens"], thinking=True)
    assert carbon.model_type == "chat" and carbon.supports_tools and carbon.supports_vision
    assert carbon.supports_streaming and carbon.supports_reasoning
    assert _parse("gemini-decision-1", ["createDecision"]).model_type == "decision"
    assert _parse("lyria-3.5", ["generateContent"]).model_type == "multimodal"
    assert _parse("gemini-3.8-live", ["bidiGenerateContent"]).model_type == "multimodal"
    assert _parse("gemini-3.5-transcribe", ["generateContent"]).model_type == "stt"
    assert _parse("gemini-3.8-flash-tts", ["generateContent"]).model_type == "tts"
    image = _parse("gemini-nano-banana-2.1", ["generateContent"])
    assert image.model_type == "multimodal" and image.supports_vision and "image-output" in image.tags
    assert _parse("gemma-4-31b-it", ["generateContent"], thinking=True).supports_reasoning
    assert _parse("aqa", ["generateAnswer"]) is None


def test_enrichment_prices_never_reach_variants():
    store = EnrichmentStore.load("google")
    assert store.get("gemini-2.5-pro").pricing is not None
    assert store.get("gemini-2.5-pro-preview-tts").pricing is None  # was given the chat price
    assert store.get("gemini-2.5-pro-001").pricing is not None  # a versioned snapshot inherits
