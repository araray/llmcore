#!/usr/bin/env python3
# tools/generate_deepgram_cards.py
"""Generate llmcore model cards for Deepgram STT / TTS / Flux models.

Deepgram is a *voice / audio* provider.  Unlike token-billed LLM providers it
bills per audio-minute (STT) or per-character (TTS), so the cards intentionally
keep the token-centric ``pricing`` field ``null`` (tokens are not a meaningful
billing unit, and llmcore's ``ModelPricing.get_cost`` is token-based — populating
it with audio rates would yield wrong costs).  Instead, the **real** pay-as-you-go
rates are recorded — with correct units, source URL, and capture date — in the
schema's free-form ``provider_extension.pricing`` object (see ``_PRICING`` below).
Nominal ``context.max_input_tokens`` values are used for the same reason:

* STT/Flux cards use a large nominal (audio is duration-bounded, not
  token-bounded) so any code that consults the context length never spuriously
  truncates.
* TTS cards use ``2000`` — the documented REST text-to-speech input character
  cap — as the closest analogue of an input limit.

Pricing provenance: rates in ``_PRICING`` were captured from
https://deepgram.com/pricing on the date in ``_PRICING_AS_OF``.  Only rates
actually published there are given as numbers; models the page does not price
(e.g. Nova-2, Whisper Cloud) are marked ``"status": "not_listed"`` with no
fabricated figure, and any inferred mapping is flagged with ``"assumption"``.

The set is deliberately representative (the most-used models + a handful of
popular voices), not exhaustive; extend the tables below to add more.  Cards are
written to ``src/llmcore/model_cards/default_cards/deepgram/`` and validated
against :class:`llmcore.model_cards.schema.ModelCard` before being written.

Usage::

    python tools/generate_deepgram_cards.py [--out DIR] [--check]

``--check`` validates and prints what *would* be written without touching disk.

References:
    * Pricing:             https://deepgram.com/pricing
    * Models & Languages: https://developers.deepgram.com/docs/models-languages-overview
    * TTS voices:          https://developers.deepgram.com/docs/tts-models
    * Flux:                https://developers.deepgram.com/docs/flux/quickstart
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Nominal context sizes (see module docstring).
_STT_NOMINAL_CONTEXT = 1_000_000
_TTS_NOMINAL_CONTEXT = 2_000

# --- Speech-to-Text models (Nova / Flux / Whisper families) ---------------
# (model_id, display_name, family, status, tags, description)
_STT_MODELS: list[tuple[str, str, str, str, list[str], str]] = [
    (
        "nova-3",
        "Nova-3",
        "nova",
        "active",
        ["streaming", "batch", "keyterm", "multilingual"],
        "Deepgram Nova-3 speech-to-text (batch + streaming). Supports keyterm "
        "prompting and multilingual code-switching. Billed per audio-minute.",
    ),
    (
        "nova-3-medical",
        "Nova-3 Medical",
        "nova",
        "active",
        ["streaming", "batch", "medical"],
        "Deepgram Nova-3 tuned for medical terminology. Billed per audio-minute.",
    ),
    (
        "nova-2",
        "Nova-2",
        "nova",
        "active",
        ["streaming", "batch"],
        "Deepgram Nova-2 speech-to-text (batch + streaming). Billed per audio-minute.",
    ),
    (
        "nova-2-phonecall",
        "Nova-2 Phonecall",
        "nova",
        "active",
        ["streaming", "batch", "telephony"],
        "Deepgram Nova-2 tuned for telephony / phone-call audio. Billed per audio-minute.",
    ),
    (
        "whisper-large",
        "Whisper Cloud (large)",
        "whisper",
        "active",
        ["batch"],
        "Deepgram-hosted OpenAI Whisper (large) for batch transcription. "
        "Billed per audio-minute.",
    ),
    (
        "flux-general-en",
        "Flux General (English)",
        "flux",
        "active",
        ["streaming", "realtime", "turn-detection", "voice-agent"],
        "Deepgram Flux conversational speech recognition with native end-of-turn "
        "detection, purpose-built for voice agents (v2 streaming). Billed per audio-minute.",
    ),
]

# --- Text-to-Speech voices (Aura families) --------------------------------
# (model_id, display_name, family, status, tags, description)
_TTS_MODELS: list[tuple[str, str, str, str, list[str], str]] = [
    (
        "aura-2-thalia-en",
        "Aura-2 Thalia (English)",
        "aura",
        "active",
        ["streaming", "batch", "aura-2"],
        "Deepgram Aura-2 text-to-speech voice 'Thalia' (English). "
        "Batch + streaming synthesis. Billed per character.",
    ),
    (
        "aura-2-andromeda-en",
        "Aura-2 Andromeda (English)",
        "aura",
        "active",
        ["streaming", "batch", "aura-2"],
        "Deepgram Aura-2 voice 'Andromeda' (English). Billed per character.",
    ),
    (
        "aura-2-apollo-en",
        "Aura-2 Apollo (English)",
        "aura",
        "active",
        ["streaming", "batch", "aura-2"],
        "Deepgram Aura-2 voice 'Apollo' (English). Billed per character.",
    ),
    (
        "aura-asteria-en",
        "Aura Asteria (English)",
        "aura",
        "active",
        ["streaming", "batch", "aura"],
        "Deepgram Aura (v1) voice 'Asteria' (English). Billed per character.",
    ),
    (
        "aura-luna-en",
        "Aura Luna (English)",
        "aura",
        "active",
        ["streaming", "batch", "aura"],
        "Deepgram Aura (v1) voice 'Luna' (English). Billed per character.",
    ),
]


# --- Pricing (captured from https://deepgram.com/pricing) -----------------
# Recorded in ``provider_extension.pricing`` (NOT the token-centric ``pricing``
# field) with correct units.  Only rates actually published on the page are
# given numerically; unpriced models are marked ``not_listed``.
_PRICING_SOURCE = "https://deepgram.com/pricing"
_PRICING_AS_OF = "2026-06-25"


def _stt_price(
    *,
    rate_per_minute: float | None,
    promo_streaming: float | None = None,
    observed: list[float] | None = None,
    growth: list[float] | None = None,
    note: str,
    status: str = "published",
    assumption: str | None = None,
) -> dict[str, Any]:
    """Build a per-audio-minute STT pricing record for ``provider_extension``."""
    rec: dict[str, Any] = {
        "currency": "USD",
        "unit": "audio_minute",
        "billing_tier": "pay-as-you-go",
        "model_improvement_program": True,
        "source": _PRICING_SOURCE,
        "as_of": _PRICING_AS_OF,
        "status": status,
        "rate_per_minute": rate_per_minute,
        "note": note,
    }
    if promo_streaming is not None:
        rec["promotional_streaming_rate_per_minute"] = promo_streaming
    if observed is not None:
        rec["observed_rates_per_minute"] = observed
    if growth is not None:
        rec["growth_rate_per_minute"] = growth
    if assumption is not None:
        rec["assumption"] = assumption
    return {"pricing": rec}


def _tts_price(*, per_1k: float, growth_per_1k: float, note: str) -> dict[str, Any]:
    """Build a per-character TTS pricing record for ``provider_extension``."""
    return {
        "pricing": {
            "currency": "USD",
            "unit": "1k_characters",
            "billing_tier": "pay-as-you-go",
            "source": _PRICING_SOURCE,
            "as_of": _PRICING_AS_OF,
            "status": "published",
            "rate_per_1k_characters": per_1k,
            "rate_per_million_characters": round(per_1k * 1000, 4),
            "growth_rate_per_1k_characters": growth_per_1k,
            "note": note,
        }
    }


#: model_id -> ``provider_extension`` dict (or ``None`` to omit the field).
_PRICING: dict[str, dict[str, Any]] = {
    # --- STT (per audio-minute) ---
    "nova-3": _stt_price(
        rate_per_minute=0.0077,
        promo_streaming=0.0048,
        observed=[0.0048, 0.0077],
        growth=[0.0042, 0.0065],
        note=(
            "Nova-3 monolingual/English. The page shows two pay-as-you-go values; "
            "the lower is the current limited-time promotional streaming rate and "
            "the higher is the standard (pre-recorded/regular) rate. Growth (prepaid) "
            "tier shown as a [min, max] range. Multilingual Nova-3 is priced higher "
            "($0.0058-$0.0092/min PAYG)."
        ),
    ),
    "nova-3-medical": _stt_price(
        rate_per_minute=0.0077,
        promo_streaming=0.0048,
        observed=[0.0048, 0.0077],
        growth=[0.0042, 0.0065],
        status="inferred",
        assumption="priced_as_nova_3_monolingual",
        note=(
            "The pricing page does not list a separate Nova-3 Medical rate; this "
            "record assumes Nova-3 monolingual pricing. Verify before relying on it "
            "for billing."
        ),
    ),
    "nova-2": _stt_price(
        rate_per_minute=None,
        status="not_listed",
        note=(
            "Nova-2 rates are not published on the current Deepgram pricing page "
            "(which prices only Nova-3 and Flux). The FAQ notes legacy Nova-2 "
            "availability. Consult the pricing page or Deepgram for current rates."
        ),
    ),
    "nova-2-phonecall": _stt_price(
        rate_per_minute=None,
        status="not_listed",
        note=(
            "Nova-2 (phonecall) rates are not published on the current Deepgram "
            "pricing page. Consult the pricing page or Deepgram for current rates."
        ),
    ),
    "whisper-large": _stt_price(
        rate_per_minute=None,
        status="not_listed",
        note=(
            "Deepgram Whisper Cloud appears under concurrency limits but no "
            "per-minute rate is published on the current pricing page. Consult the "
            "pricing page or Deepgram for current rates."
        ),
    ),
    "flux-general-en": _stt_price(
        rate_per_minute=0.0077,
        promo_streaming=0.0065,
        observed=[0.0065, 0.0077],
        growth=[0.0057, 0.0065],
        note=(
            "Flux English. The page shows two pay-as-you-go values; the lower is the "
            "current limited-time promotional streaming rate and the higher is the "
            "standard rate. Growth (prepaid) tier shown as a [min, max] range. Flux "
            "Multilingual is a single $0.0078/min PAYG."
        ),
    ),
    # --- TTS (per 1k characters) ---
    "aura-2-thalia-en": _tts_price(
        per_1k=0.030, growth_per_1k=0.027, note="Aura-2 voice. Billed per character."
    ),
    "aura-2-andromeda-en": _tts_price(
        per_1k=0.030, growth_per_1k=0.027, note="Aura-2 voice. Billed per character."
    ),
    "aura-2-apollo-en": _tts_price(
        per_1k=0.030, growth_per_1k=0.027, note="Aura-2 voice. Billed per character."
    ),
    "aura-asteria-en": _tts_price(
        per_1k=0.0150,
        growth_per_1k=0.0135,
        note="Aura-1 (original Aura) voice. Billed per character.",
    ),
    "aura-luna-en": _tts_price(
        per_1k=0.0150,
        growth_per_1k=0.0135,
        note="Aura-1 (original Aura) voice. Billed per character.",
    ),
}


def _stt_card(
    model_id: str,
    display_name: str,
    family: str,
    status: str,
    tags: list[str],
    description: str,
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "display_name": display_name,
        "provider": "deepgram",
        "model_type": "stt",
        "architecture": {"family": family, "architecture_type": "transformer"},
        "context": {"max_input_tokens": _STT_NOMINAL_CONTEXT},
        "capabilities": {
            "streaming": "streaming" in tags,
            "audio_input": True,
            "audio_output": False,
        },
        "pricing": None,  # token-centric field N/A for audio; see provider_extension
        "provider_extension": _PRICING.get(model_id),
        "lifecycle": {"status": status},
        "license": None,
        "open_weights": False,
        "aliases": [],
        "description": description,
        "tags": tags,
        "source": "generated",
    }


def _tts_card(
    model_id: str,
    display_name: str,
    family: str,
    status: str,
    tags: list[str],
    description: str,
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "display_name": display_name,
        "provider": "deepgram",
        "model_type": "tts",
        "architecture": {"family": family, "architecture_type": "transformer"},
        "context": {"max_input_tokens": _TTS_NOMINAL_CONTEXT},
        "capabilities": {
            "streaming": "streaming" in tags,
            "audio_input": False,
            "audio_output": True,
        },
        "pricing": None,  # token-centric field N/A for audio; see provider_extension
        "provider_extension": _PRICING.get(model_id),
        "lifecycle": {"status": status},
        "license": None,
        "open_weights": False,
        "aliases": [],
        "description": description,
        "tags": tags,
        "source": "generated",
    }


def build_cards() -> list[dict[str, Any]]:
    """Build all Deepgram card dicts (STT then TTS)."""
    cards: list[dict[str, Any]] = []
    for row in _STT_MODELS:
        cards.append(_stt_card(*row))
    for row in _TTS_MODELS:
        cards.append(_tts_card(*row))
    return cards


def _validate(cards: list[dict[str, Any]]) -> None:
    """Validate every card against the llmcore ModelCard schema (best-effort).

    Import is done lazily so the generator still *emits* cards even if llmcore
    is not importable in the current environment; validation is skipped with a
    warning in that case.
    """
    try:
        from llmcore.model_cards.schema import ModelCard
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"WARNING: could not import ModelCard for validation: {exc}", file=sys.stderr)
        return
    for card in cards:
        ModelCard.model_validate(card)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Deepgram model cards.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "src"
        / "llmcore"
        / "model_cards"
        / "default_cards"
        / "deepgram",
        help="Output directory for the card JSON files.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate and report without writing to disk.",
    )
    args = parser.parse_args()

    cards = build_cards()
    _validate(cards)

    if args.check:
        for card in cards:
            print(f"[check] {card['provider']}/{card['model_id']} ({card['model_type']})")
        print(f"[check] {len(cards)} cards valid; would write to {args.out}")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    for card in cards:
        path = args.out / f"{card['model_id']}.json"
        path.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    print(f"Done: {len(cards)} Deepgram model cards -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
