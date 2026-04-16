#!/usr/bin/env python3
"""
Generate llmcore model cards from the Poe ``/v1/models`` API.

Usage:
    python tools/generate_poe_cards.py [--output-dir DIR] [--top N]

This script:
1. Fetches all available models/bots from ``https://api.poe.com/v1/models``
2. Converts each model to an llmcore ModelCard JSON file
3. Writes to ``src/llmcore/model_cards/default_cards/poe/``

**Requires** a valid ``POE_API_KEY`` environment variable — unlike
OpenRouter, the Poe ``/v1/models`` endpoint requires authentication.

Environment Variables:
    POE_API_KEY: Your Poe API key (required).  Get one at https://poe.com/api/keys

Examples:
    # Generate all cards
    POE_API_KEY=sk-... python tools/generate_poe_cards.py

    # Generate only the first 50 models
    POE_API_KEY=sk-... python tools/generate_poe_cards.py --top 50

    # Custom output directory
    POE_API_KEY=sk-... python tools/generate_poe_cards.py --output-dir /tmp/poe_cards
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import httpx
except ImportError:
    print("ERROR: httpx is required. Install with: pip install httpx")
    sys.exit(1)

POE_MODELS_URL = "https://api.poe.com/v1/models"
POE_BALANCE_URL = "https://api.poe.com/usage/current_balance"
DEFAULT_OUTPUT_DIR = "src/llmcore/model_cards/default_cards/poe"


def get_api_key() -> str:
    """Resolve the Poe API key from environment."""
    key = os.environ.get("POE_API_KEY", "")
    if not key:
        print(
            "ERROR: POE_API_KEY environment variable is required.\n"
            "       Get your key at: https://poe.com/api/keys"
        )
        sys.exit(1)
    return key


def fetch_models(api_key: str) -> list[dict]:
    """Fetch all models from the Poe /v1/models endpoint.

    Args:
        api_key: A valid Poe API key.

    Returns:
        List of model dicts from the ``data`` field.
    """
    print(f"Fetching models from {POE_MODELS_URL}...")
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = httpx.get(POE_MODELS_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    models = data.get("data", [])
    print(f"  Fetched {len(models)} models/bots.")
    return models


def check_balance(api_key: str) -> int | None:
    """Optionally check and display the current point balance.

    Returns:
        Balance in points, or None on failure.
    """
    try:
        resp = httpx.get(
            POE_BALANCE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        resp.raise_for_status()
        balance = resp.json().get("current_point_balance", 0)
        print(f"  Current point balance: {balance:,}")
        return balance
    except Exception as e:
        print(f"  (Could not check balance: {e})")
        return None


def model_to_card(model: dict) -> dict:
    """Convert a Poe model dict to an llmcore ModelCard dict.

    The Poe /v1/models endpoint returns a subset of fields compared
    to OpenRouter.  We map what's available and mark unknowns.

    Args:
        model: Raw model dict from the API.

    Returns:
        A dict matching the llmcore ModelCard JSON schema.
    """
    model_id = model.get("id", "")
    architecture = model.get("architecture", {}) or {}
    pricing = model.get("pricing", {}) or {}
    input_mods = architecture.get("input_modalities", [])
    output_mods = architecture.get("output_modalities", [])
    modality_str = architecture.get("modality", "")

    # Parse prices (per-token strings)
    input_price_per_token = float(pricing.get("prompt", "0") or "0")
    output_price_per_token = float(pricing.get("completion", "0") or "0")
    input_price_per_m = input_price_per_token * 1_000_000
    output_price_per_m = output_price_per_token * 1_000_000

    # Determine model type from modalities
    model_type = "chat"
    if "video" in output_mods and "text" not in output_mods:
        model_type = "audio"  # Video generation bots
    elif "image" in output_mods and "text" not in output_mods:
        model_type = "image-generation"
    elif "image" in output_mods:
        model_type = "multimodal"

    # Capabilities
    capabilities = {
        "streaming": True,
        "function_calling": True,  # Poe supports tools for all bots
        "tool_use": True,
        "json_mode": False,  # response_format not supported
        "structured_output": False,
        "vision": "image" in input_mods,
        "audio_input": "audio" in input_mods,
        "audio_output": "audio" in output_mods,
        "video_input": "video" in input_mods,
        "image_generation": "image" in output_mods,
        "web_search": False,  # Only via Responses API tools
        "reasoning": False,  # Determined per-model below
        "file_processing": "file" in input_mods,
    }

    # Heuristic: reasoning models
    lower_id = model_id.lower()
    if any(t in lower_id for t in ("o1", "o3", "o4", "thinking", "reason")):
        capabilities["reasoning"] = True

    # Pricing (skip if free)
    pricing_dict = None
    if input_price_per_m > 0 or output_price_per_m > 0:
        pricing_dict = {
            "currency": "USD",
            "per_million_tokens": {
                "input": round(input_price_per_m, 4),
                "output": round(output_price_per_m, 4),
            },
        }

    # Determine family from owned_by or model name
    owned_by = model.get("owned_by", "")
    family = owned_by if owned_by else None

    # Context length: Poe doesn't expose this in /v1/models — we use
    # a best-effort heuristic based on the model name.
    context_length = _guess_context_length(model_id)

    card: dict = {
        "model_id": model_id,
        "display_name": model.get("description", model_id)[:200]
        if model.get("description")
        else model_id,
        "provider": "poe",
        "model_type": model_type,
        "architecture": {
            "family": family,
        },
        "context": {
            "max_input_tokens": context_length,
        },
        "capabilities": capabilities,
        "lifecycle": {"status": "active"},
        "license": None,
        "open_weights": False,
        "aliases": [],
        "description": model.get("description", ""),
        "tags": _derive_tags(model_id, input_mods, output_mods, owned_by),
        "source": "builtin",
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }

    if pricing_dict:
        card["pricing"] = pricing_dict

    # Poe-specific extension
    card["provider_extension"] = {
        "owned_by": owned_by,
        "input_modalities": input_mods,
        "output_modalities": output_mods,
        "modality": modality_str,
        "poe_object_type": model.get("object"),
        "created": model.get("created"),
    }

    return card


def _guess_context_length(model_id: str) -> int:
    """Heuristic context length based on known model families.

    Since Poe doesn't expose context length in its /v1/models response,
    we map well-known bots to their upstream provider's context limits.

    Args:
        model_id: The Poe bot name.

    Returns:
        Best-guess context length in tokens.
    """
    lower = model_id.lower()

    # GPT-5.x family
    if "gpt-5" in lower:
        return 400_000

    # GPT-4.1 family
    if "gpt-4.1" in lower:
        return 1_048_576

    # GPT-4o family
    if "gpt-4o" in lower:
        return 128_000

    # Claude families
    if "claude" in lower:
        if "opus-4" in lower or "sonnet-4" in lower:
            return 200_000
        if "haiku" in lower:
            return 200_000
        return 200_000

    # Gemini families
    if "gemini" in lower:
        if "3" in lower:
            return 1_000_000
        if "2.5" in lower or "2.0" in lower:
            return 1_000_000
        return 128_000

    # Grok
    if "grok" in lower:
        return 131_072

    # o-series reasoning
    if lower.startswith("o1") or lower.startswith("o3") or lower.startswith("o4"):
        return 200_000

    # DeepSeek
    if "deepseek" in lower:
        return 128_000

    # Llama
    if "llama" in lower:
        return 128_000

    # Image/Video generation bots (context doesn't apply the same way)
    if any(t in lower for t in ("sora", "veo", "dall-e", "imagen", "flux", "sdxl")):
        return 4_096

    # Conservative default
    return 128_000


def _derive_tags(
    model_id: str,
    input_mods: list[str],
    output_mods: list[str],
    owned_by: str,
) -> list[str]:
    """Derive categorisation tags from model metadata."""
    tags: list[str] = []

    lower = model_id.lower()

    # Provider tags
    if owned_by:
        tags.append(owned_by.lower())

    # Modality tags
    if "image" in input_mods:
        tags.append("vision")
    if "image" in output_mods:
        tags.append("image-generation")
    if "video" in output_mods:
        tags.append("video-generation")
    if "audio" in output_mods:
        tags.append("audio")

    # Reasoning
    if any(t in lower for t in ("o1", "o3", "o4", "thinking", "reason")):
        tags.append("reasoning")

    # Code
    if any(t in lower for t in ("code", "codex", "coder")):
        tags.append("code")

    return sorted(set(tags))


def sanitize_filename(model_id: str) -> str:
    """Convert bot name to a safe filename.

    Poe bot names can contain spaces, special chars, and mixed case.
    We normalise to lowercase with safe replacements.

    Args:
        model_id: The Poe bot name.

    Returns:
        Safe filename with ``.json`` extension.
    """
    safe = model_id.replace("/", "--").replace(":", "-").replace(" ", "_")
    return safe + ".json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate llmcore model cards from the Poe /v1/models API"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Only generate cards for the top N models (0=all)",
    )
    args = parser.parse_args()

    api_key = get_api_key()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    check_balance(api_key)
    models = fetch_models(api_key)

    if args.top > 0:
        models = models[: args.top]

    count = 0
    for model in models:
        model_id = model.get("id", "")
        if not model_id:
            continue

        card = model_to_card(model)
        filename = sanitize_filename(model_id)
        filepath = output_dir / filename

        with open(filepath, "w") as f:
            json.dump(card, f, indent=2, default=str)
        count += 1

    print(f"\nGenerated {count} model cards in {output_dir}/")

    # Write __init__.py
    init_path = output_dir / "__init__.py"
    if not init_path.exists():
        init_path.write_text("# src/llmcore/model_cards/default_cards/poe/__init__.py\n")

    # Summary stats
    _print_summary(models)


def _print_summary(models: list[dict]) -> None:
    """Print a brief summary of fetched models."""
    owners: dict[str, int] = {}
    for m in models:
        owner = m.get("owned_by", "unknown")
        owners[owner] = owners.get(owner, 0) + 1

    print("\nModels by provider:")
    for owner, cnt in sorted(owners.items(), key=lambda x: -x[1]):
        print(f"  {owner}: {cnt}")


if __name__ == "__main__":
    main()
