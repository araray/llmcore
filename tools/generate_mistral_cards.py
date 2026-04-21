#!/usr/bin/env python3
"""
Generate llmcore model cards from the Mistral AI /v1/models API.

Usage:
    python tools/generate_mistral_cards.py [--output-dir DIR] [--api-key KEY]

This script:
1. Fetches all models from https://api.mistral.ai/v1/models
2. Enriches with known pricing data (the API does not expose pricing)
3. Converts each model to an llmcore ModelCard JSON file
4. Writes to src/llmcore/model_cards/default_cards/mistral/

Requires: MISTRAL_API_KEY environment variable or --api-key flag.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    import httpx
except ImportError:
    print("ERROR: httpx is required. Install with: pip install httpx")
    sys.exit(1)

MISTRAL_MODELS_URL = "https://api.mistral.ai/v1/models"
DEFAULT_OUTPUT_DIR = "src/llmcore/model_cards/default_cards/mistral"

# -------------------------------------------------------------------
# Known pricing for Mistral models (USD per million tokens).
# The /v1/models API does not expose pricing; these are maintained
# manually from https://docs.mistral.ai/getting-started/models/models_overview/
# Last updated: April 2026.
# -------------------------------------------------------------------
KNOWN_PRICING: dict[str, dict[str, float]] = {
    # Frontier
    "mistral-large-latest": {"input": 2.00, "output": 6.00},
    "mistral-large-2512": {"input": 2.00, "output": 6.00},
    # Medium
    "mistral-medium-latest": {"input": 0.40, "output": 2.00},
    "mistral-medium-2508": {"input": 0.40, "output": 2.00},
    "mistral-medium-2505": {"input": 0.40, "output": 2.00},
    # Small
    "mistral-small-latest": {"input": 0.10, "output": 0.30},
    "mistral-small-2506": {"input": 0.10, "output": 0.30},
    "mistral-small-2503": {"input": 0.10, "output": 0.30},
    # Magistral (reasoning)
    "magistral-medium-latest": {"input": 0.40, "output": 2.00},
    "magistral-small-latest": {"input": 0.10, "output": 0.30},
    "magistral-medium-2509": {"input": 0.40, "output": 2.00},
    "magistral-small-2509": {"input": 0.10, "output": 0.30},
    # Codestral
    "codestral-latest": {"input": 0.30, "output": 0.90},
    "codestral-2508": {"input": 0.30, "output": 0.90},
    # Devstral
    "devstral-2-latest": {"input": 0.50, "output": 1.50},
    # Ministral
    "ministral-3-14b-latest": {"input": 0.05, "output": 0.10},
    "ministral-3-8b-latest": {"input": 0.03, "output": 0.07},
    "ministral-3-3b-latest": {"input": 0.02, "output": 0.04},
    # Voxtral
    "voxtral-mini-latest": {"input": 0.07, "output": 0.07},
    # Pixtral
    "pixtral-large-latest": {"input": 2.00, "output": 6.00},
    # Embedding
    "mistral-embed": {"input": 0.10, "output": 0.00},
    "codestral-embed-2505": {"input": 0.15, "output": 0.00},
    # OCR
    "mistral-ocr-latest": {"input": 1.00, "output": 1.00},
    # Nemo
    "open-mistral-nemo": {"input": 0.15, "output": 0.15},
    # Moderation
    "mistral-moderation-latest": {"input": 0.10, "output": 0.10},
}

# -------------------------------------------------------------------
# Known architecture info not available from the API.
# -------------------------------------------------------------------
KNOWN_ARCHITECTURE: dict[str, dict[str, str]] = {
    "mistral-large": {"parameter_count": "675B", "active_parameters": "41B", "type": "moe"},
    "mistral-medium": {"parameter_count": "N/A", "type": "dense"},
    "mistral-small": {"parameter_count": "24B", "type": "dense"},
    "magistral-medium": {"parameter_count": "N/A", "type": "dense"},
    "magistral-small": {"parameter_count": "24B", "type": "dense"},
    "codestral": {"parameter_count": "N/A", "type": "dense"},
    "devstral-2": {"parameter_count": "123B", "type": "dense"},
    "devstral-small": {"parameter_count": "24B", "type": "dense"},
    "ministral-3-14b": {"parameter_count": "14B", "type": "dense"},
    "ministral-3-8b": {"parameter_count": "8B", "type": "dense"},
    "ministral-3-3b": {"parameter_count": "3B", "type": "dense"},
    "pixtral-large": {"parameter_count": "124B", "type": "dense"},
    "pixtral-12b": {"parameter_count": "12B", "type": "dense"},
    "voxtral-mini": {"parameter_count": "4B", "type": "dense"},
    "voxtral-small": {"parameter_count": "24B", "type": "dense"},
    "open-mistral-nemo": {"parameter_count": "12B", "type": "dense"},
    "mistral-embed": {"parameter_count": "N/A", "type": "embedding"},
    "codestral-embed": {"parameter_count": "N/A", "type": "embedding"},
}


def fetch_models(api_key: str) -> list[dict]:
    """Fetch all models from Mistral API."""
    print(f"Fetching models from {MISTRAL_MODELS_URL}...")
    resp = httpx.get(
        MISTRAL_MODELS_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    models = data.get("data", [])
    print(f"  Fetched {len(models)} models.")
    return models


def _find_architecture(model_id: str) -> dict[str, str]:
    """Look up architecture info by prefix matching."""
    for prefix, info in sorted(KNOWN_ARCHITECTURE.items(), key=lambda x: -len(x[0])):
        if model_id.startswith(prefix):
            return info
    return {}


def _find_pricing(model_id: str) -> dict[str, float] | None:
    """Look up pricing by exact match, then -latest alias."""
    if model_id in KNOWN_PRICING:
        return KNOWN_PRICING[model_id]
    return None


def model_to_card(model: dict) -> dict:
    """Convert a Mistral model dict to an llmcore ModelCard dict."""
    model_id = model.get("id", "")
    caps = model.get("capabilities", {}) or {}
    deprecation = model.get("deprecation") or None

    # Context length
    context_length = model.get("max_context_length") or 32768

    # Determine model type
    model_type = "chat"
    if caps.get("completion_chat", False) is False and not caps.get("completion_fim", False):
        if model_id.startswith("mistral-embed") or model_id.startswith("codestral-embed"):
            model_type = "embedding"
        elif model_id.startswith("mistral-ocr"):
            model_type = "ocr"
        elif model_id.startswith("mistral-moderation"):
            model_type = "moderation"

    if caps.get("completion_fim", False):
        model_type = "code"

    # Build capabilities
    capabilities = {
        "streaming": caps.get("completion_chat", False),
        "function_calling": caps.get("function_calling", False),
        "tool_use": caps.get("function_calling", False),
        "json_mode": caps.get("json_mode", False),
        "structured_output": caps.get("json_mode", False),
        "vision": caps.get("vision", False),
        "audio_input": caps.get("audio", False) or caps.get("audio_transcription", False),
        "reasoning": model_id.startswith("magistral"),
        "multimodal": caps.get("vision", False) or caps.get("audio", False),
    }

    # Architecture
    arch_info = _find_architecture(model_id)
    architecture = {"family": "mistral"}
    if arch_info.get("parameter_count"):
        architecture["parameter_count"] = arch_info["parameter_count"]
    if arch_info.get("active_parameters"):
        architecture["active_parameters"] = arch_info["active_parameters"]
    if arch_info.get("type"):
        architecture["architecture_type"] = arch_info["type"]

    # Pricing
    pricing_info = _find_pricing(model_id)
    pricing = None
    if pricing_info:
        pricing = {
            "currency": "USD",
            "per_million_tokens": {
                "input": pricing_info["input"],
                "output": pricing_info["output"],
            },
        }

    # Lifecycle
    lifecycle: dict[str, str] = {"status": "active"}
    created = model.get("created")
    if created:
        try:
            dt = datetime.fromtimestamp(created)
            lifecycle["release_date"] = dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError, OSError):
            pass

    if deprecation:
        lifecycle["status"] = "deprecated"
        if isinstance(deprecation, str):
            lifecycle["deprecation_date"] = deprecation

    # License
    license_type = None
    open_weights = False
    model_type_field = model.get("type", "")
    owned_by = model.get("owned_by", "mistralai")

    # Description
    description = model.get("description", "")

    # Aliases
    aliases = model.get("aliases", []) or []

    # Tags
    tags: list[str] = []
    if open_weights:
        tags.append("open-weights")
    if capabilities.get("vision"):
        tags.append("multimodal")
    if capabilities.get("audio_input"):
        tags.append("audio")
    if model_id.startswith("magistral"):
        tags.append("reasoning")
    if model_type == "code":
        tags.append("code")
        tags.append("fim")
    if arch_info.get("type") == "moe":
        tags.append("moe")

    # Default temperature
    default_temp = model.get("default_model_temperature")

    # Build card
    card: dict = {
        "model_id": model_id,
        "display_name": model.get("name") or _model_id_to_display_name(model_id),
        "provider": "mistral",
        "model_type": model_type,
        "architecture": architecture,
        "context": {
            "max_input_tokens": int(context_length),
        },
        "capabilities": capabilities,
        "pricing": pricing,
        "lifecycle": lifecycle,
        "license": license_type,
        "open_weights": open_weights,
        "aliases": aliases,
        "description": description or f"Mistral AI model: {model_id}",
        "tags": tags,
        "provider_mistral": {
            "open_weights": open_weights,
            "license_type": license_type,
            "fill_in_middle": caps.get("completion_fim", False),
            "guardrails": {
                "system_prompt": caps.get("safe_prompt_injection", False),
            },
        },
        "source": "generated",
    }

    # Max output tokens — Mistral API doesn't report this directly
    max_output = model.get("max_output_tokens")
    if max_output:
        card["context"]["max_output_tokens"] = int(max_output)

    if default_temp is not None:
        card["provider_mistral"]["default_temperature"] = default_temp

    return card


def _model_id_to_display_name(model_id: str) -> str:
    """Convert model ID to a human-readable display name."""
    parts = model_id.replace("-", " ").replace("_", " ").split()
    # Capitalize intelligently
    result = []
    for p in parts:
        if p.isdigit() or len(p) <= 2:
            result.append(p)
        elif p.startswith("2"):
            result.append(p)  # Date-like suffixes (2512, 2506)
        else:
            result.append(p.capitalize())
    return " ".join(result)


def sanitize_filename(model_id: str) -> str:
    """Convert model ID to safe filename."""
    return model_id.replace("/", "--").replace(":", "-") + ".json"


def main():
    parser = argparse.ArgumentParser(
        description="Generate llmcore model cards from Mistral AI /v1/models API"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Mistral API key (default: $MISTRAL_API_KEY)",
    )
    parser.add_argument(
        "--include-deprecated",
        action="store_true",
        default=False,
        help="Include deprecated/legacy models",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("ERROR: MISTRAL_API_KEY not set. Pass --api-key or set the env var.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = fetch_models(api_key)

    count = 0
    skipped = 0
    for model in models:
        model_id = model.get("id", "")
        if not model_id:
            continue

        # Skip deprecated unless requested
        if not args.include_deprecated and model.get("deprecation"):
            skipped += 1
            continue

        card = model_to_card(model)
        filename = sanitize_filename(model_id)
        filepath = output_dir / filename

        with open(filepath, "w") as f:
            json.dump(card, f, indent=2)
            f.write("\n")
        count += 1
        print(f"  {model_id} → {filename}")

    # Write __init__.py
    init_path = output_dir / "__init__.py"
    if not init_path.exists():
        init_path.write_text('"""Auto-generated Mistral model cards."""\n')

    print(f"\nGenerated {count} model cards in {output_dir}/")
    if skipped:
        print(f"Skipped {skipped} deprecated models (use --include-deprecated to include)")


if __name__ == "__main__":
    main()
