#!/usr/bin/env python3
"""
Generate llmcore model cards from the OpenRouter /models API.

Usage:
    python tools/generate_openrouter_cards.py [--output-dir DIR] [--top N]

This script:
1. Fetches all models from https://openrouter.ai/api/v1/models
2. Converts each model to an llmcore ModelCard JSON file
3. Writes to src/llmcore/model_cards/default_cards/openrouter/

No API key required — the /models endpoint is public.
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

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
DEFAULT_OUTPUT_DIR = "src/llmcore/model_cards/default_cards/openrouter"


def fetch_models() -> list[dict]:
    """Fetch all models from OpenRouter API."""
    print(f"Fetching models from {OPENROUTER_MODELS_URL}...")
    resp = httpx.get(OPENROUTER_MODELS_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    models = data.get("data", [])
    print(f"  Fetched {len(models)} models.")
    return models


def model_to_card(model: dict) -> dict:
    """Convert an OpenRouter model dict to an llmcore ModelCard dict."""
    model_id = model.get("id", "")
    architecture = model.get("architecture", {}) or {}
    pricing = model.get("pricing", {}) or {}
    top_provider = model.get("top_provider", {}) or {}
    per_request_limits = model.get("per_request_limits", {}) or {}
    supported_params = model.get("supported_parameters", [])
    input_mods = architecture.get("input_modalities", [])
    output_mods = architecture.get("output_modalities", [])

    # Parse prices (OpenRouter returns per-token strings)
    input_price_per_token = float(pricing.get("prompt", "0") or "0")
    output_price_per_token = float(pricing.get("completion", "0") or "0")
    input_price_per_m = input_price_per_token * 1_000_000
    output_price_per_m = output_price_per_token * 1_000_000

    # Context length
    context_length = model.get("context_length") or 4096
    max_output = (
        top_provider.get("max_completion_tokens")
        or per_request_limits.get("completion_tokens")
    )

    # Determine model type
    model_type = "chat"
    if "embeddings" in output_mods and "text" not in output_mods:
        model_type = "embedding"
    elif "image" in output_mods and "text" not in output_mods:
        model_type = "image-generation"
    elif "image" in output_mods:
        model_type = "multimodal"

    # Build capabilities
    capabilities = {
        "streaming": True,
        "function_calling": "tools" in supported_params,
        "tool_use": "tools" in supported_params,
        "json_mode": "response_format" in supported_params,
        "structured_output": "structured_outputs" in supported_params,
        "vision": "image" in input_mods,
        "audio_input": "audio" in input_mods,
        "audio_output": "audio" in output_mods,
        "video_input": "video" in input_mods,
        "image_generation": "image" in output_mods,
        "web_search": "web_search_options" in supported_params,
        "reasoning": (
            "reasoning" in supported_params
            or "include_reasoning" in supported_params
        ),
        "file_processing": "file" in input_mods,
    }

    # Build pricing (skip if free)
    pricing_dict = None
    if input_price_per_m > 0 or output_price_per_m > 0:
        pricing_dict = {
            "currency": "USD",
            "per_million_tokens": {
                "input": round(input_price_per_m, 4),
                "output": round(output_price_per_m, 4),
            },
        }
        # Add cached pricing if available
        cache_read = pricing.get("input_cache_read")
        if cache_read and float(cache_read or "0") > 0:
            pricing_dict["per_million_tokens"]["cached_input"] = round(
                float(cache_read) * 1_000_000, 4
            )
        # Add reasoning pricing if available
        reasoning_price = pricing.get("internal_reasoning")
        if reasoning_price and float(reasoning_price or "0") > 0:
            pricing_dict["per_million_tokens"]["reasoning_output"] = round(
                float(reasoning_price) * 1_000_000, 4
            )

    # Build lifecycle
    lifecycle = {"status": "active"}
    knowledge_cutoff = model.get("knowledge_cutoff")
    if knowledge_cutoff:
        lifecycle["knowledge_cutoff"] = knowledge_cutoff
    expiration = model.get("expiration_date")
    if expiration:
        lifecycle["deprecation_date"] = expiration
        lifecycle["status"] = "deprecated"

    # Determine family from model ID
    family = None
    slug = model.get("canonical_slug", model_id)
    if "/" in slug:
        family = slug.split("/")[0].title()

    card = {
        "model_id": model_id,
        "display_name": model.get("name", model_id),
        "provider": "openrouter",
        "model_type": model_type,
        "architecture": {
            "family": family,
        },
        "context": {
            "max_input_tokens": int(context_length),
        },
        "capabilities": capabilities,
        "lifecycle": lifecycle,
        "license": None,
        "open_weights": False,
        "aliases": [],
        "description": model.get("description", ""),
        "tags": [],
        "source": "builtin",
    }

    if max_output:
        card["context"]["max_output_tokens"] = int(max_output)

    if pricing_dict:
        card["pricing"] = pricing_dict

    # Add OpenRouter-specific extension
    card["provider_extension"] = {
        "canonical_slug": model.get("canonical_slug"),
        "input_modalities": input_mods,
        "output_modalities": output_mods,
        "supported_parameters": supported_params,
        "is_moderated": top_provider.get("is_moderated", False),
        "instruct_type": architecture.get("instruct_type"),
        "tokenizer": architecture.get("tokenizer"),
    }

    return card


def sanitize_filename(model_id: str) -> str:
    """Convert model ID to safe filename."""
    return model_id.replace("/", "--").replace(":", "-") + ".json"


def main():
    parser = argparse.ArgumentParser(
        description="Generate llmcore model cards from OpenRouter API"
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = fetch_models()

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
            json.dump(card, f, indent=2)
        count += 1

    print(f"Generated {count} model cards in {output_dir}/")
    # Write __init__.py
    init_path = output_dir / "__init__.py"
    if not init_path.exists():
        init_path.write_text("")


if __name__ == "__main__":
    main()
