# examples/model_cards_and_tokens.py
"""Offline model-card and token-counting example.

This example does not call any provider API. It reads bundled model cards,
checks capabilities/context windows, estimates cost, and counts prompt tokens
through llmcore's shared token-counter interface.

Run:
    pip install -e .
    python examples/model_cards_and_tokens.py
"""

from __future__ import annotations

import logging

from llmcore import count_tokens, get_model_card, get_model_card_registry


def main() -> None:
    # Keep the example output focused even if optional bundled cards fail validation.
    logging.getLogger("llmcore.model_cards.registry").setLevel(logging.CRITICAL)

    registry = get_model_card_registry()

    card = get_model_card("openai", "gpt-4o")
    if card is None:
        raise RuntimeError("Bundled OpenAI gpt-4o model card was not found.")

    print("=== Model Card ===")
    print(f"{card.provider}/{card.model_id}: {card.display_name}")
    print(f"context={card.get_context_length():,} tokens")
    print(f"max_output={card.get_max_output():,} tokens")
    print(f"vision={card.supports_capability('vision')}")
    print(f"tool_use={card.supports_capability('tool_use')}")
    print(f"status={card.lifecycle.status}")

    prompt = "Summarize the operational risks of deploying a new LLM provider."
    tokens = count_tokens(prompt, model=card.model_id)
    print("\n=== Token Counting ===")
    print(f"prompt={prompt!r}")
    print(f"estimated tokens={tokens}")

    cost = card.estimate_cost(input_tokens=50_000, output_tokens=2_000, cached_tokens=10_000)
    print("\n=== Cost Estimate ===")
    if cost is None:
        print("No token pricing is available for this card.")
    else:
        print(f"50k input / 2k output / 10k cached input ~= ${cost:.4f}")

    print("\n=== Built-in Vision Models (first 10) ===")
    vision_models = registry.list_cards(tags=["vision"], include_deprecated=False)
    for summary in vision_models[:10]:
        print(f"- {summary.provider}/{summary.model_id}: {summary.display_name}")


if __name__ == "__main__":
    main()
