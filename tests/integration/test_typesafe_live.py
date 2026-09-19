# tests/integration/test_typesafe_live.py
"""Live smoke tests for the TypeSafe.ai provider (opt-in; needs ``TYPESAFE_API_KEY``).

Skipped automatically when the key is absent (CI has none). Run locally with::

    set -a; source /av/data/dbs/.env; set +a
    pytest tests/integration/test_typesafe_live.py -m integration -q

Each run makes three small requests (~500 input tokens each, output is free).
"""

from __future__ import annotations

import json
import os

import pytest

from llmcore import LLMCore
from llmcore.providers.typesafe_provider import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    TypeSafeProvider,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("TYPESAFE_API_KEY"), reason="TYPESAFE_API_KEY not set"),
]

STATE = {
    "subject": "Charged twice this month",
    "body": "I see two charges of $49. I only have one account. Please fix this ASAP.",
}
QUESTIONS = {
    "billing": Noul(
        instructions="Is this ticket about billing?",
        criteria={"true": "Payments or invoices", "false": "Anything else"},
    ),
    "tone": Choice(
        instructions="What is the customer's tone?",
        criteria={"calm": None, "frustrated": None, "angry": None},
    ),
    "urgency": Score(
        instructions="How urgent is this ticket?",
        criteria=["can wait", "this week", "today"],
    ),
}


@pytest.fixture
async def provider():
    p = TypeSafeProvider({"default_model": "jev-latest", "timeout": 30, "max_retries": 2})
    try:
        yield p
    finally:
        await p.close()


async def test_live_list_models(provider):
    models = await provider.list_models()
    names = {m.name for m in models}
    assert "jev-latest" in names
    for m in models:
        assert m.description and m.release_date


async def test_live_system_one(provider):
    result = await provider.system_one(STATE, QUESTIONS)
    assert result.model.startswith("jev-")
    assert result.model != "jev-latest"  # the API echoes the versioned id
    assert result.request_id
    assert result.usage.input_tokens and result.usage.input_tokens > 0
    assert result.usage.output_tokens is not None and result.usage.output_tokens >= 0

    billing = result.nouls["billing"]
    assert isinstance(billing, NoulAnswer) and 0 <= billing.noul <= 1
    assert billing.noul > 0.5  # clearly a billing ticket

    tone = result.choices["tone"]
    assert isinstance(tone, ChoiceAnswer)
    assert tone.choice in {"calm", "frustrated", "angry"}
    assert set(tone.probabilities) == {"calm", "frustrated", "angry"}
    assert sum(tone.probabilities.values()) == pytest.approx(1, abs=0.1)
    assert 0 <= tone.confidence <= 1

    urgency = result.scores["urgency"]
    assert isinstance(urgency, ScoreAnswer)
    assert 0 <= urgency.score <= 2
    assert urgency.legend == {0: "can wait", 1: "this week", 2: "today"}
    assert set(urgency.probabilities) == {0, 1, 2}
    assert sum(urgency.probabilities.values()) == pytest.approx(1, abs=0.1)

    # Budget helpers stay consistent with the card.
    assert provider.get_max_context_length(result.model) == 65536


async def test_live_chat_bridge_via_llmcore(tmp_path):
    overrides = {
        "llmcore": {"default_provider": "typesafe"},
        "providers": {"typesafe": {"default_model": "jev-latest", "timeout": 30}},
        "storage": {"session_backend": {"type": "sqlite", "db_path": str(tmp_path / "s.db")}},
    }
    llm = await LLMCore.create(config_overrides=overrides)
    try:
        answer = await llm.chat(
            "I see two charges of $49 this month and I only have one account. Please fix this ASAP.",
            provider_name="typesafe",
            questions={
                "billing": Noul(instructions="Is this about billing?"),
                "tone": Choice(instructions="Tone?", criteria={"calm": None, "frustrated": None}),
            },
            save_session=False,
        )
    finally:
        await llm.close()
    parsed = json.loads(answer)
    assert set(parsed) == {"billing", "tone"}
    assert parsed["billing"]["type"] == "noul" and 0 <= parsed["billing"]["noul"] <= 1
    assert parsed["tone"]["choice"] in {"calm", "frustrated"}
