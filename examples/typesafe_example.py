# examples/typesafe_example.py
"""
Example demonstrating the TypeSafe.ai (System One) provider in LLMCore.

TypeSafe is NOT a chat model: you send a `state` (text or JSON) plus typed
questions and get structured, calibrated answers back. This script shows:

1. Initializing LLMCore with the TypeSafe provider configured at runtime.
2. Asking Noul / Choice / Score questions about a support ticket with
   ``provider.system_one()`` and reading the typed answers.
3. Confidence-gated routing (act / confirm / escalate) driven by the answer.
4. Speculative fan-out: many questions in ONE request, consumed selectively.
5. The chat bridge: ``llm.chat(..., provider_name="typesafe", questions=...)``
   returning the answers as a JSON string.
6. Listing the models/aliases your account can use.

To run this example:
- Install with the TypeSafe extra: ``pip install llmcore[typesafe]`` (httpx only).
- Set the API key (or ``set -a; source /av/data/dbs/.env; set +a``):
    export TYPESAFE_API_KEY='your-key-here'

Docs: https://docs.typesafe.ai  ·  llmcore guide: docs/TypeSafe_provider_usage.md
"""

import asyncio
import json
import logging

from llmcore import ConfigError, LLMCore, LLMCoreError, ProviderError
from llmcore.providers.typesafe_provider import Choice, Noul, Score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# The API key is picked up from TYPESAFE_API_KEY by the provider itself.
CONFIG_OVERRIDES = {
    "llmcore": {"default_provider": "typesafe"},
    "providers": {
        "typesafe": {
            "default_model": "jev-latest",  # alias -> jev-1.13.0; pin the version if you tune thresholds
            "timeout": 30,
            "max_retries": 2,
        }
    },
}

TICKET = {
    "subject": "Charged twice this month",
    "body": (
        "I see two charges of $49 on my card and I only have one account. "
        "I've already emailed twice. Please fix this ASAP or I'm cancelling."
    ),
    "customer": {"plan": "pro", "tenure_months": 14},
}

# Keep the questions and thresholds together so they are easy to review.
TRIAGE_QUESTIONS = {
    "department": Choice(
        instructions="Which team should handle this ticket?",
        criteria={
            "billing": "Payments, invoices, refunds, duplicate charges",
            "technical": "Bugs, outages, integrations",
            "sales": "Pricing, upgrades, new accounts",
            "other": "Nothing above fits",
        },
    ),
    "frustration": Score(
        instructions="How frustrated is the customer?",
        criteria=[
            "Calm, just stating facts",
            "Frustrated but civil",
            "Very angry, strong language",
        ],
    ),
    "is_urgent": Noul(
        instructions="Does the ticket convey urgency or time-sensitivity?",
        criteria={"true": "Explicitly time-sensitive", "false": "No urgency expressed"},
    ),
    "churn_risk": Noul(instructions="Does the customer threaten to cancel or leave?"),
}
AUTO_ROUTE_CONFIDENCE = 0.75  # act without a human above this
REVIEW_CONFIDENCE = 0.5  # below this, escalate instead of guessing


def route(department, is_urgent: float, churn_risk: float) -> str:
    """Code owns the policy; the model supplies the judgments."""
    if department.confidence < REVIEW_CONFIDENCE:
        return "escalate: unsure which team (confidence %.2f)" % department.confidence
    queue = department.choice
    if churn_risk > 0.7 or is_urgent > 0.8:
        queue += "-priority"
    if department.confidence < AUTO_ROUTE_CONFIDENCE:
        return f"route to {queue} but flag for review (confidence {department.confidence:.2f})"
    return f"route to {queue}"


async def main() -> None:
    """Run the TypeSafe examples."""
    llm = None
    try:
        logger.info("Initializing LLMCore with the TypeSafe provider...")
        llm = await LLMCore.create(config_overrides=CONFIG_OVERRIDES)
        provider = llm._provider_manager.get_provider("typesafe")

        # --- Example 1: typed triage in ONE request -----------------------
        result = await provider.system_one(TICKET, TRIAGE_QUESTIONS)
        print(f"\nModel that answered: {result.model}   request_id={result.request_id}")
        print(f"Usage: {result.usage.input_tokens} input tokens (output tokens are free)\n")

        dept = result.choices["department"]
        print(f"department  -> {dept.choice!r}  confidence={dept.confidence:.2f}")
        for option, p in sorted(dept.probabilities.items(), key=lambda kv: -kv[1]):
            print(f"              {option:<10} {p:.2f}")

        frus = result.scores["frustration"]
        print(
            f"frustration -> {frus.score:.2f}  ({frus.legend[round(frus.score)]})  confidence={frus.confidence:.2f}"
        )
        print(f"is_urgent   -> {result.nouls['is_urgent'].noul:.2f}")
        print(f"churn_risk  -> {result.nouls['churn_risk'].noul:.2f}")

        # --- Example 2: confidence-gated routing ----------------------------
        decision = route(dept, result.nouls["is_urgent"].noul, result.nouls["churn_risk"].noul)
        print(f"\nRouting decision: {decision}")

        # --- Example 3: speculative fan-out ---------------------------------
        # Ask branch-specific questions up front; only read the ones that apply.
        fanout = await provider.system_one(
            TICKET,
            {
                "refund_requested": Noul(instructions="Is the customer asking for a refund?"),
                "amount_disputed": Choice(
                    instructions="Which amount is disputed? Pick 'unclear' if not stated.",
                    criteria={"49": None, "98": None, "unclear": None},
                ),
                "needs_engineering": Noul(instructions="Does resolving this require an engineer?"),
            },
        )
        if fanout.nouls["refund_requested"].noul > 0.5:
            print(f"Refund flow: disputed amount = {fanout.choices['amount_disputed'].choice}")
        else:
            print("No refund requested; skipping the refund branch.")

        # --- Example 4: the chat bridge --------------------------------------
        answer = await llm.chat(
            TICKET["body"],
            provider_name="typesafe",
            questions={
                "billing": Noul(instructions="Is this about billing?"),
                "tone": Choice(
                    instructions="Tone?", criteria={"calm": None, "frustrated": None, "angry": None}
                ),
            },
            save_session=False,
        )
        answers = json.loads(answer)  # chat() returns the answers as a JSON string
        print(
            f"\nChat bridge: tone={answers['tone']['choice']} billing={answers['billing']['noul']:.2f}"
        )

        # --- Example 5: available models -------------------------------------
        print("\nModels:")
        for m in await provider.list_models():
            print(f"  {m.name:<12} released {str(m.release_date)[:10]}  {m.description}")

    except ConfigError as e:
        logger.error("Configuration error: %s (is TYPESAFE_API_KEY set?)", e)
    except ProviderError as e:
        logger.error("Provider error (status=%s, retryable=%s): %s", e.status_code, e.retryable, e)
    except LLMCoreError as e:
        logger.error("LLMCore error: %s", e)
    finally:
        if llm:
            await llm.close()


if __name__ == "__main__":
    asyncio.run(main())
