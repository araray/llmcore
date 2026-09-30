# examples/friendli_example.py
"""
Example demonstrating the FriendliAI provider in LLMCore.

This script shows how to:
1. Initialize LLMCore with a Friendli provider configured at runtime.
2. Send a chat request to a Friendli Model APIs model.
3. Control reasoning (``reasoning_effort`` / ``reasoning_budget``) and read the
   parsed chain of thought back out of the response.
4. Stream a response token-by-token.
5. Call a tool and feed the result back for a second turn.
6. Use the Friendli-specific surfaces: catalog discovery, exact tokenization,
   and the Suite team-usage API.

Transport: the provider prefers direct transports — the ``openai`` SDK pointed at
the Friendli base URL (default) or raw ``httpx`` — over the official ``friendli``
SDK, whose generated response models drop ``reasoning_content``.  Pick one
explicitly with ``backend = "openai" | "httpx" | "sdk"``.

To run this example:
- Install with the Friendli extra: ``pip install llmcore[friendli]``
- Set a Friendli Personal API key (https://friendli.ai/suite/~/setting/keys):
    export FRIENDLI_TOKEN='flp_...'         # FRIENDLIAI_API_KEY also works
- Optionally scope requests and billing reads to a team:
    export FRIENDLI_TEAM_ID='...'           # FRIENDLIAI_TEAM_ID also works

NOTE: Friendli Model APIs rate limits are tier-based and tier 0 allows only a
couple of requests per minute, so this example paces its calls.

Docs: https://friendli.ai/docs/llms.txt
"""

import asyncio
import json
import logging

from llmcore import ConfigError, LLMCore, LLMCoreError, ProviderError
from llmcore.models import Message, Role, Tool

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Seconds to wait between chat calls so a low usage tier does not 429.
PACE_SECONDS = 35

# Configure the Friendli provider at runtime.  The API key is picked up from
# FRIENDLI_TOKEN / FRIENDLIAI_API_KEY by the provider itself.
CONFIG_OVERRIDES = {
    "llmcore": {"default_provider": "friendli"},
    "providers": {
        "friendli": {
            # Model APIs catalog id.  For a Dedicated Endpoint use the endpoint
            # ID here and set endpoint_type = "dedicated".
            "default_model": "zai-org/GLM-5.3-Flash",
            "endpoint_type": "serverless",
            "parse_reasoning": True,
            # "backend": "openai",       # "openai" (default) | "httpx" | "sdk"
            # "reasoning_effort": "high",
        }
    },
}


async def main() -> None:
    """Run the Friendli examples."""
    llm = None
    try:
        logger.info("Initializing LLMCore with the Friendli provider...")
        llm = await LLMCore.create(config_overrides=CONFIG_OVERRIDES)
        friendli = llm._provider_manager.get_provider("friendli")

        # --- Example 1: Catalog discovery (no generation, no rate-limit cost) ---
        logger.info("\n--- Model APIs catalog ---")
        for details in await friendli.get_models_details():
            pricing = details.metadata.get("pricing") or {}
            logger.info(
                "  %-28s ctx=%-9s reasoning=%-5s in=%s",
                details.id,
                details.context_length,
                details.supports_reasoning,
                pricing.get("input"),
            )

        # --- Example 2: Standard chat ---
        prompt1 = "In one sentence, what makes the Friendli Engine fast?"
        logger.info(f"\n--- Prompt 1: '{prompt1}' ---")
        response1 = await llm.chat(prompt1, provider_name="friendli")
        logger.info(f"Friendli Response 1:\n{response1}")

        # --- Example 3: Reasoning controls + parsed chain of thought ---
        await asyncio.sleep(PACE_SECONDS)
        logger.info("\n--- Prompt 2 (reasoning_effort=high, budget capped) ---")
        resp2 = await friendli.chat_completion(
            [Message(role=Role.USER, content="Is 8191 prime? Think it through.")],
            reasoning_effort="high",
            reasoning_budget=2000,
            max_tokens=1024,
        )
        logger.info(f"Answer:\n{friendli.extract_response_content(resp2)}")
        reasoning = friendli.extract_reasoning_content(resp2) or ""
        logger.info(f"Reasoning ({len(reasoning)} chars):\n{reasoning[:400]}")
        logger.info(f"Usage: {friendli.extract_usage_details(resp2)}")

        # --- Example 4: Streaming ---
        await asyncio.sleep(PACE_SECONDS)
        prompt3 = "Explain speculative decoding in two sentences."
        logger.info(f"\n--- Prompt 3 (streaming): '{prompt3}' ---")
        async for chunk in await llm.chat(prompt3, provider_name="friendli", stream=True):
            print(chunk, end="", flush=True)
        print()

        # --- Example 5: Tool calling round-trip ---
        await asyncio.sleep(PACE_SECONDS)
        logger.info("\n--- Tool calling ---")
        weather = Tool(
            name="get_weather",
            description="Get the current weather for a city.",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
        messages = [Message(role=Role.USER, content="What's the weather in Lisbon?")]
        tool_resp = await friendli.chat_completion(
            messages, tools=[weather], tool_choice="required", max_tokens=512
        )
        calls = friendli.extract_tool_calls(tool_resp)
        logger.info(f"Tool calls: {[(c.name, c.arguments) for c in calls]}")

        if calls:
            call = calls[0]
            await asyncio.sleep(PACE_SECONDS)
            follow_up = [
                *messages,
                Message(
                    role=Role.ASSISTANT,
                    content="",
                    tool_calls=[
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                    ],
                ),
                Message(role=Role.TOOL, content="18C and sunny", tool_call_id=call.id),
            ]
            final = await friendli.chat_completion(follow_up, max_tokens=256)
            logger.info(f"After the tool result:\n{friendli.extract_response_content(final)}")

        # --- Example 6: Exact tokenization with the model's own tokenizer ---
        logger.info("\n--- Native tokenizer ---")
        tokens = await friendli.tokenize("What is generative AI?")
        logger.info(f"Token IDs: {tokens} ({len(tokens)} tokens)")

        # --- Example 7: Friendli Suite team usage (needs a team ID) ---
        logger.info("\n--- Team usage (Friendli Suite) ---")
        try:
            usage = await friendli.get_team_usage(
                "2026-09-01T00:00:00Z", "2026-09-20T00:00:00Z", limit=3
            )
            logger.info(f"Usage buckets: {len(usage.get('data', []))}")
        except ProviderError as e:
            logger.warning(f"Team usage unavailable: {e}")

    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
    except ProviderError as e:
        logger.error(f"Friendli provider error (is FRIENDLI_TOKEN set?): {e}")
    except LLMCoreError as e:
        logger.error(f"An LLMCore error occurred: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
    finally:
        if llm:
            logger.info("Closing LLMCore resources...")
            await llm.close()


if __name__ == "__main__":
    asyncio.run(main())
