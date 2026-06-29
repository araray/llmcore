# examples/zai_example.py
"""
Example demonstrating the Z.ai (GLM) provider in LLMCore.

This script shows how to:
1. Initialize LLMCore with a Z.ai provider configured at runtime.
2. Send a chat request to a GLM model (default: ``glm-5.2``).
3. Toggle GLM "thinking" mode and ``reasoning_effort`` per request.
4. Stream a response token-by-token.

To run this example:
- Ensure you have llmcore installed (`pip install .` from the project root)
  along with the `openai` SDK (`pip install openai`).
- Set the Z.ai API key:
    export ZAI_API_KEY='your-key-here'
- For mainland-China users, set ``region = "china"`` (uses the
  open.bigmodel.cn endpoint) in the provider config below.

Docs: https://docs.z.ai/
"""

import asyncio
import logging

from llmcore import ConfigError, LLMCore, LLMCoreError, ProviderError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configure the Z.ai provider at runtime. The API key is picked up from the
# ZAI_API_KEY environment variable by the provider itself.
CONFIG_OVERRIDES = {
    "llmcore": {"default_provider": "zai"},
    "providers": {
        "zai": {
            "default_model": "glm-5.2",
            "thinking": "enabled",
            "reasoning_effort": "high",
            # "region": "china",  # uncomment for open.bigmodel.cn
        }
    },
}


async def main() -> None:
    """Run the Z.ai chat examples."""
    llm = None
    try:
        logger.info("Initializing LLMCore with Z.ai provider...")
        llm = await LLMCore.create(config_overrides=CONFIG_OVERRIDES)
        logger.info("LLMCore initialized successfully.")

        # --- Example 1: Standard chat (thinking enabled by default) ---
        prompt1 = "In one sentence, what makes the GLM models distinctive?"
        logger.info(f"\n--- Prompt 1 (GLM-5.2, thinking on): '{prompt1}' ---")
        response1 = await llm.chat(prompt1, provider_name="zai")
        logger.info(f"GLM Response 1:\n{response1}")

        # --- Example 2: Disable thinking, lower latency ---
        prompt2 = "List three prime numbers."
        logger.info(f"\n--- Prompt 2 (thinking off): '{prompt2}' ---")
        response2 = await llm.chat(
            prompt2,
            provider_name="zai",
            thinking="disabled",
            temperature=0.7,
        )
        logger.info(f"GLM Response 2:\n{response2}")

        # --- Example 3: Streaming with maximum reasoning effort ---
        prompt3 = "Explain why the sky is blue, briefly."
        logger.info(f"\n--- Prompt 3 (streaming, reasoning_effort=max): '{prompt3}' ---")
        async for chunk in await llm.chat(
            prompt3,
            provider_name="zai",
            reasoning_effort="max",
            stream=True,
        ):
            print(chunk, end="", flush=True)
        print()

    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
    except ProviderError as e:
        logger.error(f"Z.ai provider error (is ZAI_API_KEY set?): {e}")
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
