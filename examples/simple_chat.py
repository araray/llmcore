# examples/simple_chat.py
"""
Example demonstrating simple, stateless chat interactions using LLMCore.

This script shows how to:
1. Initialize LLMCore using default configuration.
2. Send a single chat message using the default provider (e.g., Ollama).
3. Send another message using a specifically chosen provider (e.g., OpenAI),
   demonstrating provider overrides.
4. Handle potential errors and ensure resources are closed.

To run this example:
- Ensure you have llmcore installed (`pip install .` from the project root).
- For the default Ollama provider: Make sure Ollama is running locally.
- For the OpenAI provider part: Set the `LLMCORE_PROVIDERS__OPENAI__API_KEY` environment variable.
  (e.g., `export LLMCORE_PROVIDERS__OPENAI__API_KEY='your-key-here'`)
"""

import asyncio
import logging

# Import the main class and exceptions from llmcore
from llmcore import LLMCore, LLMCoreError, ProviderError, ConfigError

# Configure logging for better visibility (optional)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def main():
    """Runs the simple chat examples."""
    llm = None # Initialize llm to None for the finally block
    try:
        # 1. Initialize LLMCore
        # This loads configuration from default locations (packaged defaults,
        # ~/.config/llmcore/config.toml, .env file, environment variables).
        # It uses asynchronous initialization.
        logger.info("Initializing LLMCore...")
        llm = await LLMCore.create()
        logger.info("LLMCore initialized successfully.")

        # --- Example 1: Simple Chat with Default Provider ---
        prompt1 = "What is the distance between the Earth and the Moon in kilometers?"
        logger.info(f"\n--- Sending prompt 1 (Default Provider): '{prompt1}' ---")

        # Call chat without a session_id for a stateless interaction.
        # stream=False (default) waits for the full response.
        response1 = await llm.chat(prompt1)
        logger.info(f"LLM Response 1:\n{response1}")

        # --- Example 2: Chat with Specific Provider and Model ---
        prompt2 = "Explain the concept of asynchronous programming in Python concisely."
        logger.info(f"\n--- Sending prompt 2 (OpenAI Provider): '{prompt2}' ---")

        try:
            # Override provider and model, pass provider-specific args like temperature.
            # This requires the OpenAI provider to be configured (e.g., via env var API key).
            response2 = await llm.chat(
                message=prompt2,
                provider_name="openai", # Explicitly choose OpenAI
                model_name="gpt-4o", # Use a specific OpenAI model
                temperature=0.6      # Pass optional parameter to the provider
            )
            logger.info(f"LLM Response 2 (OpenAI):\n{response2}")
        except (ConfigError, ProviderError) as e:
            # Handle cases where the provider might not be configured or API key is missing
            logger.warning(f"Could not chat with OpenAI. Is it configured correctly? Error: {e}")
        except LLMCoreError as e:
            logger.error(f"An LLMCore error occurred during OpenAI chat: {e}")


    except ConfigError as e:
        logger.error(f"Configuration error during initialization: {e}")
    except ProviderError as e:
         logger.error(f"Provider error during initialization or chat: {e}")
    except LLMCoreError as e:
        logger.error(f"An LLMCore error occurred: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}") # Log full traceback
    finally:
        # Ensure resources (like network connections) are closed gracefully.
        if llm:
            logger.info("Closing LLMCore resources...")
            await llm.close()
            logger.info("LLMCore resources closed.")

if __name__ == "__main__":
    # Run the main asynchronous function
    asyncio.run(main())
