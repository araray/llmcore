# examples/session_chat.py
"""
Example demonstrating stateful chat interactions using persistent sessions with LLMCore.

This script shows how to:
1. Initialize LLMCore.
2. Create or use a specific session ID for a conversation.
3. Set an initial system message for the session.
4. Send multiple messages within the same session, allowing the LLM to maintain context.
5. Ensure the conversation history is saved to the configured storage backend.
6. Optionally retrieve and inspect the saved session afterwards.

To run this example:
- Ensure you have llmcore installed (`pip install .`).
- Make sure the configured storage backend (default: SQLite at ~/.llmcore/sessions.db) is accessible.
- The default provider (Ollama) should be running, or configure API keys for other providers.
"""

import asyncio
import logging
import uuid # To generate unique session IDs for demonstration

# Import the main class and relevant exceptions
from llmcore import LLMCore, LLMCoreError, ProviderError, ConfigError, SessionNotFoundError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def main():
    """Runs the session chat example."""
    llm = None
    # Generate a unique session ID for this run to avoid conflicts
    # In a real application, you might use user IDs, conversation IDs, etc.
    session_id = f"example_session_{uuid.uuid4()}"
    logger.info(f"Using session ID: {session_id}")

    try:
        # 1. Initialize LLMCore
        logger.info("Initializing LLMCore...")
        # Use async with for automatic resource cleanup (calls llm.close() on exit)
        async with await LLMCore.create() as llm:
            logger.info("LLMCore initialized successfully.")

            # --- Interaction 1: Start the conversation with a system message ---
            prompt1 = "My name is Alex. I'm interested in learning about Large Language Models. Can you give me a brief overview?"
            logger.info(f"\n--- Sending prompt 1 (Session: {session_id}) ---")
            logger.info(f"Alex: {prompt1}")

            response1 = await llm.chat(
                message=prompt1,
                session_id=session_id,
                system_message="You are a helpful AI assistant explaining complex topics simply.",
                save_session=True # Ensure this turn is saved (default is True)
            )
            logger.info(f"LLM: {response1}")

            # --- Interaction 2: Follow-up question ---
            prompt2 = "That's interesting. How does that relate to my name?" # A slightly silly follow-up
            logger.info(f"\n--- Sending prompt 2 (Session: {session_id}) ---")
            logger.info(f"Alex: {prompt2}")

            # The LLM should have access to the previous messages in the session context,
            # including the name "Alex".
            response2 = await llm.chat(
                message=prompt2,
                session_id=session_id,
                save_session=True
                # No need to provide system_message again for existing session
            )
            logger.info(f"LLM: {response2}")

            # --- Interaction 3: Another follow-up ---
            prompt3 = "What was the first topic I asked about?"
            logger.info(f"\n--- Sending prompt 3 (Session: {session_id}) ---")
            logger.info(f"Alex: {prompt3}")

            response3 = await llm.chat(
                message=prompt3,
                session_id=session_id
            )
            logger.info(f"LLM: {response3}")


            # --- Optional: Verify session was saved ---
            logger.info(f"\n--- Verifying session '{session_id}' content ---")
            saved_session = await llm.get_session(session_id)
            if saved_session:
                 logger.info(f"Session '{saved_session.name}' found with {len(saved_session.messages)} messages.")
                 # Log first user message and last assistant message for verification
                 if len(saved_session.messages) > 1:
                      logger.info(f"  First user message content: '{saved_session.messages[1].content[:50]}...'")
                 if len(saved_session.messages) > 0:
                      logger.info(f"  Last assistant message content: '{saved_session.messages[-1].content[:50]}...'")
            else:
                 logger.warning(f"Could not retrieve session '{session_id}' after chat.")


    except SessionNotFoundError as e:
         logger.error(f"Session error: {e}")
    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
    except ProviderError as e:
         logger.error(f"Provider error: {e}")
    except LLMCoreError as e:
        logger.error(f"An LLMCore error occurred: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
    # No finally block needed for llm.close() when using 'async with'

if __name__ == "__main__":
    asyncio.run(main())
