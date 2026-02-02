# examples/gemini_example.py
"""
Example demonstrating LLMCore capabilities using the Google Gemini provider.

This script shows how to:
1. Initialize LLMCore.
2. Perform simple, streaming, and session-based chat using Gemini models.
3. Use Retrieval Augmented Generation (RAG) with Gemini.
4. Handle potential errors.

Prerequisites:
- Install LLMCore with Gemini support: `pip install llmcore[gemini,all]` (or relevant extras).
- Configure your Google AI API key:
    - Set the environment variable `LLMCORE_PROVIDERS__GEMINI__API_KEY="YOUR_API_KEY"`.
    - Or add it to your `~/.config/llmcore/config.toml` under `[providers.gemini]`.
- Ensure an embedding model and vector store are configured for RAG (see config file).
"""

import asyncio
import logging
import os
import uuid

# Import the main class and relevant exceptions/models
from llmcore import (
    ConfigError,
    ContextLengthError,
    EmbeddingError,
    LLMCore,
    LLMCoreError,
    ProviderError,
    SessionNotFoundError,
    VectorStorageError,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Configuration Check ---
# Basic check if the API key might be missing (optional but helpful)
if not os.environ.get("LLMCORE_PROVIDERS__GEMINI__API_KEY") and not os.environ.get(
    "GOOGLE_API_KEY"
):  # SDK also checks GOOGLE_API_KEY
    logger.warning(
        "Google AI API key environment variable (LLMCORE_PROVIDERS__GEMINI__API_KEY or GOOGLE_API_KEY) "
        "not detected. Ensure it's set or configured in TOML for the script to run."
    )
# --- End Configuration Check ---


async def main():
    """Runs the Gemini provider example."""
    llm = None
    # Use unique IDs/names for this run
    session_id = f"gemini_session_{uuid.uuid4().hex[:8]}"
    rag_collection_name = f"gemini_rag_docs_{uuid.uuid4().hex[:8]}"
    provider_name = "gemini"  # Explicitly specify the provider

    # Sample documents for RAG specific to this example
    rag_documents = [
        {
            "id": "gemini_doc_1",
            "content": "Gemini is a family of multimodal large language models developed by Google AI.",
            "metadata": {"source": "gemini_example"},
        },
        {
            "id": "gemini_doc_2",
            "content": "The Gemini 1.5 Pro model features a large context window, up to 1 million tokens.",
            "metadata": {"source": "gemini_example"},
        },
    ]

    try:
        # 1. Initialize LLMCore
        logger.info("Initializing LLMCore...")
        # Use async with for automatic resource cleanup
        async with await LLMCore.create() as llm:
            logger.info(f"LLMCore initialized. Using provider: {provider_name}")

            # --- Simple Chat (Stateless, Non-Streaming) ---
            logger.info("\n--- 1. Simple Chat ---")
            prompt1 = "What are the main capabilities of the Gemini models?"
            logger.info(f"User: {prompt1}")
            try:
                response1 = await llm.chat(
                    message=prompt1,
                    provider_name=provider_name,
                    # model_name="gemini-1.5-flash-latest" # Optionally specify model
                    # Pass provider-specific kwargs if needed, e.g., safety settings
                    # safety_settings={...}
                )
                logger.info(f"Gemini: {response1}")
            except (ProviderError, ConfigError, ContextLengthError) as e:
                logger.error(f"Simple chat failed: {e}")

            # --- Streaming Chat ---
            logger.info("\n--- 2. Streaming Chat ---")
            prompt2 = "Write a short poem about space exploration."
            logger.info(f"User: {prompt2}")
            print("Gemini (Streaming): ", end="", flush=True)
            try:
                response_stream = await llm.chat(
                    message=prompt2, provider_name=provider_name, stream=True
                )
                async for chunk in response_stream:
                    print(chunk, end="", flush=True)
                print("\n--- Stream finished ---")
            except (ProviderError, ConfigError, ContextLengthError) as e:
                print(f"\n--- STREAM ERROR: {e} ---", flush=True)
                logger.error(f"Streaming chat failed: {e}")

            # --- Session Chat (Stateful) ---
            logger.info("\n--- 3. Session Chat ---")
            logger.info(f"Using session ID: {session_id}")
            prompt3_1 = "My favorite color is blue. What is the color of the sky on Mars?"
            logger.info(f"User (Turn 1): {prompt3_1}")
            try:
                response3_1 = await llm.chat(
                    message=prompt3_1,
                    session_id=session_id,
                    provider_name=provider_name,
                    system_message="You are a helpful assistant knowledgeable about space and colors.",
                    # save_session=True is default
                )
                logger.info(f"Gemini (Turn 1): {response3_1}")

                prompt3_2 = "Does its sky color relate to my favorite color?"
                logger.info(f"User (Turn 2): {prompt3_2}")
                response3_2 = await llm.chat(
                    message=prompt3_2,
                    session_id=session_id,  # Continue the same session
                    provider_name=provider_name,
                )
                logger.info(f"Gemini (Turn 2): {response3_2}")

            except (ProviderError, ConfigError, ContextLengthError, SessionNotFoundError) as e:
                logger.error(f"Session chat failed: {e}")

            # --- RAG Chat ---
            logger.info("\n--- 4. RAG Chat ---")
            # a) Add documents to vector store
            logger.info(f"Adding documents to RAG collection: {rag_collection_name}")
            try:
                added_ids = await llm.add_documents_to_vector_store(
                    documents=rag_documents, collection_name=rag_collection_name
                )
                logger.info(f"Added documents with IDs: {added_ids}")
                await asyncio.sleep(1)  # Give vector store a moment
            except (VectorStorageError, EmbeddingError, ConfigError) as e:
                logger.error(f"Failed to add RAG documents: {e}. Skipping RAG chat.")
            else:
                # b) Chat with RAG enabled
                prompt4 = "Based on the documents, what is a key feature of Gemini 1.5 Pro?"
                logger.info(f"User (RAG): {prompt4}")
                try:
                    response4 = await llm.chat(
                        message=prompt4,
                        provider_name=provider_name,
                        enable_rag=True,
                        rag_collection_name=rag_collection_name,
                        rag_retrieval_k=1,
                        system_message="Answer based *only* on the provided context documents.",
                    )
                    logger.info(f"Gemini (RAG): {response4}")
                except (
                    ProviderError,
                    ConfigError,
                    ContextLengthError,
                    VectorStorageError,
                    EmbeddingError,
                ) as e:
                    logger.error(f"RAG chat failed: {e}")

    except ConfigError as e:
        logger.error(f"Initialization failed due to configuration error: {e}")
    except LLMCoreError as e:
        logger.error(f"An LLMCore error occurred: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
    # llm.close() is handled by 'async with'


if __name__ == "__main__":
    asyncio.run(main())
