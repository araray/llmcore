# examples/ollama_example.py
"""
Example demonstrating LLMCore capabilities using a local Ollama provider.

This script shows how to:
1. Initialize LLMCore (ensure Ollama provider is configured or default).
2. Perform simple, streaming, and session-based chat using Ollama models.
3. Use Retrieval Augmented Generation (RAG) with Ollama.
4. Handle potential errors (e.g., Ollama server not running, model not found).

Prerequisites:
- Install LLMCore with Ollama support: `pip install llmcore[ollama,all]` (or relevant extras).
- Have Ollama installed and running: https://ollama.com/
- Pull the desired model: `ollama pull llama3` (or the model specified in your config/example).
- Ensure an embedding model and vector store are configured for RAG (see config file).
"""

import asyncio
import logging
import uuid
import os

# Import the main class and relevant exceptions/models
from llmcore import (
    LLMCore,
    LLMCoreError,
    ProviderError,
    ConfigError,
    VectorStorageError,
    EmbeddingError,
    ContextLengthError,
    SessionNotFoundError
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Note ---
# By default, LLMCore uses Ollama. Ensure your `~/.config/llmcore/config.toml`
# points to the correct Ollama host if it's not the default (http://localhost:11434)
# Example config section:
# [providers.ollama]
# host = "http://192.168.1.100:11434" # If Ollama is on another machine
# default_model = "mistral"
# --- End Configuration Note ---

async def main():
    """Runs the Ollama provider example."""
    llm = None
    # Use unique IDs/names for this run
    session_id = f"ollama_session_{uuid.uuid4().hex[:8]}"
    rag_collection_name = f"ollama_rag_docs_{uuid.uuid4().hex[:8]}"
    provider_name = "ollama" # Explicitly specify the provider

    # Sample documents for RAG specific to this example
    rag_documents = [
        {"id": "ollama_doc_1", "content": "Ollama allows running large language models locally.", "metadata": {"source": "ollama_example"}},
        {"id": "ollama_doc_2", "content": "Common models served by Ollama include Llama 3, Mistral, and Gemma.", "metadata": {"source": "ollama_example"}},
    ]

    try:
        # 1. Initialize LLMCore
        logger.info("Initializing LLMCore...")
        # Use async with for automatic resource cleanup
        async with await LLMCore.create() as llm:
            # Check if Ollama provider was actually loaded
            if provider_name not in llm.get_available_providers():
                 logger.error(f"Ollama provider ('{provider_name}') failed to load. Is it configured correctly and dependencies installed?")
                 return
            logger.info(f"LLMCore initialized. Using provider: {provider_name}")

            # --- Simple Chat (Stateless, Non-Streaming) ---
            logger.info("\n--- 1. Simple Chat ---")
            prompt1 = "What is the capital of France?"
            logger.info(f"User: {prompt1}")
            try:
                response1 = await llm.chat(
                    message=prompt1,
                    provider_name=provider_name,
                    # model_name="mistral" # Optionally specify a different Ollama model
                    # Pass provider-specific kwargs if needed (Ollama options)
                    # temperature=0.7,
                    # top_p=0.9
                )
                logger.info(f"Ollama: {response1}")
            except ProviderError as e:
                logger.error(f"Simple chat failed: {e}. Is Ollama running and the model available?")
            except (ConfigError, ContextLengthError) as e:
                logger.error(f"Simple chat failed: {e}")

            # --- Streaming Chat ---
            logger.info("\n--- 2. Streaming Chat ---")
            prompt2 = "Write a short haiku about a running server."
            logger.info(f"User: {prompt2}")
            print(f"Ollama (Streaming): ", end="", flush=True)
            try:
                response_stream = await llm.chat(
                    message=prompt2,
                    provider_name=provider_name,
                    stream=True
                )
                async for chunk in response_stream:
                    # Ollama stream chunks often contain the full message object
                    # We extract the content delta
                    delta = chunk.get("message", {}).get("content", "")
                    print(delta, end="", flush=True)
                print("\n--- Stream finished ---")
            except ProviderError as e:
                print(f"\n--- STREAM ERROR: {e} ---", flush=True)
                logger.error(f"Streaming chat failed: {e}. Is Ollama running?")
            except (ConfigError, ContextLengthError) as e:
                print(f"\n--- STREAM ERROR: {e} ---", flush=True)
                logger.error(f"Streaming chat failed: {e}")

            # --- Session Chat (Stateful) ---
            logger.info("\n--- 3. Session Chat ---")
            logger.info(f"Using session ID: {session_id}")
            prompt3_1 = "My name is Kai. What are the benefits of running LLMs locally?"
            logger.info(f"User (Turn 1): {prompt3_1}")
            try:
                response3_1 = await llm.chat(
                    message=prompt3_1,
                    session_id=session_id,
                    provider_name=provider_name,
                    system_message="You are a helpful AI assistant discussing local LLMs."
                )
                logger.info(f"Ollama (Turn 1): {response3_1}")

                prompt3_2 = "How does that relate to my name, Kai?"
                logger.info(f"User (Turn 2): {prompt3_2}")
                response3_2 = await llm.chat(
                    message=prompt3_2,
                    session_id=session_id, # Continue the same session
                    provider_name=provider_name
                )
                logger.info(f"Ollama (Turn 2): {response3_2}")

            except ProviderError as e:
                logger.error(f"Session chat failed: {e}. Is Ollama running?")
            except (ConfigError, ContextLengthError, SessionNotFoundError) as e:
                logger.error(f"Session chat failed: {e}")

            # --- RAG Chat ---
            logger.info("\n--- 4. RAG Chat ---")
            # a) Add documents to vector store
            logger.info(f"Adding documents to RAG collection: {rag_collection_name}")
            try:
                added_ids = await llm.add_documents_to_vector_store(
                    documents=rag_documents,
                    collection_name=rag_collection_name
                )
                logger.info(f"Added documents with IDs: {added_ids}")
                await asyncio.sleep(1) # Give vector store a moment
            except (VectorStorageError, EmbeddingError, ConfigError) as e:
                logger.error(f"Failed to add RAG documents: {e}. Skipping RAG chat.")
            else:
                # b) Chat with RAG enabled
                prompt4 = "Based on the documents, what is Ollama used for?"
                logger.info(f"User (RAG): {prompt4}")
                try:
                    response4 = await llm.chat(
                        message=prompt4,
                        provider_name=provider_name,
                        enable_rag=True,
                        rag_collection_name=rag_collection_name,
                        rag_retrieval_k=1,
                        system_message="Answer based *only* on the provided context documents."
                    )
                    logger.info(f"Ollama (RAG): {response4}")
                except ProviderError as e:
                     logger.error(f"RAG chat failed: {e}. Is Ollama running?")
                except (ConfigError, ContextLengthError, VectorStorageError, EmbeddingError) as e:
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
