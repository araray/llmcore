# examples/rag_example.py
"""
Example demonstrating Retrieval Augmented Generation (RAG) with LLMCore.

This script shows how to:
1. Initialize LLMCore (ensure vector storage and embedding model are configured).
2. Add documents to the vector store.
3. Perform a similarity search (optional verification step).
4. Chat with the LLM using RAG enabled (`enable_rag=True`), allowing it
   to use the added documents as context.

To run this example:
- Ensure you have llmcore installed (`pip install .[all]`).
- Configure your `~/.config/llmcore/config.toml` (or use env vars/overrides):
    - Set `storage.vector.type` (e.g., "chromadb") and `storage.vector.path`.
    - Set `llmcore.default_embedding_model` (e.g., "all-MiniLM-L6-v2" or an API-based one).
    - Ensure the chosen LLM provider is configured (e.g., Ollama running, or API keys set).
- Run the script: `python examples/rag_example.py`
"""

import asyncio
import logging
import uuid

# Import the main class and relevant exceptions/models
from llmcore import (
    LLMCore,
    LLMCoreError,
    ProviderError,
    ConfigError,
    VectorStorageError,
    EmbeddingError,
    ContextDocument # Import ContextDocument if needed for direct search result inspection
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def main():
    """Runs the RAG example."""
    llm = None
    # Define a unique collection name for this example run
    collection_name = f"rag_example_collection_{uuid.uuid4().hex[:8]}"
    logger.info(f"Using vector store collection: {collection_name}")

    # Sample documents to add
    documents_to_add = [
        {
            "id": "doc_llmcore_readme", # Optional ID
            "content": "LLMCore is a Python library for interacting with LLMs. It supports providers like OpenAI, Anthropic, Ollama, and Gemini. Key features include session management, context handling, and Retrieval Augmented Generation (RAG).",
            "metadata": {"source": "llmcore/readme.md", "topic": "library_description"}
        },
        {
            "content": "RAG allows LLMs to access external knowledge by retrieving relevant documents from a vector store (like ChromaDB or pgvector) and adding them to the prompt context.",
            "metadata": {"source": "llmcore/spec.md", "topic": "rag_concept"}
        },
        {
            "content": "Configuration in LLMCore is managed by the 'confy' library, supporting defaults, TOML files, environment variables, and overrides.",
            "metadata": {"source": "llmcore/readme.md", "topic": "configuration"}
        },
        {
             "content": "The primary method for chat is `llm.chat()`, which accepts parameters like `message`, `session_id`, `provider_name`, `model_name`, `stream`, and `enable_rag`.",
             "metadata": {"source": "llmcore/usage.md", "topic": "api_usage"}
        }
    ]

    try:
        # 1. Initialize LLMCore
        logger.info("Initializing LLMCore...")
        # Use async with for automatic resource cleanup
        async with await LLMCore.create() as llm:
            logger.info("LLMCore initialized successfully.")

            # --- Step 2: Add documents to the vector store ---
            logger.info(f"\n--- Adding {len(documents_to_add)} documents to collection '{collection_name}' ---")
            try:
                added_ids = await llm.add_documents_to_vector_store(
                    documents=documents_to_add,
                    collection_name=collection_name
                )
                logger.info(f"Successfully added documents with IDs: {added_ids}")
            except (VectorStorageError, EmbeddingError, ConfigError) as e:
                logger.error(f"Failed to add documents: {e}. Ensure vector store and embedding model are configured correctly.")
                return # Cannot proceed without documents

            # Give ChromaDB a moment to process embeddings if needed (optional)
            await asyncio.sleep(1)

            # --- Step 3: Optional - Perform a direct similarity search ---
            search_query = "How is configuration handled?"
            logger.info(f"\n--- Performing direct similarity search for: '{search_query}' ---")
            try:
                search_results = await llm.search_vector_store(
                    query=search_query,
                    k=1, # Get the top 1 result
                    collection_name=collection_name
                )
                if search_results:
                    logger.info("Top search result:")
                    doc = search_results[0]
                    logger.info(f"  ID: {doc.id}")
                    logger.info(f"  Score: {doc.score:.4f}") # Lower score is better in ChromaDB L2
                    logger.info(f"  Content: '{doc.content[:100]}...'")
                    logger.info(f"  Metadata: {doc.metadata}")
                else:
                    logger.info("Direct search returned no results.")
            except (VectorStorageError, EmbeddingError, ConfigError) as e:
                logger.error(f"Direct search failed: {e}")


            # --- Step 4: Chat with RAG enabled ---
            chat_query = "Tell me about LLMCore's configuration system based on the provided documents."
            logger.info(f"\n--- Sending chat message with RAG enabled ---")
            logger.info(f"User: {chat_query}")
            print("\nLLM Response (RAG): ", end="", flush=True) # Use print for streaming output

            try:
                # Call chat with enable_rag=True
                response_stream = await llm.chat(
                    message=chat_query,
                    enable_rag=True,
                    rag_collection_name=collection_name, # Specify the collection
                    rag_retrieval_k=2, # Ask for top 2 relevant docs for context
                    stream=True, # Stream the response
                    # Optional: Provide a system message guiding the LLM
                    system_message="You are an assistant answering questions based ONLY on the retrieved context documents provided. Do not use prior knowledge."
                )

                # Process the streamed response
                async for chunk in response_stream:
                    print(chunk, end="", flush=True)
                print("\n--- RAG chat stream finished ---")

            except (ProviderError, ContextLengthError, VectorStorageError, EmbeddingError, ConfigError) as e:
                 print(f"\n--- RAG CHAT ERROR: {e} ---", flush=True)
                 logger.error(f"Error during RAG chat: {e}")


            # --- Step 5: Ask another question using RAG ---
            chat_query_2 = "What is RAG?"
            logger.info(f"\n--- Sending second chat message with RAG enabled ---")
            logger.info(f"User: {chat_query_2}")
            print("\nLLM Response 2 (RAG): ", end="", flush=True) # Use print for streaming output

            try:
                response_stream_2 = await llm.chat(
                    message=chat_query_2,
                    enable_rag=True,
                    rag_collection_name=collection_name,
                    stream=True,
                    system_message="Answer based ONLY on the retrieved context documents."
                )
                async for chunk in response_stream_2:
                    print(chunk, end="", flush=True)
                print("\n--- RAG chat stream 2 finished ---")
            except (ProviderError, ContextLengthError, VectorStorageError, EmbeddingError, ConfigError) as e:
                 print(f"\n--- RAG CHAT ERROR 2: {e} ---", flush=True)
                 logger.error(f"Error during second RAG chat: {e}")


            # --- Step 6: Clean up (optional) ---
            # logger.info(f"\n--- Deleting documents from collection '{collection_name}' ---")
            # try:
            #     deleted = await llm.delete_documents_from_vector_store(
            #         document_ids=added_ids, # Use the IDs returned earlier
            #         collection_name=collection_name
            #     )
            #     if deleted:
            #         logger.info("Successfully deleted documents.")
            #     else:
            #         logger.warning("Deletion attempt finished, but success status was False.")
            # except Exception as e:
            #      logger.error(f"Error deleting documents: {e}")


    except LLMCoreError as e:
        logger.error(f"An LLMCore error occurred: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
    # No finally block needed for llm.close() when using 'async with'

if __name__ == "__main__":
    asyncio.run(main())
