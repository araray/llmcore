# examples/mcp_example.py
"""
Example demonstrating LLMCore's Model Context Protocol (MCP) capabilities
along with Retrieval Augmented Generation (RAG).

This script shows how to:
1. Initialize LLMCore.
2. Configure LLMCore to use the `GithubMCPProvider`, which inherently uses MCP.
3. Add documents to a vector store for RAG.
4. Perform a chat where the context (including retrieved documents) is
   formatted using MCP.
5. Handle potential errors.

Prerequisites:
- Install LLMCore with MCP and relevant vector store/embedding support:
  `pip install llmcore[mcp,chromadb,sentence_transformers]` (or other RAG dependencies).
- An MCP-compliant server (like `github-mcp-server`) must be accessible at the
  `base_url` specified in the configuration.
- A vector store (e.g., ChromaDB) and an embedding model (e.g., a Sentence
  Transformer model) must be configured in your main LLMCore configuration
  (e.g., `~/.config/llmcore/config.toml` or environment variables).
  Example RAG config snippets for `config.toml`:
  ```toml
  [llmcore]
  default_embedding_model = "all-MiniLM-L6-v2" # Or any other configured model

  [storage.vector]
  type = "chromadb"
  path = "~/.llmcore/mcp_rag_db" # Path for ChromaDB
  default_collection = "mcp_rag_default"
  ```

What is MCP with RAG?
When MCP is used with RAG:
- LLMCore retrieves relevant documents from the vector store.
- These documents are then packaged into the `retrieved_knowledge` field
  of the MCP payload.
- The `GithubMCPProvider` sends this complete MCP object (messages + knowledge)
  to the MCP-compliant server.
"""

import asyncio
import logging
import uuid # For unique collection names

# Import the main class and relevant exceptions
from llmcore import (
    LLMCore,
    LLMCoreError,
    ProviderError,
    ConfigError,
    VectorStorageError,
    EmbeddingError
)
#from llmcore.models import Message, Role # For constructing context if needed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration for GithubMCPProvider ---
MCP_SERVER_BASE_URL = "http://localhost:8080" # Placeholder - REPLACE
PROVIDER_NAME = "github_mcp"

# Example configuration overrides
CONFIG_OVERRIDES = {
    "llmcore.default_provider": PROVIDER_NAME,
    f"providers.{PROVIDER_NAME}.base_url": MCP_SERVER_BASE_URL,
    # Optional: specify the underlying model for token counting if known
    # f"providers.{PROVIDER_NAME}.underlying_model_for_counting": "gpt-4o",
}

# Sample documents for RAG
RAG_DOCUMENTS = [
    {
        "id": "mcp_doc_1",
        "content": "The Model Context Protocol (MCP) provides a standardized way to structure information, including chat messages and retrieved documents, for language models.",
        "metadata": {"source": "mcp_spec_intro"}
    },
    {
        "id": "mcp_doc_2",
        "content": "When using Retrieval Augmented Generation (RAG) with MCP, the retrieved textual passages are typically placed in the 'retrieved_knowledge' field of the MCP payload.",
        "metadata": {"source": "mcp_rag_integration_notes"}
    }
]

async def main():
    """Runs the MCP example with RAG using GithubMCPProvider."""
    llm = None
    # Generate a unique collection name for this example run
    rag_collection_name = f"mcp_rag_example_coll_{uuid.uuid4().hex[:8]}"

    logger.info(f"Attempting to use MCP with RAG. Provider: '{PROVIDER_NAME}', Server: {MCP_SERVER_BASE_URL}")
    logger.info(f"Using RAG collection: {rag_collection_name}")
    logger.info("Ensure an MCP-compliant server is running and RAG (vector store, embedding model) is configured.")
    logger.info("Also, ensure 'modelcontextprotocol' library is installed (`pip install llmcore[mcp]`).")

    try:
        # 1. Initialize LLMCore
        logger.info("Initializing LLMCore with GithubMCPProvider configuration...")
        async with await LLMCore.create(config_overrides=CONFIG_OVERRIDES) as llm:
            logger.info("LLMCore initialized.")

            if PROVIDER_NAME not in llm.get_available_providers():
                logger.error(f"{PROVIDER_NAME} provider failed to load. Check config and dependencies.")
                return

            # --- 2. Add documents for RAG ---
            logger.info(f"\n--- Adding {len(RAG_DOCUMENTS)} documents to collection '{rag_collection_name}' ---")
            try:
                added_ids = await llm.add_documents_to_vector_store(
                    documents=RAG_DOCUMENTS,
                    collection_name=rag_collection_name
                )
                logger.info(f"Successfully added RAG documents with IDs: {added_ids}")
                # Brief pause for vector store to process, if necessary
                await asyncio.sleep(0.5)
            except (VectorStorageError, EmbeddingError, ConfigError) as e:
                logger.error(f"Failed to add RAG documents: {e}. Ensure vector store and embedding model are configured.")
                return # Cannot proceed with RAG chat

            # --- 3. Chat with RAG enabled (will use MCP) ---
            logger.info("\n--- Chatting with RAG via MCP ---")
            rag_prompt = "How are retrieved documents handled in MCP when RAG is used?"
            logger.info(f"User (RAG): {rag_prompt}")

            try:
                response_rag = await llm.chat(
                    message=rag_prompt,
                    enable_rag=True,
                    rag_collection_name=rag_collection_name,
                    rag_retrieval_k=1, # Retrieve top 1 relevant document
                    # System message can guide the LLM on how to use the context
                    system_message="You are an assistant. Please answer the question based on the provided context documents only."
                )
                logger.info(f"MCP Server RAG Response: {response_rag}")
            except ProviderError as e:
                logger.error(
                    f"RAG chat failed with {PROVIDER_NAME}: {e}. "
                    f"Is the MCP server at '{MCP_SERVER_BASE_URL}' running and accessible?"
                )
            except (VectorStorageError, EmbeddingError, ConfigError, LLMCoreError) as e:
                logger.error(f"An error occurred during RAG chat: {e}")

            # --- Cleanup (optional): Delete the test collection ---
            # try:
            #     logger.info(f"\n--- Cleaning up: Deleting RAG collection '{rag_collection_name}' ---")
            #     # Note: ChromaDB and some vector stores might not have a direct "delete_collection" API
            #     # through LLMCore. Deleting documents is more common.
            #     # If you need to delete the collection itself, you might need to use the
            #     # vector store's native client or manage its lifecycle outside LLMCore.
            #     # For this example, we'll delete the documents we added.
            #     if added_ids:
            #         deleted = await llm.delete_documents_from_vector_store(
            #             document_ids=added_ids,
            #             collection_name=rag_collection_name
            #         )
            #         logger.info(f"Document deletion from '{rag_collection_name}' attempt result: {deleted}")
            # except Exception as e:
            #     logger.error(f"Error during RAG collection cleanup: {e}")


    except ConfigError as e:
        logger.error(f"Initialization failed due to configuration error: {e}")
    except LLMCoreError as e:
        logger.error(f"An LLMCore error occurred: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
    # llm.close() is handled by 'async with'

if __name__ == "__main__":
    asyncio.run(main())
