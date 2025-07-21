# LLMCore v0.21.0: User Guide





## Introduction to LLMCore v0.21.0



Welcome to LLMCore v0.21.0. What began as a library for unified Large Language Model (LLM) interaction has evolved into a comprehensive platform for building sophisticated, scalable, and autonomous AI applications.

LLMCore provides a powerful, asynchronous Python interface that abstracts the complexities of individual provider APIs. Beyond its robust session management and Retrieval Augmented Generation (RAG) capabilities, version 0.21.0 introduces a powerful new ecosystem of interconnected components:

- **An Autonomous Agent Framework:** Build goal-oriented agents that can reason, act using tools, and learn from their experiences.
- **A Backend Task Queue:** Offload long-running operations like data ingestion and complex agent tasks to a background worker, ensuring your application remains responsive.
- **A Full-Featured API Server:** Expose all of LLMCore's functionality through a modern, asynchronous REST API, ready for integration with web frontends and other services.

This guide will get you started with the core features of the LLMCore library quickly. For a deep dive into the agentic framework, task management system, advanced configuration, and the full API reference, please consult the `DEVELOPERS_GUIDE.md`.



## Installation and Setup



**Requires Python 3.11 or later.**



### Core Installation



The standard installation provides the core library, ready for use with local backends like Ollama and SQLite.

Bash

```
pip install llmcore
```



### Installation with Extras



LLMCore uses optional dependencies ("extras") to keep your environment lean. Install only what you need for specific providers, storage backends, or embedding models. You can combine multiple extras in a single command.1

| Extra Name              | Provides Support For    | Example Command                              |
| ----------------------- | ----------------------- | -------------------------------------------- |
| `openai`                | OpenAI Models           | `pip install llmcore[openai]`                |
| `anthropic`             | Anthropic Claude Models | `pip install llmcore[anthropic]`             |
| `gemini`                | Google Gemini Models    | `pip install llmcore[gemini]`                |
| `ollama`                | Local Ollama Server     | `pip install llmcore[ollama]`                |
| `postgres`              | PostgreSQL Storage      | `pip install llmcore[postgres]`              |
| `chromadb`              | ChromaDB Vector Store   | `pip install llmcore[chromadb]`              |
| `sentence_transformers` | Local Embedding Models  | `pip install llmcore[sentence_transformers]` |
| `all`                   | All optional features   | `pip install llmcore[all]`                   |



### Initial Configuration



The quickest way to configure a provider is by setting an environment variable for its API key. For example, to use OpenAI:

Bash

```
export LLMCORE_PROVIDERS__OPENAI__API_KEY="sk-..."
```

LLMCore will automatically detect and use this key.



## Quickstart: Core Chat Functionality



This example demonstrates the fundamental chat capabilities of LLMCore: stateless chat, stateful sessions, and streaming.1

Python

```
import asyncio
from llmcore import LLMCore, LLMCoreError

async def main():
    # Initialize LLMCore using default configuration.
    # This automatically loads settings from files and environment variables.
    # The 'async with' block ensures resources are cleaned up properly.
    try:
        async with await LLMCore.create() as llm:
            # --- 1. Simple, Stateless Chat ---
            # This is a one-off query with no memory of past interactions.
            print("--- Simple Chat ---")
            response = await llm.chat("What is the capital of Brazil?")
            print(f"LLM: {response}")

            # --- 2. Streaming Chat ---
            # Receive the response in chunks as it's generated.
            print("\n--- Streaming Chat ---")
            print("LLM (Streaming): ", end="", flush=True)
            response_stream = await llm.chat(
                "Tell me a short story about a robot who discovers music.",
                stream=True
            )
            async for chunk in response_stream:
                print(chunk, end="", flush=True)
            print("\n--- Stream finished ---")

            # --- 3. Stateful Conversation with a Session ---
            # Use a session_id to maintain context across multiple turns.
            print("\n--- Session Chat ---")
            session_id = "my_first_conversation"
            print(f"Starting/Continuing session: {session_id}")

            # First turn: The LLM learns our name.
            response1 = await llm.chat(
                "My name is Alex. What is the main component of a star?",
                session_id=session_id,
                system_message="You are a helpful science tutor."
            )
            print(f"LLM: {response1}")

            # Second turn: The LLM should remember our name from the session history.
            response2 = await llm.chat(
                "Does my name have any relevance to that topic?",
                session_id=session_id
            )
            print(f"LLM: {response2}")

    except LLMCoreError as e:
        print(f"\nAn LLMCore error occurred: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```



## Quickstart: Retrieval Augmented Generation (RAG)



Enhance your LLM's knowledge by providing it with documents to use as context. This example shows the basic RAG workflow.1

Python

```
import asyncio
import uuid
from llmcore import LLMCore, LLMCoreError

async def main():
    # Use a unique collection name for this run to avoid conflicts
    collection_name = f"quickstart_rag_{uuid.uuid4().hex[:6]}"
    
    try:
        async with await LLMCore.create() as llm:
            print(f"--- RAG Example (using collection: {collection_name}) ---")

            # --- 1. Add a document to the vector store ---
            # This is the external knowledge we want the LLM to use.
            document_content = """
            The LLMCore library's configuration is managed by a library called 'confy'.
            It supports layered settings from TOML files, environment variables,
            and direct in-code overrides.
            """
            print("Adding document to vector store...")
            await llm.add_documents_to_vector_store(
                documents=[{"content": document_content, "metadata": {"source": "quickstart.py"}}],
                collection_name=collection_name
            )
            print("Document added successfully.")
            
            # Give the vector store a moment to index the document
            await asyncio.sleep(1)

            # --- 2. Ask a question using the document as context ---
            # Set enable_rag=True to activate the RAG functionality.
            question = "Based on the documents, how is configuration handled in LLMCore?"
            print(f"\nUser Question: {question}")
            
            rag_response = await llm.chat(
                message=question,
                enable_rag=True,
                rag_collection_name=collection_name,
                system_message="Answer the user's question based only on the provided context documents."
            )
            
            print(f"\nLLM (RAG) Response: {rag_response}")

    except LLMCoreError as e:
        print(f"\nA RAG error occurred: {e}. Is your vector store (e.g., ChromaDB) and embedding model configured?")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```



## Quickstart: Running the API Server



LLMCore includes a pre-built FastAPI server to expose its functionality over a REST API.

1. Start the Server:

    Run the following command from the root of the llmcore project directory. The --reload flag enables hot-reloading for development.

    Bash

    ```
    python -m uvicorn llmcore.api_server.main:app --reload
    ```

2. Interact with the API:

    Once the server is running, you can interact with it using any HTTP client, such as curl.1

    Bash

    ```
    curl -X POST "http://127.0.0.1:8000/api/v1/chat" \
    -H "Content-Type: application/json" \
    -d '{"message": "Hello from the API!"}'
    ```

    You should receive a JSON response like:

    JSON

    ```
    {"response":"Hello! How can I assist you today?","session_id":null}
    ```

## Next Steps

You have now seen the basic capabilities of the LLMCore library and API server. To unlock the full potential of the platform, including building autonomous agents, managing background tasks, and understanding the advanced configuration options, please proceed to the **`DEVELOPERS_GUIDE.md`**.