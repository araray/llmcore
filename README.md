# LLMCore

**LLMCore** is a powerful and flexible Python library designed to provide a unified, asynchronous interface for interacting with various Large Language Models (LLMs). It simplifies the process of integrating advanced AI chat capabilities into your applications by abstracting away provider-specific complexities and offering robust session and context management, including Retrieval Augmented Generation (RAG).

Built with `asyncio`, LLMCore is suitable for modern, high-performance Python applications.

## ✨ Features

* **Unified API:** Interact with different LLM providers (OpenAI, Anthropic, Ollama, Gemini) through a consistent `chat()` method.
* **Asynchronous:** Leverages `asyncio` for non-blocking operations, ideal for concurrent applications.
* **Streaming Support:** Receive LLM responses as they are generated using asynchronous generators (`stream=True`).
* **Session Management:** Persist conversation history using configurable storage backends (JSON, SQLite included; PostgreSQL planned). Maintain context across multiple interactions.
* **Retrieval Augmented Generation (RAG):** Enhance LLM responses with external knowledge by automatically retrieving relevant documents from a configured vector store (ChromaDB included; pgvector planned) during chat.
* **Flexible Configuration:** Uses the [`confy`](https://github.com/araray/confy) library for layered configuration via defaults, TOML files, environment variables, and direct overrides. Manage providers, storage, embedding models, context strategies, and more.
* **Context Window Management:** Automatically handles token counting (provider-specific) and context truncation strategies to stay within model limits.
* **Provider & Storage Abstraction:** Easily extend LLMCore by adding new LLM providers, storage backends (session & vector), or embedding models by implementing simple base classes.
* **Embedding Model Support:** Integrates with embedding models (Sentence Transformers included; OpenAI, Google AI planned) for RAG functionality.

## 🚀 Quickstart

```python
import asyncio
from llmcore import LLMCore, LLMCoreError

async def main():
    # Initialize LLMCore using default configuration
    # (Loads from packaged defaults, ~/.config/llmcore/config.toml, .env, env vars)
    # Ensure your default provider (e.g., Ollama) is running or API keys are set.
    # Use 'async with' for automatic resource cleanup (calls llm.close())
    try:
        async with await LLMCore.create() as llm:

            # --- Simple Chat (Stateless, Non-Streaming) ---
            print("--- Simple Chat ---")
            response = await llm.chat("What is the capital of Brazil?")
            print(f"LLM: {response}")

            # --- Streaming Chat ---
            print("\n--- Streaming Chat ---")
            print("LLM (Streaming): ", end="", flush=True)
            response_stream = await llm.chat(
                "Tell me a short story.",
                stream=True
            )
            async for chunk in response_stream:
                print(chunk, end="", flush=True)
            print("\n--- Stream finished ---")

            # --- Chat with Session History ---
            print("\n--- Session Chat ---")
            session_id = "my_conversation_1"
            print(f"Starting/Continuing session: {session_id}")

            response1 = await llm.chat(
                "My name is Alex. What's the weather like in Porto Alegre?",
                session_id=session_id,
                system_message="You are a helpful assistant.",
                # save_session=True is default for persistent sessions
            )
            print(f"LLM: {response1}")

            # Follow-up question - LLM should remember the name "Alex"
            response2 = await llm.chat(
                "Does my name affect the weather?",
                session_id=session_id,
            )
            print(f"LLM: {response2}")

            # List sessions
            sessions_list = await llm.list_sessions()
            print("\nAvailable Sessions:", sessions_list)

            # --- RAG Example (Requires Vector Store Setup) ---
            # See examples/rag_example.py for adding documents first.
            # print("\n--- RAG Chat ---")
            # try:
            #     rag_response = await llm.chat(
            #         "What is LLMCore configuration based on the documents?",
            #         enable_rag=True,
            #         rag_collection_name="my_llmcore_docs" # Use the collection where docs were added
            #     )
            #     print(f"LLM (RAG): {rag_response}")
            # except LLMCoreError as rag_e:
            #     print(f"RAG chat failed (is vector store set up and populated?): {rag_e}")

    except LLMCoreError as e:
        print(f"\nAn LLMCore error occurred: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
````

## ⚙️ Configuration

LLMCore uses [`confy`](https://www.google.com/url?sa=E&source=gmail&q=https://github.com/araray/confy) for configuration. Settings are loaded with the following precedence (highest priority last):

1.  **Packaged Defaults:** `src/llmcore/config/default_config.toml`
2.  **User Config File:** `~/.config/llmcore/config.toml`
3.  **Custom Config File:** Path specified via `LLMCore.create(config_file_path=...)`
4.  **`.env` File:** Variables loaded into the environment (requires `python-dotenv`)
5.  **Environment Variables:** Prefixed with `LLMCORE_` (e.g., `LLMCORE_PROVIDERS__OPENAI__API_KEY`, `LLMCORE_STORAGE__VECTOR__TYPE`)
6.  **Overrides Dictionary:** Passed via `LLMCore.create(config_overrides=...)`

Key configuration areas include:

  * `[llmcore]`: Default provider, default embedding model, log level.
  * `[providers.<name>]`: Settings for each LLM provider (API keys, default models, timeouts).
  * `[storage.session]`: Configuration for session history storage (type, path/URL).
  * `[storage.vector]`: Configuration for RAG vector storage (type, path/URL, default collection).
  * `[embedding.<name>]`: Settings for specific embedding models (API keys, model names).
  * `[context_management]`: Strategies for history selection, RAG combination, and truncation.

See the [**Usage Guide**](https://www.google.com/search?q=docs/USAGE.md%23configuration) for the full configuration structure and details.

## 💾 Installation

**Requires Python 3.11 or later.**

```bash
pip install llmcore
```

Or install directly from the source code:

```bash
git clone [https://github.com/araray/llmcore.git](https://github.com/araray/llmcore.git)
cd llmcore
pip install .
```

To install with support for specific providers, storage backends, or embedding models, use extras:

```bash
# Example: Install with OpenAI, Anthropic, ChromaDB, SentenceTransformers support
pip install llmcore[openai,anthropic,chromadb,sentence_transformers]

# Install all optional dependencies
pip install llmcore[all]
```

See `pyproject.toml` for available extras (`openai`, `anthropic`, `gemini`, `ollama`, `sqlite`, `postgres`, `chromadb`, `sentence_transformers`).

## 📖 Documentation

For detailed instructions, API reference, configuration details, and advanced examples, please refer to the [**Usage Guide**](https://www.google.com/search?q=docs/USAGE.md).

## 🤝 Contributing

Contributions are welcome\! Please see the guidelines in the specification document (`docs/llmcore_spec_v1.0.md`) or open an issue/pull request on GitHub.

## 📄 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
