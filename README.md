# LLMCore

**LLMCore** is a powerful and flexible Python library designed to provide a unified, asynchronous interface for interacting with various Large Language Models (LLMs). It simplifies the process of integrating advanced AI chat capabilities into your applications by abstracting away provider-specific complexities and offering robust session and context management.

Built with `asyncio`, LLMCore is suitable for modern, high-performance Python applications.

## ✨ Features

* **Unified API:** Interact with different LLM providers (Ollama, OpenAI, Anthropic, Gemini planned) through a consistent `chat()` method.
* **Asynchronous:** Leverages `asyncio` for non-blocking operations, ideal for concurrent applications.
* **Streaming Support:** Receive LLM responses as they are generated using asynchronous generators.
* **Session Management:** Persist conversation history using configurable storage backends (JSON, SQLite included; PostgreSQL planned). Maintain context across multiple interactions.
* **Flexible Configuration:** Uses the [`confy`](https://github.com/araray/confy) library for layered configuration via defaults, TOML files, environment variables, and direct overrides.
* **Context Window Management:** Automatically handles token counting (provider-specific) and basic context truncation to stay within model limits.
* **Provider & Storage Abstraction:** Easily extend LLMCore by adding new LLM providers or storage backends by implementing simple base classes.
* **(Planned) Retrieval Augmented Generation (RAG):** Future support for integrating vector stores (ChromaDB, pgvector planned) to provide LLMs with external knowledge.
* **(Planned) Model Context Protocol (MCP):** Future support for standardized context exchange using the MCP SDK.

## 🚀 Quickstart

```python
import asyncio
from llmcore import LLMCore, LLMCoreError

async def main():
    # Initialize LLMCore (loads config from defaults, files, env vars)
    # Ensure you have Ollama running locally for the default config,
    # or configure API keys via environment variables (e.g., LLMCORE_PROVIDERS_OPENAI_API_KEY)
    # and override the provider if needed.
    try:
        llm = await LLMCore.create() # Use await for async initialization

        # --- Simple Chat (Non-Streaming) ---
        print("--- Simple Chat ---")
        response = await llm.chat("What is the capital of Brazil?")
        print(f"LLM: {response}")

        # --- Chat with a specific provider/model ---
        print("\n--- Specific Provider Chat (requires API key env var) ---")
        # Make sure LLMCORE_PROVIDERS_OPENAI_API_KEY is set in your environment
        try:
            response_openai = await llm.chat(
                "Explain the concept of recursion concisely.",
                provider_name="openai", # Specify provider
                model_name="gpt-4o",   # Specify model
                temperature=0.5      # Pass provider-specific args
            )
            print(f"OpenAI: {response_openai}")
        except LLMCoreError as e:
            print(f"OpenAI chat failed (is API key set?): {e}")


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
        print(f"Starting session: {session_id}")

        response1 = await llm.chat(
            "My name is Alex. What is the weather like in Porto Alegre today?",
            session_id=session_id,
            system_message="You are a helpful assistant.",
            save_session=True # Default, but explicit here
        )
        print(f"LLM: {response1}")

        # Follow-up question - LLM should remember the name "Alex"
        response2 = await llm.chat(
            "Does my name affect the weather?",
            session_id=session_id,
            save_session=True
        )
        print(f"LLM: {response2}")

        # List sessions
        sessions_list = await llm.list_sessions()
        print("\nAvailable Sessions:", sessions_list)

    except LLMCoreError as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # Clean up resources (e.g., close storage connections)
        if 'llm' in locals() and llm:
             await llm.close()

if __name__ == "__main__":
    # In Python 3.8+, you can run top-level async functions like this
    asyncio.run(main())
````

## ⚙️ Configuration

LLMCore uses [`confy`](https://github.com/araray/confy) for configuration. Settings are loaded with the following precedence (highest priority last):

1.  **Packaged Defaults:** `src/llmcore/config/default_config.toml`
2.  **User Config File:** `~/.config/llmcore/config.toml` (if it exists)
3.  **Custom Config File:** Path specified via `LLMCore.create(config_file_path=...)`
4.  **`.env` File:** Variables loaded into the environment (requires `python-dotenv`)
5.  **Environment Variables:** Prefixed with `LLMCORE_` (e.g., `LLMCORE_PROVIDERS_OPENAI_API_KEY`, `LLMCORE_STORAGE_SESSION_TYPE`)
6.  **Overrides Dictionary:** Passed via `LLMCore.create(config_overrides=...)`

See the [Usage Guide](docs/USAGE.md) for details on the configuration file structure and environment variables.

## 💾 Installation

**Requires Python 3.9 or later.**

```bash
pip install llmcore
```

Or install directly from the source code:

```bash
git clone https://github.com/araray/llmcore.git
cd llmcore
pip install .
```

To install with support for specific providers or storage backends, use extras:

```bash
# Example: Install with OpenAI, Anthropic, and PostgreSQL support
pip install llmcore[openai,anthropic,postgres]

# Install all optional dependencies
pip install llmcore[all]
```

See `pyproject.toml` for available extras.

## 📖 Documentation

For detailed instructions, API reference, and advanced examples, please refer to the [**Usage Guide**](docs/USAGE.md).

## 🤝 Contributing

Contributions are welcome\! Please see the guidelines in the specification document (`docs/llmcore_spec_v1.0.md`) or open an issue/pull request on GitHub.

## 📄 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
