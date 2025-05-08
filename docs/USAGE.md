# LLMCore Usage Guide

This guide provides detailed information on how to use the LLMCore library, including initialization, configuration, core features, and examples.

## Table of Contents

1.  [Initialization (`LLMCore.create`)](#initialization-llmcorecreate)
2.  [Configuration](#configuration)
    * [Configuration Sources & Precedence](#configuration-sources--precedence)
    * [Default Configuration Structure](#default-configuration-structure)
    * [Environment Variables](#environment-variables)
    * [Overrides Dictionary](#overrides-dictionary)
3.  [Core Chat (`chat` method)](#core-chat-chat-method)
    * [Parameters](#parameters)
    * [Simple Chat Example](#simple-chat-example)
    * [Streaming Chat Example](#streaming-chat-example)
    * [Provider & Model Selection](#provider--model-selection)
4.  [Session Management](#session-management)
    * [Using Sessions](#using-sessions)
    * [Listing Sessions (`list_sessions`)](#listing-sessions-list_sessions)
    * [Retrieving a Session (`get_session`)](#retrieving-a-session-get_session)
    * [Deleting a Session (`delete_session`)](#deleting-a-session-delete_session)
5.  [Retrieval Augmented Generation (RAG) - *Phase 2*](#retrieval-augmented-generation-rag---phase-2)
    * [Concept](#concept)
    * [API Methods (Future)](#api-methods-future)
6.  [Provider Information](#provider-information)
    * [`get_available_providers`](#get_available_providers)
    * [`get_models_for_provider`](#get_models_for_provider)
7.  [Error Handling](#error-handling)
8.  [Full Examples](#full-examples)

---

## Initialization (`LLMCore.create`)

The primary way to interact with the library is through the `LLMCore` class. Since initialization involves potentially asynchronous operations (like loading storage backends), you must instantiate it using the `async` class method `create()`:

```python
import asyncio
from llmcore import LLMCore

async def initialize_llmcore():
    # Basic initialization (uses default config loading)
    llm = await LLMCore.create()
    print("LLMCore initialized!")

    # Initialization with overrides
    overrides = {
        "llmcore.default_provider": "openai",
        "providers.openai.default_model": "gpt-4o",
        "storage.session.type": "json"
    }
    llm_custom = await LLMCore.create(
        config_overrides=overrides,
        config_file_path="path/to/my_config.toml", # Optional custom config file
        env_prefix="MYAPP" # Optional custom env var prefix (default: LLMCORE)
    )
    print("Custom LLMCore initialized!")

    # Remember to close resources when done
    await llm.close()
    await llm_custom.close()

# asyncio.run(initialize_llmcore())
````

**`LLMCore.create()` Parameters:**

  * `config_overrides` (Optional\[Dict]): A dictionary where keys use dot-notation (e.g., `"providers.openai.default_model"`) to override any configuration value. Highest precedence.
  * `config_file_path` (Optional\[str]): Path to a custom TOML or JSON configuration file.
  * `env_prefix` (Optional\[str]): The prefix for environment variables (default: `"LLMCORE"`). Set to `""` to consider all non-system environment variables, or `None` to disable environment variable loading entirely.

---

## Configuration

LLMCore uses the [`confy`](https://github.com/araray/confy) library for flexible configuration management.

### Configuration Sources & Precedence

Settings are loaded from multiple sources, with later sources overriding earlier ones:

1.  **Packaged Defaults:** Sensible defaults included with the library (`src/llmcore/config/default_config.toml`).
2.  **User Config File:** A TOML file located at `~/.config/llmcore/config.toml` (created automatically if needed by some operations, but best practice is to create it manually if customization is desired).
3.  **Custom Config File:** The file specified via `config_file_path` during initialization.
4.  **`.env` File:** Variables loaded into the environment from a `.env` file found in the current or parent directories (requires `python-dotenv` to be installed). *Note:* Existing environment variables are **not** overridden by `.env` values.
5.  **Environment Variables:** System environment variables, prefixed with `LLMCORE_` (or the custom `env_prefix`). See [Environment Variables](environment-variables) below for details.
6.  **Overrides Dictionary:** The `config_overrides` dictionary passed during initialization.

### Default Configuration Structure

The default configuration (`src/llmcore/config/default_config.toml`) provides a template for all settings:

```toml
# Default configuration structure for LLMCore

[llmcore]
# Default provider if not specified in API calls (e.g., "ollama", "openai")
default_provider = "ollama"
# Default embedding model for RAG (Phase 2)
default_embedding_model = "all-MiniLM-L6-v2"
# Global flag for MCP formatting (Phase 3)
enable_mcp = false
# Log level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
log_level = "INFO"

# --- Provider Configurations ---
[providers]
  [providers.openai]
  # API Key: Best set via LLMCORE_PROVIDERS_OPENAI_API_KEY env var
  api_key = ""
  default_model = "gpt-4o"
  timeout = 60 # seconds
  # base_url = "..." # Optional base URL override

  [providers.anthropic]
  # API Key: Best set via LLMCORE_PROVIDERS_ANTHROPIC_API_KEY env var
  api_key = ""
  default_model = "claude-3-opus-20240229"
  timeout = 60

  [providers.ollama]
  base_url = "http://localhost:11434/api"
  default_model = "llama3"
  timeout = 120
  # tokenizer = "tiktoken_cl100k_base" # Optional tokenizer hint

  # [providers.gemini] # Phase 2/3
  # api_key = "" # LLMCORE_PROVIDERS_GEMINI_API_KEY
  # default_model = "gemini-1.5-pro-latest"
  # timeout = 60

# --- Storage Configurations ---
[storage]
  # Session storage (conversation history)
  [storage.session]
  # Type: 'json', 'sqlite' (more planned)
  type = "sqlite"
  # Path for file-based storage (json dir, sqlite file)
  path = "~/.llmcore/sessions.db"
  # db_url = "" # For DB-based storage (e.g., PostgreSQL - Phase 3)

  # Vector storage (for RAG - Phase 2)
  [storage.vector]
  # Type: 'chromadb', 'pgvector' (planned)
  type = "chromadb"
  default_collection = "llmcore_default_rag"
  # Path for file-based vector stores (chromadb)
  path = "~/.llmcore/chroma_db"
  # db_url = "" # For DB-based vector stores

# --- Embedding Model Configurations (Phase 2) ---
[embedding]
  # [embedding.openai]
  # default_model = "text-embedding-3-small"
  # [embedding.google]
  # default_model = "models/embedding-001"

# --- Context Management Configurations ---
[context_management]
  # Default number of docs for RAG (Phase 2)
  rag_retrieval_k = 3
  # RAG combination strategy (Phase 2)
  rag_combination_strategy = "prepend_system"
  # History selection strategy ('last_n_tokens')
  history_selection_strategy = "last_n_tokens"
  # Tokens reserved for the LLM's response
  reserved_response_tokens = 500
  # Truncation priority ('history', 'rag' - Phase 2)
  truncation_priority = "history"
  # Min history messages to keep during truncation
  minimum_history_messages = 1
```

### Environment Variables

Override any configuration setting using environment variables.

  * **Prefix:** Use `LLMCORE_` (or your custom `env_prefix`).
  * **Path Separator:** Use double underscores (`__`) to represent dots (`.`) in the configuration path.
  * **Case:** Environment variable names are case-insensitive.

**Examples:**

  * Set OpenAI API Key: `export LLMCORE_PROVIDERS__OPENAI__API_KEY="sk-..."`
  * Set default provider: `export LLMCORE_DEFAULT_PROVIDER="anthropic"`
  * Set session storage type: `export LLMCORE_STORAGE__SESSION__TYPE="json"`
  * Set Ollama URL: `export LLMCORE_PROVIDERS__OLLAMA__BASE_URL="http://192.168.1.100:11434/api"`

### Overrides Dictionary

Provide a dictionary to `LLMCore.create(config_overrides=...)` for the highest precedence overrides. Keys use dot-notation.

```python
overrides = {
    "llmcore.default_provider": "openai",
    "providers.openai.timeout": 120,
    "storage.session.path": "/data/llmcore_sessions.db"
}
llm = await LLMCore.create(config_overrides=overrides)
```

---

## Core Chat (`chat` method)

The primary method for interacting with the LLM is `llm.chat()`.

```python
response_or_stream = await llm.chat(
    message="Your prompt to the LLM",
    *, # Remaining arguments must be keyword-only
    session_id="my_conversation", # Optional: ID for conversation history
    system_message="You are a helpful assistant.", # Optional: Set LLM behavior
    provider_name="openai", # Optional: Override default provider
    model_name="gpt-4o", # Optional: Override provider's default model
    stream=False, # Optional: Set to True for streaming response
    save_session=True, # Optional: Save turn to session storage (if session_id provided)
    # RAG Parameters (Phase 2)
    # enable_rag=False,
    # rag_retrieval_k=3,
    # rag_collection_name="my_docs",
    # Provider-specific arguments
    temperature=0.7,
    max_tokens=150 # Note: Often refers to *response* length limit
)
```

### Parameters

  * `message` (str): The user's input message.
  * `session_id` (Optional\[str]): The ID of the conversation session to use or continue. If `None`, the chat is stateless (no history saved or loaded).
  * `system_message` (Optional\[str]): A message defining the role or behavior of the assistant (e.g., "You are a helpful AI assistant specialized in Python."). This is typically used at the start of a new session.
  * `provider_name` (Optional\[str]): The name of the LLM provider to use (e.g., `"openai"`, `"ollama"`). Overrides the configured default.
  * `model_name` (Optional\[str]): The specific model identifier for the chosen provider (e.g., `"gpt-4o"`, `"llama3"`). Overrides the provider's default model.
  * `stream` (bool): If `True`, returns an `AsyncGenerator[str, None]` yielding text chunks as they arrive. If `False` (default), returns the complete response content as a single string after the LLM finishes.
  * `save_session` (bool): If `True` (default) and a `session_id` is provided, the user message and the assistant's full response are saved to the persistent session storage.
  * `enable_rag` (bool, *Phase 2*): Set to `True` to enable Retrieval Augmented Generation.
  * `rag_retrieval_k` (Optional\[int], *Phase 2*): Number of documents to retrieve for RAG.
  * `rag_collection_name` (Optional\[str], *Phase 2*): Vector store collection name for RAG.
  * `**provider_kwargs`: Additional keyword arguments passed directly to the selected provider's API call (e.g., `temperature=0.7`, `max_tokens=100`, `top_p=0.9`). Consult the specific provider's documentation for available parameters.

### Simple Chat Example

```python
import asyncio
from llmcore import LLMCore

async def simple_chat():
    llm = await LLMCore.create()
    try:
        # Using default provider (e.g., Ollama)
        response = await llm.chat("What is the main export of Brazil?")
        print(f"LLM: {response}")
    finally:
        await llm.close()

# asyncio.run(simple_chat())
```

### Streaming Chat Example

```python
import asyncio
from llmcore import LLMCore

async def stream_chat():
    llm = await LLMCore.create()
    try:
        print("LLM (Streaming): ", end="", flush=True)
        async_generator = await llm.chat(
            "Write a short poem about asynchronous programming.",
            stream=True
        )
        async for chunk in async_generator:
            print(chunk, end="", flush=True)
        print("\n--- Stream Complete ---")
    finally:
        await llm.close()

# asyncio.run(stream_chat())
```

### Provider & Model Selection

```python
import asyncio
from llmcore import LLMCore, ProviderError

async def specific_model_chat():
    # Ensure LLMCORE_PROVIDERS_ANTHROPIC_API_KEY is set in environment
    llm = await LLMCore.create()
    try:
        response = await llm.chat(
            "Compare and contrast Python and Rust for web development.",
            provider_name="anthropic",
            model_name="claude-3-haiku-20240307",
            temperature=0.6
        )
        print(f"Claude Haiku: {response}")
    except ProviderError as e:
         print(f"Error: {e}") # Handle provider-specific errors (e.g., missing API key)
    finally:
        await llm.close()

# asyncio.run(specific_model_chat())
```

-----

## Session Management

LLMCore allows you to maintain conversation history using sessions. When you provide a `session_id` to the `chat` method, the library loads previous messages (respecting context limits) and saves the new turn (user message + assistant response) if `save_session=True`.

### Using Sessions

```python
import asyncio
from llmcore import LLMCore

async def session_example():
    llm = await LLMCore.create()
    session_id = "my_project_ideas"
    try:
        # First interaction
        await llm.chat(
            "Let's brainstorm some project ideas using Python.",
            session_id=session_id,
            system_message="You are a creative brainstorming partner."
        )
        # (Response will be printed by the streaming example logic if stream=True)

        # Second interaction in the same session
        await llm.chat(
            "Okay, how about an idea related to data visualization?",
            session_id=session_id
        )

        # Third interaction
        await llm.chat(
            "Expand on the data visualization idea.",
            session_id=session_id
        )
    finally:
        await llm.close()

# asyncio.run(session_example())
```

### Listing Sessions (`list_sessions`)

Retrieve metadata about all saved sessions.

```python
import asyncio
from llmcore import LLMCore

async def list_all_sessions():
    llm = await LLMCore.create()
    try:
        sessions = await llm.list_sessions()
        if not sessions:
            print("No saved sessions found.")
        else:
            print("Saved Sessions:")
            for sess_info in sessions:
                 print(f"- ID: {sess_info.get('id', 'N/A')}, "
                       f"Name: {sess_info.get('name', 'N/A')}, "
                       f"Msgs: {sess_info.get('message_count', 0)}, "
                       f"Updated: {sess_info.get('updated_at', 'N/A')}")
    finally:
        await llm.close()

# asyncio.run(list_all_sessions())
```

### Retrieving a Session (`get_session`)

Load a specific session object, including all its messages.

```python
import asyncio
from llmcore import LLMCore, SessionNotFoundError

async def retrieve_specific_session(session_id_to_load):
    llm = await LLMCore.create()
    try:
        session = await llm.get_session(session_id_to_load)
        if session:
            print(f"Session '{session.name}' loaded.")
            print(f"Total messages: {len(session.messages)}")
            # Access messages: session.messages[0].content, etc.
            if session.messages:
                 print(f"Last message: [{session.messages[-1].role.value}] {session.messages[-1].content[:80]}...")
        else:
             print(f"Session with ID '{session_id_to_load}' not found.")

    except SessionNotFoundError: # Explicitly catch if needed, though get_session returns None
         print(f"Session with ID '{session_id_to_load}' not found.")
    finally:
        await llm.close()

# asyncio.run(retrieve_specific_session("my_project_ideas"))
```

### Deleting a Session (`delete_session`)

Permanently remove a session from storage.

```python
import asyncio
from llmcore import LLMCore

async def delete_specific_session(session_id_to_delete):
    llm = await LLMCore.create()
    try:
        deleted = await llm.delete_session(session_id_to_delete)
        if deleted:
            print(f"Session '{session_id_to_delete}' deleted successfully.")
        else:
            print(f"Session '{session_id_to_delete}' not found or could not be deleted.")
    finally:
        await llm.close()

# asyncio.run(delete_specific_session("my_project_ideas"))
```

-----

## Retrieval Augmented Generation (RAG) - *Phase 2*

*(This feature is planned for Phase 2 and is not yet fully implemented)*

RAG allows the LLM to access external information stored in a vector database during conversation. This helps the model answer questions about specific documents or data it wasn't originally trained on.

### Concept

1.  **Add Documents:** You add text documents (e.g., PDFs, text files, database records) to a configured vector store (like ChromaDB or PostgreSQL+pgvector). LLMCore generates vector embeddings for these documents using a configured embedding model (like Sentence Transformers or OpenAI embeddings).
2.  **Chat with RAG:** When you call `llm.chat()` with `enable_rag=True`, LLMCore:
      * Generates an embedding for your query message.
      * Searches the vector store for documents with similar embeddings.
      * Retrieves the content of the most relevant documents.
      * Injects this retrieved content into the context sent to the LLM, along with your message and conversation history.
3.  **Informed Response:** The LLM uses the provided document context to generate a more informed and accurate response.

### API Methods (Future)

These methods will be used to manage the vector store content:

  * `llm.add_document_to_vector_store(content, metadata=None, doc_id=None, collection_name=None)`
  * `llm.add_documents_to_vector_store(documents, collection_name=None)`
  * `llm.search_vector_store(query, k, collection_name=None, filter_metadata=None)`
  * `llm.delete_documents_from_vector_store(document_ids, collection_name=None)`

-----

## Provider Information

You can query LLMCore for information about the configured providers and their models.

### `get_available_providers`

Lists the names of all providers that were successfully loaded based on your configuration.

```python
import asyncio
from llmcore import LLMCore

async def show_providers():
    llm = await LLMCore.create()
    try:
        providers = llm.get_available_providers()
        print("Loaded Providers:", providers)
    finally:
        await llm.close()

# asyncio.run(show_providers())
# Output might be: Loaded Providers: ['ollama', 'openai'] (if both configured)
```

### `get_models_for_provider`

Lists the models available for a *specific* loaded provider. Note that this might return a static list defined in the provider code or potentially involve an API call, depending on the provider implementation.

```python
import asyncio
from llmcore import LLMCore, ConfigError

async def show_provider_models(provider_name):
    llm = await LLMCore.create()
    try:
        models = llm.get_models_for_provider(provider_name)
        print(f"Models for '{provider_name}':", models)
    except ConfigError as e:
         print(e) # Handle case where provider isn't configured
    finally:
        await llm.close()

# asyncio.run(show_provider_models("openai"))
# Output might be: Models for 'openai': ['gpt-4', 'gpt-4o', 'gpt-3.5-turbo', ...]
```

-----

## Error Handling

LLMCore defines custom exceptions in `llmcore.exceptions` to help you handle specific issues:

  * `LLMCoreError`: Base exception for library errors.
  * `ConfigError`: Errors during configuration loading or validation.
  * `ProviderError`: Errors originating from an LLM provider API (e.g., connection issues, authentication failure, rate limits).
  * `StorageError`: Base class for storage-related errors.
      * `SessionStorageError`: Errors specific to session storage.
      * `VectorStorageError`: Errors specific to vector storage (Phase 2).
  * `SessionNotFoundError`: Raised when a specified `session_id` is not found in storage.
  * `ContextError`: Base class for context management errors.
      * `ContextLengthError`: Raised when context exceeds the model's limit even after truncation.
  * `EmbeddingError`: Errors during embedding generation (Phase 2).
  * `MCPError`: Errors related to MCP formatting (Phase 3).


```python
import asyncio
from llmcore import LLMCore, LLMCoreError, ProviderError, ContextLengthError

async def chat_with_error_handling():
    llm = None
    try:
        llm = await LLMCore.create()
        response = await llm.chat(
            "This is a very long message..." * 1000, # Intentionally long
            provider_name="openai",
            model_name="gpt-3.5-turbo" # Model with smaller context
        )
        print(f"LLM: {response}")

    except ContextLengthError as e:
        print(f"Error: Context too long for model '{e.model_name}'. Limit: {e.limit}, Actual: {e.actual}")
    except ProviderError as e:
        print(f"Error with provider '{e.provider_name}': {e}")
    except ConfigError as e:
        print(f"Configuration Error: {e}")
    except LLMCoreError as e:
        print(f"An LLMCore error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if llm:
            await llm.close()

# asyncio.run(chat_with_error_handling())
```

-----

## Full Examples

See the `examples/` directory in the repository for runnable scripts demonstrating various use cases:

  * `simple_chat.py`: Basic chat interaction.
  * `session_chat.py`: Demonstrates conversation history using sessions.
  * `streaming_chat.py`: Shows how to handle streaming responses.
  * `rag_example.py`: (Future) Will demonstrate adding documents and chatting with RAG enabled.
