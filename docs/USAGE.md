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
5.  [Retrieval Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
    * [Concept](#concept)
    * [Adding Documents](#adding-documents)
    * [Searching Documents](#searching-documents)
    * [Deleting Documents](#deleting-documents)
    * [Chatting with RAG](#chatting-with-rag)
6.  [Provider Information](#provider-information)
    * [`get_available_providers`](#get_available_providers)
    * [`get_models_for_provider`](#get_models_for_provider)
7.  [Error Handling](#error-handling)
8.  [Full Examples](#full-examples)

---

## Initialization (`LLMCore.create`)

The primary way to interact with the library is through the `LLMCore` class. Since initialization involves potentially asynchronous operations (like loading storage backends and embedding models), you must instantiate it using the `async` class method `create()`:

```python
import asyncio
from llmcore import LLMCore

async def initialize_llmcore():
    # Basic initialization (uses default config loading)
    # Use 'async with' for automatic resource cleanup (calls llm.close())
    async with await LLMCore.create() as llm:
        print("LLMCore initialized!")

        # Initialization with overrides
        overrides = {
            "llmcore.default_provider": "openai",
            "providers.openai.default_model": "gpt-4o",
            "storage.session.type": "json"
        }
        async with await LLMCore.create(
            config_overrides=overrides,
            config_file_path="path/to/my_config.toml", # Optional custom config file
            env_prefix="MYAPP" # Optional custom env var prefix (default: LLMCORE)
        ) as llm_custom:
            print("Custom LLMCore initialized!")

    # llm.close() is called automatically when exiting the 'async with' block

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
# src/llmcore/config/default_config.toml
# Default configuration structure for LLMCore
# This file defines the standard settings and can be overridden by user configurations,
# environment variables, or direct overrides in code.

[llmcore]
# Default provider to use if not specified in API calls.
# Options: "openai", "anthropic", "ollama", "gemini", or any custom provider name.
default_provider = "ollama"

# Default embedding model for RAG (Retrieval Augmented Generation).
# This can be a local path to a sentence-transformers model (e.g., "all-MiniLM-L6-v2")
# or an identifier for a service-based model (e.g., "openai:text-embedding-ada-002", "google:models/embedding-001").
default_embedding_model = "all-MiniLM-L6-v2" # A common sentence-transformer model

# Global flag to enable/disable Model Context Protocol (MCP) formatting.
# This can be overridden on a per-provider basis in their respective sections.
enable_mcp = false

# Log level for the library.
# Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
log_level = "INFO"


# --- Provider Configurations ---
# Each provider has its own section under [providers.<provider_name>].
# API keys are best set via environment variables (e.g., LLMCORE_PROVIDERS_OPENAI_API_KEY).
[providers]

  [providers.openai]
  # API Key: Recommended to use environment variable LLMCORE_PROVIDERS_OPENAI_API_KEY
  api_key = "" # Leave empty to rely on environment variable
  default_model = "gpt-4o" # Updated default
  timeout = 60 # Timeout in seconds for API calls
  # use_mcp = false # Provider-specific MCP toggle (overrides llmcore.enable_mcp)

  [providers.anthropic]
  # API Key: Recommended to use environment variable LLMCORE_PROVIDERS_ANTHROPIC_API_KEY
  api_key = "" # Leave empty to rely on environment variable
  default_model = "claude-3-opus-20240229"
  timeout = 60 # Timeout in seconds for API calls
  # use_mcp = false

  [providers.ollama]
  # base_url = "http://localhost:11434/api" # Base URL for the Ollama API (host defaults to http://localhost:11434)
  host = "http://localhost:11434" # Hostname/IP for the ollama library client
  default_model = "llama3" # Updated default
  timeout = 120 # Timeout in seconds for API calls
  # Optional: Specify tokenizer for Ollama models if default (tiktoken_cl100k_base) is not suitable.
  # Accurate token counting for Ollama can be tricky. Options:
  # 'tiktoken_cl100k_base': (Default) Good general-purpose tokenizer (used by GPT-3.5/4).
  # 'tiktoken_p50k_base': Another tiktoken option.
  # 'char_div_4': A very rough estimate (characters / 4). Use if tiktoken is problematic.
  # tokenizer = "tiktoken_cl100k_base"
  # use_mcp = false

  [providers.gemini]
  # API Key: Recommended to use environment variable LLMCORE_PROVIDERS_GEMINI_API_KEY
  api_key = "" # Leave empty to rely on environment variable
  default_model = "gemini-1.5-pro-latest"
  timeout = 60 # Timeout in seconds for API calls
  # Add other Gemini specific settings if needed (e.g., safety settings).
  # Example: safety_settings = { HARM_CATEGORY_SEXUALLY_EXPLICIT = "BLOCK_NONE" }
  # use_mcp = false


# --- Storage Configurations ---
# Defines where and how session history and vector embeddings are stored.
[storage]

  # Session storage configuration: for conversation history.
  [storage.session]
  # Type: 'json', 'sqlite', 'postgres' (postgres not implemented yet)
  type = "sqlite"

  # Path for file-based storage (json, sqlite).
  # '~' will be expanded to the user's home directory.
  path = "~/.llmcore/sessions.db" # For SQLite, this is the DB file. For JSON, this is the directory.

  # Connection URL for database storage (e.g., postgres).
  # Recommended to use environment variable: LLMCORE_STORAGE_SESSION_DB_URL
  # Example: db_url = "postgresql://user:pass@host:port/dbname"
  db_url = ""

  # Optional table name for database storage.
  # table_name = "llmcore_sessions"

  # Vector storage configuration: for RAG documents and embeddings.
  [storage.vector]
  # Type: 'chromadb', 'pgvector' (PostgreSQL with pgvector extension - not implemented yet)
  type = "chromadb"

  # Default collection name used for RAG if not specified in API calls.
  # Collections in vector stores are reusable across different sessions or applications.
  default_collection = "llmcore_default_rag"

  # Path for file-based vector stores (e.g., chromadb persistent client).
  # '~' will be expanded to the user's home directory.
  path = "~/.llmcore/chroma_db"

  # Connection URL for database vector stores (e.g., pgvector).
  # Recommended to use environment variable: LLMCORE_STORAGE_VECTOR_DB_URL
  # Example: db_url = "postgresql://user:pass@host:port/dbname"
  db_url = ""

  # Optional table name for database vector storage.
  # table_name = "llmcore_vectors"


# --- Embedding Model Configurations ---
# Configures embedding models used for RAG.
[embedding]
  # Configuration for specific embedding models if they require special setup,
  # like API keys for service-based ones. The 'llmcore.default_embedding_model'
  # setting determines which model is used by default.
  # If 'llmcore.default_embedding_model' is like "openai:text-embedding-3-small",
  # then settings from [embedding.openai] might be used.

  [embedding.openai]
  # API Key for OpenAI embeddings (if different from chat or if using only embeddings).
  # Recommended to use environment variable: LLMCORE_EMBEDDING_OPENAI_API_KEY
  # If empty and providers.openai.api_key is set, that might be reused by the provider logic.
  # api_key = ""
  # Default OpenAI model for embeddings.
  default_model = "text-embedding-3-small" # Example, check latest models

  [embedding.google]
  # API Key for Google AI (Gemini) embeddings.
  # Recommended to use environment variable: LLMCORE_EMBEDDING_GOOGLE_API_KEY
  # api_key = ""
  # Default Google AI model for embeddings.
  default_model = "models/embedding-001" # Example, check latest models

  # For local sentence-transformers, the model name/path is usually specified directly
  # in 'llmcore.default_embedding_model' (e.g., "all-MiniLM-L6-v2").
  # No specific section needed here unless you want to group related settings.
  [embedding.sentence_transformer]
  # device = "cpu" # Example: "cuda", "cpu", "mps" (device to run model on)


# --- Context Management Configurations ---
# Defines strategies for how context is built and managed for LLM prompts.
[context_management]
  # Default number of documents to retrieve for RAG.
  rag_retrieval_k = 3

  # Strategy for combining RAG results with conversation history.
  # 'prepend_system': RAG context is added as part of a system message or preamble.
  # 'prepend_user': RAG context is added before the latest user message. (Not fully implemented yet)
  rag_combination_strategy = "prepend_system"

  # Strategy for selecting history messages to fit within the token limit.
  # 'last_n_tokens': Prioritizes keeping the most recent messages that fit the token budget.
  # 'last_n_messages': Keeps a fixed number of the most recent messages (less precise for token limits - Not implemented yet).
  history_selection_strategy = "last_n_tokens"

  # Number of tokens to reserve for the LLM's response generation.
  # This ensures there's space for the model to actually write its answer.
  reserved_response_tokens = 500

  # Strategy for handling context overflow when total tokens (history + RAG + prompt) exceed the limit.
  # 'history': Truncate older history messages first.
  # 'rag': Truncate less relevant RAG documents first.
  truncation_priority = "history"

  # Minimum number of history messages (excluding system message) to try and keep during truncation.
  # This helps maintain some conversational flow even when context is tight.
  minimum_history_messages = 1

  # If MCP (Model Context Protocol) is enabled, you might specify a version.
  # mcp_version = "v1"
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
  * Set Ollama Host: `export LLMCORE_PROVIDERS__OLLAMA__HOST="http://192.168.1.100:11434"`

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
    # RAG Parameters
    enable_rag=False, # Optional: Enable Retrieval Augmented Generation
    rag_retrieval_k=3, # Optional: Number of documents to retrieve for RAG
    rag_collection_name="my_docs", # Optional: Vector store collection name for RAG
    # Provider-specific arguments
    temperature=0.7,
    max_tokens=150 # Note: Often refers to *response* length limit
)
```

### Parameters

  * `message` (str): The user's input message.
  * `session_id` (Optional\[str]): The ID of the conversation session to use or create. If `None`, the chat is stateless (no history saved or loaded). If an ID is provided but doesn't exist, a new persistent session with that ID is created.
  * `system_message` (Optional\[str]): A message defining the role or behavior of the assistant (e.g., "You are a helpful AI assistant specialized in Python."). This is typically used at the start of a new session. Providing it for an existing session might be ignored depending on context management strategy.
  * `provider_name` (Optional\[str]): The name of the LLM provider to use (e.g., `"openai"`, `"ollama"`). Overrides the configured default.
  * `model_name` (Optional\[str]): The specific model identifier for the chosen provider (e.g., `"gpt-4o"`, `"llama3"`). Overrides the provider's default model.
  * `stream` (bool): If `True`, returns an `AsyncGenerator[str, None]` yielding text chunks as they arrive. If `False` (default), returns the complete response content as a single string after the LLM finishes.
  * `save_session` (bool): If `True` (default) and a `session_id` is provided (indicating a persistent session), the user message and the assistant's full response are saved to the persistent session storage. Ignored if `session_id` is `None`.
  * `enable_rag` (bool): If `True` (default: `False`), enables Retrieval Augmented Generation by searching the configured vector store for context relevant to the `message`.
  * `rag_retrieval_k` (Optional\[int]): Number of documents to retrieve for RAG. Overrides the default from configuration (`context_management.rag_retrieval_k`) if provided.
  * `rag_collection_name` (Optional\[str]): Name of the vector store collection to use for RAG. Overrides the default from configuration (`storage.vector.default_collection`) if provided.
  * `**provider_kwargs`: Additional keyword arguments passed directly to the selected provider's API call (e.g., `temperature=0.7`, `max_tokens=100`, `top_p=0.9`). Consult the specific provider's documentation for available parameters. Note: `max_tokens` here usually refers to the *response* length limit, not the context window limit.

### Simple Chat Example

```python
import asyncio
from llmcore import LLMCore

async def simple_chat():
    # Use async with for automatic cleanup
    async with await LLMCore.create() as llm:
        # Using default provider (e.g., Ollama)
        response = await llm.chat("What is the main export of Brazil?")
        print(f"LLM: {response}")

# asyncio.run(simple_chat())
```

### Streaming Chat Example

```python
import asyncio
from llmcore import LLMCore

async def stream_chat():
    async with await LLMCore.create() as llm:
        print("LLM (Streaming): ", end="", flush=True)
        async_generator = await llm.chat(
            "Write a short poem about asynchronous programming.",
            stream=True
        )
        async for chunk in async_generator:
            print(chunk, end="", flush=True)
        print("\n--- Stream Complete ---")

# asyncio.run(stream_chat())
```

### Provider & Model Selection

```python
import asyncio
from llmcore import LLMCore, ProviderError

async def specific_model_chat():
    # Ensure LLMCORE_PROVIDERS_ANTHROPIC_API_KEY is set in environment
    async with await LLMCore.create() as llm:
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

# asyncio.run(specific_model_chat())
```

-----

## Session Management

LLMCore allows you to maintain conversation history using sessions. When you provide a `session_id` to the `chat` method:

  * If the session exists, the library loads previous messages (respecting context limits).
  * If the session does not exist, a new persistent session with that ID is created.
  * If `save_session=True` (default), the new turn (user message + assistant response) is saved to the session storage.

### Using Sessions

```python
import asyncio
from llmcore import LLMCore
import uuid

async def session_example():
    # Generate a unique ID for a new persistent session
    session_id = f"project_brainstorm_{uuid.uuid4().hex[:8]}"

    async with await LLMCore.create() as llm:
        print(f"Starting or continuing session: {session_id}")
        try:
            # First interaction - creates the session if it doesn't exist
            response1 = await llm.chat(
                "Let's brainstorm some project ideas using Python.",
                session_id=session_id,
                system_message="You are a creative brainstorming partner."
                # save_session=True is default
            )
            print(f"LLM: {response1}")

            # Second interaction in the same session
            response2 = await llm.chat(
                "Okay, how about an idea related to data visualization?",
                session_id=session_id
            )
            print(f"LLM: {response2}")

            # Third interaction
            response3 = await llm.chat(
                "Expand on the data visualization idea.",
                session_id=session_id
            )
            print(f"LLM: {response3}")

        except Exception as e:
             print(f"An error occurred: {e}")

# asyncio.run(session_example())
```

### Listing Sessions (`list_sessions`)

Retrieve metadata about all saved persistent sessions.

```python
import asyncio
from llmcore import LLMCore

async def list_all_sessions():
    async with await LLMCore.create() as llm:
        try:
            sessions = await llm.list_sessions() # Changed from llm.list_sessions()
            if not sessions:
                print("No saved sessions found.")
            else:
                print("Saved Sessions:")
                for sess_info in sessions:
                     # Access keys directly, handle potential missing keys gracefully
                     sess_id = sess_info.get('id', 'N/A')
                     sess_name = sess_info.get('name', 'N/A')
                     msg_count = sess_info.get('message_count', 0)
                     updated_at = sess_info.get('updated_at', 'N/A')
                     print(f"- ID: {sess_id}, Name: {sess_name}, Msgs: {msg_count}, Updated: {updated_at}")
        except Exception as e:
             print(f"Error listing sessions: {e}")


# asyncio.run(list_all_sessions())
```

### Retrieving a Session (`get_session`)

Load a specific persistent session object, including all its messages.

```python
import asyncio
from llmcore import LLMCore, SessionNotFoundError

async def retrieve_specific_session(session_id_to_load):
    async with await LLMCore.create() as llm:
        try:
            session = await llm.get_session(session_id_to_load) # Changed from llm.get_session()
            if session:
                # Note: session.name might be None if not set
                print(f"Session '{session.id}' (Name: {session.name or 'Not Set'}) loaded.")
                print(f"Total messages: {len(session.messages)}")
                # Access messages: session.messages[0].content, etc.
                if session.messages:
                     # Use repr() for potentially multi-line content
                     last_msg_content = repr(session.messages[-1].content[:80] + '...')
                     print(f"Last message: [{session.messages[-1].role}] {last_msg_content}")
            else:
                 # This case is now handled by SessionNotFoundError being raised
                 # by load_or_create_session if the ID doesn't exist.
                 # get_session itself might return None if storage fails gracefully,
                 # but SessionNotFoundError is more explicit for non-existence.
                 print(f"Session with ID '{session_id_to_load}' not found (get_session returned None).")

        except SessionNotFoundError: # Explicitly catch if ID doesn't exist
             print(f"Session with ID '{session_id_to_load}' not found.")
        except Exception as e:
             print(f"Error retrieving session: {e}")

# Example usage:
# asyncio.run(retrieve_specific_session("project_brainstorm_...")) # Use an actual ID
```

### Deleting a Session (`delete_session`)

Permanently remove a persistent session from storage.

```python
import asyncio
from llmcore import LLMCore

async def delete_specific_session(session_id_to_delete):
    async with await LLMCore.create() as llm:
        try:
            deleted = await llm.delete_session(session_id_to_delete) # Changed from llm.delete_session()
            if deleted:
                print(f"Session '{session_id_to_delete}' deleted successfully.")
            else:
                print(f"Session '{session_id_to_delete}' not found or could not be deleted.")
        except Exception as e:
             print(f"Error deleting session: {e}")

# Example usage:
# asyncio.run(delete_specific_session("project_brainstorm_...")) # Use an actual ID
```

---

## Retrieval Augmented Generation (RAG)

RAG allows the LLM to access external information stored in a vector database during conversation. This helps the model answer questions about specific documents or data it wasn't originally trained on.

### Concept

1.  **Configure:** Ensure your LLMCore configuration specifies a vector storage backend (`storage.vector.type`, e.g., `"chromadb"`) and a compatible embedding model (`llmcore.default_embedding_model`, e.g., `"all-MiniLM-L6-v2"`). Install necessary dependencies (e.g., `pip install llmcore[chromadb,sentence_transformers]`).
2.  **Add Documents:** Use `llm.add_document_to_vector_store()` or `llm.add_documents_to_vector_store()` to add text documents (e.g., from files, database records) to a specific collection in the vector store. LLMCore automatically generates vector embeddings for these documents using the configured embedding model.
3.  **Chat with RAG:** Call `llm.chat()` with `enable_rag=True`. LLMCore will:
      * Generate an embedding for your query message.
      * Search the specified (or default) vector store collection for documents with similar embeddings.
      * Retrieve the content of the most relevant documents (controlled by `rag_retrieval_k`).
      * Inject this retrieved content into the context sent to the LLM, along with your message and conversation history (strategy defined by `context_management.rag_combination_strategy`).
4.  **Informed Response:** The LLM uses the provided document context to generate a more informed and accurate response.

### Adding Documents

```python
import asyncio
from llmcore import LLMCore, VectorStorageError, EmbeddingError

async def add_rag_docs():
    docs_to_add = [
        {"id": "doc1", "content": "LLMCore uses confy for configuration.", "metadata": {"source": "readme"}},
        {"content": "ChromaDB is a supported vector store.", "metadata": {"source": "config"}}, # ID will be generated
    ]
    collection = "my_llmcore_docs"

    async with await LLMCore.create() as llm:
        try:
            # Add single document
            doc1_id = await llm.add_document_to_vector_store(
                content="RAG enhances LLMs with external knowledge.",
                metadata={"topic": "rag"},
                collection_name=collection
            )
            print(f"Added single document with ID: {doc1_id}")

            # Add multiple documents
            added_ids = await llm.add_documents_to_vector_store(
                documents=docs_to_add,
                collection_name=collection
            )
            print(f"Added batch documents with IDs: {added_ids}")

        except (VectorStorageError, EmbeddingError) as e:
            print(f"Error adding documents: {e}")

# asyncio.run(add_rag_docs())
```

  * `llm.add_document_to_vector_store(content, *, metadata=None, doc_id=None, collection_name=None)`: Adds a single document. Returns the document ID (generated if `doc_id` is `None`).
  * `llm.add_documents_to_vector_store(documents, *, collection_name=None)`: Adds a list of documents. `documents` is a list of dicts (`{"content": str, "metadata": Optional[Dict], "id": Optional[str]}`). Returns a list of document IDs.

### Searching Documents

While RAG happens automatically during chat when `enable_rag=True`, you can also perform direct similarity searches.

```python
import asyncio
from llmcore import LLMCore, VectorStorageError, EmbeddingError

async def search_rag_docs(query: str, collection: str):
    async with await LLMCore.create() as llm:
        try:
            results = await llm.search_vector_store(
                query=query,
                k=2, # Number of results to retrieve
                collection_name=collection,
                # filter_metadata={"source": "readme"} # Optional filter
            )
            print(f"Search results for '{query}':")
            if not results:
                print("  No results found.")
            for doc in results:
                print(f"  - ID: {doc.id}, Score: {doc.score:.4f}, Content: {doc.content[:60]}...")
                print(f"    Metadata: {doc.metadata}")
        except (VectorStorageError, EmbeddingError) as e:
            print(f"Error searching documents: {e}")

# asyncio.run(search_rag_docs("How is configuration managed?", "my_llmcore_docs"))
```

  * `llm.search_vector_store(query, *, k, collection_name=None, filter_metadata=None)`: Searches the vector store. Returns a list of `ContextDocument` objects.

### Deleting Documents

```python
import asyncio
from llmcore import LLMCore, VectorStorageError

async def delete_rag_docs(doc_ids: list, collection: str):
    async with await LLMCore.create() as llm:
        try:
            deleted = await llm.delete_documents_from_vector_store(
                document_ids=doc_ids,
                collection_name=collection
            )
            if deleted:
                print(f"Attempted deletion for IDs: {doc_ids}. Success: {deleted}")
            else:
                print(f"Deletion attempt failed or IDs not found for: {doc_ids}")
        except VectorStorageError as e:
            print(f"Error deleting documents: {e}")

# Example: Use IDs returned from add_documents...
# asyncio.run(delete_rag_docs(['doc1', 'generated_id_...'], "my_llmcore_docs"))
```

  * `llm.delete_documents_from_vector_store(document_ids, *, collection_name=None)`: Deletes documents by ID. Returns `True` if the operation was attempted successfully.

### Chatting with RAG

Simply set `enable_rag=True` in your `llm.chat()` call.

```python
import asyncio
from llmcore import LLMCore

async def chat_with_rag(question: str, collection: str):
     async with await LLMCore.create() as llm:
        try:
            response = await llm.chat(
                message=question,
                enable_rag=True,
                rag_collection_name=collection,
                rag_retrieval_k=2, # Optional: override default K
                stream=False, # Can also stream RAG responses
                system_message="Answer based *only* on the provided context documents." # Guide the LLM
            )
            print(f"RAG Response: {response}")
        except Exception as e:
            print(f"Error during RAG chat: {e}")

# asyncio.run(chat_with_rag("Tell me about confy configuration.", "my_llmcore_docs"))
```

See the `examples/rag_example.py` script for a complete RAG workflow demonstration.

---

## Provider Information

You can query LLMCore for information about the configured providers and their models.

### `get_available_providers`

Lists the names of all providers that were successfully loaded based on your configuration.

```python
import asyncio
from llmcore import LLMCore

async def show_providers():
    async with await LLMCore.create() as llm:
        providers = llm.get_available_providers()
        print("Loaded Providers:", providers)

# asyncio.run(show_providers())
# Output might be: Loaded Providers: ['ollama', 'openai', 'anthropic', 'gemini'] (if all configured)
```

### `get_models_for_provider`

Lists the models available for a *specific* loaded provider. Note that this might return a static list defined in the provider code or potentially involve an API call, depending on the provider implementation.

```python
import asyncio
from llmcore import LLMCore, ConfigError

async def show_provider_models(provider_name):
    async with await LLMCore.create() as llm:
        try:
            models = llm.get_models_for_provider(provider_name)
            print(f"Models for '{provider_name}':", models)
        except ConfigError as e:
             print(e) # Handle case where provider isn't configured/loaded
        except Exception as e:
             print(f"Error getting models: {e}")


# asyncio.run(show_provider_models("openai"))
# Output might be: Models for 'openai': ['gpt-4', 'gpt-4o', 'gpt-3.5-turbo', ...]
```

---

## Error Handling

LLMCore defines custom exceptions in `llmcore.exceptions` to help you handle specific issues:

  * `LLMCoreError`: Base exception for library errors.
  * `ConfigError`: Errors during configuration loading or validation.
  * `ProviderError`: Errors originating from an LLM provider API (e.g., connection issues, authentication failure, rate limits, model not found).
  * `StorageError`: Base class for storage-related errors.
      * `SessionStorageError`: Errors specific to session storage.
      * `VectorStorageError`: Errors specific to vector storage.
  * `SessionNotFoundError`: Raised by `SessionManager` when a specified `session_id` is provided but not found during loading.
  * `ContextError`: Base class for context management errors.
      * `ContextLengthError`: Raised when context exceeds the model's limit even after truncation.
  * `EmbeddingError`: Errors during embedding generation or model loading.
  * `MCPError`: Errors related to MCP formatting (Phase 3).

<!-- end list -->

```python
import asyncio
from llmcore import (
    LLMCore, LLMCoreError, ProviderError, ContextLengthError,
    ConfigError, SessionNotFoundError, VectorStorageError, EmbeddingError
)

async def chat_with_error_handling():
    llm = None # Define outside try block for finally
    try:
        # Use async with for automatic cleanup if initialization succeeds
        async with await LLMCore.create() as llm:
            response = await llm.chat(
                "This is a very long message..." * 10000, # Intentionally long
                provider_name="openai",
                model_name="gpt-3.5-turbo-0613" # Model with smaller context (4k)
            )
            print(f"LLM: {response}")

    except ContextLengthError as e:
        print(f"Error: Context too long for model '{e.model_name}'. Limit: {e.limit}, Actual: {e.actual}")
    except ProviderError as e:
        print(f"Error with provider '{e.provider_name}': {e}")
    except SessionNotFoundError as e:
         print(f"Session Error: {e}")
    except VectorStorageError as e:
         print(f"Vector Storage Error: {e}")
    except EmbeddingError as e:
         print(f"Embedding Error: {e}")
    except ConfigError as e:
        print(f"Configuration Error: {e}")
    except LLMCoreError as e: # Catch base LLMCore errors
        print(f"An LLMCore error occurred: {e}")
    except Exception as e: # Catch any other unexpected errors
        print(f"An unexpected error occurred: {e}")
    # No finally block needed for llm.close() when using 'async with'

# asyncio.run(chat_with_error_handling())
```

---

## Full Examples

See the `examples/` directory in the repository for runnable scripts demonstrating various use cases:

  * `simple_chat.py`: Basic stateless chat interaction.
  * `session_chat.py`: Demonstrates conversation history using persistent sessions.
  * `streaming_chat.py`: Shows how to handle streaming responses with sessions.
  * `rag_example.py`: Demonstrates adding documents and chatting with RAG enabled.
