# **LLMCore: Specification v1.0 (Final Draft)**

> ### Design & Implementation Plan

[TOC]

## 1. Introduction & Vision

This document outlines the specification for a Python library, **`LLMCore`**, designed to provide a unified, flexible, and extensible interface for interacting with various Large Language Models (LLMs). The primary goal is to refactor the core functionalities of the existing `llmchat` CLI tool into a reusable library/module that can be easily integrated into other Python projects.

**`LLMCore`** will offer a robust foundation for developers to build applications that require LLM-driven chat capabilities, session management, and advanced context handling, including support for Retrieval Augmented Generation (RAG). It will leverage the `confy` library for comprehensive and flexible configuration management, allowing users to easily tailor LLM providers, models, storage backends (including vector stores), embedding models, context strategies, and other operational parameters. The library will also aim to support emerging standards like the Model Context Protocol (MCP) where applicable, potentially leveraging the official Python SDK.

**Core Vision:** To empower developers with a simple yet powerful toolkit for integrating diverse LLM functionalities into their applications, abstracting away the complexities of individual provider APIs, offering sophisticated context management, and providing a consistent, configurable operational model.

## 2. Core Principles

*   **Modularity & Reusability:** Designed as a library first, ensuring components (`providers`, `storage`, `context`, `session`) are easily importable and usable in various Python applications. The existing `llmchat` CLI will become a primary consumer of this library.
*   **Extensibility:** Provide clear abstract base classes and interfaces for adding new LLM providers (e.g., `BaseProvider`), storage backends (`BaseSessionStorage`, `BaseVectorStorage`), context retrieval strategies, and potentially other related functionalities.
*   **Configurability (`confy` Integration):** All aspects of the library's operation (API keys, default models, storage paths/connections, provider settings, embedding models, vector store configurations, context strategies) will be configurable through `confy`, supporting defaults, TOML/JSON files, environment variables, and dictionary overrides.
*   **Provider Agnostic:** Offer a consistent high-level API (`LLMCore.chat()`) for interacting with different LLM providers (OpenAI, Anthropic, Ollama, Gemini, and others), abstracting provider-specific details where possible.
*   **Stateful Interactions & Advanced Context Management:**
    *   Support for managing conversation sessions/history via configurable storage backends.
    *   Robust mechanisms for managing the LLM's context window, including accurate, provider-specific token counting and intelligent context selection/truncation strategies when exceeding limits.
    *   Support for Retrieval Augmented Generation (RAG) by integrating with persistent and reusable vector store collections (ChromaDB, PostgreSQL+pgvector).
    *   Support for Model Context Protocol (MCP) to standardize context exchange with models, potentially using the `modelcontextprotocol` Python SDK based on configuration.
*   **Developer-Friendly API:** Provide a clear, intuitive, well-documented, and type-hinted Python API for developers, simplifying common LLM interaction patterns.
*   **Asynchronous Design:** Leverage Python's `asyncio` for non-blocking I/O operations, especially for provider API calls and streaming responses, making it suitable for integration into asynchronous applications.

## 3. Architecture

**`LLMCore`** will be structured as a Python library with distinct modules responsible for different aspects of LLM interaction. The design emphasizes separation of concerns and clear interfaces between components.

```mermaid
graph TD
    subgraph External Application
        App["Application (e.g., llmchat CLI, Web App)"]
    end

    subgraph LLMCore Library
        direction LR
        CoreAPI["Core API Facade (LLMCore class)"]

        subgraph Configuration
            Confy["Configuration (confy)"]
        end

        subgraph Provider Management
            ProvMgr["Provider Manager"]
            ProvBase["(BaseProvider Interface)"]
            ProvOpenAI["OpenAI Provider"]
            ProvAnthropic["Anthropic Provider"]
            ProvOllama["Ollama Provider"]
            ProvGemini["Gemini Provider"]
        end

        subgraph Session Management
            SessMgr["Session Manager"]
        end

        subgraph Storage Management
            direction TB
            StorMgr["Storage Manager"]
            subgraph Session Storage
                StorSessBase["(BaseSessionStorage)"]
                StorSessJSON["JSON Storage"]
                StorSessSQLite["SQLite Storage"]
                StorSessPG["PostgreSQL Storage"]
            end
            subgraph Vector Storage
                StorVecBase["(BaseVectorStorage)<br/>[Collections Reusable]"]
                StorVecChroma["ChromaDB Storage"]
                StorVecPG["PostgreSQL+pgvector Storage"]
            end
        end

        subgraph Context Management
            CtxMgr["Context Manager<br/>[MCP Support (SDK)]<br/>[RAG Logic]"]
        end

        subgraph Embedding Management
            EmbMgr["Embedding Manager"]
            EmbBase["(BaseEmbeddingModel)"]
            EmbST["SentenceTransformer"]
            EmbOpenAI["OpenAI Embedding"]
            EmbGoogle["GoogleAI Embedding"]
        end

        %% Core Relationships within LLMCore
        CoreAPI --> Confy
        CoreAPI --> ProvMgr
        CoreAPI --> SessMgr
        CoreAPI --> StorMgr
        CoreAPI --> CtxMgr
        CoreAPI --> EmbMgr

        %% Manager Dependencies
        ProvMgr --> ProvBase
        ProvBase --> ProvOpenAI
        ProvBase --> ProvAnthropic
        ProvBase --> ProvOllama
        ProvBase --> ProvGemini

        SessMgr --> StorMgr

        StorMgr --> StorSessBase
        StorMgr --> StorVecBase
        StorSessBase --> StorSessJSON
        StorSessBase --> StorSessSQLite
        StorSessBase --> StorSessPG
        StorVecBase --> StorVecChroma
        StorVecBase --> StorVecPG

        CtxMgr --> ProvMgr
        CtxMgr --> SessMgr
        CtxMgr --> StorMgr
        CtxMgr --> EmbMgr

        EmbMgr --> EmbBase
        EmbBase --> EmbST
        EmbBase --> EmbOpenAI
        EmbBase --> EmbGoogle

    end

    %% External Usage
    App --> CoreAPI

    %% Style (Optional)
    classDef manager fill:#f9f,stroke:#333,stroke-width:2px;
    classDef interface fill:#ccf,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5;
    classDef storage fill:#ff9,stroke:#333,stroke-width:1px;
    classDef provider fill:#9cf,stroke:#333,stroke-width:1px;
    classDef core fill:#9f9,stroke:#333,stroke-width:2px;

    class CoreAPI core;
    class ProvMgr,SessMgr,StorMgr,CtxMgr,EmbMgr manager;
    class ProvBase,StorSessBase,StorVecBase,EmbBase interface;
    class StorSessJSON,StorSessSQLite,StorSessPG,StorVecChroma,StorVecPG storage;
    class ProvOpenAI,ProvAnthropic,ProvOllama,ProvGemini,EmbST,EmbOpenAI,EmbGoogle provider;
    class Confy storage;
```
> **(Conceptual Diagram Description: A central box labeled "`LLMCore` (Python Library)" contains: "Core API Facade (`LLMCore` class)", "Provider Manager" (pointing to "BaseProvider Interface" and specific providers like OpenAI, Anthropic, Ollama, Gemini), "Session Manager", "Storage Manager" (pointing to "BaseSessionStorage", "BaseVectorStorage" and specific stores like JSON, SQLite, ChromaDB, PostgreSQL+pgvector \[noting Collections are Reusable\]), "Context Manager" (interacting with Vector Stores for RAG & potentially using MCP SDK), "Embedding Manager", and "Configuration (`confy`)". "Context Manager" also notes "MCP Support (SDK)". An external box "Application (e.g., `llmchat` CLI, Web App)" points to "`LLMCore`".)**

**Key Modules/Components:**

*   **Core API Facade (`llmcore.LLMCore`):**
    *   The primary class developers interact with.
    *   Initializes and holds instances of managers (`confy`, providers, storage, context, embedding).
    *   Provides high-level methods like `chat()`, `add_document_to_vector_store()`, `search_vector_store()`, `get_session()`, etc.
    *   Orchestrates the flow between different managers for core operations.
*   **Configuration (`llmcore.config` integrated with `confy`):**
    *   Manages all library configurations using `confy`.
    *   Loads settings for providers, storage, embedding models, context strategies, API keys, etc.
    *   Instantiated by the `LLMCore` class and passed to other managers.
*   **Provider Manager (`llmcore.providers`):**
    *   `BaseProvider`: Abstract base class defining the interface for LLM providers (`chat_completion`, `count_tokens`, etc.).
    *   Concrete provider implementations (`OpenAIProvider`, `AnthropicProvider`, `OllamaProvider`, `GeminiProvider`) inheriting from `BaseProvider`.
    *   Dynamically loads and initializes configured providers based on settings from `confy`. Responsible for handling provider-specific API interactions and authentication.
*   **Session Manager (`llmcore.sessions`):**
    *   Manages `ChatSession` objects (conversation history).
    *   Handles session creation, retrieval, and updates.
    *   Interacts with the `StorageManager` (specifically `BaseSessionStorage`) for persistence.
*   **Storage Manager (`llmcore.storage`):**
    *   `BaseSessionStorage`: Interface for session metadata storage (JSON, SQLite, PostgreSQL).
    *   `BaseVectorStorage`: Interface for vector embedding storage (ChromaDB, PostgreSQL+pgvector).
    *   Concrete storage implementations for each type.
    *   Dynamically loads and initializes configured storage backends based on `confy` settings.
*   **Context Manager (`llmcore.context`):**
    *   Responsible for assembling the final context payload sent to the LLM provider.
    *   Manages token limits using provider-specific counting via the `ProviderManager`.
    *   Handles RAG: interacts with `EmbeddingManager` and `StorageManager` (`BaseVectorStorage`) for document retrieval.
    *   Implements context combination and truncation strategies based on configuration.
    *   Optionally formats context using the MCP SDK if enabled.
*   **Embedding Manager (`llmcore.embedding`):**
    *   `BaseEmbeddingModel`: Abstract interface for embedding generation (`generate_embedding`, `generate_embeddings`).
    *   Concrete implementations (e.g., `SentenceTransformerEmbedding`, `OpenAIEmbedding`, `GoogleAIEmbedding`).
    *   Dynamically loads and initializes the configured embedding model based on `confy` settings. Used by the `ContextManager` for RAG and potentially by storage backends if they don't handle embeddings internally.
*   **Models (`llmcore.models`):**
    *   Defines core data structures using Pydantic or dataclasses: `Message`, `Role`, `ChatSession`, `ContextDocument`, configuration models.
*   **Exceptions (`llmcore.exceptions`):**
    *   Defines custom exception classes for specific error conditions (`LLMCoreError`, `ProviderError`, `StorageError`, `ContextLengthError`, etc.).
*   **Utilities (`llmcore.utils`):**
    *   Helper functions (e.g., async utilities, data validation helpers).

### **Proposed project directory tree structure**

```bash
llmcore/
├── .gitignore          # Git ignore file
├── LICENSE             # Project license file (e.g., MIT)
├── README.md           # Project description, usage, setup
├── pyproject.toml      # Build system configuration, dependencies, project metadata
├── docs/               # Documentation files (Sphinx, MkDocs, etc.)
│   ├── conf.py         # Sphinx config (example)
│   ├── index.rst       # Main doc page (example)
│   └── ...
├── examples/           # Example scripts showing how to use the library
│   ├── simple_chat.py
│   ├── session_chat.py
│   ├── streaming_chat.py
│   └── rag_example.py
├── src/
│   └── llmcore/        # Main library source code
│       ├── __init__.py # Makes llmcore a package, exports main class/functions
│       ├── api.py      # Could contain the main LLMCore class definition
│       ├── config/     # Default configurations
│       │   ├── __init__.py
│       │   └── default_config.toml
│       ├── context/    # Context management logic
│       │   ├── __init__.py
│       │   └── manager.py # ContextManager class
│       ├── embedding/  # Embedding model handling
│       │   ├── __init__.py
│       │   ├── base.py   # BaseEmbeddingModel ABC
│       │   ├── sentence_transformer.py # SentenceTransformerEmbedding
│       │   ├── openai.py # OpenAIEmbedding
│       │   └── google.py # GoogleAIEmbedding
│       ├── exceptions.py # Custom exception classes
│       ├── models.py     # Data models (Message, ChatSession, ContextDocument, Role)
│       ├── providers/    # LLM Provider implementations
│       │   ├── __init__.py
│       │   ├── base.py   # BaseProvider ABC
│       │   ├── openai_provider.py
│       │   ├── anthropic_provider.py
│       │   ├── ollama_provider.py
│       │   └── gemini_provider.py
│       ├── sessions/     # Session management logic
│       │   ├── __init__.py
│       │   └── manager.py # SessionManager class
│       ├── storage/      # Storage backend implementations
│       │   ├── __init__.py
│       │   ├── base_session.py # BaseSessionStorage ABC
│       │   ├── base_vector.py  # BaseVectorStorage ABC
│       │   ├── json_session.py
│       │   ├── sqlite_session.py
│       │   ├── postgres_storage.py # Could implement both session & vector
│       │   └── chromadb_vector.py
│       └── utils/        # Utility functions
│           ├── __init__.py
│           └── async_utils.py # Example utility module
└── tests/              # Unit and integration tests
    ├── __init__.py
    ├── conftest.py     # Pytest fixtures
    ├── test_api.py     # Tests for the main LLMCore class/API
    ├── context/
    │   ├── __init__.py
    │   └── test_manager.py
    ├── embedding/
    │   ├── __init__.py
    │   └── test_embedding_models.py
    ├── providers/
    │   ├── __init__.py
    │   └── test_providers.py # Test individual providers
    ├── sessions/
    │   ├── __init__.py
    │   └── test_manager.py
    └── storage/
        ├── __init__.py
        └── test_storage_backends.py # Test session and vector stores
```

**Explanation:**

1.  **Root Directory:** Contains project-level files (`pyproject.toml`, `README.md`, `LICENSE`, `.gitignore`).
2.  **`src/llmcore/`:** The actual Python package source code resides here, following the `src` layout pattern.
3.  **`src/llmcore/__init__.py`:** Marks the directory as a package and is the place to expose the main public API elements (like the `LLMCore` class, core models, and exceptions) for easy import (`from llmcore import LLMCore`).
4.  **Subdirectories within `src/llmcore/`:** Each major component (providers, storage, context, embedding, config, models, exceptions, utils) gets its own subdirectory, promoting modularity.
5.  **Base Classes:** Abstract base classes (`base.py`, `base_session.py`, `base_vector.py`, `base_embedding.py`) are defined within their respective modules to establish interfaces.
6.  **Implementations:** Concrete implementations reside in separate files within their module directories (e.g., `openai_provider.py`, `json_session.py`).
7.  **`config/`:** Holds default configuration files packaged with the library.
8.  **`tests/`:** Located at the root, parallel to `src/`. It mirrors the source structure to make finding tests easy. `conftest.py` holds shared pytest fixtures.
9.  **`docs/`:** Standard location for documentation source files.
10. **`examples/`:** Provides practical scripts demonstrating library usage.

## 4. Core Components Deep Dive

### 4.1. Configuration (`confy` Integration)

* **Responsibility:** Provide a flexible, unified, and layered approach to configure all operational aspects of `LLMCore`.

* **Implementation:**

    *   The `LLMCore` class constructor will initialize a `confy.Config` instance.
    *   It will accept optional arguments (`config_overrides`, `config_file_path`, `env_prefix`) to customize `confy`'s loading behavior.
    *   `LLMCore` will ship with a default configuration structure (likely defined as a dictionary or loaded from a packaged TOML file) covering all configurable parameters with sensible defaults.
    *   The precedence order defined by `confy` (Defaults -> Config File -> .env -> Environment Variables -> Overrides Dictionary) will be used.
    *   The initialized `confy.Config` object will be passed to other managers (`ProviderManager`, `StorageManager`, `ContextManager`, `EmbeddingManager`) during their initialization.

* **Key Configuration Sections (Conceptual `default_config.toml`):**
  ```toml
  
    # Default configuration structure for LLMCore
  
    [llmcore]
  
    # Default provider to use if not specified in API calls
  
    default_provider = "openai"
  
    # Default embedding model for RAG (can be local path or identifier for a service)
  
    default_embedding_model = "all-MiniLM-L6-v2"
  
    # Global flag to enable/disable MCP formatting (can be overridden per provider)
  
    enable_mcp = false
  
    # Log level for the library
  
    log_level = "INFO"
  
    # --- Provider Configurations ---
  
    [providers]
      [providers.openai]
  
      # API Key: Recommend using environment variable LLMCORE_PROVIDERS_OPENAI_API_KEY
  
      api_key = ""
      default_model = "gpt-4-turbo"
      timeout = 60
  
      # use_mcp = false # Provider-specific MCP toggle
  
      [providers.anthropic]
  
      # API Key: Recommend using environment variable LLMCORE_PROVIDERS_ANTHROPIC_API_KEY
  
      api_key = ""
      default_model = "claude-3-opus-20240229"
      timeout = 60
  
      # use_mcp = false
  
      [providers.ollama]
      base_url = "http://localhost:11434/api"
      default_model = "llama3"
      timeout = 120
  
      # Optional: Specify tokenizer for Ollama models (default: 'tiktoken_cl100k_base')
  
      # Options: 'tiktoken_cl100k_base', 'tiktoken_p50k_base', 'char_div_4'
  
      # tokenizer = "tiktoken_cl100k_base"
  
      # use_mcp = false
  
      [providers.gemini]
  
      # API Key: Recommend using environment variable LLMCORE_PROVIDERS_GEMINI_API_KEY
  
      api_key = ""
      default_model = "gemini-1.5-pro-latest"
  
      # Add other Gemini specific settings if needed (e.g., safety settings)
  
      # safety_settings = { HARM_CATEGORY_SEXUALLY_EXPLICIT = "BLOCK_NONE", ... }
  
      # use_mcp = false
  
    # --- Storage Configurations ---
  
    [storage]
  
      # Session storage configuration
  
      [storage.session]
  
      # Type: 'json', 'sqlite', 'postgres'
  
      type = "sqlite"
  
      # Path for file-based storage (json, sqlite)
  
      path = "~/.llmcore/sessions.db"
  
      # Connection URL for database storage (postgres)
  
      # db_url = "postgresql://user:pass@host:port/dbname" # Use env var LLMCORE_STORAGE_SESSION_DB_URL
  
      # table_name = "llmcore_sessions" # Optional for DB storage
  
      # Vector storage configuration
  
      [storage.vector]
  
      # Type: 'chromadb', 'pgvector'
  
      type = "chromadb"
  
      # Default collection name used for RAG if not specified in API calls
  
      default_collection = "llmcore_default_rag"
  
      # Path for file-based vector stores (chromadb)
  
      path = "~/.llmcore/chroma_db"
  
      # Connection URL for database vector stores (pgvector)
  
      # db_url = "postgresql://user:pass@host:port/dbname" # Use env var LLMCORE_STORAGE_VECTOR_DB_URL
  
      # table_name = "llmcore_vectors" # Optional for DB storage
  
    # --- Embedding Model Configurations ---
  
    [embedding]
  
      # Configuration for specific embedding models if needed (e.g., API keys for service-based ones)
  
      [embedding.openai]
  
      # api_key = "" # Use env var LLMCORE_EMBEDDING_OPENAI_API_KEY
  
      # default_model = "text-embedding-ada-002"
  
      [embedding.google]
  
      # api_key = "" # Use env var LLMCORE_EMBEDDING_GOOGLE_API_KEY
  
      # default_model = "models/embedding-001"
  
    # --- Context Management Configurations ---
  
    [context_management]
  
      # Default number of documents to retrieve for RAG
  
      rag_retrieval_k = 3
  
      # Strategy for combining RAG results with history: 'prepend_system', 'prepend_user'
  
      rag_combination_strategy = "prepend_system"
  
      # Strategy for selecting history messages: 'last_n_tokens', 'last_n_messages'
  
      history_selection_strategy = "last_n_tokens"
  
      # Tokens to reserve for the LLM's response during context calculation
  
      reserved_response_tokens = 500
  
      # Strategy for handling context overflow: 'history' (truncate history first), 'rag' (truncate RAG results first)
  
      truncation_priority = "history"
  
      # Minimum history messages to try and keep during truncation (excluding system)
  
      minimum_history_messages = 1
  
      # mcp_version = "v1" # If MCP versions need specifying
  ```

### 4.2. Provider Management

*   **Responsibility:** Abstract interactions with different LLM providers, handle authentication, manage API calls, and provide accurate token counting. Ensure extensibility for new providers.
*   **`BaseProvider` Interface (Key Methods):**
    *   `__init__(self, config: dict)`: Initialize with provider-specific config from `confy`.
    *   `async def chat_completion(self, context: Union[List[Message], MCPContextObject], model: Optional[str], stream: bool, **kwargs) -> Union[Dict, AsyncGenerator[Dict, None]]`: Sends the prepared context to the LLM API. Handles streaming logic internally, returning either the full response dict or an async generator of raw API chunks.
    *   `def count_tokens(self, text: str, model: Optional[str]) -> int`: Counts tokens for a given text string using the provider's specific tokenizer for the given model.
    *   `def count_message_tokens(self, messages: List[Message], model: Optional[str]) -> int`: Counts tokens for a list of messages, accounting for provider-specific overhead and using `count_tokens`.
    *   `def get_available_models(self) -> List[str]`: Returns a list of known/potentially available model names for the provider. Might involve an API call or return a hardcoded list.
    *   `def get_max_context_length(self, model: Optional[str]) -> int`: Returns the maximum context token limit for the specified model (or the provider's default).
    *   `def get_name(self) -> str`: Returns the unique name of the provider (e.g., "openai", "gemini").
*   **Implementations:**
    *   `OpenAIProvider`: Uses `openai` SDK, `tiktoken`.
    *   `AnthropicProvider`: Uses `anthropic` SDK.
    *   `GeminiProvider`: Uses `google-generativeai` SDK.
    *   `OllamaProvider`: Uses `aiohttp` for direct API calls. Implements configurable token counting (defaulting to `tiktoken`).
*   **Extensibility:** Developers can add support for a new provider by creating a new class inheriting from `BaseProvider`, implementing the required methods, and ensuring `confy` can load its configuration under the `[providers.<new_provider_name>]` section. The `ProviderManager` (internal to `LLMCore`) will dynamically instantiate providers based on configuration.

### 4.3. Session Management

*   **Responsibility:** Manage the lifecycle and content of conversations (`ChatSession` objects).
*   **Models:**
    *   `Message`: Represents a single turn in the conversation (role, content, timestamp, etc.).
    *   `Role`: Enum (`SYSTEM`, `USER`, `ASSISTANT`).
    *   `ChatSession`: Holds metadata and a list of `Message` objects for a conversation.
*   **Functionality:** Provides methods (used internally by `LLMCore`) to create, load (via `StorageManager`), add messages to, and retrieve messages from sessions.

### 4.4. Storage Management

*   **Responsibility:** Provide persistent storage for session history and vector embeddings, supporting multiple backends. Ensure vector collections are reusable.
*   **`BaseSessionStorage` Interface:** Defines methods for CRUD operations on `ChatSession` metadata and messages (`save_session`, `get_session`, `list_sessions`, `delete_session`).
*   **`BaseVectorStorage` Interface:** Defines methods for managing vector embeddings and associated documents (`add_documents`, `similarity_search`, `delete_documents`). Methods accept an optional `collection_name` to target specific reusable collections.
*   **Implementations:**
    *   Session: `JsonSessionStorage`, `SqliteSessionStorage`, `PostgresSessionStorage`.
    *   Vector: `ChromaVectorStorage`, `PgVectorStorage`.
    *   Note: `PostgresStorage` could potentially implement _both_ interfaces if using PostgreSQL for session data and vectors.
*   **Initialization:** The `StorageManager` (internal to `LLMCore`) reads the `confy` configuration (`storage.session.type`, `storage.vector.type`, and associated settings) and instantiates the appropriate storage backend classes.


### 4.5. Context Management

*   **Responsibility:** Assemble the final context payload for the LLM, handling token limits, RAG, and optional MCP formatting. This is a critical internal component ensuring API calls are valid and effective.
*   **Core Logic (`prepare_context` - Internal):**
    1.  **Input:** Current session state, latest user message, RAG settings, target provider/model, MCP flag.
    2.  **RAG (if enabled):**
        *   Generate query embedding via `EmbeddingManager`.
        *   Query `BaseVectorStorage` (using `rag_collection_name` or default) for top `k` documents.
        *   Extract text content from retrieved `ContextDocument`s.
    3.  **History Selection:** Retrieve conversation history (`Message` list). Select messages based on `history_selection_strategy` and token limits (using target provider's `count_message_tokens`), reserving space for prompt, RAG text, user message, and response. Prioritize system messages.
    4.  **Combine & Format:**
        *   Integrate retrieved RAG text with selected history and system prompt based on `rag_combination_strategy`.
        *   Structure the combined information:
            *   If MCP enabled: Use `modelcontextprotocol` SDK to create the MCP object.
            *   Else: Create the standard `List[Message]` payload.
    5.  **Truncation (if needed):** Calculate final token count. If over limit, apply `truncation_priority` strategy (remove oldest history or lowest-ranked RAG results) until within limits. Log warnings if truncation occurs.
    6.  **Error Check:** If essential context (e.g., system prompt + user message) still exceeds limits after truncation, raise `ContextLengthError`.
    7.  **Return:** Final context payload (`List[Message]` or `MCPContextObject`).
*   **Key Dependencies:** `SessionManager`, `StorageManager` (Vector), `EmbeddingManager`, `ProviderManager` (for token counting), `confy` (for strategies).

### 4.6. Embedding Management

*   **Responsibility:** Provide a consistent interface for generating text embeddings, required for RAG's similarity search.
*   **`BaseEmbeddingModel` Interface:**
    *   `__init__(self, config: dict)`: Initialize with model-specific config (e.g., model name, API key).
    *   `async def generate_embedding(self, text: str) -> List[float]`: Generates embedding for a single text string.
    *   `async def generate_embeddings(self, texts: List[str]) -> List[List[float]]`: Generates embeddings for a batch of texts efficiently.
*   **Implementations:**
    *   `SentenceTransformerEmbedding`: Uses the `sentence-transformers` library for local models (specified by name/path in `confy`).
    *   `OpenAIEmbedding`: Uses the `openai` SDK to call the embeddings API.
    *   `GoogleAIEmbedding`: Uses the `google-generativeai` SDK for Gemini embeddings.
*   **Initialization:** The `EmbeddingManager` (internal to `LLMCore`) reads `confy` (`llmcore.default_embedding_model` and `embedding.<provider>` sections) and instantiates the appropriate embedding model class.

### 4.7. Models (`llmcore.models`)

*   **Purpose:** Define clear, typed data structures for internal use and API consistency. Likely implemented using Pydantic or standard dataclasses.
*   **Core Models:**
    *   `Role(str, Enum)`: `SYSTEM`, `USER`, `ASSISTANT`.
    *   `Message`: `id: str`, `session_id: str`, `role: Role`, `content: str`, `timestamp: datetime`, `tokens: Optional[int] = None`, `metadata: Optional[Dict] = None`.
    *   `ChatSession`: `id: str`, `name: Optional[str] = None`, `messages: List[Message]`, `created_at: datetime`, `updated_at: datetime`, `metadata: Optional[Dict] = None`.
    *   `ContextDocument`: `id: str`, `content: str`, `embedding: Optional[List[float]] = None`, `metadata: Optional[Dict] = None`, `score: Optional[float] = None` (from similarity search).
    *   Configuration Models (Pydantic): Potentially used to validate loaded `confy` structures for providers, storage, etc.

### 4.8. Exceptions (`llmcore.exceptions`)

* **Purpose:** Provide specific error types for better error handling by library users.

* **Hierarchy:**
* ```
    Exception
    └── LLMCoreError
        ├── ConfigError
        ├── ProviderError
        ├── StorageError
        │   ├── SessionStorageError
        │   └── VectorStorageError
        ├── ContextError
        │   └── ContextLengthError
        ├── EmbeddingError
        ├── SessionNotFoundError
        └── MCPError
  ```

## 5. API Specification (Python Library API)

This section defines the primary public interface of the `LLMCore` library.

```python
# llmcore/__init__.py or llmcore/api.py
import asyncio
from typing import List, Optional, Dict, Any, Union, AsyncGenerator

# Import models and exceptions for type hinting and user access
from .models import ChatSession, ContextDocument
from .exceptions import (
    LLMCoreError, ProviderError, SessionNotFoundError, ConfigError,
    VectorStoreError, EmbeddingError, ContextLengthError, MCPError
)
from confy.loader import Config as ConfyConfig # Expose confy config type if needed

class LLMCore:
    """
    Main class for interacting with Large Language Models.

    Provides methods for chat completions, session management, and
    Retrieval Augmented Generation (RAG) using configurable providers,
    storage backends, and embedding models.
    """

    config: ConfyConfig # Publicly accessible resolved config (read-only recommended)

    def __init__(
        self,
        config_overrides: Optional[Dict] = None,
        config_file_path: Optional[str] = None,
        env_prefix: Optional[str] = "LLMCORE"
    ):
        """
        Initializes LLMCore with configuration.

        Loads configuration using confy, sets up providers, storage,
        context manager, and embedding manager based on the loaded config.

        Args:
            config_overrides: Dictionary of configuration overrides (highest precedence).
                                Keys use dot-notation (e.g., "providers.openai.default_model").
            config_file_path: Path to a custom TOML or JSON configuration file.
            env_prefix: Prefix for environment variable overrides (e.g., "LLMCORE").
                        Set to "" to consider all non-system env vars, None to disable env var loading.

        Raises:
            ConfigError: If essential configuration is missing or invalid.
            ImportError: If required dependencies (e.g., specific provider SDKs,
                         storage clients) are not installed based on config.
        """
        # 1. Initialize self.config using confy.Config(...)
        # 2. Initialize ProviderManager, StorageManager (Session & Vector),
        #    EmbeddingManager, ContextManager, SessionManager, passing self.config.
        # 3. Handle potential initialization errors.
        pass

    # --- Core Chat Method ---

    async def chat(
        self,
        message: str,
        *, # Force subsequent arguments to be keyword-only
        session_id: Optional[str] = None,
        system_message: Optional[str] = None,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        stream: bool = False,
        save_session: bool = True,
        # RAG parameters
        enable_rag: bool = False,
        rag_retrieval_k: Optional[int] = None,
        rag_collection_name: Optional[str] = None,
        # Provider specific arguments (e.g., temperature, max_tokens for response)
        **provider_kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Sends a message to the configured LLM, managing context and session.

        Handles conversation history, optional Retrieval Augmented Generation (RAG)
        from a vector store, provider-specific token counting, context window
        management (including truncation), and optional MCP formatting.

        Args:
            message: The user's message content.
            session_id: ID of the session to use/continue. If None, a temporary
                        (non-persistent) session is used for this call only.
            system_message: An optional system message for the LLM. Behavior when
                            provided for an existing session depends on context strategy
                            (e.g., it might replace or supplement existing system messages).
            provider_name: Override the default provider specified in the configuration.
            model_name: Override the default model for the selected provider.
            stream: If True, returns an async generator yielding response text chunks (str).
                    If False (default), returns the complete response content as a string.
            save_session: If True (default) and session_id is provided, the
                          conversation turn (user message + assistant response) is
                          saved to the persistent session storage. Ignored if
                          session_id is None.
            enable_rag: If True, enables Retrieval Augmented Generation by searching
                        the vector store for context relevant to the message.
            rag_retrieval_k: Number of documents to retrieve for RAG. Overrides the
                             default from configuration if provided.
            rag_collection_name: Name of the vector store collection to use for RAG.
                                 Overrides the default from configuration if provided.
            **provider_kwargs: Additional keyword arguments passed directly to the
                               selected provider's chat completion API call
                               (e.g., temperature=0.7, max_tokens=100). Note: `max_tokens`
                               here usually refers to the *response* length limit, not
                               the context window limit.

        Returns:
            If stream=False: The full response content as a string.
            If stream=True: An asynchronous generator yielding response text chunks (str).

        Raises:
            ProviderError: If the LLM provider API call fails (e.g., authentication, rate limit).
            SessionNotFoundError: If a specified session_id is not found in storage.
            ConfigError: If configuration for the selected provider/model is invalid.
            VectorStoreError: If RAG retrieval from the vector store fails.
            EmbeddingError: If embedding generation fails for the RAG query.
            ContextLengthError: If the essential context (e.g., system prompt + user message)
                                exceeds the model's limit even after internal truncation attempts.
            MCPError: If MCP formatting fails (if enabled).
            LLMCoreError: For other library-specific errors.
        """
        # Implementation details outlined in previous sections
        # Orchestrates SessionManager, ContextManager, ProviderManager
        pass

    # --- Session Management Methods ---

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Retrieves a specific chat session object (including messages).

        Args:
            session_id: The ID of the session to retrieve.

        Returns:
            The ChatSession object if found, otherwise None.

        Raises:
            StorageError: If interaction with the session storage fails.
        """
        # Calls self.session_storage.get_session(...)
        pass

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        Lists available persistent chat sessions (metadata only).

        Returns:
            A list of dictionaries, each containing basic info like 'id',
            'name', 'created_at', 'updated_at'. Does not include messages.

        Raises:
            StorageError: If interaction with the session storage fails.
        """
        # Calls self.session_storage.list_sessions(...)
        pass

    def delete_session(self, session_id: str) -> bool:
        """
        Deletes a persistent chat session from storage.

        Args:
            session_id: The ID of the session to delete.

        Returns:
            True if the session was deleted, False if it was not found.

        Raises:
            StorageError: If interaction with the session storage fails.
        """
        # Calls self.session_storage.delete_session(...)
        pass

    # --- RAG / Vector Store Management Methods ---

    async def add_document_to_vector_store(
        self,
        content: str,
        *,
        metadata: Optional[Dict] = None,
        doc_id: Optional[str] = None,
        collection_name: Optional[str] = None
    ) -> str:
        """
        Adds a single document (text content) to the configured vector store.

        Generates embedding using the configured embedding model and stores
        the document content, embedding, and metadata.

        Args:
            content: The text content of the document.
            metadata: Optional dictionary of metadata associated with the document.
            doc_id: Optional specific ID to assign to the document. If None,
                    an ID might be generated by the storage backend or library.
            collection_name: The target vector store collection. Uses the default
                             collection from configuration if None.

        Returns:
            The ID assigned to the added document.

        Raises:
            EmbeddingError: If embedding generation fails.
            VectorStoreError: If adding the document to the store fails.
            ConfigError: If vector store or embedding model is not configured.
        """
        # 1. Generate embedding via EmbeddingManager
        # 2. Create ContextDocument
        # 3. Call self.vector_storage.add_documents([doc], collection_name)
        pass

    async def add_documents_to_vector_store(
        self,
        documents: List[Dict[str, Any]],
        *,
        collection_name: Optional[str] = None
    ) -> List[str]:
        """
        Adds multiple documents to the configured vector store in a batch.

        Args:
            documents: A list of dictionaries, each representing a document.
                       Expected format: {"content": str, "metadata": Optional[Dict], "id": Optional[str]}
            collection_name: The target vector store collection. Uses the default
                             collection from configuration if None.

        Returns:
            A list of IDs assigned to the added documents.

        Raises:
            EmbeddingError: If embedding generation fails for any document.
            VectorStoreError: If adding documents to the store fails.
            ConfigError: If vector store or embedding model is not configured.
        """
        # 1. Extract content from documents list
        # 2. Generate embeddings in batch via EmbeddingManager
        # 3. Create list of ContextDocument objects
        # 4. Call self.vector_storage.add_documents(docs, collection_name)
        pass

    async def search_vector_store(
        self,
        query: str,
        *,
        k: int,
        collection_name: Optional[str] = None,
        filter_metadata: Optional[Dict] = None
    ) -> List[ContextDocument]:
        """
        Performs a similarity search for relevant documents in the vector store.

        Generates an embedding for the query text and searches the specified
        collection.

        Args:
            query: The text query to search for.
            k: The number of top similar documents to retrieve.
            collection_name: The target vector store collection. Uses the default
                             collection from configuration if None.
            filter_metadata: Optional dictionary to filter results based on metadata
                             (support depends on the vector store implementation).

        Returns:
            A list of ContextDocument objects representing the search results,
            potentially including similarity scores.

        Raises:
            EmbeddingError: If embedding generation for the query fails.
            VectorStoreError: If the search operation fails.
            ConfigError: If vector store or embedding model is not configured.
        """
        # 1. Generate query embedding via EmbeddingManager
        # 2. Call self.vector_storage.similarity_search(embedding, k, collection_name, filter_metadata)
        pass

    async def delete_documents_from_vector_store(
        self,
        document_ids: List[str],
        *,
        collection_name: Optional[str] = None
    ) -> bool:
        """
        Deletes documents from the vector store by their IDs.

        Args:
            document_ids: A list of document IDs to delete.
            collection_name: The target vector store collection. Uses the default
                             collection from configuration if None.

        Returns:
            True if deletion was successful (or partially successful), False otherwise.
            Exact behavior might depend on the storage backend.

        Raises:
            VectorStoreError: If the deletion operation fails.
            ConfigError: If vector store is not configured.
        """
        # Calls self.vector_storage.delete_documents(document_ids, collection_name)
        pass


    # --- Provider Info Methods ---

    def get_available_providers(self) -> List[str]:
        """
        Lists the names of all configured LLM providers.

        Returns:
            A list of provider name strings (e.g., ["openai", "ollama"]).
        """
        # Uses self.provider_manager
        pass

    def get_models_for_provider(self, provider_name: str) -> List[str]:
        """
        Lists available models for a specific configured provider.

        Note: This might return a cached list or perform an API call depending
              on the provider implementation.

        Args:
            provider_name: The name of the provider.

        Returns:
            A list of model name strings available for the provider.

        Raises:
            ConfigError: If the specified provider is not configured.
            ProviderError: If fetching models from the provider API fails.
        """
        # Uses self.provider_manager.get_provider(provider_name).get_available_models()
        pass

    # --- Utility / Cleanup ---
    async def close(self):
        """
        Closes connections for storage backends and potentially other resources.
        Should be called when the application using LLMCore is shutting down.
        """
        # Calls close() on storage managers, potentially provider clients if needed.
        pass

    def __await__(self):
        # Allows LLMCore instances to be awaited if initialization needs to be async
        # (e.g., async storage connection). Not strictly needed if __init__ is sync.
        async def closure():
            # If __init__ becomes async, perform await operations here.
            # For now, it just returns self.
            return self
        return closure().__await__()
```

## 6\. Use Cases & Examples

This section illustrates how developers might use the `LLMCore` library.

**Example 1: Simple Chat (No Session Persistence)**

```python
import asyncio
from llmcore import LLMCore

async def main():
    # Initialize with default config (loads from ~/.config/llmcore/config.toml, .env, env vars)
    llm = LLMCore()

    try:
        # Send a message, get the full response back
        response = await llm.chat("What is the capital of France?")
        print(f"LLM Response: {response}")

        # Example using a specific provider and model
        response_ollama = await llm.chat(
            "Explain the concept of recursion in programming.",
            provider_name="ollama",
            model_name="llama3",
            temperature=0.5 # Pass provider-specific args
        )
        print(f"\nOllama Response:\n{response_ollama}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await llm.close() # Close storage connections etc.

if __name__ == "__main__":
    asyncio.run(main())
```

**Example 2: Chat with Session History**

```python
import asyncio
from llmcore import LLMCore
import uuid

async def main():
    # Use a specific config file and prefix
    llm = LLMCore(
        config_file_path="my_app_config.toml",
        env_prefix="MYAPP"
    )
    session_id = f"user_conversation_{uuid.uuid4()}" # Generate a unique session ID

    try:
        print(f"Starting session: {session_id}")

        # First message
        response1 = await llm.chat(
            "My name is Alex. What is quantum computing?",
            session_id=session_id,
            system_message="You are a helpful physics tutor.",
            save_session=True # Ensure history is saved
        )
        print(f"Tutor: {response1}")

        # Second message - LLM should have context from the first message
        response2 = await llm.chat(
            "How does it relate to my name?", # Silly follow-up
            session_id=session_id,
            save_session=True
        )
        print(f"Tutor: {response2}")

        # List sessions to see ours
        sessions = llm.list_sessions()
        print("\nAvailable Sessions:", sessions)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await llm.close()

if __name__ == "__main__":
    asyncio.run(main())
```

**Example 3: Streaming Chat Response**

```python
import asyncio
from llmcore import LLMCore

async def main():
    llm = LLMCore()
    session_id = "streaming_test_session"

    try:
        print("User: Tell me a short story about a brave knight.")
        print("Assistant (streaming): ", end="", flush=True)

        # Use stream=True to get an async generator
        response_stream = await llm.chat(
            "Tell me a short story about a brave knight.",
            session_id=session_id,
            stream=True,
            save_session=True # Save the full response once streaming completes
        )

        async for chunk in response_stream:
            print(chunk, end="", flush=True)

        print("\n--- Stream finished ---")

        # Verify session was saved
        session = llm.get_session(session_id)
        if session:
            print(f"\nSession '{session_id}' saved with {len(session.messages)} messages.")
            # Optionally print last message content
            # print(f"Last message: {session.messages[-1].content}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        await llm.close()

if __name__ == "__main__":
    asyncio.run(main())
```

**Example 4: RAG - Adding Documents and Querying**

```python
import asyncio
from llmcore import LLMCore

async def main():
    # Assume config points to ChromaDB/pgvector and an embedding model
    llm = LLMCore()
    collection = "my_documents"

    try:
        # --- Add documents to the vector store ---
        docs_to_add = [
            {"content": "The Eiffel Tower is located in Paris, France.", "metadata": {"source": "wiki/eiffel"}, "id": "doc1"},
            {"content": "Paris is known for its museums, like the Louvre.", "metadata": {"source": "wiki/paris"}, "id": "doc2"},
            {"content": "The primary programming language used in LLMCore is Python.", "metadata": {"source": "llmcore/readme"}, "id": "doc3"}
        ]
        print(f"Adding {len(docs_to_add)} documents to collection '{collection}'...")
        doc_ids = await llm.add_documents_to_vector_store(docs_to_add, collection_name=collection)
        print(f"Added document IDs: {doc_ids}")

        # --- Perform a similarity search (optional direct check) ---
        query = "What language is used for LLMCore?"
        search_results = await llm.search_vector_store(query, k=1, collection_name=collection)
        print(f"\nDirect search results for '{query}':")
        for doc in search_results:
            print(f"- ID: {doc.id}, Score: {doc.score:.4f}, Content: {doc.content[:50]}...")

        # --- Use RAG in a chat ---
        print("\n--- Chatting with RAG enabled ---")
        rag_response = await llm.chat(
            "What programming language is mentioned in the documents?",
            enable_rag=True,
            rag_collection_name=collection,
            rag_retrieval_k=1, # Retrieve top 1 relevant doc
            system_message="Answer based *only* on the provided context documents."
        )
        print(f"RAG Response: {rag_response}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Optionally delete documents if just testing
        # await llm.delete_documents_from_vector_store(["doc1", "doc2", "doc3"], collection_name=collection)
        await llm.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## 7. Converting `llmchat` CLI to use `LLMCore`

*   **Initialization:** `llmchat`'s main entry point (`llmchat.main.main`) will instantiate `LLMCore`, passing relevant CLI options (`--config`, `--provider`, `--model`, etc.) as `config_overrides` or using them to select the provider/model in the `chat` call.
*   **`chat` Command:** Will call `llmcore_instance.chat()`, passing message, session ID, system message, stream flag, and provider kwargs. It will handle printing the streamed or full response.
*   **`session` Command:** Will use `llmcore_instance.list_sessions()` and `llmcore_instance.get_session()`.
*   **`delete` Command:** Will use `llmcore_instance.delete_session()`.
*   **`models` Command:** Will use `llmcore_instance.get_available_providers()` and `llmcore_instance.get_models_for_provider()`.
*   **Configuration:** `llmchat`'s custom `ConfigManager` will be removed. Configuration loading is handled entirely by `LLMCore` via `confy`. `llmchat` might still define its _own_ default config file path that it passes to `LLMCore`.
*   **Dependencies:** `llmchat`'s direct dependencies on provider SDKs (`openai`, `anthropic`) and storage (`tomli`, `tomli-w`, `sqlite3`) will be replaced by a single dependency on the `LLMCore` library package.

## 8 Technology Stack Summary

*   **Core Library:** Python 3.8+ (Asyncio heavily used)
*   **Configuration:** `confy`
*   **LLM Interactions:** `aiohttp`, `anthropic` SDK, `openai` SDK, `google-generativeai`.
*   **Vector Stores:** `chromadb-client`, `psycopg2-binary` (for PostgreSQL), `pgvector` (Python client or direct SQL).
*   **Embedding Models:** `sentence-transformers`. (Requires PyTorch/TensorFlow/ONNX Runtime depending on model).
*   **MCP Support:** `modelcontextprotocol` (Python SDK).
*   **Type Hinting:** Standard Python type hints. Pydantic (optional, for models/config validation).
*   **Testing:** `pytest`, `pytest-asyncio`.

## 9 Phased Development Plan

This plan breaks down the development of `LLMCore` into logical phases, focusing on delivering core functionality first and building upon it. Each task includes its purpose, key details, and expected deliverables.

### **Phase 1: Core Structure, Configuration, Session Management & Basic Providers**

*   **Task 1.1: Project Setup & Core Dependencies**

    *   **Purpose:** Establish the library's structure, build system, core dependencies, and basic package metadata.
    *   **Details:**
        *   Initialize Python project structure (`src/llmcore`, `tests/`, `docs/`).
        *   Setup `pyproject.toml` with `setuptools` build backend.
        *   Define project metadata (name `llmcore`, version, description, author, license, Python requirement >=3.8).
        *   Add core dependencies: `confy`, `aiohttp`, `python-dotenv`, `typing_extensions`.
        *   Set up basic testing infrastructure with `pytest` and `pytest-asyncio`.
        *   Create initial module structure (`__init__.py`, `api.py`, `models.py`, `exceptions.py`, `providers/`, `storage/`, `context/`, `embedding/`, `config/`).
        *   Initialize Git repository.
    *   **Deliverables:** A minimal, installable Python package structure committed to Git. Basic `pyproject.toml`. Test setup running.
*   **Task 1.2: Configuration (`confy`) Integration**

    *   **Purpose:** Integrate `confy` for loading library configuration and define the default configuration structure.
    *   **Details:**
        *   Create a default configuration file (`config/default_config.toml`) within the package, defining sections and keys as outlined in Section 4.1 (providers, storage, embedding, context\_management). Include sensible defaults.
        *   Implement logic within `llmcore.LLMCore.__init__` to initialize `confy.Config`, loading the packaged default config and allowing overrides via `config_file_path`, `config_overrides`, and `env_prefix`.
        *   Make the resolved `confy.Config` object available (e.g., as `self.config`).
        *   Define basic `ConfigError` in `llmcore.exceptions`.
    *   **Deliverables:** `LLMCore` class can initialize `confy`. Default configuration defined. Configuration loading mechanism functional. Basic tests for config loading.
*   **Task 1.3: Core Models & Exceptions**

    *   **Purpose:** Define the fundamental data structures and custom exception hierarchy.
    *   **Details:**
        *   Define core data models in `llmcore.models` using Pydantic or dataclasses: `Role` (Enum), `Message`, `ChatSession`, `ContextDocument`. Ensure proper type hinting.
        *   Define the base `LLMCoreError` and initial specific exceptions (`ConfigError`, `SessionNotFoundError`, `ProviderError`, `StorageError`) in `llmcore.exceptions`.
    *   **Deliverables:** Defined and tested core data models and exception classes.
*   **Task 1.4: Session Storage Interface & Basic Implementations**

    *   **Purpose:** Define the storage abstraction for chat sessions and implement basic file-based backends.
    *   **Details:**
        *   Define `storage.BaseSessionStorage` abstract base class with methods: `initialize`, `save_session`, `get_session`, `list_sessions`, `delete_session`, `close`. Ensure methods are async where appropriate (e.g., if DB interaction is involved later).
        *   Implement `storage.JsonSessionStorage` storing each session in a separate JSON file in a configured directory (`storage.session.path`).
        *   Implement `storage.SqliteSessionStorage` using Python's `sqlite3` module. Define table schema for sessions and messages. Handle connection management and basic CRUD operations. Configure via `storage.session.path`.
        *   Implement `StorageManager` (internal) to load the configured session storage backend based on `config.storage.session.type`.
        *   Add `SessionStorageError` to exceptions.
    *   **Deliverables:** Session storage interface defined. Functional JSON and SQLite session storage backends. Storage manager capable of loading the configured backend. Unit tests for both storage types.
*   **Task 1.5: Session Manager**

    *   **Purpose:** Implement the logic for managing `ChatSession` objects in memory and coordinating with storage.
    *   **Details:**
        *   Create `SessionManager` class (internal).
        *   Implement methods like `load_or_create_session(session_id)`, `add_message_to_session(session, message)`, which interact with the configured `BaseSessionStorage` backend via `StorageManager`.
        *   Handle creation of temporary (in-memory only) sessions when `session_id` is None.
    *   **Deliverables:** `SessionManager` capable of managing session objects and interacting with storage backends.
*   **Task 1.6: Provider Interface & Basic Implementations (Ollama)**

    *   **Purpose:** Define the provider abstraction and implement the Ollama provider as an initial test case (often easier for local testing).
    *   **Details:**
        *   Define `providers.BaseProvider` abstract base class with methods: `chat_completion`, `count_tokens`, `count_message_tokens`, `get_available_models`, `get_max_context_length`, `get_name`. Ensure `chat_completion` handles the `stream` flag correctly (returning `Dict` or `AsyncGenerator[Dict, None]`).
        *   Implement `providers.OllamaProvider`. Use `aiohttp` to interact with the Ollama REST API (`/api/chat`, potentially fallback to `/api/generate`). Implement token counting using a configurable default (e.g., `tiktoken_cl100k_base`) and document its approximate nature. Handle streaming responses correctly. Read configuration (`base_url`, `default_model`, `timeout`, `tokenizer`) from `confy`.
        *   Implement `ProviderManager` (internal) to load the configured providers based on `config.providers`.
        *   Add `ProviderError` to exceptions.
    *   **Deliverables:** Provider interface defined. Functional Ollama provider implementation. Provider manager loads configured providers. Unit/integration tests for Ollama provider (may require a running Ollama instance or mocking).
*   **Task 1.7: Basic Context Manager (Token Counting & Simple History)**

    *   **Purpose:** Implement initial context assembly focusing on token limits and basic history management.
    *   **Details:**
        *   Create `ContextManager` class (internal).
        *   Implement `prepare_context` method (internal) that takes session/messages, target provider/model.
        *   Use the target provider's `count_message_tokens` method via `ProviderManager` to calculate context size.
        *   Implement basic history selection (`last_n_tokens` strategy) to fit within `provider.get_max_context_length` minus `reserved_response_tokens`. Prioritize system messages.
        *   Initially, format context as `List[Message]`.
        *   Implement basic truncation logic (remove oldest non-system message) if over limit after initial selection.
        *   Add `ContextError` and `ContextLengthError` to exceptions.
    *   **Deliverables:** `ContextManager` capable of assembling a `List[Message]` context, respecting token limits using provider-specific counting and basic history truncation.
*   **Task 1.8: Core API Facade (`LLMCore.chat` - Basic)**

    *   **Purpose:** Implement the main `chat` method integrating the components built so far (Config, Session, Context, Provider).
    *   **Details:**
        *   Implement the `LLMCore.chat` method signature as defined in the API spec.
        *   Orchestrate the flow: get/create session, add user message, prepare context (using `ContextManager`), call provider's `chat_completion`, process response (handle non-streaming case initially), add assistant message, save session.
        *   Implement the non-streaming return path (`-> str`).
        *   Handle errors raised by managers/providers and wrap them in appropriate `LLMCore` exceptions.
    *   **Deliverables:** A functional `LLMCore.chat` method for basic, non-streaming chat interactions using the configured default provider and storage. Integration tests covering the basic chat flow.

### **Phase 2: Streaming, Additional Providers & Basic RAG Setup**

*   **Task 2.1: Streaming Implementation (`LLMCore.chat`)**

    *   **Purpose:** Enable asynchronous streaming of LLM responses through the `chat` method.
    *   **Details:**
        *   Modify `LLMCore.chat` to handle the `stream=True` case.
        *   When `stream=True`, it should receive the `AsyncGenerator[Dict, None]` from the provider's `chat_completion`.
        *   Implement the `stream_wrapper` async generator pattern described in Section 5 (API Spec): iterate through provider chunks, extract text delta, yield text delta, accumulate full response, save session after stream completion.
    *   **Deliverables:** `LLMCore.chat` method fully supports `stream=True`, returning an `AsyncGenerator[str, None]`. Tests for streaming functionality.
*   **Task 2.2: Implement OpenAI Provider**

    *   **Purpose:** Add support for OpenAI models.
    *   **Details:**
        *   Implement `providers.OpenAIProvider` inheriting from `BaseProvider`.
        *   Use the `openai` Python SDK (v1.0+).
        *   Implement `chat_completion` using `async client.chat.completions.create(...)`, handling streaming correctly.
        *   Implement token counting using `tiktoken` and model-specific encodings.
        *   Read configuration (`api_key`, `default_model`, `timeout`) from `confy`. Handle API key errors gracefully.
    *   **Deliverables:** Functional OpenAI provider. Unit/integration tests (likely requiring mocking or API keys).
*   **Task 2.3: Implement Anthropic Provider**

    *   **Purpose:** Add support for Anthropic Claude models.
    *   **Details:**
        *   Implement `providers.AnthropicProvider` inheriting from `BaseProvider`.
        *   Use the `anthropic` Python SDK.
        *   Implement `chat_completion` using `async client.messages.create(...)`, handling streaming. Map roles correctly (e.g., `assistant`).
        *   Implement token counting using `client.count_tokens()`.
        *   Read configuration (`api_key`, `default_model`, `timeout`) from `confy`.
    *   **Deliverables:** Functional Anthropic provider. Unit/integration tests.
*   **Task 2.4: Implement Gemini Provider**

    *   **Purpose:** Add support for Google Gemini models.
    *   **Details:**
        *   Implement `providers.GeminiProvider` inheriting from `BaseProvider`.
        *   Use the `google-generativeai` Python SDK.
        *   Implement `chat_completion` using `model.generate_content(..., stream=True/False)`. Map message history/roles to the Gemini API format. Handle potential safety setting blocks.
        *   Implement token counting using `model.count_tokens()`.
        *   Read configuration (`api_key`, `default_model`, `safety_settings`) from `confy`.
    *   **Deliverables:** Functional Gemini provider. Unit/integration tests.
*   **Task 2.5: Vector Storage Interface & ChromaDB Implementation**

    *   **Purpose:** Define the storage abstraction for vector embeddings and implement the ChromaDB backend.
    *   **Details:**
        *   Define `storage.BaseVectorStorage` abstract base class with methods: `initialize`, `add_documents`, `similarity_search`, `delete_documents`, `close`. Ensure methods are async.
        *   Implement `storage.ChromaVectorStorage` using the `chromadb-client` library.
        *   Handle client initialization (persistent client using configured `path` or in-memory).
        *   Implement interface methods using `collection.add`, `collection.query`, `collection.delete`. Handle collection creation (`get_or_create_collection`).
        *   Read configuration (`path`, `default_collection`) from `confy`.
        *   Update `StorageManager` to load the configured vector storage backend.
        *   Add `VectorStorageError` to exceptions.
    *   **Deliverables:** Vector storage interface defined. Functional ChromaDB vector storage backend. Storage manager loads vector backend. Unit tests for ChromaDB storage.
*   **Task 2.6: Embedding Interface & Basic Implementation (Sentence Transformers)**

    *   **Purpose:** Define the abstraction for embedding generation and implement a local model backend.
    *   **Details:**
        *   Define `embedding.BaseEmbeddingModel` abstract base class with methods: `initialize`, `generate_embedding`, `generate_embeddings`. Ensure methods are async if underlying libraries require it (though Sentence Transformers is typically sync, wrap if needed for consistency).
        *   Implement `embedding.SentenceTransformerEmbedding` using the `sentence-transformers` library.
        *   Load the model specified by `llmcore.default_embedding_model` or a specific config section during initialization. Handle model download/caching.
        *   Implement interface methods using `model.encode()`.
        *   Implement `EmbeddingManager` (internal) to load the configured embedding model.
        *   Add `EmbeddingError` to exceptions.
    *   **Deliverables:** Embedding interface defined. Functional Sentence Transformer embedding backend. Embedding manager loads the model. Unit tests for embedding generation.
*   **Task 2.7: Basic RAG Integration (`ContextManager`)**

    *   **Purpose:** Integrate the basic RAG flow into the context building process.
    *   **Details:**
        *   Modify `ContextManager.prepare_context` to perform the RAG steps if `rag_config['enabled']` is true:
            *   Call `EmbeddingManager` to embed the current message.
            *   Call `StorageManager.vector_storage.similarity_search` using the embedding and `rag_retrieval_k`/`rag_collection_name`.
            *   Retrieve `ContextDocument`s.
            *   Extract text content from documents.
            *   Combine retrieved text with history/prompt using the configured `rag_combination_strategy` (implement basic `prepend_system`).
            *   Ensure the combined context respects token limits, applying truncation if necessary (using configured `truncation_priority`).
    *   **Deliverables:** `ContextManager` can retrieve documents from vector store and incorporate their text content into the LLM context before truncation.
*   **Task 2.8: Core API Facade (`LLMCore` RAG Methods)**

    *   **Purpose:** Implement the public API methods for managing the vector store.
    *   **Details:**
        *   Implement `LLMCore.add_document_to_vector_store`. Orchestrates `EmbeddingManager` and `StorageManager.vector_storage`.
        *   Implement `LLMCore.add_documents_to_vector_store` (batch version).
        *   Implement `LLMCore.search_vector_store`. Orchestrates `EmbeddingManager` and `StorageManager.vector_storage`.
        *   Implement `LLMCore.delete_documents_from_vector_store`. Calls `StorageManager.vector_storage`.
    *   **Deliverables:** Functional public API methods for adding, searching, and deleting documents in the configured vector store. Integration tests for RAG API methods.

### **Phase 3: Advanced Storage, Context Strategies & MCP**

*   **Task 3.1: Implement PostgreSQL Storage (Session & Vector)**

    *   **Purpose:** Add support for using PostgreSQL for both session storage and vector storage (via pgvector).
    *   **Details:**
        *   Add `psycopg2-binary` and potentially a `pgvector` client library as optional dependencies.
        *   Implement `storage.PostgresSessionStorage` inheriting from `BaseSessionStorage`. Use `psycopg2` to connect (using `storage.session.db_url` from config) and perform CRUD operations on session/message tables. Define appropriate table schemas.
        *   Implement `storage.PgVectorStorage` inheriting from `BaseVectorStorage`. Use `psycopg2` and `pgvector` SQL syntax/functions. Handle connection using `storage.vector.db_url`. Implement `add_documents` (inserting embeddings), `similarity_search` (using vector distance operators like `<=>`), `delete_documents`. Define vector table schema, including index creation (e.g., HNSW).
        *   Update `StorageManager` to recognize and load `postgres` / `pgvector` types based on config.
    *   **Deliverables:** Functional PostgreSQL backend for both session and vector storage. Unit/integration tests (may require a running PostgreSQL instance with pgvector enabled).
*   **Task 3.2: Implement Additional Embedding Providers (OpenAI, Google)**

    *   **Purpose:** Provide options for using cloud-based embedding services.
    *   **Details:**
        *   Implement `embedding.OpenAIEmbedding` using the `openai` SDK (`client.embeddings.create`). Read config from `[embedding.openai]`.
        *   Implement `embedding.GoogleAIEmbedding` using the `google-generativeai` SDK (`embed_content`). Read config from `[embedding.google]`.
        *   Update `EmbeddingManager` to load these based on `llmcore.default_embedding_model` or specific identifiers.
    *   **Deliverables:** Support for OpenAI and Google AI embedding models.
*   **Task 3.3: Advanced Context/RAG Strategies**

    *   **Purpose:** Implement more sophisticated strategies for combining context and handling truncation.
    *   **Details:**
        *   In `ContextManager`, implement alternative `rag_combination_strategy` options (e.g., `prepend_user`).
        *   In `ContextManager`, implement alternative `history_selection_strategy` options (e.g., `last_n_messages`).
        *   Refine `truncation_priority` logic (e.g., allow finer control over how much history vs. RAG to keep).
        *   Ensure these strategies are selectable via `confy` settings in `[context_management]`.
    *   **Deliverables:** More flexible and configurable context assembly and truncation logic within `ContextManager`.
*   **Task 3.4: Model Context Protocol (MCP) Integration**

    *   **Purpose:** Add optional support for formatting context using the MCP standard.
    *   **Details:**
        *   Add `modelcontextprotocol` SDK as an optional dependency.
        *   In `ContextManager.prepare_context`:
            *   Check if MCP is enabled globally (`llmcore.enable_mcp`) or for the specific provider (`providers.<name>.use_mcp`).
            *   If enabled and the SDK is available:
                *   Instantiate MCP objects (`mcp.Context`, `mcp.Message`, `mcp.RetrievedKnowledge`, etc.).
                *   Map the selected conversation history, system prompt, and retrieved RAG document content to the corresponding MCP elements.
                *   Return the constructed `mcp.Context` object as the payload.
            *   If disabled, return the standard `List[Message]` payload.
        *   Modify `BaseProvider.chat_completion` signature slightly or add internal logic to detect if the received `context` is an MCP object or a `List[Message]` and handle accordingly. Providers need to know how to serialize/send the MCP object if they support it.
        *   Add `MCPError` exception.
    *   **Deliverables:** Optional MCP context formatting capability driven by configuration, using the official SDK. Provider implementations updated to potentially handle MCP context objects.
*   **Task 3.5: Refine API and Models**

    *   **Purpose:** Review and refine the public API signatures, docstrings, and data models based on implementation experience.
    *   **Details:**
        *   Ensure consistent use of keyword-only arguments for clarity in `LLMCore.chat`.
        *   Improve docstrings for all public methods and classes.
        *   Finalize Pydantic/dataclass definitions for models.
        *   Review exception hierarchy and usage.
    *   **Deliverables:** Polished and well-documented public API and models.

### **Phase 4: Testing, Documentation & Initial Release**

*   **Task 4.1: Comprehensive Testing**

    *   **Purpose:** Ensure robustness, correctness, and reliability across different configurations and use cases.
    *   **Details:**
        *   Write extensive unit tests for each module (providers, storage, context, embedding, session managers) using `pytest` and `pytest-asyncio`. Mock external services (LLM APIs, DBs) where appropriate.
        *   Write integration tests covering common workflows:
            *   Simple chat (streaming and non-streaming).
            *   Chat with session persistence (JSON, SQLite, Postgres).
            *   RAG workflow (add docs, chat with RAG enabled) using different vector stores (ChromaDB, pgvector).
            *   Test various configurations loaded via `confy` (different providers, storage types, context strategies).
            *   Test error handling paths (e.g., `ContextLengthError`, `ProviderError`).
            *   Test MCP formatting path (if enabled).
        *   Aim for high test coverage.
    *   **Deliverables:** A comprehensive test suite ensuring core functionality works as expected under various conditions.
*   **Task 4.2: Documentation**

    *   **Purpose:** Create clear documentation for users of the `LLMCore` library.
    *   **Details:**
        *   Set up documentation generation using Sphinx or MkDocs.
        *   Write User Guide:
            *   Installation instructions (including optional dependencies for different storage/providers).
            *   Quickstart example.
            *   Core Concepts (Providers, Sessions, Storage, RAG, Context, MCP).
            *   Configuration Guide (explaining `confy` integration and all settings).
            *   API Reference (detailed explanation of `LLMCore` class and public methods/models/exceptions).
            *   Examples for common use cases (simple chat, sessions, streaming, RAG).
            *   Extensibility Guide (how to add custom providers or storage backends).
        *   Ensure all public classes and methods have comprehensive docstrings (used by the API reference).
    *   **Deliverables:** Hosted or packaged documentation covering installation, usage, configuration, API reference, and examples.
*   **Task 4.3: Refactor `llmchat` CLI**

    *   **Purpose:** Update the `llmchat` tool to use `LLMCore` as its backend engine.
    *   **Details:**
        *   Remove internal provider, storage, context logic from `llmchat`.
        *   Add `llmcore` as a dependency to `llmchat`.
        *   Modify `llmchat`'s main script to instantiate `LLMCore`.
        *   Update Click command implementations (`chat`, `session`, `delete`, `models`) to call the corresponding `LLMCore` methods.
        *   Adapt `llmchat`'s CLI argument parsing to pass necessary overrides or parameters to `LLMCore`.
        *   Ensure `llmchat` handles exceptions raised by `LLMCore` gracefully.
    *   **Deliverables:** A refactored `llmchat` CLI tool that uses `LLMCore` for its core functionality.
*   **Task 4.4: Packaging & Release Preparation**

    *   **Purpose:** Prepare the `LLMCore` library for distribution via PyPI.
    *   **Details:**
        *   Finalize `pyproject.toml` (metadata, dependencies, optional dependencies for different storage/providers, e.g., `llmcore[postgres,gemini,chromadb]`).
        *   Ensure `README.md` is up-to-date.
        *   Include `LICENSE` file.
        *   Configure package data if default configs need to be included.
        *   Use `build` to create source distribution and wheel.
        *   Test installation from the built wheel in clean environments.
        *   Prepare release notes.
    *   **Deliverables:** Build artifacts (`.tar.gz`, `.whl`) ready for release.
*   **Task 4.5: Initial Release (v1.0)**

    *   **Purpose:** Publish the first stable version of `LLMCore`.
    *   **Details:**
        *   Create a Git tag for the release version.
        *   Upload distribution files to PyPI using `twine`.
        *   Announce the release.
    *   **Deliverables:** `LLMCore` v1.0 available on PyPI.

### **Phase 5: Post-Release & Future Enhancements (Ongoing)**

*   **Task 5.1: Gather Feedback & Bug Fixes**

    *   **Purpose:** Address issues and incorporate feedback from early adopters.
    *   **Details:** Monitor issue trackers, respond to user reports, fix bugs, and release patch versions as needed.
    *   **Deliverables:** Improved stability and usability based on real-world usage.
*   **Task 5.2: Performance Optimization**

    *   **Purpose:** Identify and address performance bottlenecks.
    *   **Details:** Profile key areas (API calls, RAG retrieval, context building, embedding generation). Optimize database queries, batch operations, async handling.
    *   **Deliverables:** Improved latency and resource usage.
*   **Task 5.3: Add More Providers/Storage/Embedders**

    *   **Purpose:** Expand the ecosystem of supported integrations.
    *   **Details:** Implement support for other popular LLM providers (e.g., Cohere, Mistral AI), vector databases (e.g., Qdrant, Weaviate), or embedding models based on user demand and feasibility.
    *   **Deliverables:** Increased flexibility and choice for users.
*   **Task 5.4: Enhance Context Management Strategies**

    *   **Purpose:** Provide more sophisticated ways to manage the context window.
    *   **Details:** Explore and implement advanced techniques like conversational summarization (as an optional strategy), context caching, or more complex RAG ranking/filtering.
    *   **Deliverables:** More powerful context handling capabilities.
*   **Task 5.5: Tool Use / Function Calling Support**

    *   **Purpose:** Integrate support for LLMs that can use external tools or call functions.
    *   **Details:** Adapt the `BaseProvider` interface and `chat` API to handle tool definitions, tool calls from the LLM, and tool execution results, feeding them back into the conversation.
    *   **Deliverables:** Ability to leverage LLM function calling capabilities through `LLMCore`.
*   **Task 5.6: Full MCP Implementation & Validation**

    *   **Purpose:** Fully implement and validate MCP support as the protocol matures.
    *   **Details:** Stay updated with the MCP specification and SDK. Ensure `LLMCore` correctly generates and potentially consumes MCP context objects. Test interoperability with MCP-compliant models/servers.
    *   **Deliverables:** Robust and compliant MCP support.
*   **Task 5.7: Enhanced `llmchat` CLI Features**

    *   **Purpose:** Leverage new `LLMCore` features in the example CLI.
    *   **Details:** Add commands for RAG management (`llmchat rag add/search/delete`), flags for enabling RAG/MCP in `llmchat chat`, commands for managing storage, etc.
    *   **Deliverables:** A more feature-rich `llmchat` demonstrating `LLMCore`'s capabilities.
