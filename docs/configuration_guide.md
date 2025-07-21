# Configuration Guide for LLMCore v0.21.0





## Introduction



LLMCore has evolved from a unified library for Large Language Model (LLM) interaction into a comprehensive, multi-component platform for building sophisticated and scalable AI applications. Version 0.21.0 introduces an autonomous agent framework, a backend task queue, and a full-featured API server, all designed to work in concert.1 The power, flexibility, and scalability of this entire platform are unlocked and controlled through its robust and deeply integrated configuration system.

This guide serves as the canonical reference for every setting available in LLMCore v0.21.0. It provides an exhaustive, deeply detailed exploration of the configuration schema, enabling developers to precisely tailor the behavior of the core library, the FastAPI server, and the TaskMaster background workers for any use case, from local development and experimentation to high-performance production deployments. Understanding this system is fundamental to leveraging the full capabilities of the LLMCore platform.



## The LLMCore Configuration Philosophy: A Layered Approach



At the heart of LLMCore's configuration management is the `confy` library, a system designed for a flexible, layered approach to settings management.1 This design is not merely a convenience but a core architectural pillar that enables the platform's operational versatility. Settings are loaded from multiple sources in a specific order of precedence, allowing for a powerful override mechanism that is essential for managing different environments—such as local development, staging, and production—without altering the base configuration files.

The architecture of LLMCore v0.21.0 comprises three primary, interconnected components: the core `LLMCore` library, a deployable FastAPI-based `API Server`, and an asynchronous `TaskMaster` background worker system.1 These components are often deployed in different environments; for instance, a developer might run all components locally, while a production deployment might involve running the API server and TaskMaster workers in separate Docker containers or Kubernetes pods.

A monolithic, single-file configuration system would be inadequate for such a distributed architecture, particularly for managing secrets like API keys and environment-specific settings like database connection URLs. The layered system directly addresses this challenge. A developer can rely on a base `config.toml` file for local development but seamlessly override critical settings like database URLs and provider API keys using environment variables in a production container, all without changing a single line of code. This makes the configuration system the key enabler for the platform's operational flexibility, security, and production-readiness.



## Configuration Methods and Override Precedence



LLMCore resolves its final configuration by loading settings from six distinct sources. Each subsequent source in the hierarchy overrides any settings defined in the previous ones. This ensures that the most specific and immediate configuration (e.g., a direct in-code override) always takes precedence.1

The complete override hierarchy is detailed in the table below.

| Priority    | Source                                             | Description                                                  |
| ----------- | -------------------------------------------------- | ------------------------------------------------------------ |
| 1 (Lowest)  | Packaged Defaults (`default_config.toml`)          | The base template included within the `llmcore` library package. It defines every available setting and serves as the foundational layer.1 |
| 2           | User Config File (`~/.config/llmcore/config.toml`) | User-specific settings that override the packaged defaults. This file is ideal for personal preferences and API keys used across multiple projects.1 |
| 3           | Custom Config File (`config_file_path=...`)        | A project-specific configuration file path passed during `LLMCore.create()`. This allows for self-contained project configurations that override user and default settings.1 |
| 4           | `.env` File                                        | Environment variables loaded from a `.env` file located in the project's root directory. Requires the `python-dotenv` package.1 |
| 5           | Environment Variables (`LLMCORE_...`)              | System-level environment variables. This is the standard method for configuring applications in production, CI/CD, and containerized environments.1 |
| 6 (Highest) | Overrides Dictionary (`config_overrides=...`)      | A Python dictionary of settings passed directly to the `LLMCore.create()` method. This provides the ultimate level of control for dynamic, in-code configuration.1 |



### Environment Variable Naming Convention



For production deployments, using environment variables is the recommended practice. Any setting in the `toml` configuration file can be overridden by an environment variable. The variable name is constructed by following a consistent pattern:

1. Start with the prefix `LLMCORE_`.
2. Take the full TOML path to the setting (e.g., `providers.openai.api_key`).
3. Convert the path to uppercase.
4. Replace all dots (`.`) with double underscores (`__`).

This convention ensures a clear and unambiguous mapping from the configuration file structure to environment variables.1

| TOML Path                            | Environment Variable                          |
| ------------------------------------ | --------------------------------------------- |
| `llmcore.default_provider`           | `LLMCORE_DEFAULT_PROVIDER`                    |
| `providers.openai.api_key`           | `LLMCORE_PROVIDERS__OPENAI__API_KEY`          |
| `storage.session.type`               | `LLMCORE_STORAGE__SESSION__TYPE`              |
| `storage.vector.db_url`              | `LLMCORE_STORAGE__VECTOR__DB_URL`             |
| `embedding.ollama.default_model`     | `LLMCORE_EMBEDDING__OLLAMA__DEFAULT_MODEL`    |
| `context_management.rag_retrieval_k` | `LLMCORE_CONTEXT_MANAGEMENT__RAG_RETRIEVAL_K` |



## Global Settings: The `[llmcore]` Section



The `[llmcore]` section of the configuration file defines top-level settings that control the library's default behavior and global features. These settings provide a baseline for the entire platform's operation.1



### `default_provider`



- **What it does:** Specifies the default LLM provider to be used for any `chat()` call where the `provider_name` argument is not explicitly provided.
- **Purpose:** This setting simplifies API calls by allowing the developer to configure their primary or most-used provider once, rather than specifying it in every interaction.
- **Possible Options:** Any string that corresponds to a configured provider instance name under the `[providers]` section. For example, if you have `[providers.my_openai_clone]`, you can set `default_provider = "my_openai_clone"`.
- **Default Value:** `"ollama"`
- **How to Override:**
    - **Environment Variable:** `export LLMCORE_DEFAULT_PROVIDER="openai"`
    - **Code Override:** `LLMCore.create(config_overrides={"llmcore": {"default_provider": "openai"}})`



### `default_embedding_model`



- **What it does:** Sets the default embedding model used for all Retrieval Augmented Generation (RAG) operations, including adding documents to the vector store and performing similarity searches for queries.
- **Purpose:** This centralizes the choice of embedding model for RAG, ensuring consistency across the application's knowledge base.
- **Possible Options:** A string identifier that specifies the model. The format depends on the model type:
    - For local Sentence Transformers models: The model name from Hugging Face Hub or a local path (e.g., `"all-MiniLM-L6-v2"`).
    - For service-based models: A string in the format `"provider:model_name"` (e.g., `"openai:text-embedding-3-small"`, `"google:models/embedding-001"`).
- **Default Value:** `"all-MiniLM-L6-v2"`
- **How to Override:**
    - **Environment Variable:** `export LLMCORE_DEFAULT_EMBEDDING_MODEL="openai:text-embedding-3-small"`
    - **Code Override:** `LLMCore.create(config_overrides={"llmcore": {"default_embedding_model": "ollama:mxbai-embed-large"}})`



### `log_level`



- **What it does:** Configures the logging verbosity for the entire `llmcore` library and its components.
- **Purpose:** Allows developers to control the amount of diagnostic information emitted by the library, from critical errors only to detailed debugging traces.
- **Possible Options:** Standard Python logging levels as strings: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`.
- **Default Value:** `"INFO"`
- **How to Override:**
    - **Environment Variable:** `export LLMCORE_LOG_LEVEL="DEBUG"`
    - **Code Override:** `LLMCore.create(config_overrides={"llmcore": {"log_level": "WARNING"}})`



### `log_raw_payloads`



- **What it does:** A powerful debugging feature that, when enabled, instructs `llmcore` to log the exact raw JSON request and response payloads exchanged with the underlying LLM provider APIs. This logging occurs at the `DEBUG` level.
- **Purpose:** This is an indispensable tool for advanced troubleshooting. It allows developers to inspect the precise data being sent to and received from a provider, which is essential for diagnosing `ProviderError` exceptions, understanding unexpected model behavior, verifying that custom parameters are being correctly formatted, or reverse-engineering API interactions.
- **Possible Options:** `true` or `false`.
- **Default Value:** `false`
- **How to Override:**
    - **Environment Variable:** `export LLMCORE_LOG_RAW_PAYLOADS=true`
    - **Code Override:** `LLMCore.create(config_overrides={"llmcore": {"log_raw_payloads": true}})`



## Configuring LLM Providers: The `[providers]` Section



The `[providers]` section is where all individual LLM provider instances are defined. The structure is a main `[providers]` table that contains nested tables for each configured provider instance, such as `[providers.openai]` or `[providers.ollama]`.1

A key feature of this system is the decoupling of the provider instance's name from its underlying type. A user can name an instance descriptively (e.g., `[providers.azure_openai]`) and then specify its behavior using the `type` key (e.g., `type = "openai"`). If the `type` key is omitted, the system assumes the instance name matches the type (e.g., `[providers.openai]` implicitly has `type = "openai"`).1

This design provides significant flexibility for complex setups. For example, a developer can configure multiple distinct OpenAI-compatible endpoints, each with its own name, `base_url`, and `api_key`, while all leveraging the same core `openai` provider logic.

**Example of configuring two OpenAI-compatible providers:**

Ini, TOML

```
[providers.official_openai]
type = "openai"
api_key = "${OPENAI_API_KEY}" # Using an env var
default_model = "gpt-4o"

[providers.groq_api]
type = "openai" # Leverages the OpenAI provider logic
base_url = "https://api.groq.com/openai/v1"
api_key = "${GROQ_API_KEY}"
default_model = "llama3-70b-8192"
```

The following subsections detail the specific parameters available for each supported provider type.1



### Common Parameters



These parameters are generally applicable across most providers.

- `api_key`: A string containing the secret API key for the provider. It is strongly recommended to set this using an environment variable rather than hardcoding it in the configuration file.
- `default_model`: A string specifying the model identifier to use for this provider if one is not explicitly passed to the `chat()` method.
- `timeout`: An integer representing the timeout in seconds for API requests to the provider.



### `openai` Provider



- **`type`**: `"openai"`
- **Configuration File Section**: `[providers.openai]` or `[providers.<custom_name>]` with `type = "openai"`
- **Dependencies**: `llmcore[openai]`
- **Parameters**:
    - `api_key` (string, optional): Your OpenAI API key. Can also be set via `LLMCORE_PROVIDERS__OPENAI__API_KEY` or the standard `OPENAI_API_KEY` environment variables.
    - `base_url` (string, optional): The base URL for the API endpoint. Defaults to `https://api.openai.com/v1`. This can be overridden to use proxies or other OpenAI-compatible APIs like Groq or local models served via LiteLLM.
    - `default_model` (string, optional): The default model to use. Default is `"gpt-4o"`.
    - `timeout` (integer, optional): Request timeout in seconds. Default is 60.



### `anthropic` Provider



- **`type`**: `"anthropic"`
- **Configuration File Section**: `[providers.anthropic]`
- **Dependencies**: `llmcore[anthropic]`
- **Parameters**:
    - `api_key` (string, optional): Your Anthropic API key. Can also be set via `LLMCORE_PROVIDERS__ANTHROPIC__API_KEY` or `ANTHROPIC_API_KEY` environment variables.
    - `default_model` (string, optional): The default Claude model to use. Default is `"claude-3-opus-20240229"`.
    - `timeout` (integer, optional): Request timeout in seconds. Default is 60.



### `gemini` Provider



- **`type`**: `"gemini"`
- **Configuration File Section**: `[providers.gemini]`
- **Dependencies**: `llmcore[gemini]`
- **Parameters**:
    - `api_key` (string, optional): Your Google AI API key. Can also be set via `LLMCORE_PROVIDERS__GEMINI__API_KEY` or `GOOGLE_API_KEY` environment variables.
    - `default_model` (string, optional): The default Gemini model to use. Default is `"gemini-1.5-pro-latest"`.
    - `timeout` (integer, optional): Request timeout in seconds. Default is 60.
    - `safety_settings` (table, optional): Allows configuration of Google AI's safety filters. This is passed directly to the `google-genai` SDK.



### `ollama` Provider



- **`type`**: `"ollama"`
- **Configuration File Section**: `[providers.ollama]`
- **Dependencies**: `llmcore[ollama]`
- **Parameters**:
    - `host` (string, optional): The hostname and port of the running Ollama server (e.g., `"http://localhost:11434"`). The official `ollama` Python library uses this parameter.
    - `default_model` (string, optional): The default local model to use (e.g., `"llama3"`). Ensure this model has been pulled via `ollama pull <model_name>`. Default is `"llama3"`.
    - `timeout` (integer, optional): Request timeout in seconds. Default is 120, allowing more time for local model generation.
    - `tokenizer` (string, optional): Specifies the tokenizer to use for token counting, as this can vary for local models. Options include:
        - `"tiktoken_cl100k_base"`: (Default) A good general-purpose tokenizer used by GPT-4.
        - `"tiktoken_p50k_base"`: Another `tiktoken` option.
        - `"char_div_4"`: A rough approximation based on character count. Use this if `tiktoken` is problematic for a specific model.



## Managing Persistence: The `[storage]` Section



The `[storage]` section is critical for enabling stateful capabilities in `llmcore`, such as conversation history and Retrieval Augmented Generation (RAG). It governs where and how persistent data is stored and is divided into two primary subsections: `session` and `vector`.1



### Session & Episodic Memory (`[storage.session]`)



This subsection configures the backend for storing conversation history (the sequence of user and assistant messages) and agent episodic memory (the log of an agent's thoughts, actions, and observations). The choice of backend has significant implications for application scalability.1

- **`type`** (string, required): Determines the storage backend to use.
    - **Options:**
        - `"json"`: Stores each session as a separate JSON file. Simple and human-readable, but not suitable for applications with high concurrency or a large number of sessions.
        - `"sqlite"`: Stores all sessions in a single SQLite database file. Excellent for single-process applications, local development, and desktop tools. It is the default option.
        - `"postgres"`: Stores sessions in a PostgreSQL database. This is the recommended choice for production applications, especially those running the multi-process API server and TaskMaster workers, as it provides robust, concurrent access to session data.
    - **Default:** `"sqlite"`
- **`path`** (string, optional): The filesystem path for file-based storage types (`json` and `sqlite`). The `~` character is automatically expanded to the user's home directory.
    - **Default:** `"~/.llmcore/sessions.db"`
- **`db_url`** (string, optional): The full database connection URL for `postgres`. It is strongly recommended to set this via the `LLMCORE_STORAGE__SESSION__DB_URL` environment variable to avoid committing secrets to version control.
    - **Example:** `"postgresql://user:password@host:port/database_name"`
    - **Default:** `""` (empty string)



### Vector Storage for RAG (`[storage.vector]`)



This subsection configures the vector store, which is the foundation of `llmcore`'s semantic memory and RAG capabilities. It stores documents and their corresponding numerical vector embeddings, enabling efficient similarity searches.1

- **`type`** (string, required): Determines the vector database backend to use.
    - **Options:**
        - `"chromadb"`: An open-source embedding database that is easy to set up for local development and can run in-memory or persistently on disk. It is the default option.
        - `"pgvector"`: An extension for PostgreSQL that adds vector similarity search capabilities. This is an excellent choice for production environments that already use PostgreSQL, as it co-locates vector data with other application data.
    - **Default:** `"chromadb"`
- **`default_collection`** (string, optional): Specifies a default collection name for RAG operations. A collection is analogous to a table in a relational database; it isolates a set of documents and their embeddings. This allows an application to maintain multiple distinct knowledge bases within the same vector store.
    - **Default:** `"llmcore_default_rag"`
- **`path`** (string, optional): The filesystem path for file-based vector stores like `chromadb` when running in persistent mode.
    - **Default:** `"~/.llmcore/chroma_db"`
- **`db_url`** (string, optional): The database connection URL for `pgvector`. It is recommended to set this via the `LLMCORE_STORAGE__VECTOR__DB_URL` environment variable.
    - **Default:** `""` (empty string)



## Vectorization Settings: The `[embedding]` Section



This section configures the embedding models responsible for converting text into numerical vectors (embeddings). These vectors are essential for the functioning of the RAG system, as they allow the vector store to find documents semantically similar to a user's query.1

The `EmbeddingManager` uses the global `llmcore.default_embedding_model` setting to determine which model to use. If that identifier contains a colon (e.g., `"openai:text-embedding-3-small"`), the part before the colon tells the manager which configuration block within the `[embedding]` section to consult for specific settings like API keys or host URLs.1



### `[embedding.openai]`



- **Purpose:** Configures OpenAI's embedding models.
- **Parameters:**
    - `api_key` (string, optional): API key for OpenAI embeddings. It is recommended to use the `LLMCORE_EMBEDDING__OPENAI__API_KEY` environment variable.
    - `default_model` (string, optional): The specific OpenAI embedding model to use. Default is `"text-embedding-3-small"`.
    - `base_url` (string, optional): Can be set to use a proxy or alternative OpenAI-compatible endpoint for embeddings.



### `[embedding.google]`



- **Purpose:** Configures Google AI (Gemini) embedding models.
- **Parameters:**
    - `api_key` (string, optional): API key for Google AI embeddings. Recommended to use the `LLMCORE_EMBEDDING__GOOGLE__API_KEY` environment variable.
    - `default_model` (string, optional): The specific Google embedding model to use. Default is `"models/embedding-001"`.



### `[embedding.ollama]`



- **Purpose:** Configures the use of a local Ollama instance for generating embeddings.
- **Parameters:**
    - `host` (string, optional): The URL of the Ollama server, if different from the one used for chat.
    - `default_model` (string, optional): The name of the embedding model to use from the Ollama instance (e.g., `"mxbai-embed-large"`, `"nomic-embed-text"`). Ensure the model is pulled locally. Default is `"mxbai-embed-large"`.
    - `timeout` (integer, optional): Request timeout in seconds. Default is 60.



### `[embedding.sentence_transformer]`



- **Purpose:** Provides additional configuration for local `sentence-transformers` models. The model itself is specified in `llmcore.default_embedding_model`.
- **Parameters:**
    - `device` (string, optional): The device to run the model on. Options include `"cpu"`, `"cuda"`, `"mps"`. If not specified, the library will attempt to auto-detect the best available device.



## Advanced Prompt Control: The `[context_management]` Section



This section is the control center for prompt engineering within `llmcore`. It governs the sophisticated, multi-stage process of assembling the final context that is sent to the LLM. This system is far more advanced than simple history truncation; it first prioritizes what information to *include*, and only then decides what to *truncate* if the model's token limit is exceeded.1



### The Two-Stage Context Assembly Process



1. **Inclusion Stage:** `llmcore` first gathers all potential pieces of context (system messages, chat history, RAG results, etc.). It then adds them to the prompt in the order specified by the `inclusion_priority` list. This ensures that the most critical information is always considered first.
2. **Truncation Stage:** After the inclusion stage, if the total token count of the assembled prompt exceeds the model's context window limit, `llmcore` begins to remove content. The `truncation_priority` list dictates the order in which content types are removed, protecting higher-priority information from being discarded.



### Parameter Deep Dive



- **`inclusion_priority`** (string): A comma-separated list defining the order in which context components are added to the prompt.

    - **Purpose:** To give developers fine-grained control over the structure of the final prompt. For example, to build an agent that always prioritizes its explicit instructions (`explicitly_staged`) over general chat history (`history_chat`), you would ensure `explicitly_staged` appears before `history_chat` in this list.
    - **Valid Components:** `"system_history"`, `"explicitly_staged"`, `"user_items_active"`, `"history_chat"`, `"final_user_query"`.
    - **Default:** `"system_history,explicitly_staged,user_items_active,history_chat,final_user_query"`

- **`truncation_priority`** (string): A comma-separated list defining the order in which context types are removed to meet token limits.

    - **Purpose:** To control what information is sacrificed when the context is too long. For instance, if you want to preserve as much conversational history as possible at the expense of RAG context, you would set the priority to `"rag_in_query,history_chat"`. This tells `llmcore` to discard the retrieved RAG documents before it starts removing old messages from the chat history.
    - **Valid Components:** `"history_chat"`, `"user_items_active"`, `"rag_in_query"`, `"explicitly_staged"`.
    - **Default:** `"history_chat,user_items_active,rag_in_query,explicitly_staged"`

- **`default_prompt_template`** and **`prompt_template_path`** (string): These settings control the final formatting of RAG-enabled queries. The `default_prompt_template` contains a pre-defined template string with placeholders like `{context}` for the retrieved documents and `{question}` for the user's query. The `prompt_template_path` can be used to specify a path to a custom template file, which will override the default string.

    - **Default Template:**

        ```
        You are an AI assistant specialized in answering questions about codebases based on provided context.
        Use ONLY the following pieces of retrieved context to answer the user's question.
        If the answer is not found in the context, state that you cannot answer based on the provided information.
        Do not make up an answer or use external knowledge. Keep the answer concise and relevant to the question.
        Include relevant source file paths and line numbers if possible, based *only* on the provided context metadata.
        
        Context:
        ---------------------
        {context}
        ---------------------
        
        Question: {question}
        
        Answer:
        ```

- **`rag_retrieval_k`** (integer): The default number of documents to retrieve from the vector store for RAG. Default is `3`.

- **`reserved_response_tokens`** (integer): The number of tokens to leave free in the context window for the model's response. Default is `500`.

- **`minimum_history_messages`** (integer): The minimum number of recent chat messages to try to keep during truncation. Default is `1`.

- **`user_retained_messages_count`** (integer): Prioritizes keeping the N most recent user messages (and their preceding assistant responses). Default is `5`.

- **`max_chars_per_user_item`** (integer): A safeguard that sets a maximum character limit for a single user-provided context item before it is truncated. The default value in v0.21.0 was significantly increased to `40000000`, allowing for very large documents to be included in the context pool.



## Integrated Data Ingestion: The `[apykatu]` Section



The `[apykatu]` section is a new and powerful feature in `llmcore` v0.21.0. It functions as a configuration "control plane" for the `apykatu` data ingestion library. The settings in this section are not used directly by `llmcore`'s `chat()` method; instead, they are passed to the `ingest_data_task` background job, which is executed by the TaskMaster service. This allows developers to manage the entire data ingestion pipeline—from file discovery and code-aware chunking to embedding and storage—from the same central `llmcore` configuration file.1



### `[apykatu.database]`



- **Purpose:** Configures the vector store that `apykatu` will write to during the ingestion process.
- **Parameters:**
    - `type` (string): The type of vector store. Default is `"chromadb"`.



### `[apykatu.embeddings]`



- **Purpose:** Defines the embedding models that `apykatu` should use during ingestion. This allows for using a different (perhaps more powerful or costly) model for the one-time ingestion process than the one used for real-time query embedding.
- **Parameters:**
    - `default_model` (string): An identifier that points to a model defined under `[apykatu.embeddings.models]`.
    - `[apykatu.embeddings.models.<name>]`: Nested tables defining specific embedding models, including their `provider` (e.g., `"sentence-transformers"`, `"ollama"`), `model_name`, and `device`.



### `[apykatu.chunking]`



- **Purpose:** Specifies strategies for breaking down large documents and source code files into smaller, semantically meaningful chunks suitable for embedding.
- **Parameters:**
    - `default_strategy` (string): The default chunking method (e.g., `"RecursiveSplitter"`).
    - `[apykatu.chunking.strategies]`: Defines strategies for specific file types, such as using code-aware chunking for `.py` files based on function and class definitions.
    - `[apykatu.chunking.parameters]`: Sets parameters like `chunk_size` and `chunk_overlap` for the different strategies.



### `[apykatu.discovery]`



- **Purpose:** Controls which files and directories are included or excluded from the ingestion process.
- **Parameters:**
    - `use_gitignore` (boolean): If `true`, `apykatu` will respect the rules in `.gitignore` files.
    - `excluded_dirs` (list of strings): A list of directory names to always exclude.
    - `excluded_files` (list of strings): A list of file patterns to always exclude.



## Practical Configuration Recipes



The following sections provide complete, commented `config.toml` examples for common scenarios, synthesizing the detailed explanations from the previous sections.



### Recipe 1: Local Development Setup



This configuration is optimized for getting started quickly on a local machine for development and experimentation. It relies on local, file-based services that require minimal setup.

Ini, TOML

```
# --- LLMCore Configuration for Local Development ---

[llmcore]
# Use a local Ollama instance as the default LLM provider.
# Requires Ollama to be installed and running.
default_provider = "ollama"

# Use a local Sentence Transformers model for RAG embeddings.
# The model will be downloaded automatically on first use.
default_embedding_model = "all-MiniLM-L6-v2"

# Set log level to DEBUG for detailed output during development.
log_level = "DEBUG"
log_raw_payloads = false

[providers.ollama]
# The default model to use with Ollama.
# Ensure you have run 'ollama pull llama3' locally.
default_model = "llama3"
# Increase timeout for local models which can be slower to respond.
timeout = 120

[storage.session]
# Use SQLite for session history. It's a simple, single-file database.
type = "sqlite"
# Store the session database in the user's home directory.
path = "~/.llmcore/sessions.db"

[storage.vector]
# Use ChromaDB for the RAG vector store.
type = "chromadb"
# Store the ChromaDB data persistently in the user's home directory.
path = "~/.llmcore/chroma_db"
# Define a default collection for RAG documents.
default_collection = "local_dev_docs"
```



### Recipe 2: Production-Ready Cloud Setup



This configuration is designed for a robust, scalable production deployment. It uses cloud-based services and relies on environment variables for all secrets, which is a security best practice.

Ini, TOML

```
# --- LLMCore Configuration for Production ---

[llmcore]
# Use OpenAI as the default provider.
default_provider = "openai"
# Use OpenAI's high-performance embedding model.
default_embedding_model = "openai:text-embedding-3-small"
# Set log level to INFO for production to reduce noise.
log_level = "INFO"
log_raw_payloads = false

[providers.openai]
# IMPORTANT: The API key is NOT set here. It MUST be set via the
# LLMCORE_PROVIDERS__OPENAI__API_KEY environment variable.
# api_key = ""
default_model = "gpt-4o"
timeout = 90

[storage.session]
# Use PostgreSQL for session and episodic memory storage.
# This is required for multi-process applications (API server + workers).
type = "postgres"
# IMPORTANT: The database URL is NOT set here. It MUST be set via the
# LLMCORE_STORAGE__SESSION__DB_URL environment variable.
# db_url = "postgresql://user:pass@host:port/dbname"

[storage.vector]
# Use PostgreSQL with the pgvector extension for the vector store.
# This co-locates vector data with other application data for easier management.
type = "pgvector"
# IMPORTANT: The database URL is NOT set here. It MUST be set via the
# LLMCORE_STORAGE__VECTOR__DB_URL environment variable.
# db_url = "postgresql://user:pass@host:port/dbname"
default_collection = "production_knowledge_base"

[embedding.openai]
# Configuration for the OpenAI embedding model.
# The API key will be inherited from the main OpenAI provider or can be
# set separately via LLMCORE_EMBEDDING__OPENAI__API_KEY.
default_model = "text-embedding-3-small"
```



### Recipe 3: Deep Debugging Configuration



This configuration is tailored for intensive troubleshooting and inspection of the library's internal behavior. It enables maximum logging verbosity.

Ini, TOML

```
# --- LLMCore Configuration for Deep Debugging ---

[llmcore]
# Use any provider, e.g., Ollama for local debugging.
default_provider = "ollama"
default_embedding_model = "all-MiniLM-L6-v2"

# Set log level to DEBUG to see all internal library logs.
log_level = "DEBUG"

# CRITICAL: Enable this to see the exact JSON payloads sent to and
# received from the LLM provider's API. This is invaluable for
# diagnosing provider errors or unexpected model responses.
log_raw_payloads = true

[providers.ollama]
default_model = "llama3"
timeout = 180

[storage.session]
# Use a separate SQLite DB for debugging to avoid polluting production data.
type = "sqlite"
path = "~/.llmcore/debug_sessions.db"

[storage.vector]
type = "chromadb"
path = "~/.llmcore/debug_chroma_db"
default_collection = "debug_docs"
```



## Conclusion



The configuration system of LLMCore v0.21.0 is a testament to its design as a serious, production-ready platform for AI application development. By leveraging a layered approach, it provides developers with a remarkable degree of control and flexibility. The system gracefully scales from simple, single-file setups for local scripting to complex, environment-variable-driven configurations for secure, containerized production deployments.

The granular control offered in sections like `[context_management]` and the innovative integration with data pipelines via the `[apykatu]` section demonstrate a deep understanding of the practical challenges in building advanced AI systems. By mastering this configuration system, developers can unlock the full potential of `llmcore`, precisely tailoring its behavior to meet the unique demands of their applications and ensuring that the platform is not just a library, but a robust and scalable foundation for their work.