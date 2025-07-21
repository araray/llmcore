# LLMCore v0.21.0: A Guide for Developers





## Architecture and Core Concepts



LLMCore v0.21.0 has evolved from a simple library into a comprehensive platform for building advanced AI applications. This evolution introduces a multi-component architecture designed for scalability, responsiveness, and the development of autonomous systems. Understanding this architecture is key to leveraging the full power of the platform. The system consists of three primary, interconnected components: the core `LLMCore` library, a full-featured `API Server`, and an asynchronous `TaskMaster` backend.1 This decoupled design allows long-running operations, such as data ingestion or complex agent tasks, to be offloaded to background workers. This ensures the API server remains highly responsive, which is critical for a positive user experience and overall system stability.1

At the heart of the new agentic capabilities is a sophisticated, three-tiered memory system. This structure provides a powerful mental model for building agents that can reason, remember, and learn from their interactions. The introduction of `AgentState` and `Episode` models in the codebase, alongside the existing RAG functionality, makes this hierarchical memory explicit.1

The three memory tiers are:

1. **Working Memory:** This is the agent's short-term "scratchpad" or cognitive workspace. It holds the transient state required for the agent's immediate reasoning loop, such as its current goal, plan, and recent thoughts. This is represented by the `AgentState` model.1
2. **Episodic Memory:** This is the agent's long-term log of past experiences. Every significant event—a thought, an action taken, or an observation made—is recorded as an `Episode`. This allows the agent to recall specific past interactions and learn from them over time. This tier is managed via new methods in the storage backends.1
3. **Semantic Memory:** This is the agent's long-term knowledge base of facts and information. It is powered by the Retrieval Augmented Generation (RAG) system and a vector store, allowing the agent to query a corpus of external documents to find relevant information.



## Advanced Installation and Environment Setup



To use all features of the LLMCore platform, including the API server and background workers, you will need to install all optional dependencies and set up the required external services.



### Full Installation



Install the core library along with all extras for providers, storage backends, and embedding models using the `[all]` tag.1

Bash

```
pip install llmcore[all]
```



### External Service Setup



LLMCore's backend components rely on external services for task queuing and persistent storage.

- **Ollama (for local models):**

    1. Install Ollama from [ollama.com](https://ollama.com/).

    2. Pull the models you intend to use. For example, to get Llama 3 and a common embedding model:

        Bash

        ```
        ollama pull llama3
        ollama pull mxbai-embed-large
        ```

- **PostgreSQL with pgvector (for advanced storage):**

    1. Run a PostgreSQL instance (e.g., via Docker).

    2. Connect to your database and enable the `pgvector` extension:

        SQL

        ```
        CREATE EXTENSION IF NOT EXISTS vector;
        ```

- **Redis (for the TaskMaster service):**

    1. Run a Redis instance. The simplest way is using Docker:

        Bash

        ```
        docker run -d -p 6379:6379 --name llmcore-redis redis:latest
        ```

    2. The `TaskMaster` worker and API server will connect to this instance by default.



## In-Depth Configuration (`default_config.toml`)



LLMCore's behavior is controlled by a powerful, layered configuration system. The `src/llmcore/config/default_config.toml` file serves as the comprehensive template for all available settings.1



### `[llmcore]`



This section defines global library settings.

- `default_provider` (string): Specifies the default LLM provider (e.g., `"ollama"`). Must correspond to a section name under `[providers]`.
- `default_embedding_model` (string): Sets the default embedding model for RAG (e.g., `"all-MiniLM-L6-v2"`).
- `log_level` (string): Configures the logging verbosity (e.g., `"INFO"`, `"DEBUG"`).
- `log_raw_payloads` (boolean): A new and powerful debugging feature. If `true`, LLMCore will log the raw request and response JSON payloads sent to and received from LLM providers at the `DEBUG` level.1



### `[providers.<name>]`



This section configures individual LLM provider instances. You can define multiple instances of the same provider type, for example, to connect to different OpenAI-compatible endpoints.

Ini, TOML

```
[providers.my_openai_clone]
# Explicitly state this uses the "openai" provider logic
type = "openai" 
# Specific API key for this backend (or use env var)
api_key = "..." 
base_url = "https://api.example.com/v1"
default_model = "custom-model-v1"
```



### `[storage]`



Configures session history and vector storage backends.

- **`[storage.session]`**: For conversation history.
    - `type`: `"json"`, `"sqlite"`, or `"postgres"`.
    - `path`: Filesystem path for `json` or `sqlite`.
    - `db_url`: Connection string for `postgres`.
- **`[storage.vector]`**: For RAG documents.
    - `type`: `"chromadb"` or `"pgvector"`.
    - `path`: Filesystem path for `chromadb`.
    - `db_url`: Connection string for `pgvector`.
    - `default_collection`: The default collection name for RAG operations.



### `[embedding]`



Configures specific embedding models, especially those requiring API keys or special settings. A new `[embedding.ollama]` section allows for configuring a local Ollama instance for embedding generation.1



### `[context_management]`



This section controls the sophisticated context assembly engine. The logic has been significantly enhanced from a simple truncation system to a multi-stage, rule-based process. The system first determines *what* to include in the prompt based on priority, and only then decides *what* to remove if the context window limit is exceeded. This provides far more granular and intelligent control over the final prompt sent to the LLM.1

- `inclusion_priority` (string): **New.** A comma-separated list defining the order in which context components are added to the prompt. Valid components are: `"system_history"`, `"explicitly_staged"`, `"user_items_active"`, `"history_chat"`, and `"final_user_query"`.
- `truncation_priority` (string): **Updated.** A comma-separated list defining the order in which context types are truncated if the token limit is exceeded. Valid components are: `"history_chat"`, `"user_items_active"`, `"rag_in_query"`, `"explicitly_staged"`.
- `default_prompt_template` (string): A new, more detailed default template for RAG that instructs the model to answer based only on the provided context.
- `max_chars_per_user_item` (integer): The default character limit for a single context item before it is truncated has been significantly increased to `40000000`, allowing for very large documents to be included in the context pool.1



### `[apykatu]`



This entire section is **new** and demonstrates LLMCore's deep integration with the `apykatu` data ingestion library. LLMCore acts as a configuration "control plane" for `apykatu`, allowing developers to manage the entire data pipeline—from code-aware chunking and embedding to vector store population—from a single configuration file. This tight integration streamlines the process of building and maintaining the knowledge base for RAG applications.1

Key sub-sections include:

- `[apykatu.database]`: Configures the vector store `apykatu` will write to.
- `[apykatu.embeddings]`: Defines the embedding models `apykatu` should use for ingestion.
- `[apykatu.chunking]`: Specifies code-aware and semantic chunking strategies.
- `[apykatu.discovery]`: Controls how files are discovered, including respecting `.gitignore`.



### Configuration via Environment Variables



For production deployments, using environment variables is standard practice. Any setting in the `toml` file can be overridden. The variable name is constructed by prefixing with `LLMCORE_`, converting the path to uppercase, and replacing dots (`.`) with double underscores (`__`).

| Configuration Path               | Environment Variable                       |
| -------------------------------- | ------------------------------------------ |
| `llmcore.default_provider`       | `LLMCORE_DEFAULT_PROVIDER`                 |
| `providers.openai.api_key`       | `LLMCORE_PROVIDERS__OPENAI__API_KEY`       |
| `storage.session.type`           | `LLMCORE_STORAGE__SESSION__TYPE`           |
| `storage.vector.db_url`          | `LLMCORE_STORAGE__VECTOR__DB_URL`          |
| `embedding.ollama.default_model` | `LLMCORE_EMBEDDING__OLLAMA__DEFAULT_MODEL` |



## Mastering the Core API: The `LLMCore` Class



The `LLMCore` class is the primary interface for interacting with the library's features. It should always be instantiated via the asynchronous `LLMCore.create()` class method, preferably within an `async with` block to ensure proper resource management.



### `chat()` Method Deep Dive



The `llm.chat()` method is the cornerstone of the library. It has been significantly enhanced to support standardized tool-calling, a critical feature for building agents and complex applications.1

| Parameter           | Type            | Description                                                  | Default    |
| ------------------- | --------------- | ------------------------------------------------------------ | ---------- |
| `message`           | `str`           | The user's input message.                                    | (Required) |
| `session_id`        | `Optional[str]` | ID for the conversation session.                             | `None`     |
| `system_message`    | `Optional[str]` | Sets the LLM's behavior.                                     | `None`     |
| `provider_name`     | `Optional[str]` | Overrides the default provider.                              | `None`     |
| `model_name`        | `Optional[str]` | Overrides the provider's default model.                      | `None`     |
| `stream`            | `bool`          | If `True`, returns a stream of text chunks.                  | `False`    |
| `save_session`      | `bool`          | If `True`, saves the turn to persistent storage.             | `True`     |
| `tools`             | `Optional]`     | **New:** A list of `Tool` objects for the LLM to call.       | `None`     |
| `tool_choice`       | `Optional[str]` | **New:** Controls how the model uses tools (e.g., "auto", "any"). | `None`     |
| `**provider_kwargs` | `Any`           | Additional arguments passed directly to the provider.        | `{}`       |

The `tools` and `tool_choice` parameters are passed directly to the underlying provider's `chat_completion` method, allowing `LLMCore` to leverage the native function-calling capabilities of models like GPT-4o and Claude 3.1



### Other Public Methods



- `get_agent_manager() -> AgentManager`: **New.** Returns the initialized `AgentManager` instance, which is the entry point for running autonomous agent tasks.1
- `reload_config() -> None`: **New.** Performs a live reload of the configuration from all sources, re-initializing all managers while preserving transient state.1
- `get_models_details_for_provider(provider_name: str) -> List`: **New.** Provides detailed information about a provider's models, including context length and support for tools and streaming.1
- `get_available_providers() -> List[str]`: Lists the names of all successfully loaded provider instances.
- `get_session(session_id: str) -> Optional`: Retrieves a specific session from storage if it exists.
- `list_sessions() -> List]`: Lists metadata for all persistent sessions.
- `delete_session(session_id: str) -> bool`: Deletes a session from storage.
- RAG Methods: `add_documents_to_vector_store()`, `search_vector_store()`, etc., remain the primary interface for managing the semantic memory.



## The Agentic Framework: Building Autonomous Agents



LLMCore v0.21.0 introduces a complete framework for building autonomous agents. This framework is built upon the hierarchical memory system and a reasoning loop inspired by the ReAct (Reason + Act) paradigm.



### The `AgentManager` and the ReAct Loop



The `AgentManager` orchestrates the agent's execution cycle. The primary method, `run_agent_loop`, repeatedly executes a "Think -> Act -> Observe" sequence until the agent's goal is achieved or a maximum number of iterations is reached.1

1. **Think (`_think_step`):** The agent assesses its current `goal` and `AgentState` (working memory). It queries the `MemoryManager` to retrieve relevant context from both semantic (RAG) and episodic (past experiences) memory. This combined context is used to build a prompt, asking the LLM to reason about the next best action to take. The LLM's output is a "thought" and a `ToolCall`.
2. **Act (`_act_step`):** The `AgentManager` takes the `ToolCall` from the LLM and passes it to the `ToolManager` for execution. The `ToolManager` invokes the appropriate tool with the specified arguments.
3. **Observe (`_observe_step`):** The agent receives the `ToolResult` from the executed tool. This result, or "observation," is added to its working memory (`AgentState`). The entire "Thought -> Act -> Observe" cycle is then logged as a new `Episode` in the agent's episodic memory, allowing it to learn from this experience in the future.



### Using the `ToolManager`



The `ToolManager` handles the registration and execution of all functions available to an agent. It comes with a set of powerful built-in tools 1:

- `semantic_search`: Queries the RAG vector store for factual information.
- `episodic_search`: Queries the agent's past experiences to recall prior interactions.
- `calculator`: Performs mathematical calculations.
- `finish`: A special tool the agent uses to signal that it has successfully completed its goal and to provide the final answer.

You can easily extend the agent's capabilities by creating and registering custom tools.

**Example: Creating a Custom Tool**

Python

```
import asyncio
from llmcore import LLMCore
from llmcore.models import Tool

# 1. Define the tool's function
async def get_current_weather(location: str) -> str:
    """Gets the current weather for a given location."""
    # In a real implementation, this would call a weather API
    if "tokyo" in location.lower():
        return "The weather in Tokyo is 25°C and sunny."
    return f"Sorry, I don't have weather information for {location}."

async def main():
    async with await LLMCore.create() as llm:
        # 2. Get the AgentManager and its ToolManager
        agent_manager = llm.get_agent_manager()
        tool_manager = agent_manager.get_tool_manager()

        # 3. Define the tool's schema
        weather_tool_definition = Tool(
            name="get_current_weather",
            description="Use this tool to get the current weather for a specific city.",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g., 'San Francisco'."
                    }
                },
                "required": ["location"]
            }
        )

        # 4. Register the custom tool
        tool_manager.register_tool(
            func=get_current_weather,
            definition=weather_tool_definition
        )
        print("Custom weather tool registered.")
        
        # Now the agent can use 'get_current_weather' in its run loop.

if __name__ == "__main__":
    asyncio.run(main())
```



## The Asynchronous Backend: TaskMaster and API Server



LLMCore's backend components provide a scalable and robust way to serve AI functionality and handle long-running jobs.



### The `TaskMaster` Service



The `TaskMaster` is an asynchronous task queue system built on `arq` and Redis. It runs as a separate worker process, listening for jobs to execute. This is essential for offloading intensive operations like data ingestion or multi-step agent tasks, keeping the API server responsive.1

The worker is configured in `src/llmcore/task_master/worker.py` and registers two primary tasks:

- `ingest_data_task`: Handles data ingestion using the `apykatu` library. It can process local files, ZIP archives, and clone Git repositories to populate the semantic memory.
- `run_agent_task`: Executes the `AgentManager.run_agent_loop` for a given goal, allowing agents to run completely in the background.

To start the worker, run the following command from the project root:

Bash

```
python -m llmcore.task_master.main
```



### Comprehensive API Reference



The FastAPI server exposes the full power of LLMCore through a well-defined REST API. The API is versioned, with core chat at `/v1` and newer features like memory, tasks, and agents at `/v2`.1

| Endpoint                         | Method | Description                                                  |
| -------------------------------- | ------ | ------------------------------------------------------------ |
| `/`                              | `GET`  | Root endpoint with basic service info.                       |
| `/health`                        | `GET`  | Health check for monitoring.                                 |
| `/api/v1/info`                   | `GET`  | Get detailed service capabilities and versions.              |
| `/api/v1/chat`                   | `POST` | Core endpoint for chat interactions (streaming/non-streaming). |
| `/api/v2/memory/semantic/search` | `GET`  | Search the semantic memory (vector store).                   |
| `/api/v2/submit`                 | `POST` | Submit a generic asynchronous task.                          |
| `/api/v2/{task_id}`              | `GET`  | Get the status of an asynchronous task.                      |
| `/api/v2/{task_id}/result`       | `GET`  | Get the result of a completed task.                          |
| `/api/v2/{task_id}/stream`       | `GET`  | Stream real-time progress for a task.                        |
| `/api/v2/ingestion/submit`       | `POST` | Submit a data ingestion task (file, zip, git).               |
| `/api/v2/run`                    | `POST` | **New:** Start an autonomous agent task.                     |

**Example: Starting an Agent Task via the API**

Bash

```
curl -X POST "http://127.0.0.1:8000/api/v2/run" \
-H "Content-Type: application/json" \
-d '{
  "goal": "Research the current weather in Tokyo and then write a short haiku about it.",
  "session_id": "agent-weather-task-123"
}'
```

This request will return a `task_id`. You can then use the `/api/v2/{task_id}/result` endpoint to retrieve the final haiku once the agent has completed its work.



## Error Handling and Debugging



LLMCore uses a rich hierarchy of custom exceptions to allow for specific and robust error handling. All custom exceptions inherit from `LLMCoreError`.1

Key exceptions include:

- `ConfigError`: For issues with configuration files or environment variables.
- `ProviderError`: For errors from an LLM provider's API (e.g., invalid API key, model not found).
- `StorageError`: Base class for storage-related issues.
    - `SessionNotFoundError`: When a requested `session_id` does not exist.
- `ContextLengthError`: When a prompt exceeds the model's context window limit, even after truncation.
- `EmbeddingError`: For failures during the text embedding process.

**Best Practice Error Handling:**

Python

```
import asyncio
from llmcore import (
    LLMCore, LLMCoreError, ProviderError, ContextLengthError, SessionNotFoundError
)

async def robust_chat():
    try:
        async with await LLMCore.create() as llm:
            await llm.chat(
                "This might fail.",
                session_id="non_existent_session", # This will likely cause an error if session doesn't exist
                provider_name="misconfigured_provider"
            )
    except ContextLengthError as e:
        print(f"Error: Context too long for model '{e.model_name}'. Limit: {e.limit}, Actual: {e.actual}")
    except ProviderError as e:
        print(f"Error with provider '{e.provider_name}': {e}")
    except SessionNotFoundError as e:
        print(f"Session Error: {e}")
    except LLMCoreError as e:
        print(f"A general LLMCore error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# asyncio.run(robust_chat())
```

For deep debugging of provider interactions, set `log_raw_payloads = true` in the `[llmcore]` section of your configuration file and set `log_level = "DEBUG"`. This will print the exact JSON sent to and received from the LLM APIs.1



## Extending LLMCore



LLMCore is designed for extensibility. Advanced developers can add support for new LLM providers, storage backends, or embedding models by implementing a set of well-defined abstract base classes.

- `BaseProvider` (`src/llmcore/providers/base.py`): Defines the interface for all LLM providers.
- `BaseSessionStorage` (`src/llmcore/storage/base_session.py`): Defines the interface for session and episodic memory storage.
- `BaseVectorStorage` (`src/llmcore/storage/base_vector.py`): Defines the interface for semantic memory (vector) storage.
- `BaseEmbeddingModel` (`src/llmcore/embedding/base.py`): Defines the interface for embedding model integrations.

By implementing these base classes and registering the new components in the appropriate manager maps (e.g., `PROVIDER_MAP` in `providers/manager.py`), you can seamlessly integrate new technologies into the LLMCore platform.