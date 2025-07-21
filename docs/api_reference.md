# LLMCore API Reference v0.21.0





## Service Overview and Core Concepts





### Introduction



The LLMCore API provides a comprehensive, versioned RESTful interface for building sophisticated applications on the LLMCore platform. It exposes the full range of the library's capabilities, from unified Large Language Model (LLM) interaction and persistent conversation management to advanced features like Retrieval Augmented Generation (RAG), asynchronous data ingestion, and a powerful autonomous agent framework. This document serves as the canonical technical reference for client developers, detailing every endpoint, data model, and workflow required for successful integration.



### API Architecture



The LLMCore platform is designed as a multi-component system to ensure scalability, responsiveness, and robust handling of long-running operations. Understanding this architecture is key to effectively utilizing the API.1

- **API Server (FastAPI)**: This is the primary, user-facing component that handles all incoming HTTP requests. Built on FastAPI, it provides a modern, high-performance interface for all API functionalities. Its main role is to validate requests, delegate tasks to the core `LLMCore` library, and manage the lifecycle of asynchronous jobs.1
- **TaskMaster (Arq Worker)**: This is a separate, asynchronous background worker process built on the `arq` library. Its purpose is to execute long-running and computationally intensive jobs, such as ingesting large document sets or running complex, multi-step agent tasks. By offloading these operations, the API Server remains highly responsive to new requests, which is critical for system stability and a positive user experience.1
- **Redis**: Redis serves as the high-speed message broker that decouples the API Server from the TaskMaster. When an asynchronous task is submitted to a v2 endpoint (e.g., for ingestion or agent execution), the API server enqueues a job in Redis. The TaskMaster worker listens to this queue, picks up the job, and executes it independently. This architecture is fundamental to the platform's asynchronous workflows.1



### Versioning



The API employs a clear versioning strategy to distinguish between stable, core functionalities and newer, advanced capabilities. This separation is reflected in the URL path for most endpoints and signals the maturity and expected rate of change for different parts of the API.1

- **/api/v1**: This version prefix is designated for the stable, foundational features of the platform, primarily centered around direct chat interaction. Client applications can build against v1 endpoints with a high degree of confidence in their long-term stability and backward compatibility.
- **/api/v2**: This version prefix is reserved for the platform's more advanced and actively evolving feature set. This includes direct memory access, the asynchronous task management system, data ingestion pipelines, and the autonomous agent framework. Developers integrating with v2 endpoints should anticipate a faster pace of development and potential changes in future releases, and are advised to implement more flexible client-side logic.



### Base URL and Authentication



All API endpoints documented in this reference are relative to a base URL, which is determined by the host and port where the `llmcore` server is running. For the purposes of this document, the following base URL is assumed: `http://127.0.0.1:8000`.

It is critical to note that the `llmcore` API server in version 0.21.0 does not include any built-in authentication or authorization middleware. Access to all endpoints is unrestricted by default. For production deployments or any use in untrusted environments, it is imperative to secure the API using an external solution, such as a reverse proxy (e.g., Nginx, Caddy) configured with appropriate access controls, API key validation, or OAuth2/JWT authentication.



### Master Endpoint Summary



The following table provides a high-level map of the entire API surface, allowing for quick identification of available functionalities and their corresponding endpoints.

| HTTP Method | Path                             | Version | Functionality Area | Brief Description                                            |
| ----------- | -------------------------------- | ------- | ------------------ | ------------------------------------------------------------ |
| `GET`       | `/`                              | N/A     | Core Service       | Root endpoint with basic service information.                |
| `GET`       | `/health`                        | N/A     | Core Service       | Health check for monitoring and service availability.        |
| `GET`       | `/api/v1/info`                   | v1      | Core Service       | Get detailed service capabilities, versions, and features.   |
| `POST`      | `/api/v1/chat`                   | v1      | Chat Interaction   | Core endpoint for all LLM chat interactions.                 |
| `GET`       | `/api/v2/memory/semantic/search` | v2      | Memory Access      | Search the semantic memory (vector store) for documents.     |
| `POST`      | `/api/v2/submit`                 | v2      | Asynchronous Tasks | Submit a generic asynchronous task to the background worker. |
| `GET`       | `/api/v2/{task_id}`              | v2      | Asynchronous Tasks | Get the current status of an asynchronous task.              |
| `GET`       | `/api/v2/{task_id}/result`       | v2      | Asynchronous Tasks | Get the result of a completed asynchronous task.             |
| `GET`       | `/api/v2/{task_id}/stream`       | v2      | Asynchronous Tasks | Stream real-time progress updates for a task.                |
| `POST`      | `/api/v2/ingestion/submit`       | v2      | Data Ingestion     | Submit a data ingestion task (file, zip, git).               |
| `POST`      | `/api/v2/run`                    | v2      | Autonomous Agent   | Start an autonomous agent task to achieve a high-level goal. |



## Core Service Endpoints



These endpoints provide general information about the API service and are used for health monitoring and client-side capability discovery.



### GET /



The root endpoint provides a simple confirmation that the API server is running and accessible.

- **Purpose**: To serve as a basic ping or landing point for the API.

- **Response (200 OK)**: A JSON object containing a welcome message, the API's version, and the relative URL for the interactive documentation.

    JSON

    ```
    {
      "message": "llmcore API is running",
      "version": "1.0.0",
      "docs_url": "/docs"
    }
    ```



### GET /health



The health check endpoint is designed for use by automated monitoring systems, container orchestrators (e.g., Kubernetes), and load balancers to verify the operational status of the service.1

- **Purpose**: To provide a machine-readable status of the API server and its critical dependencies.

- **Response (200 OK)**: A JSON object detailing the service's health.

    - **Healthy State**:

        JSON

        ```
        {
          "status": "healthy",
          "llmcore_available": true,
          "providers": ["ollama", "openai"],
          "task_queue_available": true
        }
        ```

    - **Degraded State**: If the core `LLMCore` instance failed to initialize or the Redis connection for the task queue is down, the status will be `"degraded"` and the corresponding boolean flags will be `false`.

        JSON

        ```
        {
          "status": "degraded",
          "llmcore_available": false,
          "providers":,
          "task_queue_available": false
        }
        ```



### GET /api/v1/info



This endpoint allows a client application to dynamically discover the capabilities and configuration of the running `llmcore` instance. This is useful for clients that need to adapt their UI or functionality based on the features enabled on the server.1

- **Purpose**: To provide detailed metadata about the service's capabilities.

- **Response (200 OK)**: A JSON object containing the API and library versions, the overall service status, and a `features` object that lists available LLM providers and boolean flags for key functionalities.

    JSON

    ```
    {
      "api_version": "1.0",
      "llmcore_version": "0.21.0",
      "service_status": "healthy",
      "features": {
        "providers": [
          "ollama",
          "openai",
          "anthropic",
          "gemini"
        ],
        "chat": true,
        "streaming": true,
        "session_management": true,
        "vector_storage": true,
        "context_management": true,
        "rag": true
      }
    }
    ```



## V1 API Reference: Chat Interaction



The v1 API provides the core, stable functionality for interacting with LLMs.



### POST /api/v1/chat



This is the central endpoint for all chat completions, serving as a direct interface to the powerful `LLMCore.chat()` method. It handles both simple, stateless queries and complex, stateful conversations, with support for real-time streaming and provider-specific customizations.1



#### Request Body (`ChatRequest`)



The endpoint accepts a JSON object conforming to the `ChatRequest` schema.1

| Field             | Type    | Required | Default | Description                                                  |
| ----------------- | ------- | -------- | ------- | ------------------------------------------------------------ |
| `message`         | string  | Yes      |         | The user's input message.                                    |
| `session_id`      | string  | No       | `null`  | The ID for the conversation session. If provided, context is maintained across calls. |
| `system_message`  | string  | No       | `null`  | A message defining the LLM's behavior or persona.            |
| `provider_name`   | string  | No       | `null`  | Overrides the default LLM provider for this call.            |
| `model_name`      | string  | No       | `null`  | Overrides the default model for the chosen provider.         |
| `stream`          | boolean | No       | `false` | If `true`, returns a streaming response.                     |
| `save_session`    | boolean | No       | `true`  | If `true`, saves the conversation turn to persistent storage. |
| `provider_kwargs` | object  | No       | `{}`    | Additional key-value arguments passed directly to the provider's API (e.g., `temperature`, `max_tokens`). |



#### Responses



The response format depends on the `stream` parameter in the request.1

- **Non-Streaming Response (`stream: false`)**

    - **Status Code**: `200 OK`

    - **Content-Type**: `application/json`

    - **Body (`ChatResponse`)**: A JSON object containing the complete, final response from the LLM.

        JSON

        ```
        {
          "response": "This is the complete response from the LLM.",
          "session_id": "my_conversation_1"
        }
        ```

- **Streaming Response (`stream: true`)**

    - **Status Code**: `200 OK`
    - **Content-Type**: `text/event-stream`
    - **Body**: A stream of Server-Sent Events (SSE). Each event contains a piece of the response text as it is generated. This allows for a real-time, "typing" effect in the client application. The client must be implemented to handle this event stream format.



#### Error Handling



This endpoint can return the following HTTP status codes for errors 1:

- `400 Bad Request`: Returned for client-side errors, such as a `ProviderError` (e.g., invalid API key), a `ContextLengthError` (prompt exceeds model limits), or a `ValueError` from an unsupported `provider_kwargs` parameter.
- `422 Unprocessable Entity`: Returned if the request body fails Pydantic validation (e.g., missing `message` field, incorrect data types).
- `500 Internal Server Error`: Returned for unexpected server-side errors or unhandled `LLMCoreError` exceptions.
- `503 Service Unavailable`: Returned if the core `llmcore` instance is not available or failed to initialize.



#### Examples



**Example 1: Simple Stateless Chat**

This example sends a single, stateless query using the default provider.

Bash

```
curl -X POST "http://127.0.0.1:8000/api/v1/chat" \
-H "Content-Type: application/json" \
-d '{
  "message": "What is the capital of Brazil?"
}'
```

**Example 2: Stateful Conversation with Provider Override**

This example starts a conversation with a `session_id`, sets a system message, and overrides the default provider to use OpenAI with a specific temperature.

Bash

```
curl -X POST "http://127.0.0.1:8000/api/v1/chat" \
-H "Content-Type: application/json" \
-d '{
  "message": "My name is Alex. What is the main component of a star?",
  "session_id": "science_tutor_convo_123",
  "system_message": "You are a helpful science tutor.",
  "provider_name": "openai",
  "provider_kwargs": {
    "temperature": 0.5
  }
}'
```

**Example 3: Streaming Chat**

This example requests a streaming response. The client would need to handle the `text/event-stream` content type.

Bash

```
curl -N -X POST "http://127.0.0.1:8000/api/v1/chat" \
-H "Content-Type: application/json" \
-d '{
  "message": "Tell me a short story about a robot who discovers music.",
  "stream": true
}'
```



## V2 API Reference: Advanced Capabilities



The v2 API exposes the advanced, asynchronous, and agentic features of the `llmcore` platform. A fundamental concept for all major v2 features (`/ingestion/submit` and `/run`) is the universal, task-based workflow. These endpoints do not perform their work synchronously; they submit a job to the background `TaskMaster` service and immediately return a `task_id`. The client application is then responsible for using the `/api/v2/tasks` endpoints to monitor the job's progress and retrieve its final result. This asynchronous pattern is essential for handling long-running operations without blocking the client or the API server.



### Memory Access



This set of endpoints provides direct, read-only access to the platform's hierarchical memory system.



#### GET /api/v2/memory/semantic/search



This endpoint allows clients to perform a direct similarity search against the semantic memory tier, which is implemented by the configured vector store. This is useful for applications that need to retrieve relevant documents for display or for implementing custom RAG logic on the client side.1

- **Purpose**: To query the vector store for documents relevant to a given text query.
- **Query Parameters** 1:
    - `query` (string, **required**): The text query to search for.
    - `collection_name` (string, optional): The name of the vector store collection to search within. If omitted, the default collection from the server's configuration is used.
    - `k` (integer, optional, default: 3): The number of top results to return. Must be between 1 and 20.
- **Response (200 OK)**: A JSON array of `ContextDocument` objects, sorted by relevance. Each object contains the document's content, metadata, and a relevance score.1
- **Error Handling**:
    - `400 Bad Request`: If the `query` parameter is missing or empty.
    - `404 Not Found`: If the specified `collection_name` does not exist.
    - `500 Internal Server Error`: For failures during the embedding or search process.
    - `503 Service Unavailable`: If the `llmcore` instance is not available.

**Example: Searching for Documents**

Bash

```
curl -G "http://127.0.0.1:8000/api/v2/memory/semantic/search" \
--data-urlencode "query=How is configuration handled in LLMCore?" \
--data-urlencode "collection_name=my_llmcore_docs" \
--data-urlencode "k=2"
```



### Asynchronous Task Management



These endpoints provide the complete lifecycle management for asynchronous jobs executed by the `TaskMaster` service. They are the foundation of the v2 workflow.1



#### POST /api/v2/submit



Submits a generic task for background execution. While available for advanced use cases, the more specific `/ingestion/submit` and `/run` endpoints are generally preferred.

- **Purpose**: To enqueue an arbitrary, named task for the `TaskMaster` worker.
- **Request Body (`TaskSubmissionRequest`)**: A JSON object specifying the task details.1
    - `task_name` (string, **required**): The name of the task function to execute (e.g., `ingest_data_task`, `run_agent_task`).
    - `args` (array, optional): A list of positional arguments for the task.
    - `kwargs` (object, optional): A dictionary of keyword arguments for the task.
- **Response (202 Accepted)**: A `TaskSubmissionResponse` JSON object containing the `task_id` for the newly created job.1



#### GET /api/v2/{task_id}



Retrieves the current status of a previously submitted asynchronous task.

- **Purpose**: To poll the state of a background job.
- **Path Parameter**:
    - `task_id` (string, **required**): The unique ID of the task, obtained from a submission response.
- **Response (200 OK)**: A `TaskStatusResponse` JSON object indicating the task's current `status` (e.g., `queued`, `in_progress`, `complete`, `failed`) and whether the `result_available` is true.1
- **Error Handling**:
    - `404 Not Found`: If the `task_id` does not exist or has expired.



#### GET /api/v2/{task_id}/result



Retrieves the final result of a successfully completed task.

- **Purpose**: To fetch the output of a finished background job.
- **Path Parameter**:
    - `task_id` (string, **required**): The unique ID of the task.
- **Response (200 OK)**: A `TaskResultResponse` JSON object containing the `task_id`, final `status`, and the `result` payload from the task.1
- **Error Handling**:
    - `400 Bad Request`: If the task is not yet complete.
    - `404 Not Found`: If the `task_id` does not exist.
    - `500 Internal Server Error`: If the task failed during execution. The response detail will contain the error message.



#### GET /api/v2/{task_id}/stream



Streams real-time progress updates for a long-running task that supports progress reporting (e.g., data ingestion).

- **Purpose**: To provide a continuous feed of progress updates using Server-Sent Events (SSE).
- **Path Parameter**:
    - `task_id` (string, **required**): The unique ID of the task.
- **Response (200 OK)**: A `text/event-stream` response. The stream will contain events for status changes, heartbeats, and the final result upon completion or failure.



### Data Ingestion



This endpoint provides a dedicated interface for populating the semantic memory by submitting data ingestion jobs to the `TaskMaster`.1



#### POST /api/v2/ingestion/submit



Submits a data ingestion task for asynchronous processing. This endpoint uses `multipart/form-data` to handle file uploads.

- **Purpose**: To add documents from local files, ZIP archives, or Git repositories to the vector store.
- **Request (Form Data)**:
    - `ingest_type` (string, **required**): The type of ingestion. Must be one of `file`, `dir_zip`, or `git`.
    - `collection_name` (string, **required**): The target vector store collection for the ingested data.
    - `files` (file, optional): One or more files to upload. Required if `ingest_type` is `file`.
    - `zip_file` (file, optional): A single ZIP archive containing a directory structure. Required if `ingest_type` is `dir_zip`.
    - `repo_name` (string, optional): An identifier for the repository or directory. Required if `ingest_type` is `git`.
    - `git_url` (string, optional): The URL of the Git repository to clone. Required if `ingest_type` is `git`.
    - `git_ref` (string, optional, default: `HEAD`): The Git branch, tag, or commit to clone.
- **Response (202 Accepted)**: A `TaskSubmissionResponse` JSON object containing the `task_id` for the ingestion job.1
- **Workflow**: After receiving the `task_id`, the client must use the `/api/v2/tasks` endpoints to monitor the ingestion progress and retrieve the final status report.
- **Server-Side Configuration**: The detailed behavior of the ingestion process (e.g., file discovery, code-aware chunking strategies, embedding models used) is governed by the `[apykatu]` section in the server's configuration file.1

**Example: Ingesting Files from a Git Repository**

Bash

```
curl -X POST "http://127.0.0.1:8000/api/v2/ingestion/submit" \
-F "ingest_type=git" \
-F "collection_name=llmcore_source_code" \
-F "repo_name=llmcore" \
-F "git_url=https://github.com/araray/llmcore.git"
```



### Autonomous Agent Execution



This endpoint is the primary entry point for leveraging the platform's advanced agentic framework.1



#### POST /api/v2/run



Starts a new autonomous agent task to achieve a high-level goal. The agent will run as a background job, using the "Think -> Act -> Observe" loop to work towards the specified goal.

- **Purpose**: To initiate a long-running, autonomous agent that can reason, use tools, and solve complex problems.
- **Request Body (`AgentRunRequest`)**: A JSON object specifying the agent's mission.1
    - `goal` (string, **required**): The high-level objective for the agent to achieve.
    - `session_id` (string, optional): A session ID for context and for logging the agent's experiences (episodic memory).
    - `provider` (string, optional): Overrides the default LLM provider for the agent's reasoning steps.
    - `model` (string, optional): Overrides the default model for the chosen provider.
- **Response (202 Accepted)**: A `TaskSubmissionResponse` JSON object containing the `task_id` for the agent run.
- **Workflow**: This endpoint initiates a background task. The final answer or result from the agent must be retrieved by polling and then calling `GET /api/v2/{task_id}/result` once the task status is `complete`.

**Example: Full Agent Workflow**

**Step 1: Start the agent task.**

Bash

```
# This command returns a task_id, e.g., "abc-123"
curl -X POST "http://127.0.0.1:8000/api/v2/run" \
-H "Content-Type: application/json" \
-d '{
  "goal": "Research the current weather in Tokyo and then write a short haiku about it.",
  "session_id": "agent-weather-task-123"
}'
```

**Step 2: Poll the task status (repeat until "complete").**

Bash

```
curl "http://127.0.0.1:8000/api/v2/abc-123"
# Response might be: {"task_id":"abc-123", "status":"in_progress", "result_available":false}
```

**Step 3: Retrieve the final result.**

Bash

```
curl "http://127.0.0.1:8000/api/v2/abc-123/result"
# Expected Response:
# {
#   "task_id": "abc-123",
#   "status": "complete",
#   "result": "Sunny skies above,\nTwenty-five degrees of warmth,\nTokyo breathes summer."
# }
```



## Data Model and Schema Reference



This section provides a detailed reference for all Pydantic models used in the API request and response bodies.



### API Model Usage Summary



| Model Name               | Used As       | Endpoint(s)                                                  |
| ------------------------ | ------------- | ------------------------------------------------------------ |
| `ChatRequest`            | Request Body  | `POST /api/v1/chat`                                          |
| `ChatResponse`           | Response Body | `POST /api/v1/chat` (non-streaming)                          |
| `ContextDocument`        | Response Body | `GET /api/v2/memory/semantic/search`                         |
| `AgentRunRequest`        | Request Body  | `POST /api/v2/run`                                           |
| `TaskSubmissionRequest`  | Request Body  | `POST /api/v2/submit`                                        |
| `TaskSubmissionResponse` | Response Body | `POST /api/v2/submit`, `POST /api/v2/ingestion/submit`, `POST /api/v2/run` |
| `TaskStatusResponse`     | Response Body | `GET /api/v2/{task_id}`                                      |
| `TaskResultResponse`     | Response Body | `GET /api/v2/{task_id}/result`                               |



### Detailed Model Schemas





#### ChatRequest



*Source: `src/llmcore/api_server/models/core.py`* 1

| Field Name        | Data Type | Required | Default | Description                                                 |
| ----------------- | --------- | -------- | ------- | ----------------------------------------------------------- |
| `message`         | string    | Yes      |         | The user's input message.                                   |
| `session_id`      | string    | No       | `null`  | The ID of the conversation session.                         |
| `system_message`  | string    | No       | `null`  | A message defining the LLM's behavior.                      |
| `provider_name`   | string    | No       | `null`  | Overrides the default LLM provider.                         |
| `model_name`      | string    | No       | `null`  | Overrides the provider's default model.                     |
| `stream`          | boolean   | No       | `false` | If true, returns a streaming response.                      |
| `save_session`    | boolean   | No       | `true`  | If true, saves the conversation turn to storage.            |
| `provider_kwargs` | object    | No       | `{}`    | Additional arguments passed directly to the provider's API. |



#### ChatResponse



*Source: `src/llmcore/api_server/models/core.py`* 1

| Field Name   | Data Type | Required | Default | Description                               |
| ------------ | --------- | -------- | ------- | ----------------------------------------- |
| `response`   | string    | Yes      |         | The LLM's full response message.          |
| `session_id` | string    | No       | `null`  | The session ID used for the conversation. |



#### ContextDocument



*Source: `src/llmcore/models.py`* 1

| Field Name  | Data Type      | Required | Default               | Description                                      |
| ----------- | -------------- | -------- | --------------------- | ------------------------------------------------ |
| `id`        | string         | No       | (auto-generated UUID) | A unique identifier for the document.            |
| `content`   | string         | Yes      |                       | The textual content of the document.             |
| `embedding` | array of float | No       | `null`                | The vector embedding of the document's content.  |
| `metadata`  | object         | No       | `{}`                  | A dictionary for storing additional information. |
| `score`     | float          | No       | `null`                | A relevance score from a similarity search.      |



#### AgentRunRequest



*Source: `src/llmcore/api_server/models/agents.py`* 1

| Field Name   | Data Type | Required | Default | Description                                     |
| ------------ | --------- | -------- | ------- | ----------------------------------------------- |
| `goal`       | string    | Yes      |         | The high-level goal for the agent to achieve.   |
| `session_id` | string    | No       | `null`  | The session ID for context and episodic memory. |
| `provider`   | string    | No       | `null`  | Overrides the default LLM provider.             |
| `model`      | string    | No       | `null`  | Overrides the default model for the provider.   |



#### TaskSubmissionResponse



*Source: `src/llmcore/api_server/models/tasks.py`* 1

| Field Name | Data Type | Required | Default    | Description                          |
| ---------- | --------- | -------- | ---------- | ------------------------------------ |
| `task_id`  | string    | Yes      |            | The unique ID of the submitted task. |
| `status`   | string    | No       | `"queued"` | The initial status of the task.      |



## Error Handling Reference



The `llmcore` API uses standard HTTP status codes to indicate the success or failure of an API request. Client applications should be prepared to handle these responses gracefully.

| Status Code | Reason Phrase         | Triggering Condition(s)                                      |
| ----------- | --------------------- | ------------------------------------------------------------ |
| `400`       | Bad Request           | A `ProviderError` (e.g., invalid API key), `ContextLengthError`, or `ConfigError` from the core library. Invalid `provider_kwargs` in a chat request. An empty `goal` in an agent run request. Attempting to retrieve the result of a task that is not yet complete. |
| `404`       | Not Found             | A requested resource, such as a RAG collection name in a semantic search or a `task_id` in a task status check, does not exist. |
| `422`       | Unprocessable Entity  | The request body failed Pydantic validation. This occurs if required fields are missing, data types are incorrect, or extraneous fields are included in the JSON payload. |
| `500`       | Internal Server Error | An unhandled exception occurred on the server. A generic `LLMCoreError` was raised from the library. An asynchronous task failed during its execution. |
| `503`       | Service Unavailable   | The core `llmcore` instance failed to initialize during server startup. A required backend service, such as the Redis server for the task queue, is unavailable or cannot be reached. |