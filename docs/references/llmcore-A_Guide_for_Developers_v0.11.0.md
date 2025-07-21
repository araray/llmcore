# **Using LLMCore: A Guide for Developers**

Llmcore v0.11.0

LLMCore is a Python library engineered to offer a unified, asynchronous, and extensible interface for interaction with a diverse array of Large Language Models (LLMs). Its primary objective is to abstract the complexities inherent in individual provider APIs, thereby streamlining the integration of advanced AI-driven chat functionalities into Python applications. The library provides robust mechanisms for session management, sophisticated context handling, and Retrieval Augmented Generation (RAG), all within an asynchronous framework tailored for high-performance applications.  
The vision underpinning LLMCore is to furnish developers with a powerful yet accessible toolkit. This toolkit aims to simplify the incorporation of varied LLM capabilities, manage context with precision, and ensure a consistent, configurable operational model across different LLM providers.  
Core principles guiding LLMCore's design include modularity, ensuring components are reusable across projects, and extensibility, facilitated by abstract base classes for providers, storage, and embedding models. Configuration is managed by the confy library, offering a layered approach through default files, user-defined files, environment variables, and direct overrides. The library maintains provider agnosticism via a consistent chat() API method. Stateful interactions are supported through advanced session and context management, including RAG and precise token counting. The API is designed to be developer-friendly, intuitive, and well-documented, leveraging Python's asyncio for non-blocking operations suitable for modern asynchronous applications.  
Key features of LLMCore encompass:

* A **Unified API** for interacting with providers like OpenAI, Anthropic, Ollama, and Gemini.  
* **Asynchronous operations** built on asyncio.  
* **Streaming support** for real-time response generation.  
* **Session management** with configurable backends (JSON, SQLite included).  
* **Retrieval Augmented Generation (RAG)** for enhancing responses with external knowledge from vector stores (ChromaDB included).  
* **Flexible configuration** via confy.  
* **Context window management**, including provider-specific token counting and truncation.  
* **Provider and storage abstraction** for easy extension.  
* **Embedding model support** (Sentence Transformers included) for RAG.

This document provides an in-depth, comprehensive guide on leveraging LLMCore within your projects, covering installation, configuration, core functionalities, and advanced features.

## **Getting Started: Installation**

To integrate LLMCore into a project, Python version 3.11 or later is required.

### **Standard Installation**

The most straightforward method for installing LLMCore is via pip:

Bash

pip install llmcore

This command fetches the latest stable release from the Python Package Index (PyPI) and installs it along with its core dependencies.

### **Installation from Source**

For developers wishing to use the latest development version or contribute to the project, LLMCore can be installed directly from its source code repository:

Bash

git clone https://github.com/araray/llmcore.git  
cd llmcore  
pip install.

This sequence clones the repository, navigates into the project directory, and then installs the library in editable mode (if preferred, though pip install. performs a standard installation).

### **Installing with Optional Dependencies (Extras)**

LLMCore is designed with a modular architecture, allowing users to install support for specific LLM providers, storage backends, and embedding models as needed. This is managed through "extras" in the installation command. This approach minimizes the number of installed dependencies, keeping project environments lean.  
Available extras include:

* **Providers**:  
  * openai: For OpenAI models (e.g., GPT-4o, GPT-3.5-turbo).  
  * anthropic: For Anthropic Claude models.  
  * gemini: For Google Gemini models.  
  * ollama: For interacting with local LLMs via an Ollama server.  
* **Storage Backends**:  
  * sqlite: SQLite is part of the standard library, so this extra primarily ensures any related Python shims or type stubs are available if specified.  
  * postgres: For using PostgreSQL as a session or vector store (requires psycopg\[binary\] and pgvector).  
  * chromadb: For using ChromaDB as a vector store.  
* **Embedding Models**:  
  * sentence\_transformers: For using local Sentence Transformer models.  
* **Convenience Extras**:  
  * all: Installs all available optional dependencies.

**Example Usage of Extras:**  
To install LLMCore with support for the OpenAI and Anthropic providers, along with ChromaDB for vector storage and Sentence Transformers for embeddings:

Bash

pip install llmcore\[openai,anthropic,chromadb,sentence\_transformers\]

To install all optional features:

Bash

pip install llmcore\[all\]

The specific packages installed by these extras are detailed in the pyproject.toml file. Selecting only the necessary extras is recommended for production environments to avoid unnecessary dependencies.

## **Project Structure Overview**

Understanding the directory structure of the LLMCore project can be beneficial for developers who wish to delve into its source code, contribute, or debug. The project is organized logically to separate concerns and facilitate maintainability.1

* **src/llmcore/**: This is the heart of the library, containing all the Python source code.  
  * api.py: Defines the main LLMCore class, which serves as the primary public interface.  
  * config/: Contains configuration-related files, notably default\_config.toml, which provides the base settings for the library.  
  * context/: Manages context assembly for LLM prompts, including RAG and token limit handling.  
  * embedding/: Handles embedding model integrations.  
  * exceptions.py: Defines custom exception classes used throughout the library.  
  * models.py: Contains Pydantic models for data structures like Message, ChatSession, and ContextDocument.  
  * providers/: Implements interfaces to various LLM providers (OpenAI, Anthropic, Ollama, Gemini).  
  * sessions/: Manages the lifecycle of chat sessions.  
  * storage/: Provides abstractions and implementations for session and vector storage backends.  
  * utils/: Contains miscellaneous utility functions.  
* **examples/**: Includes a collection of runnable Python scripts (simple\_chat.py, session\_chat.py, rag\_example.py, etc.) that demonstrate how to use different features of LLMCore. These are invaluable for understanding practical application of the library.  
* **docs/**: Contains documentation files, including the library's specification (llmcore\_spec\_v1.0.md) and usage guides (USAGE.md).1  
* **pyproject.toml**: The standard Python project configuration file. It defines project metadata, dependencies (core and optional), build system settings, and tool configurations (e.g., for linters and testers).  
* **LICENSE**: Contains the MIT License under which LLMCore is distributed.1  
* **README.md**: Provides an overview of the project, its features, and basic installation and usage instructions.  
* **run\_example.sh**: A shell script to conveniently run examples from the examples/ directory.

This structure adheres to common Python packaging best practices, making the codebase approachable for developers.

## **Configuration Deep Dive (confy System)**

LLMCore employs the confy library for a robust and flexible configuration management system. This system allows for layered configuration, enabling users to define settings at various levels, from global defaults to runtime overrides, catering to diverse operational needs.

### **Configuration Sources and Precedence**

Settings are loaded from multiple sources, with later sources in the list taking precedence over earlier ones:

1. **Packaged Defaults**: The library includes a default\_config.toml file located at src/llmcore/config/default\_config.toml. This file provides sensible default values for all configurable parameters.  
2. **User Config File**: A user-specific TOML configuration file can be placed at \~/.config/llmcore/config.toml. Settings in this file override the packaged defaults. LLMCore may create this file automatically if needed by certain operations, but manual creation is recommended for customization.  
3. **Custom Config File**: A path to a custom TOML or JSON configuration file can be specified via the config\_file\_path parameter during LLMCore.create() initialization. This overrides both packaged and user defaults.  
4. **.env File**: If a .env file is present in the current working directory or its parent directories, variables defined within it are loaded into the environment. This requires the python-dotenv package. Note that existing environment variables are *not* overridden by values from a .env file.  
5. **Environment Variables**: System environment variables can override settings from all previous sources. These variables must be prefixed with LLMCORE\_ (or a custom prefix defined by the env\_prefix parameter in LLMCore.create()). Double underscores (\_\_) are used to represent dots (.) in the configuration path (e.g., LLMCORE\_PROVIDERS\_\_OPENAI\_\_API\_KEY corresponds to providers.openai.api\_key). Environment variable names are case-insensitive.  
6. **Overrides Dictionary**: The config\_overrides dictionary, passed directly to LLMCore.create(), provides the highest level of precedence. Keys in this dictionary use dot-notation (e.g., "llmcore.default\_provider": "openai").

This layered approach ensures that developers and administrators can manage configurations flexibly, from setting global defaults to making specific runtime adjustments without modifying code. The use of environment variables is particularly beneficial for securely managing sensitive data like API keys in production environments.

### **Key Configuration Areas in default\_config.toml**

The default\_config.toml file serves as a comprehensive template, outlining all configurable aspects of LLMCore. Key sections include:

* **\[llmcore\]**: Defines global library settings.  
  * default\_provider (str): Specifies the default LLM provider (e.g., "ollama", "openai"). Default: "ollama".  
  * default\_embedding\_model (str): Sets the default embedding model for RAG (e.g., "all-MiniLM-L6-v2", "openai:text-embedding-3-small"). Default: "all-MiniLM-L6-v2".  
  * log\_level (str): Configures the logging verbosity (e.g., "INFO", "DEBUG"). Default: "INFO".  
* **\[providers.\<name\>\]**: Configures individual LLM provider instances. \<name\> is the identifier for the instance (e.g., openai, my\_custom\_ollama).  
  * api\_key (str, optional): API key for the service. Best set via environment variables.  
  * default\_model (str): Default model for this provider instance.  
  * timeout (int): API call timeout in seconds.  
  * base\_url (str, optional): Custom API endpoint, useful for proxies or self-hosted models.  
  * type (str, optional): For custom provider instances, specifies the base provider type (e.g., if \[providers.my\_azure\_openai\] uses OpenAI's API, type \= "openai").  
  * fallback\_context\_length (int, optional): Assumed maximum context length if it cannot be determined.  
  * Provider-specific settings like host for Ollama or safety\_settings for Gemini can also be included.  
* **\[storage.session\]**: Configures session history storage.  
  * type (str): Storage backend type ("json", "sqlite", "postgres"). Default: "sqlite".  
  * path (str): Filesystem path for "json" (directory) or "sqlite" (database file). \~ is expanded.  
  * db\_url (str, optional): Connection URL for database backends like "postgres".  
* **\[storage.vector\]**: Configures RAG vector storage.  
  * type (str): Vector store type ("chromadb", "pgvector"). Default: "chromadb".  
  * default\_collection (str): Default collection name for RAG operations.  
  * path (str): Filesystem path for file-based vector stores like "chromadb".  
  * db\_url (str, optional): Connection URL for database-backed vector stores.  
* **\[embedding.\<name\>\]**: Configures specific embedding models. \<name\> corresponds to the embedding service or type (e.g., openai, google, ollama, sentence\_transformer).  
  * api\_key (str, optional): API key for service-based embedding models.  
  * default\_model (str): Specific model identifier for the embedding service.  
  * host (str, optional): Host for Ollama embedding models.  
  * device (str, optional): Device for local Sentence Transformer models (e.g., "cpu", "cuda").  
* **\[context\_management\]**: Defines strategies for constructing LLM prompts.  
  * rag\_retrieval\_k (int): Default number of documents to retrieve for RAG. Default: 3\.  
  * rag\_combination\_strategy (str): How RAG results are combined with history ("prepend\_system"). Default: "prepend\_system".  
  * history\_selection\_strategy (str): How history messages are selected ("last\_n\_tokens"). Default: "last\_n\_tokens".  
  * reserved\_response\_tokens (int): Tokens reserved for the LLM's response. Default: 500\.  
  * truncation\_priority (str): Order for truncating context types if over limit (e.g., "history,rag,user\_items").  
  * minimum\_history\_messages (int): Minimum history messages to retain. Default: 1\.  
  * user\_retained\_messages\_count (int): Number of recent user/assistant turns to prioritize. Default: 5\.  
  * prioritize\_user\_context\_items (bool): If true, user-added context items are prioritized over RAG results during initial context filling. Default: true.  
  * max\_chars\_per\_user\_item (int): Maximum character length for individual user-added context items before per-item truncation. Default: 40000\.

A thorough understanding of these configuration sections and their precedence allows developers to tailor LLMCore's behavior precisely to their application's requirements and deployment environment.

## **Initializing LLMCore: The LLMCore.create() Method**

The primary entry point for utilizing the LLMCore library is through its asynchronous class method, LLMCore.create(). This method is responsible for instantiating and fully initializing an LLMCore object, preparing it for interaction with LLMs.

### **The Asynchronous Nature of Initialization**

The LLMCore.create() method is designed as an async function because the initialization process can involve several potentially I/O-bound operations. These include:

* Loading and parsing configuration files.  
* Initializing provider managers, which might involve establishing initial connections or verifying API keys for external LLM services.  
* Setting up storage managers, which could entail connecting to databases (e.g., SQLite, PostgreSQL) or preparing file system paths.  
* Initializing the embedding manager, which might load embedding models into memory (a potentially time-consuming operation for large local models) or configure API clients for embedding services.

By making create() asynchronous, LLMCore ensures that these setup tasks do not block the main execution thread of an application, which is crucial for responsive and high-performance asynchronous programs. Once await LLMCore.create() completes, the developer can be confident that the returned llm instance is fully configured and all its internal components (provider, storage, embedding, and context managers) are ready for use. Any critical failures during this setup phase, such as misconfiguration or inability to connect to a required service, will typically result in exceptions like ConfigError, ProviderError, StorageError, or EmbeddingError being raised directly from the create() call, enabling early detection and resolution of issues.

### **Basic Initialization with Defaults**

The most common way to initialize LLMCore is by using its default configuration loading mechanisms. This is typically done within an async with block to ensure proper resource management, including the automatic closure of connections when the LLMCore instance is no longer needed:

Python

import asyncio  
from llmcore import LLMCore

async def main():  
    \# Use 'async with' for automatic resource cleanup (calls llm.close())  
    async with await LLMCore.create() as llm:  
        print("LLMCore initialized with default settings\!")  
        \# llm instance is now ready to use  
        \# Example: response \= await llm.chat("Hello, world\!")  
        \# print(response)  
    \# llm.close() is called automatically when exiting the 'async with' block

if \_\_name\_\_ \== "\_\_main\_\_":  
    asyncio.run(main())

In this basic form, LLMCore.create() will load configuration according to the precedence order detailed in the configuration section (packaged defaults, user config file, .env file, environment variables).

### **Parameters of LLMCore.create()**

For more fine-grained control over the initialization process, LLMCore.create() accepts several optional parameters:

* **config\_overrides: Optional\] \= None**  
  * **Purpose**: Allows passing a dictionary of configuration values that will take the highest precedence, overriding settings from all other sources.  
  * **Format**: Keys in the dictionary use dot-notation to specify the configuration path (e.g., "llmcore.default\_provider": "openai", "providers.openai.timeout": 120).  
  * **Example**:  
    Python  
    overrides \= {  
        "llmcore.default\_provider": "openai",  
        "providers.openai.default\_model": "gpt-4o",  
        "storage.session.type": "json"  
    }  
    async with await LLMCore.create(config\_overrides=overrides) as llm\_custom:  
        \# LLMCore initialized with 'openai' as default provider, etc.  
        pass

* **config\_file\_path: Optional\[str\] \= None**  
  * **Purpose**: Specifies the path to a custom TOML or JSON configuration file. Settings from this file will override the packaged defaults and the user's global configuration file (\~/.config/llmcore/config.toml).  
  * **Example**:  
    Python  
    async with await LLMCore.create(config\_file\_path="project\_specific\_config.toml") as llm\_project:  
        \# LLMCore initialized with settings from project\_specific\_config.toml  
        pass

* **env\_prefix: Optional\[str\] \= "LLMCORE"**  
  * **Purpose**: Defines the prefix for environment variables that LLMCore will consider for configuration overrides. The default prefix is "LLMCORE".  
  * **Behavior**:  
    * If set to "" (an empty string), LLMCore will consider all non-system environment variables.  
    * If set to None, loading configuration from environment variables is disabled entirely.  
  * **Environment Variable Format**: Variable names should use the specified prefix, followed by the configuration path with dots (.) replaced by double underscores (\_\_) (e.g., MYAPP\_STORAGE\_\_SESSION\_\_TYPE="json" if env\_prefix="MYAPP").  
  * **Example**:  
    Python  
    \# Assuming environment variable MYAPP\_PROVIDERS\_\_OLLAMA\_\_HOST="http://custom.ollama.host:11434" is set  
    async with await LLMCore.create(env\_prefix="MYAPP") as llm\_myapp:  
        \# LLMCore will use the custom Ollama host  
        pass

Proper initialization using LLMCore.create() is the foundational step for any application intending to use the library. Understanding these parameters allows developers to adapt LLMCore's configuration to various deployment scenarios and specific project requirements effectively.

## **Mastering Chat Operations with llm.chat()**

The cornerstone of interacting with Large Language Models through LLMCore is the llm.chat() method. This asynchronous method provides a unified interface for sending messages to various LLMs, managing conversation history, handling streaming responses, and leveraging advanced features like Retrieval Augmented Generation (RAG) and user-provided context items.

### **Comprehensive llm.chat() Parameter Guide**

The llm.chat() method is versatile, accepting a range of parameters to control its behavior. All parameters after message must be specified as keyword arguments.

| Parameter | Type | Description | Default Value | Key Considerations/Usage |
| :---- | :---- | :---- | :---- | :---- |
| message | str | The user's input message content. | (Required) | The primary text prompt for the LLM. |
| session\_id | Optional\[str\] | The ID of the conversation session. If None, the chat is stateless (no history saved or loaded unless save\_session=False is also used for a transient named session). If an ID is provided but doesn't exist, a new persistent session with that ID is created (if save\_session=True). | None | Essential for maintaining conversational context across multiple turns. |
| system\_message | Optional\[str\] | A message defining the role or behavior of the LLM (e.g., "You are a helpful assistant specialized in Python."). | None | Typically used at the start of a new session. Behavior for existing sessions depends on context strategy. |
| provider\_name | Optional\[str\] | The name of the LLM provider to use (e.g., "openai", "ollama"). Overrides the configured default provider. | None (uses default) | Allows dynamic selection of LLM services. |
| model\_name | Optional\[str\] | The specific model identifier for the chosen provider (e.g., "gpt-4o", "llama3"). Overrides the provider's default model. | None (uses provider's default) | Enables use of specific models within a provider. |
| stream | bool | If True, returns an AsyncGenerator\[str, None\] yielding text chunks as they arrive. If False, returns the complete response content as a single string. | False | Use True for real-time display of responses. |
| save\_session | bool | If True and a session\_id is provided, the user message and assistant's response are saved to persistent storage. If False and session\_id is provided, a transient (in-memory) session is used for the duration of the LLMCore instance or until deleted. Ignored if session\_id is None. | True | Controls the persistence of the conversation turn. |
| active\_context\_item\_ids | Optional\[List\[str\]\] | A list of IDs for user-added context items (text, files, pinned RAG snippets) to be included in the LLM's context for this specific chat call. | None | Allows selective inclusion of previously added custom context. |
| enable\_rag | bool | If True, enables Retrieval Augmented Generation by searching the vector store for context relevant to the message. | False | Activates RAG for the current chat call. |
| rag\_retrieval\_k | Optional\[int\] | Number of documents to retrieve for RAG. Overrides the default from context\_management.rag\_retrieval\_k if provided. | None (uses config default) | Controls how many RAG documents are fetched. |
| rag\_collection\_name | Optional\[str\] | Name of the vector store collection to use for RAG. Overrides the default from storage.vector.default\_collection if provided. | None (uses config default) | Specifies which knowledge base to query for RAG. |
| \*\*provider\_kwargs | Any | Additional keyword arguments passed directly to the selected provider's API call (e.g., temperature=0.7, max\_tokens=100, top\_p=0.9). | N/A | Provides access to provider-specific features not directly abstracted by LLMCore. Consult provider documentation for available options. max\_tokens here usually refers to the *response* length limit. |

*Data Sources for Table:*  
The combination of session\_id and save\_session offers nuanced control over conversation state. When session\_id is supplied:

* If save\_session is True (the default), the conversation turn is appended to a persistent session, retrievable later.  
* If save\_session is False, the conversation turn is managed within a transient, in-memory session identified by session\_id. This session exists for the lifetime of the LLMCore instance or until explicitly deleted, but its history is not written to permanent storage. This is useful for temporary multi-turn interactions where long-term persistence is not required, preventing clutter in the main session storage. If session\_id is None, the interaction is stateless for that specific call, meaning no history is loaded or saved for reuse by ID, though an internal temporary session object might be used for the duration of the call itself.

The \*\*provider\_kwargs parameter is a powerful feature for advanced users. Since different LLM providers expose unique parameters for fine-tuning model behavior (e.g., temperature for creativity, top\_p for nucleus sampling, max\_tokens to limit response length, or specific safety settings for providers like Gemini), LLMCore cannot abstract all of them. \*\*provider\_kwargs allows these parameters to be passed directly to the underlying provider's SDK, enabling developers to leverage the full capabilities of each LLM service. Users should refer to the official documentation of the respective LLM provider for a list of supported keyword arguments.

### **Performing Stateless (Simple) Chats**

For interactions that do not require conversational memory, a stateless chat can be performed by calling llm.chat() without providing a session\_id. Each such call is treated as an independent query.

Python

import asyncio  
from llmcore import LLMCore

async def simple\_chat\_example():  
    async with await LLMCore.create() as llm:  
        \# Using default provider (e.g., Ollama, if configured as default)  
        response \= await llm.chat("What is the capital of Brazil?")  
        print(f"LLM: {response}")

\# asyncio.run(simple\_chat\_example())

This is suitable for one-off questions or tasks where context from previous interactions is irrelevant.

### **Implementing Real-Time Streaming Responses**

LLMCore supports receiving responses from LLMs in a streaming fashion, allowing applications to display text as it's generated. This is achieved by setting the stream=True parameter in the llm.chat() call. When stream=True, the method returns an asynchronous generator (AsyncGenerator\[str, None\]) that yields chunks of text as they become available.

Python

import asyncio  
from llmcore import LLMCore

async def streaming\_chat\_example():  
    async with await LLMCore.create() as llm:  
        print("LLM (Streaming): ", end="", flush=True)  
        response\_stream \= await llm.chat(  
            "Tell me a short, imaginative story about a brave knight.",  
            stream=True,  
            session\_id="knight\_story\_session" \# Can also be used with sessions  
        )  
        async for chunk in response\_stream:  
            print(chunk, end="", flush=True)  
        print("\\n--- Stream finished \---")

\# asyncio.run(streaming\_chat\_example())

This is particularly useful for interactive applications where users expect immediate feedback. The LLMCore internal stream wrapper handles accumulating the full response and saving it to the session if save\_session=True and a session\_id are provided, after the stream has completed.

### **Selecting Specific LLM Providers and Models**

While LLMCore uses a default provider and model based on its configuration, developers can dynamically override these settings on a per-call basis using the provider\_name and model\_name parameters in llm.chat().

Python

import asyncio  
from llmcore import LLMCore, ProviderError

async def specific\_provider\_chat\_example():  
    \# Ensure LLMCORE\_PROVIDERS\_OPENAI\_API\_KEY is set in environment or config  
    async with await LLMCore.create() as llm:  
        try:  
            response \= await llm.chat(  
                "Compare and contrast Python and Rust for web development.",  
                provider\_name="openai",  \# Explicitly select the OpenAI provider  
                model\_name="gpt-4o",     \# Specify the gpt-4o model  
                temperature=0.6          \# Pass a provider-specific argument  
            )  
            print(f"OpenAI (gpt-4o): {response}")  
        except ProviderError as e:  
            print(f"Error with OpenAI provider: {e}") \# e.g., if API key is missing or model is invalid

\# asyncio.run(specific\_provider\_chat\_example())

This flexibility allows applications to route different types of queries to the most suitable LLM service or model, or to offer users a choice of underlying models.

## **Managing Conversational State: Session Management**

LLMCore provides robust session management capabilities, crucial for building applications that require stateful conversations where the LLM can recall and utilize information from previous interactions. This is achieved by associating chat turns with a unique session\_id.

### **Understanding Stateful vs. Stateless Interactions**

* **Stateless Interactions**: In a stateless interaction, each call to llm.chat() is treated as an isolated event. The LLM has no memory of prior exchanges. This is suitable for simple, one-off queries where context is not needed. This is the default behavior if no session\_id is provided to llm.chat().  
* **Stateful Interactions**: Stateful interactions maintain a history of the conversation. By providing a consistent session\_id across multiple llm.chat() calls, LLMCore loads the relevant conversation history, allowing the LLM to generate responses that are contextually aware of the preceding dialogue.

### **Using session\_id for Persistent and Transient Conversations**

The session\_id parameter in llm.chat(), in conjunction with the save\_session parameter, dictates the nature and persistence of a conversation:

1. **Persistent Sessions**:  
   * When a session\_id is provided and save\_session is True (which is the default), LLMCore creates or uses a persistent session.  
   * If the session\_id is new, a new session is created and stored using the configured session storage backend (e.g., SQLite, JSON).  
   * If the session\_id exists, its history is loaded, and the new turn (user message and LLM response) is appended and saved.  
   * These sessions survive application restarts and can be retrieved later using llm.get\_session().

Python  
\# Example: Persistent session  
session\_id \= "long\_term\_project\_discussion"  
response1 \= await llm.chat("Let's discuss project Alpha.", session\_id=session\_id)  
\#... later...  
response2 \= await llm.chat("What were the key points about project Alpha?", session\_id=session\_id)  
*(Adapted from and)*

2. **Transient Sessions**:  
   * When a session\_id is provided but save\_session is set to False, LLMCore uses a transient, in-memory session associated with that session\_id.  
   * The conversation history for this session is maintained within the current LLMCore instance's cache but is *not* written to the persistent storage backend.  
   * This is useful for multi-turn interactions that require context for a limited duration (e.g., a single user interaction flow in a web app, a REPL session) without cluttering the persistent session store.  
   * The session and its history are lost when the LLMCore instance is closed or if the transient session is explicitly deleted using llm.delete\_session().

Python  
\# Example: Transient session  
transient\_session\_id \= "temp\_query\_flow\_123"  
response1 \= await llm.chat("First part of my query.", session\_id=transient\_session\_id, save\_session=False)  
response2 \= await llm.chat("Second part, building on the first.", session\_id=transient\_session\_id, save\_session=False)  
\# History for "temp\_query\_flow\_123" is maintained in memory for these calls  
\# but won't be saved to disk.

3. **Stateless Calls (No session\_id)**:  
   * If session\_id is None, the llm.chat() call is effectively stateless. While an internal temporary ChatSession object might be used for the duration of that single API call to structure the prompt, no history is loaded from or saved to either persistent storage or the transient cache for reuse by ID.

### **Session Lifecycle: Listing, Retrieving, and Deleting**

LLMCore offers methods to manage the lifecycle of sessions:

* **await llm.list\_sessions() \-\> List\]**:  
  * This method retrieves metadata for all *persistent* sessions stored in the configured session storage backend. It does not list transient, in-memory sessions.  
  * Each dictionary in the returned list typically contains information like id, name (if set during session creation or update, though not directly supported via llm.chat()), message\_count, created\_at, and updated\_at timestamps. This allows applications to display a list of saved conversations, for example.  
* **await llm.get\_session(session\_id: str) \-\> Optional**:  
  * This method retrieves a specific ChatSession object, including its full message history and any associated user-added context items.  
  * It first checks the internal cache for an active transient session with the given session\_id. If not found, it then queries the persistent storage backend.  
  * If the session is not found in either location, it will raise a SessionNotFoundError if the storage backend determines it's definitively not present (e.g., SQLite implementation). The api.py indicates it will try load\_or\_create\_session which might create one if not found, but the USAGE.md example shows catching SessionNotFoundError. The behavior is that if get\_session is called for an ID that doesn't exist in persistent storage and isn't a known transient session, it will not create one and will indicate it's not found (typically by returning None or raising SessionNotFoundError depending on the storage layer's strictness). The api.py's get\_session calls \_session\_manager.load\_or\_create\_session which *would* create it if not found and system\_message is None. This implies get\_session effectively ensures a session object is returned if the ID is valid, creating it if it doesn't exist. However, the USAGE.md example for retrieve\_specific\_session implies it can return None or raise SessionNotFoundError if the ID doesn't exist, which is more aligned with a "get" operation. The api.py get\_session indeed calls load\_or\_create\_session, which will create if not found. For a strict "get only if exists", a different internal method or flag would be needed. The current get\_session will return a new empty session if the ID is not found.  
* **await llm.delete\_session(session\_id: str) \-\> bool**:  
  * This method removes a session. It attempts to delete the session from both the transient in-memory cache (if present) and the persistent storage backend.  
  * Returns True if the session was found in either location and successfully deleted from at least one, or False if the session was not found in persistent storage and was not in the transient cache.

The distinction between list\_sessions and get\_session is important: list\_sessions is for discovering persistently saved conversations, while get\_session can retrieve both active transient sessions and persistent ones by their ID. A session created with save\_session=False will be accessible via get\_session(transient\_id) as long as the LLMCore instance is alive and the session hasn't been deleted, but it will *not* appear in the output of list\_sessions(). This design allows for efficient management of both long-term and short-term conversational states.

## **Leveraging Retrieval Augmented Generation (RAG)**

Retrieval Augmented Generation (RAG) is a powerful technique integrated into LLMCore that significantly enhances the capabilities of LLMs by allowing them to access and utilize external knowledge during conversations. This enables models to provide more accurate, up-to-date, and contextually relevant responses, particularly for queries about specific documents or data not present in their original training sets.

### **The Power of RAG: Concept and Workflow**

The RAG workflow within LLMCore can be summarized as follows:

1. **Configuration**: The system must be configured with a vector storage backend (e.g., "chromadb") and a compatible embedding model (e.g., "all-MiniLM-L6-v2"). This is done in the LLMCore configuration file.  
2. **Adding Documents (Knowledge Ingestion)**: Text documents, which form the external knowledge base, are processed and added to a specified collection within the configured vector store. LLMCore automatically generates vector embeddings for these documents using the chosen embedding model.  
3. **Chatting with RAG Enabled**: When a user initiates a chat with the enable\_rag=True parameter, LLMCore performs several steps:  
   * It generates a vector embedding for the user's current query message.  
   * It searches the specified vector store collection for documents whose embeddings are most similar to the query embedding.  
   * The content of the top k most relevant documents (where k is configurable by rag\_retrieval\_k) is retrieved.  
   * This retrieved textual content is then strategically injected into the context that is sent to the LLM, along with the user's message and the existing conversation history. The method of injection is determined by the context\_management.rag\_combination\_strategy setting.  
4. **Informed Response**: The LLM uses this augmented context, now enriched with relevant external information, to generate its response.

This process allows the LLM to "cite" or base its answers on the provided documents, leading to more factual and detailed outputs.

### **Configuring RAG: Vector Stores and Embedding Models**

To use RAG, specific configurations are required in your LLMCore setup:

* In the \[storage.vector\] section of your configuration file (e.g., default\_config.toml), set the type (e.g., "chromadb") and any necessary parameters like path for file-based stores or db\_url for database-backed stores.  
* In the \[llmcore\] section, specify the default\_embedding\_model (e.g., "all-MiniLM-L6-v2" for a local Sentence Transformer model, or an identifier like "openai:text-embedding-3-small" for an API-based model).  
* Ensure that any necessary dependencies for your chosen vector store and embedding model are installed. For example, pip install llmcore\[chromadb,sentence\_transformers\].

Vector stores organize documents into **collections**. LLMCore's RAG functionality allows specifying a collection\_name for operations, or it defaults to the storage.vector.default\_collection defined in the configuration. This enables the management of multiple, distinct knowledge bases within a single vector store instance, which can be useful for separating information for different projects or topics.

### **Populating Your Knowledge Base**

Documents must be added to a vector store collection before they can be used by RAG. LLMCore provides asynchronous methods for this:

* **await llm.add\_document\_to\_vector\_store(content: str, \*, metadata: Optional \= None, doc\_id: Optional\[str\] \= None, collection\_name: Optional\[str\] \= None) \-\> str**  
  * Adds a single document to the vector store.  
  * content: The text of the document.  
  * metadata (optional): A dictionary for additional information about the document.  
  * doc\_id (optional): A specific ID for the document; if None, one is generated.  
  * collection\_name (optional): The target collection; uses default if None.  
  * Returns the ID of the added document.  
  * *Example from rag\_example.py:*  
    Python  
    doc\_id \= await llm.add\_document\_to\_vector\_store(  
        content="LLMCore uses confy for configuration.",  
        metadata={"source": "readme.md"},  
        collection\_name="my\_project\_docs"  
    )

* **await llm.add\_documents\_to\_vector\_store(documents: List\], \*, collection\_name: Optional\[str\] \= None) \-\> List\[str\]**  
  * Adds a batch of documents.  
  * documents: A list of dictionaries, each with "content": str, and optional "metadata": Dict and "id": str.  
  * collection\_name (optional): The target collection.  
  * Returns a list of IDs of the added documents.  
  * *Example from rag\_example.py:*  
    Python  
    docs\_to\_add \=  
    added\_ids \= await llm.add\_documents\_to\_vector\_store(  
        documents=docs\_to\_add,  
        collection\_name="my\_project\_docs"  
    )

### **Directly Searching the Vector Store (llm.search\_vector\_store())**

While RAG performs searches automatically during chat, LLMCore also allows for direct similarity searches on the vector store:

* **await llm.search\_vector\_store(query: str, \*, k: int, collection\_name: Optional\[str\] \= None, filter\_metadata: Optional\] \= None) \-\> List**  
  * Searches for documents relevant to the given query text.  
  * k: The number of top similar documents to retrieve.  
  * collection\_name (optional): The target collection.  
  * filter\_metadata (optional): A dictionary to filter results based on metadata (support depends on the vector store implementation).  
  * Returns a list of ContextDocument objects. Each ContextDocument includes id, content, embedding (optional), metadata, and score, providing rich information about the retrieved items. This structured output is valuable not just for RAG context but also for applications that might want to display source information or apply further custom filtering.  
  * *Example from rag\_example.py:*  
    Python  
    search\_results \= await llm.search\_vector\_store(  
        query="What are the new features?",  
        k=2,  
        collection\_name="my\_project\_docs",  
        filter\_metadata={"version": "1.1"}  
    )  
    for doc in search\_results:  
        print(f"ID: {doc.id}, Score: {doc.score}, Content: {doc.content\[:50\]}...")

### **Maintaining Your Vector Store: Deleting Documents**

Documents can be removed from a vector store collection using their IDs:

* **await llm.delete\_documents\_from\_vector\_store(document\_ids: List\[str\], \*, collection\_name: Optional\[str\] \= None) \-\> bool**  
  * Deletes documents specified by the document\_ids list from the given collection\_name.  
  * Returns True if the deletion operation was attempted successfully by the backend (this does not necessarily mean all specified IDs were found and deleted, but that the backend processed the request without fundamental error).

### **Engaging in RAG-Powered Conversations**

To activate RAG during a chat interaction, set the enable\_rag=True parameter in the llm.chat() call. Additional parameters like rag\_collection\_name and rag\_retrieval\_k can be used to customize the RAG behavior for that specific call, overriding configured defaults.

Python

\# Example from USAGE.md and rag\_example.py  
rag\_response \= await llm.chat(  
    message="What is LLMCore configuration based on the documents?",  
    session\_id="rag\_chat\_session",  
    enable\_rag=True,  
    rag\_collection\_name="my\_llmcore\_docs", \# Use the collection where docs were added  
    rag\_retrieval\_k=2, \# Retrieve top 2 documents  
    system\_message="Answer based \*only\* on the provided context documents." \# Important for factual RAG  
)  
print(f"LLM (RAG): {rag\_response}")

Providing a system\_message that instructs the LLM to base its answer solely on the retrieved documents is a common practice to improve the factuality of RAG responses and reduce reliance on the LLM's parametric knowledge.

## **Fine-Grained Context Control: User-Provided Context Items**

Beyond automated RAG and conversation history, LLMCore offers mechanisms for developers to explicitly inject custom pieces of information into the LLM's context for a given session. These "user-added context items" can be arbitrary text snippets, the contents of local files, or even specific documents previously retrieved by RAG that the developer wishes to "pin" for continued relevance. This feature provides precise control over the information available to the LLM, enabling highly tailored interactions.  
These items are managed as part of a ChatSession object and are persisted if the session itself is persistent. The ContextManager considers these items when constructing the final prompt for the LLM, subject to token limits and truncation strategies.

### **Adding Text Snippets: llm.add\_text\_context\_item()**

To add a plain text snippet to a session's context pool:

* **await llm.add\_text\_context\_item(session\_id: str, content: str, item\_id: Optional\[str\] \= None, metadata: Optional\] \= None, ignore\_char\_limit: bool \= False) \-\> ContextItem**  
  * session\_id: The target session.  
  * content: The text to add.  
  * item\_id (optional): A custom ID for the item; one is generated if not provided.  
  * metadata (optional): A dictionary for custom metadata.  
  * ignore\_char\_limit (optional, default False): If True, this item will bypass the per-item character truncation defined by context\_management.max\_chars\_per\_user\_item.  
  * Returns the created ContextItem object of type USER\_TEXT.

The content is tokenized, and if ignore\_char\_limit is False and the content exceeds max\_chars\_per\_user\_item (default 40,000 characters from default\_config.toml), it will be truncated. The ContextItem's is\_truncated flag will reflect this.

### **Incorporating File Contents: llm.add\_file\_context\_item()**

To add the content of a local file:

* **await llm.add\_file\_context\_item(session\_id: str, file\_path: str, item\_id: Optional\[str\] \= None, metadata: Optional\] \= None, ignore\_char\_limit: bool \= False) \-\> ContextItem**  
  * file\_path: Path to the local file. Its content is read asynchronously.  
  * Other parameters are similar to add\_text\_context\_item.  
  * LLMCore automatically adds filename and original\_path to the item's metadata.  
  * Returns a ContextItem of type USER\_FILE.  
  * The same character truncation logic applies based on ignore\_char\_limit and max\_chars\_per\_user\_item.

### **Persisting RAG Insights: llm.pin\_rag\_document\_as\_context\_item()**

If a document retrieved by RAG in the immediately preceding chat turn was particularly useful, it can be "pinned" as a persistent context item for the session:

* **await llm.pin\_rag\_document\_as\_context\_item(session\_id: str, original\_rag\_doc\_id: str, custom\_item\_id: Optional\[str\] \= None, custom\_metadata: Optional\] \= None) \-\> ContextItem**  
  * original\_rag\_doc\_id: The ID of the ContextDocument retrieved by RAG in the *last* interaction for this session\_id.  
  * This method relies on a transient cache (\_transient\_last\_interaction\_info\_cache) that stores information about the RAG documents used in the most recent chat call for a session. If another chat call occurs, or if the LLMCore instance is restarted, this cache may be cleared or become outdated for that specific RAG interaction, potentially causing the pinning operation to fail if the original\_rag\_doc\_id is no longer found in the cache.  
  * Creates a new ContextItem of type RAG\_SNIPPET, copying the content and metadata from the original RAG document. LLMCore adds metadata like pinned\_from\_rag and original\_rag\_doc\_id.

### **Modifying Existing Items: llm.update\_context\_item()**

To change the content or metadata of an already added context item:

* **await llm.update\_context\_item(session\_id: str, item\_id: str, content: Optional\[str\] \= None, metadata: Optional\] \= None) \-\> ContextItem**  
  * If content is provided, it's updated, is\_truncated is reset, and tokens are re-estimated.  
  * If metadata is provided, it's merged with existing metadata.  
  * The item's timestamp is updated.

### **Removing Items: llm.remove\_context\_item()**

To remove a context item from a session:

* **await llm.remove\_context\_item(session\_id: str, item\_id: str) \-\> bool**  
  * Returns True if the item was found and removed, False otherwise.

### **Accessing Session Context Items**

To inspect the context items associated with a session:

* **await llm.get\_session\_context\_items(session\_id: str) \-\> List\[ContextItem\]**  
  * Returns a list of all ContextItem objects for the session.  
* **await llm.get\_context\_item(session\_id: str, item\_id: str) \-\> Optional\[ContextItem\]**  
  * Retrieves a specific ContextItem by its ID, or None if not found.

### **Lifecycle and Contextual Usage**

User-added context items are stored as part of the ChatSession object. If the session is persistent (i.e., save\_session=True was used, or it's the default behavior for a session with an ID), these context items are saved along with the messages (e.g., serialized into JSON by JsonSessionStorage or SqliteSessionStorage). They remain part of the session's context for future interactions until explicitly removed.  
When llm.chat() is called, the ContextManager incorporates active user-added items. Each item's content is formatted with a descriptive header and footer. The ignore\_char\_limit parameter for an item, context\_management.max\_chars\_per\_user\_item from the configuration, and the overall context\_management.truncation\_priority setting interact to determine the final content included. An item with ignore\_char\_limit=True bypasses the initial per-item character check but can still be entirely removed if "user\_items" are subject to truncation due to overall token budget constraints and the truncation\_priority order (default: "history,rag,user\_items"). This multi-layered control allows developers to manage verbosity while ensuring critical information can be prioritized, though not absolutely guaranteed if token limits are severely restrictive.

## **Discovering Provider Capabilities**

LLMCore provides utility methods to help developers understand the capabilities of the LLM providers that have been configured and successfully loaded. This allows applications to dynamically adapt to the available services and models.

### **Listing Loaded LLM Providers (llm.get\_available\_providers())**

To obtain a list of all LLM provider instances that were successfully initialized based on the current configuration, the llm.get\_available\_providers() method can be used.

* **Method Signature**: def get\_available\_providers(self) \-\> List\[str\]:  
* **Functionality**: This method queries the internal ProviderManager and returns a list of strings, where each string is the *configured section name* of a loaded provider instance. For example, if the configuration includes \[providers.my\_openai\_instance\], then "my\_openai\_instance" would be in the returned list if it loaded successfully. If a provider defined in the configuration fails to load (e.g., due to missing API keys, incorrect settings, or unmet dependencies), it will not be included in this list.  
* **Example**:  
  Python  
  import asyncio  
  from llmcore import LLMCore

  async def show\_loaded\_providers():  
      async with await LLMCore.create() as llm:  
          available\_providers \= llm.get\_available\_providers()  
          print("Successfully Loaded Provider Instances:", available\_providers)

  \# Example Output (assuming 'ollama' and a custom 'openai\_clone' were configured and loaded):  
  \# Successfully Loaded Provider Instances: \['ollama', 'openai\_clone'\]  
  *(Adapted from)*

### **Querying Models for a Specific Provider (llm.get\_models\_for\_provider())**

Once a provider is known to be available, an application can query for the list of models it supports using llm.get\_models\_for\_provider(provider\_name: str).

* **Method Signature**: def get\_models\_for\_provider(self, provider\_name: str) \-\> List\[str\]:  
* **Parameters**:  
  * provider\_name (str): The name of the provider instance (as returned by get\_available\_providers()) for which to list models.  
* **Functionality**: This method calls the get\_available\_models() method on the specific provider instance. The way models are listed can vary:  
  * Some providers, like OpenAI or Gemini, might make an API call to their service to fetch an up-to-date list of available models.  
  * Other providers, or specific implementations within LLMCore (like for Anthropic and Ollama as per their current provider code), might return a static list of commonly known or supported models. In such cases, the list might not always reflect the absolute latest offerings from the provider, and developers should consult the provider's official documentation for definitive information.  
* **Error Handling**:  
  * ConfigError: Raised if the provided provider\_name does not correspond to a loaded provider instance.  
  * ProviderError: May be raised if the provider attempts an API call to fetch models and that call fails (e.g., due to network issues or API key problems).  
* **Example**:  
  Python  
  import asyncio  
  from llmcore import LLMCore, ConfigError, ProviderError

  async def show\_specific\_provider\_models(provider\_name\_to\_query):  
      async with await LLMCore.create() as llm:  
          if provider\_name\_to\_query not in llm.get\_available\_providers():  
              print(f"Provider '{provider\_name\_to\_query}' is not available or failed to load.")  
              return  
          try:  
              models \= llm.get\_models\_for\_provider(provider\_name\_to\_query)  
              print(f"Models available for '{provider\_name\_to\_query}': {models}")  
          except (ConfigError, ProviderError) as e:  
              print(f"Error retrieving models for '{provider\_name\_to\_query}': {e}")

  \# Example Usage:  
  \# asyncio.run(show\_specific\_provider\_models("openai"))  
  \# Example Output: Models available for 'openai': \['gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo',...\]  
  *(Adapted from)*

These methods empower developers to build applications that can dynamically discover and adapt to the LLM services available in their LLMCore environment.

## **Robust Error Handling in LLMCore**

LLMCore is designed to interact with external services and manage complex state, operations which are inherently susceptible to various issues such as network failures, API errors, configuration problems, and data limit exceptions. To address this, the library implements a comprehensive custom exception hierarchy, enabling developers to write robust applications capable of gracefully handling these scenarios.

### **Navigating LLMCore's Custom Exception Hierarchy**

All custom exceptions in LLMCore inherit from a base class, LLMCoreError. This allows for broad error catching, while more specific exceptions provide detailed information about the nature of the problem. The primary custom exceptions are detailed in the table below:

| Exception Name | Inherits From | Description | Key Attributes (if any) |
| :---- | :---- | :---- | :---- |
| LLMCoreError | Exception | Base class for all LLMCore specific errors. | message |
| ConfigError | LLMCoreError | Raised for errors related to configuration loading or validation (e.g., missing parameters, malformed files). | message |
| ProviderError | LLMCoreError | Raised for errors originating from an LLM provider (e.g., API errors, connection issues, authentication failure, model not found). | provider\_name (str), message |
| StorageError | LLMCoreError | Base class for errors related to storage operations (session or vector stores). | message |
| SessionStorageError | StorageError | Raised for errors specific to session storage operations (e.g., saving, loading, deleting chat sessions). | message |
| VectorStorageError | StorageError | Raised for errors specific to vector storage operations (e.g., adding, searching, deleting RAG documents). | message |
| SessionNotFoundError | StorageError | Raised when a specified session ID is not found in persistent storage and is not an active transient session. | session\_id (str), message |
| ContextError | LLMCoreError | Base class for errors related to context management (e.g., assembling the prompt). | message |
| ContextLengthError | ContextError | Raised when the context length (history \+ RAG \+ user message) exceeds the model's maximum token limit, even after truncation attempts. | model\_name (str), limit (int), actual (int), message |
| EmbeddingError | LLMCoreError | Raised for errors related to text embedding generation (e.g., embedding model failure, API issues with embedding service). | model\_name (str), message |

*Data Sources for Table:*

### **Best Practices for Handling Common Errors**

It is crucial to implement proper error handling when using LLMCore, particularly around LLMCore.create() and llm.chat() calls, as these are primary interaction points where issues can arise.

Python

import asyncio  
import logging  
from llmcore import (  
    LLMCore, LLMCoreError, ProviderError, ContextLengthError,  
    ConfigError, SessionNotFoundError, VectorStorageError, EmbeddingError  
)

\# Configure logging for better visibility  
logging.basicConfig(level=logging.INFO, format='%(asctime)s \- %(levelname)s \- %(message)s')  
logger \= logging.getLogger(\_\_name\_\_)

async def robust\_chat\_operation():  
    llm\_instance \= None  
    try:  
        \# Initialization can raise ConfigError, ProviderError, etc.  
        async with await LLMCore.create() as llm\_instance:  
            logger.info("LLMCore initialized successfully.")  
              
            \# Example chat call  
            session\_id \= "my\_robust\_session"  
            user\_message \= "Tell me about quantum computing."  
              
            response \= await llm\_instance.chat(  
                message=user\_message,  
                session\_id=session\_id,  
                provider\_name="ollama" \# Example provider  
            )  
            logger.info(f"LLM Response: {response}")

    except ContextLengthError as e:  
        logger.error(f"Context too long for model '{e.model\_name}'. Limit: {e.limit}, Actual: {e.actual}. Details: {e.message}")  
        \# Application logic: e.g., inform user, suggest shortening input, or try a different model.  
    except ProviderError as e:  
        logger.error(f"Error with provider '{e.provider\_name}': {e.message}")  
        \# Application logic: e.g., try a fallback provider, inform user of service unavailability.  
    except SessionNotFoundError as e:  
        logger.error(f"Session '{e.session\_id}' not found. Details: {e.message}")  
        \# Application logic: e.g., start a new session or prompt user.  
    except VectorStorageError as e:  
        logger.error(f"Vector storage error during RAG operation: {e.message}")  
        \# Application logic: e.g., disable RAG for this query, inform user.  
    except EmbeddingError as e:  
        logger.error(f"Embedding error for model '{e.model\_name}': {e.message}")  
        \# Application logic: e.g., RAG might be unavailable, inform user.  
    except ConfigError as e:  
        logger.critical(f"LLMCore configuration error: {e.message}. Please check your setup.")  
        \# Application logic: Critical error, likely requires configuration fix.  
    except LLMCoreError as e: \# Catch any other LLMCore-specific errors  
        logger.error(f"An LLMCore operation failed: {e}")  
    except Exception as e: \# Catch any other unexpected errors  
        logger.exception(f"An unexpected system error occurred: {e}")  
    \# 'async with' ensures llm\_instance.close() is called if llm\_instance was successfully created.

\# asyncio.run(robust\_chat\_operation())

*(Adapted from and error handling patterns in)*

### **Tips for Logging and Debugging**

LLMCore utilizes Python's standard logging module. To aid in debugging and understanding the library's behavior:

* **Adjust Log Level**: The log\_level can be set in the \[llmcore\] section of the configuration file (e.g., log\_level \= "DEBUG"). Setting it to "DEBUG" provides highly verbose output, detailing internal operations, which can be invaluable for troubleshooting.  
* **Inspect Exception Attributes**: When an LLMCore custom exception is caught, its attributes (e.g., e.provider\_name, e.model\_name, e.limit, e.actual) often contain specific details that pinpoint the source or nature of the error.  
* **Review Logs**: LLMCore logs informational messages, warnings, and errors during its operations. These logs, especially at the DEBUG level, can provide a trace of actions leading up to an error.

By employing these error handling strategies and leveraging LLMCore's logging capabilities, developers can build more resilient and maintainable applications.

## **Resource Management: Closing Connections**

Proper resource management is essential for the stability and efficiency of applications using LLMCore, as the library interacts with network services, databases, and potentially large in-memory models. LLMCore provides mechanisms to ensure that these resources are released gracefully when no longer needed.

### **The Importance of llm.close()**

The LLMCore class features an asynchronous method, await llm.close(), which is designed to explicitly release all managed resources. Calling this method orchestrates the shutdown of various internal components:

* **Provider Clients**: Closes connections for all loaded LLM provider instances (e.g., underlying httpx clients used by OpenAI, Anthropic, or Gemini SDKs; Ollama client connections).  
* **Storage Backends**: Terminates connections to session storage (e.g., SQLite, PostgreSQL connection pools) and vector storage backends (e.g., ChromaDB client).  
* **Embedding Models**: Releases resources held by embedding models, particularly relevant for API-based embedding services or if local models have specific cleanup routines.  
* **Internal Caches**: Clears transient caches, such as \_transient\_last\_interaction\_info\_cache and \_transient\_sessions\_cache.

This cleanup process helps prevent resource leaks, such as open network connections or database locks, which can degrade application performance or lead to instability over time.

### **Using async with await LLMCore.create() as llm: for Automatic Cleanup**

The recommended and most Pythonic way to manage the lifecycle of an LLMCore instance is by using an async with statement. The LLMCore class implements the asynchronous context manager protocol (\_\_aenter\_\_ and \_\_aexit\_\_).

* When entering the async with block, await LLMCore.create() is called, and its result (the llm instance) is made available.  
* Upon exiting the async with block, whether normally or due to an exception, the \_\_aexit\_\_ method is automatically invoked, which in turn calls await self.close().

**Example:**

Python

import asyncio  
from llmcore import LLMCore

async def main\_with\_context\_manager():  
    try:  
        async with await LLMCore.create() as llm:  
            print("LLMCore instance created and active.")  
            response \= await llm.chat("Briefly, what is LLMCore?")  
            print(f"Response: {response}")  
        \# At this point, llm.close() has been automatically called.  
        print("LLMCore instance resources have been released.")  
    except Exception as e:  
        print(f"An error occurred: {e}")

\# asyncio.run(main\_with\_context\_manager())

This pattern ensures that llm.close() is reliably executed, simplifying resource management for the developer and making the code cleaner and less prone to errors related to unreleased resources. For most use cases, the async with statement is the preferred method for working with LLMCore instances.

## **Advanced Topics and Extensibility**

Beyond its core chat and RAG functionalities, LLMCore incorporates sophisticated mechanisms for context management and is designed with extensibility in mind.

### **Understanding Token Counting and Context Window Management**

LLMs operate with a finite context window, measured in tokens. LLMCore's ContextManager is responsible for assembling the prompt sent to the LLM, ensuring it fits within the target model's specific token limit while maximizing the inclusion of relevant information.

* **Provider-Specific Token Counting**: Different LLMs and providers use different tokenization schemes. LLMCore employs provider-specific methods for accurate token counting:  
  * For OpenAI models, it typically uses the tiktoken library.  
  * The Ollama provider also defaults to tiktoken but allows configuration for other methods like character division for models where tiktoken might not be perfectly aligned.  
  * Anthropic and Gemini providers leverage their respective SDKs for token counting (client.count\_tokens() or model.count\_tokens()).  
* **Context Assembly and Truncation Strategies**: The ContextManager uses several configured strategies from the \[context\_management\] section of default\_config.toml to build and, if necessary, truncate the context:  
  * reserved\_response\_tokens: A budget of tokens is set aside for the LLM's response, ensuring it has space to generate an answer.  
  * max\_chars\_per\_user\_item and ignore\_char\_limit: User-added context items are first checked against this character limit. If an item exceeds it and ignore\_char\_limit is False, the item's content is truncated.  
  * history\_selection\_strategy (e.g., last\_n\_tokens) and user\_retained\_messages\_count: These guide how much of the conversation history is included, prioritizing recent turns.  
  * prioritize\_user\_context\_items: If True, user-added items are generally added to the context before RAG results, if budget allows.  
  * truncation\_priority (e.g., "history,rag,user\_items"): If the combined context (history, RAG results, user items, system message, current user message) still exceeds the model's limit, items are removed based on this priority order. For instance, with the default order, older history messages are removed first, then less relevant RAG documents, and finally, user-added context items.  
  * minimum\_history\_messages: Ensures a minimum number of conversational turns are kept, if possible, during truncation to maintain some conversational flow.

This sophisticated orchestration by the ContextManager aims to automate the complex task of fitting the most relevant information into the LLM's limited context window. Advanced users can fine-tune these \[context\_management\] settings to optimize prompt construction for their specific needs.

### **Architectural Note: Extending LLMCore**

LLMCore is architected for extensibility, allowing developers to add support for new LLM providers, storage backends, or embedding models by implementing well-defined base classes. These include:

* BaseProvider ( Section 4.2,)  
* BaseSessionStorage ( Section 4.4,)  
* BaseVectorStorage ( Section 4.4,)  
* BaseEmbeddingModel ( Section 4.6,)

Developers interested in contributing new integrations or understanding the internal architecture in more detail are encouraged to consult the llmcore\_spec\_v1.0.md document found in the docs/ directory of the project repository.

## **Practical Examples and Use Cases**

The LLMCore library includes a suite of example scripts located in the examples/ directory. These scripts serve as practical demonstrations of the library's various functionalities and provide a starting point for developers integrating LLMCore into their own applications. If you have cloned the LLMCore repository, the run\_example.sh script provides a convenient way to execute these examples.

### **Walkthrough of Scripts in the examples/ Directory**

* **simple\_chat.py**: Illustrates basic, stateless chat interactions. It shows how to initialize LLMCore, send a message using the default provider, and another message using a specifically chosen provider and model (e.g., OpenAI's gpt-4o).  
* **session\_chat.py**: Demonstrates stateful conversations using persistent sessions. It covers creating/using a session ID, setting a system message, sending multiple messages within the same session to maintain context, and verifying session saving.  
* **streaming\_chat.py**: Shows how to handle streaming responses. It calls llm.chat() with stream=True and asynchronously iterates through the response chunks, printing them in real-time. It also demonstrates saving the full conversation to a session after the stream completes.  
* **rag\_example.py**: Provides a comprehensive demonstration of Retrieval Augmented Generation. It covers adding documents to a vector store, performing an optional direct similarity search, and then chatting with enable\_rag=True to allow the LLM to use the added documents as context. It also shows how to specify a RAG collection and the number of documents to retrieve (rag\_retrieval\_k).  
* **gemini\_example.py**: Focuses on using the Google Gemini provider. It includes examples of simple chat, streaming, session-based chat, and RAG, all tailored for Gemini models. It also highlights error handling specific to this provider.  
* **ollama\_example.py**: Demonstrates interaction with a local Ollama instance. Similar to the Gemini example, it covers simple chat, streaming, sessions, and RAG, along with error handling relevant to Ollama (e.g., server not running, model not pulled).

These examples are invaluable for understanding the practical application of LLMCore's API and features.

### **Conceptual Application Ideas**

LLMCore's flexibility lends itself to a variety of applications:

* **Building a Basic RAG-Powered Q\&A Application**:  
  1. Configure LLMCore with a vector store (e.g., ChromaDB) and an embedding model (e.g., Sentence Transformers).  
  2. Ingest your knowledge base (e.g., product documentation, FAQs, research papers) into a vector store collection using llm.add\_documents\_to\_vector\_store().  
  3. Create a user interface (CLI, web app) that accepts user questions.  
  4. For each question, call await llm.chat(question, enable\_rag=True, rag\_collection\_name="your\_knowledge\_base\_collection").  
  5. The LLM will provide answers based on the retrieved documents.  
* **Integration into an Asynchronous Web Service (e.g., FastAPI, Quart)**:  
  1. Instantiate LLMCore once during the web application's startup event: app.state.llm \= await LLMCore.create().  
  2. Ensure app.state.llm.close() is called during the application's shutdown event to release resources.  
  3. In your asynchronous API route handlers, access the LLMCore instance (e.g., request.app.state.llm) to perform chat operations, manage sessions, or interact with RAG features.  
  4. Streaming responses can be effectively handled using, for example, FastAPI's StreamingResponse.

These conceptual examples illustrate how LLMCore can serve as a foundational library for building more complex AI-powered applications.

## **Conclusion and Further Resources**

LLMCore offers a powerful and developer-friendly Python library for integrating diverse Large Language Model capabilities into applications. Its unified API abstracts provider-specific complexities, while its asynchronous design ensures suitability for modern, high-performance systems. Key strengths include robust session management for stateful conversations, integrated Retrieval Augmented Generation (RAG) for knowledge-enhanced responses, a highly flexible configuration system via confy, and an extensible architecture that invites community contributions.  
The library's comprehensive feature set, including detailed context window management, support for user-provided context items, and clear error handling mechanisms, empowers developers to build sophisticated AI-driven applications with greater ease and control.  
For further information, detailed API references, and community support, the following resources are available:

* **GitHub Repository**: The primary source code, issue tracker, and contribution point for LLMCore can be found at https://github.com/araray/llmcore.  
* **Project Documentation**:  
  * The README.md file in the repository provides a high-level overview and quickstart.  
  * The docs/USAGE.md file offers detailed usage instructions and examples (though this guide aims to be more comprehensive).  
  * For architectural insights and guidelines on extending LLMCore with new providers or storage backends, refer to docs/llmcore\_spec\_v1.0.md.  
* **PyPI Page**: Once LLMCore is officially published, its PyPI page will provide installation details and release history.  
* **Community and Support**: For questions, discussions, bug reports, and feature requests, please use the GitHub Issues section of the repository.

By leveraging these resources and the detailed guidance provided in this document, developers can effectively incorporate LLMCore into their projects, unlocking the potential of various Large Language Models through a consistent and powerful interface.