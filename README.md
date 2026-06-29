<p align="center">
  <img src="https://raw.githubusercontent.com/araray/llmcore/main/docs/assets/llmcore_logo.png" alt="LLMCore Logo" width="461"/>
</p>
<p align="center">
  <strong>A Production-Ready Framework for LLM Applications, Autonomous Agents, and RAG Systems</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-blue.svg"/></a>
  <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"/></a>
  <a href="https://github.com/araray/llmcore"><img alt="Version" src="https://img.shields.io/badge/version-0.50.0-green.svg"/></a>
  <a href="https://github.com/araray/llmcore/actions"><img alt="CI Status" src="https://img.shields.io/badge/CI-passing-brightgreen.svg"/></a>
  <a href="https://codecov.io/gh/araray/llmcore"><img alt="Coverage" src="https://img.shields.io/badge/coverage-85%25-green.svg"/></a>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-quickstart">Quickstart</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-documentation">Documentation</a>
</p>

---

**LLMCore** is a comprehensive Python library providing a unified, asynchronous interface for building production-grade LLM applications. From simple chat interactions to sophisticated autonomous agents with sandboxed code execution, LLMCore offers a complete toolkit for modern AI development.

## ✨ Features

### Core Capabilities

| Category | Features |
|----------|----------|
| **🔌 Multi-Provider Support** | OpenAI, Anthropic, Google Gemini, Ollama, DeepSeek, Mistral, Qwen, xAI, vLLM, DeepInfra, Deepgram |
| **💬 Chat Interface** | Unified `chat()` API, streaming responses, tool/function calling, per-call usage via `chat_with_usage()` |
| **📦 Session Management** | Persistent conversations, SQLite/PostgreSQL backends, transient sessions |
| **🔍 RAG System** | ChromaDB/pgvector storage, semantic search, context injection |
| **🌐 Search Providers** | Bright Data, Serper.dev, SerpApi, Semantic Scholar |
| **🤖 Autonomous Agents** | 8-phase cognitive cycle, goal classification, iterative reasoning |
| **🔒 Sandboxed Execution** | Docker/VM isolation, security policies, output tracking |
| **👤 Human-in-the-Loop** | Risk assessment, approval workflows, audit logging |
| **📊 Observability** | Structured event logging, metrics collection, execution replay, context diagnostics |
| **🎭 Persona System** | Customizable agent personalities and communication styles |
| **📚 Model Card Library** | Comprehensive model metadata, capability validation, cost estimation |

### What's New in v0.50.0

- **Deepgram voice/audio provider**: Native SDK support for STT, TTS, Flux streaming, Voice Agent, text intelligence, token grants, and Deepgram model cards.
- **Per-call usage accounting**: `LLMCore.chat_with_usage()` returns `ChatUsage` without requiring session persistence.
- **Search subsystem maturity**: Bright Data, Serper.dev, SerpApi, and keyless Semantic Scholar providers are available through `llmcore.search`.
- **Agent and context upgrades**: Objective-aware compression, semantic citation provenance, structured tool result summaries, iteration/phase token summaries, runtime permission metadata, and loaded-tool validation.
- **Observability upgrades**: Semantic retrieval events, context diagnostics after agent runs, context failure diagnostics, and ecosystem federation telemetry.
- **Test hygiene**: The suite now treats warning cleanup as part of release readiness; optional SDK and infrastructure tests are skipped only behind explicit environment gates.

---

## 🚀 Quickstart

### Simple Chat

```python
import asyncio
from llmcore import LLMCore

async def main():
    async with await LLMCore.create() as llm:
        # Simple question
        response = await llm.chat("What is the capital of France?")
        print(response)
        
        # Streaming response
        stream = await llm.chat("Tell me a story.", stream=True)
        async for chunk in stream:
            print(chunk, end="", flush=True)

asyncio.run(main())
```

### Conversation with Session

```python
async def conversation():
    async with await LLMCore.create() as llm:
        # First message - sets context
        await llm.chat(
            "My name is Alex and I love astronomy.",
            session_id="alex_chat",
            system_message="You are a friendly science tutor."
        )
        
        # Follow-up - LLM remembers context
        response = await llm.chat(
            "What should I observe tonight?",
            session_id="alex_chat"
        )
        print(response)
```

### RAG with Document Context

```python
async def rag_example():
    async with await LLMCore.create() as llm:
        # Add documents to vector store
        await llm.add_documents_to_vector_store(
            documents=[
                {"content": "LLMCore supports multiple providers...", "metadata": {"source": "docs"}},
                {"content": "Configuration uses the confy library...", "metadata": {"source": "docs"}},
            ],
            collection_name="my_docs"
        )
        
        # Query with RAG
        response = await llm.chat(
            "How does LLMCore handle configuration?",
            enable_rag=True,
            rag_collection_name="my_docs",
            rag_retrieval_k=3
        )
        print(response)
```

### Autonomous Agent

```python
from llmcore.agents import AgentManager, AgentMode

async def agent_example():
    async with await LLMCore.create() as llm:
        # Create agent manager
        agent = AgentManager(
            provider_manager=llm._provider_manager,
            memory_manager=llm._memory_manager,
            storage_manager=llm._storage_manager
        )
        
        # Run agent with a goal
        result = await agent.run(
            goal="Research the top 3 Python web frameworks and compare them",
            mode=AgentMode.SINGLE
        )
        print(result.final_answer)
```

---

## 📦 Installation

**Requires Python 3.11 or later.**

### Basic Installation

```bash
pip install llmcore
```

### With Specific Providers

```bash
# OpenAI support
pip install llmcore[openai]

# Anthropic Claude support
pip install llmcore[anthropic]

# Google Gemini support
pip install llmcore[gemini]

# Local Ollama support
pip install llmcore[ollama]

# Deepgram voice/audio support (STT, TTS, Voice Agent)
pip install llmcore[deepgram]

# Search providers
pip install llmcore[brightdata]
pip install llmcore[serper]
pip install llmcore[serpapi]
pip install llmcore[semanticscholar]
```

### With Storage Backends

```bash
# ChromaDB for vector storage
pip install llmcore[chromadb]

# PostgreSQL with pgvector
pip install llmcore[postgres]
```

### With Sandbox Support

```bash
# Docker sandbox
pip install llmcore[sandbox-docker]

# VM/SSH sandbox
pip install llmcore[sandbox-vm]

# Both sandbox types
pip install llmcore[sandbox]
```

### Full Installation

```bash
# Everything included
pip install llmcore[all]
```

### From Source

```bash
git clone https://github.com/araray/llmcore.git
cd llmcore
pip install -e ".[dev]"
```

---

## 🏗️ Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                              LLMCore API Facade                           │
│                          (llmcore.api.LLMCore)                            │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Provider   │  │   Session    │  │   Memory     │  │  Embedding   │   │
│  │   Manager    │  │   Manager    │  │   Manager    │  │   Manager    │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
│         │                 │                 │                 │           │
│  ┌──────┴───────┐  ┌──────┴───────┐  ┌──────┴───────┐  ┌──────┴───────┐   │
│  │  Providers   │  │   Storage    │  │    RAG       │  │  Embeddings  │   │
│  │  • OpenAI    │  │  • SQLite    │  │  • ChromaDB  │  │  • Sentence  │   │
│  │  • Anthropic │  │  • Postgres  │  │  • pgvector  │  │    Transform │   │
│  │  • Gemini    │  │  • JSON      │  │              │  │  • OpenAI    │   │
│  │  • Ollama    │  │              │  │              │  │  • Google    │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                                           │
├───────────────────────────────────────────────────────────────────────────┤
│                            Agent System (Darwin Layer 2)                  │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                         Cognitive Cycle                              │ │
│  │ PERCEIVE → PLAN → THINK → VALIDATE → ACT → OBSERVE → REFLECT → UPDATE│ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │    Goal     │ │  Fast-Path  │ │   Circuit   │ │    HITL     │          │
│  │ Classifier  │ │  Executor   │ │  Breaker    │ │   Manager   │          │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘          │
│                                                                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │   Persona   │ │   Prompt    │ │  Activity   │ │ Capability  │          │
│  │   Manager   │ │   Library   │ │   System    │ │   Checker   │          │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘          │
│                                                                           │
├───────────────────────────────────────────────────────────────────────────┤
│                              Sandbox System                               │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Docker Provider  │  VM Provider  │  Registry  │  Output Tracker     │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
├───────────────────────────────────────────────────────────────────────────┤
│                           Supporting Systems                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ Model Card  │ │Observability│ │   Tracing   │ │   Logging   │          │
│  │  Registry   │ │   System    │ │   System    │ │   Config    │          │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘          │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Configuration

LLMCore uses [`confy`](https://github.com/araray/confy) for layered configuration with the following precedence (highest priority last):

1. **Package Defaults** → `llmcore/config/default_config.toml`
2. **User Config** → `~/.config/llmcore/config.toml`
3. **Custom File** → `LLMCore.create(config_file_path="...")`
4. **Environment Variables** → `LLMCORE_*` prefix
5. **Direct Overrides** → `LLMCore.create(config_overrides={...})`

### Example Configuration

```toml
# ~/.config/llmcore/config.toml

[llmcore]
default_provider = "openai"
default_embedding_model = "text-embedding-3-small"
log_level = "INFO"

[providers.openai]
# API key via: LLMCORE_PROVIDERS__OPENAI__API_KEY or OPENAI_API_KEY
default_model = "gpt-4o"
timeout = 60

[providers.anthropic]
default_model = "claude-sonnet-4-5-20250929"
timeout = 60

[providers.ollama]
# host = "http://localhost:11434"
default_model = "llama3.2:latest"

[providers.vllm]
# Self-hosted vLLM server. base_url is required (no default).
# base_url = "http://localhost:8000/v1"
default_model = "meta-llama/Llama-3.1-8B-Instruct"
timeout = 240

[storage.session]
type = "sqlite"
path = "~/.llmcore/sessions.db"

[storage.vector]
type = "chromadb"
path = "~/.llmcore/chroma_db"
default_collection = "llmcore_default"

[agents]
max_iterations = 10
default_timeout = 600

[agents.sandbox]
mode = "docker"

[agents.sandbox.docker]
enabled = true
image = "python:3.11-slim"
memory_limit = "1g"
cpu_limit = 2.0
network_enabled = false

[agents.hitl]
enabled = true
global_risk_threshold = "medium"
default_timeout_seconds = 300
```

### Environment Variables

```bash
# Provider API Keys
export LLMCORE_PROVIDERS__OPENAI__API_KEY="sk-..."
export LLMCORE_PROVIDERS__ANTHROPIC__API_KEY="sk-ant-..."
export LLMCORE_PROVIDERS__GEMINI__API_KEY="..."

# Or use standard provider env vars
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Storage
export LLMCORE_STORAGE__SESSION__TYPE="postgres"
export LLMCORE_STORAGE__SESSION__DB_URL="postgresql://user:pass@localhost/llmcore"

# Logging
export LLMCORE_LOG_LEVEL="DEBUG"
export LLMCORE_LOG_RAW_PAYLOADS="true"
```

---

## 🔌 Providers

LLMCore supports multiple LLM providers through a unified interface:

| Provider | Models | Features |
|----------|--------|----------|
| **OpenAI** | GPT-4o, GPT-4o-mini, GPT-5.2, o1, o3-mini | Streaming, Tools, Vision |
| **Anthropic** | Claude Opus 4.5, Sonnet 4.5, Haiku 4.5 | Streaming, Tools, Vision |
| **Google** | Gemini 2.5 Pro/Flash, Gemini 3 Preview | Streaming, Tools, Vision |
| **Ollama** | Llama 3.2/3.3, Gemma 3, Phi-3, Mistral | Streaming, Local |
| **DeepSeek** | DeepSeek-R1, DeepSeek-V3.2, DeepSeek-Chat | Streaming, Reasoning |
| **Mistral** | Mistral Large 3 | Streaming, Tools |
| **Qwen** | Qwen 3 Max, Qwen3-Coder-480B | Streaming, Tools |
| **xAI** | Grok-4, Grok-4-Heavy | Streaming, Tools |
| **vLLM** | Any HuggingFace model (self-hosted) | Streaming, Tools, Vision, Structured Outputs, Guided Grammars |
| **DeepInfra** | DeepSeek, Llama, Qwen, Mistral, FLUX, Whisper, Kokoro (100+ open models) | Streaming, Tools, Vision, Reasoning, TTS, STT, Image, Embeddings |
| **Deepgram** | Nova-3, Nova-2, Whisper, Flux (STT); Aura-2 (TTS); Voice Agent | Streaming STT/TTS, Flux turn-taking, Voice Agent, Text Intelligence |

### Switching Providers

```python
# Use default provider
response = await llm.chat("Hello!")

# Override per-request
response = await llm.chat(
    "Explain quantum computing",
    provider_name="anthropic",
    model_name="claude-sonnet-4-5-20250929"
)

# With provider-specific parameters
response = await llm.chat(
    "Write a poem",
    provider_name="openai",
    model_name="gpt-4o",
    temperature=0.9,
    max_tokens=500
)
```

### Voice & Audio (Deepgram)

The **Deepgram** provider adds real-time **voice/audio** — speech-to-text (STT),
text-to-speech (TTS), conversational **Flux** STT, a bidirectional **Voice
Agent** (STT → LLM → TTS over one socket), and **text intelligence**. Deepgram
is not a chat-completion provider, so its media methods are called **directly on
the provider instance** (the `LLMCore` facade has no audio methods):

```python
from llmcore.providers.deepgram_provider import DeepgramProvider

dg = DeepgramProvider({"api_key": "dg_...", "_instance_name": "deepgram"})

# Pre-recorded STT
result = await dg.transcribe_audio(open("call.wav", "rb").read(),
                                   model="nova-3", smart_format=True, diarize=True)
print(result.text)

# TTS
speech = await dg.generate_speech("Hello from Aura.", voice="aura-2-thalia-en")
open("out.mp3", "wb").write(speech.audio_data)

# Live STT (async byte source -> streamed events)
async for ev in dg.transcribe_stream(mic_frames(), model="nova-3",
                                      encoding="linear16", sample_rate=16000,
                                      interim_results=True):
    print(ev.is_final, ev.text)

# Voice Agent (auto-answers client-side tool calls; prompt is never defaulted)
async for event in dg.run_voice_agent(mic_audio(), function_handler=handle,
                                       functions=[weather_fn],
                                       prompt="You are a concise assistant."):
    ...   # CONVERSATION_TEXT / AUDIO / FUNCTION_CALL_REQUEST events

await dg.close()
```

See **[`docs/Deepgram_provider_usage.md`](docs/Deepgram_provider_usage.md)** for
the full guide (config, streaming, Flux, Voice Agent settings shape, token auth)
and the runnable scripts in [`examples/`](examples) (`deepgram_*.py`).

---

## 🧠 Agent System (Darwin Layer 2)

The agent system implements an advanced cognitive architecture for autonomous task execution.

### Cognitive Cycle

The 8-phase cognitive cycle enables sophisticated reasoning:

```
┌───────────────────────────────────────────────────────────────────┐
│                        COGNITIVE CYCLE                            │
│                                                                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│  │ PERCEIVE │ →  │   PLAN   │ →  │  THINK   │ →  │ VALIDATE │     │
│  │          │    │          │    │          │    │          │     │
│  │ Analyze  │    │ Generate │    │ Reason & │    │ Check    │     │
│  │ goal &   │    │ strategy │    │ decide   │    │ validity │     │
│  │ context  │    │          │    │ action   │    │          │     │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘     │
│       ↑                                               │           │
│       │                                               ↓           │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│  │  UPDATE  │ ←  │ REFLECT  │ ←  │ OBSERVE  │ ←  │   ACT    │     │
│  │          │    │          │    │          │    │          │     │
│  │ Update   │    │ Learn &  │    │ Analyze  │    │ Execute  │     │
│  │ state &  │    │ improve  │    │ results  │    │ action   │     │
│  │ memory   │    │          │    │          │    │          │     │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘     │
└───────────────────────────────────────────────────────────────────┘
```

**Phase Descriptions:**

| Phase | Purpose |
|-------|---------|
| **PERCEIVE** | Analyze goal, extract entities, assess complexity |
| **PLAN** | Generate execution strategy and select approach |
| **THINK** | Reason about next action using CoT/ReAct patterns |
| **VALIDATE** | Check proposed action validity and safety |
| **ACT** | Execute the chosen action (tool call, code, etc.) |
| **OBSERVE** | Analyze results and extract observations |
| **REFLECT** | Learn from outcome, identify improvements |
| **UPDATE** | Update working memory and iteration state |

### Goal Classification

Automatic goal complexity assessment for optimal routing:

```python
from llmcore.agents import GoalClassifier, classify_goal

# Classify a goal
classification = classify_goal("What's 2 + 2?")
print(classification.complexity)  # GoalComplexity.TRIVIAL
print(classification.execution_strategy)  # ExecutionStrategy.FAST_PATH

# Complex goal
classification = classify_goal(
    "Research and compare the top 5 cloud providers, "
    "analyze their pricing, and create a recommendation report"
)
print(classification.complexity)  # GoalComplexity.COMPLEX
print(classification.max_iterations)  # 15
```

**Complexity Levels:**

| Level | Max Iterations | Examples |
|-------|----------------|----------|
| `TRIVIAL` | 1 | Greetings, simple math, factual Q&A |
| `SIMPLE` | 5 | Single-step tasks, translations |
| `MODERATE` | 10 | Multi-step tasks, analysis |
| `COMPLEX` | 15 | Research, multi-source synthesis |

### Fast-Path Execution

Bypass the full cognitive cycle for trivial goals (sub-5s responses):

```python
from llmcore.agents.learning import FastPathExecutor, should_use_fast_path

# Check if fast-path is appropriate
if should_use_fast_path("Hello, how are you?"):
    executor = FastPathExecutor(config=fast_path_config)
    result = await executor.execute(goal="Hello!")
    print(result.response)  # Instant response
```

### Circuit Breaker

Automatic detection and interruption of failing agent loops:

```python
from llmcore.agents import AgentCircuitBreaker, CircuitBreakerConfig

breaker = AgentCircuitBreaker(CircuitBreakerConfig(
    max_iterations=15,
    max_same_errors=3,
    max_execution_time_seconds=300,
    max_total_cost=1.0,
    progress_stall_threshold=5
))

# Circuit breaker trips on:
# - Maximum iterations exceeded
# - Repeated identical errors
# - Timeout exceeded
# - Cost limit exceeded
# - Progress stall detected
```

### Human-in-the-Loop (HITL)

Interactive approval workflows for sensitive operations:

```python
from llmcore.agents.hitl import HITLManager, HITLConfig, ConsoleHITLCallback

# Create HITL manager
hitl = HITLManager(
    config=HITLConfig(
        enabled=True,
        global_risk_threshold="medium",
        timeout_policy="reject"
    ),
    callback=ConsoleHITLCallback()  # Interactive console prompts
)

# Check if activity needs approval
decision = await hitl.check_approval(
    activity_type="execute_shell",
    parameters={"command": "rm -rf /tmp/test"}
)

if decision.is_approved:
    # Execute the activity
    pass
else:
    print(f"Rejected: {decision.reason}")
```

**Risk Levels:**

| Level | Requires Approval | Examples |
|-------|-------------------|----------|
| `LOW` | No | Read files, calculations |
| `MEDIUM` | Configurable | Write files, API calls |
| `HIGH` | Yes | Delete files, network access |
| `CRITICAL` | Always | System commands, credentials |

### Persona System

Customize agent behavior and communication style:

```python
from llmcore.agents import PersonaManager, AgentPersona, PersonalityTrait

# Create a custom persona
persona = AgentPersona(
    name="DataAnalyst",
    description="A meticulous data analyst focused on accuracy",
    personality=[
        PersonalityTrait.ANALYTICAL,
        PersonalityTrait.METHODICAL,
        PersonalityTrait.CAUTIOUS
    ],
    communication_style="formal",
    risk_tolerance="low",
    planning_depth="thorough"
)

# Apply to agent
manager = PersonaManager()
manager.register_persona(persona)
agent_state.persona = manager.get_persona("DataAnalyst")
```

---

## 🐳 Sandbox System

Secure, isolated execution environments for agent-generated code.

### Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         Sandbox Registry                           │
│                    (Manages sandbox lifecycle)                     │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌─────────────────────────┐    ┌─────────────────────────┐        │
│  │    Docker Provider      │    │     VM Provider         │        │
│  │                         │    │                         │        │
│  │  • Container isolation  │    │  • SSH-based access     │        │
│  │  • Image management     │    │  • Full VM isolation    │        │
│  │  • Resource limits      │    │  • Network separation   │        │
│  │  • Output capture       │    │  • Persistent storage   │        │
│  └─────────────────────────┘    └─────────────────────────┘        │
│                                                                    │
│  ┌─────────────────────────┐    ┌─────────────────────────┐        │
│  │    Output Tracker       │    │   Ephemeral Manager     │        │
│  │                         │    │                         │        │
│  │  • File lineage         │    │  • Resource cleanup     │        │
│  │  • Execution logs       │    │  • Timeout handling     │        │
│  │  • Artifact collection  │    │  • State management     │        │
│  └─────────────────────────┘    └─────────────────────────┘        │
└────────────────────────────────────────────────────────────────────┘
```

### Usage

```python
from llmcore import (
    SandboxRegistry, DockerSandboxProvider, SandboxConfig, SandboxMode
)

# Create sandbox registry
registry = SandboxRegistry()

# Register Docker provider
docker_provider = DockerSandboxProvider(SandboxConfig(
    mode=SandboxMode.DOCKER,
    image="python:3.11-slim",
    memory_limit="1g",
    cpu_limit=2.0,
    timeout_seconds=300,
    network_enabled=False
))
registry.register_provider("docker", docker_provider)

# Create and use sandbox
async with registry.create_sandbox("docker") as sandbox:
    # Execute Python code
    result = await sandbox.execute_python("""
import math
print(f"Pi = {math.pi}")
    """)
    print(result.stdout)  # "Pi = 3.141592653589793"
    
    # Execute shell command
    result = await sandbox.execute_shell("ls -la")
    print(result.stdout)
    
    # Save file
    await sandbox.save_file("output.txt", "Hello, Sandbox!")
    
    # Read file
    content = await sandbox.load_file("output.txt")
```

### Container Images

Pre-built, security-hardened images organized by tier:

| Tier | Image | Description |
|------|-------|-------------|
| **Base** | `llmcore-sandbox-base:1.0.0` | Minimal Ubuntu 24.04 |
| **Specialized** | `llmcore-sandbox-python:1.0.0` | Python 3.12 development |
| | `llmcore-sandbox-nodejs:1.0.0` | Node.js 22 development |
| | `llmcore-sandbox-shell:1.0.0` | Shell scripting |
| **Task** | `llmcore-sandbox-research:1.0.0` | Research & analysis |
| | `llmcore-sandbox-websearch:1.0.0` | Web scraping |

### Access Levels

| Level | Network | Filesystem | Tools |
|-------|---------|------------|-------|
| `RESTRICTED` | Blocked | Limited | Whitelisted only |
| `FULL` | Enabled | Extended | All tools |

### Security Features

- **Non-root execution**: All containers run as `sandbox` user (UID 1000)
- **No SUID/SGID binaries**: Privilege escalation vectors removed
- **Resource limits**: Memory, CPU, and process limits enforced
- **Network isolation**: Optional network blocking
- **Output tracking**: Full lineage and audit trail
- **AppArmor/seccomp ready**: Compatible with security profiles

---

## 📚 Model Card Library

Comprehensive metadata management for LLM models.

### Usage

```python
from llmcore import get_model_card_registry, get_model_card

# Get registry singleton
registry = get_model_card_registry()

# Lookup model card
card = registry.get("openai", "gpt-4o")
print(f"Context: {card.get_context_length():,} tokens")
print(f"Vision: {card.capabilities.vision}")
print(f"Tools: {card.capabilities.tools}")

# Cost estimation
cost = card.estimate_cost(
    input_tokens=50_000,
    output_tokens=2_000,
    cached_tokens=10_000
)
print(f"Estimated cost: ${cost:.4f}")

# List models by capability
vision_models = registry.list_cards(tags=["vision"])
for model in vision_models:
    print(f"{model.provider}/{model.model_id}")

# Alias resolution
card = registry.get("anthropic", "claude-4.5-sonnet")  # Resolves alias
```

### Supported Providers

Built-in model cards for:

- **OpenAI**: GPT-4o, GPT-4o-mini, GPT-5.2, o1, o3-mini, embeddings
- **Anthropic**: Claude Opus 4.5, Sonnet 4.5, Haiku 4.5
- **Google**: Gemini 2.5 Pro/Flash, Gemini 3 Preview
- **Ollama**: Llama 3.2/3.3, Gemma 3, Phi-3, Mistral, CodeLlama
- **DeepSeek**: DeepSeek-R1, DeepSeek-V3.2
- **Mistral**: Mistral Large 3
- **Qwen**: Qwen 3 Max, Qwen3-Coder
- **xAI**: Grok-4, Grok-4-Heavy
- **DeepInfra**: DeepSeek-V3/R1, Llama 3.x, Qwen, Mistral, FLUX (image), Whisper (STT), Kokoro (TTS), embeddings
- **Deepgram**: Nova-3, Nova-2, Whisper, Flux (STT); Aura-2 (TTS); Voice Agent

### Custom Model Cards

Add custom cards in `~/.config/llmcore/model_cards/<provider>/<model>.json`:

```json
{
  "model_id": "my-custom-model",
  "display_name": "My Custom Model",
  "provider": "ollama",
  "model_type": "chat",
  "context": {
    "max_input_tokens": 32768,
    "max_output_tokens": 4096
  },
  "capabilities": {
    "streaming": true,
    "tools": false,
    "vision": false
  }
}
```

---

## 📊 Observability

Comprehensive monitoring and debugging for agent executions.

### Event Logging

```python
from llmcore.agents.observability import EventLogger, EventCategory

# Events logged to ~/.llmcore/events.jsonl
logger = EventLogger(log_path="~/.llmcore/events.jsonl")

# Event categories
# - LIFECYCLE: Agent start/stop
# - COGNITIVE: Phase execution
# - ACTIVITY: Tool executions
# - HITL: Human approvals
# - ERROR: Exceptions
# - METRIC: Performance data
# - MEMORY: Memory operations
# - SANDBOX: Container lifecycle
# - RAG: Retrieval operations
```

### Metrics Collection

```python
from llmcore.agents.observability import MetricsCollector

collector = MetricsCollector()

# Available metrics
# - Iteration counts
# - LLM call latency (p50, p90, p95, p99)
# - Token usage (input/output)
# - Estimated costs
# - Activity execution times
# - Error counts by type
```

### Execution Replay

```python
from llmcore.agents.observability import ExecutionReplay

replay = ExecutionReplay(events_path="~/.llmcore/events.jsonl")

# List executions
executions = replay.list_executions()
for exec_id, metadata in executions.items():
    print(f"{exec_id}: {metadata['goal']}")

# Replay specific execution
events = replay.get_execution_events(exec_id)
for event in events:
    print(f"[{event.timestamp}] {event.category}: {event.type}")
```

### Configuration

```toml
[agents.observability]
enabled = true

[agents.observability.events]
enabled = true
log_path = "~/.llmcore/events.jsonl"
min_severity = "info"
categories = []  # Empty = all categories

[agents.observability.events.rotation]
strategy = "size"
max_size_mb = 100
max_files = 10
compress = true

[agents.observability.metrics]
enabled = true
track_cost = true
track_tokens = true
latency_percentiles = [50, 90, 95, 99]

[agents.observability.replay]
enabled = true
cache_enabled = true
cache_max_executions = 50
```

---

## 🔍 RAG System

Retrieval-Augmented Generation for knowledge-enhanced responses.

### Adding Documents

```python
# Add documents with metadata
await llm.add_documents_to_vector_store(
    documents=[
        {
            "content": "LLMCore is a Python library...",
            "metadata": {
                "source": "documentation",
                "version": "0.50.0",
                "category": "overview"
            }
        },
        {
            "content": "Configuration uses the confy library...",
            "metadata": {
                "source": "documentation",
                "category": "configuration"
            }
        }
    ],
    collection_name="project_docs"
)
```

### Semantic Search

```python
# Direct similarity search
results = await llm.search_vector_store(
    query="How does configuration work?",
    k=5,
    collection_name="project_docs",
    metadata_filter={"category": "configuration"}
)

for doc in results:
    print(f"Score: {doc.score:.4f}")
    print(f"Content: {doc.content[:100]}...")
    print(f"Metadata: {doc.metadata}")
```

### RAG-Enhanced Chat

```python
response = await llm.chat(
    "Explain how to configure providers",
    enable_rag=True,
    rag_collection_name="project_docs",
    rag_retrieval_k=3,
    system_message="Answer based ONLY on the provided context."
)
```

### External RAG Integration

LLMCore can serve as an LLM backend for external RAG engines:

```python
# External engine (e.g., semantiscan) handles retrieval
relevant_docs = await external_rag_engine.retrieve(query)

# Construct prompt with retrieved context
context = format_documents(relevant_docs)
full_prompt = f"Context:\n{context}\n\nQuestion: {query}"

# Use LLMCore for generation only
response = await llm.chat(
    message=full_prompt,
    enable_rag=False,  # Disable internal RAG
    explicitly_staged_items=[]  # Optional additional context
)
```

---

## 📖 API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `LLMCore` | Main facade for all LLM operations |
| `AgentManager` | Manages autonomous agent execution |
| `StorageManager` | Handles session and vector storage |
| `ProviderManager` | Manages LLM provider connections |

### Data Models

| Model | Description |
|-------|-------------|
| `ChatSession` | Conversation session with messages |
| `Message` | Individual chat message (user/assistant/system) |
| `ContextDocument` | Document for RAG/context |
| `Tool` | Function/tool definition |
| `ToolCall` | Tool invocation by LLM |
| `ToolResult` | Result of tool execution |
| `ModelCard` | Model metadata and capabilities |

### Exceptions

```python
from llmcore import (
    LLMCoreError,          # Base exception
    ConfigError,           # Configuration issues
    ProviderError,         # LLM provider errors
    StorageError,          # Storage operations
    SessionStorageError,   # Session storage specific
    VectorStorageError,    # Vector storage specific
    SessionNotFoundError,  # Session lookup failure
    ContextError,          # Context management
    ContextLengthError,    # Context exceeds limits
    EmbeddingError,        # Embedding generation
    SandboxError,          # Sandbox execution
    SandboxInitializationError,
    SandboxExecutionError,
    SandboxTimeoutError,
    SandboxAccessDenied,
    SandboxResourceError,
    SandboxConnectionError,
    SandboxCleanupError,
)
```

---

## 📚 Documentation

- [Configuration reference](docs/CONFIG_REFERENCE.md)
- [Search providers usage](docs/Search_providers_usage.md)
- [Search providers rationale](docs/Search_providers_rationale.md)
- [Deepgram provider usage](docs/Deepgram_provider_usage.md)
- [`chat_with_usage` guide](docs/USAGE_chat_with_usage.md)
- [Model cards](docs/model_cards.md)
- [Agentic system guide](docs/Agentic_System_Guide.md)
- [External RAG integration](docs/External_RAG_integration_guide.md)

---

## 📁 Project Structure

```
llmcore/
├── src/llmcore/
│   ├── __init__.py           # Public API exports
│   ├── api.py                # Main LLMCore class
│   ├── models.py             # Core data models
│   ├── exceptions.py         # Exception hierarchy
│   ├── config/               # Configuration system
│   │   ├── default_config.toml
│   │   └── models.py
│   ├── providers/            # LLM provider implementations
│   │   ├── openai_provider.py
│   │   ├── anthropic_provider.py
│   │   ├── gemini_provider.py
│   │   └── ollama_provider.py
│   ├── storage/              # Storage backends
│   │   ├── sqlite_session.py
│   │   ├── postgres_session_storage.py
│   │   ├── chromadb_vector.py
│   │   └── pgvector_storage.py
│   ├── embedding/            # Embedding models
│   │   ├── sentence_transformer.py
│   │   ├── openai.py
│   │   └── google.py
│   ├── agents/               # Agent system
│   │   ├── manager.py        # AgentManager
│   │   ├── cognitive/        # 8-phase cognitive cycle
│   │   ├── hitl/             # Human-in-the-loop
│   │   ├── sandbox/          # Sandbox execution
│   │   ├── learning/         # Learning mechanisms
│   │   ├── persona/          # Persona system
│   │   ├── prompts/          # Prompt library
│   │   ├── observability/    # Monitoring & logging
│   │   └── routing/          # Model routing
│   ├── model_cards/          # Model metadata
│   │   ├── registry.py
│   │   ├── schema.py
│   │   └── default_cards/
│   └── memory/               # Memory management
├── container_images/         # Sandbox Docker images
├── examples/                 # Usage examples
├── tests/                    # Test suite
├── docs/                     # Documentation
└── pyproject.toml            # Project configuration
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/llmcore --cov-report=term-missing

# Run specific test categories
pytest -m "not slow"           # Skip slow tests
pytest -m "not integration"    # Skip integration tests
pytest -m sandbox              # Run sandbox tests only

# Run specific test file
pytest tests/agents/test_cognitive_system.py
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Write** tests for your changes
4. **Ensure** all tests pass (`pytest`)
5. **Follow** the existing code style (`ruff check .`)
6. **Commit** with conventional commits (`feat: add amazing feature`)
7. **Push** to your branch
8. **Open** a Pull Request

### Development Setup

```bash
git clone https://github.com/araray/llmcore.git
cd llmcore
pip install -e ".[dev]"
pre-commit install
```

### Code Style

- **Formatter**: Ruff
- **Type Hints**: Required for all public APIs
- **Docstrings**: Google style
- **Line Length**: 100 characters

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🔗 Related Projects

- **[llmchat](https://github.com/araray/llmchat)** - CLI interface for llmcore
- **[semantiscan](https://github.com/araray/semantiscan)** - Advanced RAG engine
- **[confy](https://github.com/araray/confy)** - Configuration management library

---

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/araray/llmcore/issues)
- **Discussions**: [GitHub Discussions](https://github.com/araray/llmcore/discussions)

---

<p align="center">
  <sub>Built with ❤️ by <a href="https://github.com/araray">Araray Velho</a></sub>
</p>
