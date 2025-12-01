# LLMCore Agentic System: Integration Guide

## Table of Contents

1. [Introduction](#1-introduction)
2. [Architecture Overview](#2-architecture-overview)
3. [Installation & Dependencies](#3-installation--dependencies)
4. [Configuration](#4-configuration)
5. [Basic Usage Patterns](#5-basic-usage-patterns)
6. [The Cognitive Cycle](#6-the-cognitive-cycle)
7. [Sandbox System](#7-sandbox-system)
8. [Tool Development](#8-tool-development)
9. [Memory & Context Integration](#9-memory--context-integration)
10. [Advanced Patterns](#10-advanced-patterns)
11. [Security Model](#11-security-model)
12. [Error Handling](#12-error-handling)
13. [Monitoring & Observability](#13-monitoring--observability)
14. [Best Practices](#14-best-practices)
15. [Troubleshooting](#15-troubleshooting)
16. [API Reference](#16-api-reference)

---

## 1. Introduction

### 1.1 What is the LLMCore Agentic System?

The LLMCore Agentic System is a framework for building autonomous AI agents that can:

- **Think**: Reason about tasks and decide on actions
- **Act**: Execute tools to interact with the external world
- **Observe**: Process results and update understanding
- **Reflect**: Learn from experiences and adjust strategies

Unlike simple chat applications, agents operate in a loop, making decisions and taking actions until they achieve a goal or determine they cannot proceed.

### 1.2 Key Capabilities

| Capability | Description |
|------------|-------------|
| **Autonomous Execution** | Agents run cognitive loops without human intervention |
| **Tool Integration** | Extensible tool system for external interactions |
| **Sandboxed Execution** | Code execution in isolated Docker/VM environments |
| **Memory Integration** | Semantic and episodic memory for context |
| **Human-in-the-Loop** | Optional approval gates for sensitive actions |
| **Multi-Provider** | Works with OpenAI, Anthropic, Ollama, Gemini |

### 1.3 When to Use Agents vs. Simple Chat

**Use Simple Chat (`LLMCore.chat()`) when:**
- Single question/answer interactions
- No external tool execution needed
- Human controls the conversation flow

**Use Agents (`AgentManager.run_agent_loop()`) when:**
- Complex multi-step tasks
- Autonomous code generation and execution
- Research and information gathering
- File manipulation and analysis
- Tasks requiring iteration and refinement

---

## 2. Architecture Overview

### 2.1 Component Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                     Your Application                            │
├─────────────────────────────────────────────────────────────────┤
│                         LLMCore                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   LLMCore    │  │   Storage    │  │      Memory          │  │
│  │   (chat)     │  │   Manager    │  │      Manager         │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      Agent System                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    AgentManager                           │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐  │  │
│  │  │  Cognitive │  │    Tool    │  │     Sandbox        │  │  │
│  │  │   Cycle    │  │   Manager  │  │   Integration      │  │  │
│  │  └────────────┘  └────────────┘  └────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                     Sandbox Layer                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Sandbox    │  │   Docker     │  │       VM             │  │
│  │   Registry   │  │   Provider   │  │     Provider         │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
User Goal
    │
    ▼
┌─────────────────┐
│  AgentManager   │
│  .run_agent_    │
│   loop()        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│   Plan Step     │────▶│  Create high-   │
│                 │     │  level strategy │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│   Think Step    │────▶│  Decide next    │
│                 │     │  action/tool    │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│    Act Step     │────▶│  Execute tool   │
│                 │     │  in sandbox     │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  Observe Step   │────▶│  Process tool   │
│                 │     │  result         │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  Reflect Step   │────▶│  Update state,  │
│                 │     │  log episode    │
└────────┬────────┘     └─────────────────┘
         │
         │ Loop until finished or max iterations
         ▼
    Final Result
```

### 2.3 Key Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `AgentManager` | `llmcore.agents.manager` | Orchestrates the cognitive loop |
| `ToolManager` | `llmcore.agents.tools` | Registers and executes tools |
| `SandboxIntegration` | `llmcore.agents.sandbox_integration` | Bridges agents to sandbox |
| `SandboxRegistry` | `llmcore.agents.sandbox.registry` | Manages sandbox lifecycle |
| `DockerSandboxProvider` | `llmcore.agents.sandbox.docker_provider` | Docker container execution |
| `VMSandboxProvider` | `llmcore.agents.sandbox.vm_provider` | VM/SSH execution |

---

## 3. Installation & Dependencies

### 3.1 Base Installation

```bash
# Install llmcore with your preferred LLM provider
pip install llmcore[openai]      # For OpenAI
pip install llmcore[anthropic]   # For Anthropic Claude
pip install llmcore[ollama]      # For local Ollama
pip install llmcore[all]         # All providers
```

### 3.2 Sandbox Dependencies

```bash
# Docker sandbox support (recommended)
pip install llmcore[sandbox-docker]

# VM/SSH sandbox support
pip install llmcore[sandbox-vm]

# Both Docker and VM support
pip install llmcore[sandbox]

# Full installation with all features
pip install llmcore[all,sandbox]
```

### 3.3 System Requirements

**For Docker Sandbox:**
- Docker Engine 20.10+ installed and running
- User must have permission to run Docker commands
- Sufficient disk space for container images

**For VM Sandbox:**
- SSH access to a dedicated VM
- Python 3.11+ installed on the VM
- SSH key or agent authentication configured

### 3.4 Verifying Installation

```python
import asyncio
from llmcore import LLMCore, AgentManager

async def verify():
    # Check core installation
    llm = await LLMCore.create()
    print(f"LLMCore version: {llm.__version__}")

    # Check sandbox availability
    try:
        from llmcore.agents.sandbox import DockerSandboxProvider
        print("Docker sandbox: Available")
    except ImportError:
        print("Docker sandbox: Not installed (pip install llmcore[sandbox-docker])")

    try:
        from llmcore.agents.sandbox import VMSandboxProvider
        print("VM sandbox: Available")
    except ImportError:
        print("VM sandbox: Not installed (pip install llmcore[sandbox-vm])")

    await llm.close()

asyncio.run(verify())
```

---

## 4. Configuration

### 4.1 Configuration File Location

LLMCore uses the `confy` library for configuration with the following precedence (highest priority last):

1. Package defaults (`llmcore/config/default_config.toml`)
2. User config (`~/.config/llmcore/config.toml`)
3. Custom config file (specified programmatically)
4. Environment variables (`LLMCORE_*`)
5. Runtime overrides (code)

### 4.2 Agent Configuration

Add to your `~/.config/llmcore/config.toml`:

```toml
# =============================================================================
# Agent System Configuration
# =============================================================================
[agents]
# Maximum cognitive loop iterations before stopping
max_iterations = 10

# Default timeout for agent tasks (seconds)
default_timeout = 600

# =============================================================================
# Sandbox Configuration
# =============================================================================
[agents.sandbox]
# Execution mode: "docker", "vm", or "hybrid"
mode = "docker"

# Enable fallback to VM if Docker fails (hybrid mode only)
fallback_enabled = true
```

### 4.3 Docker Sandbox Configuration

```toml
[agents.sandbox.docker]
# Enable Docker sandbox
enabled = true

# Default container image
image = "python:3.11-slim"

# Allowed images (glob patterns)
image_whitelist = [
    "python:3.*-slim",
    "python:3.*-bookworm",
    "llmcore-sandbox:*",
    "my-company/*"
]

# Resource limits
memory_limit = "1g"
cpu_limit = 2.0
timeout_seconds = 600

# Network access (WARNING: reduces isolation)
network_enabled = false

# Labels/patterns that grant FULL access level
full_access_label = "llmcore.sandbox.full_access=true"
full_access_name_patterns = ["llmcore-trusted-*"]
```

### 4.4 VM Sandbox Configuration

```toml
[agents.sandbox.vm]
# Enable VM sandbox
enabled = true

# SSH connection details
host = "192.168.1.100"
port = 22
username = "agent"

# Authentication (use SSH agent or key file)
use_ssh_agent = true
# OR
private_key_path = "~/.ssh/llmcore_agent_key"

# Working directory on VM
working_directory = "/tmp/llmcore_sandbox"

# Timeout for operations
timeout_seconds = 600

# Hosts that grant FULL access level
full_access_hosts = ["trusted-vm.internal"]
```

### 4.5 Volume Mount Configuration

```toml
[agents.sandbox.volumes]
# Persistent shared data directory
share_path = "~/.llmcore/agent_share"

# Output files directory (persists after sandbox cleanup)
outputs_path = "~/.llmcore/agent_outputs"
```

### 4.6 Tool Access Control

```toml
[agents.sandbox.tools]
# Tools allowed for RESTRICTED access level
allowed = [
    "execute_shell",
    "execute_python",
    "save_file",
    "load_file",
    "replace_in_file",
    "append_to_file",
    "list_files",
    "file_exists",
    "delete_file",
    "create_directory",
    "get_state",
    "set_state",
    "list_state",
    "get_sandbox_info",
    "get_recorded_files",
    "semantic_search",
    "episodic_search",
    "calculator",
    "finish",
    "human_approval"
]

# Tools denied for ALL access levels (even FULL)
denied = [
    # "dangerous_tool_name"
]
```

### 4.7 Output Tracking Configuration

```toml
[agents.sandbox.output_tracking]
# Enable persistent output tracking
enabled = true

# Maximum log entries per run
max_log_entries = 1000

# Retention settings
max_run_age_days = 30
max_runs = 100
```

### 4.8 Environment Variables

All configuration can be overridden via environment variables:

```bash
# Sandbox mode
export LLMCORE_AGENTS__SANDBOX__MODE="docker"

# Docker image
export LLMCORE_AGENTS__SANDBOX__DOCKER__IMAGE="python:3.11-slim"

# Memory limit
export LLMCORE_AGENTS__SANDBOX__DOCKER__MEMORY_LIMIT="2g"

# VM host
export LLMCORE_AGENTS__SANDBOX__VM__HOST="192.168.1.100"

# API keys (for LLM providers)
export LLMCORE_PROVIDERS__OPENAI__API_KEY="sk-..."
export LLMCORE_PROVIDERS__ANTHROPIC__API_KEY="sk-ant-..."
```

---

## 5. Basic Usage Patterns

### 5.1 Minimal Agent Example

```python
import asyncio
from llmcore import LLMCore, AgentManager
from llmcore.models import AgentTask

async def run_simple_agent():
    """Run a simple agent task."""

    # 1. Create LLMCore instance
    llm = await LLMCore.create()

    try:
        # 2. Create AgentManager with required dependencies
        agent_manager = AgentManager(
            provider_manager=llm._provider_manager,
            memory_manager=llm._memory_manager,
            storage_manager=llm._storage_manager
        )

        # 3. Define the task
        task = AgentTask(
            goal="Calculate the factorial of 10 and explain the result",
            constraints=["Use Python for calculations", "Explain step by step"]
        )

        # 4. Run the agent loop (without sandbox - uses host execution)
        result = await agent_manager.run_agent_loop(
            task=task,
            max_iterations=10
        )

        print(f"Result: {result}")

    finally:
        await llm.close()

if __name__ == "__main__":
    asyncio.run(run_simple_agent())
```

### 5.2 Agent with Sandbox (Recommended)

```python
import asyncio
from llmcore import LLMCore, AgentManager
from llmcore.models import AgentTask

async def run_sandboxed_agent():
    """Run an agent with sandbox isolation."""

    llm = await LLMCore.create()

    try:
        agent_manager = AgentManager(
            provider_manager=llm._provider_manager,
            memory_manager=llm._memory_manager,
            storage_manager=llm._storage_manager
        )

        # Initialize sandbox support
        await agent_manager.initialize_sandbox({
            "mode": "docker",
            "docker": {
                "image": "python:3.11-slim",
                "memory_limit": "512m",
                "timeout_seconds": 300
            }
        })

        task = AgentTask(
            goal="Write a Python script that finds all prime numbers up to 100",
            constraints=[
                "Save the script to primes.py",
                "Execute the script and show the output"
            ]
        )

        # Run with sandbox (code executes in Docker, not on host)
        result = await agent_manager.run_agent_loop(
            task=task,
            use_sandbox=True,  # Explicit sandbox usage
            max_iterations=15
        )

        print(f"Result: {result}")

    finally:
        await agent_manager.cleanup()
        await llm.close()

if __name__ == "__main__":
    asyncio.run(run_sandboxed_agent())
```

### 5.3 Using Configuration from File

```python
import asyncio
from llmcore import LLMCore, AgentManager
from llmcore.models import AgentTask

async def run_configured_agent():
    """Run agent using configuration from file."""

    # LLMCore loads config from ~/.config/llmcore/config.toml
    llm = await LLMCore.create()

    try:
        agent_manager = AgentManager(
            provider_manager=llm._provider_manager,
            memory_manager=llm._memory_manager,
            storage_manager=llm._storage_manager
        )

        # Initialize sandbox from llmcore config
        # Reads [agents.sandbox] section automatically
        await agent_manager.initialize_sandbox()

        task = AgentTask(goal="Analyze the current directory structure")

        result = await agent_manager.run_agent_loop(task=task)
        print(result)

    finally:
        await agent_manager.cleanup()
        await llm.close()

asyncio.run(run_configured_agent())
```

### 5.4 Specifying Provider and Model

```python
async def run_with_specific_provider():
    """Run agent with a specific LLM provider and model."""

    llm = await LLMCore.create()

    try:
        agent_manager = AgentManager(
            provider_manager=llm._provider_manager,
            memory_manager=llm._memory_manager,
            storage_manager=llm._storage_manager
        )

        await agent_manager.initialize_sandbox()

        task = AgentTask(goal="Write a haiku about programming")

        result = await agent_manager.run_agent_loop(
            task=task,
            provider_name="anthropic",      # Use Anthropic
            model_name="claude-3-opus-20240229",  # Specific model
            max_iterations=5
        )

        print(result)

    finally:
        await agent_manager.cleanup()
        await llm.close()
```

### 5.5 With Database Session (For Tool Loading)

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

async def run_with_database():
    """Run agent with database-backed tool loading."""

    # Create database engine
    engine = create_async_engine("sqlite+aiosqlite:///./agent_tools.db")
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    llm = await LLMCore.create()

    try:
        agent_manager = AgentManager(
            provider_manager=llm._provider_manager,
            memory_manager=llm._memory_manager,
            storage_manager=llm._storage_manager
        )

        await agent_manager.initialize_sandbox()

        async with async_session() as db_session:
            task = AgentTask(goal="Process data files")

            result = await agent_manager.run_agent_loop(
                task=task,
                db_session=db_session,
                enabled_toolkits=["file_processing", "data_analysis"]
            )

            print(result)

    finally:
        await agent_manager.cleanup()
        await llm.close()
        await engine.dispose()
```

---

## 6. The Cognitive Cycle

### 6.1 Overview

The cognitive cycle is the heart of the agent system. It implements a structured approach to problem-solving:

```
┌──────────────────────────────────────────────────────────────┐
│                      COGNITIVE CYCLE                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────┐                                                │
│   │  PLAN   │  Create high-level strategy (once at start)   │
│   └────┬────┘                                                │
│        │                                                     │
│        ▼                                                     │
│   ┌─────────┐                                                │
│   │  THINK  │◀─────────────────────────────────────────┐    │
│   └────┬────┘                                          │    │
│        │  Decide next action based on state            │    │
│        ▼                                               │    │
│   ┌─────────┐                                          │    │
│   │   ACT   │  Execute chosen tool                     │    │
│   └────┬────┘                                          │    │
│        │                                               │    │
│        ▼                                               │    │
│   ┌─────────┐                                          │    │
│   │ OBSERVE │  Process tool result                     │    │
│   └────┬────┘                                          │    │
│        │                                               │    │
│        ▼                                               │    │
│   ┌─────────┐                                          │    │
│   │ REFLECT │  Update understanding, log episode       │    │
│   └────┬────┘                                          │    │
│        │                                               │    │
│        │  Not finished?                                │    │
│        └───────────────────────────────────────────────┘    │
│                                                              │
│   Finished or max iterations reached                         │
│        │                                                     │
│        ▼                                                     │
│   ┌──────────┐                                               │
│   │  RESULT  │                                               │
│   └──────────┘                                               │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 Plan Step

The planning step occurs once at the beginning of a task. The agent:

1. Analyzes the goal and constraints
2. Retrieves relevant context from memory
3. Creates a high-level strategy
4. Breaks down the task into subtasks

```python
# Internal implementation (for understanding)
async def plan_step(agent_state, session_id, provider_manager, ...):
    """
    Generate a strategic plan for the task.

    The plan includes:
    - Understanding of the goal
    - Key challenges identified
    - Proposed approach
    - Success criteria
    """
    # Calls LLM with planning prompt
    # Updates agent_state.plan
    pass
```

### 6.3 Think Step

The thinking step occurs at each iteration. The agent:

1. Reviews current state and progress
2. Considers available tools
3. Decides on the next action
4. Prepares tool call arguments

```python
# The agent reasons about what to do next
# It has access to:
# - The original goal
# - The plan
# - History of actions and observations
# - Available tools and their descriptions
# - Current context from memory
```

### 6.4 Act Step

The action step executes the chosen tool:

1. Validates tool call against access level
2. Routes execution to active sandbox (if enabled)
3. Captures stdout, stderr, exit code
4. Records execution in output tracker

```python
# Tool execution flow
tool_call = agent_state.pending_tool_call
result = await tool_manager.execute_tool(tool_call, session_id)
# Result is an ExecutionResult with stdout, stderr, exit_code
```

### 6.5 Observe Step

The observation step processes tool results:

1. Parses execution output
2. Extracts relevant information
3. Updates agent state with observation
4. Checks for errors or unexpected results

### 6.6 Reflect Step

The reflection step learns from the action:

1. Evaluates if progress was made
2. Updates understanding of the task
3. Logs episode to episodic memory
4. Decides if goal is achieved

```python
# After reflection, agent_state may have:
# - is_finished = True (goal achieved)
# - awaiting_human_approval = True (HITL pause)
# - Updated observations and insights
```

### 6.7 Termination Conditions

The loop terminates when:

1. **Goal achieved**: Agent calls `finish` tool with answer
2. **Max iterations**: Configured limit reached
3. **Human approval needed**: HITL pause requested
4. **Error**: Unrecoverable error occurs
5. **Timeout**: Task exceeds time limit

---

## 7. Sandbox System

### 7.1 Why Sandboxing?

**The Problem:**
Agents generate and execute code. Without isolation, this code runs on your host system with full access to files, network, and system resources.

**The Solution:**
The sandbox system ensures ALL agent-generated code executes in isolated environments (Docker containers or VMs), never on the host.

```
┌─────────────────────────────────────────────────────────────┐
│                      HOST SYSTEM                            │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  Your Application                    │   │
│  │                                                      │   │
│  │  agent.execute_python("import os; os.remove('/')")  │   │
│  └──────────────────────────┬───────────────────────────┘   │
│                             │                               │
│                             │ Routed to sandbox             │
│                             ▼                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              SANDBOX (Docker/VM)                     │   │
│  │  ┌───────────────────────────────────────────────┐  │   │
│  │  │  Isolated filesystem                          │  │   │
│  │  │  No access to host files                      │  │   │
│  │  │  Limited resources (CPU, memory)              │  │   │
│  │  │  Optional network isolation                   │  │   │
│  │  │                                               │  │   │
│  │  │  Code executes HERE, fails safely             │  │   │
│  │  └───────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Host system remains UNTOUCHED                              │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Sandbox Providers

#### Docker Provider (Recommended)

```python
from llmcore.agents.sandbox import DockerSandboxProvider, SandboxConfig

# Create provider
provider = DockerSandboxProvider(
    image="python:3.11-slim",
    memory_limit="1g",
    cpu_limit=2.0,
    timeout_seconds=600,
    network_enabled=False,
    image_whitelist=["python:3.*-slim"]
)

# Initialize
await provider.initialize()

# Create sandbox
config = SandboxConfig(
    task_id="task-001",
    access_level=SandboxAccessLevel.RESTRICTED
)
sandbox = await provider.create_sandbox(config)

# Execute code
result = await sandbox.execute_python("print('Hello from Docker!')")
print(result.stdout)  # "Hello from Docker!"

# Cleanup
await provider.cleanup()
```

#### VM Provider

```python
from llmcore.agents.sandbox import VMSandboxProvider, SandboxConfig

# Create provider
provider = VMSandboxProvider(
    host="192.168.1.100",
    port=22,
    username="agent",
    private_key_path="~/.ssh/agent_key",
    working_directory="/tmp/sandbox"
)

await provider.initialize()

config = SandboxConfig(task_id="task-002")
sandbox = await provider.create_sandbox(config)

result = await sandbox.execute_shell("uname -a")
print(result.stdout)

await provider.cleanup()
```

### 7.3 Access Levels

The sandbox system has two access levels:

#### RESTRICTED (Default)
- Tools filtered by `allowed` list in config
- No access to dangerous operations
- Cannot modify sandbox configuration
- Standard for untrusted agent tasks

#### FULL
- All tools available (except `denied` list)
- Can perform privileged operations
- Granted by:
  - Docker label: `llmcore.sandbox.full_access=true`
  - Container name pattern: `llmcore-trusted-*`
  - VM host in `full_access_hosts` list

```python
from llmcore.agents.sandbox import SandboxAccessLevel

# Check access level
if sandbox.access_level == SandboxAccessLevel.FULL:
    # Can use all tools
    pass
elif sandbox.access_level == SandboxAccessLevel.RESTRICTED:
    # Tool filtering applied
    pass
```

### 7.4 Using SandboxIntegration

The `SandboxIntegration` class provides high-level sandbox management:

```python
from llmcore.agents.sandbox_integration import SandboxIntegration

async def use_sandbox_integration():
    # Create from configuration
    integration = SandboxIntegration.from_dict({
        "mode": "docker",
        "docker": {
            "image": "python:3.11-slim",
            "memory_limit": "1g"
        }
    })

    # Initialize
    await integration.initialize()

    # Use sandbox context
    async with integration.sandbox_context(task) as ctx:
        # ctx.sandbox - the active sandbox
        # ctx.sandbox_id - unique identifier
        # ctx.access_level - RESTRICTED or FULL

        # Execute code
        result = await ctx.execute_python("print(2 + 2)")
        print(result.stdout)  # "4"

        # Log execution
        await ctx.log_execution("execute_python", "print(2+2)", result)

        # Check tool access
        if ctx.is_tool_allowed("execute_shell"):
            await ctx.execute_shell("ls -la")

    # Cleanup
    await integration.shutdown()
```

### 7.5 Sandbox Context in Agent Loop

When sandbox is enabled, the agent loop automatically uses it:

```python
async def run_agent_loop_sandboxed(self, task, ...):
    """Internal: How sandbox integrates with agent loop."""

    async with self._sandbox_integration.sandbox_context(task) as ctx:
        # set_active_sandbox() called automatically
        # All tool executions now route to this sandbox

        for iteration in range(max_iterations):
            # Think: Agent decides action
            await cognitive_cycle.think_step(...)

            # Act: Tool executes IN SANDBOX
            tool_result = await cognitive_cycle.act_step(...)
            # ↑ This calls tool_manager.execute_tool()
            # ↑ Which checks get_active_sandbox()
            # ↑ And routes execution to Docker/VM

            # Log execution for tracking
            await ctx.log_execution(
                tool_name,
                arguments,
                tool_result
            )

            # Observe and Reflect...

        # clear_active_sandbox() called automatically on exit
```

### 7.6 Output Tracking

The sandbox tracks all outputs and artifacts:

```python
from llmcore.agents.sandbox import OutputTracker

# Access tracker
tracker = OutputTracker(base_path="~/.llmcore/agent_outputs")
await tracker.initialize()

# Start a run
run_id = await tracker.start_run(task_id="task-001", metadata={
    "goal": "Process data files",
    "provider": "openai"
})

# Log executions (done automatically by sandbox context)
await tracker.log_execution(
    run_id=run_id,
    tool_name="execute_python",
    input_data="print('hello')",
    output_data="hello",
    success=True
)

# Track files
await tracker.track_file(
    run_id=run_id,
    file_path="/sandbox/output.txt",
    file_type="output",
    size_bytes=1024
)

# Complete run
summary = await tracker.complete_run(
    run_id=run_id,
    success=True,
    final_result="Task completed successfully"
)

# Query runs
recent_runs = await tracker.get_recent_runs(limit=10)
run_details = await tracker.get_run_details(run_id)
```

### 7.7 Ephemeral State Management

Sandboxes support ephemeral state that persists within a run:

```python
# Inside sandbox context
async with integration.sandbox_context(task) as ctx:
    # Set state (persists across tool calls within this run)
    await ctx.sandbox.set_state("processed_files", ["a.txt", "b.txt"])
    await ctx.sandbox.set_state("iteration", 1)

    # Get state
    files = await ctx.sandbox.get_state("processed_files")
    iteration = await ctx.sandbox.get_state("iteration", default=0)

    # List all state
    all_state = await ctx.sandbox.list_state()

    # Delete state
    await ctx.sandbox.delete_state("iteration")

# State is cleared when sandbox is destroyed
```

---

## 8. Tool Development

### 8.1 Built-in Tools

The sandbox system provides these built-in tools:

| Tool | Description |
|------|-------------|
| `execute_shell` | Run shell commands |
| `execute_python` | Execute Python code |
| `save_file` | Write content to file |
| `load_file` | Read file contents |
| `replace_in_file` | Find and replace in file |
| `append_to_file` | Append content to file |
| `list_files` | List directory contents |
| `file_exists` | Check if file exists |
| `delete_file` | Delete a file |
| `create_directory` | Create directory |
| `get_state` | Get ephemeral state |
| `set_state` | Set ephemeral state |
| `list_state` | List all state keys |
| `get_sandbox_info` | Get sandbox metadata |
| `get_recorded_files` | List tracked files |
| `semantic_search` | Search vector store |
| `episodic_search` | Search episodic memory |
| `calculator` | Safe math calculations |
| `finish` | Complete task with answer |
| `human_approval` | Request HITL approval |

### 8.2 Creating Custom Tools

```python
from llmcore.agents.tools import register_implementation
from llmcore.agents.sandbox import get_active_sandbox

async def my_custom_tool(
    query: str,
    options: dict = None,
    memory_manager = None  # Injected by ToolManager
) -> str:
    """
    A custom tool that does something useful.

    Args:
        query: The search query
        options: Optional configuration
        memory_manager: Injected dependency

    Returns:
        Result string
    """
    # Check if running in sandbox
    sandbox = get_active_sandbox()
    if sandbox:
        # Execute in sandbox if available
        result = await sandbox.execute_python(f"""
import json
# Custom logic here
result = {{"query": "{query}", "processed": True}}
print(json.dumps(result))
""")
        return result.stdout
    else:
        # Direct execution (no sandbox)
        return f"Processed: {query}"

# Register the tool
register_implementation(
    key="llmcore.tools.custom.my_tool",
    func=my_custom_tool,
    description="A custom tool that processes queries"
)
```

### 8.3 Tool Schema Definition

Tools need schemas for LLM function calling:

```python
MY_TOOL_SCHEMA = {
    "name": "my_custom_tool",
    "description": "A custom tool that processes queries with optional configuration",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The query to process"
            },
            "options": {
                "type": "object",
                "description": "Optional configuration",
                "properties": {
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["json", "text", "markdown"]
                    }
                }
            }
        },
        "required": ["query"]
    }
}
```

### 8.4 Registering Tools with ToolManager

```python
from llmcore.agents.tools import ToolManager, register_implementation
from llmcore.models import Tool

# 1. Register implementation function
register_implementation(
    key="llmcore.tools.custom.my_tool",
    func=my_custom_tool,
    description="Process queries"
)

# 2. Create tool definition
tool_def = Tool(
    name="my_custom_tool",
    description="A custom tool that processes queries",
    parameters=MY_TOOL_SCHEMA["parameters"]
)

# 3. Add to ToolManager
tool_manager._tool_definitions.append(tool_def)
tool_manager._implementation_map["my_custom_tool"] = "llmcore.tools.custom.my_tool"
```

### 8.5 Tool Best Practices

1. **Always check for sandbox**: Route execution appropriately

```python
sandbox = get_active_sandbox()
if sandbox:
    # Use sandbox execution
    result = await sandbox.execute_python(code)
else:
    # Direct execution or error
    raise RuntimeError("Sandbox required for this tool")
```

2. **Handle errors gracefully**: Return informative error messages

```python
try:
    result = await sandbox.execute_shell(command)
    if result.exit_code != 0:
        return f"Command failed (exit {result.exit_code}): {result.stderr}"
    return result.stdout
except SandboxTimeoutError:
    return "Command timed out - try a simpler approach"
except SandboxExecutionError as e:
    return f"Execution error: {e}"
```

3. **Document thoroughly**: LLMs use descriptions to decide when to use tools

```python
async def analyze_code(
    code: str,
    language: str = "python"
) -> str:
    """
    Analyze source code for issues and improvements.

    Use this tool when you need to:
    - Find bugs or potential issues in code
    - Suggest improvements or optimizations
    - Check for security vulnerabilities
    - Verify code follows best practices

    Args:
        code: The source code to analyze
        language: Programming language (default: python)

    Returns:
        Analysis report with findings and suggestions
    """
    pass
```

---

## 9. Memory & Context Integration

### 9.1 Semantic Memory

Agents can search the vector store for relevant information:

```python
# Built-in semantic_search tool
result = await tool_manager.execute_tool(
    ToolCall(
        name="semantic_search",
        arguments={
            "query": "How to implement authentication",
            "top_k": 5,
            "collection_name": "codebase"
        }
    )
)
```

### 9.2 Episodic Memory

Agents can recall past experiences:

```python
# Built-in episodic_search tool
result = await tool_manager.execute_tool(
    ToolCall(
        name="episodic_search",
        arguments={
            "query": "Previous attempts at file processing",
            "session_id": session_id,
            "limit": 10
        }
    )
)
```

### 9.3 Using Memory in Custom Tools

```python
async def my_memory_aware_tool(
    query: str,
    memory_manager = None,  # Injected
    storage_manager = None  # Injected
) -> str:
    """Tool that uses memory context."""

    # Search semantic memory
    if memory_manager:
        context = await memory_manager.search_semantic(
            query=query,
            top_k=3
        )

    # Search episodic memory
    if storage_manager:
        episodes = await storage_manager.search_episodes(
            query=query,
            limit=5
        )

    # Use context in processing
    return process_with_context(query, context, episodes)
```

### 9.4 Staging Context for Agent Tasks

```python
from llmcore.models import ContextItem, ContextItemType

# Create context items
context_items = [
    ContextItem(
        id="doc-1",
        type=ContextItemType.DOCUMENT,
        content="Relevant documentation content...",
        metadata={"source": "README.md"}
    ),
    ContextItem(
        id="code-1",
        type=ContextItemType.CODE,
        content="def example(): pass",
        metadata={"file": "example.py", "language": "python"}
    )
]

# Include in agent task
task = AgentTask(
    goal="Improve the example function",
    context_items=context_items
)
```

---

## 10. Advanced Patterns

### 10.1 Custom Agent Manager

```python
from llmcore.agents import AgentManager
from llmcore.agents.sandbox_integration import SandboxAgentMixin

class MyCustomAgentManager(AgentManager, SandboxAgentMixin):
    """Custom agent manager with additional features."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._custom_state = {}

    async def run_agent_loop(self, task, **kwargs):
        """Override with custom pre/post processing."""

        # Pre-processing
        self._custom_state["start_time"] = time.time()
        await self._notify_task_start(task)

        try:
            # Run with sandbox
            result = await self.run_agent_loop_sandboxed(task, **kwargs)

            # Post-processing
            await self._notify_task_complete(task, result)
            return result

        except Exception as e:
            await self._notify_task_error(task, e)
            raise

    async def _notify_task_start(self, task):
        """Custom notification on task start."""
        pass

    async def _notify_task_complete(self, task, result):
        """Custom notification on task completion."""
        pass
```

### 10.2 Parallel Agent Execution

```python
import asyncio
from llmcore import LLMCore, AgentManager
from llmcore.models import AgentTask

async def run_parallel_agents():
    """Run multiple agents in parallel."""

    llm = await LLMCore.create()

    # Create multiple agent managers (each gets own sandbox)
    agents = []
    for i in range(3):
        agent = AgentManager(
            provider_manager=llm._provider_manager,
            memory_manager=llm._memory_manager,
            storage_manager=llm._storage_manager
        )
        await agent.initialize_sandbox()
        agents.append(agent)

    # Define tasks
    tasks = [
        AgentTask(goal="Analyze Python files"),
        AgentTask(goal="Generate test cases"),
        AgentTask(goal="Create documentation")
    ]

    # Run in parallel
    results = await asyncio.gather(*[
        agent.run_agent_loop(task, use_sandbox=True)
        for agent, task in zip(agents, tasks)
    ])

    # Cleanup
    for agent in agents:
        await agent.cleanup()
    await llm.close()

    return results
```

### 10.3 Human-in-the-Loop (HITL)

```python
async def run_with_hitl():
    """Run agent with human approval gates."""

    llm = await LLMCore.create()
    agent = AgentManager(...)
    await agent.initialize_sandbox()

    task = AgentTask(
        goal="Delete old log files",
        constraints=["Require human approval before deletion"]
    )

    while True:
        result = await agent.run_agent_loop(task=task)

        # Check if paused for approval
        if result.startswith("HITL_PAUSE:"):
            prompt = result[11:]  # Extract prompt

            # Present to human
            print(f"Agent requests approval: {prompt}")
            approval = input("Approve? (yes/no): ")

            if approval.lower() == "yes":
                # Continue with approval
                task = AgentTask(
                    goal=task.goal,
                    context={"human_approval": True, "approved_action": prompt}
                )
            else:
                # Abort or redirect
                break
        else:
            # Task completed
            print(f"Result: {result}")
            break

    await agent.cleanup()
    await llm.close()
```

### 10.4 Streaming Agent Progress

```python
from typing import AsyncGenerator

async def stream_agent_progress(
    agent: AgentManager,
    task: AgentTask
) -> AsyncGenerator[dict, None]:
    """Stream agent progress events."""

    # Hook into cognitive cycle events
    original_think = cognitive_cycle.think_step
    original_act = cognitive_cycle.act_step

    async def hooked_think(*args, **kwargs):
        yield {"event": "think_start", "iteration": kwargs.get("iteration")}
        result = await original_think(*args, **kwargs)
        yield {"event": "think_complete", "action": result}
        return result

    async def hooked_act(*args, **kwargs):
        yield {"event": "act_start", "tool": kwargs.get("tool_name")}
        result = await original_act(*args, **kwargs)
        yield {"event": "act_complete", "result": result}
        return result

    # Run with hooks
    # (Implementation details omitted for brevity)
```

### 10.5 Custom Sandbox Images

```dockerfile
# Dockerfile.agent-sandbox
FROM python:3.11-slim

# Install common dependencies
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    requests \
    beautifulsoup4 \
    pyyaml

# Create non-root user
RUN useradd -m -s /bin/bash agent
USER agent
WORKDIR /home/agent

# Add custom tools
COPY tools/ /home/agent/tools/
```

```python
# Use custom image
await agent.initialize_sandbox({
    "mode": "docker",
    "docker": {
        "image": "my-company/agent-sandbox:latest",
        "image_whitelist": ["my-company/*"]
    }
})
```

---

## 11. Security Model

### 11.1 Security Principles

1. **Isolation**: Agent code NEVER runs on host
2. **Least Privilege**: RESTRICTED access by default
3. **Whitelisting**: Only approved images/hosts allowed
4. **Resource Limits**: CPU, memory, timeout enforced
5. **Network Isolation**: Disabled by default

### 11.2 Threat Model

| Threat | Mitigation |
|--------|------------|
| Malicious code execution | Sandbox isolation |
| File system access | Container/VM filesystem isolation |
| Network attacks | Network disabled by default |
| Resource exhaustion | CPU/memory limits, timeouts |
| Privilege escalation | Non-root containers, tool filtering |
| Data exfiltration | No network, limited volume mounts |

### 11.3 Access Level Security

```python
# RESTRICTED access (default)
# - Can only use tools in "allowed" list
# - Cannot access host filesystem
# - Limited resource access

# FULL access (explicitly granted)
# - All tools except "denied" list
# - Still sandboxed, but fewer restrictions
# - Grant via:
#   - Docker label
#   - Container name pattern
#   - VM host whitelist
```

### 11.4 Image Whitelisting

```toml
[agents.sandbox.docker]
# Only these images can be used
image_whitelist = [
    "python:3.*-slim",           # Official Python slim
    "python:3.*-bookworm",       # Official Python Debian
    "my-company/agent-*",        # Company-approved images
    "llmcore-sandbox:*"          # LLMCore official images
]
```

### 11.5 Network Security

```toml
[agents.sandbox.docker]
# Default: network disabled
network_enabled = false

# If network needed (use with caution):
# network_enabled = true
# Consider:
# - Firewall rules
# - Proxy configuration
# - Egress filtering
```

### 11.6 Volume Mount Security

```toml
[agents.sandbox.volumes]
# Shared directory (agent can read/write)
share_path = "~/.llmcore/agent_share"

# Output directory (preserved after cleanup)
outputs_path = "~/.llmcore/agent_outputs"

# NEVER mount:
# - Home directory
# - System directories
# - Sensitive configuration
# - SSH keys
```

### 11.7 Security Checklist

- [ ] Using Docker or VM sandbox (not host execution)
- [ ] Image whitelist configured
- [ ] Network disabled unless required
- [ ] Resource limits set appropriately
- [ ] Volume mounts minimized
- [ ] RESTRICTED access level for untrusted tasks
- [ ] Tool allowed/denied lists configured
- [ ] Timeout configured to prevent runaway processes
- [ ] Output tracking enabled for audit trail

---

## 12. Error Handling

### 12.1 Exception Hierarchy

```
LLMCoreError
└── SandboxError
    ├── SandboxInitializationError  # Failed to create sandbox
    ├── SandboxExecutionError       # Command/code execution failed
    ├── SandboxTimeoutError         # Operation exceeded time limit
    ├── SandboxAccessDenied         # Security policy violation
    ├── SandboxResourceError        # Resource limits exceeded
    ├── SandboxConnectionError      # Failed to connect (VM/Docker)
    └── SandboxCleanupError         # Failed to cleanup resources
```

### 12.2 Handling Errors

```python
from llmcore.exceptions import (
    SandboxError,
    SandboxInitializationError,
    SandboxExecutionError,
    SandboxTimeoutError,
    SandboxAccessDenied,
    SandboxResourceError,
    SandboxConnectionError
)

async def robust_agent_execution():
    try:
        result = await agent.run_agent_loop(task, use_sandbox=True)
        return result

    except SandboxInitializationError as e:
        # Docker not running, image not found, etc.
        logger.error(f"Failed to initialize sandbox: {e}")
        # Fallback or notify user

    except SandboxTimeoutError as e:
        # Task took too long
        logger.warning(f"Agent task timed out: {e}")
        # Return partial results or retry with simpler task

    except SandboxAccessDenied as e:
        # Tool not allowed
        logger.error(f"Access denied: {e}")
        # Check tool configuration

    except SandboxResourceError as e:
        # Out of memory, CPU, etc.
        logger.error(f"Resource limit exceeded: {e}")
        # Increase limits or simplify task

    except SandboxExecutionError as e:
        # Code failed to execute
        logger.error(f"Execution failed: {e}")
        # Check code/command

    except SandboxConnectionError as e:
        # Can't reach Docker/VM
        logger.error(f"Connection failed: {e}")
        # Check Docker daemon, SSH config

    except SandboxError as e:
        # Catch-all for sandbox errors
        logger.error(f"Sandbox error: {e}")
```

### 12.3 Error Details

All sandbox exceptions include detailed information:

```python
try:
    await sandbox.execute_python("bad code")
except SandboxExecutionError as e:
    print(f"Message: {e.message}")
    print(f"Exit code: {e.exit_code}")
    print(f"Stderr: {e.stderr}")
    print(f"Sandbox ID: {e.sandbox_id}")
    print(f"Details: {e.details}")

    # Serialize for logging
    error_dict = e.to_dict()
```

### 12.4 Retry Strategies

```python
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
async def resilient_sandbox_execution(sandbox, code):
    """Execute with automatic retry."""
    return await sandbox.execute_python(code)

# Or manual retry
async def execute_with_retry(sandbox, code, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await sandbox.execute_python(code)
        except SandboxTimeoutError:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                raise
```

---

## 13. Monitoring & Observability

### 13.1 Logging

```python
import logging

# Configure llmcore logging
logging.getLogger("llmcore").setLevel(logging.DEBUG)
logging.getLogger("llmcore.agents").setLevel(logging.DEBUG)
logging.getLogger("llmcore.agents.sandbox").setLevel(logging.DEBUG)

# Or via config
# [llmcore]
# log_level = "DEBUG"
```

### 13.2 Output Tracking Queries

```python
from llmcore.agents.sandbox import OutputTracker

tracker = OutputTracker()
await tracker.initialize()

# Get recent runs
recent = await tracker.get_recent_runs(limit=20)
for run in recent:
    print(f"Run {run['run_id']}: {run['status']} ({run['duration_seconds']}s)")

# Get run details
details = await tracker.get_run_details(run_id)
print(f"Task: {details['task_id']}")
print(f"Executions: {len(details['executions'])}")
print(f"Files: {len(details['files'])}")

# Search runs
runs = await tracker.search_runs(
    task_id="task-*",
    start_date="2024-01-01",
    end_date="2024-12-31",
    success_only=False
)
```

### 13.3 Metrics Collection

```python
import time
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class AgentMetrics:
    task_id: str
    start_time: float
    end_time: float
    iterations: int
    tool_calls: List[Dict]
    success: bool
    error: str = None

class MetricsCollector:
    def __init__(self):
        self.metrics: List[AgentMetrics] = []

    async def collect(self, task, agent_result):
        # Collect metrics from run
        pass

    def summary(self):
        # Return aggregated metrics
        total = len(self.metrics)
        success = sum(1 for m in self.metrics if m.success)
        avg_duration = sum(m.end_time - m.start_time for m in self.metrics) / total
        return {
            "total_runs": total,
            "success_rate": success / total,
            "avg_duration": avg_duration
        }
```

### 13.4 Integration with Observability Platforms

```python
# OpenTelemetry integration (if available)
from llmcore.tracing import get_tracer, create_span

tracer = get_tracer("my_application")

async def traced_agent_run():
    with create_span(tracer, "agent.task") as span:
        span.set_attribute("task.goal", task.goal)

        result = await agent.run_agent_loop(task)

        span.set_attribute("task.success", True)
        span.set_attribute("task.result_length", len(result))

        return result
```

---

## 14. Best Practices

### 14.1 Task Design

```python
# GOOD: Clear, specific goal with constraints
task = AgentTask(
    goal="Create a Python function that validates email addresses using regex",
    constraints=[
        "Function should be named 'validate_email'",
        "Should return True for valid emails, False otherwise",
        "Include docstring with examples",
        "Write unit tests"
    ]
)

# BAD: Vague, open-ended goal
task = AgentTask(
    goal="Do something with emails"
)
```

### 14.2 Resource Management

```python
# ALWAYS use context managers or explicit cleanup

# Good: Context manager
async with LLMCore.create() as llm:
    agent = AgentManager(...)
    await agent.initialize_sandbox()
    try:
        result = await agent.run_agent_loop(task)
    finally:
        await agent.cleanup()

# Good: Explicit cleanup
llm = await LLMCore.create()
try:
    # ... use agent
finally:
    await agent.cleanup()
    await llm.close()
```

### 14.3 Timeout Configuration

```python
# Set appropriate timeouts based on task complexity

# Simple tasks: 60-120 seconds
await agent.initialize_sandbox({
    "docker": {"timeout_seconds": 120}
})

# Complex tasks: 300-600 seconds
await agent.initialize_sandbox({
    "docker": {"timeout_seconds": 600}
})

# Per-task override
result = await agent.run_agent_loop(
    task=complex_task,
    max_iterations=20  # More iterations for complex tasks
)
```

### 14.4 Memory Limits

```python
# Adjust based on task requirements

# Light tasks (text processing): 256-512MB
"memory_limit": "512m"

# Medium tasks (data analysis): 1-2GB
"memory_limit": "1g"

# Heavy tasks (ML, large datasets): 4-8GB
"memory_limit": "4g"
```

### 14.5 Error Recovery

```python
async def robust_task_execution(goal: str, max_attempts: int = 3):
    """Execute task with progressive simplification on failure."""

    for attempt in range(max_attempts):
        try:
            task = AgentTask(
                goal=goal,
                constraints=[f"Attempt {attempt + 1} of {max_attempts}"]
            )
            return await agent.run_agent_loop(task)

        except SandboxTimeoutError:
            # Simplify task
            goal = f"Simplified: {goal}"

        except SandboxResourceError:
            # Increase resources
            await agent.shutdown_sandbox()
            await agent.initialize_sandbox({
                "docker": {
                    "memory_limit": f"{2 ** (attempt + 1)}g"
                }
            })

    raise RuntimeError("Task failed after all attempts")
```

### 14.6 Testing Agents

```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_agent_task():
    """Test agent completes task successfully."""

    llm = await LLMCore.create()
    agent = AgentManager(...)
    await agent.initialize_sandbox()

    try:
        task = AgentTask(goal="Calculate 2 + 2")
        result = await agent.run_agent_loop(task, max_iterations=5)

        assert "4" in result

    finally:
        await agent.cleanup()
        await llm.close()

@pytest.mark.asyncio
async def test_agent_with_mock_sandbox():
    """Test agent with mocked sandbox."""

    mock_sandbox = AsyncMock()
    mock_sandbox.execute_python.return_value = ExecutionResult(
        stdout="4",
        stderr="",
        exit_code=0
    )

    with patch.object(agent, '_sandbox_integration') as mock_integration:
        mock_integration.sandbox_context.return_value.__aenter__.return_value.sandbox = mock_sandbox

        result = await agent.run_agent_loop(task)

        mock_sandbox.execute_python.assert_called()
```

---

## 15. Troubleshooting

### 15.1 Docker Issues

**Problem: Docker daemon not running**
```
SandboxInitializationError: Cannot connect to Docker daemon
```
**Solution:**
```bash
# Start Docker
sudo systemctl start docker
# Or on macOS
open -a Docker
```

**Problem: Permission denied**
```
SandboxInitializationError: Permission denied while connecting to Docker
```
**Solution:**
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Log out and back in
```

**Problem: Image not found**
```
SandboxInitializationError: Image python:3.11-slim not found
```
**Solution:**
```bash
# Pull image manually
docker pull python:3.11-slim
```

### 15.2 VM/SSH Issues

**Problem: SSH connection refused**
```
SandboxConnectionError: Connection refused to 192.168.1.100:22
```
**Solution:**
- Check VM is running
- Verify SSH service is active
- Check firewall rules

**Problem: Authentication failed**
```
SandboxConnectionError: Authentication failed
```
**Solution:**
```bash
# Test SSH manually
ssh -i ~/.ssh/agent_key agent@192.168.1.100

# Check key permissions
chmod 600 ~/.ssh/agent_key
```

### 15.3 Timeout Issues

**Problem: Tasks frequently timeout**
```
SandboxTimeoutError: Operation timed out after 600 seconds
```
**Solutions:**
1. Increase timeout: `timeout_seconds = 1200`
2. Simplify task with better constraints
3. Break task into smaller subtasks
4. Use faster LLM model

### 15.4 Memory Issues

**Problem: Container killed (OOM)**
```
SandboxResourceError: Container exceeded memory limit
```
**Solutions:**
1. Increase limit: `memory_limit = "2g"`
2. Process data in chunks
3. Use more efficient algorithms

### 15.5 Tool Access Issues

**Problem: Tool not allowed**
```
SandboxAccessDenied: Tool 'execute_shell' not allowed for RESTRICTED access
```
**Solutions:**
1. Add tool to `allowed` list in config
2. Use FULL access (if appropriate)
3. Use alternative allowed tool

### 15.6 Debugging Tips

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("llmcore").setLevel(logging.DEBUG)

# Check sandbox state
async with integration.sandbox_context(task) as ctx:
    print(f"Sandbox ID: {ctx.sandbox_id}")
    print(f"Access level: {ctx.access_level}")
    print(f"Working dir: {ctx.sandbox.working_directory}")

    # Test execution
    result = await ctx.execute_shell("pwd")
    print(f"Working directory: {result.stdout}")

    result = await ctx.execute_shell("ls -la")
    print(f"Files: {result.stdout}")

# Check Docker container directly
# docker ps  # See running containers
# docker logs <container_id>  # See container logs
# docker exec -it <container_id> bash  # Enter container
```

---

## 16. API Reference

### 16.1 AgentManager

```python
class AgentManager:
    """Orchestrates the autonomous agent execution loop."""

    def __init__(
        self,
        provider_manager: ProviderManager,
        memory_manager: MemoryManager,
        storage_manager: StorageManager
    ):
        """
        Initialize AgentManager.

        Args:
            provider_manager: For LLM interactions
            memory_manager: For context retrieval
            storage_manager: For episodic memory logging
        """

    async def initialize_sandbox(
        self,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize sandbox support.

        Args:
            config: Optional sandbox configuration. If None, reads from llmcore config.

        Raises:
            SandboxError: If initialization fails
        """

    async def shutdown_sandbox(self) -> None:
        """Shutdown sandbox and cleanup resources."""

    @property
    def sandbox_enabled(self) -> bool:
        """Check if sandbox is enabled."""

    async def run_agent_loop(
        self,
        task: AgentTask,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        max_iterations: int = 10,
        session_id: Optional[str] = None,
        db_session: Optional[AsyncSession] = None,
        enabled_toolkits: Optional[List[str]] = None,
        use_sandbox: Optional[bool] = None
    ) -> str:
        """
        Run the cognitive loop to complete a task.

        Args:
            task: The AgentTask to complete
            provider_name: Override default LLM provider
            model_name: Override default model
            max_iterations: Maximum cognitive cycles
            session_id: Optional session for context
            db_session: Database session for tool loading
            enabled_toolkits: Toolkits to enable
            use_sandbox: True=require sandbox, False=skip, None=auto

        Returns:
            Final result string

        Raises:
            LLMCoreError: If agent loop fails
            SandboxError: If sandbox required but not available
        """

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""

    async def cleanup(self) -> None:
        """Cleanup all resources."""
```

### 16.2 SandboxIntegration

```python
class SandboxIntegration:
    """High-level sandbox management for agent integration."""

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "SandboxIntegration":
        """Create from configuration dictionary."""

    @classmethod
    def from_llmcore_config(cls, config) -> "SandboxIntegration":
        """Create from LLMCore config object."""

    async def initialize(self) -> None:
        """Initialize sandbox registry and output tracker."""

    @asynccontextmanager
    async def sandbox_context(
        self,
        task: AgentTask
    ) -> AsyncGenerator[SandboxContext, None]:
        """
        Context manager for sandbox-scoped execution.

        Usage:
            async with integration.sandbox_context(task) as ctx:
                result = await ctx.execute_python("print(42)")
        """

    async def shutdown(self) -> None:
        """Shutdown and cleanup all resources."""
```

### 16.3 SandboxContext

```python
class SandboxContext:
    """Provides sandbox access during execution."""

    @property
    def sandbox(self) -> SandboxProvider:
        """Get the active sandbox provider."""

    @property
    def sandbox_id(self) -> str:
        """Get sandbox identifier."""

    @property
    def run_id(self) -> str:
        """Get output tracking run ID."""

    @property
    def access_level(self) -> SandboxAccessLevel:
        """Get access level (RESTRICTED or FULL)."""

    async def log_execution(
        self,
        tool_name: str,
        input_data: str,
        output: ExecutionResult
    ) -> None:
        """Log tool execution for tracking."""

    async def track_file(
        self,
        file_path: str,
        file_type: str = "output"
    ) -> None:
        """Track file for output preservation."""

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if tool is allowed for current access level."""

    async def execute_shell(self, command: str) -> ExecutionResult:
        """Execute shell command in sandbox."""

    async def execute_python(self, code: str) -> ExecutionResult:
        """Execute Python code in sandbox."""
```

### 16.4 SandboxProvider (Abstract)

```python
class SandboxProvider(ABC):
    """Abstract base for sandbox providers."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider."""

    @abstractmethod
    async def create_sandbox(self, config: SandboxConfig) -> "SandboxProvider":
        """Create a new sandbox instance."""

    @abstractmethod
    async def execute_shell(
        self,
        command: str,
        timeout: Optional[int] = None,
        working_dir: Optional[str] = None
    ) -> ExecutionResult:
        """Execute shell command."""

    @abstractmethod
    async def execute_python(
        self,
        code: str,
        timeout: Optional[int] = None
    ) -> ExecutionResult:
        """Execute Python code."""

    @abstractmethod
    async def write_file(
        self,
        path: str,
        content: str,
        mode: str = "w"
    ) -> None:
        """Write file in sandbox."""

    @abstractmethod
    async def read_file(self, path: str) -> str:
        """Read file from sandbox."""

    @abstractmethod
    async def list_files(
        self,
        path: str = ".",
        recursive: bool = False
    ) -> List[FileInfo]:
        """List files in sandbox."""

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup sandbox resources."""
```

### 16.5 Data Models

```python
@dataclass
class AgentTask:
    """Defines a task for an agent to complete."""
    goal: str
    constraints: List[str] = field(default_factory=list)
    context_items: List[ContextItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionResult:
    """Result of code/command execution."""
    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FileInfo:
    """Information about a file in sandbox."""
    path: str
    name: str
    size: int
    is_directory: bool
    modified_time: Optional[datetime] = None

class SandboxAccessLevel(Enum):
    """Access level for sandbox operations."""
    RESTRICTED = "restricted"  # Tool filtering applied
    FULL = "full"              # All tools available

class SandboxStatus(Enum):
    """Status of a sandbox instance."""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"

class SandboxMode(Enum):
    """Sandbox execution mode."""
    DOCKER = "docker"
    VM = "vm"
    HYBRID = "hybrid"
```

---

## Appendix A: Complete Example Application

```python
#!/usr/bin/env python3
"""
Complete example: Code analysis agent using LLMCore.

This agent analyzes Python files, finds issues, and suggests improvements.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from llmcore import LLMCore, AgentManager
from llmcore.models import AgentTask, ContextItem, ContextItemType
from llmcore.exceptions import SandboxError, LLMCoreError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CodeAnalysisAgent:
    """Agent that analyzes code and suggests improvements."""

    def __init__(self):
        self.llm: Optional[LLMCore] = None
        self.agent: Optional[AgentManager] = None

    async def initialize(self):
        """Initialize LLMCore and agent manager."""
        logger.info("Initializing Code Analysis Agent...")

        # Create LLMCore instance
        self.llm = await LLMCore.create()

        # Create agent manager
        self.agent = AgentManager(
            provider_manager=self.llm._provider_manager,
            memory_manager=self.llm._memory_manager,
            storage_manager=self.llm._storage_manager
        )

        # Initialize sandbox with appropriate settings
        await self.agent.initialize_sandbox({
            "mode": "docker",
            "docker": {
                "image": "python:3.11-slim",
                "memory_limit": "1g",
                "timeout_seconds": 300,
                "network_enabled": False
            }
        })

        logger.info("Agent initialized successfully")

    async def analyze_file(self, file_path: str) -> str:
        """
        Analyze a Python file and return suggestions.

        Args:
            file_path: Path to Python file

        Returns:
            Analysis results with suggestions
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read file content
        content = path.read_text()

        # Create context item with file content
        context = ContextItem(
            id=f"file-{path.name}",
            type=ContextItemType.CODE,
            content=content,
            metadata={
                "file_path": str(path),
                "language": "python",
                "lines": len(content.splitlines())
            }
        )

        # Define analysis task
        task = AgentTask(
            goal=f"Analyze the Python file '{path.name}' and provide actionable suggestions",
            constraints=[
                "Focus on code quality, not style preferences",
                "Identify potential bugs or issues",
                "Suggest performance improvements if applicable",
                "Check for security concerns",
                "Provide specific line numbers when referencing issues",
                "Format output as a structured report"
            ],
            context_items=[context]
        )

        # Run analysis
        logger.info(f"Analyzing {file_path}...")
        result = await self.agent.run_agent_loop(
            task=task,
            use_sandbox=True,
            max_iterations=10
        )

        return result

    async def analyze_directory(self, dir_path: str) -> dict:
        """
        Analyze all Python files in a directory.

        Args:
            dir_path: Path to directory

        Returns:
            Dictionary mapping file paths to analysis results
        """
        path = Path(dir_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        results = {}
        python_files = list(path.glob("**/*.py"))

        logger.info(f"Found {len(python_files)} Python files in {dir_path}")

        for py_file in python_files:
            try:
                results[str(py_file)] = await self.analyze_file(str(py_file))
            except Exception as e:
                logger.error(f"Failed to analyze {py_file}: {e}")
                results[str(py_file)] = f"Analysis failed: {e}"

        return results

    async def cleanup(self):
        """Cleanup resources."""
        if self.agent:
            await self.agent.cleanup()
        if self.llm:
            await self.llm.close()
        logger.info("Agent cleanup complete")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Code Analysis Agent")
    parser.add_argument("path", help="File or directory to analyze")
    parser.add_argument("--output", "-o", help="Output file for results")
    args = parser.parse_args()

    agent = CodeAnalysisAgent()

    try:
        await agent.initialize()

        path = Path(args.path)
        if path.is_file():
            result = await agent.analyze_file(args.path)
            print("\n" + "=" * 60)
            print(f"ANALYSIS RESULTS: {path.name}")
            print("=" * 60)
            print(result)
        elif path.is_dir():
            results = await agent.analyze_directory(args.path)
            for file_path, result in results.items():
                print("\n" + "=" * 60)
                print(f"FILE: {file_path}")
                print("=" * 60)
                print(result)
        else:
            print(f"Error: {args.path} is not a valid file or directory")
            return

        # Save to file if requested
        if args.output:
            with open(args.output, "w") as f:
                if isinstance(result, dict):
                    import json
                    json.dump(result, f, indent=2)
                else:
                    f.write(result)
            print(f"\nResults saved to {args.output}")

    except SandboxError as e:
        logger.error(f"Sandbox error: {e}")
        print(f"Error: Sandbox failed - {e}")
    except LLMCoreError as e:
        logger.error(f"LLMCore error: {e}")
        print(f"Error: LLMCore failed - {e}")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"Error: {e}")
    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Appendix B: Configuration Template

```toml
# ~/.config/llmcore/config.toml
# Complete configuration template for LLMCore with agent system

# =============================================================================
# Core Settings
# =============================================================================
[llmcore]
default_provider = "ollama"
default_embedding_model = "all-MiniLM-L6-v2"
log_level = "INFO"
log_raw_payloads = false

# =============================================================================
# LLM Providers
# =============================================================================
[providers.ollama]
host = "http://localhost:11434"
default_model = "llama3.2:latest"
timeout = 120

[providers.openai]
# api_key = ""  # Set via LLMCORE_PROVIDERS__OPENAI__API_KEY
default_model = "gpt-4o"
timeout = 60

[providers.anthropic]
# api_key = ""  # Set via LLMCORE_PROVIDERS__ANTHROPIC__API_KEY
default_model = "claude-3-opus-20240229"
timeout = 60

# =============================================================================
# Storage
# =============================================================================
[storage.session]
type = "sqlite"
path = "~/.local/share/llmcore/sessions.db"

[storage.vector]
type = "chromadb"
path = "~/.local/share/llmcore/chroma_db"
default_collection = "default_collection"

# =============================================================================
# Agent System
# =============================================================================
[agents]
max_iterations = 10
default_timeout = 600

[agents.sandbox]
mode = "docker"
fallback_enabled = true

[agents.sandbox.docker]
enabled = true
image = "python:3.11-slim"
image_whitelist = ["python:3.*-slim", "python:3.*-bookworm"]
memory_limit = "1g"
cpu_limit = 2.0
timeout_seconds = 600
network_enabled = false
full_access_label = "llmcore.sandbox.full_access=true"
full_access_name_patterns = ["llmcore-trusted-*"]

[agents.sandbox.vm]
enabled = false
host = ""
port = 22
username = "agent"
private_key_path = ""
use_ssh_agent = false
timeout_seconds = 600
working_directory = "/tmp/llmcore_sandbox"
full_access_hosts = []

[agents.sandbox.volumes]
share_path = "~/.llmcore/agent_share"
outputs_path = "~/.llmcore/agent_outputs"

[agents.sandbox.tools]
allowed = [
    "execute_shell", "execute_python",
    "save_file", "load_file", "replace_in_file", "append_to_file",
    "list_files", "file_exists", "delete_file", "create_directory",
    "get_state", "set_state", "list_state",
    "get_sandbox_info", "get_recorded_files",
    "semantic_search", "episodic_search", "calculator",
    "finish", "human_approval"
]
denied = []

[agents.sandbox.output_tracking]
enabled = true
max_log_entries = 1000
max_run_age_days = 30
max_runs = 100
```

---

*Document Version: 1.0.0*
*Last Updated: 2024*
*LLMCore Version: 0.26.0+*
