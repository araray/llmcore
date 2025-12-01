# src/llmcore/agents/__init__.py
"""
Agents module for the LLMCore library.

This package contains the agentic execution engine including:
- AgentManager: Orchestrates the Think -> Act -> Observe loop
- ToolManager: Dynamic tool registration and execution
- Sandbox: Isolated execution environments for agent tasks

UPDATED: Added sandbox system exports for Layer 1 integration.
"""

from .manager import AgentManager
from .tools import ToolManager

# Sandbox system exports
from .sandbox import (
    # Core classes
    SandboxRegistry,
    SandboxRegistryConfig,
    SandboxConfig,
    SandboxProvider,
    SandboxMode,
    SandboxAccessLevel,
    SandboxStatus,
    ExecutionResult,
    FileInfo,

    # Providers
    DockerSandboxProvider,
    VMSandboxProvider,

    # Utilities
    EphemeralResourceManager,
    OutputTracker,

    # Configuration
    load_sandbox_config,
    create_registry_config,

    # Exceptions
    SandboxError,
    SandboxInitializationError,
    SandboxExecutionError,
    SandboxTimeoutError,
    SandboxAccessDenied,
    SandboxResourceError,
    SandboxConnectionError,
    SandboxCleanupError,

    # Tool management
    set_active_sandbox,
    clear_active_sandbox,
    get_active_sandbox,
    SANDBOX_TOOL_IMPLEMENTATIONS,
    SANDBOX_TOOL_SCHEMAS,
)

# Integration bridge exports
from .sandbox_integration import (
    SandboxIntegration,
    SandboxContext,
    SandboxAgentMixin,
    register_sandbox_tools,
    get_sandbox_tool_definitions,
)

__all__ = [
    # Core agent classes
    "AgentManager",
    "ToolManager",

    # Sandbox core
    "SandboxRegistry",
    "SandboxRegistryConfig",
    "SandboxConfig",
    "SandboxProvider",
    "SandboxMode",
    "SandboxAccessLevel",
    "SandboxStatus",
    "ExecutionResult",
    "FileInfo",

    # Sandbox providers
    "DockerSandboxProvider",
    "VMSandboxProvider",

    # Sandbox utilities
    "EphemeralResourceManager",
    "OutputTracker",

    # Sandbox configuration
    "load_sandbox_config",
    "create_registry_config",

    # Sandbox exceptions
    "SandboxError",
    "SandboxInitializationError",
    "SandboxExecutionError",
    "SandboxTimeoutError",
    "SandboxAccessDenied",
    "SandboxResourceError",
    "SandboxConnectionError",
    "SandboxCleanupError",

    # Sandbox tool management
    "set_active_sandbox",
    "clear_active_sandbox",
    "get_active_sandbox",
    "SANDBOX_TOOL_IMPLEMENTATIONS",
    "SANDBOX_TOOL_SCHEMAS",

    # Integration bridge
    "SandboxIntegration",
    "SandboxContext",
    "SandboxAgentMixin",
    "register_sandbox_tools",
    "get_sandbox_tool_definitions",
]
