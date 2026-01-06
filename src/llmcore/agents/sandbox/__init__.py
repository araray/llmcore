# src/llmcore/agents/sandbox/__init__.py
"""
Sandbox System for LLMCore Agent Execution.

Isolated execution environments for AI agents with configurable access levels.

NOTE: SandboxIntegration, SandboxContext, SandboxAgentMixin are in
      llmcore.agents.sandbox_integration, not this package.
"""

from .base import (
    ExecutionResult,
    FileInfo,
    SandboxAccessLevel,
    SandboxConfig,
    SandboxProvider,
    SandboxStatus,
)
from .config import (
    create_registry_config,
    load_sandbox_config,
)
from .docker_provider import DockerSandboxProvider
from .ephemeral import EphemeralResourceManager
from .exceptions import (
    SandboxAccessDenied,
    SandboxCleanupError,
    SandboxConnectionError,
    SandboxError,
    SandboxExecutionError,
    SandboxImageNotFoundError,
    SandboxInitializationError,
    SandboxNotInitializedError,
    SandboxResourceError,
    SandboxTimeoutError,
)
from .output_tracker import OutputTracker
from .registry import (
    SandboxMode,
    SandboxRegistry,
    SandboxRegistryConfig,
)
from .tools import (
    SANDBOX_TOOL_IMPLEMENTATIONS,
    SANDBOX_TOOL_SCHEMAS,
    clear_active_sandbox,
    execute_python,
    execute_shell,
    get_active_sandbox,
    get_sandbox_info,
    get_state,
    list_files,
    load_file,
    save_file,
    set_active_sandbox,
    set_state,
)
from .vm_provider import VMSandboxProvider

__all__ = [
    "SandboxProvider",
    "SandboxConfig",
    "SandboxAccessLevel",
    "SandboxStatus",
    "ExecutionResult",
    "FileInfo",
    "SandboxError",
    "SandboxInitializationError",
    "SandboxExecutionError",
    "SandboxTimeoutError",
    "SandboxAccessDenied",
    "SandboxResourceError",
    "SandboxConnectionError",
    "SandboxCleanupError",
    "SandboxNotInitializedError",
    "SandboxImageNotFoundError",
    "DockerSandboxProvider",
    "VMSandboxProvider",
    "SandboxRegistry",
    "SandboxRegistryConfig",
    "SandboxMode",
    "EphemeralResourceManager",
    "OutputTracker",
    "load_sandbox_config",
    "create_registry_config",
    "set_active_sandbox",
    "clear_active_sandbox",
    "get_active_sandbox",
    "SANDBOX_TOOL_IMPLEMENTATIONS",
    "SANDBOX_TOOL_SCHEMAS",
    "execute_shell",
    "execute_python",
    "save_file",
    "load_file",
    "list_files",
    "get_state",
    "set_state",
    "get_sandbox_info",
]
