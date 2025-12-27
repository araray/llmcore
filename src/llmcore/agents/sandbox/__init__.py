# src/llmcore/agents/sandbox/__init__.py
"""
Sandbox System for LLMCore Agent Execution.

Isolated execution environments for AI agents with configurable access levels.

NOTE: SandboxIntegration, SandboxContext, SandboxAgentMixin are in
      llmcore.agents.sandbox_integration, not this package.
"""

from .base import (
    SandboxProvider,
    SandboxConfig,
    SandboxAccessLevel,
    SandboxStatus,
    ExecutionResult,
    FileInfo,
)

from .exceptions import (
    SandboxError,
    SandboxInitializationError,
    SandboxExecutionError,
    SandboxTimeoutError,
    SandboxAccessDenied,
    SandboxResourceError,
    SandboxConnectionError,
    SandboxCleanupError,
    SandboxNotInitializedError,
    SandboxImageNotFoundError,
)

from .docker_provider import DockerSandboxProvider
from .vm_provider import VMSandboxProvider

from .registry import (
    SandboxRegistry,
    SandboxRegistryConfig,
    SandboxMode,
)

from .ephemeral import EphemeralResourceManager
from .output_tracker import OutputTracker

from .config import (
    load_sandbox_config,
    create_registry_config,
)

from .tools import (
    set_active_sandbox,
    clear_active_sandbox,
    get_active_sandbox,
    SANDBOX_TOOL_IMPLEMENTATIONS,
    SANDBOX_TOOL_SCHEMAS,
)

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
]
