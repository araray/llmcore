# src/llmcore/agents/sandbox/__init__.py
"""
Sandbox System for LLMCore Agent Execution.

This package provides isolated execution environments for AI agents,
enabling safe code execution with configurable access levels and
comprehensive output tracking.

Main Components:
    - SandboxRegistry: Central management and routing
    - DockerSandboxProvider: Docker container-based isolation
    - VMSandboxProvider: SSH-based VM isolation
    - EphemeralResourceManager: Temporary resource handling
    - OutputTracker: Execution output capture and storage

Access Levels:
    - RESTRICTED: Limited tool access, network disabled
    - FULL: Complete access for whitelisted images/VMs

Usage:
    >>> from llmcore.agents.sandbox import (
    ...     SandboxRegistry,
    ...     SandboxRegistryConfig,
    ...     SandboxMode,
    ...     SandboxConfig,
    ... )
    >>>
    >>> config = SandboxRegistryConfig(mode=SandboxMode.DOCKER)
    >>> registry = SandboxRegistry(config)
    >>> sandbox = await registry.create_sandbox(SandboxConfig())
    >>> result = await sandbox.execute_shell("echo 'Hello'")
    >>> await registry.cleanup_sandbox(sandbox.get_config().sandbox_id)

NOTE: SandboxIntegration, SandboxContext, SandboxAgentMixin, and related
      integration utilities are in llmcore.agents.sandbox_integration, NOT
      in this package. Import them from llmcore.agents instead.

References:
    - Technical Spec: Section 4 (Sandbox System)
    - Implementation Dossiers: Steps 1.1-1.14
"""

# =============================================================================
# BASE CLASSES AND DATA MODELS
# =============================================================================

from .base import (
    SandboxProvider,
    SandboxConfig,
    SandboxAccessLevel,
    SandboxStatus,
    ExecutionResult,
    FileInfo,
)

# =============================================================================
# EXCEPTIONS
# =============================================================================

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

# =============================================================================
# PROVIDERS
# =============================================================================

from .docker_provider import DockerSandboxProvider
from .vm_provider import VMSandboxProvider

# =============================================================================
# REGISTRY
# =============================================================================

from .registry import (
    SandboxRegistry,
    SandboxRegistryConfig,
    SandboxMode,
)

# =============================================================================
# UTILITIES
# =============================================================================

from .ephemeral import EphemeralResourceManager
from .output_tracker import OutputTracker

# =============================================================================
# CONFIGURATION HELPERS
# =============================================================================

from .config import (
    load_sandbox_config,
    create_registry_config,
)

# =============================================================================
# TOOLS
# =============================================================================

from .tools import (
    set_active_sandbox,
    clear_active_sandbox,
    get_active_sandbox,
    SANDBOX_TOOL_IMPLEMENTATIONS,
    SANDBOX_TOOL_SCHEMAS,
)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Base classes
    "SandboxProvider",
    "SandboxConfig",
    "SandboxAccessLevel",
    "SandboxStatus",
    "ExecutionResult",
    "FileInfo",
    # Exceptions
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
    # Providers
    "DockerSandboxProvider",
    "VMSandboxProvider",
    # Registry
    "SandboxRegistry",
    "SandboxRegistryConfig",
    "SandboxMode",
    # Utilities
    "EphemeralResourceManager",
    "OutputTracker",
    # Configuration
    "load_sandbox_config",
    "create_registry_config",
    # Tools
    "set_active_sandbox",
    "clear_active_sandbox",
    "get_active_sandbox",
    "SANDBOX_TOOL_IMPLEMENTATIONS",
    "SANDBOX_TOOL_SCHEMAS",
]
