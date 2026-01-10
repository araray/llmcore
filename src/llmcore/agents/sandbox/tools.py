# src/llmcore/agents/sandbox/tools.py
"""
Sandbox-aware tools for agent execution.

This module provides the implementation of tools that agents can use
to interact with their sandbox environment. All tools delegate to the
active sandbox, ensuring code NEVER runs on the host system.

Tool Categories:
    - Execution: execute_shell, execute_python
    - File Operations: save_file, load_file, replace_in_file, list_files
    - State Management: get_state, set_state
    - Information: get_sandbox_info

Usage:
    These tools are registered in the ToolManager and called by the
    agent during the Act phase of the cognitive cycle.

    # Setup active sandbox (done by AgentManager)
    >>> set_active_sandbox(sandbox, registry)
    >>>
    >>> # Tools can now be called
    >>> result = await execute_shell("echo 'Hello'")
    >>>
    >>> # Cleanup
    >>> clear_active_sandbox()

Thread Safety:
    The active sandbox is stored in module-level state. This module
    is designed for single-threaded async use. For multi-threaded
    scenarios, use contextvars or pass sandbox explicitly.
"""

import json
import logging
from typing import Optional, Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import SandboxProvider
    from .registry import SandboxRegistry
    from .ephemeral import EphemeralResourceManager

logger = logging.getLogger(__name__)

# Module-level state for active sandbox
# These are set by the AgentManager when running an agent
_active_sandbox: Optional["SandboxProvider"] = None
_sandbox_registry: Optional["SandboxRegistry"] = None
_ephemeral_manager: Optional["EphemeralResourceManager"] = None


def set_active_sandbox(
    sandbox: "SandboxProvider",
    registry: "SandboxRegistry"
) -> None:
    """
    Set the active sandbox for tool execution.

    This should be called by AgentManager before running an agent.

    Args:
        sandbox: The sandbox provider to use
        registry: The sandbox registry for policy enforcement
    """
    global _active_sandbox, _sandbox_registry, _ephemeral_manager

    _active_sandbox = sandbox
    _sandbox_registry = registry

    # Create ephemeral resource manager
    from .ephemeral import EphemeralResourceManager
    _ephemeral_manager = EphemeralResourceManager(sandbox)

    logger.debug(f"Active sandbox set: {sandbox.get_info().get('provider')}")


def clear_active_sandbox() -> None:
    """
    Clear the active sandbox reference.

    This should be called after agent execution completes.
    """
    global _active_sandbox, _sandbox_registry, _ephemeral_manager

    _active_sandbox = None
    _sandbox_registry = None
    _ephemeral_manager = None

    logger.debug("Active sandbox cleared")


def get_active_sandbox() -> Optional["SandboxProvider"]:
    """
    Get the currently active sandbox.

    Returns:
        Active SandboxProvider or None
    """
    return _active_sandbox


def _check_tool_access(tool_name: str) -> Optional[str]:
    """
    Check if a tool is allowed in the current sandbox.

    Args:
        tool_name: Name of the tool

    Returns:
        Error message if denied, None if allowed
    """
    if not _active_sandbox:
        return "No active sandbox. Code execution is not available."

    if _sandbox_registry:
        access_level = _active_sandbox.get_access_level()
        if not _sandbox_registry.is_tool_allowed(tool_name, access_level):
            return f"Tool '{tool_name}' is not allowed in {access_level.value} sandbox mode."

    return None


# =============================================================================
# EXECUTION TOOLS
# =============================================================================

async def execute_shell(
    command: str,
    timeout: Optional[int] = None,
    working_dir: Optional[str] = None
) -> str:
    """
    Execute a shell command in the sandbox.

    This tool runs shell commands in an isolated environment. The sandbox
    has its own filesystem, and depending on access level, may or may not
    have network access.

    Args:
        command: Shell command to execute (runs via bash -c)
        timeout: Optional timeout in seconds (default from config)
        working_dir: Optional working directory

    Returns:
        Command output (stdout/stderr combined with exit code info)

    Examples:
        >>> await execute_shell("ls -la")
        'total 0\ndrwxr-xr-x  2 root root ...'

        >>> await execute_shell("python --version")
        'Python 3.11.4'

        >>> await execute_shell("echo $PATH")
        '/usr/local/bin:/usr/bin:/bin'
    """
    error = _check_tool_access("execute_shell")
    if error:
        return f"ERROR: {error}"

    result = await _active_sandbox.execute_shell(command, timeout, working_dir)
    return result.to_tool_output()


async def execute_python(
    code: str,
    timeout: Optional[int] = None
) -> str:
    """
    Execute Python code in the sandbox.

    This tool writes the provided code to a temporary file and executes it
    using the Python interpreter in the sandbox.

    Args:
        code: Python code to execute
        timeout: Optional timeout in seconds

    Returns:
        Execution output (stdout/stderr with exit code info)

    Examples:
        >>> await execute_python("print('Hello, World!')")
        'Hello, World!'

        >>> await execute_python('''
        ... import sys
        ... print(f"Python version: {sys.version}")
        ... ''')
        'Python version: 3.11.4 ...'

        >>> await execute_python("x = 1/0")
        '❌ EXIT CODE: 1\nSTDERR:\nTraceback ...'
    """
    error = _check_tool_access("execute_python")
    if error:
        return f"ERROR: {error}"

    result = await _active_sandbox.execute_python(code, timeout)
    return result.to_tool_output()


# =============================================================================
# FILE OPERATION TOOLS
# =============================================================================

async def save_file(
    path: str,
    content: str,
    description: Optional[str] = None
) -> str:
    """
    Save content to a file in the sandbox.

    Creates parent directories if they don't exist. Files saved to the
    output directory will be preserved after sandbox cleanup.

    Args:
        path: File path (relative to workspace or absolute)
        content: Content to write to the file
        description: Optional description for tracking

    Returns:
        Success message or error

    Examples:
        >>> await save_file("script.py", "print('hello')")
        'Successfully saved file: script.py'

        >>> await save_file("output/result.txt", "Analysis complete")
        'Successfully saved file: output/result.txt'
    """
    error = _check_tool_access("save_file")
    if error:
        return f"ERROR: {error}"

    success = await _active_sandbox.write_file(path, content)

    if success:
        # Record in ephemeral database
        if _ephemeral_manager:
            await _ephemeral_manager.record_file(
                path,
                len(content.encode('utf-8')),
                description or ""
            )
        return f"Successfully saved file: {path}"

    return f"ERROR: Failed to save file: {path}"


async def load_file(path: str) -> str:
    """
    Load content from a file in the sandbox.

    Args:
        path: File path (relative to workspace or absolute)

    Returns:
        File content or error message

    Examples:
        >>> await load_file("config.json")
        '{"setting": "value"}'

        >>> await load_file("nonexistent.txt")
        'ERROR: File not found: nonexistent.txt'
    """
    error = _check_tool_access("load_file")
    if error:
        return f"ERROR: {error}"

    content = await _active_sandbox.read_file(path)

    if content is not None:
        return content

    return f"ERROR: File not found or could not be read: {path}"


async def replace_in_file(
    path: str,
    old_value: str,
    new_value: str
) -> str:
    """
    Find and replace text in a file in the sandbox.

    Reads the file, performs the replacement, and writes it back.

    Args:
        path: File path
        old_value: Text to find
        new_value: Text to replace with

    Returns:
        Success message or error

    Examples:
        >>> await replace_in_file("config.py", "DEBUG = False", "DEBUG = True")
        "Successfully replaced 'DEBUG = False' with 'DEBUG = True' in config.py"

        >>> await replace_in_file("config.py", "NONEXISTENT", "value")
        "ERROR: 'NONEXISTENT' not found in file"
    """
    error = _check_tool_access("replace_in_file")
    if error:
        return f"ERROR: {error}"

    content = await _active_sandbox.read_file(path)

    if content is None:
        return f"ERROR: File not found: {path}"

    if old_value not in content:
        return f"ERROR: '{old_value}' not found in file"

    new_content = content.replace(old_value, new_value)
    success = await _active_sandbox.write_file(path, new_content)

    if success:
        return f"Successfully replaced '{old_value}' with '{new_value}' in {path}"

    return f"ERROR: Failed to write updated file: {path}"


async def append_to_file(
    path: str,
    content: str
) -> str:
    """
    Append content to a file in the sandbox.

    If the file doesn't exist, it will be created.

    Args:
        path: File path
        content: Content to append

    Returns:
        Success message or error
    """
    error = _check_tool_access("save_file")  # Same permission as save
    if error:
        return f"ERROR: {error}"

    success = await _active_sandbox.write_file(path, content, mode="a")

    if success:
        return f"Successfully appended to file: {path}"

    return f"ERROR: Failed to append to file: {path}"


async def list_files(
    path: str = ".",
    recursive: bool = False
) -> str:
    """
    List files in a directory in the sandbox.

    Args:
        path: Directory path (default: current workspace)
        recursive: If True, list files recursively

    Returns:
        Formatted list of files or error

    Examples:
        >>> await list_files()
        'script.py\nconfig.json\noutput/'

        >>> await list_files("output", recursive=True)
        'output/result.txt\noutput/data/processed.csv'
    """
    error = _check_tool_access("list_files")
    if error:
        return f"ERROR: {error}"

    files = await _active_sandbox.list_files(path, recursive)

    if not files:
        return "(empty directory or path not found)"

    # Format output
    lines = []
    for f in files:
        suffix = "/" if f.is_directory else ""
        size_str = f" ({f.size_bytes} bytes)" if f.size_bytes and not f.is_directory else ""
        lines.append(f"{f.name}{suffix}{size_str}")

    return "\n".join(lines)


async def file_exists(path: str) -> str:
    """
    Check if a file or directory exists in the sandbox.

    Args:
        path: Path to check

    Returns:
        "true" or "false"
    """
    error = _check_tool_access("file_exists")
    if error:
        return f"ERROR: {error}"

    exists = await _active_sandbox.file_exists(path)
    return "true" if exists else "false"


async def delete_file(path: str) -> str:
    """
    Delete a file in the sandbox.

    Args:
        path: File path to delete

    Returns:
        Success message or error
    """
    error = _check_tool_access("delete_file")
    if error:
        return f"ERROR: {error}"

    success = await _active_sandbox.delete_file(path)

    if success:
        return f"Successfully deleted: {path}"

    return f"ERROR: Failed to delete file: {path}"


async def create_directory(path: str) -> str:
    """
    Create a directory in the sandbox.

    Creates parent directories if needed.

    Args:
        path: Directory path to create

    Returns:
        Success message or error
    """
    error = _check_tool_access("create_directory")
    if error:
        return f"ERROR: {error}"

    success = await _active_sandbox.create_directory(path)

    if success:
        return f"Successfully created directory: {path}"

    return f"ERROR: Failed to create directory: {path}"


# =============================================================================
# STATE MANAGEMENT TOOLS
# =============================================================================

async def get_state(key: str) -> str:
    """
    Get a value from the agent's ephemeral state.

    The state is stored in an ephemeral SQLite database that persists
    across iterations but is destroyed when the sandbox is cleaned up.

    Args:
        key: State key to retrieve

    Returns:
        The value as a string, or "(not set)" if key doesn't exist
    """
    if not _ephemeral_manager:
        return "ERROR: State management not available"

    value = await _ephemeral_manager.get_state(key)

    if value is None:
        return "(not set)"

    if isinstance(value, (dict, list)):
        return json.dumps(value)

    return str(value)


async def set_state(key: str, value: str) -> str:
    """
    Store a value in the agent's ephemeral state.

    Use this to remember information across iterations.

    Args:
        key: State key
        value: Value to store

    Returns:
        Success message or error
    """
    if not _ephemeral_manager:
        return "ERROR: State management not available"

    # Try to parse as JSON for complex values
    try:
        parsed_value = json.loads(value)
        success = await _ephemeral_manager.set_state(key, parsed_value)
    except json.JSONDecodeError:
        success = await _ephemeral_manager.set_state(key, value)

    if success:
        return f"State '{key}' updated"

    return f"ERROR: Failed to update state '{key}'"


async def list_state() -> str:
    """
    List all keys in the agent's ephemeral state.

    Returns:
        Newline-separated list of keys
    """
    if not _ephemeral_manager:
        return "ERROR: State management not available"

    keys = await _ephemeral_manager.list_state_keys()

    if not keys:
        return "(no state stored)"

    return "\n".join(keys)


# =============================================================================
# INFORMATION TOOLS
# =============================================================================

async def get_sandbox_info() -> str:
    """
    Get information about the current sandbox environment.

    Returns details about the sandbox type, access level, and capabilities.

    Returns:
        Formatted sandbox information
    """
    if not _active_sandbox:
        return "No active sandbox"

    info = _active_sandbox.get_info()
    healthy = await _active_sandbox.is_healthy()

    access_level = _active_sandbox.get_access_level()

    lines = [
        f"Provider: {info.get('provider', 'unknown')}",
        f"Status: {'healthy' if healthy else 'unhealthy'}",
        f"Access Level: {access_level.value}",
    ]

    if access_level.value == "full":
        lines.extend([
            "Network: enabled",
            "Package Installation: allowed",
            "Tool Restrictions: none"
        ])
    else:
        lines.extend([
            "Network: disabled",
            "Package Installation: restricted",
            "Tool Restrictions: whitelist enforced"
        ])

    if 'container_id' in info:
        lines.append(f"Container ID: {info['container_id']}")
    if 'host' in info:
        lines.append(f"VM Host: {info['host']}")
    if 'workspace' in info:
        lines.append(f"Workspace: {info['workspace']}")
    if 'working_directory' in info:
        lines.append(f"Working Directory: {info['working_directory']}")

    return "\n".join(lines)


async def get_recorded_files() -> str:
    """
    Get list of files recorded as created by the agent.

    Returns:
        Formatted list of recorded files
    """
    if not _ephemeral_manager:
        return "ERROR: File tracking not available"

    files = await _ephemeral_manager.list_recorded_files()

    if not files:
        return "(no files recorded)"

    lines = []
    for f in files:
        desc = f" - {f.description}" if f.description else ""
        lines.append(f"{f.path} ({f.size_bytes} bytes){desc}")

    return "\n".join(lines)


# =============================================================================
# TOOL REGISTRY
# =============================================================================

# Implementation registry for the ToolManager
SANDBOX_TOOL_IMPLEMENTATIONS = {
    "llmcore.tools.sandbox.execute_shell": execute_shell,
    "llmcore.tools.sandbox.execute_python": execute_python,
    "llmcore.tools.sandbox.save_file": save_file,
    "llmcore.tools.sandbox.load_file": load_file,
    "llmcore.tools.sandbox.replace_in_file": replace_in_file,
    "llmcore.tools.sandbox.append_to_file": append_to_file,
    "llmcore.tools.sandbox.list_files": list_files,
    "llmcore.tools.sandbox.file_exists": file_exists,
    "llmcore.tools.sandbox.delete_file": delete_file,
    "llmcore.tools.sandbox.create_directory": create_directory,
    "llmcore.tools.sandbox.get_state": get_state,
    "llmcore.tools.sandbox.set_state": set_state,
    "llmcore.tools.sandbox.list_state": list_state,
    "llmcore.tools.sandbox.get_sandbox_info": get_sandbox_info,
    "llmcore.tools.sandbox.get_recorded_files": get_recorded_files,
}

# Human-readable descriptions for the implementation keys
SANDBOX_TOOL_DESCRIPTIONS = {
    "llmcore.tools.sandbox.execute_shell": "Execute shell commands in the isolated sandbox environment",
    "llmcore.tools.sandbox.execute_python": "Execute Python code in the isolated sandbox environment",
    "llmcore.tools.sandbox.save_file": "Save content to a file in the sandbox workspace",
    "llmcore.tools.sandbox.load_file": "Load content from a file in the sandbox workspace",
    "llmcore.tools.sandbox.replace_in_file": "Find and replace text in a file in the sandbox",
    "llmcore.tools.sandbox.append_to_file": "Append content to a file in the sandbox",
    "llmcore.tools.sandbox.list_files": "List files in a directory in the sandbox",
    "llmcore.tools.sandbox.file_exists": "Check if a file or directory exists in the sandbox",
    "llmcore.tools.sandbox.delete_file": "Delete a file in the sandbox",
    "llmcore.tools.sandbox.create_directory": "Create a directory in the sandbox",
    "llmcore.tools.sandbox.get_state": "Get a value from ephemeral agent state",
    "llmcore.tools.sandbox.set_state": "Store a value in ephemeral agent state",
    "llmcore.tools.sandbox.list_state": "List all keys in ephemeral agent state",
    "llmcore.tools.sandbox.get_sandbox_info": "Get information about the current sandbox environment",
    "llmcore.tools.sandbox.get_recorded_files": "Get list of files recorded as created by the agent",
}

# Tool parameter schemas (for OpenAI-style function calling)
SANDBOX_TOOL_SCHEMAS = {
    "execute_shell": {
        "type": "function",
        "function": {
            "name": "execute_shell",
            "description": "Execute a shell command in the isolated sandbox environment",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Optional timeout in seconds"
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Optional working directory"
                    }
                },
                "required": ["command"]
            }
        }
    },
    "execute_python": {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Execute Python code in the isolated sandbox environment",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Optional timeout in seconds"
                    }
                },
                "required": ["code"]
            }
        }
    },
        "append_to_file": {
        "type": "function",
        "function": {
            "name": "append_to_file",
            "description": "Append content to an existing file in the sandbox",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to append to"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to append"
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    "save_file": {
        "type": "function",
        "function": {
            "name": "save_file",
            "description": "Save content to a file in the sandbox workspace",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path (relative to workspace or absolute)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional description for tracking"
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    "load_file": {
        "type": "function",
        "function": {
            "name": "load_file",
            "description": "Load content from a file in the sandbox workspace",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to read"
                    }
                },
                "required": ["path"]
            }
        }
    },
    "replace_in_file": {
        "type": "function",
        "function": {
            "name": "replace_in_file",
            "description": "Find and replace text in a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path"
                    },
                    "old_value": {
                        "type": "string",
                        "description": "Text to find"
                    },
                    "new_value": {
                        "type": "string",
                        "description": "Text to replace with"
                    }
                },
                "required": ["path", "old_value", "new_value"]
            }
        }
    },
    "list_files": {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path (default: current workspace)"
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "If true, list files recursively"
                    }
                },
                "required": []
            }
        }
    },
    "file_exists": {
        "type": "function",
        "function": {
            "name": "file_exists",
            "description": "Check if a file or directory exists in the sandbox",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to check"
                    }
                },
                "required": ["path"]
            }
        }
    },
    "get_sandbox_info": {
        "type": "function",
        "function": {
            "name": "get_sandbox_info",
            "description": "Get information about the current sandbox environment",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
}
