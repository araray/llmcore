# src/llmcore/agents/tools.py
"""
Tool Management for LLMCore Agents.

This module provides the ToolManager which handles the registration, validation,
and execution of tools available to agents. Tools are the actions that agents
can take to interact with the external world.

UPDATED: Now supports dynamic registration of sandbox tools for isolated
code execution. Sandbox tools run inside Docker/VM containers.

Security Model:
    - Only functions in _IMPLEMENTATION_REGISTRY can be executed
    - Tool definitions are loaded from database per-tenant
    - Implementation keys map tool names to actual functions
    - Sandbox tools execute in isolated environments (when sandbox is active)
"""

import asyncio
import logging
import math
from typing import Dict, List, Optional
from collections.abc import Callable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..exceptions import LLMCoreError
from ..memory.manager import MemoryManager
from ..models import Tool, ToolCall, ToolResult
from ..storage.manager import StorageManager

logger = logging.getLogger(__name__)


# =============================================================================
# BUILT-IN TOOL IMPLEMENTATIONS
# =============================================================================


async def semantic_search(
    query: str, memory_manager: MemoryManager, top_k: int = 5, collection_name: str | None = None
) -> str:
    """
    Search the knowledge base (vector store) for relevant information.

    Args:
        query: The search query
        memory_manager: MemoryManager instance for retrieval
        top_k: Number of results to return
        collection_name: Optional collection to search

    Returns:
        Formatted search results as a string
    """
    logger.debug(f"Semantic search: {query}")

    try:
        results = await memory_manager.search_semantic(
            query=query, top_k=top_k, collection_name=collection_name
        )

        if not results:
            return "No relevant results found."

        formatted = []
        for i, result in enumerate(results, 1):
            content = result.get("content", "")[:500]
            score = result.get("score", 0)
            formatted.append(f"[{i}] (score: {score:.2f}) {content}")

        return "\n\n".join(formatted)

    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return f"Search error: {e!s}"


async def episodic_search(
    query: str, storage_manager: StorageManager, session_id: str | None = None, limit: int = 10
) -> str:
    """
    Search past experiences and interactions in episodic memory.

    Args:
        query: The search query
        storage_manager: StorageManager instance
        session_id: Optional session to filter by
        limit: Maximum results to return

    Returns:
        Formatted episodic memories as a string
    """
    logger.debug(f"Episodic search: {query}")

    try:
        results = await storage_manager.search_episodes(
            query=query, session_id=session_id, limit=limit
        )

        if not results:
            return "No relevant past experiences found."

        formatted = []
        for i, episode in enumerate(results, 1):
            summary = episode.get("summary", "")[:300]
            timestamp = episode.get("timestamp", "unknown")
            formatted.append(f"[{i}] ({timestamp}) {summary}")

        return "\n\n".join(formatted)

    except Exception as e:
        logger.error(f"Episodic search failed: {e}")
        return f"Search error: {e!s}"


def calculator(expression: str) -> str:
    """
    Perform mathematical calculations safely.

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        Result of the calculation as a string
    """
    logger.debug(f"Calculator: {expression}")

    # Safe math functions
    safe_dict = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
    }

    try:
        # Remove any potentially dangerous characters
        clean_expr = expression.replace("__", "").replace("import", "")
        result = eval(clean_expr, {"__builtins__": {}}, safe_dict)
        return str(result)
    except Exception as e:
        return f"Calculation error: {e!s}"


def finish(answer: str) -> str:
    """
    Complete the agent task with a final answer.

    This tool signals that the agent has completed its task and provides
    the final result.

    Args:
        answer: The final answer/result to return

    Returns:
        The answer (passthrough)
    """
    logger.info(f"Agent finishing with answer: {answer[:100]}...")
    return answer


def human_approval(prompt: str, pending_action: str) -> str:
    """
    Request human approval before executing sensitive or irreversible actions.

    This tool does not perform any external action. Instead, it acts as a signal
    to the AgentManager to pause execution and request human approval.

    Args:
        prompt: The question/request to present to the human operator
        pending_action: JSON representation of the tool call that needs approval

    Returns:
        A placeholder message indicating approval is being requested
    """
    logger.info(f"Human approval requested: {prompt}")
    logger.debug(f"Pending action for approval: {pending_action}")
    return "Pausing for human approval."


# =============================================================================
# SECURE IMPLEMENTATION REGISTRY
# =============================================================================

# This is the security boundary - only functions registered here can be executed
_IMPLEMENTATION_REGISTRY: dict[str, Callable] = {
    # Core search tools
    "llmcore.tools.search.semantic": semantic_search,
    "llmcore.tools.search.episodic": episodic_search,
    # Calculation tools
    "llmcore.tools.calculation.calculator": calculator,
    # Flow control tools
    "llmcore.tools.flow.finish": finish,
    "llmcore.tools.flow.human_approval": human_approval,
    # NOTE: Sandbox tools are registered dynamically via register_sandbox_tools()
    # This keeps the sandbox system optional and modular
}

# Human-readable descriptions for the implementation keys
_IMPLEMENTATION_DESCRIPTIONS: dict[str, str] = {
    "llmcore.tools.search.semantic": "Search the knowledge base (vector store) for relevant information",
    "llmcore.tools.search.episodic": "Search past experiences and interactions in episodic memory",
    "llmcore.tools.calculation.calculator": "Perform mathematical calculations safely",
    "llmcore.tools.flow.finish": "Complete the agent task with a final answer",
    "llmcore.tools.flow.human_approval": "Request human approval before executing sensitive actions",
}


def register_implementation(key: str, func: Callable, description: str = "") -> None:
    """
    Register a new tool implementation.

    This is the approved way to add new tools to the registry.

    Args:
        key: Unique implementation key (e.g., "llmcore.tools.mypackage.mytool")
        func: The function to execute
        description: Human-readable description

    Raises:
        ValueError: If key already exists
    """
    if key in _IMPLEMENTATION_REGISTRY:
        raise ValueError(f"Implementation key '{key}' already registered")

    _IMPLEMENTATION_REGISTRY[key] = func
    if description:
        _IMPLEMENTATION_DESCRIPTIONS[key] = description

    logger.debug(f"Registered tool implementation: {key}")


def register_implementations(implementations: dict[str, Callable]) -> None:
    """
    Register multiple tool implementations at once.

    Args:
        implementations: Dict mapping keys to functions
    """
    for key, func in implementations.items():
        if key not in _IMPLEMENTATION_REGISTRY:
            _IMPLEMENTATION_REGISTRY[key] = func
            logger.debug(f"Registered tool implementation: {key}")


def get_registered_implementations() -> list[str]:
    """Get list of all registered implementation keys."""
    return list(_IMPLEMENTATION_REGISTRY.keys())


# =============================================================================
# TOOLMANAGER CLASS
# =============================================================================


class ToolManager:
    """
    Manages the registration, validation, and execution of tools available to agents.

    UPDATED: Now supports dynamic tool loading from database with secure implementation
    registry. Tools are loaded per-tenant and per-run rather than globally at startup.

    Also supports sandbox tools when sandbox integration is active.

    Attributes:
        _memory_manager: MemoryManager for memory-related tools
        _storage_manager: StorageManager for storage-related tools
        _tool_definitions: List of loaded tool definitions
        _implementation_map: Maps tool names to implementation keys
    """

    def __init__(self, memory_manager: MemoryManager, storage_manager: StorageManager):
        """
        Initialize the ToolManager with required dependencies.

        Args:
            memory_manager: The MemoryManager instance for memory-related tools.
            storage_manager: The StorageManager instance for storage-related tools.
        """
        self._memory_manager = memory_manager
        self._storage_manager = storage_manager

        # These will be populated dynamically per run
        self._tool_definitions: list[Tool] = []
        self._implementation_map: dict[str, str] = {}  # tool_name -> implementation_key

        logger.info("ToolManager initialized for dynamic tool loading")

    async def load_tools_for_run(
        self, db_session: AsyncSession, enabled_toolkits: list[str] | None = None
    ) -> None:
        """
        Load tool definitions from the database for a specific tenant and toolkits.

        This method queries the tenant's database schema to load the tools that are
        available for the current agent run. It supports filtering by toolkit names.

        Args:
            db_session: Tenant-scoped database session
            enabled_toolkits: List of toolkit names to enable (None = all tools)

        Raises:
            LLMCoreError: If database query fails or invalid implementation keys found
        """
        try:
            logger.debug(f"Loading tools for run with toolkits: {enabled_toolkits}")

            # Clear any existing tool definitions
            self._tool_definitions.clear()
            self._implementation_map.clear()

            # Build query based on whether toolkits are specified
            if enabled_toolkits:
                # Load tools from specific toolkits
                placeholders = ", ".join([f":toolkit_{i}" for i in range(len(enabled_toolkits))])
                query = text(f"""
                    SELECT DISTINCT t.name, t.description, t.parameters_schema, t.implementation_key
                    FROM tools t
                    JOIN toolkit_tools tt ON t.name = tt.tool_name
                    WHERE t.is_enabled = TRUE
                    AND tt.toolkit_name IN ({placeholders})
                    ORDER BY t.name
                """)
                params = {f"toolkit_{i}": name for i, name in enumerate(enabled_toolkits)}
            else:
                # Load all enabled tools
                query = text("""
                    SELECT name, description, parameters_schema, implementation_key
                    FROM tools
                    WHERE is_enabled = TRUE
                    ORDER BY name
                """)
                params = {}

            # Execute query and process results
            result = await db_session.execute(query, params)
            rows = result.fetchall()

            for row in rows:
                tool_name = row.name
                implementation_key = row.implementation_key

                # Security check: ensure implementation key exists in secure registry
                if implementation_key not in _IMPLEMENTATION_REGISTRY:
                    logger.error(
                        f"Invalid implementation key '{implementation_key}' for tool '{tool_name}'"
                    )
                    raise LLMCoreError(
                        f"Tool '{tool_name}' has invalid implementation key: {implementation_key}"
                    )

                # Create Tool model from database data
                tool = Tool(
                    name=tool_name, description=row.description, parameters=row.parameters_schema
                )

                # Store the mappings
                self._tool_definitions.append(tool)
                self._implementation_map[tool_name] = implementation_key

            logger.info(f"Loaded {len(self._tool_definitions)} tools for agent run")

        except Exception as e:
            logger.error(f"Error loading tools for run: {e}", exc_info=True)
            raise LLMCoreError(f"Failed to load tools: {e!s}")

    def load_default_tools(self) -> None:
        """
        Load default built-in tools without database.

        Useful for testing or when running without database.
        """
        self._tool_definitions.clear()
        self._implementation_map.clear()

        default_tools = [
            (
                "semantic_search",
                "Search the knowledge base for relevant information",
                "llmcore.tools.search.semantic",
            ),
            (
                "episodic_search",
                "Search past experiences in episodic memory",
                "llmcore.tools.search.episodic",
            ),
            (
                "calculator",
                "Perform mathematical calculations",
                "llmcore.tools.calculation.calculator",
            ),
            ("finish", "Complete the task with a final answer", "llmcore.tools.flow.finish"),
            (
                "human_approval",
                "Request human approval for sensitive actions",
                "llmcore.tools.flow.human_approval",
            ),
        ]

        for name, desc, impl_key in default_tools:
            if impl_key in _IMPLEMENTATION_REGISTRY:
                tool = Tool(name=name, description=desc, parameters={})
                self._tool_definitions.append(tool)
                self._implementation_map[name] = impl_key

        logger.info(f"Loaded {len(self._tool_definitions)} default tools")

    def get_tool_definitions(self) -> list[Tool]:
        """
        Get all loaded tool definitions for the current run.

        Returns:
            List of Tool definitions available for agent use.
        """
        return self._tool_definitions.copy()

    async def execute_tool(
        self, tool_call: ToolCall, session_id: str | None = None
    ) -> ToolResult:
        """
        Execute a tool call and return the result.

        Args:
            tool_call: The ToolCall object containing the tool name and arguments.
            session_id: Optional session ID for tools that need session context.

        Returns:
            ToolResult containing the execution result.

        Raises:
            LLMCoreError: If the tool is not found or execution fails.
        """
        tool_name = tool_call.name

        if tool_name not in self._implementation_map:
            available_tools = list(self._implementation_map.keys())
            error_msg = (
                f"Tool '{tool_name}' not loaded for this run. Available tools: {available_tools}"
            )
            logger.error(error_msg)
            return ToolResult(tool_call_id=tool_call.id, content=f"ERROR: {error_msg}")

        try:
            # Get the implementation key and function
            implementation_key = self._implementation_map[tool_name]
            tool_func = _IMPLEMENTATION_REGISTRY[implementation_key]

            arguments = tool_call.arguments.copy()

            # Inject dependencies based on tool function signature
            import inspect

            sig = inspect.signature(tool_func)

            # Inject memory_manager if the tool needs it
            if "memory_manager" in sig.parameters:
                arguments["memory_manager"] = self._memory_manager

            # Inject storage_manager if the tool needs it
            if "storage_manager" in sig.parameters:
                arguments["storage_manager"] = self._storage_manager

            # Inject session_id if the tool needs it and we have one
            if "session_id" in sig.parameters and session_id:
                arguments["session_id"] = session_id

            logger.debug(
                f"Executing tool '{tool_name}' (key: {implementation_key}) with arguments: {list(arguments.keys())}"
            )

            # Execute the tool function
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(**arguments)
            else:
                result = tool_func(**arguments)

            logger.debug(f"Tool '{tool_name}' executed successfully")

            return ToolResult(tool_call_id=tool_call.id, content=str(result))

        except TypeError as e:
            error_msg = f"Invalid arguments for tool '{tool_name}': {e!s}"
            logger.error(error_msg, exc_info=True)
            return ToolResult(tool_call_id=tool_call.id, content=f"ERROR: {error_msg}")
        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {e!s}"
            logger.error(error_msg, exc_info=True)
            return ToolResult(tool_call_id=tool_call.id, content=f"ERROR: {error_msg}")

    def get_tool_names(self) -> list[str]:
        """Get a list of all loaded tool names for the current run."""
        return list(self._implementation_map.keys())

    def is_tool_loaded(self, tool_name: str) -> bool:
        """Check if a tool is loaded for this run."""
        return tool_name in self._implementation_map

    def get_implementation_key(self, tool_name: str) -> str | None:
        """Get the implementation key for a tool."""
        return self._implementation_map.get(tool_name)


# =============================================================================
# SANDBOX TOOL REGISTRATION (NEW)
# =============================================================================


def register_sandbox_tools_to_manager(tool_manager: ToolManager) -> None:
    """
    Register sandbox tools with a ToolManager instance.

    This adds the sandbox execution tools to the manager's available tools.
    Called when sandbox integration is initialized.

    Args:
        tool_manager: The ToolManager to register tools with
    """
    try:
        from .sandbox import SANDBOX_TOOL_IMPLEMENTATIONS, SANDBOX_TOOL_SCHEMAS

        # Register implementations in global registry
        register_implementations(SANDBOX_TOOL_IMPLEMENTATIONS)

        # Add tool definitions to manager
        for tool_name, schema in SANDBOX_TOOL_SCHEMAS.items():
            # Extract implementation key from the SANDBOX_TOOL_IMPLEMENTATIONS
            impl_key = f"llmcore.tools.sandbox.{tool_name}"

            if impl_key in _IMPLEMENTATION_REGISTRY:
                tool = Tool(
                    name=tool_name,
                    description=schema.get("description", ""),
                    parameters=schema.get("parameters", {}),
                )
                tool_manager._tool_definitions.append(tool)
                tool_manager._implementation_map[tool_name] = impl_key

        logger.info("Sandbox tools registered with ToolManager")

    except ImportError as e:
        logger.warning(f"Sandbox module not available: {e}")
    except Exception as e:
        logger.error(f"Failed to register sandbox tools: {e}")
