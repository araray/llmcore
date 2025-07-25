# src/llmcore/agents/tools.py
"""
Tool Management for LLMCore Agents.

Handles the registration, validation, and execution of tools available to agents.
Includes built-in tools for semantic search, episodic search, and basic calculations.

UPDATED: Refactored to support dynamic tool loading from database with secure
implementation registry for tenant-specific tool management.
"""

import asyncio
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Union

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from ..exceptions import LLMCoreError
from ..memory.manager import MemoryManager
from ..models import Tool, ToolCall, ToolResult
from ..storage.manager import StorageManager

logger = logging.getLogger(__name__)


# --- Built-in Tool Functions ---

async def semantic_search(memory_manager: MemoryManager, query: str, k: int = 3, collection: Optional[str] = None) -> str:
    """
    Searches the semantic memory (vector store) for relevant information.

    Args:
        memory_manager: The MemoryManager instance for accessing memory systems.
        query: The search query string.
        k: Number of results to retrieve (default: 3).
        collection: Optional collection name to search in.

    Returns:
        Formatted string containing the search results.
    """
    try:
        logger.debug(f"Semantic search: query='{query}', k={k}, collection={collection}")

        # Use the retrieve_relevant_context method which handles semantic memory
        context_items = await memory_manager.retrieve_relevant_context(query)

        if not context_items:
            return f"No relevant information found in semantic memory for query: '{query}'"

        # Format the results for the agent
        results = []
        for i, item in enumerate(context_items[:k]):
            score_info = ""
            if item.metadata.get("retrieval_score"):
                score_info = f" (relevance: {item.metadata['retrieval_score']:.3f})"

            source_info = ""
            if item.source_id:
                source_info = f" [Source: {item.source_id}]"

            results.append(f"{i+1}. {item.content[:500]}{'...' if len(item.content) > 500 else ''}{source_info}{score_info}")

        return f"Semantic search results for '{query}':\n\n" + "\n\n".join(results)

    except Exception as e:
        logger.error(f"Error in semantic_search tool: {e}", exc_info=True)
        return f"Error searching semantic memory: {str(e)}"


async def episodic_search(storage_manager: StorageManager, session_id: str, query: str, limit: int = 10) -> str:
    """
    Searches the episodic memory for past experiences and interactions.

    Args:
        storage_manager: The StorageManager instance for accessing episode storage.
        session_id: The session ID to search episodes for.
        query: The search query (currently supports basic text matching).
        limit: Maximum number of episodes to return.

    Returns:
        Formatted string containing the episode search results.
    """
    try:
        logger.debug(f"Episodic search: session_id='{session_id}', query='{query}', limit={limit}")

        # Get recent episodes from the session
        episodes = await storage_manager.get_episodes(session_id, limit=limit * 2)  # Get more to filter

        if not episodes:
            return f"No episodes found in session '{session_id}'"

        # Basic text search through episodes (in a full implementation, this would use semantic search)
        query_lower = query.lower()
        matching_episodes = []

        for episode in episodes:
            # Search in the episode data content
            episode_text = json.dumps(episode.data, default=str).lower()
            if query_lower in episode_text:
                matching_episodes.append(episode)
                if len(matching_episodes) >= limit:
                    break

        if not matching_episodes:
            return f"No episodes matching '{query}' found in recent history"

        # Format the results
        results = []
        for i, episode in enumerate(matching_episodes):
            timestamp_str = episode.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            event_type = episode.event_type.value if hasattr(episode.event_type, 'value') else str(episode.event_type)

            # Extract relevant content from episode data
            content_preview = ""
            if isinstance(episode.data, dict):
                if 'content' in episode.data:
                    content_preview = str(episode.data['content'])[:200]
                elif 'thought' in episode.data:
                    content_preview = f"Thought: {episode.data['thought']}"[:200]
                elif 'observation' in episode.data:
                    content_preview = f"Observation: {episode.data['observation']}"[:200]
                else:
                    content_preview = str(episode.data)[:200]

            if len(content_preview) > 200:
                content_preview += "..."

            results.append(f"{i+1}. [{timestamp_str}] {event_type}: {content_preview}")

        return f"Episodic search results for '{query}' in session '{session_id}':\n\n" + "\n\n".join(results)

    except Exception as e:
        logger.error(f"Error in episodic_search tool: {e}", exc_info=True)
        return f"Error searching episodic memory: {str(e)}"


async def calculator(expression: str) -> str:
    """
    Safely evaluates a mathematical expression.

    Args:
        expression: The mathematical expression to evaluate.

    Returns:
        The result of the calculation or an error message.
    """
    try:
        logger.debug(f"Calculator: evaluating '{expression}'")

        # Simple safety check - only allow basic math operations
        if not re.match(r'^[0-9+\-*/().\s]+, expression):
            return f"Invalid expression: '{expression}'. Only basic arithmetic operations are allowed."

        # Prevent potentially dangerous operations
        if any(dangerous in expression.lower() for dangerous in ['import', 'exec', 'eval', '__']):
            return f"Expression contains potentially dangerous operations: '{expression}'"

        try:
            result = eval(expression)
            return f"Result of '{expression}' = {result}"
        except ZeroDivisionError:
            return f"Error: Division by zero in expression '{expression}'"
        except Exception as e:
            return f"Error evaluating expression '{expression}': {str(e)}"

    except Exception as e:
        logger.error(f"Error in calculator tool: {e}", exc_info=True)
        return f"Calculator error: {str(e)}"


async def finish(answer: str) -> str:
    """
    Special tool to indicate the agent has completed its task.

    Args:
        answer: The final answer or result.

    Returns:
        The final answer formatted for completion.
    """
    logger.info(f"Agent finishing with answer: {answer[:100]}...")
    return f"TASK_COMPLETE: {answer}"


# --- Secure Implementation Registry ---

# This is the security boundary - only functions registered here can be executed
_IMPLEMENTATION_REGISTRY: Dict[str, Callable] = {
    "llmcore.tools.search.semantic": semantic_search,
    "llmcore.tools.search.episodic": episodic_search,
    "llmcore.tools.calculation.calculator": calculator,
    "llmcore.tools.flow.finish": finish,
}

# Human-readable descriptions for the implementation keys
_IMPLEMENTATION_DESCRIPTIONS: Dict[str, str] = {
    "llmcore.tools.search.semantic": "Search the knowledge base (vector store) for relevant information",
    "llmcore.tools.search.episodic": "Search past experiences and interactions in episodic memory",
    "llmcore.tools.calculation.calculator": "Perform mathematical calculations safely",
    "llmcore.tools.flow.finish": "Complete the agent task with a final answer",
}


# --- ToolManager Class ---

class ToolManager:
    """
    Manages the registration, validation, and execution of tools available to agents.

    UPDATED: Now supports dynamic tool loading from database with secure implementation
    registry. Tools are loaded per-tenant and per-run rather than globally at startup.
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
        self._tool_definitions: List[Tool] = []
        self._implementation_map: Dict[str, str] = {}  # tool_name -> implementation_key

        logger.info("ToolManager initialized for dynamic tool loading")

    async def load_tools_for_run(
        self,
        db_session: AsyncSession,
        enabled_toolkits: Optional[List[str]] = None
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
                    logger.error(f"Invalid implementation key '{implementation_key}' for tool '{tool_name}'")
                    raise LLMCoreError(f"Tool '{tool_name}' has invalid implementation key: {implementation_key}")

                # Create Tool model from database data
                tool = Tool(
                    name=tool_name,
                    description=row.description,
                    parameters=row.parameters_schema
                )

                # Store the mappings
                self._tool_definitions.append(tool)
                self._implementation_map[tool_name] = implementation_key

            logger.info(f"Loaded {len(self._tool_definitions)} tools for agent run")

        except Exception as e:
            logger.error(f"Error loading tools for run: {e}", exc_info=True)
            raise LLMCoreError(f"Failed to load tools: {str(e)}")

    def get_tool_definitions(self) -> List[Tool]:
        """
        Get all loaded tool definitions for the current run.

        Returns:
            List of Tool definitions available for agent use.
        """
        return self._tool_definitions.copy()

    async def execute_tool(self, tool_call: ToolCall, session_id: Optional[str] = None) -> ToolResult:
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
            error_msg = f"Tool '{tool_name}' not loaded for this run. Available tools: {available_tools}"
            logger.error(error_msg)
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"ERROR: {error_msg}"
            )

        try:
            # Get the implementation key and function
            implementation_key = self._implementation_map[tool_name]
            tool_func = _IMPLEMENTATION_REGISTRY[implementation_key]

            arguments = tool_call.arguments.copy()

            # Inject dependencies based on tool function signature
            import inspect
            sig = inspect.signature(tool_func)

            # Inject memory_manager if the tool needs it
            if 'memory_manager' in sig.parameters:
                arguments['memory_manager'] = self._memory_manager

            # Inject storage_manager if the tool needs it
            if 'storage_manager' in sig.parameters:
                arguments['storage_manager'] = self._storage_manager

            # Inject session_id if the tool needs it and we have one
            if 'session_id' in sig.parameters and session_id:
                arguments['session_id'] = session_id

            logger.debug(f"Executing tool '{tool_name}' (key: {implementation_key}) with arguments: {arguments}")

            # Execute the tool function
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(**arguments)
            else:
                result = tool_func(**arguments)

            logger.debug(f"Tool '{tool_name}' executed successfully")

            return ToolResult(
                tool_call_id=tool_call.id,
                content=str(result)
            )

        except TypeError as e:
            error_msg = f"Invalid arguments for tool '{tool_name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"ERROR: {error_msg}"
            )
        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"ERROR: {error_msg}"
            )

    def get_tool_names(self) -> List[str]:
        """Get a list of all loaded tool names for the current run."""
        return list(self._implementation_map.keys())

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is loaded for the current run."""
        return tool_name in self._implementation_map

    @classmethod
    def is_valid_implementation_key(cls, implementation_key: str) -> bool:
        """
        Check if an implementation key is valid (exists in the secure registry).

        Args:
            implementation_key: The key to validate

        Returns:
            True if the key is valid, False otherwise
        """
        return implementation_key in _IMPLEMENTATION_REGISTRY

    @classmethod
    def get_available_implementations(cls) -> Dict[str, str]:
        """
        Get all available implementation keys and their descriptions.

        Returns:
            Dictionary mapping implementation keys to descriptions
        """
        return _IMPLEMENTATION_DESCRIPTIONS.copy()
