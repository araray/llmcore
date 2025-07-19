# src/llmcore/agents/tools.py
"""
Tool Management for LLMCore Agents.

Handles the registration, validation, and execution of tools available to agents.
Includes built-in tools for semantic search, episodic search, and basic calculations.
"""

import asyncio
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Union

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
        if not re.match(r'^[0-9+\-*/().\s]+$', expression):
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


# --- ToolManager Class ---

class ToolManager:
    """
    Manages the registration, validation, and execution of tools available to agents.

    Provides a dynamic registry for tools and handles their execution with proper
    dependency injection and error handling.
    """

    def __init__(self, memory_manager: MemoryManager, storage_manager: StorageManager):
        """
        Initialize the ToolManager with required dependencies.

        Args:
            memory_manager: The MemoryManager instance for memory-related tools.
            storage_manager: The StorageManager instance for storage-related tools.
        """
        self._tools: Dict[str, Callable] = {}
        self._tool_definitions: List[Tool] = []
        self._memory_manager = memory_manager
        self._storage_manager = storage_manager

        # Register built-in tools
        self._register_built_in_tools()
        logger.info(f"ToolManager initialized with {len(self._tool_definitions)} tools")

    def _register_built_in_tools(self) -> None:
        """Register all built-in tools with their definitions."""

        # Register semantic_search tool
        self.register_tool(
            func=semantic_search,
            definition=Tool(
                name="semantic_search",
                description="Searches the knowledge base (Semantic Memory) for relevant information on a topic. Use this when you need factual information or documentation.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant information"
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of results to retrieve (default: 3)",
                            "default": 3
                        },
                        "collection": {
                            "type": "string",
                            "description": "Optional collection name to search in"
                        }
                    },
                    "required": ["query"]
                }
            )
        )

        # Register episodic_search tool
        self.register_tool(
            func=episodic_search,
            definition=Tool(
                name="episodic_search",
                description="Searches past experiences and interactions (Episodic Memory) to recall previous conversations, actions, or observations. Use this to remember what happened before.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for in past experiences"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of episodes to return (default: 10)",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            )
        )

        # Register calculator tool
        self.register_tool(
            func=calculator,
            definition=Tool(
                name="calculator",
                description="Performs mathematical calculations. Use this for arithmetic operations, calculations, or mathematical problem solving.",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate (e.g., '2 + 3 * 4', '(10 - 3) / 2')"
                        }
                    },
                    "required": ["expression"]
                }
            )
        )

        # Register finish tool
        self.register_tool(
            func=finish,
            definition=Tool(
                name="finish",
                description="Use this tool when you have completed the task and have a final answer. This will end the agent's execution.",
                parameters={
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The final answer or result of the task"
                        }
                    },
                    "required": ["answer"]
                }
            )
        )

    def register_tool(self, func: Callable, definition: Tool) -> None:
        """
        Register a new tool with its function and definition.

        Args:
            func: The Python function to execute for this tool.
            definition: The Tool definition including name, description, and parameters.
        """
        self._tools[definition.name] = func

        # Remove existing definition if present (for updates)
        self._tool_definitions = [t for t in self._tool_definitions if t.name != definition.name]
        self._tool_definitions.append(definition)

        logger.debug(f"Registered tool: {definition.name}")

    def get_tool_definitions(self) -> List[Tool]:
        """
        Get all registered tool definitions.

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

        if tool_name not in self._tools:
            available_tools = list(self._tools.keys())
            error_msg = f"Tool '{tool_name}' not found. Available tools: {available_tools}"
            logger.error(error_msg)
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"ERROR: {error_msg}"
            )

        try:
            tool_func = self._tools[tool_name]
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

            logger.debug(f"Executing tool '{tool_name}' with arguments: {arguments}")

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
        """Get a list of all registered tool names."""
        return list(self._tools.keys())

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self._tools
