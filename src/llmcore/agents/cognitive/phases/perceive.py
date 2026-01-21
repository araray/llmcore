# src/llmcore/agents/cognitive/phases/perceive.py
"""
PERCEIVE Phase Implementation.

The PERCEIVE phase is the first phase of the enhanced cognitive cycle. It is
responsible for gathering and processing all inputs needed for decision-making:
- Retrieving relevant context from memory systems
- Capturing current environmental state
- Creating a snapshot of working memory
- Preparing inputs for subsequent phases

This phase does NOT involve LLM calls - it's purely data gathering and
preparation.

References:
    - Technical Spec: Section 5.3.1 (PERCEIVE Phase)
    - Dossier: Step 2.5 (Cognitive Phases - PERCEIVE)
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..models import EnhancedAgentState, PerceiveInput, PerceiveOutput

if TYPE_CHECKING:
    from ....memory.manager import MemoryManager
    from ...sandbox import SandboxProvider

logger = logging.getLogger(__name__)


# =============================================================================
# PERCEIVE PHASE FUNCTION
# =============================================================================


async def perceive_phase(
    agent_state: EnhancedAgentState,
    perceive_input: PerceiveInput,
    memory_manager: "MemoryManager",
    sandbox: Optional["SandboxProvider"] = None,
    tracer: Optional[Any] = None,
) -> PerceiveOutput:
    """
    Execute the PERCEIVE phase of the cognitive cycle.

    This phase gathers all necessary inputs for decision-making:
    1. Retrieves relevant context from semantic and episodic memory
    2. Captures current sandbox/environmental state
    3. Creates snapshot of working memory
    4. Prepares data for subsequent phases

    Args:
        agent_state: Current enhanced agent state
        perceive_input: Input configuration for perception
        memory_manager: Memory manager for context retrieval
        sandbox: Optional active sandbox provider
        tracer: Optional OpenTelemetry tracer

    Returns:
        PerceiveOutput with gathered context and state

    Example:
        >>> perceive_input = PerceiveInput(
        ...     goal="Calculate factorial of 10",
        ...     force_refresh=False
        ... )
        >>>
        >>> output = await perceive_phase(
        ...     agent_state=state,
        ...     perceive_input=perceive_input,
        ...     memory_manager=memory_manager
        ... )
        >>>
        >>> print(f"Retrieved {len(output.retrieved_context)} context items")
    """
    from ....tracing import add_span_attributes, create_span

    with create_span(tracer, "cognitive.perceive") as span:
        logger.debug("Starting PERCEIVE phase")

        # 1. Retrieve context from memory
        retrieved_context = await _retrieve_context(
            goal=perceive_input.goal,
            context_query=perceive_input.context_query,
            memory_manager=memory_manager,
            force_refresh=perceive_input.force_refresh,
        )

        # 2. Capture working memory snapshot
        working_memory_snapshot = dict(agent_state.working_memory)

        # 3. Capture environmental state
        environmental_state = await _capture_environmental_state(
            agent_state=agent_state, sandbox=sandbox
        )

        # 4. Create output
        output = PerceiveOutput(
            retrieved_context=retrieved_context,
            working_memory_snapshot=working_memory_snapshot,
            environmental_state=environmental_state,
            perceived_at=datetime.utcnow(),
        )

        # 5. Add tracing attributes
        if span:
            add_span_attributes(
                span,
                {
                    "perceive.context_items": len(retrieved_context),
                    "perceive.working_memory_size": len(working_memory_snapshot),
                    "perceive.has_sandbox": sandbox is not None,
                },
            )

        logger.debug(
            f"PERCEIVE phase complete: {len(retrieved_context)} context items, "
            f"{len(working_memory_snapshot)} working memory entries"
        )

        return output


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


async def _retrieve_context(
    goal: str, context_query: Optional[str], memory_manager: "MemoryManager", force_refresh: bool
) -> List[str]:
    """
    Retrieve relevant context from memory systems.

    Args:
        goal: The current goal
        context_query: Optional specific query
        memory_manager: Memory manager instance
        force_refresh: Whether to force refresh

    Returns:
        List of context strings
    """
    query = context_query or goal

    try:
        # Retrieve from semantic memory (RAG)
        context_items = await memory_manager.retrieve_relevant_context(
            goal=query,  # Parameter is named 'goal', not 'query'
        )

        # Convert to strings
        context_strings = []
        for item in context_items:
            # Extract content based on item type
            if hasattr(item, "content"):
                context_strings.append(item.content)
            elif hasattr(item, "text"):
                context_strings.append(item.text)
            elif isinstance(item, str):
                context_strings.append(item)
            else:
                context_strings.append(str(item))

        logger.debug(f"Retrieved {len(context_strings)} context items from memory")
        return context_strings

    except Exception as e:
        logger.warning(f"Failed to retrieve context from memory: {e}")
        return []


async def _capture_environmental_state(
    agent_state: EnhancedAgentState, sandbox: Optional["SandboxProvider"]
) -> Dict[str, Any]:
    """
    Capture current environmental state.

    Args:
        agent_state: Current agent state
        sandbox: Optional active sandbox

    Returns:
        Dictionary of environmental state
    """
    env_state = {
        "has_sandbox": sandbox is not None,
        "iteration_count": agent_state.iteration_count,
        "total_tool_calls": agent_state.total_tool_calls,
        "plan_version": agent_state.plan_version,
    }

    # Add sandbox info if available
    if sandbox:
        try:
            sandbox_info = sandbox.get_info()
            env_state.update(
                {
                    "sandbox_provider": sandbox_info.get("provider"),
                    "sandbox_status": sandbox_info.get("status"),
                    "sandbox_access_level": sandbox_info.get("access_level"),
                }
            )
        except Exception as e:
            logger.warning(f"Failed to get sandbox info: {e}")

    return env_state


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ["perceive_phase"]
