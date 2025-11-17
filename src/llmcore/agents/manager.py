# src/llmcore/agents/manager.py
"""
Agent Management for LLMCore.

This module provides the AgentManager, which orchestrates the agentic execution loop.
It coordinates between the LLM provider, memory systems, and tool execution to
achieve complex goals autonomously.

REFACTORED: This class now acts as a high-level orchestrator, delegating the
implementation of the cognitive cycle (Plan, Think, Act, etc.) and prompt
management to the `cognitive_cycle` and `prompt_utils` modules respectively.
This separation of concerns makes the agent's architecture clearer and more maintainable.
"""

import logging
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from ..exceptions import LLMCoreError
from ..memory.manager import MemoryManager
from ..models import AgentTask
from ..providers.manager import ProviderManager
from ..storage.manager import StorageManager
from . import cognitive_cycle
from .tools import ToolManager

logger = logging.getLogger(__name__)

class AgentManager:
    """
    Orchestrates the autonomous agent execution loop with strategic planning and reflection.

    Manages the Plan -> (Think -> Act -> Observe -> Reflect) cycle by calling
    functions from the `cognitive_cycle` module. It initializes and holds the
    necessary dependencies (managers for providers, memory, storage, tools) and
    passes them to the cognitive functions as needed.
    """

    def __init__(
        self,
        provider_manager: ProviderManager,
        memory_manager: MemoryManager,
        storage_manager: StorageManager
    ):
        """
        Initialize the AgentManager with required dependencies.

        Args:
            provider_manager: The ProviderManager for LLM interactions.
            memory_manager: The MemoryManager for context retrieval.
            storage_manager: The StorageManager for episodic memory logging.
        """
        self._provider_manager = provider_manager
        self._memory_manager = memory_manager
        self._storage_manager = storage_manager
        self._tool_manager = ToolManager(memory_manager, storage_manager)

        # Initialize tracing
        self._tracer = None
        try:
            from ..tracing import get_tracer
            self._tracer = get_tracer("llmcore.agents.manager")
        except Exception as e:
            logger.debug(f"Tracing not available for AgentManager: {e}")

        logger.info("AgentManager initialized as orchestrator.")

    async def run_agent_loop(
        self,
        task: AgentTask,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        max_iterations: int = 10,
        session_id: Optional[str] = None,
        db_session: Optional[AsyncSession] = None,
        enabled_toolkits: Optional[List[str]] = None
    ) -> str:
        """
        Orchestrates the Plan -> (Think -> Act -> Observe -> Reflect) loop.

        This method manages the overall flow of the agent's execution, calling
        the appropriate functions from the `cognitive_cycle` module to perform
        each step.

        Args:
            task: The AgentTask containing the goal and initial state.
            provider_name: Optional override for the LLM provider.
            model_name: Optional override for the model name.
            max_iterations: Maximum number of loop iterations.
            session_id: Optional session ID for episodic memory context.
            db_session: Tenant-scoped database session for dynamic tool loading.
            enabled_toolkits: List of toolkit names to enable for this run.

        Returns:
            The final result or answer from the agent.
        """
        agent_state = task.agent_state
        actual_session_id = session_id or task.task_id

        logger.info(f"Starting orchestrated agent loop for goal: '{agent_state.goal[:100]}...'")

        span_attributes = {
            "agent.task_id": task.task_id,
            "agent.goal": agent_state.goal[:200],
            "agent.max_iterations": max_iterations,
        }

        from ..tracing import create_span, add_span_attributes, record_span_exception
        with create_span(self._tracer, "agent.execution_loop_orchestration", **span_attributes) as main_span:
            try:
                # Handle task resumption from HITL workflow
                is_resuming = cognitive_cycle.check_if_resuming_task(task)
                if is_resuming:
                    logger.info(f"Resuming agent task {task.task_id} from HITL workflow")
                    if main_span:
                        add_span_attributes(main_span, {"agent.resuming_from_hitl": True})

                    resume_result = await cognitive_cycle.handle_task_resumption(
                        task, actual_session_id, self._tool_manager, self._storage_manager, self._tracer, db_session
                    )
                    if resume_result:
                        return resume_result

                # Load tools for this agent run
                if db_session:
                    await self._tool_manager.load_tools_for_run(db_session, enabled_toolkits)
                    logger.info(f"Loaded {len(self._tool_manager.get_tool_names())} tools for agent run.")

                # STEP 1: PLANNING (if not resuming)
                if not is_resuming:
                    await cognitive_cycle.plan_step(
                        agent_state, actual_session_id, self._provider_manager, self._tracer, provider_name, model_name
                    )
                    if agent_state.plan and not agent_state.plan_steps_status:
                        agent_state.plan_steps_status = ['pending'] * len(agent_state.plan)
                    logger.info(f"Agent generated plan with {len(agent_state.plan)} steps.")

                # MAIN LOOP
                for iteration in range(max_iterations):
                    with create_span(self._tracer, "agent.iteration", iteration=iteration + 1) as iter_span:
                        logger.debug(f"Agent iteration {iteration + 1}/{max_iterations}")

                        # 1. THINK
                        thought, tool_call = await cognitive_cycle.think_step(
                            agent_state, actual_session_id, self._memory_manager, self._provider_manager,
                            self._tool_manager, self._tracer, provider_name, model_name
                        )
                        if not thought or not tool_call:
                            raise LLMCoreError(f"Failed to get valid thought/action at iteration {iteration + 1}")

                        # 2. ACT
                        tool_result = await cognitive_cycle.act_step(
                            tool_call, actual_session_id, task, self._tool_manager, self._tracer, db_session
                        )
                        if tool_result.content == "PAUSED_FOR_APPROVAL":
                            logger.info(f"Agent task {task.task_id} paused for human approval.")
                            return "Agent task paused for human approval. Use the API to approve or reject."

                        # 3. OBSERVE
                        await cognitive_cycle.observe_step(
                            agent_state, thought, tool_call, tool_result.content,
                            actual_session_id, self._storage_manager, self._tracer
                        )

                        # 4. REFLECT
                        await cognitive_cycle.reflect_step(
                            agent_state, tool_call, tool_result, actual_session_id,
                            self._provider_manager, self._tracer, provider_name, model_name
                        )

                        # 5. CHECK FOR FINISH
                        if tool_call.name == "finish":
                            final_answer = tool_result.content.replace("TASK_COMPLETE: ", "")
                            logger.info(f"Agent completed task after {iteration + 1} iterations.")
                            # Record successful completion metrics
                            try:
                                tenant_id = context.get('tenant_id', 'unknown')
                            except Exception as e:
                                logger.debug(f"Failed to record agent metrics: {e}")
                            return final_answer

                # Max iterations reached
                logger.warning(f"Agent reached max iterations ({max_iterations}) without completion.")
                # Record timeout metrics
                try:
                    tenant_id = context.get('tenant_id', 'unknown')
                except Exception as e:
                    logger.debug(f"Failed to record agent metrics: {e}")
                return f"Agent reached maximum iterations ({max_iterations}) without completing the task."

            except Exception as e:
                error_msg = f"Agent loop failed: {str(e)}"
                logger.error(error_msg, exc_info=True)
                if main_span:
                    record_span_exception(main_span, e)
                # Record error metrics
                try:
                    tenant_id = context.get('tenant_id', 'unknown')
                except Exception as metrics_error:
                    logger.debug(f"Failed to record agent metrics: {metrics_error}")
                return f"Agent error: {error_msg}"

    def get_tool_manager(self) -> ToolManager:
        """Get the ToolManager instance for external access."""
        return self._tool_manager
