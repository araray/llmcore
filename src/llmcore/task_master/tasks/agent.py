# llmcore/src/llmcore/task_master/tasks/agent.py
"""
Agent execution tasks for the TaskMaster service.

This module contains arq tasks for running autonomous agent loops
that can execute complex, multi-step reasoning and tool usage to achieve goals.
"""

import logging
from typing import Optional

from llmcore.api import LLMCore
from llmcore.models import AgentTask, AgentState

logger = logging.getLogger(__name__)


async def run_agent_task(
    ctx,
    task_id: str,
    goal: str,
    session_id: Optional[str] = None,
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None
):
    """
    Arq task to run a complete agent loop for a given goal.

    This task creates an AgentManager instance and executes the Think -> Act -> Observe
    loop until the goal is achieved or maximum iterations are reached.

    Args:
        ctx: Arq context containing shared resources like the LLMCore instance
        task_id: Unique identifier for this agent task
        goal: The high-level goal for the agent to achieve
        session_id: Optional session ID for context and episodic memory
        provider_name: Optional override for the default LLM provider
        model_name: Optional override for the default model

    Returns:
        Dictionary containing the task result and status
    """
    llmcore: LLMCore = ctx['llmcore_instance']
    logger.info(f"Starting agent task {task_id} for goal: {goal}")

    try:
        # Create the initial agent state and task
        agent_state = AgentState(goal=goal)
        task = AgentTask(
            task_id=task_id,
            goal=goal,
            agent_state=agent_state,
            status="RUNNING"
        )

        # Get the agent manager from LLMCore
        agent_manager = llmcore.get_agent_manager()

        # Execute the agent loop
        final_result = await agent_manager.run_agent_loop(
            task=task,
            provider_name=provider_name,
            model_name=model_name,
            session_id=session_id or task_id
        )

        # Update task status to SUCCESS
        task.status = "SUCCESS"
        logger.info(f"Agent task {task_id} completed successfully.")

        return {
            "status": "success",
            "result": final_result,
            "task_id": task_id,
            "goal": goal
        }

    except Exception as e:
        logger.error(f"Agent task {task_id} failed: {e}", exc_info=True)

        # Update task status to FAILURE
        if 'task' in locals():
            task.status = "FAILURE"

        return {
            "status": "error",
            "message": str(e),
            "task_id": task_id,
            "goal": goal
        }
