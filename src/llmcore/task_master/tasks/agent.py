# llmcore/src/llmcore/task_master/tasks/agent.py
"""
Agent execution tasks for the TaskMaster service.

This module contains arq tasks for running autonomous agent loops
that can execute complex, multi-step reasoning and tool usage to achieve goals.

UPDATED: Added database session support for dynamic tool loading.
"""

import logging
import os
from typing import Optional

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from llmcore.api import LLMCore
from llmcore.models import AgentTask, AgentState

logger = logging.getLogger(__name__)


async def run_agent_task(
    ctx,
    goal: str,
    session_id: Optional[str] = None,
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None,
    enabled_toolkits: Optional[list] = None
):
    """
    Arq task to run a complete agent loop for a given goal.

    This task creates an AgentManager instance and executes the Think -> Act -> Observe
    loop until the goal is achieved or maximum iterations are reached.

    UPDATED: Added database session support for dynamic tool loading per tenant.

    Args:
        ctx: Arq context containing shared resources like the LLMCore instance
        goal: The high-level goal for the agent to achieve
        session_id: Optional session ID for context and episodic memory
        provider_name: Optional override for the default LLM provider
        model_name: Optional override for the default model
        enabled_toolkits: Optional list of toolkit names to enable for this run

    Returns:
        Dictionary containing the task result and status
    """
    llmcore: LLMCore = ctx['llmcore_instance']
    task_id = f"agent_{session_id or 'anonymous'}"

    logger.info(f"Starting agent task {task_id} for goal: {goal}")

    # Create database session for tool loading
    # Note: In a production environment, this would need proper tenant context
    # For now, we'll create a basic session that can be extended
    db_session = None
    try:
        # Get database URL from environment
        database_url = os.environ.get(
            'LLMCORE_TENANT_DATABASE_URL',
            os.environ.get('LLMCORE_AUTH_DATABASE_URL',
                           'postgresql+asyncpg://postgres:password@localhost:5432/llmcore')
        )

        # Create async engine and session
        engine = create_async_engine(database_url, echo=False)
        session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        db_session = session_factory()

        # Note: In production, we would need to set the search_path to the correct tenant schema
        # For now, this assumes we're working with a default schema or single tenant setup
        # await db_session.execute(f"SET search_path TO {tenant_schema}, public")

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

        # Execute the agent loop with database session for tool loading
        final_result = await agent_manager.run_agent_loop(
            task=task,
            provider_name=provider_name,
            model_name=model_name,
            session_id=session_id or task_id,
            db_session=db_session,
            enabled_toolkits=enabled_toolkits or ['basic_tools']  # Default to basic tools
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

    finally:
        # Clean up database session
        if db_session:
            try:
                await db_session.close()
            except Exception as e:
                logger.warning(f"Error closing database session: {e}")
