# src/llmcore/task_master/tasks/agent.py
"""
Agent execution tasks for the TaskMaster service.

This module contains arq tasks for running autonomous agent loops
that can execute complex, multi-step reasoning and tool usage to achieve goals.

UPDATED: Added database session support for dynamic tool loading.
UPDATED: Added Human-in-the-Loop (HITL) workflow support for task resumption.
"""

import logging
import os
from typing import Optional
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

from llmcore.api import LLMCore
from llmcore.models import AgentTask, AgentState

logger = logging.getLogger(__name__)


async def run_agent_task(
    ctx,
    goal: str = None,
    session_id: Optional[str] = None,
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None,
    enabled_toolkits: Optional[list] = None,
    task_id: Optional[str] = None
):
    """
    Arq task to run a complete agent loop for a given goal.

    This task creates an AgentManager instance and executes the Think -> Act -> Observe
    loop until the goal is achieved or maximum iterations are reached.

    UPDATED: Added database session support for dynamic tool loading per tenant.
    UPDATED: Added HITL workflow support - can resume tasks from approval/rejection.

    Args:
        ctx: Arq context containing shared resources like the LLMCore instance
        goal: The high-level goal for the agent to achieve (for new tasks)
        session_id: Optional session ID for context and episodic memory
        provider_name: Optional override for the default LLM provider
        model_name: Optional override for the default model
        enabled_toolkits: Optional list of toolkit names to enable for this run
        task_id: Optional task ID for resuming existing tasks (HITL workflow)

    Returns:
        Dictionary containing the task result and status
    """
    llmcore: LLMCore = ctx['llmcore_instance']

    # Determine if this is a new task or resuming an existing one
    if task_id:
        # This is a resuming task - load from database
        logger.info(f"Resuming agent task {task_id}")
        task = await _load_existing_task(task_id)
        if not task:
            return {
                "status": "error",
                "message": f"Task {task_id} not found in database",
                "task_id": task_id
            }
        actual_task_id = task_id
        actual_goal = task.goal
    else:
        # This is a new task - create it
        if not goal:
            return {
                "status": "error",
                "message": "Goal is required for new agent tasks",
                "task_id": None
            }

        actual_task_id = f"agent_{session_id or 'anonymous'}_{int(datetime.now().timestamp())}"
        actual_goal = goal

        # Create the initial agent state and task
        agent_state = AgentState(goal=actual_goal)
        task = AgentTask(
            task_id=actual_task_id,
            goal=actual_goal,
            agent_state=agent_state,
            status="RUNNING"
        )

        logger.info(f"Starting new agent task {actual_task_id} for goal: {actual_goal}")

    # Create database session for tool loading and task persistence
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

        # Get the agent manager from LLMCore
        agent_manager = llmcore.get_agent_manager()

        # Execute the agent loop with database session for tool loading and HITL support
        final_result = await agent_manager.run_agent_loop(
            task=task,
            provider_name=provider_name,
            model_name=model_name,
            session_id=session_id or actual_task_id,
            db_session=db_session,
            enabled_toolkits=enabled_toolkits or ['basic_tools']  # Default to basic tools
        )

        # Check if the task was paused for approval
        if "paused for human approval" in final_result.lower():
            # Task is paused - this is not an error, it's expected HITL behavior
            logger.info(f"Agent task {actual_task_id} paused for human approval")
            return {
                "status": "pending_approval",
                "result": final_result,
                "task_id": actual_task_id,
                "goal": actual_goal,
                "message": "Task paused pending human approval"
            }

        # Update task status to SUCCESS
        task.status = "SUCCESS"
        await _update_task_status(actual_task_id, "SUCCESS", db_session)

        logger.info(f"Agent task {actual_task_id} completed successfully.")

        return {
            "status": "success",
            "result": final_result,
            "task_id": actual_task_id,
            "goal": actual_goal
        }

    except Exception as e:
        logger.error(f"Agent task {actual_task_id} failed: {e}", exc_info=True)

        # Update task status to FAILURE
        if 'task' in locals():
            task.status = "FAILURE"
            try:
                await _update_task_status(actual_task_id, "FAILURE", db_session)
            except Exception as update_error:
                logger.error(f"Failed to update task status to FAILURE: {update_error}")

        return {
            "status": "error",
            "message": str(e),
            "task_id": actual_task_id,
            "goal": actual_goal if 'actual_goal' in locals() else goal
        }

    finally:
        # Clean up database session
        if db_session:
            try:
                await db_session.close()
            except Exception as e:
                logger.warning(f"Error closing database session: {e}")


async def _load_existing_task(task_id: str) -> Optional[AgentTask]:
    """
    Load an existing task from the database.

    Args:
        task_id: The ID of the task to load

    Returns:
        AgentTask object if found, None otherwise
    """
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

        async with session_factory() as db_session:
            # Query for the task
            query = text("""
                SELECT task_id, goal, status, agent_state, pending_action_data,
                       approval_prompt, created_at, updated_at
                FROM agent_tasks
                WHERE task_id = :task_id
            """)

            result = await db_session.execute(query, {"task_id": task_id})
            row = result.fetchone()

            if not row:
                logger.warning(f"Task {task_id} not found in database")
                return None

            # Reconstruct the AgentTask object
            # Note: In a real implementation, you might want to use proper JSON serialization
            # for the agent_state field
            agent_state_data = row.agent_state if row.agent_state else {}
            if isinstance(agent_state_data, str):
                import json
                agent_state_data = json.loads(agent_state_data)

            agent_state = AgentState(
                goal=agent_state_data.get('goal', row.goal),
                plan=agent_state_data.get('plan', []),
                current_plan_step_index=agent_state_data.get('current_plan_step_index', 0),
                plan_steps_status=agent_state_data.get('plan_steps_status', []),
                history_of_thoughts=agent_state_data.get('history_of_thoughts', []),
                observations=agent_state_data.get('observations', {}),
                scratchpad=agent_state_data.get('scratchpad', '')
            )

            # Parse pending action data if present
            pending_action_data = None
            if row.pending_action_data:
                if isinstance(row.pending_action_data, str):
                    import json
                    pending_action_data = json.loads(row.pending_action_data)
                else:
                    pending_action_data = row.pending_action_data

            task = AgentTask(
                task_id=row.task_id,
                status=row.status,
                goal=row.goal,
                agent_state=agent_state,
                pending_action_data=pending_action_data,
                approval_prompt=row.approval_prompt,
                created_at=row.created_at,
                updated_at=row.updated_at
            )

            logger.info(f"Successfully loaded task {task_id} from database")
            return task

    except Exception as e:
        logger.error(f"Error loading task {task_id} from database: {e}", exc_info=True)
        return None


async def _update_task_status(task_id: str, status: str, db_session: Optional[AsyncSession] = None) -> None:
    """
    Update the status of a task in the database.

    Args:
        task_id: The ID of the task to update
        status: The new status to set
        db_session: Optional existing database session to use
    """
    try:
        if db_session:
            # Use the provided session
            update_query = text("""
                UPDATE agent_tasks
                SET status = :status, updated_at = :updated_at
                WHERE task_id = :task_id
            """)

            await db_session.execute(update_query, {
                "task_id": task_id,
                "status": status,
                "updated_at": datetime.now(timezone.utc)
            })
            await db_session.commit()
        else:
            # Create a new session
            database_url = os.environ.get(
                'LLMCORE_TENANT_DATABASE_URL',
                os.environ.get('LLMCORE_AUTH_DATABASE_URL',
                               'postgresql+asyncpg://postgres:password@localhost:5432/llmcore')
            )

            engine = create_async_engine(database_url, echo=False)
            session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

            async with session_factory() as session:
                update_query = text("""
                    UPDATE agent_tasks
                    SET status = :status, updated_at = :updated_at
                    WHERE task_id = :task_id
                """)

                await session.execute(update_query, {
                    "task_id": task_id,
                    "status": status,
                    "updated_at": datetime.now(timezone.utc)
                })
                await session.commit()

        logger.debug(f"Updated task {task_id} status to {status}")

    except Exception as e:
        logger.error(f"Error updating task {task_id} status to {status}: {e}", exc_info=True)
