# llmcore/src/llmcore/api_server/routes/agents.py
"""
Agent execution API routes for the llmcore API server.

This module contains the implementation of the agent endpoints that provide
API access to the autonomous agent execution functionality via the TaskMaster service.
"""

import logging
from fastapi import APIRouter, HTTPException, status, Depends

from ..services.redis_client import get_redis_pool, is_redis_available
from ..models.agents import AgentRunRequest
from ..models.tasks import TaskSubmissionResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/run",
    response_model=TaskSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED
)
async def run_agent(
    request: AgentRunRequest,
    redis_pool=Depends(get_redis_pool)
) -> TaskSubmissionResponse:
    """
    Start a new autonomous agent task to achieve a high-level goal.

    This endpoint accepts a goal and optional parameters, creates an agent task,
    and enqueues it for asynchronous execution by the TaskMaster service. The agent
    will use the Think -> Act -> Observe loop to autonomously work towards the goal.

    Args:
        request: The agent run request containing the goal and optional parameters
        redis_pool: Redis connection pool for task queue (injected dependency)

    Returns:
        TaskSubmissionResponse: Contains the task_id for monitoring progress

    Raises:
        HTTPException:
            - 503 if Redis/task queue is not available
            - 400 if request parameters are invalid
            - 500 for internal server errors
    """
    # Check if Redis is available
    if not is_redis_available():
        logger.error("Attempted to run agent but Redis pool is not available")
        raise HTTPException(
            status_code=503,
            detail="Task queue service is not available. Please try again later."
        )

    # Validate the goal
    if not request.goal or not request.goal.strip():
        raise HTTPException(
            status_code=400,
            detail="A non-empty goal is required for the agent."
        )

    try:
        logger.info(f"Submitting agent task for goal: '{request.goal[:100]}...'")

        # Enqueue the agent task
        job = await redis_pool.enqueue_job(
            'run_agent_task',
            goal=request.goal,
            session_id=request.session_id,
            provider_name=request.provider,
            model_name=request.model
        )

        logger.info(f"Enqueued agent task {job.job_id} for goal: {request.goal[:50]}...")

        return TaskSubmissionResponse(
            task_id=job.job_id,
            status="queued"
        )

    except Exception as e:
        logger.error(f"Error submitting agent task: {e}", exc_info=True)

        # Check for common error patterns
        error_message = str(e).lower()
        if "connection" in error_message or "redis" in error_message:
            raise HTTPException(
                status_code=503,
                detail="Task queue service is temporarily unavailable."
            )
        elif "serialize" in error_message or "json" in error_message:
            raise HTTPException(
                status_code=400,
                detail="Agent parameters could not be serialized. Please check your input data."
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="An internal error occurred while submitting the agent task."
            )
