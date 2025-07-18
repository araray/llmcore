# src/llmcore/api_server/routes/tasks.py
"""
Task management API routes for the llmcore API server.

This module contains the implementation of the task management endpoints
that provide API access to the asynchronous TaskMaster service functionality.
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

try:
    from arq.jobs import Job
    from arq.connections import ArqRedis
except ImportError as e:
    raise ImportError(
        "arq library is required for task management. Install with: pip install arq>=0.25.0"
    ) from e

from ..services.redis_client import get_redis_pool, is_redis_available
from ..models.tasks import (
    TaskSubmissionRequest,
    TaskSubmissionResponse,
    TaskStatusResponse,
    TaskResultResponse
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/tasks/submit",
    response_model=TaskSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED
)
async def submit_task(request: TaskSubmissionRequest) -> TaskSubmissionResponse:
    """
    Submit a new asynchronous task to the TaskMaster service.

    This endpoint accepts a task name and parameters, enqueues the job using arq,
    and immediately returns a task_id with a 202 Accepted status. The task will
    be processed asynchronously by the TaskMaster worker.

    Args:
        request: The task submission request containing task name and parameters

    Returns:
        TaskSubmissionResponse: Contains the task_id and initial status

    Raises:
        HTTPException:
            - 503 if Redis/task queue is not available
            - 400 if task parameters are invalid
            - 500 for internal server errors
    """
    # Check if Redis is available
    if not is_redis_available():
        logger.error("Attempted to submit task but Redis pool is not available")
        raise HTTPException(
            status_code=503,
            detail="Task queue service is not available. Please try again later."
        )

    try:
        # Get the Redis pool
        redis: ArqRedis = get_redis_pool()

        logger.info(
            f"Submitting task '{request.task_name}' with args={request.args}, "
            f"kwargs={request.kwargs}"
        )

        # Enqueue the job
        job = await redis.enqueue_job(
            request.task_name,
            *request.args,
            **request.kwargs
        )

        logger.info(f"Task submitted successfully with ID: {job.job_id}")

        return TaskSubmissionResponse(
            task_id=job.job_id,
            status="queued"
        )

    except Exception as e:
        logger.error(f"Error submitting task '{request.task_name}': {e}", exc_info=True)

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
                detail="Task parameters could not be serialized. Please check your input data."
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="An internal error occurred while submitting the task."
            )


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """
    Get the current status of an asynchronous task.

    This endpoint queries the task queue to determine the current status of
    a previously submitted task. Possible statuses include: queued, in_progress,
    complete, failed.

    Args:
        task_id: The unique identifier of the task to query

    Returns:
        TaskStatusResponse: Contains the task status and result availability

    Raises:
        HTTPException:
            - 404 if the task is not found
            - 503 if the task queue is not available
            - 500 for internal server errors
    """
    # Check if Redis is available
    if not is_redis_available():
        logger.error("Attempted to get task status but Redis pool is not available")
        raise HTTPException(
            status_code=503,
            detail="Task queue service is not available."
        )

    try:
        # Get the Redis pool and job info
        redis: ArqRedis = get_redis_pool()
        job = Job(task_id, redis)
        info = await job.info()

        if not info:
            logger.warning(f"Task not found: {task_id}")
            raise HTTPException(
                status_code=404,
                detail="Task not found. It may have expired or never existed."
            )

        logger.debug(f"Retrieved status for task {task_id}: {info.status.value}")

        return TaskStatusResponse(
            task_id=task_id,
            status=info.status.value,
            result_available=info.success is not None
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error getting status for task {task_id}: {e}", exc_info=True)

        error_message = str(e).lower()
        if "connection" in error_message or "redis" in error_message:
            raise HTTPException(
                status_code=503,
                detail="Task queue service is temporarily unavailable."
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="An internal error occurred while retrieving task status."
            )


@router.get("/tasks/{task_id}/result")
async def get_task_result(task_id: str) -> JSONResponse:
    """
    Get the final result of a completed asynchronous task.

    This endpoint retrieves the result of a task that has completed successfully.
    If the task is not complete or has failed, appropriate error responses are returned.

    Args:
        task_id: The unique identifier of the task to get results for

    Returns:
        JSONResponse: Contains the task result data

    Raises:
        HTTPException:
            - 404 if the task is not found
            - 400 if the task result is not yet available
            - 500 if the task failed or for internal server errors
            - 503 if the task queue is not available
    """
    # Check if Redis is available
    if not is_redis_available():
        logger.error("Attempted to get task result but Redis pool is not available")
        raise HTTPException(
            status_code=503,
            detail="Task queue service is not available."
        )

    try:
        # Get the Redis pool and job info
        redis: ArqRedis = get_redis_pool()
        job = Job(task_id, redis)
        info = await job.info()

        if not info:
            logger.warning(f"Task not found: {task_id}")
            raise HTTPException(
                status_code=404,
                detail="Task not found. It may have expired or never existed."
            )

        # Check if the task failed
        if info.status.value == 'failed':
            logger.warning(f"Task {task_id} failed: {info.result}")
            raise HTTPException(
                status_code=500,
                detail=f"Task failed: {info.result}"
            )

        # Check if the task is not yet complete
        if not info.success:
            current_status = info.status.value
            logger.debug(f"Task {task_id} result not yet available, status: {current_status}")
            raise HTTPException(
                status_code=400,
                detail=f"Task result not yet available. Current status: {current_status}"
            )

        logger.info(f"Retrieved result for completed task {task_id}")

        # Return the result using TaskResultResponse structure
        result_data = TaskResultResponse(
            task_id=task_id,
            status="complete",
            result=info.result
        )

        return JSONResponse(
            content=result_data.model_dump(),
            status_code=200
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error getting result for task {task_id}: {e}", exc_info=True)

        error_message = str(e).lower()
        if "connection" in error_message or "redis" in error_message:
            raise HTTPException(
                status_code=503,
                detail="Task queue service is temporarily unavailable."
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="An internal error occurred while retrieving task result."
            )
