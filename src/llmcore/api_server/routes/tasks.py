# src/llmcore/api_server/routes/tasks.py
"""
Task management API routes for the llmcore API server.

This module contains the implementation of the generic task management endpoints
that provide API access to the asynchronous TaskMaster service functionality.

REFACTORED: The Human-in-the-Loop (HITL) specific endpoints for agent task
approval and rejection have been moved to the `hitl.py` module to improve
separation of concerns. This router now handles only the core, generic task
queue operations.
"""

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse

try:
    from arq.connections import ArqRedis
    from arq.jobs import Job
except ImportError:
    ArqRedis = None
    Job = None

from ..models.tasks import (TaskSubmissionRequest, TaskSubmissionResponse,
                            TaskStatusResponse, TaskResultResponse)
from ..services.redis_client import get_redis_pool, is_redis_available

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/submit",
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
        request: The task submission request containing task name and parameters.

    Returns:
        TaskSubmissionResponse: Contains the task_id and initial status.

    Raises:
        HTTPException: For service unavailability or invalid request parameters.
    """
    if not is_redis_available():
        logger.error("Attempted to submit task but Redis pool is not available")
        raise HTTPException(
            status_code=503,
            detail="Task queue service is not available. Please try again later."
        )

    try:
        redis: ArqRedis = get_redis_pool()
        logger.info(f"Submitting task '{request.task_name}' with args={request.args}, kwargs={request.kwargs}")
        job = await redis.enqueue_job(request.task_name, *request.args, **request.kwargs)
        logger.info(f"Task submitted successfully with ID: {job.job_id}")
        return TaskSubmissionResponse(task_id=job.job_id, status="queued")
    except Exception as e:
        logger.error(f"Error submitting task '{request.task_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while submitting the task.")


@router.get("/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """
    Get the current status of an asynchronous task.

    Queries the task queue to determine the current status of a previously
    submitted task (e.g., queued, in_progress, complete, failed).

    Args:
        task_id: The unique identifier of the task to query.

    Returns:
        TaskStatusResponse: Contains the task status and result availability.

    Raises:
        HTTPException: If the task is not found or the service is unavailable.
    """
    if not is_redis_available():
        raise HTTPException(status_code=503, detail="Task queue service is not available.")

    try:
        redis: ArqRedis = get_redis_pool()
        job = Job(task_id, redis)
        info = await job.info()

        if not info:
            raise HTTPException(status_code=404, detail="Task not found. It may have expired or never existed.")

        logger.debug(f"Retrieved status for task {task_id}: {info.status.value}")
        return TaskStatusResponse(task_id=task_id, status=info.status.value, result_available=info.success is not None)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status for task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while retrieving task status.")


@router.get("/{task_id}/result")
async def get_task_result(task_id: str) -> JSONResponse:
    """
    Get the final result of a completed asynchronous task.

    Retrieves the result of a task that has completed successfully. If the task
    is not complete or has failed, appropriate error responses are returned.

    Args:
        task_id: The unique identifier of the task to get results for.

    Returns:
        JSONResponse: Contains the task result data.

    Raises:
        HTTPException: If task not found, result not ready, or task failed.
    """
    if not is_redis_available():
        raise HTTPException(status_code=503, detail="Task queue service is not available.")

    try:
        redis: ArqRedis = get_redis_pool()
        job = Job(task_id, redis)
        info = await job.info()

        if not info:
            raise HTTPException(status_code=404, detail="Task not found.")
        if info.status.value == 'failed':
            raise HTTPException(status_code=500, detail=f"Task failed: {info.result}")
        if not info.success:
            raise HTTPException(status_code=400, detail=f"Task result not yet available. Current status: {info.status.value}")

        logger.info(f"Retrieved result for completed task {task_id}")
        result_data = TaskResultResponse(task_id=task_id, status="complete", result=info.result)
        return JSONResponse(content=result_data.model_dump(), status_code=200)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting result for task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while retrieving task result.")


@router.get("/{task_id}/stream")
async def stream_task_progress(task_id: str):
    """
    Stream real-time progress updates for a long-running task using Server-Sent Events.

    Provides a continuous stream of progress updates for tasks that support
    progress reporting, useful for ingestion or long agent runs.

    Args:
        task_id: The unique identifier of the task to stream.

    Returns:
        StreamingResponse: A Server-Sent Events stream with progress updates.
    """
    if not is_redis_available():
        raise HTTPException(status_code=503, detail="Task queue service is not available.")

    async def event_stream():
        try:
            redis: ArqRedis = get_redis_pool()
            job = Job(task_id, redis)
            info = await job.info()
            if not info:
                yield f"data: {json.dumps({'type': 'error', 'error': 'Task not found'})}\n\n"
                return

            yield f"data: {json.dumps({'type': 'status', 'status': info.status.value, 'task_id': task_id})}\n\n"
            last_status = info.status.value

            for _ in range(360):  # Poll for up to 30 minutes
                await asyncio.sleep(5)
                info = await job.info()
                if not info:
                    yield f"data: {json.dumps({'type': 'error', 'error': 'Task disappeared'})}\n\n"
                    break

                if info.status.value != last_status:
                    yield f"data: {json.dumps({'type': 'status_change', 'status': info.status.value})}\n\n"
                    last_status = info.status.value

                if info.status.value in ['complete', 'failed']:
                    result_type = 'complete' if info.status.value == 'complete' else 'failed'
                    payload = {'result': info.result} if result_type == 'complete' else {'error': str(info.result)}
                    yield f"data: {json.dumps({'type': result_type, **payload})}\n\n"
                    break
            else:
                yield f"data: {json.dumps({'type': 'timeout'})}\n\n"

            yield f"data: {json.dumps({'type': 'end'})}\n\n"
        except Exception as e:
            logger.error(f"Error in task stream for {task_id}: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': 'Stream error occurred'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
