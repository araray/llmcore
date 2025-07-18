# src/llmcore/api_server/routes/tasks.py
"""
Task management API routes for the llmcore API server.

This module contains the implementation of the task management endpoints
that provide API access to the asynchronous TaskMaster service functionality.
"""

import asyncio
import json
import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse

try:
    from arq.jobs import Job, JobStatus
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


@router.get("/{task_id}", response_model=TaskStatusResponse)
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


@router.get("/{task_id}/result")
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


@router.get("/{task_id}/stream")
async def stream_task_progress(task_id: str):
    """
    Stream real-time progress updates for a long-running task using Server-Sent Events.

    This endpoint provides a continuous stream of progress updates for tasks that
    support progress reporting. It's particularly useful for ingestion tasks and
    other long-running operations.

    Args:
        task_id: The unique identifier of the task to stream

    Returns:
        StreamingResponse: Server-Sent Events stream with progress updates

    Raises:
        HTTPException:
            - 404 if the task is not found
            - 503 if the task queue is not available
            - 500 for internal server errors
    """
    # Check if Redis is available
    if not is_redis_available():
        logger.error("Attempted to stream task progress but Redis pool is not available")
        raise HTTPException(
            status_code=503,
            detail="Task queue service is not available."
        )

    async def event_stream():
        """Generate Server-Sent Events for task progress."""
        try:
            redis: ArqRedis = get_redis_pool()
            job = Job(task_id, redis)

            # Check if task exists
            info = await job.info()
            if not info:
                yield f"data: {json.dumps({'type': 'error', 'error': 'Task not found'})}\n\n"
                return

            # Initial status event
            yield f"data: {json.dumps({'type': 'status', 'status': info.status.value, 'task_id': task_id})}\n\n"

            # Monitor task until completion
            last_status = info.status.value
            poll_count = 0
            max_polls = 360  # 30 minutes with 5-second intervals

            while poll_count < max_polls:
                await asyncio.sleep(5)  # Poll every 5 seconds
                poll_count += 1

                try:
                    # Get updated job info
                    info = await job.info()
                    if not info:
                        yield f"data: {json.dumps({'type': 'error', 'error': 'Task disappeared'})}\n\n"
                        break

                    current_status = info.status.value

                    # Send status update if changed
                    if current_status != last_status:
                        yield f"data: {json.dumps({'type': 'status_change', 'status': current_status, 'previous_status': last_status})}\n\n"
                        last_status = current_status

                    # Send periodic heartbeat
                    if poll_count % 6 == 0:  # Every 30 seconds
                        yield f"data: {json.dumps({'type': 'heartbeat', 'status': current_status, 'poll_count': poll_count})}\n\n"

                    # Check if task is complete
                    if current_status in ['complete', 'failed']:
                        if current_status == 'complete':
                            # Send final result
                            yield f"data: {json.dumps({'type': 'complete', 'result': info.result})}\n\n"
                        else:
                            # Send failure notification
                            yield f"data: {json.dumps({'type': 'failed', 'error': str(info.result)})}\n\n"

                        yield f"data: {json.dumps({'type': 'end'})}\n\n"
                        break

                except Exception as poll_error:
                    logger.error(f"Error polling task {task_id} during stream: {poll_error}")
                    yield f"data: {json.dumps({'type': 'error', 'error': 'Polling error occurred'})}\n\n"
                    break

            else:
                # Max polls reached - timeout
                yield f"data: {json.dumps({'type': 'timeout', 'message': 'Maximum polling time reached'})}\n\n"
                yield f"data: {json.dumps({'type': 'end'})}\n\n"

        except Exception as stream_error:
            logger.error(f"Error in task stream for {task_id}: {stream_error}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': 'Stream error occurred'})}\n\n"
            yield f"data: {json.dumps({'type': 'end'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )
