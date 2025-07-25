# src/llmcore/api_server/routes/tasks.py
"""
Task management API routes for the llmcore API server.

This module contains the implementation of the task management endpoints
that provide API access to the asynchronous TaskMaster service functionality.

UPDATED: Added Human-in-the-Loop (HITL) workflow endpoints for task approval/rejection.
"""

import asyncio
import json
import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

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
from ..db import get_tenant_db_session
from ..auth import get_current_tenant
from ..schemas.security import Tenant

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


# ============================================================================
# NEW: Human-in-the-Loop (HITL) Workflow Endpoints
# ============================================================================

@router.get("/pending_approval")
async def list_pending_approval_tasks(
    db_session: AsyncSession = Depends(get_tenant_db_session),
    tenant: Tenant = Depends(get_current_tenant)
) -> Dict[str, Any]:
    """
    List all tasks awaiting human approval for the current tenant.

    This endpoint returns tasks that are in PENDING_APPROVAL status,
    providing the information needed for human operators to review and
    make approval decisions.

    Args:
        db_session: Tenant-scoped database session
        tenant: Current authenticated tenant

    Returns:
        Dictionary containing pending tasks and their approval prompts

    Raises:
        HTTPException: For database errors or service unavailability
    """
    try:
        logger.info(f"Listing pending approval tasks for tenant: {tenant.name}")

        # Query for tasks in PENDING_APPROVAL status
        query = text("""
            SELECT task_id, goal, approval_prompt, pending_action_data, created_at, updated_at
            FROM agent_tasks
            WHERE status = 'PENDING_APPROVAL'
            ORDER BY updated_at DESC
        """)

        result = await db_session.execute(query)
        rows = result.fetchall()

        pending_tasks = []
        for row in rows:
            task_data = {
                "task_id": row.task_id,
                "goal": row.goal,
                "approval_prompt": row.approval_prompt,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            }

            # Include pending action details if available
            if row.pending_action_data:
                try:
                    pending_data = json.loads(row.pending_action_data) if isinstance(row.pending_action_data, str) else row.pending_action_data
                    approved_action = pending_data.get("approved_action", {})
                    task_data["pending_action"] = {
                        "tool_name": approved_action.get("name", "unknown"),
                        "arguments": approved_action.get("arguments", {}),
                        "requested_at": pending_data.get("requested_at")
                    }
                except Exception as e:
                    logger.warning(f"Error parsing pending action data for task {row.task_id}: {e}")
                    task_data["pending_action"] = {"tool_name": "unknown", "arguments": {}}

            pending_tasks.append(task_data)

        logger.info(f"Found {len(pending_tasks)} tasks pending approval for tenant {tenant.name}")

        return {
            "pending_tasks": pending_tasks,
            "total_count": len(pending_tasks),
            "tenant_id": str(tenant.id)
        }

    except Exception as e:
        logger.error(f"Error listing pending approval tasks: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while retrieving pending approval tasks."
        )


@router.post("/{task_id}/approve")
async def approve_task(
    task_id: str,
    db_session: AsyncSession = Depends(get_tenant_db_session),
    tenant: Tenant = Depends(get_current_tenant)
) -> Dict[str, Any]:
    """
    Approve a pending task and resume its execution.

    This endpoint approves a task that was paused for human review,
    updates its status, and re-enqueues it for execution by the TaskMaster.

    Args:
        task_id: The ID of the task to approve
        db_session: Tenant-scoped database session
        tenant: Current authenticated tenant

    Returns:
        Confirmation message and task details

    Raises:
        HTTPException: If task not found, not pending approval, or system errors
    """
    try:
        logger.info(f"Approving task {task_id} for tenant {tenant.name}")

        # Check if Redis is available for re-enqueuing
        if not is_redis_available():
            raise HTTPException(
                status_code=503,
                detail="Task queue service is not available for task resumption."
            )

        # Find the task and verify it's in PENDING_APPROVAL status
        query = text("""
            SELECT task_id, goal, status, pending_action_data, approval_prompt
            FROM agent_tasks
            WHERE task_id = :task_id AND status = 'PENDING_APPROVAL'
        """)

        result = await db_session.execute(query, {"task_id": task_id})
        row = result.fetchone()

        if not row:
            logger.warning(f"Task {task_id} not found or not pending approval")
            raise HTTPException(
                status_code=404,
                detail="Task not found or not in pending approval status."
            )

        # Update task status back to QUEUED
        update_query = text("""
            UPDATE agent_tasks
            SET status = 'QUEUED', updated_at = NOW()
            WHERE task_id = :task_id
        """)

        await db_session.execute(update_query, {"task_id": task_id})
        await db_session.commit()

        # Re-enqueue the task in Redis for the TaskMaster to pick up
        redis: ArqRedis = get_redis_pool()
        job = await redis.enqueue_job(
            'run_agent_task',
            task_id=task_id  # The task will load its state from the database
        )

        logger.info(f"Task {task_id} approved and re-enqueued with job ID: {job.job_id}")

        return {
            "message": f"Task {task_id} has been approved and resumed.",
            "task_id": task_id,
            "goal": row.goal,
            "approval_prompt": row.approval_prompt,
            "new_job_id": job.job_id,
            "status": "approved_and_resumed"
        }

    except HTTPException:
        raise
    except Exception as e:
        await db_session.rollback()
        logger.error(f"Error approving task {task_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while approving the task."
        )


@router.post("/{task_id}/reject")
async def reject_task(
    task_id: str,
    rejection_data: Dict[str, Any] = None,
    db_session: AsyncSession = Depends(get_tenant_db_session),
    tenant: Tenant = Depends(get_current_tenant)
) -> Dict[str, Any]:
    """
    Reject a pending task with optional feedback.

    This endpoint rejects a task that was paused for human review,
    provides the rejection reason to the agent, and re-enqueues it
    for the agent to process the feedback and potentially try a different approach.

    Args:
        task_id: The ID of the task to reject
        rejection_data: Optional dictionary containing rejection reason
        db_session: Tenant-scoped database session
        tenant: Current authenticated tenant

    Returns:
        Confirmation message and task details

    Raises:
        HTTPException: If task not found, not pending approval, or system errors
    """
    try:
        # Extract rejection reason from request body (if provided)
        rejection_reason = "Action rejected by human operator"
        if rejection_data and isinstance(rejection_data, dict):
            rejection_reason = rejection_data.get("reason", rejection_reason)

        logger.info(f"Rejecting task {task_id} for tenant {tenant.name} with reason: {rejection_reason}")

        # Check if Redis is available for re-enqueuing
        if not is_redis_available():
            raise HTTPException(
                status_code=503,
                detail="Task queue service is not available for task resumption."
            )

        # Find the task and verify it's in PENDING_APPROVAL status
        query = text("""
            SELECT task_id, goal, status, pending_action_data, approval_prompt
            FROM agent_tasks
            WHERE task_id = :task_id AND status = 'PENDING_APPROVAL'
        """)

        result = await db_session.execute(query, {"task_id": task_id})
        row = result.fetchone()

        if not row:
            logger.warning(f"Task {task_id} not found or not pending approval")
            raise HTTPException(
                status_code=404,
                detail="Task not found or not in pending approval status."
            )

        # Update task with rejection information
        rejection_data_json = json.dumps({
            "rejection_reason": rejection_reason,
            "rejected_at": json.dumps(datetime.now().isoformat())  # Convert datetime to string for JSON
        })

        update_query = text("""
            UPDATE agent_tasks
            SET status = 'QUEUED',
                pending_action_data = :rejection_data,
                updated_at = NOW()
            WHERE task_id = :task_id
        """)

        await db_session.execute(update_query, {
            "task_id": task_id,
            "rejection_data": rejection_data_json
        })
        await db_session.commit()

        # Re-enqueue the task in Redis for the TaskMaster to pick up
        redis: ArqRedis = get_redis_pool()
        job = await redis.enqueue_job(
            'run_agent_task',
            task_id=task_id  # The task will load its state and process the rejection
        )

        logger.info(f"Task {task_id} rejected and re-enqueued with job ID: {job.job_id}")

        return {
            "message": f"Task {task_id} has been rejected and will receive feedback.",
            "task_id": task_id,
            "goal": row.goal,
            "approval_prompt": row.approval_prompt,
            "rejection_reason": rejection_reason,
            "new_job_id": job.job_id,
            "status": "rejected_with_feedback"
        }

    except HTTPException:
        raise
    except Exception as e:
        await db_session.rollback()
        raise HTTPException(
            status_code=500,
            detail="An error occurred while rejecting the task."
        )
