# src/llmcore/api_server/routes/hitl.py
"""
API routes for the Human-in-the-Loop (HITL) workflow.

This module provides secure endpoints for managing agent tasks that have been
paused and are awaiting human review and approval or rejection. This is a critical
component for ensuring safe and controlled autonomous agent operations.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

try:
    from arq.connections import ArqRedis
except ImportError:
    ArqRedis = None

from ..auth import get_current_tenant
from ..db import get_tenant_db_session
from ..schemas.security import Tenant
from ..services.redis_client import get_redis_pool, is_redis_available

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/tasks/pending_approval")
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
        db_session: Tenant-scoped database session.
        tenant: Current authenticated tenant.

    Returns:
        Dictionary containing pending tasks and their approval prompts.

    Raises:
        HTTPException: For database errors or service unavailability.
    """
    try:
        logger.info(f"Listing pending approval tasks for tenant: {tenant.name}")

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


@router.post("/tasks/{task_id}/approve")
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
        task_id: The ID of the task to approve.
        db_session: Tenant-scoped database session.
        tenant: Current authenticated tenant.

    Returns:
        Confirmation message and task details.

    Raises:
        HTTPException: If task not found, not pending approval, or system errors.
    """
    try:
        logger.info(f"Approving task {task_id} for tenant {tenant.name}")

        if not is_redis_available():
            raise HTTPException(status_code=503, detail="Task queue service is not available for task resumption.")

        query = text("""
            SELECT task_id, goal, status, approval_prompt
            FROM agent_tasks
            WHERE task_id = :task_id AND status = 'PENDING_APPROVAL'
        """)
        result = await db_session.execute(query, {"task_id": task_id})
        row = result.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Task not found or not in pending approval status.")

        update_query = text("UPDATE agent_tasks SET status = 'QUEUED', updated_at = NOW() WHERE task_id = :task_id")
        await db_session.execute(update_query, {"task_id": task_id})
        await db_session.commit()

        redis: ArqRedis = get_redis_pool()
        job = await redis.enqueue_job('run_agent_task', task_id=task_id)

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
        raise HTTPException(status_code=500, detail="An error occurred while approving the task.")


@router.post("/tasks/{task_id}/reject")
async def reject_task(
    task_id: str,
    rejection_data: Dict[str, Any],
    db_session: AsyncSession = Depends(get_tenant_db_session),
    tenant: Tenant = Depends(get_current_tenant)
) -> Dict[str, Any]:
    """
    Reject a pending task with optional feedback.

    This endpoint rejects a task, provides the rejection reason to the agent,
    and re-enqueues it for the agent to process the feedback.

    Args:
        task_id: The ID of the task to reject.
        rejection_data: Dictionary containing a "reason" for the rejection.
        db_session: Tenant-scoped database session.
        tenant: Current authenticated tenant.

    Returns:
        Confirmation message and task details.

    Raises:
        HTTPException: If task not found, not pending approval, or system errors.
    """
    try:
        rejection_reason = rejection_data.get("reason", "Action rejected by human operator")
        logger.info(f"Rejecting task {task_id} for tenant {tenant.name} with reason: {rejection_reason}")

        if not is_redis_available():
            raise HTTPException(status_code=503, detail="Task queue service is not available for task resumption.")

        query = text("""
            SELECT task_id, goal, status, approval_prompt
            FROM agent_tasks
            WHERE task_id = :task_id AND status = 'PENDING_APPROVAL'
        """)
        result = await db_session.execute(query, {"task_id": task_id})
        row = result.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Task not found or not in pending approval status.")

        rejection_data_json = json.dumps({
            "rejection_reason": rejection_reason,
            "rejected_at": datetime.now().isoformat()
        })
        update_query = text("""
            UPDATE agent_tasks
            SET status = 'QUEUED', pending_action_data = :rejection_data, updated_at = NOW()
            WHERE task_id = :task_id
        """)
        await db_session.execute(update_query, {"task_id": task_id, "rejection_data": rejection_data_json})
        await db_session.commit()

        redis: ArqRedis = get_redis_pool()
        job = await redis.enqueue_job('run_agent_task', task_id=task_id)

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
        raise HTTPException(status_code=500, detail="An error occurred while rejecting the task.")
