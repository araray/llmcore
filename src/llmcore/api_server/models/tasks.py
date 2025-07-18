# src/llmcore/api_server/models/tasks.py
"""
Pydantic models for task management API endpoints.

This module defines the request and response models for the asynchronous
task management system, ensuring proper data validation and clear
API contracts for task operations.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TaskSubmissionRequest(BaseModel):
    """
    Request model for task submission.

    This model defines the parameters for submitting a new asynchronous task
    to the TaskMaster service via the API.
    """
    task_name: str = Field(description="The name of the task function to execute.")
    args: List[Any] = Field(
        default_factory=list,
        description="Positional arguments for the task."
    )
    kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments for the task."
    )

    class Config:
        extra = 'forbid'  # Reject unknown fields for strict validation


class TaskSubmissionResponse(BaseModel):
    """
    Response model for task submission.

    Returned when a task is successfully enqueued.
    """
    task_id: str = Field(description="The unique ID of the submitted task.")
    status: str = Field(
        default="queued",
        description="The initial status of the task."
    )


class TaskStatusResponse(BaseModel):
    """
    Response model for task status queries.

    Provides information about the current state of a task.
    """
    task_id: str = Field(description="The unique ID of the task.")
    status: str = Field(description="The current status of the task.")
    result_available: bool = Field(
        description="Whether the task result is available for retrieval."
    )


class TaskResultResponse(BaseModel):
    """
    Response model for task result retrieval.

    Contains the final result of a completed task.
    """
    task_id: str = Field(description="The unique ID of the task.")
    status: str = Field(description="The status of the task.")
    result: Any = Field(description="The result data from the completed task.")
