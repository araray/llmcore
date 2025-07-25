# llmcore/src/llmcore/api_server/models/__init__.py
"""
API models package initialization.

This module exports all the Pydantic models used for API request/response validation
and data structures across the different API endpoints.

UPDATED: Added tools models for dynamic tool management.
"""

from .core import *
from .memory import *
from .tasks import *
from .ingestion import *
from .agents import *
from .tools import *  # NEW: Add tools models import

__all__ = [
    # Core models
    "ChatRequest",
    "ChatResponse",
    "ErrorResponse",

    # Memory models
    "SemanticSearchRequest",

    # Task models
    "TaskSubmissionRequest",
    "TaskSubmissionResponse",
    "TaskStatusResponse",
    "TaskResultResponse",

    # Ingestion models
    "FileIngestionRequest",
    "DirectoryZipIngestionRequest",
    "GitIngestionRequest",
    "IngestionSubmitRequest",
    "IngestionSubmitResponse",
    "IngestionResult",

    # Agent models
    "AgentRunRequest",

    # NEW: Tool management models
    "ToolCreateRequest",
    "ToolUpdateRequest",
    "ToolResponse",
    "ToolkitCreateRequest",
    "ToolkitUpdateRequest",
    "ToolkitResponse",
    "ToolExecutionRequest",
    "AvailableImplementationsResponse",
]
