# src/llmcore/api_server/routes/__init__.py
"""
API routes package initialization.

This module exports all the API routers for easy import and registration
with the main FastAPI application.

UPDATED: Replaced the monolithic tools_router with specialized routers for
         tool_management and toolkit_management.
UPDATED: Added hitl_router for Human-in-the-Loop workflows.
"""

from .chat import router as chat_router
from .core import router as core_router
from .memory import router as memory_router
from .tasks import router as tasks_router
from .ingestion import router as ingestion_router
from .agents import router as agents_router
from .tool_management import router as tool_management_router
from .toolkit_management import router as toolkit_management_router
from .hitl import router as hitl_router

__all__ = [
    "chat_router",
    "core_router",
    "memory_router",
    "tasks_router",
    "ingestion_router",
    "agents_router",
    "tool_management_router",
    "toolkit_management_router",
    "hitl_router",
]
