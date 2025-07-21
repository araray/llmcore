# llmcore/src/llmcore/api_server/routes/__init__.py
"""
API routes package initialization.

This module exports all the API routers for easy import and registration
with the main FastAPI application.
"""

from .chat import router as chat_router
from .core import router as core_router
from .memory import router as memory_router
from .tasks import router as tasks_router
from .ingestion import router as ingestion_router
from .agents import router as agents_router  # Add agents router import

__all__ = [
    "chat_router",
    "core_router",
    "memory_router",
    "tasks_router",
    "ingestion_router",
    "agents_router",  # Add agents router export
]
