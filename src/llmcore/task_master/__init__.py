# src/llmcore/task_master/__init__.py
"""
TaskMaster - Asynchronous Task Queue System for llmcore.

This module provides the foundation for running long-running, non-blocking tasks
within the llmcore ecosystem using the arq (async-redis-queue) library.
"""

from .worker import WorkerSettings
from .tasks.ingestion import ingest_data_task
from .tasks.agent import run_agent_task  # Add agent task import

__all__ = [
    "WorkerSettings",
    "ingest_data_task",
    "run_agent_task",  # Add agent task export
]
