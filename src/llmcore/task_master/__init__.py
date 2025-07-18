# src/llmcore/task_master/__init__.py
"""
TaskMaster - Asynchronous Task Queue System for llmcore.

This module provides the foundation for running long-running, non-blocking tasks
within the llmcore ecosystem using the arq (async-redis-queue) library.
"""

from .worker import WorkerSettings

__all__ = ["WorkerSettings"]
