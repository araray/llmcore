# src/llmcore/task_master/worker.py
"""
TaskMaster Worker Configuration and Task Definitions.

This module defines the arq worker configuration and implements sample tasks
to validate the task queue functionality. This serves as the foundation for
all future long-running operations in the llmcore ecosystem.
"""

import asyncio
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


async def sample_task(ctx: Dict[str, Any], x: int, y: int) -> int:
    """
    A simple asynchronous task for testing the arq worker.

    This task demonstrates the basic functionality of the TaskMaster service
    by performing a simple calculation with a simulated delay to represent
    I/O-bound work.

    Args:
        ctx: The arq context dictionary (contains job info, Redis connection, etc.)
        x: First integer operand
        y: Second integer operand

    Returns:
        The sum of x and y

    Example:
        This task would typically be enqueued from the API server:
        ```python
        await redis_pool.enqueue_job('sample_task', 5, 10)
        ```
    """
    logger.info(f"Executing sample_task with arguments: x={x}, y={y}")

    # Simulate I/O-bound work (e.g., network request, file operation, etc.)
    await asyncio.sleep(2)

    result = x + y
    logger.info(f"sample_task completed with result: {result}")

    return result


async def startup(ctx: Dict[str, Any]) -> None:
    """
    Worker startup function.

    This function is called when the arq worker process starts up.
    It can be used to initialize shared resources, database connections,
    or in future iterations, a shared LLMCore instance for the worker process.

    Args:
        ctx: The arq context dictionary
    """
    logger.info("TaskMaster worker is starting up...")

    # Future enhancement: Initialize shared LLMCore instance
    # ctx['llmcore_instance'] = await LLMCore.create()

    logger.info("TaskMaster worker startup complete.")


async def shutdown(ctx: Dict[str, Any]) -> None:
    """
    Worker shutdown function.

    This function is called when the arq worker process is shutting down.
    It should be used to clean up resources, close connections, and ensure
    a graceful shutdown of any long-running operations.

    Args:
        ctx: The arq context dictionary
    """
    logger.info("TaskMaster worker is shutting down...")

    # Future enhancement: Clean up shared LLMCore instance
    # if 'llmcore_instance' in ctx:
    #     await ctx['llmcore_instance'].close()

    logger.info("TaskMaster worker shutdown complete.")


class WorkerSettings:
    """
    Configuration class for the arq worker.

    This class defines the tasks, startup/shutdown hooks, and Redis connection
    settings for the TaskMaster worker. The Redis connection settings are
    automatically picked up by arq from the REDIS_URL environment variable
    or default to localhost:6379.

    Attributes:
        functions: List of task functions available to the worker
        on_startup: Function to call when the worker starts
        on_shutdown: Function to call when the worker shuts down
        redis_settings: Redis connection configuration (auto-detected)

    Environment Variables:
        REDIS_URL: Redis connection URL (optional, defaults to redis://localhost:6379)

    Example Usage:
        To start the worker:
        ```bash
        python -m llmcore.task_master.main
        ```

        Or programmatically:
        ```python
        from arq import run_worker
        from llmcore.task_master.worker import WorkerSettings

        await run_worker(WorkerSettings)
        ```
    """

    # List of task functions that the worker can execute
    functions = [sample_task]

    # Lifecycle hooks
    on_startup = startup
    on_shutdown = shutdown

    # Redis connection settings will be automatically picked up by arq
    # from the REDIS_URL environment variable or default to localhost:6379
    #
    # For custom Redis settings, you can uncomment and modify:
    # redis_settings = {
    #     'host': 'localhost',
    #     'port': 6379,
    #     'database': 0,
    # }
