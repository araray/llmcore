# llmcore/src/llmcore/task_master/worker.py
"""
TaskMaster worker configuration and task registration.

This module configures the arq worker that processes background tasks
including data ingestion, agent execution, tenant provisioning, and other long-running operations.

UPDATED: Added provision_tenant_task registration for multi-tenant schema provisioning.
"""

import logging
from arq import create_pool
from arq.connections import RedisSettings
from arq.worker import Worker

from llmcore.api import LLMCore
from .tasks.ingestion import ingest_data_task
from .tasks.agent import run_agent_task
from .tasks.provisioning import provision_tenant_task  # NEW: Tenant provisioning task

logger = logging.getLogger(__name__)


async def startup(ctx):
    """
    Worker startup function that initializes shared resources.

    Creates a shared LLMCore instance that will be available to all tasks
    in this worker process through the context.
    """
    logger.info("TaskMaster worker is starting up.")
    # Create a shared LLMCore instance for all tasks in this worker
    ctx['llmcore_instance'] = await LLMCore.create()
    logger.info("LLMCore instance created and stored in worker context.")


async def shutdown(ctx):
    """
    Worker shutdown function that cleans up shared resources.
    """
    logger.info("TaskMaster worker is shutting down.")
    if 'llmcore_instance' in ctx:
        await ctx['llmcore_instance'].close()
        logger.info("LLMCore instance closed.")


async def sample_task(ctx, x, y):
    """Sample task for testing the task queue functionality."""
    logger.info(f"Sample task called with x={x}, y={y}")
    return x + y


class WorkerSettings:
    """
    Configuration settings for the arq worker.
    """
    # Redis connection settings
    redis_settings = RedisSettings(host='localhost', port=6379, database=0)

    # Task functions to register with the worker
    functions = [
        sample_task,
        ingest_data_task,
        run_agent_task,
        provision_tenant_task,  # NEW: Added tenant provisioning task
    ]

    # Worker lifecycle functions
    on_startup = startup
    on_shutdown = shutdown

    # Worker configuration
    max_jobs = 10
    job_timeout = 3600  # 1 hour timeout for long-running tasks
    keep_result = 3600  # Keep results for 1 hour

    # Logging
    log_results = True


if __name__ == "__main__":
    """
    Entry point for running the worker directly.
    This allows the worker to be started with: python -m llmcore.task_master.worker
    """
    import asyncio

    async def main():
        logger.info("Starting TaskMaster worker...")
        worker = Worker(WorkerSettings)
        await worker.async_run()

    asyncio.run(main())
