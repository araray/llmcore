# llmcore/src/llmcore/task_master/worker.py
"""
TaskMaster worker configuration and task registration.

This module configures the arq worker that processes background tasks
including data ingestion, agent execution, tenant provisioning, and other long-running operations.

UPDATED: Added distributed tracing initialization and trace context propagation
for end-to-end observability across API server and worker processes.
"""

import logging
from typing import Dict, Any
from arq import create_pool
from arq.connections import RedisSettings
from arq.worker import Worker

from llmcore.api import LLMCore
from ..tracing import configure_tracer, extract_and_set_trace_context
from .tasks.ingestion import ingest_data_task
from .tasks.agent import run_agent_task
from .tasks.provisioning import provision_tenant_task

logger = logging.getLogger(__name__)


async def startup(ctx):
    """
    Worker startup function that initializes shared resources.

    Creates a shared LLMCore instance and configures distributed tracing
    for all tasks in this worker process.

    UPDATED: Added distributed tracing initialization for the worker.
    """
    logger.info("TaskMaster worker is starting up.")

    # Initialize distributed tracing for the worker
    try:
        logger.info("Initializing distributed tracing for TaskMaster worker...")
        configure_tracer("llmcore-worker")
        logger.info("Distributed tracing successfully initialized for worker")
    except Exception as e:
        logger.warning(f"Failed to initialize tracing for worker: {e}")

    # Create a shared LLMCore instance for all tasks in this worker
    try:
        ctx['llmcore_instance'] = await LLMCore.create()
        logger.info("LLMCore instance created and stored in worker context.")
    except Exception as e:
        logger.error(f"Failed to create LLMCore instance in worker: {e}", exc_info=True)
        ctx['llmcore_instance'] = None


async def shutdown(ctx):
    """
    Worker shutdown function that cleans up shared resources.
    """
    logger.info("TaskMaster worker is shutting down.")
    if 'llmcore_instance' in ctx:
        try:
            await ctx['llmcore_instance'].close()
            logger.info("LLMCore instance closed.")
        except Exception as e:
            logger.error(f"Error closing LLMCore instance: {e}")


async def before_job_run(ctx: Dict[str, Any], job_id: str, **kwargs) -> None:
    """
    Hook that runs before each job execution.

    This hook extracts and applies trace context from the job payload,
    enabling distributed tracing across the API server and worker processes.

    Args:
        ctx: Worker context containing shared resources
        job_id: Unique identifier for the job
        **kwargs: Job arguments that may contain trace context
    """
    try:
        # Extract trace context from job kwargs if present
        trace_context = kwargs.get('_trace_context')
        if trace_context:
            logger.debug(f"Extracting trace context for job {job_id}")
            extract_and_set_trace_context(trace_context)
        else:
            logger.debug(f"No trace context found for job {job_id}")

    except Exception as e:
        logger.debug(f"Failed to extract trace context for job {job_id}: {e}")


async def after_job_run(ctx: Dict[str, Any], job_id: str, **kwargs) -> None:
    """
    Hook that runs after each job execution.

    This hook can be used for cleanup or additional logging after job completion.

    Args:
        ctx: Worker context containing shared resources
        job_id: Unique identifier for the job
        **kwargs: Job arguments and results
    """
    try:
        logger.debug(f"Completed job {job_id}")
    except Exception as e:
        logger.debug(f"Error in after_job_run hook for job {job_id}: {e}")


async def sample_task(ctx, x, y):
    """Sample task for testing the task queue functionality."""
    logger.info(f"Sample task called with x={x}, y={y}")
    return x + y


class WorkerSettings:
    """
    Configuration settings for the arq worker.

    UPDATED: Added before_job_run and after_job_run hooks for trace context propagation.
    """
    # Redis connection settings
    redis_settings = RedisSettings(host='localhost', port=6379, database=0)

    # Task functions to register with the worker
    functions = [
        sample_task,
        ingest_data_task,
        run_agent_task,
        provision_tenant_task,
    ]

    # Worker lifecycle functions
    on_startup = startup
    on_shutdown = shutdown

    # Job lifecycle hooks for trace context propagation
    on_job_start = before_job_run
    on_job_end = after_job_run

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
