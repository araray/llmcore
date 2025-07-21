# src/llmcore/task_master/main.py
"""
Main entry point for the TaskMaster arq worker.

This module provides the command-line interface for starting the TaskMaster
worker process. The worker connects to Redis and listens for jobs enqueued
by the llmcore API server or other producers.

Usage:
    python -m llmcore.task_master.main

    Or from the command line after installation:
    python -m llmcore.task_master.main

Environment Variables:
    REDIS_URL: Redis connection URL (default: redis://localhost:6379)
    LOG_LEVEL: Logging level (default: INFO)

Example:
    # Start the worker with default settings
    python -m llmcore.task_master.main

    # Start with custom Redis URL
    REDIS_URL=redis://redis-server:6379/1 python -m llmcore.task_master.main
"""

import asyncio
import logging
import os
import sys
from typing import NoReturn

try:
    from arq import run_worker
except ImportError as e:
    print(f"Error: arq library is not installed. Please install with: pip install arq>=0.25.0")
    print(f"Original error: {e}")
    sys.exit(1)

try:
    from .worker import WorkerSettings
except ImportError:
    # Handle relative import when running as script
    from worker import WorkerSettings


def setup_logging() -> None:
    """
    Configure logging for the TaskMaster worker.

    Sets up structured logging with appropriate formatters and handlers
    for both development and production environments.
    """
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - [TaskMaster] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Set specific log levels for noisy libraries
    logging.getLogger('arq').setLevel(logging.INFO)
    logging.getLogger('redis').setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(f"TaskMaster logging configured at {log_level} level")


def main() -> NoReturn:
    """
    Main entry point to start the TaskMaster arq worker.

    This function:
    1. Sets up logging configuration
    2. Validates the Redis connection
    3. Starts the arq worker with the defined WorkerSettings

    The worker will run indefinitely until interrupted (Ctrl+C) or
    terminated by a signal.

    Raises:
        SystemExit: If there's an error starting the worker
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting TaskMaster worker...")
    logger.info(f"Redis URL: {os.getenv('REDIS_URL', 'redis://localhost:6379 (default)')}")

    try:
        # Use asyncio.run() for the top-level async function
        # This ensures proper cleanup of async resources
        asyncio.run(run_worker(WorkerSettings))

    except KeyboardInterrupt:
        logger.info("TaskMaster worker interrupted by user (Ctrl+C)")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error starting TaskMaster worker: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
