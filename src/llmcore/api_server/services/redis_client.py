# src/llmcore/api_server/services/redis_client.py
"""
Redis client service for arq task queue management.

This module provides a centralized service to manage the arq Redis pool,
ensuring it's a shared resource across the FastAPI application lifecycle.
"""

import logging
from typing import Optional

try:
    from arq import create_pool
    from arq.connections import ArqRedis, RedisSettings
except ImportError as e:
    raise ImportError(
        "arq library is required for task management. Install with: pip install arq>=0.25.0"
    ) from e

logger = logging.getLogger(__name__)

# Global Redis pool instance
redis_pool: Optional[ArqRedis] = None


async def initialize_redis_pool() -> None:
    """
    Initialize the global Redis pool for arq task queue operations.

    This function creates and stores a Redis pool that can be used throughout
    the application for enqueuing jobs and querying job status. The pool
    uses the default Redis settings which can be configured via environment
    variables (REDIS_URL).

    Raises:
        ConnectionError: If unable to connect to Redis
        ImportError: If arq dependencies are not installed
    """
    global redis_pool

    if redis_pool is not None:
        logger.warning("Redis pool is already initialized")
        return

    try:
        logger.info("Initializing Redis pool for task queue...")

        # Use default RedisSettings which automatically picks up REDIS_URL
        # environment variable or defaults to localhost:6379
        redis_pool = await create_pool(RedisSettings())

        # Test the connection
        await redis_pool.ping()

        logger.info("Redis pool successfully initialized and tested")

    except Exception as e:
        logger.error(f"Failed to initialize Redis pool: {e}", exc_info=True)
        redis_pool = None
        raise ConnectionError(f"Could not connect to Redis: {e}") from e


async def close_redis_pool() -> None:
    """
    Close the global Redis pool and clean up connections.

    This function should be called during application shutdown to ensure
    all Redis connections are properly closed and resources are freed.
    """
    global redis_pool

    if redis_pool is None:
        logger.debug("No Redis pool to close")
        return

    try:
        logger.info("Closing Redis pool...")
        await redis_pool.close()
        redis_pool = None
        logger.info("Redis pool successfully closed")

    except Exception as e:
        logger.error(f"Error closing Redis pool: {e}", exc_info=True)
        # Still set to None to prevent reuse
        redis_pool = None


def get_redis_pool() -> ArqRedis:
    """
    Get the global Redis pool instance.

    Returns:
        ArqRedis: The active Redis pool for task operations

    Raises:
        RuntimeError: If the Redis pool is not initialized
    """
    if redis_pool is None:
        raise RuntimeError(
            "Redis pool is not initialized. Call initialize_redis_pool() first."
        )

    return redis_pool


def is_redis_available() -> bool:
    """
    Check if the Redis pool is available and ready for use.

    Returns:
        bool: True if Redis pool is initialized and ready, False otherwise
    """
    return redis_pool is not None
