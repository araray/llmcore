# src/llmcore/api_server/main.py
"""
Main FastAPI application for the llmcore API server.

This module contains the FastAPI application instance with proper lifecycle
management for the LLMCore instance and all API route definitions.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from ..api import LLMCore
from ..exceptions import LLMCoreError, ConfigError
from .routes import chat_router, core_router, ingestion_router, memory_router, tasks_router, agents_router
from .services.redis_client import initialize_redis_pool, close_redis_pool
from .auth import get_current_tenant, initialize_auth_db_session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifecycle of the FastAPI application.

    Handles startup (LLMCore initialization, Redis pool setup, and authentication DB setup)
    and shutdown (graceful cleanup) of the application resources.
    """
    # Startup: Initialize LLMCore, Redis pool, and authentication database
    logger.info("API Server starting up...")

    try:
        logger.info("Initializing LLMCore instance...")
        llmcore_instance = await LLMCore.create()
        app.state.llmcore_instance = llmcore_instance
        logger.info("LLMCore instance successfully created and attached to app state")

        # Log available providers for debugging
        available_providers = llmcore_instance.get_available_providers()
        logger.info(f"Available LLM providers: {available_providers}")

    except (ConfigError, LLMCoreError) as e:
        logger.critical(f"Fatal error during LLMCore initialization: {e}", exc_info=True)
        app.state.llmcore_instance = None
        # Don't raise here - let the server start but mark service as unavailable
        logger.warning("API server will start but LLMCore service will be unavailable")
    except Exception as e:
        logger.critical(f"Unexpected error during LLMCore initialization: {e}", exc_info=True)
        app.state.llmcore_instance = None
        logger.warning("API server will start but LLMCore service will be unavailable")

    # Initialize Redis pool for task queue
    try:
        logger.info("Initializing Redis pool for task queue...")
        await initialize_redis_pool()
        logger.info("Redis pool successfully initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Redis pool: {e}", exc_info=True)
        logger.warning("API server will start but task queue will be unavailable")

    # Initialize authentication database session
    try:
        logger.info("Initializing authentication database session...")
        # Get database URL from environment or configuration
        # For now, using a default - this should be configured properly in production
        database_url = os.environ.get(
            'LLMCORE_AUTH_DATABASE_URL',
            'postgresql+asyncpg://postgres:password@localhost:5432/llmcore'
        )
        initialize_auth_db_session(database_url)
        logger.info("Authentication database session successfully initialized")
    except Exception as e:
        logger.error(f"Failed to initialize authentication database: {e}", exc_info=True)
        logger.warning("API server will start but authentication will be unavailable")

    yield  # The application runs while in this yield block

    # Shutdown: Cleanly close the LLMCore instance and Redis pool
    logger.info("API Server shutting down...")

    # Close LLMCore instance
    if hasattr(app.state, 'llmcore_instance') and app.state.llmcore_instance:
        try:
            await app.state.llmcore_instance.close()
            logger.info("LLMCore instance successfully closed")
        except Exception as e:
            logger.error(f"Error during LLMCore cleanup: {e}", exc_info=True)
    else:
        logger.info("No LLMCore instance to clean up")

    # Close Redis pool
    try:
        await close_redis_pool()
        logger.info("Redis pool successfully closed")
    except Exception as e:
        logger.error(f"Error during Redis pool cleanup: {e}", exc_info=True)


# Create the FastAPI application
app = FastAPI(
    title="llmcore API",
    description="A unified, flexible API for interacting with various Large Language Models (LLMs)",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for web client compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with security dependency applied to protected endpoints
# Note: Root and health endpoints remain public as specified in the requirements

# V1 API routes - secured
app.include_router(
    core_router,
    prefix="/api/v1",
    tags=["core_v1"],
    dependencies=[Depends(get_current_tenant)]
)
app.include_router(
    chat_router,
    prefix="/api/v1",
    tags=["chat_v1"],
    dependencies=[Depends(get_current_tenant)]
)

# V2 API routes - secured
app.include_router(
    memory_router,
    prefix="/api/v2",
    tags=["memory_v2"],
    dependencies=[Depends(get_current_tenant)]
)
app.include_router(
    tasks_router,
    prefix="/api/v2",
    tags=["tasks_v2"],
    dependencies=[Depends(get_current_tenant)]
)
app.include_router(
    ingestion_router,
    prefix="/api/v2/ingestion",
    tags=["ingestion_v2"],
    dependencies=[Depends(get_current_tenant)]
)
app.include_router(
    agents_router,
    prefix="/api/v2",
    tags=["agents_v2"],
    dependencies=[Depends(get_current_tenant)]
)


@app.get("/")
async def root() -> Dict[str, str]:
    """
    Root endpoint providing basic service information.

    This endpoint remains public and does not require authentication.
    """
    return {
        "message": "llmcore API is running",
        "version": "1.0.0",
        "docs_url": "/docs"
    }


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for monitoring and load balancers.

    This endpoint remains public and does not require authentication.
    """
    llmcore_instance = getattr(app.state, 'llmcore_instance', None)

    if llmcore_instance:
        # Basic health check - could be expanded to test actual functionality
        available_providers = llmcore_instance.get_available_providers()

        # Check Redis availability for task queue
        from .services.redis_client import is_redis_available
        redis_available = is_redis_available()

        return {
            "status": "healthy",
            "llmcore_available": True,
            "providers": available_providers,
            "task_queue_available": redis_available,
            "authentication": "enabled"
        }
    else:
        from .services.redis_client import is_redis_available
        redis_available = is_redis_available()

        return {
            "status": "degraded",
            "llmcore_available": False,
            "providers": [],
            "task_queue_available": redis_available,
            "authentication": "enabled"
        }
