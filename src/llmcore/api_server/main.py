# src/llmcore/api_server/main.py
"""
Main FastAPI application for the llmcore API server.

This module contains the FastAPI application instance with proper lifecycle
management for the LLMCore instance and all API route definitions.

UPDATED: Added comprehensive observability stack with structured logging,
Prometheus metrics, and distributed tracing integration.
"""

import asyncio
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
from .db import initialize_tenant_db_session
from .middleware.observability import ObservabilityMiddleware
from ..tracing import configure_tracer

# Import observability components with fallback handling
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    from .metrics import initialize_system_info, update_queue_depth_metrics
    PROMETHEUS_AVAILABLE = True
except ImportError:
    Instrumentator = None
    PROMETHEUS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global task for metrics updates
metrics_update_task = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifecycle of the FastAPI application.

    Handles startup (LLMCore initialization, Redis pool setup, authentication DB setup,
    tenant database session factory setup, and observability stack initialization) and
    shutdown (graceful cleanup) of the application resources.

    UPDATED: Added comprehensive observability stack initialization including tracing,
    metrics, and background metrics collection.
    """
    global metrics_update_task

    # Startup: Initialize all components
    logger.info("API Server starting up...")

    # Step 1: Initialize distributed tracing
    try:
        logger.info("Initializing distributed tracing...")
        configure_tracer("llmcore-api")
        logger.info("Distributed tracing successfully initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize tracing: {e}")

    # Step 2: Initialize LLMCore instance
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

    # Step 3: Initialize Redis pool for task queue
    try:
        logger.info("Initializing Redis pool for task queue...")
        await initialize_redis_pool()
        logger.info("Redis pool successfully initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Redis pool: {e}", exc_info=True)
        logger.warning("API server will start but task queue will be unavailable")

    # Step 4: Initialize authentication database session
    try:
        logger.info("Initializing authentication database session...")
        # Get database URL from environment or configuration
        database_url = os.environ.get(
            'LLMCORE_AUTH_DATABASE_URL',
            'postgresql+asyncpg://postgres:password@localhost:5432/llmcore'
        )
        initialize_auth_db_session(database_url)
        logger.info("Authentication database session successfully initialized")
    except Exception as e:
        logger.error(f"Failed to initialize authentication database: {e}", exc_info=True)
        logger.warning("API server will start but authentication will be unavailable")

    # Step 5: Initialize tenant database session factory
    try:
        logger.info("Initializing tenant database session factory...")
        # Use the same database URL for tenant operations
        tenant_database_url = os.environ.get(
            'LLMCORE_TENANT_DATABASE_URL',
            os.environ.get('LLMCORE_AUTH_DATABASE_URL',
                           'postgresql+asyncpg://postgres:password@localhost:5432/llmcore')
        )
        initialize_tenant_db_session(tenant_database_url)
        logger.info("Tenant database session factory successfully initialized")
    except Exception as e:
        logger.error(f"Failed to initialize tenant database session factory: {e}", exc_info=True)
        logger.warning("API server will start but tenant-scoped operations will be unavailable")

    # Step 6: Initialize Prometheus metrics
    if PROMETHEUS_AVAILABLE:
        try:
            logger.info("Initializing Prometheus metrics...")
            initialize_system_info()

            # Start background task for queue depth metrics
            metrics_update_task = asyncio.create_task(metrics_update_loop())
            logger.info("Prometheus metrics and background collection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Prometheus metrics: {e}", exc_info=True)
    else:
        logger.warning("Prometheus instrumentation not available - metrics will be disabled")

    logger.info("API Server startup complete")

    yield  # The application runs while in this yield block

    # Shutdown: Cleanly close all resources
    logger.info("API Server shutting down...")

    # Stop metrics update task
    if metrics_update_task and not metrics_update_task.done():
        try:
            metrics_update_task.cancel()
            await metrics_update_task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error stopping metrics update task: {e}")

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

    logger.info("API Server shutdown complete")


async def metrics_update_loop():
    """
    Background task to periodically update metrics that require polling.
    """
    logger.info("Starting metrics update background task")

    try:
        while True:
            await update_queue_depth_metrics()
            await asyncio.sleep(30)  # Update every 30 seconds
    except asyncio.CancelledError:
        logger.info("Metrics update task cancelled")
        raise
    except Exception as e:
        logger.error(f"Error in metrics update loop: {e}", exc_info=True)


# Create the FastAPI application
app = FastAPI(
    title="llmcore API",
    description="A unified, flexible API for interacting with various Large Language Models (LLMs)",
    version="2.0.0",
    lifespan=lifespan
)

# Add observability middleware (must be added early in the middleware stack)
app.add_middleware(
    ObservabilityMiddleware,
    enable_request_logging=True
)

# Add CORS middleware for web client compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Prometheus instrumentation if available
if PROMETHEUS_AVAILABLE:
    try:
        instrumentator = Instrumentator(
            should_group_status_codes=False,
            should_ignore_untemplated=True,
            should_instrument_requests_inprogress=True,
            should_instrument_requests_request_size=True,
            should_instrument_requests_response_size=True,
            excluded_handlers=["/metrics", "/health"],
        )
        instrumentator.instrument(app).expose(app)
        logger.info("Prometheus FastAPI instrumentation enabled on /metrics endpoint")
    except Exception as e:
        logger.error(f"Failed to initialize Prometheus instrumentation: {e}")

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
        "version": "2.0.0",
        "docs_url": "/docs",
        "observability_enabled": PROMETHEUS_AVAILABLE
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
            "authentication": "enabled",
            "multi_tenancy": "enabled",
            "observability": {
                "structured_logging": True,
                "distributed_tracing": True,
                "prometheus_metrics": PROMETHEUS_AVAILABLE
            }
        }
    else:
        from .services.redis_client import is_redis_available
        redis_available = is_redis_available()

        return {
            "status": "degraded",
            "llmcore_available": False,
            "providers": [],
            "task_queue_available": redis_available,
            "authentication": "enabled",
            "multi_tenancy": "enabled",
            "observability": {
                "structured_logging": True,
                "distributed_tracing": True,
                "prometheus_metrics": PROMETHEUS_AVAILABLE
            }
        }


@app.get("/metrics")
async def metrics_endpoint():
    """
    Prometheus metrics endpoint.

    This endpoint is automatically exposed by the prometheus-fastapi-instrumentator
    but we define it here for documentation purposes.
    """
    pass  # The actual implementation is handled by the instrumentator
