# src/llmcore/api_server/main.py
"""
Main FastAPI application for the llmcore API server.

This module contains the FastAPI application instance with proper lifecycle
management for the LLMCore instance and all API route definitions.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..api import LLMCore
from ..exceptions import LLMCoreError, ConfigError
from .routes import chat_router, core_router, memory_router

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

    Handles startup (LLMCore initialization) and shutdown (graceful cleanup)
    of the application resources.
    """
    # Startup: Initialize LLMCore and store it in the app's state
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

    yield  # The application runs while in this yield block

    # Shutdown: Cleanly close the LLMCore instance
    logger.info("API Server shutting down...")
    if hasattr(app.state, 'llmcore_instance') and app.state.llmcore_instance:
        try:
            await app.state.llmcore_instance.close()
            logger.info("LLMCore instance successfully closed")
        except Exception as e:
            logger.error(f"Error during LLMCore cleanup: {e}", exc_info=True)
    else:
        logger.info("No LLMCore instance to clean up")


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

# Include routers under the v1 API prefix
app.include_router(core_router, prefix="/api/v1", tags=["core_v1"])
app.include_router(chat_router, prefix="/api/v1", tags=["chat_v1"])

# Add the new v2 router for memory operations
app.include_router(memory_router, prefix="/api/v2", tags=["memory_v2"])


@app.get("/")
async def root() -> Dict[str, str]:
    """
    Root endpoint providing basic service information.
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
    """
    llmcore_instance = getattr(app.state, 'llmcore_instance', None)

    if llmcore_instance:
        # Basic health check - could be expanded to test actual functionality
        available_providers = llmcore_instance.get_available_providers()
        return {
            "status": "healthy",
            "llmcore_available": True,
            "providers": available_providers
        }
    else:
        return {
            "status": "degraded",
            "llmcore_available": False,
            "providers": []
        }
