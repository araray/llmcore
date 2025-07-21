# src/llmcore/api_server/routes/core.py
"""
Core API routes for the llmcore API server.

This module contains core endpoints that provide metadata and introspection
capabilities for the API service.
"""

import logging
from typing import Dict, Any, List

from fastapi import APIRouter, Request, HTTPException

from ... import __version__ as llmcore_version

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/info")
async def get_api_info(request: Request) -> Dict[str, Any]:
    """
    Returns API version information and service capabilities.

    This endpoint provides clients with essential metadata about the running
    service, including API version, llmcore library version, and available features.

    Args:
        request: The FastAPI request object containing app state

    Returns:
        Dictionary containing version info and feature list

    Raises:
        HTTPException: If the service is not properly initialized
    """
    try:
        # Get the LLMCore instance from app state
        llmcore_instance = getattr(request.app.state, 'llmcore_instance', None)

        if not llmcore_instance:
            logger.warning("LLMCore instance not available for info endpoint")
            # Still return basic info even if LLMCore is not available
            return {
                "api_version": "1.0",
                "llmcore_version": llmcore_version,
                "service_status": "degraded",
                "features": {
                    "providers": [],
                    "chat": False,
                    "streaming": False,
                    "session_management": False
                }
            }

        # Get available providers
        try:
            available_providers = llmcore_instance.get_available_providers()
        except Exception as e:
            logger.warning(f"Could not get available providers: {e}")
            available_providers = []

        # Build feature set based on available functionality
        features = {
            "providers": available_providers,
            "chat": True,
            "streaming": True,
            "session_management": True,
            "vector_storage": True,  # Assuming vector storage is available
            "context_management": True,
            "rag": True  # Retrieval Augmented Generation
        }

        return {
            "api_version": "1.0",
            "llmcore_version": llmcore_version,
            "service_status": "healthy",
            "features": features
        }

    except Exception as e:
        logger.error(f"Error in info endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve service information"
        )
