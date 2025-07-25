# llmcore/src/llmcore/api_server/routes/admin.py
"""
Administrative API routes for the llmcore platform.

This module defines secure administrative endpoints that provide high-privilege
operations for platform management. All endpoints in this router require
admin-level authentication using a dedicated admin API key.

Current endpoints:
- POST /reload-config: Performs live configuration reloading without service restart
"""

import logging
from typing import Dict, Any

from fastapi import APIRouter, Request, Depends, HTTPException, status
from pydantic import BaseModel

from ..auth_admin import get_admin_user
from ...exceptions import ConfigError

logger = logging.getLogger(__name__)

# Create the admin router
admin_router = APIRouter()


class AdminResponse(BaseModel):
    """Standard response model for administrative operations."""
    status: str
    message: str
    details: Dict[str, Any] = {}


class ConfigReloadResponse(AdminResponse):
    """Response model for configuration reload operations."""
    preserved_sessions_count: int = 0
    preserved_context_info_count: int = 0


@admin_router.post(
    "/reload-config",
    response_model=ConfigReloadResponse,
    summary="Reload configuration without service restart",
    description="""
    Performs a live reload of the entire llmcore configuration from all sources
    (files, environment variables) and re-initializes all core components without
    requiring a service restart.

    This endpoint:
    1. Preserves transient state (in-memory chat sessions)
    2. Gracefully shuts down existing connections
    3. Reloads configuration from all sources
    4. Re-initializes all managers and providers
    5. Restores preserved transient state

    **Security Note**: This endpoint requires admin-level authentication using
    the X-LLMCore-Admin-Key header with a valid admin API key.

    **Use Cases**:
    - Update API keys for LLM providers
    - Change default models or providers
    - Modify logging levels
    - Update database connection settings
    - Add or remove provider configurations

    **Caution**: While this operation preserves transient sessions, it may
    briefly impact new requests during the reload process.
    """,
    responses={
        200: {
            "description": "Configuration successfully reloaded",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "message": "Configuration reloaded successfully",
                        "preserved_sessions_count": 3,
                        "preserved_context_info_count": 3
                    }
                }
            }
        },
        401: {"description": "Invalid or missing admin API key"},
        403: {"description": "Administrative access not configured"},
        500: {"description": "Configuration reload failed"}
    }
)
async def reload_config(
    request: Request,
    admin_user: bool = Depends(get_admin_user)
) -> ConfigReloadResponse:
    """
    Reload the llmcore configuration without restarting the service.

    This endpoint provides zero-downtime configuration updates by performing
    a controlled reload of all configuration sources and component re-initialization
    while preserving critical transient state.

    Args:
        request: FastAPI request object containing the LLMCore instance
        admin_user: Admin authentication dependency (always True if reached)

    Returns:
        ConfigReloadResponse with operation status and preserved state counts

    Raises:
        HTTPException:
            - 500 if LLMCore instance is unavailable
            - 500 if configuration reload fails
    """
    try:
        # Step 1: Validate that LLMCore instance is available
        if not hasattr(request.app.state, 'llmcore_instance') or request.app.state.llmcore_instance is None:
            logger.error("Configuration reload requested but LLMCore instance not available")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="LLMCore service not available"
            )

        llmcore_instance = request.app.state.llmcore_instance

        # Step 2: Capture initial state for reporting
        initial_sessions_count = len(llmcore_instance._transient_sessions_cache)
        initial_context_info_count = len(llmcore_instance._transient_last_interaction_info_cache)

        logger.info(f"Starting configuration reload with {initial_sessions_count} transient sessions and {initial_context_info_count} context info entries")

        # Step 3: Perform the actual configuration reload
        # The reload_config method handles all the complex state preservation logic
        await llmcore_instance.reload_config()

        # Step 4: Verify state preservation
        final_sessions_count = len(llmcore_instance._transient_sessions_cache)
        final_context_info_count = len(llmcore_instance._transient_last_interaction_info_cache)

        logger.info(f"Configuration reload completed successfully. Preserved {final_sessions_count} sessions and {final_context_info_count} context info entries")

        # Step 5: Return success response with detailed information
        return ConfigReloadResponse(
            status="success",
            message="Configuration reloaded successfully",
            preserved_sessions_count=final_sessions_count,
            preserved_context_info_count=final_context_info_count,
            details={
                "operation": "reload_config",
                "state_preservation": {
                    "sessions_before": initial_sessions_count,
                    "sessions_after": final_sessions_count,
                    "context_info_before": initial_context_info_count,
                    "context_info_after": final_context_info_count
                }
            }
        )

    except ConfigError as e:
        logger.error(f"Configuration reload failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration reload failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during configuration reload: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Configuration reload failed due to internal error"
        )


# Health check endpoint specifically for admin operations
@admin_router.get(
    "/health",
    response_model=AdminResponse,
    summary="Admin health check",
    description="Health check endpoint for administrative operations and admin authentication validation."
)
async def admin_health_check(
    request: Request,
    admin_user: bool = Depends(get_admin_user)
) -> AdminResponse:
    """
    Administrative health check endpoint.

    This endpoint serves as both a health check for administrative functionality
    and a way to validate admin API key authentication.

    Args:
        request: FastAPI request object
        admin_user: Admin authentication dependency

    Returns:
        AdminResponse indicating admin system health
    """
    try:
        # Check if LLMCore instance is available
        llmcore_available = (
            hasattr(request.app.state, 'llmcore_instance')
            and request.app.state.llmcore_instance is not None
        )

        if llmcore_available:
            llmcore_instance = request.app.state.llmcore_instance
            available_providers = llmcore_instance.get_available_providers()
            transient_sessions = len(llmcore_instance._transient_sessions_cache)
        else:
            available_providers = []
            transient_sessions = 0

        return AdminResponse(
            status="healthy",
            message="Administrative system operational",
            details={
                "admin_auth": "configured",
                "llmcore_available": llmcore_available,
                "available_providers": available_providers,
                "transient_sessions_count": transient_sessions,
                "config_reload_available": llmcore_available
            }
        )

    except Exception as e:
        logger.error(f"Error in admin health check: {e}", exc_info=True)
        return AdminResponse(
            status="degraded",
            message="Administrative system partially operational",
            details={
                "admin_auth": "configured",
                "error": str(e)
            }
        )
