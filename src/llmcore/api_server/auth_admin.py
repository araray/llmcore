# llmcore/src/llmcore/api_server/auth_admin.py
"""
Administrative authentication dependency for the llmcore API server.

This module provides a separate, more stringent authentication system specifically
for administrative endpoints. It uses a dedicated admin API key that is independent
of the tenant-based authentication system, providing an additional security layer
for high-privilege operations like live configuration reloading.
"""

import asyncio
import logging
import secrets
from typing import Optional

from fastapi import Request, Depends, HTTPException, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

# Define the custom header for the admin API key
admin_api_key_header_scheme = APIKeyHeader(name="X-LLMCore-Admin-Key", auto_error=True)


async def get_admin_user(
    request: Request,
    admin_api_key: str = Depends(admin_api_key_header_scheme)
) -> bool:
    """
    FastAPI dependency that validates admin API keys for administrative operations.

    This dependency implements a secure authentication mechanism specifically for
    administrative endpoints. It validates the provided admin API key against the
    configuration and ensures only authorized administrators can access privileged
    operations like live configuration reloading.

    Security Features:
    - Uses a dedicated admin API key separate from tenant authentication
    - Performs constant-time comparison to prevent timing attacks
    - Reads the admin key from live configuration for immediate key rotation support
    - Provides detailed logging for security auditing

    Args:
        request: FastAPI request object containing the LLMCore instance
        admin_api_key: Admin API key from the X-LLMCore-Admin-Key header

    Returns:
        True if authentication succeeds (for consistency with FastAPI dependency patterns)

    Raises:
        HTTPException:
            - 401 for invalid or missing admin API keys
            - 403 for access denied scenarios
            - 500 for internal configuration errors
    """
    try:
        # Step 1: Access the LLMCore instance to get the current configuration
        if not hasattr(request.app.state, 'llmcore_instance') or request.app.state.llmcore_instance is None:
            logger.error("LLMCore instance not available for admin authentication")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Service configuration unavailable"
            )

        llmcore_instance = request.app.state.llmcore_instance

        # Step 2: Retrieve the admin API key from the live configuration
        try:
            configured_admin_key = llmcore_instance.config.get('llmcore.admin_api_key')
        except Exception as e:
            logger.error(f"Failed to retrieve admin API key from configuration: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Configuration access error"
            )

        # Step 3: Validate that an admin key is configured
        if not configured_admin_key or not configured_admin_key.strip():
            logger.warning(f"Admin API access attempted but no admin key configured from {request.client.host if request.client else 'unknown'}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Administrative access not configured"
            )

        # Step 4: Perform secure, constant-time comparison
        # This prevents timing attacks that could leak information about the key
        is_valid_key = secrets.compare_digest(
            admin_api_key.encode('utf-8'),
            configured_admin_key.encode('utf-8')
        )

        if not is_valid_key:
            logger.warning(f"Invalid admin API key provided from {request.client.host if request.client else 'unknown'}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid admin API key"
            )

        # Step 5: Log successful authentication for security auditing
        logger.info(f"Successful admin authentication from {request.client.host if request.client else 'unknown'}")

        # Return True to indicate successful authentication
        return True

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error during admin authentication: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal authentication error"
        )
