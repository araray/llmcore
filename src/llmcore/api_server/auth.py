# llmcore/src/llmcore/api_server/auth.py
"""
Authentication dependency for the llmcore API server.

This module contains the core authentication dependency that validates API keys
and identifies the tenant for each request, establishing the foundational
security layer for the platform.
"""

import asyncio
import logging
from typing import Optional

import bcrypt
from fastapi import Request, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from .db_utils import get_api_key_by_prefix, get_tenant_by_id, update_api_key_last_used
from .schemas.security import Tenant

logger = logging.getLogger(__name__)

# Define the custom header for the API key
api_key_header_scheme = APIKeyHeader(name="X-LLMCore-API-Key", auto_error=True)

# Global database session factory - will be initialized during app startup
_db_session_factory: Optional[sessionmaker] = None


def initialize_auth_db_session(database_url: str) -> None:
    """
    Initialize the database session factory for authentication operations.

    This should be called during application startup to configure the
    database connection used by the authentication system.

    Args:
        database_url: PostgreSQL connection URL
    """
    global _db_session_factory

    engine = create_async_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False  # Set to True for SQL debugging
    )

    _db_session_factory = sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    logger.info("Authentication database session factory initialized")


async def get_auth_db_session() -> AsyncSession:
    """
    Get a database session for authentication operations.

    Returns:
        Configured AsyncSession for database operations

    Raises:
        RuntimeError: If the session factory is not initialized
    """
    if _db_session_factory is None:
        raise RuntimeError("Authentication database session factory not initialized. Call initialize_auth_db_session() first.")

    return _db_session_factory()


def _extract_key_prefix(api_key: str) -> Optional[str]:
    """
    Extract the key prefix from a full API key.

    Expected format: llmk_<prefix>_<secret>

    Args:
        api_key: The full API key string

    Returns:
        The prefix portion if valid format, None otherwise
    """
    try:
        # Expected format: llmk_<prefix>_<secret>
        if not api_key.startswith('llmk_'):
            return None

        parts = api_key.split('_')
        if len(parts) < 3:  # Need at least llmk, prefix, and secret parts
            return None

        # Return the prefix part (second element)
        return f"llmk_{parts[1]}"

    except Exception as e:
        logger.warning(f"Error extracting key prefix from API key: {e}")
        return None


async def _verify_key_hash(api_key: str, hashed_key: str) -> bool:
    """
    Verify an API key against its stored hash using bcrypt.

    Args:
        api_key: The plaintext API key to verify
        hashed_key: The stored bcrypt hash

    Returns:
        True if the key matches the hash, False otherwise
    """
    try:
        # Run bcrypt verification in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()

        def verify():
            return bcrypt.checkpw(api_key.encode('utf-8'), hashed_key.encode('utf-8'))

        return await loop.run_in_executor(None, verify)

    except Exception as e:
        logger.error(f"Error verifying API key hash: {e}", exc_info=True)
        return False


async def get_current_tenant(
    request: Request,
    api_key: str = Depends(api_key_header_scheme)
) -> Tenant:
    """
    FastAPI dependency that validates API keys and identifies the current tenant.

    This dependency implements the core authentication logic for the llmcore platform.
    It validates the provided API key, verifies it against the database, and returns
    the associated tenant information. The tenant object is also attached to the
    request state for use throughout the request lifecycle.

    Args:
        request: FastAPI request object
        api_key: API key from the X-LLMCore-API-Key header

    Returns:
        Tenant object for the authenticated request

    Raises:
        HTTPException:
            - 401 for invalid API keys or authentication failures
            - 403 for inactive tenants or access denied
            - 500 for internal server errors
    """
    try:
        # Step 1: Extract the key prefix from the API key
        key_prefix = _extract_key_prefix(api_key)
        if not key_prefix:
            logger.warning(f"Invalid API key format received from {request.client.host if request.client else 'unknown'}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key format"
            )

        # Step 2: Get database session and look up the API key record
        async with await get_auth_db_session() as db_session:
            api_key_record = await get_api_key_by_prefix(db_session, key_prefix)

            if not api_key_record:
                logger.warning(f"API key not found for prefix: {key_prefix}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key"
                )

            # Step 3: Verify the full key against the stored hash
            is_valid = await _verify_key_hash(api_key, api_key_record.hashed_key)
            if not is_valid:
                logger.warning(f"API key verification failed for prefix: {key_prefix}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key"
                )

            # Step 4: Retrieve the associated tenant
            tenant = await get_tenant_by_id(db_session, api_key_record.tenant_id)
            if not tenant:
                logger.error(f"Tenant not found for API key: {key_prefix}, tenant_id: {api_key_record.tenant_id}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Associated tenant not found"
                )

            # Step 5: Validate tenant status
            if tenant.status != 'active':
                logger.warning(f"Access attempt with inactive tenant: {tenant.name} (status: {tenant.status})")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Tenant account is inactive"
                )

            # Step 6: Update last used timestamp (fire and forget)
            try:
                await update_api_key_last_used(db_session, api_key_record.id)
            except Exception as e:
                # Don't fail the request if we can't update the timestamp
                logger.warning(f"Failed to update last_used_at for API key {key_prefix}: {e}")

            # Step 7: Attach tenant to request state and return
            request.state.tenant = tenant

            logger.debug(f"Successfully authenticated tenant: {tenant.name} ({tenant.id})")
            return tenant

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error during authentication: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal authentication error"
        )
