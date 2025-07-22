# llmcore/src/llmcore/api_server/db_utils.py
"""
Database interaction utilities for tenant and API key management.

This module encapsulates all database operations related to tenants and API keys,
abstracting the data layer from the authentication logic to maintain separation of concerns.
"""

import logging
from typing import Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from .schemas.security import Tenant, APIKey

logger = logging.getLogger(__name__)


async def get_api_key_by_prefix(db_session: AsyncSession, prefix: str) -> Optional[APIKey]:
    """
    Retrieve an API key record by its prefix.

    Args:
        db_session: Active database session
        prefix: The key prefix to search for

    Returns:
        APIKey object if found, None otherwise

    Raises:
        Exception: Database connection or query errors
    """
    try:
        # Query the api_keys table for a record matching the provided key prefix
        query = text("""
            SELECT id, hashed_key, key_prefix, tenant_id, created_at, expires_at, last_used_at
            FROM api_keys
            WHERE key_prefix = :prefix
            AND (expires_at IS NULL OR expires_at > NOW())
        """)

        result = await db_session.execute(query, {"prefix": prefix})
        row = result.fetchone()

        if row:
            return APIKey(
                id=row.id,
                hashed_key=row.hashed_key,
                key_prefix=row.key_prefix,
                tenant_id=row.tenant_id,
                created_at=row.created_at,
                expires_at=row.expires_at,
                last_used_at=row.last_used_at
            )

        logger.debug(f"No API key found for prefix: {prefix}")
        return None

    except Exception as e:
        logger.error(f"Database error while retrieving API key by prefix '{prefix}': {e}", exc_info=True)
        raise


async def get_tenant_by_id(db_session: AsyncSession, tenant_id: UUID) -> Optional[Tenant]:
    """
    Retrieve a tenant record by its ID.

    Args:
        db_session: Active database session
        tenant_id: The tenant UUID to search for

    Returns:
        Tenant object if found, None otherwise

    Raises:
        Exception: Database connection or query errors
    """
    try:
        # Query the tenants table for a record matching the tenant ID
        query = text("""
            SELECT id, name, db_schema_name, created_at, status
            FROM tenants
            WHERE id = :tenant_id
        """)

        result = await db_session.execute(query, {"tenant_id": tenant_id})
        row = result.fetchone()

        if row:
            return Tenant(
                id=row.id,
                name=row.name,
                db_schema_name=row.db_schema_name,
                created_at=row.created_at,
                status=row.status
            )

        logger.debug(f"No tenant found for ID: {tenant_id}")
        return None

    except Exception as e:
        logger.error(f"Database error while retrieving tenant by ID '{tenant_id}': {e}", exc_info=True)
        raise


async def update_api_key_last_used(db_session: AsyncSession, api_key_id: UUID) -> None:
    """
    Update the last_used_at timestamp for an API key.

    Args:
        db_session: Active database session
        api_key_id: The API key UUID to update

    Raises:
        Exception: Database connection or query errors
    """
    try:
        query = text("""
            UPDATE api_keys
            SET last_used_at = NOW()
            WHERE id = :api_key_id
        """)

        await db_session.execute(query, {"api_key_id": api_key_id})
        await db_session.commit()

        logger.debug(f"Updated last_used_at for API key: {api_key_id}")

    except Exception as e:
        logger.error(f"Database error while updating last_used_at for API key '{api_key_id}': {e}", exc_info=True)
        raise
