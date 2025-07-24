# llmcore/src/llmcore/api_server/db.py
"""
Tenant-scoped database dependency for the llmcore API server.

This module provides the core mechanism for schema switching that ensures
all data operations for a given tenant are confined within their designated schema.
"""

import logging
import os
from typing import Optional

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from .auth import get_current_tenant
from .schemas.security import Tenant

logger = logging.getLogger(__name__)

# Global database session factory - will be initialized during app startup
_tenant_db_session_factory: Optional[sessionmaker] = None


def initialize_tenant_db_session(database_url: str) -> None:
    """
    Initialize the tenant database session factory.

    This should be called during application startup to configure the
    database connection used for tenant-scoped operations.

    Args:
        database_url: PostgreSQL connection URL
    """
    global _tenant_db_session_factory

    engine = create_async_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False  # Set to True for SQL debugging
    )

    _tenant_db_session_factory = sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    logger.info("Tenant database session factory initialized")


async def get_tenant_db_session(
    tenant: Tenant = Depends(get_current_tenant)
) -> AsyncSession:
    """
    FastAPI dependency that provides a tenant-scoped database session.

    This dependency is the core mechanism for schema switching in the multi-tenant
    architecture. It configures the database session to operate within the
    specific tenant's schema, ensuring complete data isolation.

    The dependency:
    1. Gets the authenticated tenant from get_current_tenant
    2. Acquires a database connection from the global pool
    3. Executes 'SET search_path TO tenant_schema, public' to switch schema context
    4. Yields the configured session to the route handler
    5. Automatically resets the connection state when returned to the pool

    Args:
        tenant: The authenticated tenant from get_current_tenant dependency

    Returns:
        Configured AsyncSession operating in the tenant's schema context

    Raises:
        HTTPException:
            - 500 if the session factory is not initialized
            - 500 if schema switching fails
            - 503 if database connection fails
    """
    if _tenant_db_session_factory is None:
        logger.error("Tenant database session factory not initialized")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database session factory not initialized"
        )

    session = None
    try:
        # Create a new session from the factory
        session = _tenant_db_session_factory()

        # Critical step: Set the search_path to the tenant's schema
        # This ensures all subsequent queries operate within the tenant's isolated schema
        await session.execute(
            f"SET search_path TO {tenant.db_schema_name}, public"
        )

        logger.debug(f"Database session configured for tenant schema: {tenant.db_schema_name}")

        # Yield the configured session to the route handler
        yield session

    except Exception as e:
        logger.error(f"Error configuring tenant database session for schema '{tenant.db_schema_name}': {e}", exc_info=True)

        # Close the session if it was created
        if session:
            try:
                await session.close()
            except Exception as close_error:
                logger.error(f"Error closing database session: {close_error}")

        # Determine appropriate error response
        error_message = str(e).lower()
        if "connection" in error_message or "timeout" in error_message:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database service temporarily unavailable"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database session configuration failed"
            )

    finally:
        # Ensure session is always closed, which returns the connection to the pool
        # The connection pool will reset the session state automatically
        if session:
            try:
                await session.close()
                logger.debug(f"Database session closed for tenant: {tenant.name}")
            except Exception as e:
                logger.error(f"Error closing tenant database session: {e}")


async def execute_tenant_query(
    session: AsyncSession,
    query: str,
    params: Optional[dict] = None
) -> any:
    """
    Helper function to execute a query within a tenant-scoped session.

    This function provides a convenient way to execute raw SQL queries
    within the context of a tenant's schema.

    Args:
        session: Tenant-scoped database session
        query: SQL query to execute
        params: Optional query parameters

    Returns:
        Query result

    Raises:
        Exception: Database execution errors
    """
    try:
        if params:
            result = await session.execute(query, params)
        else:
            result = await session.execute(query)

        await session.commit()
        return result

    except Exception as e:
        await session.rollback()
        logger.error(f"Error executing tenant query: {e}", exc_info=True)
        raise


async def get_tenant_schema_info(
    session: AsyncSession
) -> dict:
    """
    Helper function to get information about the current tenant schema.

    This function can be used for debugging or administrative purposes
    to verify which schema is currently active.

    Args:
        session: Tenant-scoped database session

    Returns:
        Dictionary containing schema information

    Raises:
        Exception: Database query errors
    """
    try:
        # Get current search path
        result = await session.execute("SHOW search_path")
        search_path = result.scalar()

        # Get current schema
        result = await session.execute("SELECT current_schema()")
        current_schema = result.scalar()

        # Get list of tables in current schema
        result = await session.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = current_schema()
            ORDER BY table_name
        """)
        tables = [row[0] for row in result.fetchall()]

        return {
            "search_path": search_path,
            "current_schema": current_schema,
            "tables": tables
        }

    except Exception as e:
        logger.error(f"Error getting tenant schema info: {e}", exc_info=True)
        raise
