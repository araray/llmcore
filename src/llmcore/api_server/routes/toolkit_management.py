# src/llmcore/api_server/routes/toolkit_management.py
"""
API routes for dynamic management of Toolkits.

This module provides secure CRUD endpoints for managing named groups of tools
(Toolkits) within the context of the authenticated tenant.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_tenant_db_session
from ..models.tools import (ToolkitCreateRequest, ToolkitResponse,
                            ToolkitUpdateRequest)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/toolkits", response_model=ToolkitResponse, status_code=status.HTTP_201_CREATED)
async def create_toolkit(
    request: ToolkitCreateRequest,
    db_session: AsyncSession = Depends(get_tenant_db_session)
) -> ToolkitResponse:
    """
    Create a new toolkit for the authenticated tenant.

    Args:
        request: Toolkit creation parameters.
        db_session: Tenant-scoped database session.

    Returns:
        The created toolkit information.

    Raises:
        HTTPException: If toolkit name already exists or specified tools don't exist.
    """
    try:
        check_query = text("SELECT name FROM toolkits WHERE name = :name")
        result = await db_session.execute(check_query, {"name": request.name})
        if result.fetchone():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Toolkit '{request.name}' already exists"
            )

        if request.tool_names:
            placeholders = ", ".join([f":tool_{i}" for i in range(len(request.tool_names))])
            tool_check_query = text(f"SELECT name FROM tools WHERE name IN ({placeholders}) AND is_enabled = TRUE")
            tool_params = {f"tool_{i}": name for i, name in enumerate(request.tool_names)}
            result = await db_session.execute(tool_check_query, tool_params)
            existing_tools = {row[0] for row in result.fetchall()}
            missing_tools = set(request.tool_names) - existing_tools
            if missing_tools:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Tools not found or disabled: {list(missing_tools)}"
                )

        now = datetime.now(timezone.utc)
        insert_query = text("""
            INSERT INTO toolkits (name, description, created_at, updated_at)
            VALUES (:name, :description, :created_at, :updated_at)
        """)
        await db_session.execute(insert_query, {
            "name": request.name, "description": request.description,
            "created_at": now, "updated_at": now
        })

        if request.tool_names:
            for tool_name in request.tool_names:
                assoc_query = text("INSERT INTO toolkit_tools (toolkit_name, tool_name) VALUES (:toolkit_name, :tool_name)")
                await db_session.execute(assoc_query, {"toolkit_name": request.name, "tool_name": tool_name})

        await db_session.commit()

        created_toolkit = await _get_toolkit_by_name(db_session, request.name)
        if not created_toolkit:
            raise HTTPException(status_code=500, detail="Failed to retrieve toolkit after creation.")
        return created_toolkit

    except HTTPException:
        raise
    except Exception as e:
        await db_session.rollback()
        logger.error(f"Error creating toolkit '{request.name}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create toolkit")


@router.get("/toolkits", response_model=List[ToolkitResponse])
async def list_toolkits(
    db_session: AsyncSession = Depends(get_tenant_db_session)
) -> List[ToolkitResponse]:
    """
    List all toolkits for the authenticated tenant.

    Args:
        db_session: Tenant-scoped database session.

    Returns:
        List of toolkits with their associated tools.
    """
    try:
        query = text("""
            SELECT tk.name, tk.description, tk.created_at, tk.updated_at,
                   COALESCE(array_agg(tt.tool_name ORDER BY tt.tool_name)
                           FILTER (WHERE tt.tool_name IS NOT NULL), ARRAY[]::text[]) as tool_names
            FROM toolkits tk
            LEFT JOIN toolkit_tools tt ON tk.name = tt.toolkit_name
            GROUP BY tk.name, tk.description, tk.created_at, tk.updated_at
            ORDER BY tk.name
        """)
        result = await db_session.execute(query)
        rows = result.fetchall()
        return [ToolkitResponse.model_validate(row._mapping) for row in rows]
    except Exception as e:
        logger.error(f"Error listing toolkits: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list toolkits")


@router.get("/toolkits/{toolkit_name}", response_model=ToolkitResponse)
async def get_toolkit(
    toolkit_name: str,
    db_session: AsyncSession = Depends(get_tenant_db_session)
) -> ToolkitResponse:
    """
    Get a specific toolkit by name.

    Args:
        toolkit_name: Name of the toolkit to retrieve.
        db_session: Tenant-scoped database session.

    Returns:
        Toolkit information with associated tools.
    """
    toolkit = await _get_toolkit_by_name(db_session, toolkit_name)
    if not toolkit:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Toolkit '{toolkit_name}' not found")
    return toolkit


@router.put("/toolkits/{toolkit_name}", response_model=ToolkitResponse)
async def update_toolkit(
    toolkit_name: str,
    request: ToolkitUpdateRequest,
    db_session: AsyncSession = Depends(get_tenant_db_session)
) -> ToolkitResponse:
    """
    Update an existing toolkit.

    Args:
        toolkit_name: Name of the toolkit to update.
        request: Update parameters.
        db_session: Tenant-scoped database session.

    Returns:
        Updated toolkit information.
    """
    try:
        if not await _get_toolkit_by_name(db_session, toolkit_name):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Toolkit '{toolkit_name}' not found")

        if request.description is not None:
            update_query = text("UPDATE toolkits SET description = :description, updated_at = :updated_at WHERE name = :name")
            await db_session.execute(update_query, {
                "name": toolkit_name, "description": request.description, "updated_at": datetime.now(timezone.utc)
            })

        if request.tool_names is not None:
            # Full replacement of tool associations
            delete_query = text("DELETE FROM toolkit_tools WHERE toolkit_name = :toolkit_name")
            await db_session.execute(delete_query, {"toolkit_name": toolkit_name})

            if request.tool_names:
                for tool_name in request.tool_names:
                    assoc_query = text("INSERT INTO toolkit_tools (toolkit_name, tool_name) VALUES (:toolkit_name, :tool_name)")
                    await db_session.execute(assoc_query, {"toolkit_name": toolkit_name, "tool_name": tool_name})

        await db_session.commit()

        updated_toolkit = await _get_toolkit_by_name(db_session, toolkit_name)
        if not updated_toolkit:
            raise HTTPException(status_code=500, detail="Failed to retrieve toolkit after update.")
        return updated_toolkit

    except HTTPException:
        raise
    except Exception as e:
        await db_session.rollback()
        logger.error(f"Error updating toolkit '{toolkit_name}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update toolkit")


@router.delete("/toolkits/{toolkit_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_toolkit(
    toolkit_name: str,
    db_session: AsyncSession = Depends(get_tenant_db_session)
) -> None:
    """
    Delete a toolkit and all its tool associations.

    Args:
        toolkit_name: Name of the toolkit to delete.
        db_session: Tenant-scoped database session.
    """
    try:
        if not await _get_toolkit_by_name(db_session, toolkit_name):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Toolkit '{toolkit_name}' not found")

        delete_query = text("DELETE FROM toolkits WHERE name = :name")
        await db_session.execute(delete_query, {"name": toolkit_name})
        await db_session.commit()
    except HTTPException:
        raise
    except Exception as e:
        await db_session.rollback()
        logger.error(f"Error deleting toolkit '{toolkit_name}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete toolkit")


async def _get_toolkit_by_name(db_session: AsyncSession, toolkit_name: str) -> Optional[ToolkitResponse]:
    """Helper function to get a toolkit by name with its associated tools."""
    query = text("""
        SELECT tk.name, tk.description, tk.created_at, tk.updated_at,
               COALESCE(array_agg(tt.tool_name ORDER BY tt.tool_name)
                       FILTER (WHERE tt.tool_name IS NOT NULL), ARRAY[]::text[]) as tool_names
        FROM toolkits tk
        LEFT JOIN toolkit_tools tt ON tk.name = tt.toolkit_name
        WHERE tk.name = :name
        GROUP BY tk.name, tk.description, tk.created_at, tk.updated_at
    """)
    result = await db_session.execute(query, {"name": toolkit_name})
    row = result.fetchone()
    return ToolkitResponse.model_validate(row._mapping) if row else None
