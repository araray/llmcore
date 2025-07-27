# src/llmcore/api_server/routes/tool_management.py
"""
API routes for dynamic management of individual Tools.

This module provides secure CRUD endpoints for managing tools within the context
of the authenticated tenant, enabling runtime configuration of agent capabilities.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_tenant_db_session
from ..models.tools import (AvailableImplementationsResponse, ToolCreateRequest,
                            ToolResponse, ToolUpdateRequest)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/tools", response_model=ToolResponse, status_code=status.HTTP_201_CREATED)
async def create_tool(
    request: ToolCreateRequest,
    db_session: AsyncSession = Depends(get_tenant_db_session)
) -> ToolResponse:
    """
    Create a new tool for the authenticated tenant.

    Args:
        request: Tool creation parameters.
        db_session: Tenant-scoped database session.

    Returns:
        The created tool information.

    Raises:
        HTTPException: If tool name already exists or implementation key is invalid.
    """
    try:
        from ...agents.tools import ToolManager
        if not ToolManager.is_valid_implementation_key(request.implementation_key):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid implementation key: {request.implementation_key}"
            )

        check_query = text("SELECT name FROM tools WHERE name = :name")
        result = await db_session.execute(check_query, {"name": request.name})
        if result.fetchone():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Tool '{request.name}' already exists"
            )

        now = datetime.now(timezone.utc)
        insert_query = text("""
            INSERT INTO tools (name, description, parameters_schema, implementation_key, created_at, updated_at)
            VALUES (:name, :description, :parameters_schema, :implementation_key, :created_at, :updated_at)
        """)
        await db_session.execute(insert_query, {
            "name": request.name, "description": request.description,
            "parameters_schema": request.parameters_schema,
            "implementation_key": request.implementation_key,
            "created_at": now, "updated_at": now
        })
        await db_session.commit()

        created_tool = await _get_tool_by_name(db_session, request.name)
        if not created_tool:
             raise HTTPException(status_code=500, detail="Failed to retrieve tool after creation.")
        return created_tool

    except HTTPException:
        raise
    except Exception as e:
        await db_session.rollback()
        logger.error(f"Error creating tool '{request.name}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create tool")


@router.get("/tools", response_model=List[ToolResponse])
async def list_tools(
    enabled_only: bool = True,
    db_session: AsyncSession = Depends(get_tenant_db_session)
) -> List[ToolResponse]:
    """
    List all tools for the authenticated tenant.

    Args:
        enabled_only: If True, only return enabled tools.
        db_session: Tenant-scoped database session.

    Returns:
        List of tools.
    """
    try:
        query_str = "SELECT * FROM tools"
        params = {}
        if enabled_only:
            query_str += " WHERE is_enabled = :enabled"
            params["enabled"] = True
        query_str += " ORDER BY name"

        result = await db_session.execute(text(query_str), params)
        rows = result.fetchall()
        return [ToolResponse.model_validate(row._mapping) for row in rows]
    except Exception as e:
        logger.error(f"Error listing tools: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list tools")


@router.get("/tools/{tool_name}", response_model=ToolResponse)
async def get_tool(
    tool_name: str,
    db_session: AsyncSession = Depends(get_tenant_db_session)
) -> ToolResponse:
    """
    Get a specific tool by name.

    Args:
        tool_name: Name of the tool to retrieve.
        db_session: Tenant-scoped database session.

    Returns:
        Tool information.
    """
    tool = await _get_tool_by_name(db_session, tool_name)
    if not tool:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tool '{tool_name}' not found")
    return tool


@router.put("/tools/{tool_name}", response_model=ToolResponse)
async def update_tool(
    tool_name: str,
    request: ToolUpdateRequest,
    db_session: AsyncSession = Depends(get_tenant_db_session)
) -> ToolResponse:
    """
    Update an existing tool.

    Args:
        tool_name: Name of the tool to update.
        request: Update parameters.
        db_session: Tenant-scoped database session.

    Returns:
        Updated tool information.
    """
    try:
        if not await _get_tool_by_name(db_session, tool_name):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tool '{tool_name}' not found")

        update_fields = {"updated_at": datetime.now(timezone.utc)}
        if request.description is not None:
            update_fields["description"] = request.description
        if request.parameters_schema is not None:
            update_fields["parameters_schema"] = request.parameters_schema
        if request.is_enabled is not None:
            update_fields["is_enabled"] = request.is_enabled

        set_clause = ", ".join([f"{key} = :{key}" for key in update_fields])
        update_query = text(f"UPDATE tools SET {set_clause} WHERE name = :name")

        await db_session.execute(update_query, {**update_fields, "name": tool_name})
        await db_session.commit()

        updated_tool = await _get_tool_by_name(db_session, tool_name)
        if not updated_tool:
            raise HTTPException(status_code=500, detail="Failed to retrieve tool after update.")
        return updated_tool
    except HTTPException:
        raise
    except Exception as e:
        await db_session.rollback()
        logger.error(f"Error updating tool '{tool_name}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update tool")


@router.delete("/tools/{tool_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_tool(
    tool_name: str,
    db_session: AsyncSession = Depends(get_tenant_db_session)
) -> None:
    """
    Delete a tool by marking it as disabled (soft delete).

    Args:
        tool_name: Name of the tool to delete.
        db_session: Tenant-scoped database session.
    """
    try:
        if not await _get_tool_by_name(db_session, tool_name):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tool '{tool_name}' not found")

        update_query = text("UPDATE tools SET is_enabled = FALSE, updated_at = :updated_at WHERE name = :name")
        await db_session.execute(update_query, {"name": tool_name, "updated_at": datetime.now(timezone.utc)})
        await db_session.commit()
    except HTTPException:
        raise
    except Exception as e:
        await db_session.rollback()
        logger.error(f"Error deleting tool '{tool_name}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete tool")


@router.get("/implementations", response_model=AvailableImplementationsResponse)
async def get_available_implementations() -> AvailableImplementationsResponse:
    """
    Get the list of available, securely registered implementation keys.

    Returns:
        A dictionary mapping available implementation keys to their descriptions.
    """
    try:
        from ...agents.tools import ToolManager
        implementations = ToolManager.get_available_implementations()
        return AvailableImplementationsResponse(
            implementation_keys=list(implementations.keys()),
            descriptions=implementations
        )
    except Exception as e:
        logger.error(f"Error getting available implementations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get available implementations")


async def _get_tool_by_name(db_session: AsyncSession, tool_name: str) -> Optional[ToolResponse]:
    """Helper function to get a tool by name from the database."""
    query = text("SELECT * FROM tools WHERE name = :name")
    result = await db_session.execute(query, {"name": tool_name})
    row = result.fetchone()
    return ToolResponse.model_validate(row._mapping) if row else None
