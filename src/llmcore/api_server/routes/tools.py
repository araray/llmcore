# llmcore/src/llmcore/api_server/routes/tools.py
"""
API routes for dynamic tool and toolkit management.

This module provides secure CRUD endpoints for managing tools and toolkits
within the context of the authenticated tenant, enabling runtime configuration
of agent capabilities.
"""

import logging
from typing import List, Dict, Any
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select, insert, update, delete

from ..db import get_tenant_db_session
from ..models.tools import (
    ToolCreateRequest, ToolUpdateRequest, ToolResponse,
    ToolkitCreateRequest, ToolkitUpdateRequest, ToolkitResponse,
    AvailableImplementationsResponse
)

logger = logging.getLogger(__name__)

router = APIRouter()


# Tool Management Endpoints

@router.post("/tools", response_model=ToolResponse, status_code=status.HTTP_201_CREATED)
async def create_tool(
    request: ToolCreateRequest,
    db_session: AsyncSession = Depends(get_tenant_db_session)
) -> ToolResponse:
    """
    Create a new tool for the authenticated tenant.

    Args:
        request: Tool creation parameters
        db_session: Tenant-scoped database session

    Returns:
        The created tool information

    Raises:
        HTTPException: If tool name already exists or implementation key is invalid
    """
    try:
        # Validate that the implementation key exists in the secure registry
        from ...agents.tools import ToolManager
        if not ToolManager.is_valid_implementation_key(request.implementation_key):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid implementation key: {request.implementation_key}"
            )

        # Check if tool already exists
        check_query = text("SELECT name FROM tools WHERE name = :name")
        result = await db_session.execute(check_query, {"name": request.name})
        if result.fetchone():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Tool '{request.name}' already exists"
            )

        # Insert new tool
        now = datetime.now(timezone.utc)
        insert_query = text("""
            INSERT INTO tools (name, description, parameters_schema, implementation_key, created_at, updated_at)
            VALUES (:name, :description, :parameters_schema, :implementation_key, :created_at, :updated_at)
        """)

        await db_session.execute(insert_query, {
            "name": request.name,
            "description": request.description,
            "parameters_schema": request.parameters_schema,
            "implementation_key": request.implementation_key,
            "created_at": now,
            "updated_at": now
        })
        await db_session.commit()

        # Return the created tool
        return await _get_tool_by_name(db_session, request.name)

    except HTTPException:
        raise
    except Exception as e:
        await db_session.rollback()
        logger.error(f"Error creating tool '{request.name}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create tool"
        )


@router.get("/tools", response_model=List[ToolResponse])
async def list_tools(
    enabled_only: bool = True,
    db_session: AsyncSession = Depends(get_tenant_db_session)
) -> List[ToolResponse]:
    """
    List all tools for the authenticated tenant.

    Args:
        enabled_only: If True, only return enabled tools
        db_session: Tenant-scoped database session

    Returns:
        List of tools
    """
    try:
        query = "SELECT * FROM tools"
        params = {}

        if enabled_only:
            query += " WHERE is_enabled = :enabled"
            params["enabled"] = True

        query += " ORDER BY name"

        result = await db_session.execute(text(query), params)
        rows = result.fetchall()

        return [ToolResponse(**dict(row._asdict())) for row in rows]

    except Exception as e:
        logger.error(f"Error listing tools: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list tools"
        )


@router.get("/tools/{tool_name}", response_model=ToolResponse)
async def get_tool(
    tool_name: str,
    db_session: AsyncSession = Depends(get_tenant_db_session)
) -> ToolResponse:
    """
    Get a specific tool by name.

    Args:
        tool_name: Name of the tool to retrieve
        db_session: Tenant-scoped database session

    Returns:
        Tool information

    Raises:
        HTTPException: If tool not found
    """
    tool = await _get_tool_by_name(db_session, tool_name)
    if not tool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_name}' not found"
        )
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
        tool_name: Name of the tool to update
        request: Update parameters
        db_session: Tenant-scoped database session

    Returns:
        Updated tool information

    Raises:
        HTTPException: If tool not found
    """
    try:
        # Check if tool exists
        existing_tool = await _get_tool_by_name(db_session, tool_name)
        if not existing_tool:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tool '{tool_name}' not found"
            )

        # Build update fields
        update_fields = {"updated_at": datetime.now(timezone.utc)}
        if request.description is not None:
            update_fields["description"] = request.description
        if request.parameters_schema is not None:
            update_fields["parameters_schema"] = request.parameters_schema
        if request.is_enabled is not None:
            update_fields["is_enabled"] = request.is_enabled

        # Update tool
        set_clause = ", ".join([f"{key} = :{key}" for key in update_fields.keys()])
        update_query = text(f"UPDATE tools SET {set_clause} WHERE name = :name")
        update_fields["name"] = tool_name

        await db_session.execute(update_query, update_fields)
        await db_session.commit()

        # Return updated tool
        return await _get_tool_by_name(db_session, tool_name)

    except HTTPException:
        raise
    except Exception as e:
        await db_session.rollback()
        logger.error(f"Error updating tool '{tool_name}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update tool"
        )


@router.delete("/tools/{tool_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_tool(
    tool_name: str,
    db_session: AsyncSession = Depends(get_tenant_db_session)
) -> None:
    """
    Delete a tool (sets is_enabled to False).

    Args:
        tool_name: Name of the tool to delete
        db_session: Tenant-scoped database session

    Raises:
        HTTPException: If tool not found
    """
    try:
        # Check if tool exists
        existing_tool = await _get_tool_by_name(db_session, tool_name)
        if not existing_tool:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tool '{tool_name}' not found"
            )

        # Soft delete by disabling
        update_query = text("""
            UPDATE tools
            SET is_enabled = FALSE, updated_at = :updated_at
            WHERE name = :name
        """)
        await db_session.execute(update_query, {
            "name": tool_name,
            "updated_at": datetime.now(timezone.utc)
        })
        await db_session.commit()

    except HTTPException:
        raise
    except Exception as e:
        await db_session.rollback()
        logger.error(f"Error deleting tool '{tool_name}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete tool"
        )


# Toolkit Management Endpoints

@router.post("/toolkits", response_model=ToolkitResponse, status_code=status.HTTP_201_CREATED)
async def create_toolkit(
    request: ToolkitCreateRequest,
    db_session: AsyncSession = Depends(get_tenant_db_session)
) -> ToolkitResponse:
    """
    Create a new toolkit for the authenticated tenant.

    Args:
        request: Toolkit creation parameters
        db_session: Tenant-scoped database session

    Returns:
        The created toolkit information

    Raises:
        HTTPException: If toolkit name already exists or tools don't exist
    """
    try:
        # Check if toolkit already exists
        check_query = text("SELECT name FROM toolkits WHERE name = :name")
        result = await db_session.execute(check_query, {"name": request.name})
        if result.fetchone():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Toolkit '{request.name}' already exists"
            )

        # Validate that all specified tools exist
        if request.tool_names:
            placeholders = ", ".join([f":tool_{i}" for i in range(len(request.tool_names))])
            tool_check_query = text(f"""
                SELECT name FROM tools
                WHERE name IN ({placeholders}) AND is_enabled = TRUE
            """)
            tool_params = {f"tool_{i}": name for i, name in enumerate(request.tool_names)}
            result = await db_session.execute(tool_check_query, tool_params)
            existing_tools = {row[0] for row in result.fetchall()}

            missing_tools = set(request.tool_names) - existing_tools
            if missing_tools:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Tools not found or disabled: {list(missing_tools)}"
                )

        # Create toolkit
        now = datetime.now(timezone.utc)
        insert_query = text("""
            INSERT INTO toolkits (name, description, created_at, updated_at)
            VALUES (:name, :description, :created_at, :updated_at)
        """)
        await db_session.execute(insert_query, {
            "name": request.name,
            "description": request.description,
            "created_at": now,
            "updated_at": now
        })

        # Add tool associations
        if request.tool_names:
            for tool_name in request.tool_names:
                assoc_query = text("""
                    INSERT INTO toolkit_tools (toolkit_name, tool_name)
                    VALUES (:toolkit_name, :tool_name)
                """)
                await db_session.execute(assoc_query, {
                    "toolkit_name": request.name,
                    "tool_name": tool_name
                })

        await db_session.commit()

        # Return the created toolkit
        return await _get_toolkit_by_name(db_session, request.name)

    except HTTPException:
        raise
    except Exception as e:
        await db_session.rollback()
        logger.error(f"Error creating toolkit '{request.name}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create toolkit"
        )


@router.get("/toolkits", response_model=List[ToolkitResponse])
async def list_toolkits(
    db_session: AsyncSession = Depends(get_tenant_db_session)
) -> List[ToolkitResponse]:
    """
    List all toolkits for the authenticated tenant.

    Args:
        db_session: Tenant-scoped database session

    Returns:
        List of toolkits with their associated tools
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

        return [
            ToolkitResponse(
                name=row.name,
                description=row.description,
                tool_names=row.tool_names,
                created_at=row.created_at,
                updated_at=row.updated_at
            )
            for row in rows
        ]

    except Exception as e:
        logger.error(f"Error listing toolkits: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list toolkits"
        )


@router.get("/toolkits/{toolkit_name}", response_model=ToolkitResponse)
async def get_toolkit(
    toolkit_name: str,
    db_session: AsyncSession = Depends(get_tenant_db_session)
) -> ToolkitResponse:
    """
    Get a specific toolkit by name.

    Args:
        toolkit_name: Name of the toolkit to retrieve
        db_session: Tenant-scoped database session

    Returns:
        Toolkit information with associated tools

    Raises:
        HTTPException: If toolkit not found
    """
    toolkit = await _get_toolkit_by_name(db_session, toolkit_name)
    if not toolkit:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Toolkit '{toolkit_name}' not found"
        )
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
        toolkit_name: Name of the toolkit to update
        request: Update parameters
        db_session: Tenant-scoped database session

    Returns:
        Updated toolkit information

    Raises:
        HTTPException: If toolkit not found or tools don't exist
    """
    try:
        # Check if toolkit exists
        existing_toolkit = await _get_toolkit_by_name(db_session, toolkit_name)
        if not existing_toolkit:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Toolkit '{toolkit_name}' not found"
            )

        # Update toolkit metadata if provided
        if request.description is not None:
            update_query = text("""
                UPDATE toolkits
                SET description = :description, updated_at = :updated_at
                WHERE name = :name
            """)
            await db_session.execute(update_query, {
                "name": toolkit_name,
                "description": request.description,
                "updated_at": datetime.now(timezone.utc)
            })

        # Update tool associations if provided
        if request.tool_names is not None:
            # Validate all tools exist
            if request.tool_names:
                placeholders = ", ".join([f":tool_{i}" for i in range(len(request.tool_names))])
                tool_check_query = text(f"""
                    SELECT name FROM tools
                    WHERE name IN ({placeholders}) AND is_enabled = TRUE
                """)
                tool_params = {f"tool_{i}": name for i, name in enumerate(request.tool_names)}
                result = await db_session.execute(tool_check_query, tool_params)
                existing_tools = {row[0] for row in result.fetchall()}

                missing_tools = set(request.tool_names) - existing_tools
                if missing_tools:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Tools not found or disabled: {list(missing_tools)}"
                    )

            # Remove all existing associations
            delete_query = text("DELETE FROM toolkit_tools WHERE toolkit_name = :toolkit_name")
            await db_session.execute(delete_query, {"toolkit_name": toolkit_name})

            # Add new associations
            for tool_name in request.tool_names:
                assoc_query = text("""
                    INSERT INTO toolkit_tools (toolkit_name, tool_name)
                    VALUES (:toolkit_name, :tool_name)
                """)
                await db_session.execute(assoc_query, {
                    "toolkit_name": toolkit_name,
                    "tool_name": tool_name
                })

        await db_session.commit()

        # Return updated toolkit
        return await _get_toolkit_by_name(db_session, toolkit_name)

    except HTTPException:
        raise
    except Exception as e:
        await db_session.rollback()
        logger.error(f"Error updating toolkit '{toolkit_name}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update toolkit"
        )


@router.delete("/toolkits/{toolkit_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_toolkit(
    toolkit_name: str,
    db_session: AsyncSession = Depends(get_tenant_db_session)
) -> None:
    """
    Delete a toolkit and all its tool associations.

    Args:
        toolkit_name: Name of the toolkit to delete
        db_session: Tenant-scoped database session

    Raises:
        HTTPException: If toolkit not found
    """
    try:
        # Check if toolkit exists
        existing_toolkit = await _get_toolkit_by_name(db_session, toolkit_name)
        if not existing_toolkit:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Toolkit '{toolkit_name}' not found"
            )

        # Delete toolkit (cascade will handle toolkit_tools)
        delete_query = text("DELETE FROM toolkits WHERE name = :name")
        await db_session.execute(delete_query, {"name": toolkit_name})
        await db_session.commit()

    except HTTPException:
        raise
    except Exception as e:
        await db_session.rollback()
        logger.error(f"Error deleting toolkit '{toolkit_name}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete toolkit"
        )


# Utility Endpoints

@router.get("/implementations", response_model=AvailableImplementationsResponse)
async def get_available_implementations() -> AvailableImplementationsResponse:
    """
    Get the list of available implementation keys.

    Returns:
        Available implementation keys and their descriptions
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get available implementations"
        )


# Helper Functions

async def _get_tool_by_name(db_session: AsyncSession, tool_name: str) -> ToolResponse:
    """Get a tool by name from the database."""
    query = text("SELECT * FROM tools WHERE name = :name")
    result = await db_session.execute(query, {"name": tool_name})
    row = result.fetchone()

    if not row:
        return None

    return ToolResponse(**dict(row._asdict()))


async def _get_toolkit_by_name(db_session: AsyncSession, toolkit_name: str) -> ToolkitResponse:
    """Get a toolkit by name with its associated tools."""
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

    if not row:
        return None

    return ToolkitResponse(
        name=row.name,
        description=row.description,
        tool_names=row.tool_names,
        created_at=row.created_at,
        updated_at=row.updated_at
    )
