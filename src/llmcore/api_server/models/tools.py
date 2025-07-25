# llmcore/src/llmcore/api_server/models/tools.py
"""
Pydantic models for tool and toolkit management API endpoints.

This module defines the request and response models for the dynamic tool
management system, ensuring proper data validation and clear API contracts.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


# Tool Management Models

class ToolCreateRequest(BaseModel):
    """Request model for creating a new tool."""
    name: str = Field(description="Unique name of the tool")
    description: str = Field(description="Natural language description for the LLM")
    parameters_schema: Dict[str, Any] = Field(description="JSON Schema defining tool parameters")
    implementation_key: str = Field(description="Key mapping to the secure implementation function")

    class Config:
        extra = 'forbid'


class ToolUpdateRequest(BaseModel):
    """Request model for updating an existing tool."""
    description: Optional[str] = Field(default=None, description="Updated description")
    parameters_schema: Optional[Dict[str, Any]] = Field(default=None, description="Updated parameter schema")
    is_enabled: Optional[bool] = Field(default=None, description="Enable/disable the tool")

    class Config:
        extra = 'forbid'


class ToolResponse(BaseModel):
    """Response model for tool information."""
    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    parameters_schema: Dict[str, Any] = Field(description="Tool parameter schema")
    implementation_key: str = Field(description="Implementation key")
    is_enabled: bool = Field(description="Whether the tool is enabled")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")

    class Config:
        from_attributes = True


# Toolkit Management Models

class ToolkitCreateRequest(BaseModel):
    """Request model for creating a new toolkit."""
    name: str = Field(description="Unique name of the toolkit")
    description: Optional[str] = Field(default=None, description="Description of the toolkit's purpose")
    tool_names: List[str] = Field(default_factory=list, description="List of tool names to include in this toolkit")

    class Config:
        extra = 'forbid'


class ToolkitUpdateRequest(BaseModel):
    """Request model for updating an existing toolkit."""
    description: Optional[str] = Field(default=None, description="Updated description")
    tool_names: Optional[List[str]] = Field(default=None, description="Updated list of tool names")

    class Config:
        extra = 'forbid'


class ToolkitResponse(BaseModel):
    """Response model for toolkit information."""
    name: str = Field(description="Toolkit name")
    description: Optional[str] = Field(description="Toolkit description")
    tool_names: List[str] = Field(description="List of tools in this toolkit")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")

    class Config:
        from_attributes = True


# Utility Models

class ToolExecutionRequest(BaseModel):
    """Request model for testing tool execution."""
    tool_name: str = Field(description="Name of the tool to execute")
    arguments: Dict[str, Any] = Field(description="Arguments to pass to the tool")

    class Config:
        extra = 'forbid'


class AvailableImplementationsResponse(BaseModel):
    """Response model listing available implementation keys."""
    implementation_keys: List[str] = Field(description="List of available implementation keys")
    descriptions: Dict[str, str] = Field(description="Mapping of keys to their descriptions")

    class Config:
        extra = 'forbid'
