# llmcore/src/llmcore/api_server/schemas/security.py
"""
Pydantic models for security-related database entities.

This module defines the data models for tenant and API key management,
representing the database records used in the authentication system.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class Tenant(BaseModel):
    """
    Represents a tenant entity in the multi-tenant system.

    This model represents the database record for a tenant, providing
    the foundational element for data segregation and per-tenant resource management.
    """
    id: UUID = Field(description="Unique identifier for the tenant")
    name: str = Field(description="Human-readable name for tenant identification")
    db_schema_name: str = Field(description="PostgreSQL schema name for this tenant's data")
    created_at: datetime = Field(description="Timestamp of tenant creation")
    status: str = Field(description="Current status of the tenant (active, suspended, etc.)")

    class Config:
        """Pydantic model configuration."""
        from_attributes = True  # Enable ORM mode for SQLAlchemy compatibility


class APIKey(BaseModel):
    """
    Represents an API key record in the authentication system.

    This model represents the database record for API keys, linking
    authentication tokens to specific tenants for secure access control.
    """
    id: UUID = Field(description="Unique identifier for the API key")
    hashed_key: str = Field(description="Securely hashed API key value")
    key_prefix: str = Field(description="Non-sensitive prefix for key identification")
    tenant_id: UUID = Field(description="ID of the tenant this key belongs to")
    created_at: datetime = Field(description="Timestamp of key creation")
    expires_at: Optional[datetime] = Field(default=None, description="Optional expiration timestamp")
    last_used_at: Optional[datetime] = Field(default=None, description="Last usage timestamp for auditing")

    class Config:
        """Pydantic model configuration."""
        from_attributes = True  # Enable ORM mode for SQLAlchemy compatibility
