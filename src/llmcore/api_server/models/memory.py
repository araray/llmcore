# src/llmcore/api_server/models/memory.py
"""
Pydantic models for memory-related API endpoints.

This module defines the request and response models for the hierarchical
memory system API endpoints, ensuring proper data validation and clear
API contracts for memory operations.
"""

from typing import Optional
from pydantic import BaseModel, Field


class SemanticSearchRequest(BaseModel):
    """
    Request model for semantic memory search.

    This model defines the parameters for searching the semantic memory
    (vector store) via the API. Since this is used with a GET request,
    the actual endpoint will use query parameters, but this model serves
    as documentation and potential future POST endpoint support.
    """
    query: str = Field(description="The text query to search for.")
    collection_name: Optional[str] = Field(
        default=None,
        description="The target collection name. Uses default if not provided."
    )
    k: int = Field(
        default=3,
        ge=1,
        le=20,
        description="The number of results to return."
    )

    class Config:
        extra = 'forbid'  # Reject unknown fields for strict validation
