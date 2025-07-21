# src/llmcore/api_server/models.py
"""
Pydantic models for the llmcore API server.

This module defines the request and response models for the FastAPI server,
ensuring proper data validation and clear API contracts.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Request model for the chat endpoint.

    This model mirrors the parameters of the LLMCore.chat() method,
    ensuring API compatibility with the core library functionality.
    """
    message: str = Field(description="The user's input message")
    session_id: Optional[str] = Field(default=None, description="The ID of the conversation session")
    system_message: Optional[str] = Field(default=None, description="A message defining the LLM's behavior")
    provider_name: Optional[str] = Field(default=None, description="Override the default LLM provider for this call")
    model_name: Optional[str] = Field(default=None, description="Override the default model for the chosen provider")
    stream: bool = Field(default=False, description="If True, returns a streaming response")
    save_session: bool = Field(default=True, description="If True, saves the conversation turn to storage")
    provider_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional arguments passed directly to the provider's API"
    )

    class Config:
        extra = 'forbid'  # Reject unknown fields for strict validation


class ChatResponse(BaseModel):
    """
    Response model for non-streaming chat responses.
    """
    response: str = Field(description="The LLM's response message")
    session_id: Optional[str] = Field(default=None, description="The session ID used for this conversation")


class ErrorResponse(BaseModel):
    """
    Standard error response model.
    """
    detail: str = Field(description="Error message describing what went wrong")
    error_type: Optional[str] = Field(default=None, description="The type of error that occurred")
