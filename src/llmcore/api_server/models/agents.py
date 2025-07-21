# llmcore/src/llmcore/api_server/models/agents.py
"""
Pydantic models for agent-related API endpoints.

This module defines the request and response models for the agent execution
API endpoints, ensuring proper data validation and clear API contracts.
"""

from typing import Optional
from pydantic import BaseModel, Field


class AgentRunRequest(BaseModel):
    """
    Request model for running an autonomous agent task.

    This model defines the parameters for starting a new agent that will
    work autonomously to achieve a high-level goal using the Think -> Act -> Observe loop.
    """
    goal: str = Field(description="The high-level goal for the agent to achieve.")
    session_id: Optional[str] = Field(
        default=None,
        description="The session ID for context and episodic memory."
    )
    provider: Optional[str] = Field(
        default=None,
        description="Override the default LLM provider."
    )
    model: Optional[str] = Field(
        default=None,
        description="Override the default model for the provider."
    )

    class Config:
        extra = 'forbid'  # Reject unknown fields for strict validation
