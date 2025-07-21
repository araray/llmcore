# src/llmcore/agents/__init__.py
"""
Agents module for the LLMCore library.

This package contains the agentic execution engine including the AgentManager
for orchestrating the Think -> Act -> Observe loop and the ToolManager for
dynamic tool registration and execution.
"""

from .manager import AgentManager
from .tools import ToolManager

__all__ = ["AgentManager", "ToolManager"]
