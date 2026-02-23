# src/llmcore/sessions/__init__.py
"""
Session management module for the LLMCore library.

This package handles the lifecycle and content of chat sessions,
including conversation history and crash recovery for autonomous agents.

Components:
    - SessionManager: Core session CRUD operations
    - SessionRecovery: Checkpoint-based crash recovery
"""

from .manager import SessionManager
from .recovery import CheckpointStatus, RecoveryCheckpoint, SessionRecovery

__all__ = [
    "SessionManager",
    "SessionRecovery",
    "RecoveryCheckpoint",
    "CheckpointStatus",
]
