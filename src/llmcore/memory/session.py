# src/llmcore/memory/session.py
"""
Session Memory — Conversation-scoped persistent memory.

This module provides the spec-mandated ``memory/session.py`` entry-point.
Session memory tracks the full conversation history within a single session,
including messages, tool call results, and metadata.

The actual session management lives in:

- :class:`~llmcore.sessions.manager.SessionManager` — CRUD operations
- :class:`~llmcore.storage.base_session.BaseSessionStorage` — persistence
- :class:`~llmcore.sessions.recovery.SessionRecovery` — crash recovery

This module re-exports the key types so they are accessible via the
``memory/`` namespace as the spec envisions.

References:
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §8 (Memory System)
"""

from __future__ import annotations

from ..models import ChatSession, Message, Role
from ..sessions.manager import SessionManager

__all__ = [
    "ChatSession",
    "Message",
    "Role",
    "SessionManager",
]
