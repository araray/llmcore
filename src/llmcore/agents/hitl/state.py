# src/llmcore/agents/hitl/state.py
"""
State Persistence for HITL System.

Provides persistence for:
- Pending approval requests (survive restart)
- Approval history
- Scope data

Implementations:
- InMemoryHITLStore: For testing
- FileHITLStore: JSON file-based persistence
- (Future: SQLiteHITLStore, RedisHITLStore)

References:
    - Master Plan: Section 20.3 (State Persistence)
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .models import (
    ApprovalStatus,
    HITLRequest,
    HITLResponse,
    PersistentScope,
    SessionScope,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ABSTRACT STATE STORE
# =============================================================================


class HITLStateStore(ABC):
    """
    Abstract base class for HITL state persistence.

    Stores:
    - Pending approval requests
    - Approval history
    - Scope data
    """

    @abstractmethod
    async def save_request(self, request: HITLRequest) -> None:
        """Save a pending approval request."""
        ...

    @abstractmethod
    async def get_request(self, request_id: str) -> Optional[HITLRequest]:
        """Get request by ID."""
        ...

    @abstractmethod
    async def update_request_status(
        self,
        request_id: str,
        status: ApprovalStatus,
        response: Optional[HITLResponse] = None,
    ) -> bool:
        """Update request status."""
        ...

    @abstractmethod
    async def get_pending_requests(
        self,
        session_id: Optional[str] = None,
    ) -> List[HITLRequest]:
        """Get all pending requests, optionally filtered by session."""
        ...

    @abstractmethod
    async def delete_request(self, request_id: str) -> bool:
        """Delete a request."""
        ...

    @abstractmethod
    async def save_response(self, response: HITLResponse) -> None:
        """Save an approval response."""
        ...

    @abstractmethod
    async def get_response(self, request_id: str) -> Optional[HITLResponse]:
        """Get response for a request."""
        ...

    @abstractmethod
    async def save_session_scope(self, scope: SessionScope) -> None:
        """Save session scope."""
        ...

    @abstractmethod
    async def get_session_scope(self, session_id: str) -> Optional[SessionScope]:
        """Get session scope."""
        ...

    @abstractmethod
    async def save_persistent_scope(self, scope: PersistentScope) -> None:
        """Save persistent scope."""
        ...

    @abstractmethod
    async def get_persistent_scope(self, user_id: str) -> Optional[PersistentScope]:
        """Get persistent scope."""
        ...

    @abstractmethod
    async def cleanup_expired(self) -> int:
        """Remove expired requests. Returns count removed."""
        ...

    @abstractmethod
    async def count_pending(self) -> int:
        """Count pending requests."""
        ...


# =============================================================================
# IN-MEMORY STORE
# =============================================================================


class InMemoryHITLStore(HITLStateStore):
    """
    In-memory HITL state store for testing.

    Data is lost when process ends.
    """

    def __init__(self):
        self._requests: Dict[str, HITLRequest] = {}
        self._responses: Dict[str, HITLResponse] = {}
        self._session_scopes: Dict[str, SessionScope] = {}
        self._persistent_scopes: Dict[str, PersistentScope] = {}
        self._lock = asyncio.Lock()

    async def save_request(self, request: HITLRequest) -> None:
        async with self._lock:
            self._requests[request.request_id] = request
            logger.debug(f"Saved request {request.request_id} to memory")

    async def get_request(self, request_id: str) -> Optional[HITLRequest]:
        async with self._lock:
            return self._requests.get(request_id)

    async def update_request_status(
        self,
        request_id: str,
        status: ApprovalStatus,
        response: Optional[HITLResponse] = None,
    ) -> bool:
        async with self._lock:
            if request_id not in self._requests:
                return False

            self._requests[request_id].status = status

            if response:
                self._responses[request_id] = response

            return True

    async def get_pending_requests(
        self,
        session_id: Optional[str] = None,
    ) -> List[HITLRequest]:
        async with self._lock:
            pending = [r for r in self._requests.values() if r.status == ApprovalStatus.PENDING]

            if session_id:
                pending = [r for r in pending if r.session_id == session_id]

            return pending

    async def delete_request(self, request_id: str) -> bool:
        async with self._lock:
            if request_id in self._requests:
                del self._requests[request_id]
                self._responses.pop(request_id, None)
                return True
            return False

    async def save_response(self, response: HITLResponse) -> None:
        async with self._lock:
            self._responses[response.request_id] = response

    async def get_response(self, request_id: str) -> Optional[HITLResponse]:
        async with self._lock:
            return self._responses.get(request_id)

    async def save_session_scope(self, scope: SessionScope) -> None:
        async with self._lock:
            self._session_scopes[scope.session_id] = scope

    async def get_session_scope(self, session_id: str) -> Optional[SessionScope]:
        async with self._lock:
            return self._session_scopes.get(session_id)

    async def save_persistent_scope(self, scope: PersistentScope) -> None:
        async with self._lock:
            self._persistent_scopes[scope.user_id] = scope

    async def get_persistent_scope(self, user_id: str) -> Optional[PersistentScope]:
        async with self._lock:
            return self._persistent_scopes.get(user_id)

    async def cleanup_expired(self) -> int:
        """Remove expired requests."""
        async with self._lock:
            now = datetime.now()
            expired = [
                rid
                for rid, req in self._requests.items()
                if req.status == ApprovalStatus.PENDING and req.is_expired
            ]

            for rid in expired:
                self._requests[rid].status = ApprovalStatus.TIMEOUT
                logger.debug(f"Marked request {rid} as expired")

            return len(expired)

    async def count_pending(self) -> int:
        async with self._lock:
            return sum(1 for r in self._requests.values() if r.status == ApprovalStatus.PENDING)

    def clear(self) -> None:
        """Clear all data (for testing)."""
        self._requests.clear()
        self._responses.clear()
        self._session_scopes.clear()
        self._persistent_scopes.clear()


# =============================================================================
# FILE-BASED STORE
# =============================================================================


class FileHITLStore(HITLStateStore):
    """
    File-based HITL state store using JSON.

    Persists to disk:
    - requests.json: All requests
    - responses.json: All responses
    - scopes/session_{id}.json: Session scopes
    - scopes/persistent_{user_id}.json: Persistent scopes
    """

    def __init__(
        self,
        storage_path: str | Path,
        auto_save: bool = True,
    ):
        """
        Initialize file store.

        Args:
            storage_path: Directory for storage files
            auto_save: Whether to save immediately on changes
        """
        self.storage_path = Path(storage_path)
        self.auto_save = auto_save

        self._requests: Dict[str, HITLRequest] = {}
        self._responses: Dict[str, HITLResponse] = {}
        self._session_scopes: Dict[str, SessionScope] = {}
        self._persistent_scopes: Dict[str, PersistentScope] = {}
        self._lock = asyncio.Lock()
        self._loaded = False

    async def _ensure_loaded(self) -> None:
        """Ensure data is loaded from disk."""
        if self._loaded:
            return

        async with self._lock:
            if self._loaded:
                return

            self.storage_path.mkdir(parents=True, exist_ok=True)
            scopes_path = self.storage_path / "scopes"
            scopes_path.mkdir(exist_ok=True)

            # Load requests
            requests_file = self.storage_path / "requests.json"
            if requests_file.exists():
                try:
                    data = json.loads(requests_file.read_text())
                    for req_data in data.get("requests", []):
                        req = HITLRequest.from_dict(req_data)
                        self._requests[req.request_id] = req
                    logger.info(f"Loaded {len(self._requests)} requests from {requests_file}")
                except Exception as e:
                    logger.warning(f"Failed to load requests: {e}")

            # Load responses
            responses_file = self.storage_path / "responses.json"
            if responses_file.exists():
                try:
                    data = json.loads(responses_file.read_text())
                    for resp_data in data.get("responses", []):
                        resp = HITLResponse(
                            request_id=resp_data["request_id"],
                            approved=resp_data["approved"],
                            modified_parameters=resp_data.get("modified_parameters"),
                            feedback=resp_data.get("feedback"),
                            responded_at=datetime.fromisoformat(resp_data["responded_at"])
                            if "responded_at" in resp_data
                            else datetime.now(),
                            responder_id=resp_data.get("responder_id", ""),
                        )
                        self._responses[resp.request_id] = resp
                    logger.info(f"Loaded {len(self._responses)} responses")
                except Exception as e:
                    logger.warning(f"Failed to load responses: {e}")

            # Load persistent scopes
            for scope_file in scopes_path.glob("persistent_*.json"):
                try:
                    data = json.loads(scope_file.read_text())
                    user_id = data["user_id"]
                    scope = PersistentScope(
                        user_id=user_id,
                        approved_tools=[],  # Will be populated below
                        created_at=datetime.fromisoformat(data["created_at"])
                        if "created_at" in data
                        else datetime.now(),
                        updated_at=datetime.fromisoformat(data["updated_at"])
                        if "updated_at" in data
                        else datetime.now(),
                    )
                    self._persistent_scopes[user_id] = scope
                except Exception as e:
                    logger.warning(f"Failed to load scope from {scope_file}: {e}")

            self._loaded = True

    async def _save_requests(self) -> None:
        """Save requests to disk."""
        if not self.auto_save:
            return

        try:
            requests_file = self.storage_path / "requests.json"
            data = {"requests": [req.to_dict() for req in self._requests.values()]}
            requests_file.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.warning(f"Failed to save requests: {e}")

    async def _save_responses(self) -> None:
        """Save responses to disk."""
        if not self.auto_save:
            return

        try:
            responses_file = self.storage_path / "responses.json"
            data = {"responses": [resp.to_dict() for resp in self._responses.values()]}
            responses_file.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.warning(f"Failed to save responses: {e}")

    async def save_request(self, request: HITLRequest) -> None:
        await self._ensure_loaded()
        async with self._lock:
            self._requests[request.request_id] = request
            await self._save_requests()
            logger.debug(f"Saved request {request.request_id}")

    async def get_request(self, request_id: str) -> Optional[HITLRequest]:
        await self._ensure_loaded()
        async with self._lock:
            return self._requests.get(request_id)

    async def update_request_status(
        self,
        request_id: str,
        status: ApprovalStatus,
        response: Optional[HITLResponse] = None,
    ) -> bool:
        await self._ensure_loaded()
        async with self._lock:
            if request_id not in self._requests:
                return False

            self._requests[request_id].status = status
            await self._save_requests()

            if response:
                self._responses[request_id] = response
                await self._save_responses()

            return True

    async def get_pending_requests(
        self,
        session_id: Optional[str] = None,
    ) -> List[HITLRequest]:
        await self._ensure_loaded()
        async with self._lock:
            pending = [r for r in self._requests.values() if r.status == ApprovalStatus.PENDING]

            if session_id:
                pending = [r for r in pending if r.session_id == session_id]

            return pending

    async def delete_request(self, request_id: str) -> bool:
        await self._ensure_loaded()
        async with self._lock:
            if request_id in self._requests:
                del self._requests[request_id]
                self._responses.pop(request_id, None)
                await self._save_requests()
                await self._save_responses()
                return True
            return False

    async def save_response(self, response: HITLResponse) -> None:
        await self._ensure_loaded()
        async with self._lock:
            self._responses[response.request_id] = response
            await self._save_responses()

    async def get_response(self, request_id: str) -> Optional[HITLResponse]:
        await self._ensure_loaded()
        async with self._lock:
            return self._responses.get(request_id)

    async def save_session_scope(self, scope: SessionScope) -> None:
        await self._ensure_loaded()
        async with self._lock:
            self._session_scopes[scope.session_id] = scope

            # Session scopes are ephemeral, optionally save
            if self.auto_save:
                try:
                    scope_file = self.storage_path / "scopes" / f"session_{scope.session_id}.json"
                    scope_file.write_text(json.dumps(scope.to_dict(), indent=2, default=str))
                except Exception as e:
                    logger.warning(f"Failed to save session scope: {e}")

    async def get_session_scope(self, session_id: str) -> Optional[SessionScope]:
        await self._ensure_loaded()
        async with self._lock:
            if session_id in self._session_scopes:
                return self._session_scopes[session_id]

            # Try loading from disk
            scope_file = self.storage_path / "scopes" / f"session_{session_id}.json"
            if scope_file.exists():
                try:
                    data = json.loads(scope_file.read_text())
                    scope = SessionScope.from_dict(data)
                    self._session_scopes[session_id] = scope
                    return scope
                except Exception as e:
                    logger.warning(f"Failed to load session scope: {e}")

            return None

    async def save_persistent_scope(self, scope: PersistentScope) -> None:
        await self._ensure_loaded()
        async with self._lock:
            self._persistent_scopes[scope.user_id] = scope

            if self.auto_save:
                try:
                    scope_file = self.storage_path / "scopes" / f"persistent_{scope.user_id}.json"
                    scope_file.write_text(json.dumps(scope.to_dict(), indent=2, default=str))
                except Exception as e:
                    logger.warning(f"Failed to save persistent scope: {e}")

    async def get_persistent_scope(self, user_id: str) -> Optional[PersistentScope]:
        await self._ensure_loaded()
        async with self._lock:
            if user_id in self._persistent_scopes:
                return self._persistent_scopes[user_id]

            # Try loading from disk
            scope_file = self.storage_path / "scopes" / f"persistent_{user_id}.json"
            if scope_file.exists():
                try:
                    data = json.loads(scope_file.read_text())
                    scope = PersistentScope(
                        user_id=data["user_id"],
                        approved_tools=[],
                        created_at=datetime.fromisoformat(
                            data.get("created_at", datetime.now().isoformat())
                        ),
                        updated_at=datetime.fromisoformat(
                            data.get("updated_at", datetime.now().isoformat())
                        ),
                    )
                    self._persistent_scopes[user_id] = scope
                    return scope
                except Exception as e:
                    logger.warning(f"Failed to load persistent scope: {e}")

            return None

    async def cleanup_expired(self) -> int:
        await self._ensure_loaded()
        async with self._lock:
            now = datetime.now()
            expired = [
                rid
                for rid, req in self._requests.items()
                if req.status == ApprovalStatus.PENDING and req.is_expired
            ]

            for rid in expired:
                self._requests[rid].status = ApprovalStatus.TIMEOUT

            if expired:
                await self._save_requests()

            return len(expired)

    async def count_pending(self) -> int:
        await self._ensure_loaded()
        async with self._lock:
            return sum(1 for r in self._requests.values() if r.status == ApprovalStatus.PENDING)

    async def flush(self) -> None:
        """Force save all data to disk."""
        await self._ensure_loaded()
        async with self._lock:
            await self._save_requests()
            await self._save_responses()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "FileHITLStore",
    "HITLStateStore",
    "InMemoryHITLStore",
]
