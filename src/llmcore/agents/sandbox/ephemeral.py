# src/llmcore/agents/sandbox/ephemeral.py
"""
Ephemeral resource management for sandbox environments.

This module provides utilities for managing ephemeral resources within
sandboxes, including:
    - SQLite databases for agent state persistence during a task
    - Temporary file management
    - Resource cleanup

The ephemeral SQLite database is created when a sandbox starts and
destroyed when it's cleaned up. It provides agents with a way to
persist state across iterations without affecting the host system.

Usage:
    >>> manager = EphemeralResourceManager(sandbox)
    >>> await manager.init_database()
    >>> await manager.set_state("current_step", "3")
    >>> step = await manager.get_state("current_step")
    >>> await manager.log_event("INFO", "Processing step 3")
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .base import SandboxProvider

logger = logging.getLogger(__name__)


@dataclass
class AgentLogEntry:
    """
    A log entry from the ephemeral database.

    Attributes:
        id: Unique log entry ID
        timestamp: When the log was created
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        message: Log message content
    """

    id: int
    timestamp: datetime
    level: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "message": self.message,
        }


@dataclass
class FileRecord:
    """
    Record of a file created by the agent.

    Attributes:
        path: File path in sandbox
        created_at: When the file was created
        size_bytes: File size
        description: Optional description of the file
    """

    path: str
    created_at: datetime
    size_bytes: int = 0
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "path": self.path,
            "created_at": self.created_at.isoformat(),
            "size_bytes": self.size_bytes,
            "description": self.description,
        }


class EphemeralResourceManager:
    """
    Manages ephemeral resources within a sandbox.

    This class provides a high-level interface for agents to:
        - Store and retrieve key-value state
        - Log events during execution
        - Track files created during the task

    All data is stored in an ephemeral SQLite database that is
    destroyed when the sandbox is cleaned up.

    Example:
        >>> manager = EphemeralResourceManager(sandbox)
        >>>
        >>> # Store agent state
        >>> await manager.set_state("iteration", "5")
        >>> await manager.set_state("plan", json.dumps(["step1", "step2"]))
        >>>
        >>> # Retrieve state
        >>> iteration = await manager.get_state("iteration")
        >>>
        >>> # Log events
        >>> await manager.log_event("INFO", "Starting iteration 5")
        >>>
        >>> # Track created files
        >>> await manager.record_file("/workspace/output.py", 1024, "Generated code")
    """

    def __init__(self, sandbox: "SandboxProvider", db_path: Optional[str] = None):
        """
        Initialize the ephemeral resource manager.

        Args:
            sandbox: The sandbox provider to use for execution
            db_path: Path to SQLite database (uses config default if None)
        """
        self._sandbox = sandbox
        self._db_path = db_path or (
            sandbox.get_config().ephemeral_db_path if sandbox.get_config() else "/tmp/agent_task.db"
        )

    async def init_database(self) -> bool:
        """
        Initialize the ephemeral SQLite database.

        Creates the necessary tables if they don't exist.

        Returns:
            True if initialization was successful
        """
        init_sql = """
CREATE TABLE IF NOT EXISTS agent_state (
    key TEXT PRIMARY KEY,
    value TEXT,
    value_type TEXT DEFAULT 'string',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS agent_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    level TEXT NOT NULL,
    message TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_files (
    path TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    size_bytes INTEGER DEFAULT 0,
    description TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON agent_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_logs_level ON agent_logs(level);
"""

        result = await self._sandbox.execute_shell(
            f"sqlite3 '{self._db_path}' << 'EOF'\n{init_sql}\nEOF"
        )

        if result.success:
            logger.debug(f"Initialized ephemeral database at {self._db_path}")
            return True
        else:
            logger.warning(f"Failed to initialize ephemeral database: {result.stderr}")
            return False

    async def set_state(self, key: str, value: Any, value_type: str = "auto") -> bool:
        """
        Store a value in the agent state.

        Args:
            key: State key
            value: Value to store (will be JSON-serialized if not string)
            value_type: Type hint ("string", "json", "number", "auto")

        Returns:
            True if successful
        """
        # Determine value type and serialize
        if value_type == "auto":
            if isinstance(value, str):
                value_type = "string"
                serialized = value
            elif isinstance(value, (int, float)):
                value_type = "number"
                serialized = str(value)
            else:
                value_type = "json"
                serialized = json.dumps(value)
        elif value_type == "json":
            serialized = json.dumps(value) if not isinstance(value, str) else value
        else:
            serialized = str(value)

        # Escape for SQL
        key_escaped = key.replace("'", "''")
        value_escaped = serialized.replace("'", "''")

        sql = f"""
INSERT OR REPLACE INTO agent_state (key, value, value_type, updated_at)
VALUES ('{key_escaped}', '{value_escaped}', '{value_type}', datetime('now'));
"""

        result = await self._sandbox.execute_shell(f"sqlite3 '{self._db_path}' \"{sql}\"")

        return result.success

    async def get_state(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value from the agent state.

        Args:
            key: State key
            default: Default value if key not found

        Returns:
            The stored value (deserialized if JSON), or default
        """
        key_escaped = key.replace("'", "''")
        sql = f"SELECT value, value_type FROM agent_state WHERE key = '{key_escaped}';"

        result = await self._sandbox.execute_shell(
            f"sqlite3 -separator '|' '{self._db_path}' \"{sql}\""
        )

        if result.success and result.stdout.strip():
            parts = result.stdout.strip().split("|", 1)
            if len(parts) >= 1:
                value = parts[0]
                value_type = parts[1] if len(parts) > 1 else "string"

                if value_type == "json":
                    try:
                        return json.loads(value)
                    except json.JSONDecodeError:
                        return value
                elif value_type == "number":
                    try:
                        if "." in value:
                            return float(value)
                        return int(value)
                    except ValueError:
                        return value
                return value

        return default

    async def delete_state(self, key: str) -> bool:
        """
        Delete a state entry.

        Args:
            key: State key to delete

        Returns:
            True if successful
        """
        key_escaped = key.replace("'", "''")
        sql = f"DELETE FROM agent_state WHERE key = '{key_escaped}';"

        result = await self._sandbox.execute_shell(f"sqlite3 '{self._db_path}' \"{sql}\"")

        return result.success

    async def list_state_keys(self) -> List[str]:
        """
        List all state keys.

        Returns:
            List of state keys
        """
        sql = "SELECT key FROM agent_state ORDER BY key;"

        result = await self._sandbox.execute_shell(f"sqlite3 '{self._db_path}' \"{sql}\"")

        if result.success and result.stdout.strip():
            return [k.strip() for k in result.stdout.strip().split("\n") if k.strip()]
        return []

    async def get_all_state(self) -> Dict[str, Any]:
        """
        Get all state as a dictionary.

        Returns:
            Dictionary of all state key-value pairs
        """
        sql = "SELECT key, value, value_type FROM agent_state ORDER BY key;"

        result = await self._sandbox.execute_shell(
            f"sqlite3 -separator '|' '{self._db_path}' \"{sql}\""
        )

        state = {}
        if result.success and result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                parts = line.split("|", 2)
                if len(parts) >= 2:
                    key = parts[0]
                    value = parts[1]
                    value_type = parts[2] if len(parts) > 2 else "string"

                    if value_type == "json":
                        try:
                            state[key] = json.loads(value)
                        except json.JSONDecodeError:
                            state[key] = value
                    elif value_type == "number":
                        try:
                            state[key] = float(value) if "." in value else int(value)
                        except ValueError:
                            state[key] = value
                    else:
                        state[key] = value

        return state

    async def log_event(self, level: str, message: str) -> bool:
        """
        Log an event to the ephemeral database.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            message: Log message

        Returns:
            True if successful
        """
        level_escaped = level.upper().replace("'", "''")
        message_escaped = message.replace("'", "''")

        sql = f"""
INSERT INTO agent_logs (timestamp, level, message)
VALUES (datetime('now'), '{level_escaped}', '{message_escaped}');
"""

        result = await self._sandbox.execute_shell(f"sqlite3 '{self._db_path}' \"{sql}\"")

        return result.success

    async def get_logs(
        self, level: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> List[AgentLogEntry]:
        """
        Retrieve log entries.

        Args:
            level: Optional filter by log level
            limit: Maximum entries to return
            offset: Offset for pagination

        Returns:
            List of AgentLogEntry objects
        """
        where_clause = ""
        if level:
            level_escaped = level.upper().replace("'", "''")
            where_clause = f"WHERE level = '{level_escaped}'"

        sql = f"""
SELECT id, timestamp, level, message
FROM agent_logs
{where_clause}
ORDER BY timestamp DESC
LIMIT {limit} OFFSET {offset};
"""

        result = await self._sandbox.execute_shell(
            f"sqlite3 -separator '|' '{self._db_path}' \"{sql}\""
        )

        logs = []
        if result.success and result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                parts = line.split("|", 3)
                if len(parts) >= 4:
                    try:
                        logs.append(
                            AgentLogEntry(
                                id=int(parts[0]),
                                timestamp=datetime.fromisoformat(parts[1].replace(" ", "T")),
                                level=parts[2],
                                message=parts[3],
                            )
                        )
                    except (ValueError, IndexError):
                        continue

        return logs

    async def record_file(self, path: str, size_bytes: int = 0, description: str = "") -> bool:
        """
        Record a file created by the agent.

        Args:
            path: File path in sandbox
            size_bytes: File size
            description: Description of the file

        Returns:
            True if successful
        """
        path_escaped = path.replace("'", "''")
        desc_escaped = description.replace("'", "''")

        sql = f"""
INSERT OR REPLACE INTO agent_files (path, created_at, size_bytes, description)
VALUES ('{path_escaped}', datetime('now'), {size_bytes}, '{desc_escaped}');
"""

        result = await self._sandbox.execute_shell(f"sqlite3 '{self._db_path}' \"{sql}\"")

        return result.success

    async def list_recorded_files(self) -> List[FileRecord]:
        """
        List all recorded files.

        Returns:
            List of FileRecord objects
        """
        sql = """
SELECT path, created_at, size_bytes, description
FROM agent_files
ORDER BY created_at DESC;
"""

        result = await self._sandbox.execute_shell(
            f"sqlite3 -separator '|' '{self._db_path}' \"{sql}\""
        )

        files = []
        if result.success and result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                parts = line.split("|", 3)
                if len(parts) >= 4:
                    try:
                        files.append(
                            FileRecord(
                                path=parts[0],
                                created_at=datetime.fromisoformat(parts[1].replace(" ", "T")),
                                size_bytes=int(parts[2]) if parts[2] else 0,
                                description=parts[3],
                            )
                        )
                    except (ValueError, IndexError):
                        continue

        return files

    async def clear_logs(self) -> bool:
        """
        Clear all log entries.

        Returns:
            True if successful
        """
        result = await self._sandbox.execute_shell(
            f"sqlite3 '{self._db_path}' 'DELETE FROM agent_logs;'"
        )
        return result.success

    async def clear_state(self) -> bool:
        """
        Clear all state entries.

        Returns:
            True if successful
        """
        result = await self._sandbox.execute_shell(
            f"sqlite3 '{self._db_path}' 'DELETE FROM agent_state;'"
        )
        return result.success

    async def get_database_size(self) -> int:
        """
        Get the size of the ephemeral database in bytes.

        Returns:
            Database file size in bytes
        """
        result = await self._sandbox.execute_shell(
            f"stat -c %s '{self._db_path}' 2>/dev/null || echo '0'"
        )

        if result.success:
            try:
                return int(result.stdout.strip())
            except ValueError:
                return 0
        return 0

    async def export_state(self) -> Dict[str, Any]:
        """
        Export all ephemeral data as a dictionary.

        Useful for preserving agent state before cleanup.

        Returns:
            Dictionary with state, logs, and files
        """
        return {
            "state": await self.get_all_state(),
            "logs": [log.to_dict() for log in await self.get_logs(limit=1000)],
            "files": [f.to_dict() for f in await self.list_recorded_files()],
        }
