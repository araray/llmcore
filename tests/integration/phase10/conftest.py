# tests/integration/phase10/conftest.py
"""
Phase 10 Integration Test Fixtures

Provides common fixtures and configuration for Phase 10 integration tests.
These fixtures set up isolated test environments to ensure tests don't
interfere with each other.
"""

import os
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# ENVIRONMENT FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def clean_environment() -> Generator[None, None, None]:
    """Ensure clean environment for each test."""
    # Save original environment
    original_env = os.environ.copy()

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_api_keys() -> Generator[Dict[str, str], None, None]:
    """Provide mock API keys for testing."""
    mock_keys = {
        "OPENAI_API_KEY": "sk-test-mock-openai-key",
        "ANTHROPIC_API_KEY": "sk-test-mock-anthropic-key",
        "GOOGLE_API_KEY": "test-mock-google-key",
    }

    with patch.dict(os.environ, mock_keys):
        yield mock_keys


# =============================================================================
# STORAGE FIXTURES
# =============================================================================


@pytest.fixture
def temp_storage_dir(tmp_path: Path) -> Path:
    """Create a temporary storage directory."""
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


@pytest.fixture
def temp_sqlite_db(temp_storage_dir: Path) -> str:
    """Create a temporary SQLite database path."""
    return str(temp_storage_dir / "test.db")


@pytest.fixture
def temp_chromadb_dir(temp_storage_dir: Path) -> str:
    """Create a temporary ChromaDB directory."""
    chroma_dir = temp_storage_dir / "chromadb"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return str(chroma_dir)


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================


@pytest.fixture
def minimal_config_dict() -> Dict[str, Any]:
    """Minimal configuration dictionary for testing."""
    return {
        "llm": {
            "default_provider": "openai",
            "default_model": "gpt-4o",
            "temperature": 0.7,
        },
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dimensions": 1536,
        },
        "storage": {
            "session_backend": "sqlite",
            "vector_backend": "chromadb",
        },
        "observability": {
            "enabled": True,
            "metrics_enabled": True,
            "events_enabled": True,
            "cost_tracking_enabled": True,
        },
    }


@pytest.fixture
def full_config_dict(
    minimal_config_dict: Dict[str, Any],
    temp_storage_dir: Path,
) -> Dict[str, Any]:
    """Full configuration dictionary with all paths set to temp."""
    config = minimal_config_dict.copy()

    config["embedding"]["cache"] = {
        "enabled": True,
        "memory_size": 1000,
        "disk_enabled": True,
        "disk_path": str(temp_storage_dir / "embedding_cache.db"),
    }

    config["storage"]["sqlite"] = {
        "path": str(temp_storage_dir / "sessions.db"),
    }

    config["storage"]["chromadb"] = {
        "persist_directory": str(temp_storage_dir / "chromadb"),
    }

    config["observability"]["events"] = {
        "log_dir": str(temp_storage_dir / "events"),
    }

    config["observability"]["cost_tracking"] = {
        "db_path": str(temp_storage_dir / "costs.db"),
    }

    config["agents"] = {
        "enabled": True,
        "hitl": {
            "enabled": True,
            "db_path": str(temp_storage_dir / "hitl.db"),
            "require_approval_for": ["file_write", "shell_command"],
        },
    }

    return config


@pytest.fixture
def temp_config_file(tmp_path: Path, full_config_dict: Dict[str, Any]) -> Path:
    """Create a temporary TOML config file."""
    import toml

    config_path = tmp_path / "config.toml"

    def dict_to_toml(d: Dict[str, Any], prefix: str = "") -> str:
        """Recursively convert dict to TOML string."""
        lines = []
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                lines.append(f"\n[{full_key}]")
                for k, v in value.items():
                    if isinstance(v, dict):
                        lines.append(dict_to_toml({k: v}, full_key))
                    elif isinstance(v, str):
                        lines.append(f'{k} = "{v}"')
                    elif isinstance(v, bool):
                        lines.append(f"{k} = {str(v).lower()}")
                    elif isinstance(v, (int, float)):
                        lines.append(f"{k} = {v}")
                    elif isinstance(v, list):
                        lines.append(f"{k} = {v}")
            elif isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            elif isinstance(value, bool):
                lines.append(f"{key} = {str(value).lower()}")
            elif isinstance(value, (int, float)):
                lines.append(f"{key} = {value}")
            elif isinstance(value, list):
                lines.append(f"{key} = {value}")
        return "\n".join(lines)

    # Use toml library if available, otherwise write manually
    try:
        config_path.write_text(toml.dumps(full_config_dict))
    except Exception:
        # Simple manual TOML writing
        content = dict_to_toml(full_config_dict)
        config_path.write_text(content)

    return config_path


# =============================================================================
# MOCK FIXTURES
# =============================================================================


@pytest.fixture
def mock_llm_response() -> MagicMock:
    """Create a mock LLM response."""
    response = MagicMock()
    response.content = "This is a mock LLM response for testing."
    response.model = "gpt-4o"
    response.usage = {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
    }
    response.finish_reason = "stop"
    return response


@pytest.fixture
def mock_embedding() -> list:
    """Create a mock embedding vector."""
    return [0.1 + (i * 0.001) for i in range(1536)]


@pytest.fixture
def mock_llmcore():
    """Create a mock LLMCore instance."""
    mock = MagicMock()
    mock.chat.return_value = MagicMock(
        content="Mock response",
        usage={"input_tokens": 100, "output_tokens": 50},
    )
    mock.embed.return_value = [[0.1] * 1536]
    return mock


# =============================================================================
# HITL FIXTURES
# =============================================================================


@pytest.fixture
def mock_hitl_config(temp_storage_dir: Path) -> Dict[str, Any]:
    """Create HITL configuration for testing."""
    return {
        "enabled": True,
        "db_path": str(temp_storage_dir / "hitl.db"),
        "default_timeout_seconds": 3600,
        "require_approval_for": ["file_write", "shell_command", "network_request"],
        "auto_approve_low_risk": False,
    }


@pytest.fixture
def mock_approval_request() -> Dict[str, Any]:
    """Create a mock approval request."""
    from uuid import uuid4

    return {
        "task_id": f"task_{uuid4().hex[:8]}",
        "agent_run_id": f"run_{uuid4().hex[:8]}",
        "action_type": "file_write",
        "description": "Write test results to output file",
        "risk_level": "medium",
        "details": {
            "file_path": "/tmp/test_output.txt",
            "content_preview": "Test results...",
        },
    }


# =============================================================================
# SAMPLE DATA FIXTURES
# =============================================================================


@pytest.fixture
def sample_code_files() -> Dict[str, str]:
    """Sample code files for testing."""
    return {
        "src/auth.py": '''"""Authentication module."""

def authenticate(username: str, password: str) -> bool:
    """Authenticate a user."""
    # Check credentials
    return verify_password(password, get_user_hash(username))

def verify_password(plain: str, hashed: str) -> bool:
    """Verify password against hash."""
    import bcrypt
    return bcrypt.checkpw(plain.encode(), hashed.encode())
''',
        "src/api.py": '''"""API routes module."""

from fastapi import FastAPI, HTTPException
from auth import authenticate

app = FastAPI()

@app.post("/login")
async def login(username: str, password: str):
    """Login endpoint."""
    if not authenticate(username, password):
        raise HTTPException(401, "Invalid credentials")
    return {"status": "ok"}
''',
        "tests/test_auth.py": '''"""Tests for auth module."""

import pytest
from auth import authenticate, verify_password

def test_authenticate_valid():
    """Test valid authentication."""
    assert authenticate("admin", "correct_password")

def test_authenticate_invalid():
    """Test invalid authentication."""
    assert not authenticate("admin", "wrong_password")
''',
    }


@pytest.fixture
def sample_chunks(sample_code_files: Dict[str, str]) -> list:
    """Create sample chunks from code files."""
    chunks = []
    for i, (path, content) in enumerate(sample_code_files.items()):
        chunks.append(
            {
                "id": f"chunk_{i}",
                "content": content,
                "metadata": {
                    "file_path": path,
                    "language": "python",
                    "start_line": 1,
                    "end_line": content.count("\n") + 1,
                },
                "embedding": [0.1 + (i * 0.1)] * 1536,
                "score": 1.0 - (i * 0.1),
            }
        )
    return chunks


# =============================================================================
# CLEANUP FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def cleanup_metrics():
    """Clean up metrics registry after each test."""
    yield

    # Reset singleton registries
    try:
        from llmcore.observability.metrics import MetricsRegistry

        MetricsRegistry._instances = {}
    except Exception:
        pass


@pytest.fixture(autouse=True)
def cleanup_loggers():
    """Clean up loggers after each test."""
    import logging

    yield

    # Reset logging handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


# =============================================================================
# ASYNC FIXTURES
# =============================================================================


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# MARKERS
# =============================================================================


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "benchmark: marks tests as benchmarks")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
