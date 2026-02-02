# tests/storage/conftest.py
"""
Pytest configuration and fixtures for storage tests.

Phase 2 (NEXUS): Provides configurable PostgreSQL connection for integration tests.

Configuration via environment variables:
    LLMCORE_TEST_PG_HOST: PostgreSQL host (default: localhost)
    LLMCORE_TEST_PG_PORT: PostgreSQL port (default: 5432)
    LLMCORE_TEST_PG_USER: PostgreSQL user (default: postgres)
    LLMCORE_TEST_PG_PASSWORD: PostgreSQL password (default: postgres)
    LLMCORE_TEST_PG_DATABASE: PostgreSQL database (default: llmcore_test)
    LLMCORE_SKIP_PG_TESTS: Skip PostgreSQL integration tests (default: false)

Usage:
    # Run with default local PostgreSQL
    pytest tests/storage/test_pgvector_enhanced.py

    # Run with custom PostgreSQL
    LLMCORE_TEST_PG_HOST=myserver LLMCORE_TEST_PG_PORT=5433 pytest tests/storage/

    # Skip PostgreSQL tests
    LLMCORE_SKIP_PG_TESTS=1 pytest tests/storage/
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, AsyncGenerator, Dict

import pytest

# =============================================================================
# POSTGRESQL CONFIGURATION
# =============================================================================

def get_pg_config() -> Dict[str, Any]:
    """
    Get PostgreSQL configuration from environment variables.

    Returns:
        Dictionary with PostgreSQL connection parameters
    """
    return {
        "host": os.environ.get("LLMCORE_TEST_PG_HOST", "localhost"),
        "port": int(os.environ.get("LLMCORE_TEST_PG_PORT", "5432")),
        "user": os.environ.get("LLMCORE_TEST_PG_USER", "postgres"),
        "password": os.environ.get("LLMCORE_TEST_PG_PASSWORD", "postgres"),
        "database": os.environ.get("LLMCORE_TEST_PG_DATABASE", "llmcore_test"),
    }


def get_pg_url() -> str:
    """
    Build PostgreSQL connection URL from environment variables.

    Returns:
        PostgreSQL connection URL
    """
    config = get_pg_config()
    return (
        f"postgresql://{config['user']}:{config['password']}"
        f"@{config['host']}:{config['port']}/{config['database']}"
    )


def should_skip_pg_tests() -> bool:
    """
    Check if PostgreSQL tests should be skipped.

    Returns:
        True if LLMCORE_SKIP_PG_TESTS is set to a truthy value
    """
    skip = os.environ.get("LLMCORE_SKIP_PG_TESTS", "").lower()
    return skip in ("1", "true", "yes", "on")


# =============================================================================
# PYTEST MARKERS
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers",
        "requires_postgres: mark test as requiring PostgreSQL (skip if not available)"
    )
    config.addinivalue_line(
        "markers",
        "requires_pgvector: mark test as requiring pgvector extension"
    )


def pytest_collection_modifyitems(config, items):
    """Skip PostgreSQL tests if configured or unavailable."""
    if should_skip_pg_tests():
        skip_pg = pytest.mark.skip(reason="PostgreSQL tests disabled via LLMCORE_SKIP_PG_TESTS")
        for item in items:
            if "requires_postgres" in item.keywords or "requires_pgvector" in item.keywords:
                item.add_marker(skip_pg)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def pg_config() -> Dict[str, Any]:
    """PostgreSQL configuration fixture."""
    return get_pg_config()


@pytest.fixture(scope="session")
def pg_url() -> str:
    """PostgreSQL URL fixture."""
    return get_pg_url()


@pytest.fixture(scope="session")
def storage_config(pg_url: str) -> Dict[str, Any]:
    """
    Storage configuration for testing.

    Returns:
        Configuration dictionary for EnhancedPgVectorStorage
    """
    return {
        "db_url": pg_url,
        "vectors_table_name": "test_vectors",
        "collections_table_name": "test_vector_collections",
        "default_collection": "test_collection",
        "default_vector_dimension": 384,
        "hnsw_m": 8,
        "hnsw_ef_construction": 32,
        "hnsw_ef_search": 20,
        "min_pool_size": 1,
        "max_pool_size": 3,
    }


@pytest.fixture
async def pg_pool(pg_url: str) -> AsyncGenerator[Any, None]:
    """
    Create a PostgreSQL connection pool for testing.

    Yields:
        Connection pool (or None if psycopg not available)
    """
    try:
        from psycopg_pool import AsyncConnectionPool
        pool = AsyncConnectionPool(conninfo=pg_url, min_size=1, max_size=3)
        yield pool
        await pool.close()
    except ImportError:
        yield None


@pytest.fixture
async def clean_test_tables(pg_pool) -> AsyncGenerator[None, None]:
    """
    Clean up test tables before and after tests.

    This fixture ensures test isolation by dropping and recreating tables.
    """
    if pg_pool is None:
        yield
        return

    async def drop_tables():
        async with pg_pool.connection() as conn:
            async with conn.transaction():
                await conn.execute("DROP TABLE IF EXISTS test_vectors CASCADE")
                await conn.execute("DROP TABLE IF EXISTS test_vector_collections CASCADE")

    # Clean before test
    await drop_tables()
    yield
    # Clean after test
    await drop_tables()


# =============================================================================
# HELPER FIXTURES FOR MOCKING
# =============================================================================

@pytest.fixture
def mock_context():
    """Create a mock StorageContext."""
    # Try llmcore package first, then direct import
    try:
        from llmcore.storage.abstraction import StorageContext
    except ImportError:
        _storage_path = Path(__file__).parent.parent.parent / "src" / "llmcore" / "storage"
        if str(_storage_path) not in sys.path:
            sys.path.insert(0, str(_storage_path))
        from abstraction import StorageContext
    return StorageContext(user_id="test_user_123", namespace="test_ns")


@pytest.fixture
def mock_documents():
    """Create mock documents for testing."""
    # Import here to avoid issues if models aren't available
    try:
        from llmcore.models import ContextDocument
    except ImportError:
        # Create a simple stand-in
        from dataclasses import dataclass, field
        from typing import List, Optional

        @dataclass
        class ContextDocument:
            id: str
            content: str = ""
            metadata: Dict[str, Any] = field(default_factory=dict)
            embedding: Optional[List[float]] = None
            score: Optional[float] = None

    # Generate test documents with random embeddings
    import random
    random.seed(42)

    docs = []
    for i in range(5):
        docs.append(ContextDocument(
            id=f"doc_{i}",
            content=f"This is test document number {i} with some content for testing.",
            metadata={"source": "test", "index": i},
            embedding=[random.random() for _ in range(384)],
        ))
    return docs


@pytest.fixture
def sample_embedding():
    """Create a sample embedding vector."""
    import random
    random.seed(42)
    return [random.random() for _ in range(384)]


# =============================================================================
# SKIP HELPERS
# =============================================================================

requires_postgres = pytest.mark.requires_postgres
requires_pgvector = pytest.mark.requires_pgvector


def skip_if_no_postgres(func):
    """Decorator to skip test if PostgreSQL is not available."""
    return pytest.mark.skipif(
        should_skip_pg_tests(),
        reason="PostgreSQL tests disabled"
    )(func)


# =============================================================================
# ASYNC TEST HELPERS
# =============================================================================

@pytest.fixture
def async_test_timeout():
    """Default timeout for async tests."""
    return 30.0
