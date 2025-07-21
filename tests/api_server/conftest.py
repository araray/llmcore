# tests/api_server/conftest.py
"""
Pytest configuration and fixtures for API server tests.

This module provides shared fixtures and configuration for testing
the llmcore API server components.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient

from llmcore.api_server.main import app


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_llmcore_instance():
    """
    Create a mock LLMCore instance for testing.

    This fixture provides a fully mocked LLMCore instance with reasonable
    default behaviors for testing API endpoints.
    """
    mock = AsyncMock()

    # Configure default mock behaviors
    mock.get_available_providers.return_value = ["openai", "anthropic", "ollama"]
    mock.chat.return_value = "This is a mock response from the LLM."
    mock.close.return_value = None

    return mock


@pytest.fixture
def api_client():
    """
    Create a FastAPI test client.

    This fixture provides a test client for making HTTP requests to the API
    endpoints during testing.
    """
    return TestClient(app)


@pytest.fixture
def api_client_with_llmcore(api_client, mock_llmcore_instance):
    """
    Create a FastAPI test client with a mocked LLMCore instance.

    This fixture combines the test client with a mocked LLMCore instance,
    simulating a fully functional API server for testing.
    """
    # Inject the mock LLMCore instance into the app state
    app.state.llmcore_instance = mock_llmcore_instance

    yield api_client

    # Cleanup: reset app state after test
    app.state.llmcore_instance = None


@pytest.fixture
def sample_chat_request():
    """
    Provide a sample chat request payload for testing.
    """
    return {
        "message": "Hello, this is a test message",
        "session_id": "test-session-123",
        "provider_name": "openai",
        "model_name": "gpt-4o",
        "stream": False,
        "save_session": True,
        "provider_kwargs": {
            "temperature": 0.7,
            "max_tokens": 100
        }
    }


@pytest.fixture
def sample_streaming_request():
    """
    Provide a sample streaming chat request payload for testing.
    """
    return {
        "message": "Tell me a story",
        "stream": True,
        "provider_name": "openai"
    }


class MockAsyncGenerator:
    """
    Helper class to create mock async generators for testing streaming responses.
    """

    def __init__(self, items):
        self.items = items
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item


@pytest.fixture
def mock_streaming_response():
    """
    Create a mock streaming response for testing.
    """
    return MockAsyncGenerator([
        "Hello", " ", "there", "! ", "This", " ", "is", " ", "a", " ",
        "streaming", " ", "response", "."
    ])


# Test markers for categorizing tests
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "api: mark test as an API test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
