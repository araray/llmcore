# tests/api_server/test_api_server.py
"""
Comprehensive unit and functional tests for the llmcore API server.

This test suite covers the FastAPI application, endpoints, error handling,
and integration with the LLMCore library.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException

from llmcore.api_server.main import app
from llmcore.api_server.models import ChatRequest, ChatResponse, ErrorResponse
from llmcore.exceptions import ProviderError, ContextLengthError, ConfigError, LLMCoreError


class TestAPIServerModels:
    """Test Pydantic models for request/response validation."""

    def test_chat_request_valid(self):
        """Test valid ChatRequest creation."""
        request = ChatRequest(
            message="Hello, world!",
            session_id="test-session",
            provider_name="openai",
            model_name="gpt-4o",
            stream=True,
            save_session=True,
            provider_kwargs={"temperature": 0.7}
        )

        assert request.message == "Hello, world!"
        assert request.session_id == "test-session"
        assert request.provider_name == "openai"
        assert request.model_name == "gpt-4o"
        assert request.stream is True
        assert request.save_session is True
        assert request.provider_kwargs == {"temperature": 0.7}

    def test_chat_request_minimal(self):
        """Test ChatRequest with only required fields."""
        request = ChatRequest(message="Hello")

        assert request.message == "Hello"
        assert request.session_id is None
        assert request.system_message is None
        assert request.provider_name is None
        assert request.model_name is None
        assert request.stream is False
        assert request.save_session is True
        assert request.provider_kwargs == {}

    def test_chat_request_extra_fields_rejected(self):
        """Test that extra fields are rejected."""
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            ChatRequest(
                message="Hello",
                invalid_field="should_fail"
            )

    def test_chat_response_valid(self):
        """Test valid ChatResponse creation."""
        response = ChatResponse(
            response="Hello back!",
            session_id="test-session"
        )

        assert response.response == "Hello back!"
        assert response.session_id == "test-session"

    def test_error_response_valid(self):
        """Test valid ErrorResponse creation."""
        error = ErrorResponse(
            detail="Something went wrong",
            error_type="ProviderError"
        )

        assert error.detail == "Something went wrong"
        assert error.error_type == "ProviderError"


class TestAPIServerRoutes:
    """Test API server route functionality."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)

    @pytest.fixture
    def mock_llmcore(self):
        """Create a mock LLMCore instance."""
        mock = AsyncMock()
        mock.get_available_providers.return_value = ["openai", "anthropic", "ollama"]
        mock.chat.return_value = "Hello, this is a test response!"
        return mock

    def test_root_endpoint(self, client):
        """Test the root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "llmcore API is running"
        assert data["version"] == "1.0.0"
        assert "docs_url" in data

    def test_health_endpoint_healthy(self, client, mock_llmcore):
        """Test health endpoint when service is healthy."""
        app.state.llmcore_instance = mock_llmcore

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["llmcore_available"] is True
        assert "openai" in data["providers"]

    def test_health_endpoint_degraded(self, client):
        """Test health endpoint when LLMCore is not available."""
        app.state.llmcore_instance = None

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["llmcore_available"] is False
        assert data["providers"] == []

    def test_info_endpoint_healthy(self, client, mock_llmcore):
        """Test info endpoint when service is healthy."""
        app.state.llmcore_instance = mock_llmcore

        response = client.get("/api/v1/info")

        assert response.status_code == 200
        data = response.json()
        assert data["api_version"] == "1.0"
        assert "llmcore_version" in data
        assert data["service_status"] == "healthy"
        assert data["features"]["chat"] is True
        assert data["features"]["streaming"] is True
        assert "openai" in data["features"]["providers"]

    def test_info_endpoint_degraded(self, client):
        """Test info endpoint when LLMCore is not available."""
        app.state.llmcore_instance = None

        response = client.get("/api/v1/info")

        assert response.status_code == 200
        data = response.json()
        assert data["service_status"] == "degraded"
        assert data["features"]["chat"] is False
        assert data["features"]["providers"] == []


class TestChatEndpoint:
    """Test the chat endpoint functionality."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)

    @pytest.fixture
    def mock_llmcore(self):
        """Create a mock LLMCore instance."""
        mock = AsyncMock()
        mock.get_available_providers.return_value = ["openai", "anthropic"]
        return mock

    def test_chat_non_streaming_success(self, client, mock_llmcore):
        """Test successful non-streaming chat request."""
        mock_llmcore.chat.return_value = "This is a test response"
        app.state.llmcore_instance = mock_llmcore

        response = client.post("/api/v1/chat", json={
            "message": "Hello, world!",
            "session_id": "test-session",
            "provider_name": "openai",
            "model_name": "gpt-4o",
            "stream": False
        })

        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "This is a test response"
        assert data["session_id"] == "test-session"

        # Verify the mock was called with correct parameters
        mock_llmcore.chat.assert_called_once_with(
            message="Hello, world!",
            session_id="test-session",
            provider_name="openai",
            model_name="gpt-4o",
            stream=False,
            save_session=True
        )

    def test_chat_streaming_success(self, client, mock_llmcore):
        """Test successful streaming chat request."""
        async def mock_stream():
            for chunk in ["Hello", " ", "world", "!"]:
                yield chunk

        mock_llmcore.chat.return_value = mock_stream()
        app.state.llmcore_instance = mock_llmcore

        response = client.post("/api/v1/chat", json={
            "message": "Tell me a story",
            "stream": True
        })

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # For streaming, we get the response as text
        content = response.text
        assert "Hello world!" in content

    def test_chat_with_provider_kwargs(self, client, mock_llmcore):
        """Test chat request with provider-specific kwargs."""
        mock_llmcore.chat.return_value = "Response with custom parameters"
        app.state.llmcore_instance = mock_llmcore

        response = client.post("/api/v1/chat", json={
            "message": "Generate creative content",
            "provider_kwargs": {
                "temperature": 0.9,
                "max_tokens": 500,
                "top_p": 0.95
            }
        })

        assert response.status_code == 200

        # Verify provider_kwargs were unpacked and passed to chat method
        mock_llmcore.chat.assert_called_once_with(
            message="Generate creative content",
            save_session=True,
            stream=False,
            temperature=0.9,
            max_tokens=500,
            top_p=0.95
        )

    def test_chat_llmcore_not_available(self, client):
        """Test chat request when LLMCore is not available."""
        app.state.llmcore_instance = None

        response = client.post("/api/v1/chat", json={
            "message": "Hello"
        })

        assert response.status_code == 503
        data = response.json()
        assert "LLMCore service is not available" in data["detail"]

    def test_chat_provider_error(self, client, mock_llmcore):
        """Test chat request with provider error."""
        mock_llmcore.chat.side_effect = ProviderError("openai", "API key invalid")
        app.state.llmcore_instance = mock_llmcore

        response = client.post("/api/v1/chat", json={
            "message": "Hello"
        })

        assert response.status_code == 400
        data = response.json()
        assert "Error with provider 'openai': API key invalid" in data["detail"]

    def test_chat_context_length_error(self, client, mock_llmcore):
        """Test chat request with context length error."""
        mock_llmcore.chat.side_effect = ContextLengthError(
            model_name="gpt-4o",
            limit=8000,
            actual=9000
        )
        app.state.llmcore_instance = mock_llmcore

        response = client.post("/api/v1/chat", json={
            "message": "Very long message..."
        })

        assert response.status_code == 400
        data = response.json()
        assert "Context length exceeded" in data["detail"]
        assert "gpt-4o" in data["detail"]

    def test_chat_config_error(self, client, mock_llmcore):
        """Test chat request with configuration error."""
        mock_llmcore.chat.side_effect = ConfigError("Invalid configuration")
        app.state.llmcore_instance = mock_llmcore

        response = client.post("/api/v1/chat", json={
            "message": "Hello"
        })

        assert response.status_code == 400
        data = response.json()
        assert "Invalid configuration" in data["detail"]

    def test_chat_value_error(self, client, mock_llmcore):
        """Test chat request with parameter validation error."""
        mock_llmcore.chat.side_effect = ValueError("Invalid parameter 'temperature'")
        app.state.llmcore_instance = mock_llmcore

        response = client.post("/api/v1/chat", json={
            "message": "Hello"
        })

        assert response.status_code == 400
        data = response.json()
        assert "Invalid parameter 'temperature'" in data["detail"]

    def test_chat_generic_llmcore_error(self, client, mock_llmcore):
        """Test chat request with generic LLMCore error."""
        mock_llmcore.chat.side_effect = LLMCoreError("Generic error")
        app.state.llmcore_instance = mock_llmcore

        response = client.post("/api/v1/chat", json={
            "message": "Hello"
        })

        assert response.status_code == 500
        data = response.json()
        assert "Internal LLMCore error: Generic error" in data["detail"]

    def test_chat_unexpected_error(self, client, mock_llmcore):
        """Test chat request with unexpected error."""
        mock_llmcore.chat.side_effect = RuntimeError("Unexpected error")
        app.state.llmcore_instance = mock_llmcore

        response = client.post("/api/v1/chat", json={
            "message": "Hello"
        })

        assert response.status_code == 500
        data = response.json()
        assert "An internal server error occurred" in data["detail"]

    def test_chat_invalid_request_body(self, client, mock_llmcore):
        """Test chat request with invalid JSON body."""
        app.state.llmcore_instance = mock_llmcore

        response = client.post("/api/v1/chat", json={
            "invalid_field": "should fail"
        })

        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "Field required" in str(data)


class TestAPIServerLifecycle:
    """Test FastAPI application lifecycle management."""

    @pytest.mark.asyncio
    async def test_lifespan_startup_success(self):
        """Test successful startup with LLMCore initialization."""
        with patch('llmcore.api_server.main.LLMCore.create') as mock_create:
            mock_instance = AsyncMock()
            mock_instance.get_available_providers.return_value = ["openai"]
            mock_create.return_value = mock_instance

            # Simulate FastAPI lifespan startup
            from llmcore.api_server.main import lifespan

            mock_app = MagicMock()
            mock_app.state = MagicMock()

            async with lifespan(mock_app):
                # During startup
                assert mock_app.state.llmcore_instance == mock_instance
                mock_create.assert_called_once()

            # During shutdown
            mock_instance.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_startup_failure(self):
        """Test startup behavior when LLMCore initialization fails."""
        with patch('llmcore.api_server.main.LLMCore.create') as mock_create:
            mock_create.side_effect = ConfigError("Configuration failed")

            from llmcore.api_server.main import lifespan

            mock_app = MagicMock()
            mock_app.state = MagicMock()

            # Should not raise exception, but set instance to None
            async with lifespan(mock_app):
                assert mock_app.state.llmcore_instance is None

    @pytest.mark.asyncio
    async def test_lifespan_shutdown_cleanup(self):
        """Test proper cleanup during shutdown."""
        mock_instance = AsyncMock()

        from llmcore.api_server.main import lifespan

        mock_app = MagicMock()
        mock_app.state = MagicMock()
        mock_app.state.llmcore_instance = mock_instance

        async with lifespan(mock_app):
            pass  # Just test the shutdown cleanup

        mock_instance.close.assert_called_once()


class TestAPIServerIntegration:
    """Integration tests for the complete API server."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)

    def test_complete_chat_flow(self, client):
        """Test complete chat flow from request to response."""
        with patch('llmcore.api_server.main.LLMCore.create') as mock_create:
            # Setup mock LLMCore instance
            mock_instance = AsyncMock()
            mock_instance.get_available_providers.return_value = ["openai", "anthropic"]
            mock_instance.chat.return_value = "Hello! How can I help you today?"
            mock_create.return_value = mock_instance

            # Start the app (triggers lifespan startup)
            app.state.llmcore_instance = mock_instance

            # Test the complete flow
            # 1. Check service is healthy
            health_response = client.get("/health")
            assert health_response.status_code == 200
            assert health_response.json()["status"] == "healthy"

            # 2. Get service info
            info_response = client.get("/api/v1/info")
            assert info_response.status_code == 200
            info_data = info_response.json()
            assert info_data["service_status"] == "healthy"
            assert "openai" in info_data["features"]["providers"]

            # 3. Make a chat request
            chat_response = client.post("/api/v1/chat", json={
                "message": "Hello, AI!",
                "session_id": "integration-test",
                "provider_name": "openai",
                "save_session": True
            })

            assert chat_response.status_code == 200
            chat_data = chat_response.json()
            assert chat_data["response"] == "Hello! How can I help you today?"
            assert chat_data["session_id"] == "integration-test"

            # Verify the chat method was called correctly
            mock_instance.chat.assert_called_once_with(
                message="Hello, AI!",
                session_id="integration-test",
                provider_name="openai",
                save_session=True,
                stream=False
            )


# Pytest configuration for async tests
pytest_asyncio.main = pytest.main
