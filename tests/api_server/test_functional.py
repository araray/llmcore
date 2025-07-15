# tests/api_server/test_functional.py
"""
Functional tests for the llmcore API server.

These tests verify end-to-end functionality and realistic usage scenarios
of the API server, including integration between components.
"""

import pytest
import asyncio
import json
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from llmcore.api_server.main import app
from llmcore.exceptions import ProviderError, ContextLengthError


@pytest.mark.integration
class TestAPIServerFunctional:
    """Functional tests for complete API server workflows."""

    @pytest.fixture
    def client(self):
        """Create a test client for functional tests."""
        return TestClient(app)

    def test_full_chat_session_workflow(self, client):
        """Test a complete chat session workflow with multiple interactions."""
        # Mock LLMCore for the entire workflow
        mock_llmcore = AsyncMock()
        mock_llmcore.get_available_providers.return_value = ["openai", "anthropic"]

        # Define responses for different messages
        def chat_side_effect(**kwargs):
            message = kwargs.get('message', '')
            if 'hello' in message.lower():
                return "Hello! How can I help you today?"
            elif 'name' in message.lower():
                return "I'm Claude, an AI assistant created by Anthropic."
            elif 'python' in message.lower():
                return "Python is a versatile programming language..."
            else:
                return "I understand. How else can I assist you?"

        mock_llmcore.chat.side_effect = chat_side_effect
        app.state.llmcore_instance = mock_llmcore

        session_id = "functional-test-session"

        # Step 1: Initial greeting
        response1 = client.post("/api/v1/chat", json={
            "message": "Hello there!",
            "session_id": session_id,
            "save_session": True
        })

        assert response1.status_code == 200
        data1 = response1.json()
        assert "Hello" in data1["response"]
        assert data1["session_id"] == session_id

        # Step 2: Follow-up question (should remember context)
        response2 = client.post("/api/v1/chat", json={
            "message": "What's your name?",
            "session_id": session_id
        })

        assert response2.status_code == 200
        data2 = response2.json()
        assert "Claude" in data2["response"]

        # Step 3: Technical question
        response3 = client.post("/api/v1/chat", json={
            "message": "Tell me about Python programming",
            "session_id": session_id,
            "provider_name": "anthropic",  # Switch provider
            "model_name": "claude-3-sonnet-20240229"
        })

        assert response3.status_code == 200
        data3 = response3.json()
        assert "Python" in data3["response"]

        # Verify all calls were made with the correct session_id
        assert mock_llmcore.chat.call_count == 3
        for call in mock_llmcore.chat.call_args_list:
            assert call.kwargs.get('session_id') == session_id

    def test_streaming_chat_functional(self, client):
        """Test functional streaming chat with realistic data flow."""
        # Create a mock streaming response
        async def mock_stream():
            story_parts = [
                "Once", " upon", " a", " time", ",", " in", " a", " land",
                " far", " away", ",", " there", " lived", " a", " wise",
                " old", " wizard", " who", " knew", " the", " secrets",
                " of", " the", " digital", " realm", "."
            ]
            for part in story_parts:
                yield part
                await asyncio.sleep(0.01)  # Simulate realistic streaming delay

        mock_llmcore = AsyncMock()
        mock_llmcore.get_available_providers.return_value = ["openai"]
        mock_llmcore.chat.return_value = mock_stream()
        app.state.llmcore_instance = mock_llmcore

        response = client.post("/api/v1/chat", json={
            "message": "Tell me a short fantasy story",
            "stream": True,
            "provider_kwargs": {
                "temperature": 0.8,
                "max_tokens": 200
            }
        })

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        # Verify the streaming content contains expected elements
        content = response.text
        assert "Once upon a time" in content
        assert "wizard" in content
        assert "digital realm" in content

        # Verify the mock was called with streaming enabled and provider_kwargs
        mock_llmcore.chat.assert_called_once()
        call_kwargs = mock_llmcore.chat.call_args.kwargs
        assert call_kwargs["stream"] is True
        assert call_kwargs["temperature"] == 0.8
        assert call_kwargs["max_tokens"] == 200

    def test_provider_switching_functional(self, client):
        """Test switching between different providers within the same session."""
        mock_llmcore = AsyncMock()
        mock_llmcore.get_available_providers.return_value = ["openai", "anthropic", "ollama"]

        # Different responses for different providers
        def provider_specific_response(**kwargs):
            provider = kwargs.get('provider_name', 'default')
            return f"Response from {provider} provider"

        mock_llmcore.chat.side_effect = provider_specific_response
        app.state.llmcore_instance = mock_llmcore

        session_id = "provider-switching-session"

        # Use OpenAI
        response1 = client.post("/api/v1/chat", json={
            "message": "Hello",
            "session_id": session_id,
            "provider_name": "openai",
            "model_name": "gpt-4o"
        })

        assert response1.status_code == 200
        assert "openai" in response1.json()["response"]

        # Switch to Anthropic
        response2 = client.post("/api/v1/chat", json={
            "message": "Continue our conversation",
            "session_id": session_id,
            "provider_name": "anthropic",
            "model_name": "claude-3-opus-20240229"
        })

        assert response2.status_code == 200
        assert "anthropic" in response2.json()["response"]

        # Switch to Ollama
        response3 = client.post("/api/v1/chat", json={
            "message": "And now with a local model",
            "session_id": session_id,
            "provider_name": "ollama",
            "model_name": "llama3.1"
        })

        assert response3.status_code == 200
        assert "ollama" in response3.json()["response"]

        # Verify all calls maintained the same session but used different providers
        calls = mock_llmcore.chat.call_args_list
        assert len(calls) == 3
        assert calls[0].kwargs["provider_name"] == "openai"
        assert calls[1].kwargs["provider_name"] == "anthropic"
        assert calls[2].kwargs["provider_name"] == "ollama"

        for call in calls:
            assert call.kwargs["session_id"] == session_id

    def test_error_recovery_functional(self, client):
        """Test error handling and recovery in realistic scenarios."""
        mock_llmcore = AsyncMock()
        mock_llmcore.get_available_providers.return_value = ["openai", "anthropic"]

        # First call fails with provider error, second succeeds
        call_count = 0
        def chat_with_failure(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ProviderError("openai", "Rate limit exceeded")
            else:
                return "This request succeeded after the error"

        mock_llmcore.chat.side_effect = chat_with_failure
        app.state.llmcore_instance = mock_llmcore

        session_id = "error-recovery-session"

        # First request fails
        response1 = client.post("/api/v1/chat", json={
            "message": "This will fail",
            "session_id": session_id,
            "provider_name": "openai"
        })

        assert response1.status_code == 400
        error_data = response1.json()
        assert "Rate limit exceeded" in error_data["detail"]
        assert "openai" in error_data["detail"]

        # Second request succeeds (simulating retry or provider switch)
        response2 = client.post("/api/v1/chat", json={
            "message": "This should work",
            "session_id": session_id,
            "provider_name": "anthropic"  # Switch provider
        })

        assert response2.status_code == 200
        success_data = response2.json()
        assert "succeeded after the error" in success_data["response"]
        assert success_data["session_id"] == session_id

        # Verify both calls were attempted
        assert mock_llmcore.chat.call_count == 2

    def test_system_message_functional(self, client):
        """Test functional behavior with system messages."""
        mock_llmcore = AsyncMock()
        mock_llmcore.get_available_providers.return_value = ["openai"]

        # Mock response that reflects the system message influence
        def system_aware_response(**kwargs):
            system_msg = kwargs.get('system_message', '')
            message = kwargs.get('message', '')

            if 'professional' in system_msg.lower():
                return f"Good day. Regarding your inquiry about {message.lower()}, I shall provide a formal response."
            else:
                return f"Hey! About {message.lower()}, here's what I think..."

        mock_llmcore.chat.side_effect = system_aware_response
        app.state.llmcore_instance = mock_llmcore

        # Test with professional system message
        response1 = client.post("/api/v1/chat", json={
            "message": "Python programming",
            "system_message": "You are a professional technical consultant. Always use formal language.",
            "session_id": "professional-session"
        })

        assert response1.status_code == 200
        data1 = response1.json()
        assert "Good day" in data1["response"]
        assert "formal response" in data1["response"]

        # Test with casual system message
        response2 = client.post("/api/v1/chat", json={
            "message": "JavaScript frameworks",
            "system_message": "You are a friendly coding buddy. Keep it casual and fun!",
            "session_id": "casual-session"
        })

        assert response2.status_code == 200
        data2 = response2.json()
        assert "Hey!" in data2["response"]
        assert "here's what I think" in data2["response"]

    def test_concurrent_requests_functional(self, client):
        """Test handling multiple concurrent requests."""
        import threading
        import time

        mock_llmcore = AsyncMock()
        mock_llmcore.get_available_providers.return_value = ["openai"]

        # Add artificial delay to simulate real processing
        async def delayed_response(**kwargs):
            await asyncio.sleep(0.1)  # 100ms delay
            message = kwargs.get('message', '')
            session_id = kwargs.get('session_id', 'unknown')
            return f"Processed '{message}' for session {session_id}"

        mock_llmcore.chat.side_effect = delayed_response
        app.state.llmcore_instance = mock_llmcore

        results = []
        errors = []

        def make_request(session_num):
            try:
                response = client.post("/api/v1/chat", json={
                    "message": f"Request from session {session_num}",
                    "session_id": f"concurrent-session-{session_num}"
                })
                results.append((session_num, response.status_code, response.json()))
            except Exception as e:
                errors.append((session_num, str(e)))

        # Create multiple concurrent requests
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all requests to complete
        for thread in threads:
            thread.join(timeout=5)  # 5 second timeout

        # Verify all requests succeeded
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5

        for session_num, status_code, data in results:
            assert status_code == 200
            assert f"session {session_num}" in data["response"]
            assert data["session_id"] == f"concurrent-session-{session_num}"

    def test_large_context_handling_functional(self, client):
        """Test handling of large context scenarios."""
        mock_llmcore = AsyncMock()
        mock_llmcore.get_available_providers.return_value = ["openai"]

        # First attempt fails with context length error
        # Second attempt succeeds (simulating context truncation)
        call_count = 0
        def context_length_simulation(**kwargs):
            nonlocal call_count
            call_count += 1

            message = kwargs.get('message', '')

            if call_count == 1 and len(message) > 100:
                raise ContextLengthError(
                    model_name="gpt-4o",
                    limit=8000,
                    actual=9500,
                    message="Context length exceeded"
                )
            else:
                return f"Successfully processed message of {len(message)} characters"

        mock_llmcore.chat.side_effect = context_length_simulation
        app.state.llmcore_instance = mock_llmcore

        # Large message that will trigger context length error
        large_message = "This is a very long message. " * 50  # ~1500 characters

        response1 = client.post("/api/v1/chat", json={
            "message": large_message,
            "session_id": "large-context-session",
            "model_name": "gpt-4o"
        })

        assert response1.status_code == 400
        error_data = response1.json()
        assert "Context length exceeded" in error_data["detail"]
        assert "gpt-4o" in error_data["detail"]
        assert "8000" in error_data["detail"]  # limit
        assert "9500" in error_data["detail"]  # actual

        # Smaller message should succeed
        small_message = "This is a short message."

        response2 = client.post("/api/v1/chat", json={
            "message": small_message,
            "session_id": "large-context-session",
            "model_name": "gpt-4o"
        })

        assert response2.status_code == 200
        success_data = response2.json()
        assert "Successfully processed" in success_data["response"]
        assert str(len(small_message)) in success_data["response"]


@pytest.mark.integration
class TestAPIServerPerformance:
    """Performance and load testing for the API server."""

    @pytest.fixture
    def client(self):
        """Create a test client for performance tests."""
        return TestClient(app)

    @pytest.mark.slow
    def test_response_time_performance(self, client):
        """Test API response times are within acceptable limits."""
        import time

        mock_llmcore = AsyncMock()
        mock_llmcore.get_available_providers.return_value = ["openai"]
        mock_llmcore.chat.return_value = "Quick response"
        app.state.llmcore_instance = mock_llmcore

        # Test non-streaming response time
        start_time = time.time()
        response = client.post("/api/v1/chat", json={
            "message": "Hello",
            "stream": False
        })
        end_time = time.time()

        assert response.status_code == 200
        response_time = end_time - start_time
        assert response_time < 1.0, f"Response time {response_time:.3f}s exceeded 1 second"

        # Test info endpoint response time
        start_time = time.time()
        info_response = client.get("/api/v1/info")
        end_time = time.time()

        assert info_response.status_code == 200
        info_response_time = end_time - start_time
        assert info_response_time < 0.1, f"Info response time {info_response_time:.3f}s exceeded 100ms"

    @pytest.mark.slow
    def test_memory_usage_stability(self, client):
        """Test that repeated requests don't cause memory leaks."""
        import gc
        import sys

        mock_llmcore = AsyncMock()
        mock_llmcore.get_available_providers.return_value = ["openai"]
        mock_llmcore.chat.return_value = "Memory test response"
        app.state.llmcore_instance = mock_llmcore

        # Get baseline memory usage
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Make multiple requests
        for i in range(50):
            response = client.post("/api/v1/chat", json={
                "message": f"Memory test message {i}",
                "session_id": f"memory-test-{i % 5}"  # Reuse some sessions
            })
            assert response.status_code == 200

        # Check memory usage hasn't grown significantly
        gc.collect()
        final_objects = len(gc.get_objects())

        # Allow for some growth, but not excessive
        growth_ratio = final_objects / initial_objects
        assert growth_ratio < 1.5, f"Object count grew by {growth_ratio:.2f}x, indicating possible memory leak"


@pytest.mark.api
class TestAPIServerValidation:
    """Test API input validation and edge cases."""

    @pytest.fixture
    def client(self):
        """Create a test client for validation tests."""
        return TestClient(app)

    def test_request_validation_edge_cases(self, client):
        """Test edge cases in request validation."""
        mock_llmcore = AsyncMock()
        mock_llmcore.get_available_providers.return_value = ["openai"]
        mock_llmcore.chat.return_value = "Valid response"
        app.state.llmcore_instance = mock_llmcore

        # Test empty message
        response = client.post("/api/v1/chat", json={
            "message": ""
        })
        assert response.status_code == 200  # Empty string is technically valid

        # Test very long message
        long_message = "A" * 10000
        response = client.post("/api/v1/chat", json={
            "message": long_message
        })
        assert response.status_code == 200

        # Test special characters in message
        special_message = "Hello! @#$%^&*()_+ 中文 emoji: 🚀🎉"
        response = client.post("/api/v1/chat", json={
            "message": special_message
        })
        assert response.status_code == 200

        # Test very long session_id
        response = client.post("/api/v1/chat", json={
            "message": "Hello",
            "session_id": "x" * 1000
        })
        assert response.status_code == 200

        # Test special characters in session_id
        response = client.post("/api/v1/chat", json={
            "message": "Hello",
            "session_id": "session-with-special-chars_123!@#"
        })
        assert response.status_code == 200

    def test_provider_kwargs_validation(self, client):
        """Test validation of provider-specific parameters."""
        mock_llmcore = AsyncMock()
        mock_llmcore.get_available_providers.return_value = ["openai"]
        app.state.llmcore_instance = mock_llmcore

        # Valid provider_kwargs should work
        mock_llmcore.chat.return_value = "Success with valid params"
        response = client.post("/api/v1/chat", json={
            "message": "Hello",
            "provider_kwargs": {
                "temperature": 0.7,
                "max_tokens": 100,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
        })
        assert response.status_code == 200

        # Invalid parameter should cause LLMCore to raise ValueError
        mock_llmcore.chat.side_effect = ValueError("Unsupported parameter 'invalid_param'")
        response = client.post("/api/v1/chat", json={
            "message": "Hello",
            "provider_kwargs": {
                "invalid_param": "should_fail"
            }
        })
        assert response.status_code == 400
        assert "invalid_param" in response.json()["detail"]

    def test_malformed_json_handling(self, client):
        """Test handling of malformed JSON requests."""
        # This test uses the raw requests approach since TestClient
        # automatically handles JSON serialization
        import requests
        import json as json_module

        mock_llmcore = AsyncMock()
        app.state.llmcore_instance = mock_llmcore

        # Test with TestClient for comparison
        valid_response = client.post("/api/v1/chat", json={"message": "Hello"})
        assert valid_response.status_code in [200, 503]  # Either works or service unavailable

        # Test missing required field
        response = client.post("/api/v1/chat", json={})
        assert response.status_code == 422  # Validation error

        # Test wrong data types
        response = client.post("/api/v1/chat", json={
            "message": 123,  # Should be string
            "stream": "yes"  # Should be boolean
        })
        assert response.status_code == 422


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
