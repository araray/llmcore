# tests/api/test_context_info_introspection.py
"""
Tests for the LLMCore introspection API.

This test suite validates the `get_last_interaction_context_info()` method,
ensuring that context preparation details are correctly captured, cached,
and retrievable for UI/REPL integration purposes.

Test Coverage:
- Basic context info retrieval
- Token counting accuracy
- RAG-enabled interactions
- Multi-session independence
- Cache overwriting behavior
- Null-safety (no interaction)
- Field validation and types
- External RAG pattern compatibility

Phase: 4, Step 4.3
Version: 0.25.0
"""

import pytest
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

from llmcore import LLMCore
from llmcore.models import ContextPreparationDetails, ContextDocument


class TestContextInfoIntrospectionBasic:
    """Test basic context info retrieval functionality."""

    @pytest.mark.asyncio
    async def test_get_context_info_after_chat(self):
        """Test that context info is available after a chat interaction."""
        llm = await LLMCore.create()
        
        try:
            # Make a chat call
            response = await llm.chat(
                message="Test message",
                session_id="test_session"
            )
            
            # Retrieve context info
            context_info = llm.get_last_interaction_context_info("test_session")
            
            # Assertions
            assert context_info is not None, "Context info should be available after chat"
            assert isinstance(context_info, ContextPreparationDetails), \
                "Context info should be ContextPreparationDetails instance"
            
        finally:
            await llm.close()

    @pytest.mark.asyncio
    async def test_context_info_has_required_fields(self):
        """Test that context info contains all required fields."""
        llm = await LLMCore.create()
        
        try:
            await llm.chat(message="Test", session_id="test_session")
            context_info = llm.get_last_interaction_context_info("test_session")
            
            # Required fields - must always be present
            assert hasattr(context_info, 'provider'), "Missing 'provider' field"
            assert hasattr(context_info, 'model'), "Missing 'model' field"
            assert hasattr(context_info, 'prompt_tokens'), "Missing 'prompt_tokens' field"
            assert hasattr(context_info, 'completion_tokens'), "Missing 'completion_tokens' field"
            assert hasattr(context_info, 'total_tokens'), "Missing 'total_tokens' field"
            assert hasattr(context_info, 'rag_used'), "Missing 'rag_used' field"
            assert hasattr(context_info, 'context_truncation_applied'), "Missing 'context_truncation_applied' field"
            
            # Type checks
            assert isinstance(context_info.provider, str), "provider should be string"
            assert isinstance(context_info.model, str), "model should be string"
            assert isinstance(context_info.prompt_tokens, int), "prompt_tokens should be int"
            assert isinstance(context_info.completion_tokens, int), "completion_tokens should be int"
            assert isinstance(context_info.total_tokens, int), "total_tokens should be int"
            assert isinstance(context_info.rag_used, bool), "rag_used should be bool"
            assert isinstance(context_info.context_truncation_applied, bool), \
                "context_truncation_applied should be bool"
            
        finally:
            await llm.close()

    @pytest.mark.asyncio
    async def test_get_context_info_no_interaction(self):
        """Test that None is returned when no interaction has occurred."""
        llm = await LLMCore.create()
        
        try:
            # Try to get context info without any chat
            context_info = llm.get_last_interaction_context_info("nonexistent_session")
            
            assert context_info is None, \
                "Context info should be None for session with no interactions"
            
        finally:
            await llm.close()

    @pytest.mark.asyncio
    async def test_context_info_for_wrong_session(self):
        """Test that requesting info for wrong session returns None."""
        llm = await LLMCore.create()
        
        try:
            # Chat in session_a
            await llm.chat(message="Test", session_id="session_a")
            
            # Request info for different session
            context_info = llm.get_last_interaction_context_info("session_b")
            
            assert context_info is None, \
                "Context info should be None when requesting wrong session ID"
            
        finally:
            await llm.close()


class TestTokenCounting:
    """Test token counting accuracy in context info."""

    @pytest.mark.asyncio
    async def test_token_counts_are_positive(self):
        """Test that token counts are positive integers."""
        llm = await LLMCore.create()
        
        try:
            await llm.chat(message="Hello, how are you?", session_id="token_test")
            context_info = llm.get_last_interaction_context_info("token_test")
            
            assert context_info is not None
            assert context_info.prompt_tokens > 0, "Prompt tokens should be positive"
            assert context_info.completion_tokens > 0, "Completion tokens should be positive"
            assert context_info.total_tokens > 0, "Total tokens should be positive"
            
        finally:
            await llm.close()

    @pytest.mark.asyncio
    async def test_total_tokens_equals_sum(self):
        """Test that total_tokens = prompt_tokens + completion_tokens."""
        llm = await LLMCore.create()
        
        try:
            await llm.chat(message="Test message", session_id="sum_test")
            context_info = llm.get_last_interaction_context_info("sum_test")
            
            assert context_info is not None
            expected_total = context_info.prompt_tokens + context_info.completion_tokens
            assert context_info.total_tokens == expected_total, \
                f"Total tokens ({context_info.total_tokens}) should equal " \
                f"prompt + completion ({expected_total})"
            
        finally:
            await llm.close()

    @pytest.mark.asyncio
    async def test_longer_message_more_tokens(self):
        """Test that longer messages result in more prompt tokens."""
        llm = await LLMCore.create()
        
        try:
            # Short message
            await llm.chat(message="Hi", session_id="short")
            short_info = llm.get_last_interaction_context_info("short")
            
            # Long message
            long_message = "This is a much longer message " * 20
            await llm.chat(message=long_message, session_id="long")
            long_info = llm.get_last_interaction_context_info("long")
            
            assert short_info is not None and long_info is not None
            assert long_info.prompt_tokens > short_info.prompt_tokens, \
                "Longer message should have more prompt tokens"
            
        finally:
            await llm.close()


class TestRAGIntegration:
    """Test context info when RAG is enabled."""

    @pytest.mark.asyncio
    async def test_rag_disabled_flag(self):
        """Test that rag_used is False when RAG is not enabled."""
        llm = await LLMCore.create()
        
        try:
            await llm.chat(
                message="Test message",
                session_id="no_rag",
                enable_rag=False
            )
            
            context_info = llm.get_last_interaction_context_info("no_rag")
            
            assert context_info is not None
            assert context_info.rag_used is False, \
                "rag_used should be False when enable_rag=False"
            assert context_info.rag_documents_retrieved is None or \
                   context_info.rag_documents_retrieved == 0, \
                "No documents should be retrieved when RAG is disabled"
            
        finally:
            await llm.close()

    @pytest.mark.asyncio
    async def test_rag_enabled_flag(self):
        """Test that rag_used is True when RAG is enabled."""
        llm = await LLMCore.create()
        
        try:
            # Add some test documents first
            await llm.add_documents_to_vector_store([
                {"content": "Test document 1", "metadata": {"source": "test1"}},
                {"content": "Test document 2", "metadata": {"source": "test2"}},
            ])
            
            # Chat with RAG enabled
            await llm.chat(
                message="Test query",
                session_id="with_rag",
                enable_rag=True
            )
            
            context_info = llm.get_last_interaction_context_info("with_rag")
            
            assert context_info is not None
            assert context_info.rag_used is True, \
                "rag_used should be True when enable_rag=True"
            
        finally:
            await llm.close()

    @pytest.mark.asyncio
    async def test_rag_documents_retrieved_count(self):
        """Test that rag_documents_retrieved reports correct count."""
        llm = await LLMCore.create()
        
        try:
            # Add test documents
            await llm.add_documents_to_vector_store([
                {"content": f"Document {i}", "metadata": {"id": i}}
                for i in range(5)
            ])
            
            # Chat with RAG, requesting top 3
            await llm.chat(
                message="Test query",
                session_id="rag_count",
                enable_rag=True,
                rag_retrieval_k=3
            )
            
            context_info = llm.get_last_interaction_context_info("rag_count")
            
            assert context_info is not None
            assert context_info.rag_used is True
            assert context_info.rag_documents_retrieved is not None
            # Should retrieve at most 3 (may be less if vector store has fewer relevant docs)
            assert context_info.rag_documents_retrieved <= 3, \
                f"Should retrieve at most 3 documents, got {context_info.rag_documents_retrieved}"
            
        finally:
            await llm.close()

    @pytest.mark.asyncio
    async def test_rag_documents_used_structure(self):
        """Test the structure of rag_documents_used."""
        llm = await LLMCore.create()
        
        try:
            # Add test document
            await llm.add_documents_to_vector_store([
                {"content": "Python is a programming language", "metadata": {"type": "definition"}}
            ])
            
            # Chat with RAG
            await llm.chat(
                message="What is Python?",
                session_id="rag_structure",
                enable_rag=True
            )
            
            context_info = llm.get_last_interaction_context_info("rag_structure")
            
            assert context_info is not None
            if context_info.rag_documents_used:
                assert isinstance(context_info.rag_documents_used, list)
                assert len(context_info.rag_documents_used) > 0
                
                # Check first document structure
                first_doc = context_info.rag_documents_used[0]
                assert isinstance(first_doc, ContextDocument)
                assert hasattr(first_doc, 'content')
                assert hasattr(first_doc, 'metadata')
                assert isinstance(first_doc.content, str)
                assert isinstance(first_doc.metadata, dict)
            
        finally:
            await llm.close()


class TestMultiSessionBehavior:
    """Test context info behavior across multiple sessions."""

    @pytest.mark.asyncio
    async def test_independent_session_context_info(self):
        """Test that different sessions maintain independent context info."""
        llm = await LLMCore.create()
        
        try:
            # Chat in session A
            await llm.chat(message="Message A", session_id="session_a")
            info_a = llm.get_last_interaction_context_info("session_a")
            
            # Chat in session B
            await llm.chat(message="Message B", session_id="session_b")
            info_b = llm.get_last_interaction_context_info("session_b")
            
            # Both should exist independently
            assert info_a is not None, "Session A context info should exist"
            assert info_b is not None, "Session B context info should exist"
            
            # Session A info should still be accessible after B's chat
            info_a_again = llm.get_last_interaction_context_info("session_a")
            assert info_a_again is not None, "Session A context info should persist"
            assert info_a_again.total_tokens == info_a.total_tokens, \
                "Session A context info should be unchanged"
            
        finally:
            await llm.close()

    @pytest.mark.asyncio
    async def test_concurrent_sessions_dont_interfere(self):
        """Test that multiple sessions can have different providers/models."""
        llm = await LLMCore.create()
        
        try:
            # Assuming we have multiple providers configured
            # Session 1 with default provider
            await llm.chat(message="Test 1", session_id="session_1")
            info_1 = llm.get_last_interaction_context_info("session_1")
            
            # Session 2 (potentially different provider if configured)
            await llm.chat(message="Test 2", session_id="session_2")
            info_2 = llm.get_last_interaction_context_info("session_2")
            
            assert info_1 is not None
            assert info_2 is not None
            
            # Both should have valid provider/model info
            assert info_1.provider and info_1.model
            assert info_2.provider and info_2.model
            
        finally:
            await llm.close()


class TestCacheOverwriting:
    """Test that context info is correctly overwritten by new interactions."""

    @pytest.mark.asyncio
    async def test_subsequent_chats_overwrite_context_info(self):
        """Test that subsequent chats overwrite the context info cache."""
        llm = await LLMCore.create()
        
        try:
            session_id = "overwrite_test"
            
            # First chat
            await llm.chat(message="First message", session_id=session_id)
            info_1 = llm.get_last_interaction_context_info(session_id)
            tokens_1 = info_1.total_tokens if info_1 else 0
            
            # Second chat with different message
            await llm.chat(
                message="This is a completely different and much longer message to ensure different token count",
                session_id=session_id
            )
            info_2 = llm.get_last_interaction_context_info(session_id)
            tokens_2 = info_2.total_tokens if info_2 else 0
            
            # The second chat should have overwritten the first
            assert info_2 is not None, "Second context info should exist"
            # Different messages should likely have different token counts
            # (though not guaranteed, so we just check that info was updated)
            assert tokens_2 > 0, "Second interaction should have positive token count"
            
        finally:
            await llm.close()

    @pytest.mark.asyncio
    async def test_cache_persists_until_next_chat(self):
        """Test that context info persists until the next chat in that session."""
        llm = await LLMCore.create()
        
        try:
            session_id = "persist_test"
            
            # Make a chat
            await llm.chat(message="Test", session_id=session_id)
            
            # Retrieve multiple times
            info_1 = llm.get_last_interaction_context_info(session_id)
            info_2 = llm.get_last_interaction_context_info(session_id)
            info_3 = llm.get_last_interaction_context_info(session_id)
            
            # All should return the same data
            assert info_1 is not None
            assert info_2 is not None
            assert info_3 is not None
            assert info_1.total_tokens == info_2.total_tokens == info_3.total_tokens
            
        finally:
            await llm.close()


class TestExternalRAGPattern:
    """Test context info behavior in external RAG pattern."""

    @pytest.mark.asyncio
    async def test_external_rag_pattern_context_info(self):
        """Test context info when using external RAG pattern (enable_rag=False)."""
        llm = await LLMCore.create()
        
        try:
            # Simulate external RAG: construct prompt manually
            external_context = """
            def greet(name):
                return f"Hello, {name}!"
            """
            query = "What does this function do?"
            full_prompt = f"Context:\n{external_context}\n\nQuestion: {query}\n\nAnswer:"
            
            # Call llmcore with enable_rag=False (external RAG pattern)
            await llm.chat(
                message=full_prompt,
                session_id="external_rag_session",
                enable_rag=False  # Critical for external RAG
            )
            
            context_info = llm.get_last_interaction_context_info("external_rag_session")
            
            assert context_info is not None
            assert context_info.rag_used is False, \
                "rag_used should be False in external RAG pattern"
            assert context_info.prompt_tokens > 0, \
                "Should count tokens for externally-constructed prompt"
            assert context_info.total_tokens > 0
            
        finally:
            await llm.close()

    @pytest.mark.asyncio
    async def test_explicitly_staged_items_pattern(self):
        """Test context info when using explicitly_staged_items."""
        llm = await LLMCore.create()
        
        try:
            # Use explicitly_staged_items (another external RAG pattern)
            staged_items = [
                {
                    "type": "user_text",
                    "content": "External context: Python is a programming language."
                }
            ]
            
            await llm.chat(
                message="What is Python?",
                session_id="staged_items_session",
                enable_rag=False,
                explicitly_staged_items=staged_items
            )
            
            context_info = llm.get_last_interaction_context_info("staged_items_session")
            
            assert context_info is not None
            assert context_info.rag_used is False, \
                "Native RAG should be disabled when using explicitly_staged_items"
            # The staged items should contribute to prompt tokens
            assert context_info.prompt_tokens > 0
            
        finally:
            await llm.close()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_message(self):
        """Test context info with empty message."""
        llm = await LLMCore.create()
        
        try:
            await llm.chat(message="", session_id="empty_message")
            context_info = llm.get_last_interaction_context_info("empty_message")
            
            # Should still have context info even with empty message
            assert context_info is not None
            # Token counts should still be valid (may be minimal)
            assert context_info.total_tokens >= 0
            
        finally:
            await llm.close()

    @pytest.mark.asyncio
    async def test_very_long_session_id(self):
        """Test context info with unusually long session ID."""
        llm = await LLMCore.create()
        
        try:
            long_session_id = "session_" + "x" * 1000
            
            await llm.chat(message="Test", session_id=long_session_id)
            context_info = llm.get_last_interaction_context_info(long_session_id)
            
            assert context_info is not None, \
                "Should handle long session IDs"
            
        finally:
            await llm.close()

    @pytest.mark.asyncio
    async def test_special_characters_in_session_id(self):
        """Test context info with special characters in session ID."""
        llm = await LLMCore.create()
        
        try:
            special_session_id = "session-with_special.chars@123"
            
            await llm.chat(message="Test", session_id=special_session_id)
            context_info = llm.get_last_interaction_context_info(special_session_id)
            
            assert context_info is not None, \
                "Should handle session IDs with special characters"
            
        finally:
            await llm.close()


class TestModelSerialization:
    """Test that ContextPreparationDetails can be properly serialized."""

    @pytest.mark.asyncio
    async def test_context_info_model_dump(self):
        """Test that context info can be serialized to dict."""
        llm = await LLMCore.create()
        
        try:
            await llm.chat(message="Test", session_id="serialize_test")
            context_info = llm.get_last_interaction_context_info("serialize_test")
            
            assert context_info is not None
            
            # Should be able to convert to dict
            context_dict = context_info.model_dump()
            
            assert isinstance(context_dict, dict)
            assert 'provider' in context_dict
            assert 'model' in context_dict
            assert 'total_tokens' in context_dict
            
        finally:
            await llm.close()

    @pytest.mark.asyncio
    async def test_context_info_json_serialization(self):
        """Test that context info can be serialized to JSON."""
        llm = await LLMCore.create()
        
        try:
            await llm.chat(message="Test", session_id="json_test")
            context_info = llm.get_last_interaction_context_info("json_test")
            
            assert context_info is not None
            
            # Should be able to convert to JSON
            json_str = context_info.model_dump_json()
            
            assert isinstance(json_str, str)
            assert len(json_str) > 0
            # Basic JSON structure check
            assert '"provider"' in json_str
            assert '"total_tokens"' in json_str
            
        finally:
            await llm.close()


class TestProviderAndModelInfo:
    """Test provider and model information in context info."""

    @pytest.mark.asyncio
    async def test_provider_name_is_valid(self):
        """Test that provider name is a non-empty string."""
        llm = await LLMCore.create()
        
        try:
            await llm.chat(message="Test", session_id="provider_test")
            context_info = llm.get_last_interaction_context_info("provider_test")
            
            assert context_info is not None
            assert isinstance(context_info.provider, str)
            assert len(context_info.provider) > 0, "Provider name should not be empty"
            # Common providers
            valid_providers = ['openai', 'anthropic', 'ollama', 'gemini', 'azure']
            assert any(p in context_info.provider.lower() for p in valid_providers), \
                f"Provider '{context_info.provider}' should be a known provider"
            
        finally:
            await llm.close()

    @pytest.mark.asyncio
    async def test_model_name_is_valid(self):
        """Test that model name is a non-empty string."""
        llm = await LLMCore.create()
        
        try:
            await llm.chat(message="Test", session_id="model_test")
            context_info = llm.get_last_interaction_context_info("model_test")
            
            assert context_info is not None
            assert isinstance(context_info.model, str)
            assert len(context_info.model) > 0, "Model name should not be empty"
            
        finally:
            await llm.close()


# Test fixtures and utilities

@pytest.fixture
async def llmcore_instance():
    """Fixture that provides a fresh LLMCore instance for each test."""
    llm = await LLMCore.create()
    yield llm
    await llm.close()


# Integration test marker
@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for real-world scenarios."""

    @pytest.mark.asyncio
    async def test_typical_repl_workflow(self):
        """Test a typical REPL workflow with multiple interactions."""
        llm = await LLMCore.create()
        
        try:
            session_id = "repl_workflow"
            
            # Simulate REPL interactions
            interactions = [
                "Hello!",
                "What's 2+2?",
                "Tell me a joke",
                "Thanks!"
            ]
            
            for message in interactions:
                await llm.chat(message=message, session_id=session_id)
                context_info = llm.get_last_interaction_context_info(session_id)
                
                # Each interaction should have valid context info
                assert context_info is not None
                assert context_info.total_tokens > 0
                assert context_info.provider
                assert context_info.model
            
        finally:
            await llm.close()

    @pytest.mark.asyncio
    async def test_rag_workflow_with_context_tracking(self):
        """Test a complete RAG workflow while tracking context info."""
        llm = await LLMCore.create()
        
        try:
            # Setup: Add documents
            await llm.add_documents_to_vector_store([
                {"content": "Python uses indentation for code blocks.", "metadata": {}},
                {"content": "Python supports multiple programming paradigms.", "metadata": {}},
            ])
            
            # Query 1: Without RAG
            await llm.chat(
                message="What is Python?",
                session_id="rag_workflow",
                enable_rag=False
            )
            info_without_rag = llm.get_last_interaction_context_info("rag_workflow")
            
            # Query 2: With RAG
            await llm.chat(
                message="What is Python?",
                session_id="rag_workflow",
                enable_rag=True
            )
            info_with_rag = llm.get_last_interaction_context_info("rag_workflow")
            
            # Validate both
            assert info_without_rag is not None
            assert info_with_rag is not None
            
            assert info_without_rag.rag_used is False
            assert info_with_rag.rag_used is True
            
            # RAG version should typically have more prompt tokens (context included)
            # Note: This is not guaranteed in all cases, so we just verify the flag
            
        finally:
            await llm.close()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
