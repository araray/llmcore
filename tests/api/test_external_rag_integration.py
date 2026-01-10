# tests/api/test_external_rag_integration.py
"""
Test suite for external RAG engine integration patterns with LLMCore.

This test suite validates that LLMCore's API properly supports external RAG engines
like semantiscan by testing the three main integration patterns:
1. Fully-constructed prompts with enable_rag=False
2. Structured context via explicitly_staged_items
3. Preview context functionality for token estimation

These tests ensure zero breaking changes and full backward compatibility.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

import pytest

from llmcore.api import LLMCore, LLMCoreProtocol
from llmcore.exceptions import LLMCoreError, ProviderError, SessionNotFoundError
from llmcore.models import (
    ChatSession,
    ContextItem,
    ContextItemType,
    ContextPreparationDetails,
    Message,
    Role,
)

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
async def llmcore_instance():
    """Create a test LLMCore instance with minimal configuration."""
    config_overrides = {
        "llmcore": {"default_provider": "ollama", "log_raw_payloads": False},
        "providers": {
            "ollama": {
                "host": "http://localhost:11434",
                "default_model": "llama3:8b",
                "timeout": 30,
            }
        },
        "storage": {
            "session": {"type": "json", "path": ":memory:"},
            "vector": {"type": "chromadb", "path": ":memory:"},
        },
    }

    async with await LLMCore.create(config_overrides=config_overrides) as llm:
        yield llm


@pytest.fixture
def sample_rag_documents() -> List[Dict[str, str]]:
    """Sample documents simulating semantiscan RAG retrieval results."""
    return [
        {
            "content": "LLMCore is a library for interacting with multiple LLM providers through a unified API.",
            "source": "llmcore/README.md",
            "score": 0.95,
        },
        {
            "content": "The chat() method accepts parameters like message, session_id, provider_name, and model_name.",
            "source": "llmcore/docs/api.md",
            "score": 0.88,
        },
        {
            "content": "External RAG engines should pass enable_rag=False to prevent double-RAG scenarios.",
            "source": "llmcore/docs/integration.md",
            "score": 0.82,
        },
    ]


# ==============================================================================
# Pattern 1: Fully-Constructed Prompts
# ==============================================================================


class TestExternalRAGPattern1_ConstructedPrompts:
    """
    Test Pattern 1: External engine constructs full prompt (context + query)
    and passes as 'message' parameter with enable_rag=False.

    This is the simplest integration pattern and the recommended approach
    for semantiscan query pipeline.
    """

    @pytest.mark.asyncio
    async def test_basic_constructed_prompt_with_rag_disabled(self, llmcore_instance: LLMCore):
        """
        Test that LLMCore accepts a fully-constructed prompt and doesn't
        attempt internal RAG when enable_rag=False.
        """
        # Simulate semantiscan building a prompt
        query = "What is LLMCore?"
        context = "LLMCore is a library for LLM interactions."
        full_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        # Call chat with constructed prompt and RAG disabled
        response = await llmcore_instance.chat(
            message=full_prompt,
            enable_rag=False,  # Critical: prevent double-RAG
            stream=False,
        )

        # Verify response is generated
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_constructed_prompt_with_session(self, llmcore_instance: LLMCore):
        """
        Test constructed prompt pattern with session management for
        maintaining conversation history across multiple queries.
        """
        session_id = f"test_session_{uuid.uuid4()}"

        # First query with context
        query1 = "What is LLMCore?"
        context1 = "LLMCore provides a unified API for multiple LLM providers."
        full_prompt1 = f"Context:\n{context1}\n\nQuestion: {query1}\n\nAnswer:"

        response1 = await llmcore_instance.chat(
            message=full_prompt1, session_id=session_id, enable_rag=False, save_session=True
        )

        assert response1 is not None

        # Follow-up query (should have access to history)
        query2 = "How do I use it?"
        context2 = "Use the chat() method to send messages to LLMs."
        full_prompt2 = f"Context:\n{context2}\n\nQuestion: {query2}\n\nAnswer:"

        response2 = await llmcore_instance.chat(
            message=full_prompt2, session_id=session_id, enable_rag=False, save_session=True
        )

        assert response2 is not None

        # Verify session contains both interactions
        session = await llmcore_instance.get_session(session_id)
        assert len(session.messages) >= 4  # 2 user + 2 assistant messages

    @pytest.mark.asyncio
    async def test_constructed_prompt_with_provider_override(self, llmcore_instance: LLMCore):
        """
        Test that external RAG engines can override provider and model
        while using constructed prompts.
        """
        query = "Test query"
        context = "Test context"
        full_prompt = f"Context:\n{context}\n\nQuestion: {query}"

        # This should work even if we specify a different provider
        # (will use default if specified provider not available)
        response = await llmcore_instance.chat(
            message=full_prompt,
            provider_name="ollama",  # Explicit provider
            model_name="llama3:8b",  # Explicit model
            enable_rag=False,
        )

        assert response is not None

    @pytest.mark.asyncio
    async def test_constructed_prompt_with_streaming(self, llmcore_instance: LLMCore):
        """
        Test streaming responses with constructed prompts.
        External RAG engines should be able to stream LLM responses.
        """
        query = "Tell me about LLMCore"
        context = "LLMCore is an LLM library."
        full_prompt = f"Context:\n{context}\n\nQuestion: {query}"

        # Get streaming response
        response_stream = await llmcore_instance.chat(
            message=full_prompt, enable_rag=False, stream=True
        )

        # Collect chunks
        chunks = []
        async for chunk in response_stream:
            assert isinstance(chunk, str)
            chunks.append(chunk)

        # Verify we got streamed content
        assert len(chunks) > 0
        full_response = "".join(chunks)
        assert len(full_response) > 0


# ==============================================================================
# Pattern 2: Structured Context via explicitly_staged_items
# ==============================================================================


class TestExternalRAGPattern2_StagedItems:
    """
    Test Pattern 2: External engine passes retrieved documents as
    structured ContextItem objects via explicitly_staged_items parameter.

    This pattern provides more structure and allows LLMCore to manage
    token counting and truncation of the external context.
    """

    @pytest.mark.asyncio
    async def test_basic_explicitly_staged_items(
        self, llmcore_instance: LLMCore, sample_rag_documents: List[Dict[str, str]]
    ):
        """
        Test passing external RAG context as explicitly_staged_items.
        """
        # Convert sample documents to ContextItems
        context_items = [
            ContextItem(
                id=f"rag_doc_{i}",
                type=ContextItemType.RAG_SNIPPET,
                content=doc["content"],
                source_id=doc["source"],
                metadata={"score": doc["score"], "external_rag": True},
            )
            for i, doc in enumerate(sample_rag_documents)
        ]

        query = "What is LLMCore and how do I use it?"

        # Pass structured context via explicitly_staged_items
        response = await llmcore_instance.chat(
            message=query,
            explicitly_staged_items=context_items,
            enable_rag=False,  # Don't do internal RAG
            stream=False,
        )

        assert response is not None
        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_staged_items_with_session(
        self, llmcore_instance: LLMCore, sample_rag_documents: List[Dict[str, str]]
    ):
        """
        Test explicitly_staged_items with session management.
        Verify that staged items are included but not permanently added to session.
        """
        session_id = f"test_staged_{uuid.uuid4()}"

        context_items = [
            ContextItem(id=f"rag_doc_{i}", type=ContextItemType.USER_TEXT, content=doc["content"])
            for i, doc in enumerate(sample_rag_documents[:2])  # Use first 2 docs
        ]

        response = await llmcore_instance.chat(
            message="Question about the context above",
            session_id=session_id,
            explicitly_staged_items=context_items,
            enable_rag=False,
            save_session=True,
        )

        assert response is not None

        # Verify session doesn't contain the staged items permanently
        session = await llmcore_instance.get_session(session_id)
        # Session should have user query and assistant response,
        # but staged items shouldn't be in messages list
        assert all(msg.role in [Role.USER, Role.ASSISTANT] for msg in session.messages)

    @pytest.mark.asyncio
    async def test_staged_items_mixed_types(self, llmcore_instance: LLMCore):
        """
        Test mixing different types of staged items (messages + context items).
        """
        # Create mixed staged items
        staged_items = [
            # A previous message to include
            Message(id=str(uuid.uuid4()), role=Role.USER, content="Previous important question"),
            # External RAG context
            ContextItem(
                id="external_context_1",
                type=ContextItemType.RAG_SNIPPET,
                content="Relevant documentation snippet",
            ),
            # User-provided text
            ContextItem(
                id="user_note_1",
                type=ContextItemType.USER_TEXT,
                content="Important note: Always use async methods",
            ),
        ]

        response = await llmcore_instance.chat(
            message="Answer based on all the context above",
            explicitly_staged_items=staged_items,
            enable_rag=False,
        )

        assert response is not None

    @pytest.mark.asyncio
    async def test_staged_items_token_management(self, llmcore_instance: LLMCore):
        """
        Test that LLMCore properly counts tokens for staged items
        and can truncate if necessary.
        """
        # Create many large context items
        large_context_items = [
            ContextItem(
                id=f"large_doc_{i}",
                type=ContextItemType.USER_TEXT,
                content="Lorem ipsum " * 500,  # Large content
            )
            for i in range(10)
        ]

        # This should not fail even with large context
        # LLMCore should handle truncation based on configuration
        response = await llmcore_instance.chat(
            message="Summarize the key points",
            explicitly_staged_items=large_context_items,
            enable_rag=False,
        )

        assert response is not None


# ==============================================================================
# Pattern 3: Context Preview for Token Estimation
# ==============================================================================


class TestExternalRAGPattern3_ContextPreview:
    """
    Test Pattern 3: preview_context_for_chat() for token estimation
    and context debugging without making actual LLM API calls.

    This is essential for external RAG engines to validate prompts
    and estimate costs before generation.
    """

    @pytest.mark.asyncio
    async def test_basic_context_preview(self, llmcore_instance: LLMCore):
        """
        Test basic context preview functionality.
        """
        query = "What is LLMCore?"

        preview = await llmcore_instance.preview_context_for_chat(
            current_user_query=query, enable_rag=False
        )

        # Verify preview structure
        assert isinstance(preview, dict)
        assert "prepared_messages" in preview
        assert "final_token_count" in preview
        assert "max_tokens_for_model" in preview

        # Verify messages are prepared
        assert len(preview["prepared_messages"]) > 0
        assert preview["final_token_count"] > 0
        assert preview["max_tokens_for_model"] > 0

    @pytest.mark.asyncio
    async def test_preview_with_staged_items(
        self, llmcore_instance: LLMCore, sample_rag_documents: List[Dict[str, str]]
    ):
        """
        Test context preview with explicitly_staged_items.
        This is how semantiscan can estimate tokens before calling chat().
        """
        context_items = [
            ContextItem(
                id=f"preview_doc_{i}", type=ContextItemType.RAG_SNIPPET, content=doc["content"]
            )
            for i, doc in enumerate(sample_rag_documents)
        ]

        query = "Based on the context, explain LLMCore"

        preview = await llmcore_instance.preview_context_for_chat(
            current_user_query=query, explicitly_staged_items=context_items, enable_rag=False
        )

        # Verify token counts include staged items
        assert preview["final_token_count"] > 0

        # Count of messages should include:
        # - System message (if any)
        # - Staged context items (converted to messages)
        # - User query
        assert len(preview["prepared_messages"]) >= len(context_items) + 1

    @pytest.mark.asyncio
    async def test_preview_with_session_history(self, llmcore_instance: LLMCore):
        """
        Test preview includes session history in token estimates.
        """
        session_id = f"test_preview_{uuid.uuid4()}"

        # Create session with some history
        await llmcore_instance.chat(
            message="First message", session_id=session_id, enable_rag=False, save_session=True
        )

        # Preview next message
        preview = await llmcore_instance.preview_context_for_chat(
            current_user_query="Second message", session_id=session_id, enable_rag=False
        )

        # Should include history in token count
        assert preview["final_token_count"] > 0
        # Should have at least 3 messages: user1, assistant1, user2(preview)
        assert len(preview["prepared_messages"]) >= 3

    @pytest.mark.asyncio
    async def test_preview_doesnt_modify_session(self, llmcore_instance: LLMCore):
        """
        Critical test: Verify preview doesn't modify the actual session.
        """
        session_id = f"test_preview_immutable_{uuid.uuid4()}"

        # Create session
        await llmcore_instance.chat(
            message="Initial message", session_id=session_id, enable_rag=False, save_session=True
        )

        session_before = await llmcore_instance.get_session(session_id)
        message_count_before = len(session_before.messages)

        # Preview a message
        await llmcore_instance.preview_context_for_chat(
            current_user_query="Preview message - should not be saved",
            session_id=session_id,
            enable_rag=False,
        )

        # Verify session unchanged
        session_after = await llmcore_instance.get_session(session_id)
        assert len(session_after.messages) == message_count_before

    @pytest.mark.asyncio
    async def test_preview_with_prompt_template_values(self, llmcore_instance: LLMCore):
        """
        Test preview with custom prompt template values.
        """
        preview = await llmcore_instance.preview_context_for_chat(
            current_user_query="What is {project_name}?",
            prompt_template_values={"project_name": "LLMCore", "version": "0.24.0"},
            enable_rag=False,
        )

        assert preview is not None
        assert "prepared_messages" in preview


# ==============================================================================
# Protocol Compliance Tests
# ==============================================================================


class TestLLMCoreProtocolCompliance:
    """
    Test that LLMCore properly implements the LLMCoreProtocol interface
    for type-safe integration with external RAG engines.
    """

    @pytest.mark.asyncio
    async def test_llmcore_implements_protocol(self, llmcore_instance: LLMCore):
        """
        Test that LLMCore instance is recognized as implementing LLMCoreProtocol.
        """
        assert isinstance(llmcore_instance, LLMCoreProtocol)

    @pytest.mark.asyncio
    async def test_protocol_method_signature_compliance(self):
        """
        Test that LLMCore.chat() signature matches LLMCoreProtocol.chat().
        This ensures external code using the protocol will work with LLMCore.
        """
        import inspect

        # Get signatures
        llmcore_chat_sig = inspect.signature(LLMCore.chat)
        protocol_chat_sig = inspect.signature(LLMCoreProtocol.chat)

        # Extract parameter names (excluding 'self')
        llmcore_params = [p for p in llmcore_chat_sig.parameters.keys() if p != "self"]
        protocol_params = [p for p in protocol_chat_sig.parameters.keys() if p != "self"]

        # Verify all protocol parameters exist in LLMCore
        for param in protocol_params:
            assert param in llmcore_params, (
                f"Parameter '{param}' from protocol not in LLMCore.chat()"
            )


# ==============================================================================
# Integration Tests with Semantiscan Patterns
# ==============================================================================


class TestSemantiscanIntegrationPatterns:
    """
    High-level integration tests simulating actual semantiscan usage patterns.
    """

    @pytest.mark.asyncio
    async def test_semantiscan_query_pipeline_simulation(
        self, llmcore_instance: LLMCore, sample_rag_documents: List[Dict[str, str]]
    ):
        """
        Simulate complete semantiscan query pipeline:
        1. Retrieve documents (mocked)
        2. Construct prompt
        3. Call LLMCore
        4. Return response
        """
        # Step 1: Retrieve (simulated - would be semantiscan.retrieve())
        retrieved_docs = sample_rag_documents

        # Step 2: Construct prompt (semantiscan pattern)
        query = "How do I use LLMCore?"
        context_str = "\n\n".join(
            [f"Source: {doc['source']}\n{doc['content']}" for doc in retrieved_docs]
        )
        full_prompt = f"""Based on the following context, answer the question.

Context:
{context_str}

Question: {query}

Answer:"""

        # Step 3: Call LLMCore as semantiscan would
        response = await llmcore_instance.chat(
            message=full_prompt,
            enable_rag=False,  # Critical: semantiscan handles RAG
            stream=False,
        )

        # Step 4: Verify response
        assert response is not None
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_semantiscan_with_conversation_history(
        self, llmcore_instance: LLMCore, sample_rag_documents: List[Dict[str, str]]
    ):
        """
        Simulate semantiscan maintaining conversation history across queries.
        """
        session_id = f"semantiscan_session_{uuid.uuid4()}"

        # First query
        query1 = "What is LLMCore?"
        context1 = sample_rag_documents[0]["content"]
        prompt1 = f"Context: {context1}\n\nQuestion: {query1}"

        response1 = await llmcore_instance.chat(
            message=prompt1, session_id=session_id, enable_rag=False, save_session=True
        )

        # Follow-up query with different context
        query2 = "How do I use the chat method?"
        context2 = sample_rag_documents[1]["content"]
        prompt2 = f"Context: {context2}\n\nQuestion: {query2}"

        response2 = await llmcore_instance.chat(
            message=prompt2, session_id=session_id, enable_rag=False, save_session=True
        )

        # Verify both responses received
        assert response1 is not None
        assert response2 is not None

        # Verify conversation history maintained
        session = await llmcore_instance.get_session(session_id)
        assert len(session.messages) >= 4


# ==============================================================================
# Backward Compatibility Tests
# ==============================================================================


class TestBackwardCompatibility:
    """
    Ensure new parameters don't break existing usage patterns.
    """

    @pytest.mark.asyncio
    async def test_chat_without_new_parameters(self, llmcore_instance: LLMCore):
        """
        Test that chat() still works without any of the new parameters.
        Ensures backward compatibility for existing code.
        """
        # Simple chat call without new parameters
        response = await llmcore_instance.chat(message="Hello!", stream=False)

        assert response is not None

    @pytest.mark.asyncio
    async def test_chat_with_internal_rag_still_works(self, llmcore_instance: LLMCore):
        """
        Test that internal RAG (enable_rag=True) still functions.
        """
        # This test verifies the original RAG functionality is preserved
        # Even though we're adding external RAG support
        response = await llmcore_instance.chat(
            message="Test query",
            enable_rag=True,  # Original internal RAG
            rag_retrieval_k=3,
            stream=False,
        )

        assert response is not None

    @pytest.mark.asyncio
    async def test_all_original_parameters_work(self, llmcore_instance: LLMCore):
        """
        Test all pre-existing parameters still function correctly.
        """
        session_id = f"compat_test_{uuid.uuid4()}"

        response = await llmcore_instance.chat(
            message="Test message",
            session_id=session_id,
            system_message="You are a helpful assistant",
            provider_name="ollama",
            stream=False,
            save_session=True,
            temperature=0.7,  # provider_kwarg
            max_tokens=100,  # provider_kwarg
        )

        assert response is not None


# ==============================================================================
# Error Handling Tests
# ==============================================================================


class TestErrorHandling:
    """
    Test proper error handling with new parameters.
    """

    @pytest.mark.asyncio
    async def test_invalid_staged_item_type(self, llmcore_instance: LLMCore):
        """
        Test handling of invalid items in explicitly_staged_items.
        """
        # Try to pass invalid type
        with pytest.raises((TypeError, ValueError, LLMCoreError)):
            await llmcore_instance.chat(
                message="Test",
                explicitly_staged_items=["not_a_valid_item"],  # type: ignore
                enable_rag=False,
            )

    @pytest.mark.asyncio
    async def test_preview_with_invalid_session(self, llmcore_instance: LLMCore):
        """
        Test preview with non-existent session ID.
        """
        # Should handle gracefully or raise appropriate error
        try:
            preview = await llmcore_instance.preview_context_for_chat(
                current_user_query="Test", session_id="non_existent_session_12345", enable_rag=False
            )
            # If it succeeds, it should create a temp session
            assert preview is not None
        except Exception as e:
            # If it fails, should be a clear error
            assert isinstance(e, (SessionNotFoundError, LLMCoreError))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
