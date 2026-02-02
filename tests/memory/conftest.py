"""
Shared pytest fixtures for memory module tests.

This conftest provides mocked managers, sessions, and test data
for all memory module tests.
"""

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from llmcore.embedding.manager import EmbeddingManager
from llmcore.models import (
    ChatSession,
    ContextDocument,
    ContextItem,
    ContextItemType,
    Message,
    Role,
)
from llmcore.providers.manager import ProviderManager
from llmcore.storage.manager import StorageManager

# ============================================================================
# CONFIG FIXTURES
# ============================================================================


@pytest.fixture
def mock_context_management_config() -> Dict[str, Any]:
    """Provides a valid context management configuration."""
    return {
        "inclusion_priority": "system_history,explicitly_staged,user_items_active,history_chat,final_user_query",
        "truncation_priority": "history_chat,user_items_active,rag_in_query,explicitly_staged",
        "reserved_response_tokens": 500,
        "default_prompt_template": "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        "prompt_template_path": "",
    }


@pytest.fixture
def mock_full_config(mock_context_management_config: Dict[str, Any]) -> Dict[str, Any]:
    """Provides a complete LLMCore configuration."""
    return {
        "context_management": mock_context_management_config,
        "embedding": {
            "model": "openai",
            "cache_embeddings": True,
        },
        "providers": {
            "default": "openai",
        },
    }


# ============================================================================
# MOCK MANAGERS
# ============================================================================


@pytest.fixture
def mock_provider_manager() -> MagicMock:
    """Provides a mocked ProviderManager."""
    manager = MagicMock(spec=ProviderManager)
    manager.get_provider.return_value = MagicMock()
    return manager


@pytest.fixture
def mock_storage_manager() -> MagicMock:
    """Provides a mocked StorageManager."""
    manager = MagicMock(spec=StorageManager)
    return manager


@pytest.fixture
def mock_embedding_manager() -> MagicMock:
    """Provides a mocked EmbeddingManager."""
    manager = MagicMock(spec=EmbeddingManager)
    manager.embed.return_value = [0.1, 0.2, 0.3]  # Simple embedding
    return manager


# ============================================================================
# MOCK PROVIDER (FOR TOKEN COUNTING)
# ============================================================================


@pytest.fixture
def mock_provider():
    """Provides a mocked BaseProvider with token counting capabilities."""
    provider = MagicMock()

    # Mock count_message_tokens to return realistic token counts
    async def mock_count_tokens(messages: List[Message], model: str) -> int:
        """Mock token counting: ~4 chars per token average."""
        total_tokens = 0
        for msg in messages:
            if msg.content:
                # Approximate: 1 token per 4 characters + metadata tokens
                total_tokens += max(1, len(msg.content) // 4) + 10
        return total_tokens

    provider.count_message_tokens = AsyncMock(side_effect=mock_count_tokens)
    provider.get_context_length.return_value = 4096
    provider.model_name = "gpt-4"
    return provider


# ============================================================================
# MOCK CHAT SESSIONS
# ============================================================================


@pytest.fixture
def empty_session() -> ChatSession:
    """Provides an empty ChatSession."""
    session = ChatSession(
        id="test-session-1",
        title="Test Session",
        messages=[],
        context_items=[],
    )
    return session


@pytest.fixture
def session_with_system_message() -> ChatSession:
    """Provides a ChatSession with a system message."""
    session = ChatSession(
        id="test-session-2",
        title="Test Session",
        messages=[
            Message(
                role=Role.SYSTEM,
                content="You are a helpful AI assistant.",
                tokens=10,
            )
        ],
        context_items=[],
    )
    return session


@pytest.fixture
def session_with_history() -> ChatSession:
    """Provides a ChatSession with conversation history."""
    session = ChatSession(
        id="test-session-3",
        title="Test Session with History",
        messages=[
            Message(
                role=Role.SYSTEM,
                content="You are a helpful AI assistant.",
                tokens=10,
            ),
            Message(
                role=Role.USER,
                content="Hello, what is Python?",
                tokens=8,
            ),
            Message(
                role=Role.ASSISTANT,
                content="Python is a programming language known for its simplicity and readability.",
                tokens=15,
            ),
            Message(
                role=Role.USER,
                content="Tell me more about it.",
                tokens=6,
            ),
        ],
        context_items=[],
    )
    return session


@pytest.fixture
def session_with_context_items() -> ChatSession:
    """Provides a ChatSession with context items."""
    session = ChatSession(
        id="test-session-4",
        title="Test Session with Context",
        messages=[
            Message(
                role=Role.SYSTEM,
                content="You are a helpful AI assistant.",
                tokens=10,
            ),
        ],
        context_items=[
            ContextItem(
                id="ctx-1",
                item_type=ContextItemType.DOCUMENT,
                content="Python is a high-level programming language.",
                metadata={"source": "documentation"},
                tokens=10,
            ),
            ContextItem(
                id="ctx-2",
                item_type=ContextItemType.DOCUMENT,
                content="It was created by Guido van Rossum in 1991.",
                metadata={"source": "history"},
                tokens=9,
            ),
        ],
    )
    return session


# ============================================================================
# MOCK CONTEXT DOCUMENTS
# ============================================================================


@pytest.fixture
def simple_context_document() -> ContextDocument:
    """Provides a simple ContextDocument."""
    return ContextDocument(
        content="Machine learning is a subset of artificial intelligence.",
        metadata={
            "source": "encyclopedia",
            "page": 1,
        },
    )


@pytest.fixture
def context_documents_list() -> List[ContextDocument]:
    """Provides a list of ContextDocuments."""
    return [
        ContextDocument(
            content="Machine learning uses algorithms to learn from data.",
            metadata={"source": "article1", "relevance_score": 0.95},
        ),
        ContextDocument(
            content="Deep learning is a subset of machine learning.",
            metadata={"source": "article2", "relevance_score": 0.87},
        ),
        ContextDocument(
            content="Neural networks are inspired by biological neurons.",
            metadata={"source": "article3", "relevance_score": 0.82},
        ),
    ]


@pytest.fixture
def context_documents_with_metadata() -> List[ContextDocument]:
    """Provides ContextDocuments with rich metadata."""
    return [
        ContextDocument(
            content="Supervised learning requires labeled training data.",
            metadata={
                "source": "research_paper.pdf",
                "page_number": 42,
                "relevance_score": 0.93,
                "timestamp": "2024-01-15",
            },
        ),
        ContextDocument(
            content="Unsupervised learning discovers patterns in unlabeled data.",
            metadata={
                "source": "textbook.pdf",
                "page_number": 156,
                "relevance_score": 0.88,
                "chapter": "Chapter 5",
            },
        ),
    ]


# ============================================================================
# MOCK MESSAGES
# ============================================================================


@pytest.fixture
def mock_user_message() -> Message:
    """Provides a mock user message."""
    return Message(
        role=Role.USER,
        content="What is machine learning?",
        tokens=5,
    )


@pytest.fixture
def mock_assistant_message() -> Message:
    """Provides a mock assistant message."""
    return Message(
        role=Role.ASSISTANT,
        content="Machine learning is a field of artificial intelligence that enables systems to learn from data.",
        tokens=16,
    )


@pytest.fixture
def mock_system_message() -> Message:
    """Provides a mock system message."""
    return Message(
        role=Role.SYSTEM,
        content="You are an expert AI assistant specializing in machine learning and data science.",
        tokens=14,
    )


# ============================================================================
# CONTEXT ITEM FIXTURES
# ============================================================================


@pytest.fixture
def mock_context_item_rag() -> ContextItem:
    """Provides a mock RAG context item."""
    return ContextItem(
        id="rag-item-1",
        item_type=ContextItemType.DOCUMENT,
        content="The transformer architecture revolutionized natural language processing in 2017.",
        metadata={"source": "arxiv:1706.03762", "relevance": 0.98},
        tokens=14,
    )


@pytest.fixture
def mock_context_items_collection() -> List[ContextItem]:
    """Provides a collection of context items."""
    return [
        ContextItem(
            id="item-1",
            item_type=ContextItemType.DOCUMENT,
            content="Content about topic A.",
            metadata={"source": "source_a"},
            tokens=5,
        ),
        ContextItem(
            id="item-2",
            item_type=ContextItemType.DOCUMENT,
            content="Content about topic B with more information.",
            metadata={"source": "source_b"},
            tokens=8,
        ),
        ContextItem(
            id="item-3",
            item_type=ContextItemType.DOCUMENT,
            content="Additional context for reference.",
            metadata={"source": "source_c"},
            tokens=6,
        ),
    ]


# ============================================================================
# HELPERS
# ============================================================================


@pytest.fixture
def make_message():
    """Factory for creating custom messages."""

    def _make_message(
        role: Role = Role.USER,
        content: str = "Test message",
        tokens: Optional[int] = None,
    ) -> Message:
        if tokens is None:
            tokens = max(1, len(content) // 4)
        return Message(role=role, content=content, tokens=tokens)

    return _make_message


@pytest.fixture
def make_context_document():
    """Factory for creating custom context documents."""

    def _make_document(
        content: str = "Test content",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ContextDocument:
        if metadata is None:
            metadata = {"source": "test"}
        return ContextDocument(content=content, metadata=metadata)

    return _make_document
