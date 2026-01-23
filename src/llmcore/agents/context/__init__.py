# src/llmcore/agents/context/__init__.py
"""
Context management module for agents.

Provides RAG filtering, context compression, priority-based
component selection, and token budget management.

Features:
    - Priority-based content selection (REQUIRED → OPTIONAL)
    - Automatic compression for verbose content
    - Conversation summarization for history management
    - Token counting and budget tracking
    - RAG chunk quality filtering

Usage:
    from llmcore.agents.context import ContextManager, RAGContextFilter

    # Context management
    ctx_manager = ContextManager(max_tokens=100000)
    built = ctx_manager.build_context(
        system_prompt="You are helpful",
        goal="Write a poem",
        history=messages,
        observations=obs,
    )
    print(f"Built context: {built.total_tokens} tokens")

    # RAG filtering
    rag_filter = RAGContextFilter()
    filtered = rag_filter.filter_chunks(chunks, query)
"""

from .context_manager import (
    BuiltContext,
    ContentType,
    # Data models
    ContextComponent,
    ContextManager,
    # Config
    ContextManagerConfig,
    ConversationSummarizer,
    Message,
    # Enums
    Priority,
    # Implementations
    SimpleTokenCounter,
    TextCompressor,
    # Protocols
    TokenCounter,
)
from .rag_filter import (
    FilterStats,
    RAGContextFilter,
    RAGResult,
)

__all__ = [
    # RAG filter
    "RAGContextFilter",
    "RAGResult",
    "FilterStats",
    # Context manager enums
    "Priority",
    "ContentType",
    # Context manager protocols
    "TokenCounter",
    # Context manager data models
    "ContextComponent",
    "Message",
    "BuiltContext",
    # Context manager config
    "ContextManagerConfig",
    # Context manager implementations
    "SimpleTokenCounter",
    "TextCompressor",
    "ConversationSummarizer",
    "ContextManager",
]
