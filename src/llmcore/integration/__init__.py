# src/llmcore/integration/__init__.py
"""
LLMCore Integration Module for External Engines.

Phase 3 (SYMBIOSIS): This module provides integration points for external
RAG engines (like SemantiScan) to delegate vector operations to LLMCore.

The goal is to enable SemantiScan to use LLMCore's unified storage layer
instead of directly managing ChromaDB or other vector stores.

Usage:
    from llmcore.integration import LLMCoreVectorClient

    # Initialize with an LLMCore instance
    client = await LLMCoreVectorClient.create(llmcore_instance, collection_name="my_collection")

    # Use with the same interface as ChromaDBClient
    client.add_chunks(chunks, embeddings)
    results = client.query(query_embedding, top_k=10)
"""

from .grimoire_runes import (
    RuneCommandTool,
    register_rune_tools,
    rune_command_metadata,
    rune_command_to_tool,
    rune_command_tool_name,
    rune_commands_to_tools,
)
from .vector_client import (
    LLMCoreVectorClient,
    LLMCoreVectorClientConfig,
    VectorClientProtocol,
)

__all__ = [
    "LLMCoreVectorClient",
    "LLMCoreVectorClientConfig",
    "RuneCommandTool",
    "VectorClientProtocol",
    "register_rune_tools",
    "rune_command_metadata",
    "rune_command_to_tool",
    "rune_command_tool_name",
    "rune_commands_to_tools",
]
