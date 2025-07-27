# src/llmcore/memory/__init__.py
"""
Memory management module for the LLMCore library.

This package is responsible for assembling the final context payload
sent to the LLM provider and managing the hierarchical memory system.
It handles token limits, Retrieval Augmented Generation (RAG) logic,
episodic memory retrieval, and other context-aware operations.

The MemoryManager serves as the central, intelligent retrieval interface
for the entire three-tiered memory system (Semantic, Episodic, and Working Memory).
"""

from .manager import MemoryManager

__all__ = ["MemoryManager"]
