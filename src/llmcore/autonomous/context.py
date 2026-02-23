# src/llmcore/autonomous/context.py
"""
Autonomous Context Management.

This module bridges the autonomous subsystem to the main
:mod:`llmcore.context` package, providing convenience access to the
Adaptive Context Synthesis engine from the ``autonomous`` namespace.

The actual implementations live in:

- :mod:`llmcore.context.synthesis` — ``ContextSynthesizer``, ``ContextSource``
- :mod:`llmcore.context.compression` — ``ContextCompressor``
- :mod:`llmcore.context.prioritization` — ``ContentPrioritizer``
- :mod:`llmcore.context.sources` — Goal, Recent, Semantic, Episodic sources

This redirect exists to satisfy the spec's expected
``autonomous/context.py`` path while keeping the canonical implementations
in the dedicated ``context/`` package.

Example::

    from llmcore.autonomous.context import ContextSynthesizer

    synth = ContextSynthesizer(max_tokens=100_000)
    ctx = await synth.synthesize(current_task="analyze logs")

References:
    - UNIFIED_ECOSYSTEM_SPECIFICATION.md §12 (Adaptive Context Synthesis)
"""

from __future__ import annotations

# Re-export everything from the canonical context package so that
# ``from llmcore.autonomous.context import <X>`` works identically to
# ``from llmcore.context import <X>``.
from ..context import (
    CompressionResult,
    CompressionStrategy,
    ContentPrioritizer,
    ContextChunk,
    ContextCompressor,
    ContextSource,
    ContextSynthesizer,
    EstimateCounter,
    PriorityWeights,
    ScoredChunk,
    SynthesizedContext,
    TiktokenCounter,
    TokenCounter,
)

__all__ = [
    # Core synthesizer
    "ContextSynthesizer",
    "SynthesizedContext",
    "ContextChunk",
    # Protocols
    "ContextSource",
    "TokenCounter",
    # Token counters
    "TiktokenCounter",
    "EstimateCounter",
    # Compression
    "ContextCompressor",
    "CompressionStrategy",
    "CompressionResult",
    # Prioritization
    "ContentPrioritizer",
    "PriorityWeights",
    "ScoredChunk",
]
