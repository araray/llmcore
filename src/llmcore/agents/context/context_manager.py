# src/llmcore/agents/context/context_manager.py
"""
Context Manager for LLM Context Window Management.

Manages the limited context window by prioritizing essential content,
compressing verbose sections, and offloading to external stores.

Key Features:
    - Priority-based content inclusion
    - Context compression for long content
    - Token budget management
    - Conversation summarization
    - RAG context integration

Problem:
    - LLM performance degrades as context grows ("context rot")
    - Important info gets lost in the middle
    - Token costs increase linearly

Solution:
    - OFFLOAD: Store full data externally, send summary
    - REDUCE: Compress conversations, remove old messages
    - RETRIEVE: Fetch on demand via RAG

Usage:
    from llmcore.agents.context import ContextManager, Priority

    manager = ContextManager(max_tokens=100000)

    context = manager.build_context(
        system_prompt="You are a helpful assistant",
        goal="Analyze this data",
        history=conversation_messages,
        observations=tool_outputs,
        rag_context=retrieved_docs,
    )

    # Use context.messages for LLM call
    response = await llm.chat(context.messages)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
)

try:
    from pydantic import BaseModel, Field

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object

    def Field(*args, **kwargs):
        return kwargs.get("default")


logger = logging.getLogger(__name__)


# =============================================================================
# Priority System
# =============================================================================


class Priority(IntEnum):
    """Priority levels for context components."""

    REQUIRED = 100  # Always include (system prompt, goal)
    CRITICAL = 90  # Almost always include (recent history, observations)
    HIGH = 70  # Include if space permits (reflections, errors)
    MEDIUM = 50  # Include if budget allows (older history)
    LOW = 30  # Optional (RAG context, examples)
    OPTIONAL = 10  # Nice to have (metadata, verbose logs)


class ContentType(str, Enum):
    """Types of content in context."""

    SYSTEM = "system"
    GOAL = "goal"
    HISTORY = "history"
    OBSERVATION = "observation"
    REFLECTION = "reflection"
    RAG = "rag"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    EXAMPLE = "example"
    METADATA = "metadata"


# =============================================================================
# Token Counting
# =============================================================================


class TokenCounter(Protocol):
    """Protocol for token counting."""

    def count(self, text: str) -> int:
        """Count tokens in text."""
        ...


class SimpleTokenCounter:
    """
    Simple token counter using word/character estimation.

    For production, use tiktoken or model-specific tokenizers.
    """

    def __init__(self, chars_per_token: float = 4.0):
        self.chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        """Estimate token count."""
        if not text:
            return 0
        return int(len(text) / self.chars_per_token)


# =============================================================================
# Content Components
# =============================================================================


@dataclass
class ContextComponent:
    """A component of the context window."""

    content: str
    content_type: ContentType
    priority: Priority
    tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    compressible: bool = True
    source: str = ""

    def __post_init__(self):
        if self.tokens == 0 and self.content:
            # Estimate tokens
            self.tokens = int(len(self.content) / 4)


@dataclass
class Message:
    """A chat message."""

    role: str
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass
class BuiltContext:
    """Result of building context."""

    messages: List[Dict[str, Any]]
    total_tokens: int
    budget_remaining: int
    included_components: List[str]
    excluded_components: List[str]
    compression_applied: bool
    warnings: List[str]

    def get_message_count(self) -> int:
        """Get number of messages."""
        return len(self.messages)


# =============================================================================
# Text Compression
# =============================================================================


class TextCompressor:
    """
    Compresses text while preserving key information.

    Strategies:
    - Remove redundant whitespace
    - Shorten verbose phrases
    - Extract key sentences
    - Summarize long sections
    """

    def __init__(
        self,
        max_sentence_length: int = 200,
        preserve_code: bool = True,
    ):
        self.max_sentence_length = max_sentence_length
        self.preserve_code = preserve_code

        # Verbose phrases to shorten
        self._replacements = [
            (r"\bin order to\b", "to"),
            (r"\bdue to the fact that\b", "because"),
            (r"\bat this point in time\b", "now"),
            (r"\bin the event that\b", "if"),
            (r"\bprior to\b", "before"),
            (r"\bsubsequent to\b", "after"),
            (r"\bwith regard to\b", "about"),
            (r"\bin spite of the fact that\b", "although"),
            (r"\bfor the purpose of\b", "to"),
            (r"\bin close proximity to\b", "near"),
            (r"\s{2,}", " "),  # Multiple spaces
            (r"\n{3,}", "\n\n"),  # Multiple newlines
        ]

    def compress(
        self,
        text: str,
        target_ratio: float = 0.7,
        preserve_first_n_chars: int = 500,
    ) -> str:
        """
        Compress text to target ratio.

        Args:
            text: Text to compress
            target_ratio: Target size as ratio of original
            preserve_first_n_chars: Always preserve beginning

        Returns:
            Compressed text
        """
        if not text:
            return text

        original_len = len(text)
        target_len = int(original_len * target_ratio)

        # Preserve beginning
        preserved = text[:preserve_first_n_chars]
        rest = text[preserve_first_n_chars:]

        # Apply replacements
        for pattern, replacement in self._replacements:
            rest = re.sub(pattern, replacement, rest, flags=re.IGNORECASE)

        # If still too long, extract key sentences
        if len(preserved) + len(rest) > target_len:
            rest = self._extract_key_content(rest, target_len - len(preserved))

        result = preserved + rest

        logger.debug(
            f"Compressed {original_len} -> {len(result)} chars ({len(result) / original_len:.0%})"
        )

        return result

    def _extract_key_content(self, text: str, max_length: int) -> str:
        """Extract key sentences from text."""
        if len(text) <= max_length:
            return text

        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)

        if not sentences:
            return text[:max_length]

        # Score sentences by importance
        scored = []
        for i, sentence in enumerate(sentences):
            score = self._score_sentence(sentence, i, len(sentences))
            scored.append((score, sentence))

        # Sort by score and select top sentences
        scored.sort(key=lambda x: -x[0])

        selected = []
        total_len = 0

        for score, sentence in scored:
            if total_len + len(sentence) + 1 > max_length:
                break
            selected.append(sentence)
            total_len += len(sentence) + 1

        # Reconstruct in original order
        sentence_indices = {s: i for i, s in enumerate(sentences)}
        selected.sort(key=lambda s: sentence_indices.get(s, 0))

        return " ".join(selected)

    def _score_sentence(
        self,
        sentence: str,
        position: int,
        total: int,
    ) -> float:
        """Score sentence importance."""
        score = 1.0

        # Position bonus (first and last sentences often important)
        if position < 2:
            score += 0.3
        if position >= total - 2:
            score += 0.2

        # Length penalty (very short sentences less informative)
        if len(sentence) < 20:
            score -= 0.3

        # Key phrase bonus
        key_phrases = [
            "important",
            "must",
            "should",
            "error",
            "warning",
            "result",
            "output",
            "because",
            "therefore",
            "however",
            "conclusion",
            "summary",
            "finally",
            "key",
            "critical",
        ]
        for phrase in key_phrases:
            if phrase in sentence.lower():
                score += 0.2
                break

        # Code detection (preserve code)
        if self.preserve_code:
            if "```" in sentence or re.search(r"^\s{4}", sentence):
                score += 0.5

        return score


# =============================================================================
# Conversation Summarizer
# =============================================================================


class ConversationSummarizer:
    """
    Summarizes conversation history to save context space.

    Converts:
        User: Hello, how are you?
        Assistant: I'm doing well, thank you!
        User: Can you help me with Python?
        Assistant: Of course! What would you like help with?

    To:
        [Previous: User greeted, asked for Python help. Assistant agreed to help.]
    """

    def __init__(self, max_summary_length: int = 200):
        self.max_summary_length = max_summary_length

    def summarize_messages(
        self,
        messages: List[Dict[str, Any]],
        preserve_last_n: int = 4,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Summarize older messages while preserving recent ones.

        Args:
            messages: List of messages
            preserve_last_n: Number of recent messages to keep

        Returns:
            Tuple of (preserved messages, summary of old messages)
        """
        if len(messages) <= preserve_last_n:
            return messages, ""

        to_summarize = messages[:-preserve_last_n]
        to_preserve = messages[-preserve_last_n:]

        summary = self._create_summary(to_summarize)

        return to_preserve, summary

    def _create_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Create summary of messages."""
        topics = []
        actions = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")[:200]  # Limit per message

            if role == "user":
                # Extract user's intent
                if "?" in content:
                    topics.append("asked question")
                elif any(w in content.lower() for w in ["please", "can you", "help"]):
                    topics.append("requested help")
                elif any(w in content.lower() for w in ["create", "write", "make"]):
                    topics.append("requested creation")
            elif role == "assistant":
                if any(w in content.lower() for w in ["here is", "created", "done"]):
                    actions.append("provided")
                elif any(w in content.lower() for w in ["sorry", "cannot", "unable"]):
                    actions.append("declined/clarified")

        parts = []
        if topics:
            unique_topics = list(set(topics))[:3]
            parts.append(f"User: {', '.join(unique_topics)}")
        if actions:
            unique_actions = list(set(actions))[:2]
            parts.append(f"Assistant: {', '.join(unique_actions)}")

        summary = "; ".join(parts) if parts else "Previous conversation context"

        return f"[Earlier: {summary}]"


# =============================================================================
# Context Manager
# =============================================================================


class ContextManagerConfig:
    """Configuration for context manager."""

    def __init__(
        self,
        max_tokens: int = 100000,
        reserve_for_output: int = 4000,
        compression_threshold: float = 0.8,
        auto_summarize_history: bool = True,
        history_preserve_count: int = 6,
        max_rag_chunks: int = 5,
        max_observations: int = 10,
    ):
        self.max_tokens = max_tokens
        self.reserve_for_output = reserve_for_output
        self.compression_threshold = compression_threshold
        self.auto_summarize_history = auto_summarize_history
        self.history_preserve_count = history_preserve_count
        self.max_rag_chunks = max_rag_chunks
        self.max_observations = max_observations


class ContextManager:
    """
    Manages the LLM context window.

    Responsibilities:
    - Prioritize essential content
    - Compress verbose sections
    - Summarize old conversation history
    - Integrate RAG context
    - Track token budget

    Args:
        config: Context manager configuration
        token_counter: Token counting implementation
    """

    def __init__(
        self,
        config: Optional[ContextManagerConfig] = None,
        token_counter: Optional[TokenCounter] = None,
    ):
        self.config = config or ContextManagerConfig()
        self.token_counter = token_counter or SimpleTokenCounter()

        self._compressor = TextCompressor()
        self._summarizer = ConversationSummarizer()

        # Default priorities for content types
        self._default_priorities = {
            ContentType.SYSTEM: Priority.REQUIRED,
            ContentType.GOAL: Priority.REQUIRED,
            ContentType.HISTORY: Priority.HIGH,
            ContentType.OBSERVATION: Priority.CRITICAL,
            ContentType.REFLECTION: Priority.HIGH,
            ContentType.RAG: Priority.LOW,
            ContentType.TOOL_RESULT: Priority.CRITICAL,
            ContentType.ERROR: Priority.HIGH,
            ContentType.EXAMPLE: Priority.LOW,
            ContentType.METADATA: Priority.OPTIONAL,
        }

    def build_context(
        self,
        system_prompt: str,
        goal: str,
        history: Optional[List[Dict[str, Any]]] = None,
        observations: Optional[List[str]] = None,
        reflections: Optional[List[str]] = None,
        rag_context: Optional[List[str]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
        additional_context: Optional[Dict[str, str]] = None,
    ) -> BuiltContext:
        """
        Build optimized context from components.

        Args:
            system_prompt: System prompt (always included)
            goal: Current goal (always included)
            history: Conversation history
            observations: Tool/action observations
            reflections: Reflection outputs
            rag_context: Retrieved documents
            tool_results: Tool execution results
            additional_context: Custom context additions

        Returns:
            BuiltContext ready for LLM call
        """
        budget = self.config.max_tokens - self.config.reserve_for_output
        warnings: List[str] = []
        compression_applied = False

        # Create components
        components: List[ContextComponent] = []

        # System prompt (required)
        components.append(
            ContextComponent(
                content=system_prompt,
                content_type=ContentType.SYSTEM,
                priority=Priority.REQUIRED,
                compressible=False,
            )
        )

        # Goal (required)
        goal_content = f"## Current Goal\n{goal}"
        components.append(
            ContextComponent(
                content=goal_content,
                content_type=ContentType.GOAL,
                priority=Priority.REQUIRED,
                compressible=False,
            )
        )

        # History (high priority, may need summarization)
        if history:
            history_content, summary = self._process_history(history)
            if summary:
                warnings.append(f"Summarized {len(history) - len(history_content)} messages")
                components.append(
                    ContextComponent(
                        content=summary,
                        content_type=ContentType.HISTORY,
                        priority=Priority.MEDIUM,
                        source="summarized",
                    )
                )
            # Recent history as individual messages (handled separately)
            self._recent_history = history_content
        else:
            self._recent_history = []

        # Observations (critical)
        if observations:
            obs_limited = observations[-self.config.max_observations :]
            obs_content = "\n".join(f"- {o[:500]}" for o in obs_limited)
            components.append(
                ContextComponent(
                    content=f"## Observations\n{obs_content}",
                    content_type=ContentType.OBSERVATION,
                    priority=Priority.CRITICAL,
                )
            )

        # Reflections (high)
        if reflections:
            ref_content = "\n".join(reflections[-3:])  # Last 3 reflections
            components.append(
                ContextComponent(
                    content=f"## Reflections\n{ref_content}",
                    content_type=ContentType.REFLECTION,
                    priority=Priority.HIGH,
                )
            )

        # RAG context (low priority, limited)
        if rag_context:
            rag_limited = rag_context[: self.config.max_rag_chunks]
            rag_content = "\n---\n".join(rag_limited)
            components.append(
                ContextComponent(
                    content=f"## Retrieved Context\n{rag_content}",
                    content_type=ContentType.RAG,
                    priority=Priority.LOW,
                )
            )

        # Tool results (critical)
        if tool_results:
            for result in tool_results[-5:]:  # Last 5 results
                tool_name = result.get("tool", "unknown")
                output = str(result.get("output", ""))[:1000]
                components.append(
                    ContextComponent(
                        content=f"[Tool: {tool_name}]\n{output}",
                        content_type=ContentType.TOOL_RESULT,
                        priority=Priority.CRITICAL,
                    )
                )

        # Additional context
        if additional_context:
            for name, content in additional_context.items():
                components.append(
                    ContextComponent(
                        content=f"## {name}\n{content}",
                        content_type=ContentType.METADATA,
                        priority=Priority.MEDIUM,
                    )
                )

        # Count tokens for all components
        for comp in components:
            comp.tokens = self.token_counter.count(comp.content)

        # Select components within budget
        included, excluded = self._select_components(components, budget)

        # Check if compression needed
        total_tokens = sum(c.tokens for c in included)
        if total_tokens > budget * self.config.compression_threshold:
            included = self._compress_components(included, budget)
            compression_applied = True
            warnings.append("Applied compression to fit context budget")

        # Build messages
        messages = self._build_messages(included)

        # Add recent history messages
        for msg in self._recent_history:
            messages.append(msg)

        # Final token count
        final_tokens = sum(self.token_counter.count(str(m)) for m in messages)

        return BuiltContext(
            messages=messages,
            total_tokens=final_tokens,
            budget_remaining=budget - final_tokens,
            included_components=[c.content_type.value for c in included],
            excluded_components=[c.content_type.value for c in excluded],
            compression_applied=compression_applied,
            warnings=warnings,
        )

    def _process_history(
        self,
        history: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Process conversation history with optional summarization."""
        if not self.config.auto_summarize_history:
            return history, ""

        if len(history) <= self.config.history_preserve_count:
            return history, ""

        preserved, summary = self._summarizer.summarize_messages(
            history,
            preserve_last_n=self.config.history_preserve_count,
        )

        return preserved, summary

    def _select_components(
        self,
        components: List[ContextComponent],
        budget: int,
    ) -> Tuple[List[ContextComponent], List[ContextComponent]]:
        """Select components within budget based on priority."""
        # Sort by priority (descending)
        sorted_components = sorted(
            components,
            key=lambda c: c.priority,
            reverse=True,
        )

        included = []
        excluded = []
        used_tokens = 0

        for comp in sorted_components:
            # Required components always included
            if comp.priority >= Priority.REQUIRED:
                included.append(comp)
                used_tokens += comp.tokens
                continue

            # Check if fits in budget
            if used_tokens + comp.tokens <= budget:
                included.append(comp)
                used_tokens += comp.tokens
            else:
                excluded.append(comp)

        return included, excluded

    def _compress_components(
        self,
        components: List[ContextComponent],
        budget: int,
    ) -> List[ContextComponent]:
        """Compress compressible components to fit budget."""
        total_tokens = sum(c.tokens for c in components)
        if total_tokens <= budget:
            return components

        target_ratio = budget / total_tokens

        compressed = []
        for comp in components:
            if comp.compressible and comp.priority < Priority.REQUIRED:
                new_content = self._compressor.compress(
                    comp.content,
                    target_ratio=max(target_ratio, 0.5),
                )
                comp.content = new_content
                comp.tokens = self.token_counter.count(new_content)
            compressed.append(comp)

        return compressed

    def _build_messages(
        self,
        components: List[ContextComponent],
    ) -> List[Dict[str, Any]]:
        """Build message list from components."""
        messages = []

        # Find system component
        system_content_parts = []
        other_components = []

        for comp in components:
            if comp.content_type == ContentType.SYSTEM:
                system_content_parts.append(comp.content)
            else:
                other_components.append(comp)

        # System message
        if system_content_parts:
            messages.append(
                {
                    "role": "system",
                    "content": "\n\n".join(system_content_parts),
                }
            )

        # Combine other components into user context message
        context_parts = []
        for comp in other_components:
            if comp.content:
                context_parts.append(comp.content)

        if context_parts:
            messages.append(
                {
                    "role": "user",
                    "content": "\n\n".join(context_parts),
                }
            )

        return messages

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return self.token_counter.count(text)

    def get_remaining_budget(self, current_tokens: int) -> int:
        """Get remaining token budget."""
        return self.config.max_tokens - self.config.reserve_for_output - current_tokens


# =============================================================================
# Convenience Functions
# =============================================================================


def create_context_manager(
    max_tokens: int = 100000,
    reserve_for_output: int = 4000,
) -> ContextManager:
    """Create a context manager with default settings."""
    config = ContextManagerConfig(
        max_tokens=max_tokens,
        reserve_for_output=reserve_for_output,
    )
    return ContextManager(config=config)


def estimate_conversation_tokens(
    messages: List[Dict[str, Any]],
    chars_per_token: float = 4.0,
) -> int:
    """Estimate total tokens in a conversation."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        total += int(len(content) / chars_per_token)
    return total


__all__ = [
    # Enums
    "Priority",
    "ContentType",
    # Token counting
    "TokenCounter",
    "SimpleTokenCounter",
    # Data models
    "ContextComponent",
    "Message",
    "BuiltContext",
    # Compression
    "TextCompressor",
    "ConversationSummarizer",
    # Config
    "ContextManagerConfig",
    # Main class
    "ContextManager",
    # Convenience
    "create_context_manager",
    "estimate_conversation_tokens",
]
