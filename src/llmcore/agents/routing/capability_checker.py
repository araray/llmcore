# src/llmcore/agents/routing/capability_checker.py
"""
Model Capability Checker for LLMCore Agent System.

This module provides pre-flight capability checking for models before
agent execution, preventing failures from attempting unsupported operations.

Key capabilities checked:
- Tool/function calling support
- Vision/image support
- Context window size
- Streaming support
- JSON mode support

Usage:
    from llmcore.agents.routing.capability_checker import CapabilityChecker

    checker = CapabilityChecker(model_registry)

    result = checker.check_compatibility(
        model="gemma3:4b",
        requires_tools=True,
    )

    if not result.compatible:
        print(f"Model incompatible: {result.issues}")
        print(f"Suggestion: {result.suggestions}")

Integration Point:
    Call at start of CognitiveCycle.run() before execution begins.

Author: llmcore team
Date: 2026-01-21
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Models
# =============================================================================


class Capability(str, Enum):
    """Model capabilities that can be checked."""

    TOOLS = "tools"  # Function/tool calling
    VISION = "vision"  # Image understanding
    STREAMING = "streaming"  # Streaming responses
    JSON_MODE = "json_mode"  # Structured JSON output
    SYSTEM_MESSAGE = "system_message"  # System message support
    MULTI_TURN = "multi_turn"  # Multi-turn conversation
    CODE_EXECUTION = "code_execution"  # Code interpreter

    # Context sizes
    CONTEXT_4K = "context_4k"
    CONTEXT_8K = "context_8k"
    CONTEXT_16K = "context_16k"
    CONTEXT_32K = "context_32k"
    CONTEXT_128K = "context_128k"


class IssueSeverity(str, Enum):
    """Severity of compatibility issues."""

    ERROR = "error"  # Blocks execution
    WARNING = "warning"  # May cause problems
    INFO = "info"  # Informational only


@dataclass
class CapabilityIssue:
    """A single capability compatibility issue."""

    capability: Capability
    severity: IssueSeverity
    message: str
    suggestion: str

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.capability.value}: {self.message}"


@dataclass
class CompatibilityResult:
    """Result of a capability compatibility check."""

    compatible: bool
    model: str
    issues: List[CapabilityIssue] = field(default_factory=list)
    model_capabilities: Set[Capability] = field(default_factory=set)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == IssueSeverity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == IssueSeverity.WARNING for i in self.issues)

    @property
    def suggestions(self) -> List[str]:
        return [i.suggestion for i in self.issues if i.suggestion]

    def __str__(self) -> str:
        status = "✅ Compatible" if self.compatible else "❌ Incompatible"
        lines = [f"{status}: {self.model}"]
        for issue in self.issues:
            lines.append(f"  {issue}")
        return "\n".join(lines)


@dataclass
class ModelInfo:
    """Information about a model's capabilities."""

    name: str
    provider: str
    capabilities: Set[Capability]
    context_window: int
    max_output_tokens: int
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0

    @property
    def supports_tools(self) -> bool:
        return Capability.TOOLS in self.capabilities

    @property
    def supports_vision(self) -> bool:
        return Capability.VISION in self.capabilities

    @property
    def supports_streaming(self) -> bool:
        return Capability.STREAMING in self.capabilities


# =============================================================================
# Model Registry (Stub - integrate with actual model cards)
# =============================================================================

# This is a simplified registry - in production, integrate with llmcore model cards
DEFAULT_MODEL_INFO: Dict[str, ModelInfo] = {
    # OpenAI
    "gpt-4-turbo": ModelInfo(
        name="gpt-4-turbo",
        provider="openai",
        capabilities={
            Capability.TOOLS,
            Capability.VISION,
            Capability.STREAMING,
            Capability.JSON_MODE,
            Capability.SYSTEM_MESSAGE,
            Capability.MULTI_TURN,
            Capability.CONTEXT_128K,
        },
        context_window=128000,
        max_output_tokens=4096,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
    ),
    "gpt-4o": ModelInfo(
        name="gpt-4o",
        provider="openai",
        capabilities={
            Capability.TOOLS,
            Capability.VISION,
            Capability.STREAMING,
            Capability.JSON_MODE,
            Capability.SYSTEM_MESSAGE,
            Capability.MULTI_TURN,
            Capability.CONTEXT_128K,
        },
        context_window=128000,
        max_output_tokens=16384,
        cost_per_1k_input=0.005,
        cost_per_1k_output=0.015,
    ),
    "gpt-4o-mini": ModelInfo(
        name="gpt-4o-mini",
        provider="openai",
        capabilities={
            Capability.TOOLS,
            Capability.VISION,
            Capability.STREAMING,
            Capability.JSON_MODE,
            Capability.SYSTEM_MESSAGE,
            Capability.MULTI_TURN,
            Capability.CONTEXT_128K,
        },
        context_window=128000,
        max_output_tokens=16384,
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
    ),
    # Anthropic
    "claude-3-opus": ModelInfo(
        name="claude-3-opus",
        provider="anthropic",
        capabilities={
            Capability.TOOLS,
            Capability.VISION,
            Capability.STREAMING,
            Capability.SYSTEM_MESSAGE,
            Capability.MULTI_TURN,
            Capability.CONTEXT_128K,
        },
        context_window=200000,
        max_output_tokens=4096,
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
    ),
    "claude-3-5-sonnet": ModelInfo(
        name="claude-3-5-sonnet",
        provider="anthropic",
        capabilities={
            Capability.TOOLS,
            Capability.VISION,
            Capability.STREAMING,
            Capability.SYSTEM_MESSAGE,
            Capability.MULTI_TURN,
            Capability.CONTEXT_128K,
        },
        context_window=200000,
        max_output_tokens=8192,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
    ),
    "claude-3-haiku": ModelInfo(
        name="claude-3-haiku",
        provider="anthropic",
        capabilities={
            Capability.TOOLS,
            Capability.VISION,
            Capability.STREAMING,
            Capability.SYSTEM_MESSAGE,
            Capability.MULTI_TURN,
            Capability.CONTEXT_128K,
        },
        context_window=200000,
        max_output_tokens=4096,
        cost_per_1k_input=0.00025,
        cost_per_1k_output=0.00125,
    ),
    # Google
    "gemini-pro": ModelInfo(
        name="gemini-pro",
        provider="google",
        capabilities={
            Capability.TOOLS,
            Capability.STREAMING,
            Capability.SYSTEM_MESSAGE,
            Capability.MULTI_TURN,
            Capability.CONTEXT_32K,
        },
        context_window=32000,
        max_output_tokens=8192,
    ),
    "gemini-pro-vision": ModelInfo(
        name="gemini-pro-vision",
        provider="google",
        capabilities={
            Capability.VISION,
            Capability.STREAMING,
            Capability.SYSTEM_MESSAGE,
            Capability.MULTI_TURN,
            Capability.CONTEXT_16K,
        },
        context_window=16000,
        max_output_tokens=4096,
    ),
    # Ollama (local models - typically NO tool support)
    "llama3:8b": ModelInfo(
        name="llama3:8b",
        provider="ollama",
        capabilities={
            Capability.STREAMING,
            Capability.SYSTEM_MESSAGE,
            Capability.MULTI_TURN,
            Capability.CONTEXT_8K,
        },
        context_window=8192,
        max_output_tokens=4096,
    ),
    "llama3:70b": ModelInfo(
        name="llama3:70b",
        provider="ollama",
        capabilities={
            Capability.STREAMING,
            Capability.SYSTEM_MESSAGE,
            Capability.MULTI_TURN,
            Capability.CONTEXT_8K,
        },
        context_window=8192,
        max_output_tokens=4096,
    ),
    "gemma3:4b": ModelInfo(
        name="gemma3:4b",
        provider="ollama",
        capabilities={
            Capability.STREAMING,
            Capability.SYSTEM_MESSAGE,
            Capability.MULTI_TURN,
            Capability.CONTEXT_8K,
        },
        context_window=8192,
        max_output_tokens=4096,
    ),
    "mistral:7b": ModelInfo(
        name="mistral:7b",
        provider="ollama",
        capabilities={
            Capability.STREAMING,
            Capability.SYSTEM_MESSAGE,
            Capability.MULTI_TURN,
            Capability.CONTEXT_8K,
        },
        context_window=8192,
        max_output_tokens=4096,
    ),
    "codellama:13b": ModelInfo(
        name="codellama:13b",
        provider="ollama",
        capabilities={
            Capability.STREAMING,
            Capability.SYSTEM_MESSAGE,
            Capability.MULTI_TURN,
            Capability.CONTEXT_16K,
            Capability.CODE_EXECUTION,
        },
        context_window=16384,
        max_output_tokens=4096,
    ),
}


# =============================================================================
# Capability Checker Implementation
# =============================================================================


class CapabilityChecker:
    """
    Check model capabilities before agent execution.

    Prevents failures by verifying model can support required operations
    before starting agent execution.

    Args:
        model_registry: Optional custom model registry (dict of name -> ModelInfo)
    """

    def __init__(self, model_registry: Optional[Dict[str, ModelInfo]] = None):
        self.model_registry = model_registry or DEFAULT_MODEL_INFO

    def get_model_info(self, model: str) -> Optional[ModelInfo]:
        """
        Get model info, handling variations in model names.

        Handles cases like:
        - Exact match: "gpt-4-turbo"
        - With version: "gpt-4-turbo-2024-01-01"
        - Provider prefix: "openai/gpt-4-turbo"
        - Ollama format: "ollama:llama3:8b"
        """
        # Try exact match first
        if model in self.model_registry:
            return self.model_registry[model]

        # Try without provider prefix
        if "/" in model:
            model_name = model.split("/")[-1]
            if model_name in self.model_registry:
                return self.model_registry[model_name]

        # Try base model name (remove version suffix)
        for registered_name in self.model_registry:
            if model.startswith(registered_name):
                return self.model_registry[registered_name]

        # Try partial match for Ollama models
        model_lower = model.lower()
        for registered_name, info in self.model_registry.items():
            if registered_name in model_lower or model_lower in registered_name:
                return info

        return None

    def check_compatibility(
        self,
        model: str,
        requires_tools: bool = False,
        requires_vision: bool = False,
        requires_streaming: bool = False,
        requires_json_mode: bool = False,
        min_context_window: int = 4096,
        estimated_tokens: Optional[int] = None,
    ) -> CompatibilityResult:
        """
        Check if model is compatible with requirements.

        Args:
            model: Model name
            requires_tools: Whether task needs tool/function calling
            requires_vision: Whether task needs vision/image support
            requires_streaming: Whether streaming is required
            requires_json_mode: Whether JSON mode is required
            min_context_window: Minimum required context window
            estimated_tokens: Estimated tokens for this task

        Returns:
            CompatibilityResult with compatibility status and any issues
        """
        issues: List[CapabilityIssue] = []

        # Get model info
        model_info = self.get_model_info(model)

        if model_info is None:
            return CompatibilityResult(
                compatible=False,
                model=model,
                issues=[
                    CapabilityIssue(
                        capability=Capability.TOOLS,  # Generic
                        severity=IssueSeverity.ERROR,
                        message=f"Unknown model: {model}",
                        suggestion="Check model name or add to registry",
                    )
                ],
            )

        # Check tool support
        if requires_tools and not model_info.supports_tools:
            issues.append(
                CapabilityIssue(
                    capability=Capability.TOOLS,
                    severity=IssueSeverity.ERROR,
                    message=f"Model {model} does not support tool/function calling",
                    suggestion=(
                        "Use --use-activities flag for activity-based execution, "
                        "or switch to a tool-capable model (gpt-4o, claude-3)"
                    ),
                )
            )

        # Check vision support
        if requires_vision and not model_info.supports_vision:
            issues.append(
                CapabilityIssue(
                    capability=Capability.VISION,
                    severity=IssueSeverity.ERROR,
                    message=f"Model {model} does not support vision/images",
                    suggestion="Switch to a vision-capable model (gpt-4o, claude-3, gemini-pro-vision)",
                )
            )

        # Check streaming support
        if requires_streaming and not model_info.supports_streaming:
            issues.append(
                CapabilityIssue(
                    capability=Capability.STREAMING,
                    severity=IssueSeverity.WARNING,
                    message=f"Model {model} does not support streaming",
                    suggestion="Responses will be returned all at once",
                )
            )

        # Check JSON mode
        if requires_json_mode and Capability.JSON_MODE not in model_info.capabilities:
            issues.append(
                CapabilityIssue(
                    capability=Capability.JSON_MODE,
                    severity=IssueSeverity.WARNING,
                    message=f"Model {model} does not have native JSON mode",
                    suggestion="Will use prompt-based JSON extraction (less reliable)",
                )
            )

        # Check context window
        if model_info.context_window < min_context_window:
            issues.append(
                CapabilityIssue(
                    capability=Capability.CONTEXT_8K,
                    severity=IssueSeverity.ERROR,
                    message=(
                        f"Model context window ({model_info.context_window}) "
                        f"is smaller than required ({min_context_window})"
                    ),
                    suggestion="Enable context compression or use model with larger context",
                )
            )

        # Check estimated token usage
        if estimated_tokens and estimated_tokens > model_info.context_window * 0.9:
            issues.append(
                CapabilityIssue(
                    capability=Capability.CONTEXT_32K,
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"Estimated tokens ({estimated_tokens}) approaching "
                        f"context limit ({model_info.context_window})"
                    ),
                    suggestion="Consider enabling context compression",
                )
            )

        # Determine overall compatibility
        compatible = not any(i.severity == IssueSeverity.ERROR for i in issues)

        return CompatibilityResult(
            compatible=compatible,
            model=model,
            issues=issues,
            model_capabilities=model_info.capabilities,
        )

    def check_agent_compatibility(
        self,
        model: str,
        use_native_tools: bool = True,
        use_activities: bool = False,
        task_complexity: str = "moderate",
    ) -> CompatibilityResult:
        """
        High-level check for agent execution compatibility.

        Args:
            model: Model name
            use_native_tools: Whether to use native tool calling
            use_activities: Whether to use activity-based execution
            task_complexity: "trivial", "simple", "moderate", "complex"

        Returns:
            CompatibilityResult with recommendations
        """
        # Determine requirements based on complexity
        complexity_requirements = {
            "trivial": {"requires_tools": False, "min_context": 4096},
            "simple": {
                "requires_tools": use_native_tools and not use_activities,
                "min_context": 8192,
            },
            "moderate": {
                "requires_tools": use_native_tools and not use_activities,
                "min_context": 16384,
            },
            "complex": {
                "requires_tools": use_native_tools and not use_activities,
                "min_context": 32768,
            },
        }

        reqs = complexity_requirements.get(task_complexity, complexity_requirements["moderate"])

        result = self.check_compatibility(
            model=model,
            requires_tools=reqs["requires_tools"],
            min_context_window=reqs["min_context"],
        )

        # Add activity system suggestion if tools not supported
        if not result.compatible:
            for issue in result.issues:
                if issue.capability == Capability.TOOLS:
                    # Already has tool suggestion, but emphasize activities
                    pass

        return result

    def suggest_alternative_models(
        self,
        required_capabilities: Set[Capability],
        exclude_providers: Optional[Set[str]] = None,
        max_suggestions: int = 3,
    ) -> List[ModelInfo]:
        """
        Suggest alternative models that meet requirements.

        Args:
            required_capabilities: Set of required capabilities
            exclude_providers: Providers to exclude (e.g., {"ollama"} for cloud-only)
            max_suggestions: Maximum number of suggestions

        Returns:
            List of compatible ModelInfo sorted by cost
        """
        exclude_providers = exclude_providers or set()
        compatible = []

        for name, info in self.model_registry.items():
            # Check provider exclusion
            if info.provider in exclude_providers:
                continue

            # Check all required capabilities
            if required_capabilities.issubset(info.capabilities):
                compatible.append(info)

        # Sort by cost (cheapest first)
        compatible.sort(key=lambda m: m.cost_per_1k_input + m.cost_per_1k_output)

        return compatible[:max_suggestions]


# =============================================================================
# Convenience Functions
# =============================================================================


def check_model_for_agent(
    model: str,
    use_tools: bool = True,
) -> CompatibilityResult:
    """
    Quick check if model can run agent tasks.

    Args:
        model: Model name
        use_tools: Whether native tools will be used

    Returns:
        CompatibilityResult
    """
    checker = CapabilityChecker()
    return checker.check_compatibility(model=model, requires_tools=use_tools)


def get_model_capabilities(model: str) -> Optional[Set[Capability]]:
    """Get capabilities of a model."""
    checker = CapabilityChecker()
    info = checker.get_model_info(model)
    return info.capabilities if info else None


def model_supports_tools(model: str) -> bool:
    """Check if model supports tool calling."""
    checker = CapabilityChecker()
    info = checker.get_model_info(model)
    return info.supports_tools if info else False


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    print("Running Capability Checker self-tests...\n")

    passed = 0
    failed = 0

    checker = CapabilityChecker()

    # Test 1: Tool-capable model
    print("Test 1: Tool-capable model (gpt-4o)")
    result = checker.check_compatibility("gpt-4o", requires_tools=True)
    if result.compatible:
        print("  ✅ gpt-4o is compatible with tools")
        passed += 1
    else:
        print("  ❌ gpt-4o should be compatible")
        failed += 1

    # Test 2: Non-tool model
    print("\nTest 2: Non-tool model (gemma3:4b)")
    result = checker.check_compatibility("gemma3:4b", requires_tools=True)
    if not result.compatible:
        print("  ✅ gemma3:4b correctly flagged as incompatible for tools")
        print(f"     Issues: {[str(i) for i in result.issues]}")
        passed += 1
    else:
        print("  ❌ gemma3:4b should be flagged as incompatible")
        failed += 1

    # Test 3: Vision check
    print("\nTest 3: Vision capability (gpt-4o vs llama3)")
    result_gpt = checker.check_compatibility("gpt-4o", requires_vision=True)
    result_llama = checker.check_compatibility("llama3:8b", requires_vision=True)

    if result_gpt.compatible and not result_llama.compatible:
        print("  ✅ Vision check: gpt-4o=yes, llama3=no")
        passed += 1
    else:
        print("  ❌ Vision check failed")
        failed += 1

    # Test 4: Context window check
    print("\nTest 4: Context window check")
    result = checker.check_compatibility("llama3:8b", min_context_window=100000)
    if not result.compatible:
        print("  ✅ Correctly flagged insufficient context window")
        passed += 1
    else:
        print("  ❌ Should flag context window issue")
        failed += 1

    # Test 5: Model suggestions
    print("\nTest 5: Alternative model suggestions")
    suggestions = checker.suggest_alternative_models(
        required_capabilities={Capability.TOOLS, Capability.VISION},
        exclude_providers={"ollama"},
    )
    if len(suggestions) > 0 and all(
        Capability.TOOLS in s.capabilities and Capability.VISION in s.capabilities
        for s in suggestions
    ):
        print(f"  ✅ Found {len(suggestions)} alternatives: {[s.name for s in suggestions]}")
        passed += 1
    else:
        print("  ❌ Failed to find suitable alternatives")
        failed += 1

    # Test 6: Convenience function
    print("\nTest 6: Convenience function")
    if model_supports_tools("gpt-4o") and not model_supports_tools("gemma3:4b"):
        print("  ✅ model_supports_tools() works correctly")
        passed += 1
    else:
        print("  ❌ Convenience function failed")
        failed += 1

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("All tests passed! ✅")
    else:
        print("Some tests failed ❌")
        exit(1)
