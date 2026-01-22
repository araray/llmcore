# tests/agents/test_phase1_fixes.py
"""
Phase 1 Foundation Tests.

Tests the critical bug fixes and new infrastructure:
- BUG-001: Tool rendering fix
- BUG-002: Progress extraction fix
- Goal Classifier
- Circuit Breaker
- RAG Filter
"""

import re
from typing import Any, Dict, List

import pytest

# =============================================================================
# BUG-001 TESTS: Tool Rendering
# =============================================================================


class TestToolRendering:
    """Tests for the tool rendering fix in think.py."""

    def test_format_tools_direct_format(self):
        """Test formatting tools in direct/Pydantic format (name at top level)."""
        from llmcore.agents.cognitive.phases.think import _format_tools

        tools = [
            {"name": "search", "description": "Search the knowledge base", "parameters": {}},
            {"name": "calculator", "description": "Perform calculations", "parameters": {}},
        ]

        result = _format_tools(tools)

        assert "search" in result
        assert "calculator" in result
        assert "Search the knowledge base" in result
        assert "Perform calculations" in result
        assert "unknown" not in result

    def test_format_tools_openai_format(self):
        """Test formatting tools in OpenAI function-calling format."""
        from llmcore.agents.cognitive.phases.think import _format_tools

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                },
            },
        ]

        result = _format_tools(tools)

        assert "web_search" in result
        assert "Search the web" in result
        assert "unknown" not in result

    def test_format_tools_with_parameters(self):
        """Test that parameter names are included in tool description."""
        from llmcore.agents.cognitive.phases.think import _format_tools

        tools = [
            {
                "name": "file_read",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "encoding": {"type": "string"},
                    },
                },
            },
        ]

        result = _format_tools(tools)

        assert "file_read" in result
        assert "params:" in result
        assert "path" in result

    def test_format_tools_empty_list(self):
        """Test formatting empty tool list."""
        from llmcore.agents.cognitive.phases.think import _format_tools

        result = _format_tools([])

        assert result == "No tools available."

    def test_format_tools_mixed_formats(self):
        """Test formatting a mix of direct and OpenAI formats."""
        from llmcore.agents.cognitive.phases.think import _format_tools

        tools = [
            {"name": "tool1", "description": "Direct format", "parameters": {}},
            {
                "type": "function",
                "function": {"name": "tool2", "description": "OpenAI format", "parameters": {}},
            },
        ]

        result = _format_tools(tools)

        assert "tool1" in result
        assert "tool2" in result
        assert "Direct format" in result
        assert "OpenAI format" in result


# =============================================================================
# BUG-002 TESTS: Progress Extraction
# =============================================================================


class TestProgressExtraction:
    """Tests for the progress extraction fix in reflect.py."""

    def test_extract_progress_labeled(self):
        """Test extracting progress from labeled format."""
        from llmcore.agents.cognitive.phases.reflect import _extract_progress

        test_cases = [
            ("PROGRESS: 75%", 0.75),
            ("progress: 50%", 0.50),
            ("PROGRESS: 100%", 1.00),
            ("progress: 0%", 0.00),
            ("PROGRESS: 25 percent", 0.25),
        ]

        for text, expected in test_cases:
            result = _extract_progress(text)
            assert abs(result - expected) < 0.05, (
                f"Failed for '{text}': got {result}, expected {expected}"
            )

    def test_extract_progress_xml_format(self):
        """Test extracting progress from XML format."""
        from llmcore.agents.cognitive.phases.reflect import _extract_progress

        test_cases = [
            ("<progress>75</progress>", 0.75),
            ("<progress>50</progress>", 0.50),
            ("<PROGRESS>100</PROGRESS>", 1.00),
        ]

        for text, expected in test_cases:
            result = _extract_progress(text)
            assert abs(result - expected) < 0.05, (
                f"Failed for '{text}': got {result}, expected {expected}"
            )

    def test_extract_progress_natural_language(self):
        """Test extracting progress from natural language."""
        from llmcore.agents.cognitive.phases.reflect import _extract_progress

        test_cases = [
            ("I am 75% complete with the task", 0.75),
            ("Task is 50% done", 0.50),
            ("Approximately 25% progress", 0.25),
            ("We are 80% of the way there", 0.80),
        ]

        for text, expected in test_cases:
            result = _extract_progress(text)
            assert abs(result - expected) < 0.1, (
                f"Failed for '{text}': got {result}, expected {expected}"
            )

    def test_extract_progress_fraction_format(self):
        """Test extracting progress from fraction format."""
        from llmcore.agents.cognitive.phases.reflect import _extract_progress

        test_cases = [
            ("Step 3 of 4", 0.75),
            ("Completed 2 of 5 tasks", 0.40),
            ("step 1/4 done", 0.25),
        ]

        for text, expected in test_cases:
            result = _extract_progress(text)
            assert abs(result - expected) < 0.1, (
                f"Failed for '{text}': got {result}, expected {expected}"
            )

    def test_extract_progress_estimation(self):
        """Test progress estimation from content when no explicit value."""
        from llmcore.agents.cognitive.phases.reflect import _extract_progress

        # High progress indicators
        assert _extract_progress("Task completed successfully") > 0.7
        assert _extract_progress("I have finished the analysis") > 0.7

        # Low progress indicators
        assert _extract_progress("Just getting started") < 0.3
        assert _extract_progress("Beginning the first step") < 0.3

        # Blocked/error indicators
        assert _extract_progress("I am stuck and cannot proceed") < 0.2

    def test_extract_progress_no_match_default(self):
        """Test default value when no progress indicators found."""
        from llmcore.agents.cognitive.phases.reflect import _extract_progress

        result = _extract_progress("Some random text without progress info")

        # Should return a moderate default, not 0.5 hardcoded
        assert 0.2 < result < 0.5


# =============================================================================
# GOAL CLASSIFIER TESTS
# =============================================================================


class TestGoalClassifier:
    """Tests for the goal classification system."""

    @pytest.fixture
    def classifier(self):
        from llmcore.agents.cognitive.goal_classifier import GoalClassifier

        return GoalClassifier()

    def test_trivial_greetings(self, classifier):
        """Test that greetings are classified as trivial."""
        from llmcore.agents.cognitive.goal_classifier import GoalComplexity

        trivial_goals = ["hello", "hi", "hey", "good morning", "thanks", "bye"]

        for goal in trivial_goals:
            result = classifier.classify(goal)
            assert result.complexity == GoalComplexity.TRIVIAL, f"'{goal}' should be TRIVIAL"
            assert not result.requires_tools, f"'{goal}' should not require tools"

    def test_simple_tasks(self, classifier):
        """Test that simple tasks are classified correctly."""
        from llmcore.agents.cognitive.goal_classifier import GoalComplexity

        simple_goals = [
            "read file config.yaml",
            "list the files in /var/log",
            "what is 2+2",
        ]

        for goal in simple_goals:
            result = classifier.classify(goal)
            assert result.complexity == GoalComplexity.SIMPLE, f"'{goal}' should be SIMPLE"

    def test_complex_tasks(self, classifier):
        """Test that complex tasks are classified correctly."""
        from llmcore.agents.cognitive.goal_classifier import GoalComplexity

        complex_goals = [
            "analyze the sales data and create a comprehensive report with visualizations",
            "build a web scraper application",
        ]

        for goal in complex_goals:
            result = classifier.classify(goal)
            assert result.complexity == GoalComplexity.COMPLEX, f"'{goal}' should be COMPLEX"

    def test_ambiguous_tasks(self, classifier):
        """Test that ambiguous tasks are classified correctly."""
        from llmcore.agents.cognitive.goal_classifier import GoalComplexity

        ambiguous_goals = ["do it", "fix that", "handle the thing"]

        for goal in ambiguous_goals:
            result = classifier.classify(goal)
            assert result.complexity == GoalComplexity.AMBIGUOUS, f"'{goal}' should be AMBIGUOUS"
            assert result.clarification_needed
            assert len(result.clarification_questions) > 0

    def test_execution_strategy_mapping(self, classifier):
        """Test that complexity maps to appropriate execution strategy."""
        from llmcore.agents.cognitive.goal_classifier import ExecutionStrategy, GoalComplexity

        # Trivial -> DIRECT
        result = classifier.classify("hello")
        assert result.suggested_strategy == ExecutionStrategy.DIRECT

        # Complex -> HIERARCHICAL or similar
        result = classifier.classify("analyze data and create comprehensive report")
        assert result.suggested_strategy in [
            ExecutionStrategy.HIERARCHICAL,
            ExecutionStrategy.REFLEXION,
        ]

    def test_classification_confidence(self, classifier):
        """Test that high-confidence patterns have high confidence scores."""
        result = classifier.classify("hello")
        assert result.confidence >= 0.9

        result = classifier.classify("do something")
        assert result.confidence < 0.9


# =============================================================================
# CIRCUIT BREAKER TESTS
# =============================================================================


class TestCircuitBreaker:
    """Tests for the circuit breaker system."""

    def test_max_iterations(self):
        """Test that circuit breaker trips at max iterations."""
        from llmcore.agents.resilience.circuit_breaker import AgentCircuitBreaker, TripReason

        breaker = AgentCircuitBreaker(max_iterations=5)
        breaker.start()

        for i in range(5):
            result = breaker.check(iteration=i, progress=0.1 * i)
            if i < 5:
                assert not result.tripped

        result = breaker.check(iteration=5, progress=0.5)
        assert result.tripped
        assert result.reason == TripReason.MAX_ITERATIONS

    def test_repeated_errors(self):
        """Test that circuit breaker trips on repeated identical errors."""
        from llmcore.agents.resilience.circuit_breaker import AgentCircuitBreaker, TripReason

        breaker = AgentCircuitBreaker(max_same_errors=3, max_iterations=100)
        breaker.start()

        same_error = "Model does not support tools"

        for i in range(2):
            result = breaker.check(iteration=i, progress=0.0, error=same_error)
            assert not result.tripped

        result = breaker.check(iteration=2, progress=0.0, error=same_error)
        assert result.tripped
        assert result.reason == TripReason.REPEATED_ERROR

    def test_progress_stall(self):
        """Test that circuit breaker trips on progress stall."""
        from llmcore.agents.resilience.circuit_breaker import AgentCircuitBreaker, TripReason

        breaker = AgentCircuitBreaker(progress_stall_threshold=3, max_iterations=100)
        breaker.start()

        # Same progress for multiple iterations
        for i in range(5):
            result = breaker.check(iteration=i, progress=0.5)
            if result.tripped:
                assert result.reason == TripReason.PROGRESS_STALL
                break
        else:
            pytest.fail("Circuit breaker should have tripped for progress stall")

    def test_cost_limit(self):
        """Test that circuit breaker trips at cost limit."""
        from llmcore.agents.resilience.circuit_breaker import AgentCircuitBreaker, TripReason

        breaker = AgentCircuitBreaker(max_total_cost=0.05, max_iterations=100)
        breaker.start()

        for i in range(10):
            result = breaker.check(iteration=i, progress=0.1 * i, cost=0.02)
            if result.tripped:
                assert result.reason == TripReason.COST_LIMIT
                break
        else:
            pytest.fail("Circuit breaker should have tripped for cost limit")

    def test_normal_completion(self):
        """Test that circuit breaker doesn't trip during normal operation."""
        from llmcore.agents.resilience.circuit_breaker import AgentCircuitBreaker

        breaker = AgentCircuitBreaker(max_iterations=10)
        breaker.start()

        # Normal progress with no errors
        for i in range(5):
            result = breaker.check(iteration=i, progress=0.2 * i)
            assert not result.tripped

    def test_reset(self):
        """Test that circuit breaker can be reset."""
        from llmcore.agents.resilience.circuit_breaker import AgentCircuitBreaker

        breaker = AgentCircuitBreaker(max_iterations=3)
        breaker.start()

        # Trip the breaker
        for i in range(4):
            breaker.check(iteration=i, progress=0.0)

        assert breaker.is_tripped

        # Reset
        breaker.reset()
        assert not breaker.is_tripped

    def test_factory_presets(self):
        """Test factory preset configurations."""
        from llmcore.agents.resilience.circuit_breaker import create_circuit_breaker

        strict = create_circuit_breaker("strict")
        assert strict.config.max_iterations == 5

        permissive = create_circuit_breaker("permissive")
        assert permissive.config.max_iterations == 50

        # Test override
        custom = create_circuit_breaker("default", max_iterations=25)
        assert custom.config.max_iterations == 25


# =============================================================================
# RAG FILTER TESTS
# =============================================================================


class TestRAGFilter:
    """Tests for the RAG context filter."""

    @pytest.fixture
    def rag_filter(self):
        from llmcore.agents.context.rag_filter import RAGContextFilter

        return RAGContextFilter(min_similarity=0.7, max_results=5)

    def test_filter_low_similarity(self, rag_filter):
        """Test that low-similarity results are filtered."""
        from llmcore.agents.context.rag_filter import RAGResult

        results = [
            RAGResult(content="Highly relevant content here", similarity=0.9),
            RAGResult(content="Somewhat relevant content", similarity=0.75),
            RAGResult(content="Barely relevant content", similarity=0.5),  # Below threshold
            RAGResult(content="Not very relevant stuff", similarity=0.3),  # Below threshold
        ]

        filtered = rag_filter.filter(results)

        assert len(filtered) == 2
        assert all(r.similarity >= 0.7 for r in filtered)

    def test_filter_garbage_content(self, rag_filter):
        """Test that garbage/placeholder content is filtered."""
        from llmcore.agents.context.rag_filter import RAGResult

        results = [
            RAGResult(content="Valid meaningful content here", similarity=0.9),
            RAGResult(content="Document 1", similarity=0.9),  # Garbage pattern
            RAGResult(content="TODO: implement this", similarity=0.9),  # Garbage pattern
            RAGResult(content="Lorem ipsum dolor sit amet", similarity=0.9),  # Garbage
        ]

        filtered = rag_filter.filter(results)

        assert len(filtered) == 1
        assert "Valid meaningful content" in filtered[0].content

    def test_filter_duplicates(self, rag_filter):
        """Test that near-duplicate results are deduplicated."""
        from llmcore.agents.context.rag_filter import RAGResult

        results = [
            RAGResult(content="The quick brown fox jumps over the lazy dog", similarity=0.9),
            RAGResult(
                content="The quick brown fox jumps over the lazy dog.", similarity=0.85
            ),  # Near duplicate
            RAGResult(content="Something completely different here", similarity=0.8),
        ]

        filtered = rag_filter.filter(results)

        # Should keep first of duplicates and the different one
        assert len(filtered) == 2

    def test_filter_max_results(self, rag_filter):
        """Test that results are limited to max_results."""
        from llmcore.agents.context.rag_filter import RAGResult

        results = [
            RAGResult(content=f"Unique content number {i}", similarity=0.9 - i * 0.01)
            for i in range(10)
        ]

        filtered = rag_filter.filter(results)

        assert len(filtered) <= 5

    def test_filter_stats(self, rag_filter):
        """Test that filter stats are tracked correctly."""
        from llmcore.agents.context.rag_filter import RAGResult

        results = [
            RAGResult(content="Valid meaningful content here", similarity=0.9),
            RAGResult(content="Too short", similarity=0.9),  # Too short
            RAGResult(content="Low similarity content here", similarity=0.5),  # Low similarity
        ]

        filtered, stats = rag_filter.filter_with_stats(results)

        assert stats.input_count == 3
        assert stats.output_count <= stats.input_count


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestPhase1Integration:
    """Integration tests for Phase 1 components working together."""

    def test_imports(self):
        """Test that all new components can be imported."""
        # Goal classifier
        # Circuit breaker
        # RAG filter
        # Capability checker
        from llmcore.agents import (
            AgentCircuitBreaker,
            Capability,
            CapabilityChecker,
            CircuitBreakerConfig,
            CompatibilityResult,
            FilterStats,
            GoalClassification,
            GoalClassifier,
            GoalComplexity,
            RAGContextFilter,
            RAGResult,
            TripReason,
            classify_goal,
            create_circuit_breaker,
            is_trivial_goal,
        )

    def test_goal_classifier_with_circuit_breaker(self):
        """Test goal classifier determining circuit breaker settings."""
        from llmcore.agents import AgentCircuitBreaker, GoalClassifier, GoalComplexity

        classifier = GoalClassifier()

        # Trivial goal -> minimal iterations
        result = classifier.classify("hello")
        assert result.complexity == GoalComplexity.TRIVIAL

        breaker = AgentCircuitBreaker(max_iterations=result.max_iterations)
        assert breaker.config.max_iterations <= 3  # Trivial tasks need few iterations

        # Complex goal -> more iterations allowed
        result = classifier.classify("analyze data and create report")
        breaker = AgentCircuitBreaker(max_iterations=result.max_iterations)
        assert breaker.config.max_iterations >= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
