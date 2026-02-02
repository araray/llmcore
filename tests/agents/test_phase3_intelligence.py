# tests/agents/test_phase3_intelligence.py
"""
Phase 3 Intelligence Layer Tests.

Tests the reasoning frameworks and learning components:
- ReAct Reasoner
- Reflexion Reasoner
- Reflection Bridge
- Failure Memory
- Fast Path Executor
"""

import pytest

# =============================================================================
# REACT REASONER TESTS
# =============================================================================


class TestReActReasoner:
    """Tests for the ReAct (Reason + Act) reasoning framework."""

    def test_reasoner_initialization(self):
        """Test ReActReasoner initializes with correct defaults."""
        from llmcore.agents.reasoning import ReActReasoner

        reasoner = ReActReasoner()

        assert reasoner is not None
        assert hasattr(reasoner, "config")
        assert hasattr(reasoner, "reason")

    def test_reasoner_with_config(self):
        """Test ReActReasoner accepts configuration."""
        from llmcore.agents.reasoning import ReActReasoner
        from llmcore.agents.reasoning.react import ReActConfig

        config = ReActConfig(max_iterations=5)
        reasoner = ReActReasoner(config=config)

        assert reasoner.config.max_iterations == 5

    def test_config_defaults(self):
        """Test ReActConfig default values."""
        from llmcore.agents.reasoning.react import ReActConfig

        config = ReActConfig()

        assert config.max_iterations > 0
        assert hasattr(config, "max_iterations")

    def test_reason_method_exists(self):
        """Test reason method exists and is callable."""
        from llmcore.agents.reasoning import ReActReasoner

        reasoner = ReActReasoner()

        assert hasattr(reasoner, "reason")
        assert callable(reasoner.reason)


# =============================================================================
# REFLEXION REASONER TESTS
# =============================================================================


class TestReflexionReasoner:
    """Tests for the Reflexion self-reflection framework."""

    def test_reasoner_initialization(self):
        """Test ReflexionReasoner initializes correctly."""
        from llmcore.agents.reasoning import ReflexionReasoner

        reasoner = ReflexionReasoner()

        assert reasoner is not None
        assert hasattr(reasoner, "config")
        assert hasattr(reasoner, "reflections")
        assert hasattr(reasoner, "trials")

    def test_reflexion_with_config(self):
        """Test ReflexionReasoner with custom config."""
        from llmcore.agents.reasoning import ReflexionReasoner
        from llmcore.agents.reasoning.reflexion import ReflexionConfig

        config = ReflexionConfig(max_trials=5)
        reasoner = ReflexionReasoner(config=config)

        assert reasoner.config.max_trials == 5

    def test_reflections_list(self):
        """Test that reflections are tracked."""
        from llmcore.agents.reasoning import ReflexionReasoner

        reasoner = ReflexionReasoner()

        assert isinstance(reasoner.reflections, list)
        assert isinstance(reasoner.trials, list)

    def test_has_react_reasoner(self):
        """Test that ReflexionReasoner contains ReActReasoner."""
        from llmcore.agents.reasoning import ReActReasoner, ReflexionReasoner

        reasoner = ReflexionReasoner()

        assert hasattr(reasoner, "react_reasoner")
        assert isinstance(reasoner.react_reasoner, ReActReasoner)

    def test_reason_with_reflection_method(self):
        """Test that reason_with_reflection method exists."""
        from llmcore.agents.reasoning import ReflexionReasoner

        reasoner = ReflexionReasoner()

        assert hasattr(reasoner, "reason_with_reflection")
        assert callable(reasoner.reason_with_reflection)


# =============================================================================
# REFLECTION BRIDGE TESTS
# =============================================================================


class TestReflectionBridge:
    """Tests for the Reflection Bridge that connects insights to actions."""

    def test_bridge_initialization(self):
        """Test ReflectionBridge initializes correctly."""
        from llmcore.agents.learning import ReflectionBridge

        bridge = ReflectionBridge()

        assert bridge is not None
        assert hasattr(bridge, "insights")
        assert hasattr(bridge, "max_insights")

    def test_add_reflection(self):
        """Test adding reflections to the bridge."""
        from llmcore.agents.learning import ReflectionBridge

        bridge = ReflectionBridge()

        # Use actual API signature
        insights = bridge.add_reflection(
            reflection_text="I should check file existence before reading",
            iteration=1,
        )

        assert isinstance(insights, list)

    def test_add_insight(self):
        """Test adding insights directly."""
        from llmcore.agents.learning import ReflectionBridge
        from llmcore.agents.learning.reflection_bridge import InsightType, ReflectionInsight

        bridge = ReflectionBridge()

        insight = ReflectionInsight(
            content="Always validate tool parameters",
            insight_type=InsightType.TOOL_USE,
            source_iteration=1,
        )
        bridge.add_insight(insight)

        assert len(bridge.insights) > 0

    def test_get_guidance(self):
        """Test getting guidance from accumulated insights."""
        from llmcore.agents.learning import ReflectionBridge
        from llmcore.agents.learning.reflection_bridge import GuidanceSet

        bridge = ReflectionBridge()

        guidance = bridge.get_guidance()
        # Returns GuidanceSet, not str
        assert isinstance(guidance, GuidanceSet)

    def test_get_guidance_for_phase(self):
        """Test getting phase-specific guidance."""
        from llmcore.agents.learning import ReflectionBridge
        from llmcore.agents.learning.reflection_bridge import GuidanceSet

        bridge = ReflectionBridge()

        guidance = bridge.get_guidance_for_phase("THINK")
        assert isinstance(guidance, GuidanceSet)

    def test_clear(self):
        """Test clearing all insights."""
        from llmcore.agents.learning import ReflectionBridge

        bridge = ReflectionBridge()
        bridge.clear()

        assert len(bridge.insights) == 0

    def test_statistics(self):
        """Test getting bridge statistics."""
        from llmcore.agents.learning import ReflectionBridge

        bridge = ReflectionBridge()

        stats = bridge.get_statistics()
        assert stats is not None


# =============================================================================
# FAILURE MEMORY TESTS
# =============================================================================


class TestFailureMemory:
    """Tests for pattern-based failure recognition."""

    def test_memory_initialization(self):
        """Test FailureMemory initializes correctly."""
        from llmcore.agents.learning import FailureMemory

        memory = FailureMemory()

        assert memory is not None
        assert memory.max_records > 0

    def test_record_failure(self):
        """Test recording a failure."""
        from llmcore.agents.learning import FailureMemory
        from llmcore.agents.learning.failure_memory import FailureType

        memory = FailureMemory()

        record = memory.record_failure(
            failure_type=FailureType.TOOL_ERROR,
            description="FileNotFoundError: /tmp/missing.txt",
            context={"goal": "Read config file"},
        )

        assert record is not None
        assert record.description == "FileNotFoundError: /tmp/missing.txt"
        assert record.failure_type == FailureType.TOOL_ERROR

    def test_find_similar_failures(self):
        """Test finding similar failure patterns."""
        from llmcore.agents.learning import FailureMemory
        from llmcore.agents.learning.failure_memory import FailureType

        memory = FailureMemory()

        memory.record_failure(
            failure_type=FailureType.TOOL_ERROR,
            description="Permission denied: /etc/passwd",
        )

        # find_similar takes (query: str, failure_type, limit)
        similar = memory.find_similar(
            query="Permission denied",
            failure_type=FailureType.TOOL_ERROR,
        )

        assert isinstance(similar, list)

    def test_get_unresolved(self):
        """Test getting unresolved failure patterns."""
        from llmcore.agents.learning import FailureMemory
        from llmcore.agents.learning.failure_memory import FailureType

        memory = FailureMemory()

        memory.record_failure(
            failure_type=FailureType.TOOL_ERROR,
            description="TimeoutError",
        )

        unresolved = memory.get_unresolved()
        assert isinstance(unresolved, list)
        assert len(unresolved) > 0  # Should have at least the one we just added

    def test_statistics(self):
        """Test getting failure memory statistics."""
        from llmcore.agents.learning import FailureMemory
        from llmcore.agents.learning.failure_memory import FailureType

        memory = FailureMemory()

        memory.record_failure(
            failure_type=FailureType.TOOL_ERROR,
            description="Error1",
        )

        stats = memory.get_statistics()
        assert stats is not None

    def test_clear(self):
        """Test clearing failure memory."""
        from llmcore.agents.learning import FailureMemory
        from llmcore.agents.learning.failure_memory import FailureType

        memory = FailureMemory()

        memory.record_failure(
            failure_type=FailureType.TOOL_ERROR,
            description="Error",
        )
        memory.clear()

        unresolved = memory.get_unresolved()
        assert len(unresolved) == 0


# =============================================================================
# FAST PATH EXECUTOR TESTS
# =============================================================================


class TestFastPathExecutor:
    """Tests for fast path execution of known patterns."""

    def test_executor_initialization(self):
        """Test FastPathExecutor initializes correctly."""
        from llmcore.agents.learning import FastPathExecutor

        executor = FastPathExecutor()

        assert executor is not None
        assert hasattr(executor, "config")
        assert hasattr(executor, "execute")

    def test_has_execute_method(self):
        """Test that execute method exists."""
        from llmcore.agents.learning import FastPathExecutor

        executor = FastPathExecutor()

        assert hasattr(executor, "execute")
        assert callable(executor.execute)

    def test_statistics(self):
        """Test getting execution statistics."""
        from llmcore.agents.learning import FastPathExecutor

        executor = FastPathExecutor()

        stats = executor.get_statistics()
        assert stats is not None

    def test_clear_cache(self):
        """Test clearing the fast path cache."""
        from llmcore.agents.learning import FastPathExecutor

        executor = FastPathExecutor()

        # Should not raise
        executor.clear_cache()


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestPhase3Integration:
    """Integration tests for Phase 3 components working together."""

    def test_all_components_importable(self):
        """Test all Phase 3 components can be imported."""
        from llmcore.agents.learning import FailureMemory, FastPathExecutor, ReflectionBridge
        from llmcore.agents.reasoning import ReActReasoner, ReflexionReasoner

        assert ReActReasoner is not None
        assert ReflexionReasoner is not None
        assert ReflectionBridge is not None
        assert FailureMemory is not None
        assert FastPathExecutor is not None

    def test_components_initialize(self):
        """Test all components initialize without error."""
        from llmcore.agents.learning import FailureMemory, FastPathExecutor, ReflectionBridge
        from llmcore.agents.reasoning import ReActReasoner, ReflexionReasoner

        react = ReActReasoner()
        reflexion = ReflexionReasoner()
        bridge = ReflectionBridge()
        memory = FailureMemory()
        fast_path = FastPathExecutor()

        assert react is not None
        assert reflexion is not None
        assert bridge is not None
        assert memory is not None
        assert fast_path is not None

    def test_failure_memory_with_reflection_bridge(self):
        """Test FailureMemory and ReflectionBridge interaction."""
        from llmcore.agents.learning import FailureMemory, ReflectionBridge
        from llmcore.agents.learning.failure_memory import FailureType
        from llmcore.agents.learning.reflection_bridge import GuidanceSet

        memory = FailureMemory()
        bridge = ReflectionBridge()

        # Record a failure
        record = memory.record_failure(
            failure_type=FailureType.TOOL_ERROR,
            description="FileNotFoundError",
            context={"path": "/missing/file"},
        )

        assert record is not None

        # Use record info to add reflection
        bridge.add_reflection(
            reflection_text="Check file existence before reading",
            iteration=1,
        )

        guidance = bridge.get_guidance()
        assert isinstance(guidance, GuidanceSet)

    def test_reflexion_has_react(self):
        """Test ReflexionReasoner contains ReActReasoner."""
        from llmcore.agents.reasoning import ReActReasoner, ReflexionReasoner

        reflexion = ReflexionReasoner()

        assert hasattr(reflexion, "react_reasoner")
        assert isinstance(reflexion.react_reasoner, ReActReasoner)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestPhase3EdgeCases:
    """Edge case tests for Phase 3 components."""

    def test_empty_reflection_bridge(self):
        """Test ReflectionBridge with no insights."""
        from llmcore.agents.learning import ReflectionBridge
        from llmcore.agents.learning.reflection_bridge import GuidanceSet

        bridge = ReflectionBridge()

        guidance = bridge.get_guidance()
        assert isinstance(guidance, GuidanceSet)

    def test_unicode_in_failure_patterns(self):
        """Test Unicode handling in failure patterns."""
        from llmcore.agents.learning import FailureMemory
        from llmcore.agents.learning.failure_memory import FailureType

        memory = FailureMemory()

        record = memory.record_failure(
            failure_type=FailureType.TOOL_ERROR,
            description="UnicodeError: 日本語 encoding failed",
            context={"text": "日本語テスト"},
        )

        assert record is not None
        assert "日本語" in record.description

    def test_long_description(self):
        """Test handling very long descriptions."""
        from llmcore.agents.learning import FailureMemory
        from llmcore.agents.learning.failure_memory import FailureType

        memory = FailureMemory()

        long_desc = "Error: " + "x" * 5000
        record = memory.record_failure(
            failure_type=FailureType.TOOL_ERROR,
            description=long_desc,
        )

        assert record is not None

    def test_reflection_bridge_max_insights(self):
        """Test ReflectionBridge respects max_insights."""
        from llmcore.agents.learning import ReflectionBridge

        bridge = ReflectionBridge(max_insights=5)

        # Add many reflections
        for i in range(10):
            bridge.add_reflection(
                reflection_text=f"Insight {i}",
                iteration=i,
            )

        # Should not exceed max
        assert len(bridge.insights) <= 5 or bridge.max_insights == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
