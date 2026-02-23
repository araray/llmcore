# tests/test_audit_sprint_1_3.py
"""
Tests for all Sprint 1-3 modules from the LLMCORE_AUDIT_REPORT.

Covers:
- autonomous/scheduler.py
- autonomous/context.py
- sessions/recovery.py
- storage/tiers/cached.py
- storage/tiers/persistent.py
- memory/volatile.py, session.py, semantic.py, episodic.py, hierarchical.py
- observability/tracing.py
- embedding/cohere.py, voyageai.py (import-level)
- __init__.py comprehensive exports
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# =============================================================================
# Sprint 1: autonomous/scheduler.py
# =============================================================================


class TestAutonomousScheduler:
    """Tests for the AutonomousScheduler module."""

    def test_import(self):
        """Verify the module imports correctly."""
        from llmcore.autonomous.scheduler import (
            AutonomousScheduler,
            HeartbeatManager,
            ScheduledTask,
            TaskPriority,
        )

        assert TaskPriority.LOW < TaskPriority.NORMAL < TaskPriority.HIGH < TaskPriority.CRITICAL

    def test_task_priority_ordering(self):
        """Verify priority enum has correct ordering."""
        from llmcore.autonomous.scheduler import TaskPriority

        assert TaskPriority.LOW == 1
        assert TaskPriority.NORMAL == 5
        assert TaskPriority.HIGH == 10
        assert TaskPriority.CRITICAL == 20

    def test_scheduled_task_dataclass(self):
        """Verify ScheduledTask creation."""
        from llmcore.autonomous.scheduler import ScheduledTask, TaskPriority

        task = ScheduledTask(
            name="test_task",
            interval_seconds=60.0,
            priority=TaskPriority.HIGH,
        )
        assert task.name == "test_task"
        assert task.interval_seconds == 60.0
        assert task.priority == TaskPriority.HIGH
        assert task.enabled is True
        assert task.depends_on == []
        assert task.resource_aware is True

    def test_scheduler_schedule_and_list(self):
        """Test scheduling tasks and listing them."""
        from llmcore.autonomous.scheduler import AutonomousScheduler, TaskPriority

        mock_hb = MagicMock()
        mock_hb.register_task = MagicMock()

        scheduler = AutonomousScheduler(heartbeat_manager=mock_hb)

        async def callback1():
            pass

        async def callback2():
            pass

        scheduler.schedule("low_task", callback1, 60, priority=TaskPriority.LOW)
        scheduler.schedule("high_task", callback2, 30, priority=TaskPriority.HIGH)

        tasks = scheduler.list_tasks()
        assert len(tasks) == 2
        assert tasks[0].name == "high_task"  # Sorted by priority desc
        assert tasks[1].name == "low_task"

    def test_scheduler_unschedule(self):
        """Test removing a scheduled task."""
        from llmcore.autonomous.scheduler import AutonomousScheduler

        mock_hb = MagicMock()
        scheduler = AutonomousScheduler(heartbeat_manager=mock_hb)

        async def cb():
            pass

        scheduler.schedule("temp", cb, 10)
        assert scheduler.get_task("temp") is not None

        removed = scheduler.unschedule("temp")
        assert removed is True
        assert scheduler.get_task("temp") is None

        removed_again = scheduler.unschedule("nonexistent")
        assert removed_again is False


# =============================================================================
# Sprint 1: autonomous/context.py
# =============================================================================


class TestAutonomousContext:
    """Tests for the autonomous context redirect."""

    def test_import_redirects(self):
        """All context imports should work via the autonomous namespace."""
        from llmcore.autonomous.context import (
            ContentPrioritizer,
            ContextChunk,
            ContextCompressor,
            ContextSource,
            ContextSynthesizer,
            EstimateCounter,
            SynthesizedContext,
            TiktokenCounter,
            TokenCounter,
        )

        # Verify they're the same classes as the canonical ones
        from llmcore.context import (
            ContextSynthesizer as CanonicalSynthesizer,
        )

        assert ContextSynthesizer is CanonicalSynthesizer


# =============================================================================
# Sprint 2: sessions/recovery.py
# =============================================================================


class TestSessionRecovery:
    """Tests for the session recovery system."""

    def test_checkpoint_status_enum(self):
        """Verify checkpoint status values."""
        from llmcore.sessions.recovery import CheckpointStatus

        assert CheckpointStatus.IN_PROGRESS == "in_progress"
        assert CheckpointStatus.COMPLETED == "completed"
        assert CheckpointStatus.FAILED == "failed"
        assert CheckpointStatus.RECOVERING == "recovering"

    def test_checkpoint_creation(self):
        """Test RecoveryCheckpoint data class."""
        from llmcore.sessions.recovery import CheckpointStatus, RecoveryCheckpoint

        cp = RecoveryCheckpoint(
            agent_id="agent-1",
            session_id="sess-42",
            phase="THINK",
            turn_index=5,
        )
        assert cp.agent_id == "agent-1"
        assert cp.session_id == "sess-42"
        assert cp.phase == "THINK"
        assert cp.turn_index == 5
        assert cp.status == CheckpointStatus.IN_PROGRESS
        assert isinstance(cp.checkpoint_id, str)
        assert cp.created_at > 0

    def test_checkpoint_serialization(self):
        """Test checkpoint to_dict / from_dict round-trip."""
        from llmcore.sessions.recovery import CheckpointStatus, RecoveryCheckpoint

        cp = RecoveryCheckpoint(
            agent_id="a1",
            session_id="s1",
            phase="ACT",
            turn_index=3,
            goal_snapshot={"goal": "deploy"},
            pending_tool_calls=[{"tool": "bash", "args": {"cmd": "ls"}}],
        )

        data = cp.to_dict()
        assert data["agent_id"] == "a1"
        assert data["status"] == "in_progress"

        restored = RecoveryCheckpoint.from_dict(data)
        assert restored.agent_id == cp.agent_id
        assert restored.phase == cp.phase
        assert restored.goal_snapshot == {"goal": "deploy"}
        assert len(restored.pending_tool_calls) == 1

    def test_sessions_init_exports(self):
        """Verify sessions __init__.py exports recovery types."""
        from llmcore.sessions import (
            CheckpointStatus,
            RecoveryCheckpoint,
            SessionManager,
            SessionRecovery,
        )


# =============================================================================
# Sprint 3: storage/tiers/cached.py
# =============================================================================


class TestCachedStorageTier:
    """Tests for the cached storage tier."""

    def test_config_defaults(self):
        """Verify CachedStorageConfig defaults."""
        from llmcore.storage.tiers.cached import CachedStorageConfig

        config = CachedStorageConfig()
        assert config.enabled is True
        assert config.max_items == 50_000
        assert config.default_ttl_seconds == 86400.0
        assert config.enable_stats is True

    def test_tier_creation(self):
        """Verify tier can be instantiated."""
        from llmcore.storage.tiers.cached import CachedStorageConfig, CachedStorageTier

        tier = CachedStorageTier(CachedStorageConfig(db_path="/tmp/test_cache.db"))
        assert tier._db is None  # Not initialized yet
        assert tier._stats["hits"] == 0

    def test_stats_default(self):
        """Verify stats returns correct structure."""
        from llmcore.storage.tiers.cached import CachedStorageTier

        tier = CachedStorageTier()
        stats = tier.stats()
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert stats["hit_rate"] == 0.0

    def test_factory_function(self):
        """Test the create_cached_tier factory."""
        from llmcore.storage.tiers.cached import CachedStorageConfig, create_cached_tier

        tier = create_cached_tier(CachedStorageConfig(max_items=100))
        assert tier._config.max_items == 100


# =============================================================================
# Sprint 3: storage/tiers/persistent.py
# =============================================================================


class TestPersistentStorageTier:
    """Tests for the persistent storage tier."""

    def test_config_defaults(self):
        """Verify PersistentStorageConfig defaults."""
        from llmcore.storage.tiers.persistent import PersistentStorageConfig

        config = PersistentStorageConfig()
        assert config.backend == "sqlite"
        assert config.table_name == "persistent_kv"
        assert config.enable_compression is False

    def test_compression_helpers(self):
        """Test compress/decompress round-trip."""
        from llmcore.storage.tiers.persistent import PersistentStorageTier

        original = "Hello, world! " * 100
        compressed = PersistentStorageTier._compress(original)
        assert compressed != original
        decompressed = PersistentStorageTier._decompress(compressed)
        assert decompressed == original

    def test_factory_function(self):
        """Test the create_persistent_tier factory."""
        from llmcore.storage.tiers.persistent import (
            PersistentStorageConfig,
            create_persistent_tier,
        )

        tier = create_persistent_tier(PersistentStorageConfig(backend="sqlite"))
        assert tier._config.backend == "sqlite"


# =============================================================================
# Sprint 3: storage/tiers/__init__.py
# =============================================================================


class TestStorageTiersInit:
    """Test that all three tiers are exported."""

    def test_all_tiers_importable(self):
        from llmcore.storage.tiers import (
            CachedStorageConfig,
            CachedStorageTier,
            PersistentStorageConfig,
            PersistentStorageTier,
            VolatileMemoryConfig,
            VolatileMemoryTier,
            create_cached_tier,
            create_persistent_tier,
            create_volatile_tier,
        )


# =============================================================================
# Sprint 3: Memory tier files
# =============================================================================


class TestMemoryTierModules:
    """Tests for the spec-mandated memory tier files."""

    def test_volatile_redirect(self):
        from llmcore.memory.volatile import VolatileMemoryConfig, VolatileMemoryTier

    def test_session_redirect(self):
        from llmcore.memory.session import ChatSession, Message, Role, SessionManager

    def test_semantic_memory(self):
        from llmcore.memory.semantic import SemanticMemory, SemanticResult

        result = SemanticResult(content="test", score=0.95, source="doc:1")
        assert result.content == "test"
        assert result.score == 0.95

    def test_episodic_memory(self):
        from llmcore.memory.episodic import Episode, EpisodeRecord, EpisodeType, EpisodicMemory

    def test_hierarchical_memory(self):
        from llmcore.memory.hierarchical import (
            HierarchicalMemoryManager,
            MemoryItem,
            MemoryTier,
        )

        assert MemoryTier.VOLATILE == "volatile"
        assert MemoryTier.SESSION == "session"
        assert MemoryTier.SEMANTIC == "semantic"
        assert MemoryTier.EPISODIC == "episodic"

        item = MemoryItem(key="k1", content="hello", tier=MemoryTier.VOLATILE, score=1.0)
        assert item.key == "k1"

    def test_memory_init_exports(self):
        from llmcore.memory import (
            HierarchicalMemoryManager,
            MemoryItem,
            MemoryManager,
            MemoryTier,
        )


# =============================================================================
# Sprint 3: observability/tracing.py
# =============================================================================


class TestObservabilityTracing:
    """Tests for the consolidated tracing module."""

    def test_imports(self):
        from llmcore.observability.tracing import (
            configure_tracer,
            get_tracer,
            trace_agent_phase,
            trace_llm_call,
            trace_span,
        )

    def test_get_tracer_noop(self):
        """get_tracer should return a no-op when OTel is not configured."""
        from llmcore.observability.tracing import get_tracer

        tracer = get_tracer("test")
        # Should not raise
        span = tracer.start_as_current_span("test_span")
        span.set_attribute("key", "value")
        span.end()

    def test_trace_span_context_manager(self):
        """trace_span should work as a context manager."""
        from llmcore.observability.tracing import trace_span

        with trace_span("test_operation", {"key": "val"}) as span:
            span.set_attribute("extra", "data")
            # Should not raise

    def test_trace_llm_call_noop(self):
        """trace_llm_call should not raise when OTel is absent."""
        from llmcore.observability.tracing import trace_llm_call

        trace_llm_call(
            provider="openai",
            model="gpt-4o",
            operation="chat",
            input_tokens=100,
            output_tokens=50,
            duration_ms=1500.0,
        )

    def test_trace_agent_phase_noop(self):
        """trace_agent_phase should not raise when OTel is absent."""
        from llmcore.observability.tracing import trace_agent_phase

        trace_agent_phase(
            phase="THINK",
            agent_id="test-agent",
            iteration=3,
            duration_ms=500.0,
        )


# =============================================================================
# Sprint 2: Embedding providers
# =============================================================================


class TestEmbeddingProviders:
    """Tests for Cohere and VoyageAI embedding providers."""

    def test_cohere_import(self):
        """CohereEmbedding class should be importable."""
        try:
            from llmcore.embedding.cohere import CohereEmbedding

            assert CohereEmbedding is not None
        except ImportError as e:
            if "cohere" in str(e).lower():
                pytest.skip("cohere SDK not installed")
            raise

    def test_voyageai_import(self):
        """VoyageAIEmbedding class should be importable."""
        try:
            from llmcore.embedding.voyageai import VoyageAIEmbedding

            assert VoyageAIEmbedding is not None
        except ImportError as e:
            if "voyageai" in str(e).lower():
                pytest.skip("voyageai SDK not installed")
            raise

    def test_lazy_provider_map(self):
        """Embedding manager should have lazy entries for new providers."""
        from llmcore.embedding.manager import _LAZY_PROVIDER_MAP

        assert "cohere" in _LAZY_PROVIDER_MAP
        assert "voyageai" in _LAZY_PROVIDER_MAP
        assert "voyage" in _LAZY_PROVIDER_MAP  # alias

    def test_embedding_init_exports(self):
        """Embedding __init__ should list new providers."""
        from llmcore.embedding import __all__ as exports

        assert "CohereEmbedding" in exports
        assert "VoyageAIEmbedding" in exports


# =============================================================================
# Top-level __init__.py comprehensive exports
# =============================================================================


class TestTopLevelExports:
    """Validate the comprehensive __init__.py exports."""

    def test_core_api(self):
        from llmcore import LLMCore

    def test_providers(self):
        from llmcore import BaseProvider, ProviderManager

    def test_agents(self):
        from llmcore import (
            AgentManager,
            AgentMode,
            CognitiveCycle,
            CognitivePhase,
            EnhancedAgentManager,
            PersonaManager,
            ToolManager,
        )

    def test_hitl(self):
        from llmcore import HITLConfig, HITLManager, RiskAssessor, RiskLevel

    def test_autonomous(self):
        from llmcore import (
            AutonomousScheduler,
            AutonomousState,
            EscalationManager,
            Goal,
            GoalManager,
            HeartbeatManager,
            ResourceMonitor,
            Skill,
            SkillLoader,
            StateManager,
            TaskPriority,
        )

    def test_context(self):
        from llmcore import (
            ContentPrioritizer,
            ContextChunk,
            ContextCompressor,
            ContextSource,
            ContextSynthesizer,
            SynthesizedContext,
        )

    def test_memory_and_sessions(self):
        from llmcore import MemoryManager, SessionManager

    def test_observability(self):
        from llmcore import CostAnalyzer, CostTracker, MetricsRegistry, ObservabilityLogger

    def test_all_count(self):
        """Should export a comprehensive set of symbols."""
        import llmcore

        assert len(llmcore.__all__) >= 150, f"Only {len(llmcore.__all__)} exports, expected >= 150"

    def test_no_duplicates_in_all(self):
        """__all__ should not contain duplicate entries."""
        import llmcore

        seen = set()
        duplicates = []
        for name in llmcore.__all__:
            if name in seen:
                duplicates.append(name)
            seen.add(name)
        assert duplicates == [], f"Duplicate exports: {duplicates}"
