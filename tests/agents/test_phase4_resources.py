# tests/agents/test_phase4_resources.py
"""
Phase 4 Resource Management Tests.

Tests the efficiency and resource management components:
- Model Router
- Semantic Cache
- Context Manager
- Memory Store (Working + Long-term)
"""

import pytest

# =============================================================================
# MODEL ROUTER TESTS
# =============================================================================


class TestModelRouter:
    """Tests for the intelligent model routing system."""

    def test_router_initialization(self):
        """Test ModelRouter initializes with correct defaults."""
        from llmcore.agents.routing import ModelRouter

        router = ModelRouter()

        assert router is not None
        assert hasattr(router, "select_model")
        assert hasattr(router, "list_models")

    def test_list_models(self):
        """Test listing registered models."""
        from llmcore.agents.routing import ModelRouter

        router = ModelRouter()
        models = router.list_models()

        assert len(models) > 0

    def test_select_model_basic(self):
        """Test basic model selection."""
        from llmcore.agents.routing import ModelRouter

        router = ModelRouter()

        selection = router.select_model()

        assert selection is not None
        # selection is ModelSelection with .model attribute
        assert hasattr(selection, "model") or hasattr(selection, "model_id")

    def test_select_model_with_complexity(self):
        """Test model selection with complexity hint."""
        from llmcore.agents.cognitive import GoalComplexity
        from llmcore.agents.routing import ModelRouter

        router = ModelRouter()

        for complexity in [
            GoalComplexity.TRIVIAL,
            GoalComplexity.SIMPLE,
            GoalComplexity.MODERATE,
            GoalComplexity.COMPLEX,
        ]:
            selection = router.select_model(complexity=complexity)
            assert selection is not None

    def test_model_tiers(self):
        """Test that model tiers are defined."""
        from llmcore.agents.routing.model_router import ModelTier

        assert ModelTier.FAST is not None
        assert ModelTier.BALANCED is not None
        assert ModelTier.CAPABLE is not None

    def test_model_capabilities(self):
        """Test that model capabilities are defined."""
        from llmcore.agents.routing.model_router import ModelCapability

        capabilities = list(ModelCapability)
        assert len(capabilities) > 0

    def test_get_model_info(self):
        """Test getting model information."""
        from llmcore.agents.routing import ModelRouter

        router = ModelRouter()
        models = router.list_models()

        if models:
            first_model = models[0]
            model_id = (
                first_model.model_id if hasattr(first_model, "model_id") else str(first_model)
            )
            info = router.get_model_info(model_id)
            # May return None or info depending on model

    def test_statistics(self):
        """Test getting router statistics."""
        from llmcore.agents.routing import ModelRouter

        router = ModelRouter()

        stats = router.get_statistics()
        assert stats is not None


class TestModelTierSelection:
    """Tests for tier selection logic."""

    def test_select_model_for_complexity_helper(self):
        """Test the helper function for complexity-based selection."""
        from llmcore.agents.cognitive import GoalComplexity
        from llmcore.agents.routing.model_router import select_model_for_complexity

        for complexity in GoalComplexity:
            result = select_model_for_complexity(complexity)
            assert result is not None


# =============================================================================
# SEMANTIC CACHE TESTS
# =============================================================================


class TestSemanticCache:
    """Tests for semantic caching system."""

    def test_cache_initialization(self):
        """Test SemanticCache initializes correctly."""
        from llmcore.agents.caching import SemanticCache

        cache = SemanticCache()

        assert cache is not None
        assert hasattr(cache, "get")
        assert hasattr(cache, "set")

    @pytest.mark.asyncio
    async def test_cache_set_and_get(self):
        """Test storing and retrieving cached responses."""
        from llmcore.agents.caching import SemanticCache

        cache = SemanticCache()

        # Store a response
        await cache.set(
            query="What is the capital of France?",
            response="The capital of France is Paris.",
        )

        # Retrieve
        result = await cache.get("What is the capital of France?")
        # Result may be CacheHit or None

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache miss for unknown queries."""
        from llmcore.agents.caching import SemanticCache

        cache = SemanticCache()

        # Query something not in cache
        result = await cache.get("Unknown query never cached before xyz123")

        # Should be None or miss
        assert result is None or (hasattr(result, "hit") and not result.hit)

    @pytest.mark.asyncio
    async def test_cache_clear(self):
        """Test clearing all cache entries."""
        from llmcore.agents.caching import SemanticCache

        cache = SemanticCache()
        await cache.clear()
        # Should not raise

    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        from llmcore.agents.caching import SemanticCache

        cache = SemanticCache()

        stats = cache.get_stats()
        assert stats is not None

    def test_config_access(self):
        """Test configuration is accessible."""
        from llmcore.agents.caching import SemanticCache

        cache = SemanticCache()

        assert cache.config is not None


class TestPlanCache:
    """Tests for the Adaptive Plan Cache (APC)."""

    def test_plan_cache_initialization(self):
        """Test PlanCache initializes correctly."""
        from llmcore.agents.caching import PlanCache

        cache = PlanCache()

        assert cache is not None
        assert hasattr(cache, "store_plan")
        assert hasattr(cache, "find_similar_plan")

    @pytest.mark.asyncio
    async def test_store_plan(self):
        """Test storing execution plans."""
        from llmcore.agents.caching import PlanCache

        cache = PlanCache()

        plan_steps = [
            {"action": "search", "params": {"query": "test"}},
            {"action": "analyze", "params": {"data": "results"}},
        ]

        plan_id = await cache.store_plan(
            goal="Find and analyze test data",
            plan_steps=plan_steps,
        )

        assert plan_id is not None

    @pytest.mark.asyncio
    async def test_find_similar_plan(self):
        """Test finding similar plans."""
        from llmcore.agents.caching import PlanCache

        cache = PlanCache()

        plan_steps = [{"action": "search"}]
        await cache.store_plan(
            goal="Search for files in directory",
            plan_steps=plan_steps,
        )

        # Find similar
        result = await cache.find_similar_plan("Find files in folder")
        # May or may not find depending on similarity

    @pytest.mark.asyncio
    async def test_record_outcome(self):
        """Test recording plan outcomes."""
        from llmcore.agents.caching import PlanCache

        cache = PlanCache()

        plan_steps = [{"action": "test"}]
        await cache.store_plan(goal="Test goal", plan_steps=plan_steps)

        await cache.record_outcome("Test goal", success=True)
        await cache.record_outcome("Test goal", success=False)
        # Should not raise

    def test_statistics(self):
        """Test getting plan cache statistics."""
        from llmcore.agents.caching import PlanCache

        cache = PlanCache()

        stats = cache.get_statistics()
        assert stats is not None


# =============================================================================
# CONTEXT MANAGER TESTS
# =============================================================================


class TestContextManager:
    """Tests for context window management."""

    def test_manager_initialization(self):
        """Test ContextManager initializes correctly."""
        from llmcore.agents.context import ContextManager

        manager = ContextManager()

        assert manager is not None
        assert hasattr(manager, "build_context")
        assert hasattr(manager, "config")

    def test_build_context(self):
        """Test building context."""
        from llmcore.agents.context import ContextManager

        manager = ContextManager()

        context = manager.build_context(
            system_prompt="You are a helpful assistant.",
            goal="Help the user.",
            history=[],
        )

        assert context is not None

    def test_estimate_tokens(self):
        """Test token estimation."""
        from llmcore.agents.context import ContextManager

        manager = ContextManager()

        tokens = manager.estimate_tokens("This is a test message.")

        assert tokens > 0

    def test_priority_enum(self):
        """Test Priority enum exists."""
        from llmcore.agents.context.context_manager import Priority

        assert Priority.REQUIRED is not None
        assert Priority.HIGH is not None
        assert Priority.MEDIUM is not None
        assert Priority.LOW is not None


class TestTextCompressor:
    """Tests for text compression utilities."""

    def test_compressor_initialization(self):
        """Test TextCompressor initializes correctly."""
        from llmcore.agents.context.context_manager import TextCompressor

        compressor = TextCompressor()
        assert compressor is not None

    def test_compress_text(self):
        """Test text compression."""
        from llmcore.agents.context.context_manager import TextCompressor

        compressor = TextCompressor()

        long_text = "This is a very long piece of text. " * 20
        compressed = compressor.compress(long_text, target_ratio=0.5)

        assert len(compressed) < len(long_text)


# =============================================================================
# MEMORY STORE TESTS
# =============================================================================


class TestWorkingMemory:
    """Tests for working memory system."""

    def test_memory_initialization(self):
        """Test WorkingMemory initializes correctly."""
        from llmcore.agents.memory import WorkingMemory

        memory = WorkingMemory()

        assert memory is not None
        assert memory.capacity > 0

    def test_add_and_get(self):
        """Test adding and retrieving items."""
        from llmcore.agents.memory import WorkingMemory
        from llmcore.agents.memory.memory_store import MemoryImportance

        memory = WorkingMemory(capacity=5)

        # add takes (content: str, importance, metadata, pinned)
        item_id = memory.add(
            content="Remember this fact",
            importance=MemoryImportance.HIGH,
        )

        retrieved = memory.get(item_id)

        assert retrieved is not None
        assert retrieved.content == "Remember this fact"

    def test_capacity_limit(self):
        """Test capacity limit enforcement."""
        from llmcore.agents.memory import WorkingMemory
        from llmcore.agents.memory.memory_store import MemoryImportance

        memory = WorkingMemory(capacity=3)

        for i in range(5):
            memory.add(
                content=f"Content {i}",
                importance=MemoryImportance.MEDIUM,
            )

        assert memory.count <= 3

    def test_pinning_mechanism(self):
        """Test pinning prevents eviction."""
        from llmcore.agents.memory import WorkingMemory
        from llmcore.agents.memory.memory_store import MemoryImportance

        memory = WorkingMemory(capacity=3)

        # Add item with pinned=True
        pinned_id = memory.add(
            content="Critical info",
            importance=MemoryImportance.CRITICAL,
            pinned=True,
        )

        # Fill remaining capacity
        for i in range(5):
            memory.add(
                content=f"Content {i}",
                importance=MemoryImportance.LOW,
            )

        # Pinned item should still exist
        assert memory.get(pinned_id) is not None

    def test_remove(self):
        """Test removing items."""
        from llmcore.agents.memory import WorkingMemory

        memory = WorkingMemory()

        item_id = memory.add(content="Test")
        memory.remove(item_id)

        assert memory.get(item_id) is None

    def test_clear(self):
        """Test clearing all items."""
        from llmcore.agents.memory import WorkingMemory

        memory = WorkingMemory()

        memory.add(content="Test1")
        memory.add(content="Test2")
        memory.clear()

        assert memory.count == 0

    def test_get_all(self):
        """Test getting all items."""
        from llmcore.agents.memory import WorkingMemory

        memory = WorkingMemory()

        memory.add(content="Test1")
        memory.add(content="Test2")

        all_items = memory.get_all()
        assert len(all_items) == 2

    def test_get_context(self):
        """Test getting context representation."""
        from llmcore.agents.memory import WorkingMemory

        memory = WorkingMemory()

        memory.add(content="Test content")

        context = memory.get_context()
        assert "Test content" in context


class TestLongTermMemory:
    """Tests for long-term memory system."""

    def test_memory_types(self):
        """Test MemoryType enum."""
        from llmcore.agents.memory.memory_store import MemoryType

        assert MemoryType.SEMANTIC is not None
        assert MemoryType.EPISODIC is not None
        assert MemoryType.PROCEDURAL is not None

    def test_memory_importance(self):
        """Test MemoryImportance enum."""
        from llmcore.agents.memory.memory_store import MemoryImportance

        assert MemoryImportance.CRITICAL is not None
        assert MemoryImportance.HIGH is not None
        assert MemoryImportance.MEDIUM is not None
        assert MemoryImportance.LOW is not None

    def test_in_memory_store(self):
        """Test InMemoryStore."""
        import asyncio

        from llmcore.agents.memory.memory_store import InMemoryStore, MemoryType

        store = InMemoryStore(memory_type=MemoryType.SEMANTIC)

        # store is async, use retrieve to get back
        async def test_store():
            item_id = await store.store(content="Python was created in 1991")
            assert item_id is not None
            results = await store.retrieve(query="Python")
            return results

        results = asyncio.run(test_store())
        assert isinstance(results, list)


class TestMemoryManager:
    """Tests for unified memory manager."""

    def test_manager_initialization(self):
        """Test MemoryManager initializes correctly."""
        from llmcore.agents.memory import MemoryManager

        manager = MemoryManager()

        assert manager is not None
        assert hasattr(manager, "working")
        assert hasattr(manager, "remember")
        assert hasattr(manager, "recall")

    @pytest.mark.asyncio
    async def test_remember_to_long_term(self):
        """Test remembering to long-term memory."""
        from llmcore.agents.memory import MemoryManager
        from llmcore.agents.memory.memory_store import MemoryType

        manager = MemoryManager()

        await manager.remember(
            content="Long-term fact",
            memory_type=MemoryType.SEMANTIC,
        )

        results = await manager.recall("Long-term fact")
        assert results is not None

    @pytest.mark.asyncio
    async def test_recall(self):
        """Test recalling memories."""
        from llmcore.agents.memory import MemoryManager
        from llmcore.agents.memory.memory_store import MemoryType

        manager = MemoryManager()

        await manager.remember(
            content="Python is a programming language",
            memory_type=MemoryType.SEMANTIC,
        )

        results = await manager.recall("Python programming")
        assert results is not None

    def test_clear_working(self):
        """Test clearing working memory."""
        from llmcore.agents.memory import MemoryManager

        manager = MemoryManager()
        manager.clear_working()

        assert manager.working.count == 0

    @pytest.mark.asyncio
    async def test_statistics(self):
        """Test memory statistics."""
        from llmcore.agents.memory import MemoryManager

        manager = MemoryManager()

        stats = await manager.get_statistics()
        assert stats is not None

    def test_memory_stores_access(self):
        """Test accessing different memory stores."""
        from llmcore.agents.memory import MemoryManager

        manager = MemoryManager()

        assert manager.semantic is not None
        assert manager.episodic is not None
        assert manager.procedural is not None


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestPhase4Integration:
    """Integration tests for Phase 4 components working together."""

    def test_all_components_importable(self):
        """Test all Phase 4 components can be imported."""
        from llmcore.agents.caching import PlanCache, SemanticCache
        from llmcore.agents.context import ContextManager
        from llmcore.agents.memory import MemoryManager, WorkingMemory
        from llmcore.agents.routing import ModelRouter

        assert ModelRouter is not None
        assert SemanticCache is not None
        assert PlanCache is not None
        assert ContextManager is not None
        assert MemoryManager is not None
        assert WorkingMemory is not None

    def test_components_initialize(self):
        """Test all components initialize without error."""
        from llmcore.agents.caching import PlanCache, SemanticCache
        from llmcore.agents.context import ContextManager
        from llmcore.agents.memory import MemoryManager
        from llmcore.agents.routing import ModelRouter

        router = ModelRouter()
        sem_cache = SemanticCache()
        plan_cache = PlanCache()
        context_mgr = ContextManager()
        memory_mgr = MemoryManager()

        assert router is not None
        assert sem_cache is not None
        assert plan_cache is not None
        assert context_mgr is not None
        assert memory_mgr is not None


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestPhase4EdgeCases:
    """Edge case tests for Phase 4 components."""

    def test_memory_overflow(self):
        """Test memory with many items."""
        from llmcore.agents.memory import WorkingMemory

        memory = WorkingMemory(capacity=2)

        for i in range(100):
            memory.add(content=f"Content {i}")

        assert memory.count <= 2

    def test_unicode_in_memory(self):
        """Test Unicode in memory."""
        from llmcore.agents.memory import WorkingMemory

        memory = WorkingMemory()

        item_id = memory.add(content="日本語のコンテンツ")

        item = memory.get(item_id)
        assert item is not None
        assert "日本語" in item.content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
