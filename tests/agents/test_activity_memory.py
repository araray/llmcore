# tests/agents/test_activity_memory.py
"""
Tests for Activity Memory Handlers (Phase 3).

Verifies that memory_store and memory_search activities properly:
- Store data in working memory
- Search working memory for matches
- Gracefully degrade when MemoryManager unavailable
- Integrate with MemoryManager when available

Phase: 3 - Activity System Completion
Reference: LLMCORE_CORRECTION_MASTER_PLAN.md Section 5
"""

from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from llmcore.agents.activities.executor import ActivityExecutor
from llmcore.agents.activities.registry import ExecutionContext, get_default_registry
from llmcore.agents.activities.schema import ActivityRequest, ActivityStatus

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def execution_context() -> ExecutionContext:
    """Create a fresh execution context for each test."""
    return ExecutionContext(
        working_dir="/tmp",
        sandbox_available=False,
        session_id="test-session-001",
    )


@pytest.fixture
def executor_no_memory_manager() -> ActivityExecutor:
    """Create executor without memory manager (graceful degradation mode)."""
    return ActivityExecutor(
        registry=get_default_registry(),
        memory_manager=None,
    )


@pytest.fixture
def mock_memory_manager() -> MagicMock:
    """Create a mock MemoryManager from agents.memory.memory_store."""
    manager = MagicMock()

    # Mock remember() async method
    manager.remember = AsyncMock(return_value="mem-12345")

    # Mock recall() async method with RecallResult-like response
    recall_result = MagicMock()
    recall_result.memories = []
    manager.recall = AsyncMock(return_value=recall_result)

    return manager


@pytest.fixture
def executor_with_memory_manager(mock_memory_manager) -> ActivityExecutor:
    """Create executor with mock memory manager."""
    return ActivityExecutor(
        registry=get_default_registry(),
        memory_manager=mock_memory_manager,
    )


def make_memory_store_request(
    key: str,
    value: Any,
    memory_type: str = "working",
    metadata: Optional[Dict[str, Any]] = None,
) -> ActivityRequest:
    """Helper to create memory_store request."""
    params = {
        "key": key,
        "value": value,
        "memory_type": memory_type,
    }
    if metadata:
        params["metadata"] = metadata
    return ActivityRequest(
        activity="memory_store",
        parameters=params,
        request_id="test-req-001",
    )


def make_memory_search_request(
    query: str,
    max_results: int = 5,
    search_working: bool = True,
    search_longterm: bool = True,
) -> ActivityRequest:
    """Helper to create memory_search request."""
    return ActivityRequest(
        activity="memory_search",
        parameters={
            "query": query,
            "max_results": max_results,
            "search_working": search_working,
            "search_longterm": search_longterm,
        },
        request_id="test-req-002",
    )


# =============================================================================
# Test: memory_store Activity - Working Memory
# =============================================================================


class TestMemoryStoreWorkingMemory:
    """Tests for storing data in working memory."""

    @pytest.mark.asyncio
    async def test_store_in_working_memory(
        self, executor_no_memory_manager, execution_context
    ):
        """Verify store_memory stores data in working memory by default."""
        request = make_memory_store_request(
            key="capital_of_france",
            value="Paris",
            memory_type="working",
        )

        result = await executor_no_memory_manager.execute(request, execution_context)

        # Check result
        assert result.status == ActivityStatus.SUCCESS
        assert "Stored in working memory" in result.output
        assert "capital_of_france" in result.output

        # Verify data actually stored
        assert "capital_of_france" in execution_context.working_memory
        stored = execution_context.working_memory["capital_of_france"]
        assert stored["value"] == "Paris"
        assert "stored_at" in stored["metadata"]
        assert stored["metadata"]["source"] == "activity_memory_store"

    @pytest.mark.asyncio
    async def test_store_default_to_working_memory(
        self, executor_no_memory_manager, execution_context
    ):
        """Verify store_memory defaults to working memory when type not specified."""
        # Create request without explicit memory_type
        request = ActivityRequest(
            activity="memory_store",
            parameters={"key": "test_key", "value": "test_value"},
            request_id="test-req-003",
        )

        result = await executor_no_memory_manager.execute(request, execution_context)

        assert result.status == ActivityStatus.SUCCESS
        assert "test_key" in execution_context.working_memory

    @pytest.mark.asyncio
    async def test_store_with_metadata(
        self, executor_no_memory_manager, execution_context
    ):
        """Verify custom metadata is preserved in stored memory."""
        custom_metadata = {"importance": "high", "category": "geography"}
        request = make_memory_store_request(
            key="population",
            value="67 million",
            memory_type="working",
            metadata=custom_metadata,
        )

        await executor_no_memory_manager.execute(request, execution_context)

        stored = execution_context.working_memory["population"]
        assert stored["metadata"]["importance"] == "high"
        assert stored["metadata"]["category"] == "geography"

    @pytest.mark.asyncio
    async def test_store_complex_value(
        self, executor_no_memory_manager, execution_context
    ):
        """Verify complex values (as JSON strings) can be stored."""
        import json
        complex_value = {
            "name": "France",
            "capital": "Paris",
            "cities": ["Paris", "Lyon", "Marseille"],
        }
        # Store complex data as JSON string
        request = make_memory_store_request(
            key="country_info",
            value=json.dumps(complex_value),
        )

        result = await executor_no_memory_manager.execute(request, execution_context)

        assert result.status == ActivityStatus.SUCCESS
        stored = execution_context.working_memory["country_info"]["value"]
        # Verify the JSON string was stored and can be parsed back
        parsed = json.loads(stored)
        assert parsed["name"] == "France"
        assert "Lyon" in parsed["cities"]


# =============================================================================
# Test: memory_store Activity - Long-term Memory
# =============================================================================


class TestMemoryStoreLongterm:
    """Tests for storing data in long-term memory."""

    @pytest.mark.asyncio
    async def test_longterm_without_manager_fallback(
        self, executor_no_memory_manager, execution_context
    ):
        """Verify longterm storage falls back to working memory when no manager."""
        request = make_memory_store_request(
            key="important_fact",
            value="The sky is blue",
            memory_type="longterm",
        )

        result = await executor_no_memory_manager.execute(request, execution_context)

        # Should succeed with fallback message
        assert result.status == ActivityStatus.SUCCESS
        assert "FALLBACK" in result.output
        assert "working memory" in result.output.lower()

        # Data should be in working memory
        assert "important_fact" in execution_context.working_memory
        assert execution_context.working_memory["important_fact"]["metadata"]["fallback"] is True

    @pytest.mark.asyncio
    async def test_longterm_with_manager_success(
        self, executor_with_memory_manager, mock_memory_manager, execution_context
    ):
        """Verify longterm storage uses MemoryManager when available."""
        request = make_memory_store_request(
            key="user_preference",
            value="prefers Python",
            memory_type="longterm",
        )

        result = await executor_with_memory_manager.execute(request, execution_context)

        # Should succeed with long-term message
        assert result.status == ActivityStatus.SUCCESS
        assert "long-term memory" in result.output.lower()
        assert "mem-12345" in result.output  # The mock item ID

        # MemoryManager.remember should have been called
        mock_memory_manager.remember.assert_called_once()
        call_args = mock_memory_manager.remember.call_args
        assert "user_preference: prefers Python" in call_args.kwargs["content"]

    @pytest.mark.asyncio
    async def test_longterm_manager_error_fallback(
        self, executor_with_memory_manager, mock_memory_manager, execution_context
    ):
        """Verify errors during long-term storage fall back to working memory."""
        # Make remember() raise an exception
        mock_memory_manager.remember.side_effect = RuntimeError("Connection failed")

        request = make_memory_store_request(
            key="critical_data",
            value="should not be lost",
            memory_type="longterm",
        )

        result = await executor_with_memory_manager.execute(request, execution_context)

        # Should succeed with error message but data preserved
        assert result.status == ActivityStatus.SUCCESS
        assert "ERROR" in result.output
        assert "working memory instead" in result.output.lower()

        # Data should be in working memory as fallback
        assert "critical_data" in execution_context.working_memory

    @pytest.mark.asyncio
    async def test_invalid_memory_type(
        self, executor_no_memory_manager, execution_context
    ):
        """Verify invalid memory_type is rejected by validation."""
        request = make_memory_store_request(
            key="test",
            value="test",
            memory_type="invalid_type",
        )

        result = await executor_no_memory_manager.execute(request, execution_context)

        # With enum constraint in activity definition, validation fails
        assert result.status == ActivityStatus.FAILED
        assert "Invalid value" in result.error or "invalid_type" in (result.error or "")


# =============================================================================
# Test: memory_search Activity - Working Memory
# =============================================================================


class TestMemorySearchWorkingMemory:
    """Tests for searching working memory."""

    @pytest.mark.asyncio
    async def test_search_finds_matching_key(
        self, executor_no_memory_manager, execution_context
    ):
        """Verify search finds items by key match."""
        # Pre-populate working memory
        execution_context.working_memory["capital_france"] = {
            "value": "Paris",
            "metadata": {},
        }
        execution_context.working_memory["capital_germany"] = {
            "value": "Berlin",
            "metadata": {},
        }

        request = make_memory_search_request(query="france")

        result = await executor_no_memory_manager.execute(request, execution_context)

        assert result.status == ActivityStatus.SUCCESS
        assert "Found" in result.output
        assert "capital_france" in result.output
        assert "Paris" in result.output
        # Should NOT include Germany
        assert "Berlin" not in result.output

    @pytest.mark.asyncio
    async def test_search_finds_matching_value(
        self, executor_no_memory_manager, execution_context
    ):
        """Verify search finds items by value match."""
        execution_context.working_memory["city_info"] = {
            "value": "Paris is the capital of France",
            "metadata": {},
        }

        request = make_memory_search_request(query="Paris")

        result = await executor_no_memory_manager.execute(request, execution_context)

        assert result.status == ActivityStatus.SUCCESS
        assert "Found" in result.output
        assert "city_info" in result.output

    @pytest.mark.asyncio
    async def test_search_no_results(
        self, executor_no_memory_manager, execution_context
    ):
        """Verify search returns appropriate message when nothing found."""
        execution_context.working_memory["unrelated"] = {"value": "xyz", "metadata": {}}

        request = make_memory_search_request(query="nonexistent")

        result = await executor_no_memory_manager.execute(request, execution_context)

        assert result.status == ActivityStatus.SUCCESS
        assert "No memories found" in result.output

    @pytest.mark.asyncio
    async def test_search_respects_max_results(
        self, executor_no_memory_manager, execution_context
    ):
        """Verify search limits results to max_results."""
        # Add many matching items
        for i in range(10):
            execution_context.working_memory[f"item_{i}"] = {
                "value": f"test value {i}",
                "metadata": {},
            }

        request = make_memory_search_request(query="test", max_results=3)

        result = await executor_no_memory_manager.execute(request, execution_context)

        assert result.status == ActivityStatus.SUCCESS
        # Count how many items appear (look for "[Working]" markers)
        count = result.output.count("[Working]")
        assert count <= 3


# =============================================================================
# Test: memory_search Activity - Long-term Memory Integration
# =============================================================================


class TestMemorySearchLongterm:
    """Tests for searching long-term memory via MemoryManager."""

    @pytest.mark.asyncio
    async def test_search_calls_memory_manager(
        self, executor_with_memory_manager, mock_memory_manager, execution_context
    ):
        """Verify search calls MemoryManager.recall() when available."""
        request = make_memory_search_request(
            query="user preferences",
            search_longterm=True,
        )

        await executor_with_memory_manager.execute(request, execution_context)

        mock_memory_manager.recall.assert_called_once()
        call_args = mock_memory_manager.recall.call_args
        assert call_args.kwargs["query"] == "user preferences"

    @pytest.mark.asyncio
    async def test_search_combines_working_and_longterm(
        self, executor_with_memory_manager, mock_memory_manager, execution_context
    ):
        """Verify search combines results from both memory types."""
        # Add working memory item
        execution_context.working_memory["working_item"] = {
            "value": "from working memory",
            "metadata": {},
        }

        # Configure mock to return long-term items
        mock_item = MagicMock()
        mock_item.content = "from long-term memory"
        mock_item.relevance = 0.95
        mock_item.memory_type = MagicMock(value="semantic")
        mock_item.metadata = {}

        recall_result = MagicMock()
        recall_result.memories = [mock_item]
        mock_memory_manager.recall = AsyncMock(return_value=recall_result)

        request = make_memory_search_request(
            query="memory",
            search_working=True,
            search_longterm=True,
        )

        result = await executor_with_memory_manager.execute(request, execution_context)

        assert result.status == ActivityStatus.SUCCESS
        assert "[Working]" in result.output
        assert "[Long-term]" in result.output

    @pytest.mark.asyncio
    async def test_search_longterm_only(
        self, executor_with_memory_manager, mock_memory_manager, execution_context
    ):
        """Verify search can target only long-term memory."""
        execution_context.working_memory["should_not_appear"] = {
            "value": "test",
            "metadata": {},
        }

        request = make_memory_search_request(
            query="test",
            search_working=False,
            search_longterm=True,
        )

        result = await executor_with_memory_manager.execute(request, execution_context)

        assert result.status == ActivityStatus.SUCCESS
        assert "[Working]" not in result.output

    @pytest.mark.asyncio
    async def test_search_manager_error_continues(
        self, executor_with_memory_manager, mock_memory_manager, execution_context
    ):
        """Verify search continues gracefully if long-term search fails."""
        execution_context.working_memory["working_item"] = {
            "value": "test value",
            "metadata": {},
        }

        mock_memory_manager.recall.side_effect = RuntimeError("Search failed")

        request = make_memory_search_request(query="test")

        result = await executor_with_memory_manager.execute(request, execution_context)

        # Should still find working memory item
        assert result.status == ActivityStatus.SUCCESS
        assert "[Working]" in result.output
        # And include error message
        assert "[Error]" in result.output


# =============================================================================
# Test: End-to-End Integration
# =============================================================================


class TestMemoryEndToEnd:
    """End-to-end tests for memory store and search together."""

    @pytest.mark.asyncio
    async def test_store_then_search(
        self, executor_no_memory_manager, execution_context
    ):
        """Verify stored data can be searched immediately."""
        # Store data
        store_request = make_memory_store_request(
            key="user_name",
            value="Alice",
        )
        await executor_no_memory_manager.execute(store_request, execution_context)

        # Search for it
        search_request = make_memory_search_request(query="Alice")
        result = await executor_no_memory_manager.execute(search_request, execution_context)

        assert result.status == ActivityStatus.SUCCESS
        assert "user_name" in result.output
        assert "Alice" in result.output

    @pytest.mark.asyncio
    async def test_multiple_stores_and_search(
        self, executor_no_memory_manager, execution_context
    ):
        """Verify multiple items can be stored and searched."""
        # Store multiple items
        for name, city in [("alice_city", "Paris"), ("bob_city", "London"), ("carol_city", "Tokyo")]:
            request = make_memory_store_request(key=name, value=city)
            await executor_no_memory_manager.execute(request, execution_context)

        # Search for cities
        search_request = make_memory_search_request(query="city")
        result = await executor_no_memory_manager.execute(search_request, execution_context)

        assert result.status == ActivityStatus.SUCCESS
        assert "3" in result.output or "Found" in result.output
        assert "Paris" in result.output
        assert "London" in result.output
        assert "Tokyo" in result.output

    @pytest.mark.asyncio
    async def test_realistic_agent_scenario(
        self, executor_no_memory_manager, execution_context
    ):
        """Simulate realistic agent memory usage."""
        # Agent stores user preference
        await executor_no_memory_manager.execute(
            make_memory_store_request(
                key="user_pref_language",
                value="Python",
                metadata={"category": "preferences"},
            ),
            execution_context,
        )

        # Agent stores a fact learned during task
        await executor_no_memory_manager.execute(
            make_memory_store_request(
                key="project_status",
                value="Phase 2 complete, starting Phase 3",
                metadata={"category": "context"},
            ),
            execution_context,
        )

        # Later, agent searches for relevant context
        result = await executor_no_memory_manager.execute(
            make_memory_search_request(query="Python"),
            execution_context,
        )

        assert "user_pref_language" in result.output
        assert "Python" in result.output

        # Agent searches for project context
        result2 = await executor_no_memory_manager.execute(
            make_memory_search_request(query="project status Phase"),
            execution_context,
        )

        assert "project_status" in result2.output


# =============================================================================
# Test: API Compatibility
# =============================================================================


class TestActivityExecutorAPICompatibility:
    """Verify backward compatibility of ActivityExecutor API."""

    def test_executor_without_memory_manager_works(self):
        """Verify executor works without memory_manager (backward compatible)."""
        executor = ActivityExecutor()
        assert executor.memory_manager is None
        assert executor.registry is not None

    def test_executor_with_memory_manager_accepts(self, mock_memory_manager):
        """Verify executor accepts memory_manager parameter."""
        executor = ActivityExecutor(memory_manager=mock_memory_manager)
        assert executor.memory_manager is mock_memory_manager

    def test_executor_all_parameters(self, mock_memory_manager):
        """Verify executor accepts all parameters together."""
        from llmcore.agents.activities.executor import HITLApprover

        executor = ActivityExecutor(
            registry=get_default_registry(),
            validator=None,
            hitl_approver=HITLApprover(),
            sandbox=None,
            memory_manager=mock_memory_manager,
            default_timeout=120,
        )
        assert executor.memory_manager is mock_memory_manager
        assert executor.default_timeout == 120


# =============================================================================
# Test: ExecutionContext Enhancement
# =============================================================================


class TestExecutionContextWorkingMemory:
    """Test ExecutionContext.working_memory attribute."""

    def test_execution_context_has_working_memory(self):
        """Verify ExecutionContext has working_memory attribute."""
        ctx = ExecutionContext()
        assert hasattr(ctx, "working_memory")
        assert isinstance(ctx.working_memory, dict)
        assert len(ctx.working_memory) == 0

    def test_execution_context_working_memory_isolated(self):
        """Verify each context has isolated working memory."""
        ctx1 = ExecutionContext()
        ctx2 = ExecutionContext()

        ctx1.working_memory["key1"] = "value1"

        assert "key1" in ctx1.working_memory
        assert "key1" not in ctx2.working_memory


# =============================================================================
# Regression Tests
# =============================================================================


class TestRegressionBugFixes:
    """Regression tests for specific bugs fixed in Phase 3."""

    @pytest.mark.asyncio
    async def test_bug_memory_store_returned_fake_success(
        self, executor_no_memory_manager, execution_context
    ):
        """
        REGRESSION: memory_store used to return fake success without storing.

        Bug: The old implementation returned "Stored value under key 'X'"
        without actually storing anything.

        Fix: Now stores in ExecutionContext.working_memory.
        """
        request = make_memory_store_request(key="test_key", value="test_value")

        result = await executor_no_memory_manager.execute(request, execution_context)

        # Verify it's not just a fake message
        assert result.status == ActivityStatus.SUCCESS

        # CRITICAL: Verify data is actually stored
        assert "test_key" in execution_context.working_memory
        assert execution_context.working_memory["test_key"]["value"] == "test_value"

    @pytest.mark.asyncio
    async def test_bug_memory_search_returned_fake_results(
        self, executor_no_memory_manager, execution_context
    ):
        """
        REGRESSION: memory_search used to return fake results without searching.

        Bug: The old implementation returned "Memory search for 'X' (max Y results)"
        without actually searching anything.

        Fix: Now searches ExecutionContext.working_memory and returns real results.
        """
        # Store something first
        execution_context.working_memory["actual_data"] = {
            "value": "real stored value",
            "metadata": {},
        }

        request = make_memory_search_request(query="actual")

        result = await executor_no_memory_manager.execute(request, execution_context)

        # Verify it's not just a fake message
        assert result.status == ActivityStatus.SUCCESS

        # CRITICAL: Verify actual search was performed
        assert "actual_data" in result.output
        assert "real stored value" in result.output

        # The old fake message should NOT appear
        assert "Memory search for" not in result.output or "Found" in result.output
