from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from llmcore.agents import AgentMode, EnhancedAgentManager
from llmcore.api import LLMCore


def _initialized_llmcore() -> LLMCore:
    llm = LLMCore()
    llm._provider_manager = Mock()
    llm._memory_manager = Mock()
    llm._storage_manager = Mock()
    llm._memory_manager.retrieve_relevant_context = AsyncMock(return_value=[])
    llm._memory_manager.store_memory = AsyncMock(return_value=None)
    llm._storage_manager.add_episode = AsyncMock(return_value=None)
    llm._storage_manager.get_episodes = AsyncMock(return_value=[])
    return llm


def test_create_enhanced_agent_manager_uses_owned_managers() -> None:
    llm = _initialized_llmcore()

    manager = llm.create_enhanced_agent_manager()

    assert isinstance(manager, EnhancedAgentManager)
    assert manager._provider_manager is llm._provider_manager
    assert manager._memory_manager is llm._memory_manager
    assert manager._storage_manager is llm._storage_manager
    assert manager.default_mode == AgentMode.SINGLE


def test_create_enhanced_agent_manager_passes_optional_components() -> None:
    llm = _initialized_llmcore()
    prompt_registry = Mock()
    tracer = Mock()
    observability = Mock()
    agents_config = Mock()

    manager = llm.create_enhanced_agent_manager(
        prompt_registry=prompt_registry,
        tracer=tracer,
        default_mode=AgentMode.LEGACY,
        observability=observability,
        agents_config=agents_config,
    )

    assert manager.prompt_registry is prompt_registry
    assert manager.default_mode == AgentMode.LEGACY
    assert manager._tracer is tracer
    assert manager._observability is observability
    assert manager._agents_config is agents_config


def test_create_enhanced_agent_manager_passes_context_synthesizer() -> None:
    llm = _initialized_llmcore()
    context_synthesizer = Mock()

    manager = llm.create_enhanced_agent_manager(context_synthesizer=context_synthesizer)

    assert manager.single_agent.context_synthesizer is context_synthesizer
    assert manager.single_agent.cognitive_cycle.context_synthesizer is context_synthesizer


@pytest.mark.asyncio
async def test_create_enhanced_agent_manager_builds_semantic_source_from_memory_backend() -> None:
    llm = _initialized_llmcore()

    class FakeMemoryBackend:
        def as_retrieval_fn(self):
            async def retrieve(query: str, top_k: int = 10):
                return [
                    {
                        "content": f"semantiscan memory for {query}",
                        "source": "semantiscan://repo/chunk-1",
                        "score": 0.91,
                    }
                ]

            return retrieve

    manager = llm.create_enhanced_agent_manager(memory_backend=FakeMemoryBackend())
    synthesizer = manager.single_agent.cognitive_cycle.context_synthesizer

    assert synthesizer is not None

    context = await synthesizer.synthesize(current_task=SimpleNamespace(description="adapter"))

    assert "semantiscan memory for adapter" in context.content
    assert "semantiscan://repo/chunk-1" in context.content


def test_create_enhanced_agent_manager_requires_initialized_instance() -> None:
    llm = LLMCore()

    with pytest.raises(RuntimeError, match="LLMCore is not initialized"):
        llm.create_enhanced_agent_manager()
