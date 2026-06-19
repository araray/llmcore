from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest


def test_memory_backends_do_not_import_semantiscan_at_module_import():
    removed = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "semantiscan" or name.startswith("semantiscan.")
    }
    for name in removed:
        sys.modules.pop(name, None)

    try:
        module = importlib.import_module("llmcore.memory.backends")
        importlib.reload(module)

        assert not any(
            name == "semantiscan" or name.startswith("semantiscan.") for name in sys.modules
        )
    finally:
        sys.modules.update(removed)


@pytest.mark.asyncio
async def test_semantiscan_backend_maps_retrieval_batch_to_memory_records():
    from llmcore.memory import SemantiscanMemoryBackend

    calls = []

    async def fake_retrieve(query: str, **kwargs):
        calls.append((query, kwargs))
        return SimpleNamespace(
            strategy="sota",
            metadata={"retrieval_confidence": 0.93},
            results=[
                SimpleNamespace(
                    chunk_id="chunk-1",
                    content="retrieved content",
                    score=0.87,
                    strategy="hybrid",
                    metadata={
                        "file_path": "pkg/module.py",
                        "start_line": "10",
                        "end_line": 14,
                        "chunk_entity_id": "entity-1",
                        "chunk_pid": "pid-abc",
                        "run_id": "run-42",
                        "provenance": {"namespace": "parser"},
                    },
                )
            ],
        )

    backend = SemantiscanMemoryBackend(
        retrieve_fn=fake_retrieve,
        collection="repo",
        storage=object(),
        embedder=object(),
        strategy="sota",
    )

    records = await backend.retrieve("where is the parser?", top_k=3, filters={"lang": "py"})

    assert calls[0][0] == "where is the parser?"
    assert calls[0][1]["collection"] == "repo"
    assert calls[0][1]["top_k"] == 3
    assert calls[0][1]["filters"] == {"lang": "py"}
    assert records[0].content == "retrieved content"
    assert records[0].score == 0.87
    assert records[0].source == "pkg/module.py"
    assert records[0].metadata["retrieval_batch"] == {"retrieval_confidence": 0.93}

    citation = records[0].citations[0]
    assert citation.chunk_id == "chunk-1"
    assert citation.path == "pkg/module.py"
    assert citation.start_line == 10
    assert citation.end_line == 14
    assert citation.metastore_entity_id == "entity-1"
    assert citation.metastore_pid == "pid-abc"
    assert citation.run_id == "run-42"
    assert citation.provenance == {"namespace": "parser"}


@pytest.mark.asyncio
async def test_semantiscan_backend_prefers_typed_citation_when_available():
    from llmcore.memory import SemantiscanMemoryBackend

    typed_citation = SimpleNamespace(
        chunk_id="chunk-typed",
        source_file="typed.py",
        start_line=3,
        end_line=9,
        document_id="repo",
        uri="file://typed.py",
        metastore_entity_id="entity-typed",
        metastore_pid="pid-typed",
        run_id="run-typed",
        provenance={"namespace": "parser"},
        metadata={"strategy": "hybrid"},
    )

    async def fake_retrieve(query: str, **kwargs):
        return [
            SimpleNamespace(
                chunk_id="chunk-raw",
                content="typed citation content",
                score=0.77,
                citation=typed_citation,
                metadata={"file_path": "raw.py"},
            )
        ]

    records = await SemantiscanMemoryBackend(retrieve_fn=fake_retrieve).retrieve("query")

    citation = records[0].citations[0]
    assert citation.chunk_id == "chunk-typed"
    assert citation.path == "typed.py"
    assert citation.document_id == "repo"
    assert citation.uri == "file://typed.py"
    assert citation.metastore_entity_id == "entity-typed"
    assert citation.metastore_pid == "pid-typed"
    assert citation.provenance == {"namespace": "parser"}


@pytest.mark.asyncio
async def test_semantiscan_backend_can_feed_semantic_context_source():
    from llmcore.context.sources.semantic import SemanticContextSource
    from llmcore.memory import SemantiscanMemoryBackend

    async def fake_retrieve(query: str, **kwargs):
        return [
            {
                "chunk_id": "c1",
                "content": f"answer for {query}",
                "score": 0.9,
                "metadata": {"source": "docs/spec.md"},
            }
        ]

    backend = SemantiscanMemoryBackend(retrieve_fn=fake_retrieve)
    source = SemanticContextSource(backend.as_retrieval_fn())

    context = await source.get_context(task=SimpleNamespace(description="adapter"))

    assert "answer for adapter" in context.content
    assert "docs/spec.md" in context.content
