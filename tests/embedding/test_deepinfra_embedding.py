# tests/embedding/test_deepinfra_embedding.py
"""DeepInfra embeddings keep ``dimensions`` (live: 256 -> 256-dim vectors)."""

from __future__ import annotations

import pytest

pytest.importorskip("openai")

from llmcore.embedding.deepinfra import DeepInfraEmbedding  # noqa: E402


def test_dimensions_are_kept():
    emb = DeepInfraEmbedding({"api_key": "k", "dimensions": 256})
    assert emb._dimensions == 256
    assert emb._base_url == "https://api.deepinfra.com/v1/openai"
