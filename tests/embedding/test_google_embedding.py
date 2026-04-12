# tests/embedding/test_google_embedding.py
"""
Unit tests for the GoogleAIEmbedding provider.

Validates the fixes for google-genai SDK v1.72.0 API changes:
- embed_content() now uses `contents=` (not `content=`)
- task_type is inside EmbedContentConfig (not a top-level kwarg)
- Response is EmbedContentResponse with .embeddings[i].values (not dict)
"""

import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock google modules before import
mock_genai = MagicMock()
mock_types = MagicMock()
mock_errors = MagicMock()


# Create a real-ish GoogleAPIError
class FakeGoogleAPIError(Exception):
    pass


mock_errors.GoogleAPIError = FakeGoogleAPIError

with patch.dict(
    sys.modules,
    {
        "google": MagicMock(),
        "google.genai": mock_genai,
        "google.genai.types": mock_types,
        "google.genai.errors": mock_errors,
    },
):
    from llmcore.embedding import google as ge

    ge.google_genai_available = True
    ge.genai = mock_genai
    ge.genai_types = mock_types
    ge.genai_errors = mock_errors


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def embedding_config():
    return {
        "api_key": "fake-key",
        "default_model": "text-embedding-004",
        "task_type": "RETRIEVAL_DOCUMENT",
    }


@pytest.fixture
def embedding_model(embedding_config):
    model = ge.GoogleAIEmbedding(embedding_config)
    # Mock the client
    mock_client = MagicMock()
    model._client = mock_client
    return model


# ---------------------------------------------------------------------------
# Test: Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_strips_models_prefix(self):
        config = {"default_model": "models/text-embedding-004"}
        m = ge.GoogleAIEmbedding(config)
        assert m._model_name == "text-embedding-004"

    def test_default_task_type(self):
        m = ge.GoogleAIEmbedding({})
        assert m._task_type == "RETRIEVAL_DOCUMENT"

    def test_custom_task_type(self):
        m = ge.GoogleAIEmbedding({"task_type": "retrieval_query"})
        assert m._task_type == "RETRIEVAL_QUERY"

    def test_output_dimensionality(self):
        m = ge.GoogleAIEmbedding({"output_dimensionality": 256})
        assert m._output_dimensionality == 256


# ---------------------------------------------------------------------------
# Test: _build_embed_config
# ---------------------------------------------------------------------------


class TestBuildEmbedConfig:
    def test_creates_config_with_task_type(self, embedding_model):
        """Verify task_type is passed inside EmbedContentConfig."""
        config = embedding_model._build_embed_config()
        # The mock will have been called with task_type kwarg
        mock_types.EmbedContentConfig.assert_called()
        call_kwargs = mock_types.EmbedContentConfig.call_args
        assert call_kwargs.kwargs.get("task_type") == "RETRIEVAL_DOCUMENT"

    def test_creates_config_with_dimensionality(self):
        m = ge.GoogleAIEmbedding(
            {
                "task_type": "CLUSTERING",
                "output_dimensionality": 128,
            }
        )
        m._build_embed_config()
        call_kwargs = mock_types.EmbedContentConfig.call_args
        assert call_kwargs.kwargs.get("output_dimensionality") == 128


# ---------------------------------------------------------------------------
# Test: generate_embedding API call signature
# ---------------------------------------------------------------------------


class TestGenerateEmbedding:
    @pytest.mark.asyncio
    async def test_uses_contents_not_content(self, embedding_model):
        """Verify we call embed_content(contents=..., config=...) not content=."""
        # Mock the async embed_content
        mock_embed = AsyncMock()
        mock_response = SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2, 0.3])])
        mock_embed.return_value = mock_response
        embedding_model._client.aio.models.embed_content = mock_embed

        result = await embedding_model.generate_embedding("test text")

        # Verify call signature
        mock_embed.assert_called_once()
        call_kwargs = mock_embed.call_args.kwargs
        assert "contents" in call_kwargs, "Should use 'contents', not 'content'"
        assert "content" not in call_kwargs, "Old 'content' param should not be used"
        assert "task_type" not in call_kwargs, "task_type should be in config, not top-level"
        assert "config" in call_kwargs
        assert call_kwargs["contents"] == "test text"
        assert call_kwargs["model"] == "models/text-embedding-004"

    @pytest.mark.asyncio
    async def test_parses_pydantic_response(self, embedding_model):
        """Verify we access result.embeddings[0].values, not result.get('embedding')."""
        mock_embed = AsyncMock()
        mock_response = SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0, 2.0, 3.0, 4.0])])
        mock_embed.return_value = mock_response
        embedding_model._client.aio.models.embed_content = mock_embed

        result = await embedding_model.generate_embedding("hello")
        assert result == [1.0, 2.0, 3.0, 4.0]

    @pytest.mark.asyncio
    async def test_empty_text_raises(self, embedding_model):
        from llmcore.exceptions import EmbeddingError

        with pytest.raises(EmbeddingError):
            await embedding_model.generate_embedding("")


# ---------------------------------------------------------------------------
# Test: generate_embeddings (batch) API call
# ---------------------------------------------------------------------------


class TestGenerateEmbeddings:
    @pytest.mark.asyncio
    async def test_batch_uses_contents_list(self, embedding_model):
        """Batch call passes list to contents= parameter."""
        mock_embed = AsyncMock()
        mock_response = SimpleNamespace(
            embeddings=[
                SimpleNamespace(values=[0.1, 0.2]),
                SimpleNamespace(values=[0.3, 0.4]),
            ]
        )
        mock_embed.return_value = mock_response
        embedding_model._client.aio.models.embed_content = mock_embed

        result = await embedding_model.generate_embeddings(["text1", "text2"])

        call_kwargs = mock_embed.call_args.kwargs
        assert call_kwargs["contents"] == ["text1", "text2"]
        assert len(result) == 2
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.3, 0.4]

    @pytest.mark.asyncio
    async def test_batch_handles_empty_strings(self, embedding_model):
        """Empty strings in batch get zero-vector placeholders."""
        mock_embed = AsyncMock()
        # ["hello", "", "world"] → 2 non-empty texts sent to API
        mock_response = SimpleNamespace(
            embeddings=[
                SimpleNamespace(values=[1.0, 2.0]),
                SimpleNamespace(values=[3.0, 4.0]),
            ]
        )
        mock_embed.return_value = mock_response
        embedding_model._client.aio.models.embed_content = mock_embed

        result = await embedding_model.generate_embeddings(["hello", "", "world"])
        assert len(result) == 3
        assert result[0] == [1.0, 2.0]
        assert result[1] == [0.0, 0.0]  # zero-vector for empty
        assert result[2] == [3.0, 4.0]

    @pytest.mark.asyncio
    async def test_empty_list(self, embedding_model):
        result = await embedding_model.generate_embeddings([])
        assert result == []
