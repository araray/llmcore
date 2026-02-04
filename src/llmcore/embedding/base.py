# src/llmcore/embedding/base.py
"""
Abstract Base Class for Text Embedding Models.

This module defines the common interface that all specific embedding model
implementations (e.g., Sentence Transformers, OpenAI Embeddings, Google AI Embeddings)
must adhere to within the LLMCore library. This is essential for the RAG functionality.
"""

import abc
from typing import Any, Dict, List


class BaseEmbeddingModel(abc.ABC):
    """
    Abstract Base Class for text embedding model integrations.

    Ensures all embedding models provide a consistent way to generate
    vector representations (embeddings) for text strings.
    """

    @abc.abstractmethod
    def __init__(self, config: dict[str, Any]):
        """
        Initialize the embedding model with its specific configuration.

        Args:
            config: A dictionary containing model-specific settings loaded
                    from the main LLMCore configuration (e.g., model_name,
                    api_key, device preference).
        """
        pass

    @abc.abstractmethod
    async def initialize(self) -> None:
        """
        Perform any necessary asynchronous initialization.

        This could include loading models into memory, establishing connections, etc.
        Should be called after the instance is created.
        """
        pass

    @abc.abstractmethod
    async def generate_embedding(self, text: str) -> list[float]:
        """
        Generate a vector embedding for a single text string.

        Args:
            text: The input text string to embed.

        Returns:
            A list of floats representing the vector embedding.

        Raises:
            EmbeddingError: If the embedding generation fails.
        """
        pass

    @abc.abstractmethod
    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate vector embeddings for a batch of text strings efficiently.

        Implementing this method is recommended for performance when embedding
        multiple documents. If not overridden, it can default to calling
        `generate_embedding` repeatedly, but batch processing is preferred.

        Args:
            texts: A list of input text strings to embed.

        Returns:
            A list of lists of floats, where each inner list is the vector
            embedding for the corresponding input text.

        Raises:
            EmbeddingError: If the batch embedding generation fails.
        """
        # Default implementation (can be overridden for efficiency)
        embeddings = []
        for text in texts:
            embeddings.append(await self.generate_embedding(text))
        return embeddings

    # Optional: Add a method to get embedding dimension if needed
    # @abc.abstractmethod
    # def get_embedding_dimension(self) -> int:
    #     """Return the dimensionality of the embeddings produced by this model."""
    #     pass

    # Optional: Add an async close method if models need cleanup
    # async def close(self) -> None:
    #     """Clean up resources like loaded models or connections."""
    #     pass
