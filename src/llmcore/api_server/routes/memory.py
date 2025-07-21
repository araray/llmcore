# src/llmcore/api_server/routes/memory.py
"""
Memory-related API routes for the llmcore API server.

This module contains the implementation of memory system endpoints that provide
API access to the hierarchical memory architecture, starting with semantic memory.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Request, HTTPException, Query

from ...api import LLMCore
from ...exceptions import VectorStorageError, ConfigError
from ...models import ContextDocument

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/memory/semantic/search", response_model=List[ContextDocument])
async def search_semantic_memory(
    request: Request,
    query: str = Query(description="The text query to search for"),
    collection_name: Optional[str] = Query(
        default=None,
        description="The target collection name. Uses default if not provided"
    ),
    k: int = Query(
        default=3,
        ge=1,
        le=20,
        description="The number of results to return"
    )
) -> List[ContextDocument]:
    """
    Performs a similarity search against the semantic memory (vector store).

    This endpoint exposes the full functionality of the existing vector store
    search capability as a formal, queryable API endpoint. It represents the
    first pillar of the new hierarchical memory architecture.

    Args:
        request: The FastAPI request object containing app state
        query: The text query to search for in the vector store
        collection_name: Optional target collection name
        k: Number of results to return (1-20)

    Returns:
        List of ContextDocument objects containing search results

    Raises:
        HTTPException: For various error conditions (400, 404, 500, 503)
    """
    # Get the LLMCore instance from app state
    llmcore_instance: LLMCore = getattr(request.app.state, 'llmcore_instance', None)
    if not llmcore_instance:
        logger.error("LLMCore instance not found in app state")
        raise HTTPException(
            status_code=503,
            detail="LLMCore service is not available."
        )

    # Validate query parameter
    if not query or not query.strip():
        raise HTTPException(
            status_code=400,
            detail="A non-empty 'query' parameter is required."
        )

    try:
        logger.debug(f"Performing semantic search: query='{query}', collection='{collection_name}', k={k}")

        # Call the existing search_vector_store method
        search_results = await llmcore_instance.search_vector_store(
            query=query,
            k=k,
            collection_name=collection_name
            # Note: filter_metadata is not yet exposed in this simple GET endpoint
        )

        logger.debug(f"Semantic search completed successfully, found {len(search_results)} results")
        return search_results

    except (VectorStorageError, ConfigError) as e:
        logger.error(f"API Error during semantic search: {e}", exc_info=True)

        # Return a 404 if the collection is not found, otherwise a 500
        error_message = str(e).lower()
        if "collection not found" in error_message or "collection" in error_message and "not found" in error_message:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' not found."
            )

        raise HTTPException(
            status_code=500,
            detail=f"Error during semantic search: {str(e)}"
        )

    except Exception as e:
        logger.error(f"Unexpected server error during semantic search: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal server error occurred during semantic search."
        )
