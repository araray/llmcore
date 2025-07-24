# src/llmcore/api_server/routes/memory.py
"""
Memory-related API routes for the llmcore API server.

This module contains the implementation of memory system endpoints that provide
API access to the hierarchical memory architecture, starting with semantic memory.

UPDATED: Modified to use tenant-scoped database sessions for multi-tenant data isolation.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Request, HTTPException, Query, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ...api import LLMCore
from ...exceptions import VectorStorageError, ConfigError
from ...models import ContextDocument
from ..db import get_tenant_db_session  # NEW: Tenant-scoped database dependency

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
    ),
    db_session: AsyncSession = Depends(get_tenant_db_session)  # NEW: Tenant-scoped database session
) -> List[ContextDocument]:
    """
    Performs a similarity search against the semantic memory (vector store).

    This endpoint exposes the full functionality of the existing vector store
    search capability as a formal, queryable API endpoint. It represents the
    first pillar of the new hierarchical memory architecture.

    UPDATED: Now uses tenant-scoped database session to ensure data isolation.

    Args:
        request: The FastAPI request object containing app state
        query: The text query to search for in the vector store
        collection_name: Optional target collection name
        k: Number of results to return (1-20)
        db_session: Tenant-scoped database session for data isolation

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

        # NEW: Use tenant-scoped vector storage
        storage_manager = llmcore_instance._storage_manager
        tenant_vector_storage = storage_manager.get_vector_storage(db_session)

        # Generate embedding using the tenant-scoped embedding manager
        query_embedding = await llmcore_instance._embedding_manager.generate_embedding(query)

        # Perform similarity search with tenant-scoped storage
        search_results = await tenant_vector_storage.similarity_search(
            query_embedding=query_embedding,
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


@router.get("/memory/episodic/{session_id}", response_model=List)
async def get_episodic_memory(
    request: Request,
    session_id: str,
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum number of episodes to return"),
    offset: int = Query(default=0, ge=0, description="Number of episodes to skip"),
    db_session: AsyncSession = Depends(get_tenant_db_session)  # NEW: Tenant-scoped database session
) -> List:
    """
    Retrieves episodic memory (episodes) for a specific session.

    NEW: Added episodic memory endpoint with tenant-scoped data isolation.

    Args:
        request: The FastAPI request object containing app state
        session_id: The session ID to retrieve episodes for
        limit: Maximum number of episodes to return
        offset: Number of episodes to skip for pagination
        db_session: Tenant-scoped database session for data isolation

    Returns:
        List of Episode objects for the session

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

    try:
        logger.debug(f"Retrieving episodic memory for session '{session_id}' (limit={limit}, offset={offset})")

        # Use tenant-scoped storage manager to get episodes
        storage_manager = llmcore_instance._storage_manager
        episodes = await storage_manager.get_episodes(
            session_id=session_id,
            limit=limit,
            offset=offset,
            db_session=db_session
        )

        logger.debug(f"Retrieved {len(episodes)} episodes for session '{session_id}'")
        return episodes

    except Exception as e:
        logger.error(f"Error retrieving episodic memory for session '{session_id}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal server error occurred while retrieving episodic memory."
        )
