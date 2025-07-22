# src/llmcore/api_server/routes/chat.py
"""
Chat-related API routes for the llmcore API server.

This module contains the implementation of the /chat endpoint that provides
API access to the core LLMCore.chat() functionality.

UPDATED: Modified to use tenant-scoped database sessions for multi-tenant data isolation.
"""

import logging
from typing import Union

from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ...exceptions import LLMCoreError, ProviderError, ContextLengthError, ConfigError
from ..models import ChatRequest, ChatResponse
from ..db import get_tenant_db_session  # NEW: Tenant-scoped database dependency

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def handle_chat(
    request: Request,
    chat_request: ChatRequest,
    db_session: AsyncSession = Depends(get_tenant_db_session)  # NEW: Tenant-scoped database session
) -> Union[ChatResponse, StreamingResponse]:
    """
    Handles chat requests by forwarding them to the LLMCore instance.

    Supports both streaming and non-streaming responses based on the 'stream' parameter.

    UPDATED: Now uses tenant-scoped database session to ensure data isolation.

    Args:
        request: The FastAPI request object containing app state
        chat_request: The validated chat request payload
        db_session: Tenant-scoped database session for data isolation

    Returns:
        ChatResponse for non-streaming requests, StreamingResponse for streaming requests

    Raises:
        HTTPException: For various error conditions (400, 500, 503)
    """
    # Get the LLMCore instance from app state
    llmcore_instance = getattr(request.app.state, 'llmcore_instance', None)
    if not llmcore_instance:
        logger.error("LLMCore instance not found in app state")
        raise HTTPException(
            status_code=503,
            detail="LLMCore service is not available. The service may be starting up or experiencing issues."
        )

    try:
        # Prepare arguments for the chat method
        chat_params = chat_request.model_dump(exclude_none=True)

        # Extract and merge provider_kwargs into the main parameters
        provider_kwargs = chat_params.pop('provider_kwargs', {})
        chat_params.update(provider_kwargs)

        # NEW: Pass the tenant-scoped database session to the storage manager
        # This ensures all data operations are isolated to the tenant's schema
        storage_manager = llmcore_instance._storage_manager
        tenant_session_storage = storage_manager.get_session_storage(db_session)

        # Configure the session manager to use the tenant-scoped storage
        original_storage = llmcore_instance._session_manager._storage
        llmcore_instance._session_manager._storage = tenant_session_storage

        logger.debug(f"Calling LLMCore.chat with tenant-scoped storage: {list(chat_params.keys())}")

        try:
            # Call the LLMCore chat method with tenant-scoped storage
            response = await llmcore_instance.chat(**chat_params)

            if chat_request.stream:
                # For streaming responses, wrap the async generator in StreamingResponse
                logger.debug("Returning streaming response")
                return StreamingResponse(
                    response,  # response is an AsyncGenerator[str, None]
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    }
                )
            else:
                # For non-streaming responses, return the string wrapped in our response model
                logger.debug("Returning non-streaming response")
                return ChatResponse(
                    response=response,  # response is a str
                    session_id=chat_request.session_id
                )
        finally:
            # Restore the original storage to prevent side effects
            llmcore_instance._session_manager._storage = original_storage

    except (ProviderError, ContextLengthError, ConfigError) as e:
        # These are expected errors from LLMCore that should be returned as 400 Bad Request
        logger.warning(f"Client error during chat processing: {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except ValueError as e:
        # Parameter validation errors
        logger.warning(f"Parameter validation error: {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except LLMCoreError as e:
        # Other LLMCore errors - treat as server errors
        logger.error(f"LLMCore error during chat processing: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal LLMCore error: {str(e)}"
        )
    except Exception as e:
        # Unexpected errors - log with full stack trace and return generic error
        logger.error(f"Unexpected error during chat processing: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal server error occurred while processing your request."
        )
