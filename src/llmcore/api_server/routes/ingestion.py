# llmcore/src/llmcore/api_server/routes/ingestion.py
"""
API routes for data ingestion operations.

This module provides REST endpoints for submitting asynchronous data ingestion tasks.
The actual ingestion processing is handled by the TaskMaster service via arq tasks.
"""

import logging
import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
# Note: secure_filename is a Werkzeug function, but this is FastAPI
# We'll use a simple filename sanitization instead
import re

from ..models.ingestion import IngestionSubmitResponse
from ..services.redis_client import get_redis_pool

logger = logging.getLogger(__name__)

# Create the router for ingestion endpoints
router = APIRouter()


def secure_filename(filename):
    """Simple filename sanitization for FastAPI context."""
    # Remove path separators and dangerous characters
    filename = re.sub(r'[^\w\s-.]', '', filename)
    return filename.strip()


@router.post("/submit", response_model=IngestionSubmitResponse)
async def submit_ingestion_task(
    ingest_type: str = Form(...),
    collection_name: str = Form(...),
    # File ingestion
    files: List[UploadFile] = File(default=[]),
    # Directory ZIP ingestion
    zip_file: UploadFile = File(default=None),
    repo_name: str = Form(default=None),
    # Git ingestion
    git_url: str = Form(default=None),
    git_ref: str = Form(default="HEAD"),
    redis_pool=Depends(get_redis_pool)
):
    """
    Submit a data ingestion task for asynchronous processing.

    This endpoint validates the request, saves any uploaded files to temporary locations,
    and enqueues an ingestion task for processing by the TaskMaster service.

    Args:
        ingest_type: Type of ingestion ('file', 'dir_zip', 'git')
        collection_name: Target collection name for the ingested data
        files: List of files for file ingestion
        zip_file: ZIP file for directory ingestion
        repo_name: Repository identifier name
        git_url: Git repository URL for git ingestion
        git_ref: Git branch/tag/commit reference
        redis_pool: Redis connection pool for task queue

    Returns:
        Response containing the task_id for monitoring the ingestion progress

    Raises:
        HTTPException: For validation errors or service unavailability
    """
    logger.info(f"Received ingestion request: type={ingest_type}, collection={collection_name}")

    if not ingest_type or not collection_name:
        raise HTTPException(
            status_code=400,
            detail="Missing required fields: ingest_type and collection_name"
        )

    # Prepare task parameters based on ingestion type
    task_params = {
        "ingest_type": ingest_type,
        "collection_name": collection_name
    }

    try:
        if ingest_type == "file":
            if not files or not any(f.filename for f in files):
                raise HTTPException(
                    status_code=400,
                    detail="No files provided for file ingestion"
                )

            # Save uploaded files to temporary locations
            temp_file_paths = []
            for uploaded_file in files:
                if uploaded_file.filename:
                    # Create temporary file
                    temp_file_fd, temp_file_path = tempfile.mkstemp(
                        prefix="llmcore_ingest_",
                        suffix=f"_{secure_filename(uploaded_file.filename)}"
                    )

                    try:
                        # Write uploaded content to temporary file
                        content = await uploaded_file.read()
                        with open(temp_file_fd, 'wb') as f:
                            f.write(content)
                        temp_file_paths.append(temp_file_path)
                        logger.debug(f"Saved uploaded file to temporary path: {temp_file_path}")
                    except Exception as e:
                        # Clean up on error
                        try:
                            Path(temp_file_path).unlink()
                        except:
                            pass
                        raise e

            task_params["file_paths"] = temp_file_paths

        elif ingest_type == "dir_zip":
            if not zip_file or not zip_file.filename:
                raise HTTPException(
                    status_code=400,
                    detail="No ZIP file provided for directory ingestion"
                )

            # Save ZIP file to temporary location
            temp_file_fd, temp_zip_path = tempfile.mkstemp(
                prefix="llmcore_ingest_zip_",
                suffix=f"_{secure_filename(zip_file.filename)}"
            )

            try:
                content = await zip_file.read()
                with open(temp_file_fd, 'wb') as f:
                    f.write(content)

                task_params["zip_file_path"] = temp_zip_path
                task_params["repo_name"] = repo_name or "uploaded_directory"
                logger.debug(f"Saved ZIP file to temporary path: {temp_zip_path}")
            except Exception as e:
                # Clean up on error
                try:
                    Path(temp_zip_path).unlink()
                except:
                    pass
                raise e

        elif ingest_type == "git":
            if not git_url or not repo_name:
                raise HTTPException(
                    status_code=400,
                    detail="Missing Git URL or repository identifier for Git ingestion"
                )

            task_params["git_url"] = git_url
            task_params["repo_name"] = repo_name
            task_params["git_ref"] = git_ref

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported ingestion type: {ingest_type}"
            )

        # Enqueue the ingestion task
        try:
            job = await redis_pool.enqueue_job(
                'ingest_data_task',
                task_params
            )
            task_id = job.job_id
            logger.info(f"Enqueued ingestion task {task_id} for {ingest_type} ingestion into collection '{collection_name}'")

            return IngestionSubmitResponse(
                task_id=task_id,
                message=f"Ingestion task submitted successfully. Task ID: {task_id}"
            )

        except Exception as e:
            logger.error(f"Failed to enqueue ingestion task: {e}", exc_info=True)
            # Clean up any temporary files on enqueue failure
            _cleanup_temp_files(task_params)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to submit ingestion task: {str(e)}"
            )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error in ingestion submission: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during ingestion submission: {str(e)}"
        )


def _cleanup_temp_files(task_params: dict) -> None:
    """Clean up temporary files in case of errors."""
    try:
        # Clean up file paths
        if "file_paths" in task_params:
            for file_path in task_params["file_paths"]:
                try:
                    Path(file_path).unlink()
                except Exception as e:
                    logger.warning(f"Could not clean up temporary file {file_path}: {e}")

        # Clean up ZIP file
        if "zip_file_path" in task_params:
            try:
                Path(task_params["zip_file_path"]).unlink()
            except Exception as e:
                logger.warning(f"Could not clean up temporary ZIP file {task_params['zip_file_path']}: {e}")

    except Exception as e:
        logger.warning(f"Error during temporary file cleanup: {e}")


@router.get("/status")
async def get_ingestion_status():
    """
    Get the status of the ingestion service.

    Returns:
        Dictionary with service status information
    """
    # This could be expanded to check Apykatu availability, etc.
    return {
        "service": "ingestion",
        "status": "operational",
        "supported_types": ["file", "dir_zip", "git"]
    }
