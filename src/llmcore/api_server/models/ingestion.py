# llmcore/src/llmcore/api_server/models/ingestion.py
"""
Pydantic models for data ingestion API requests and responses.

This module defines the data models used for ingestion operations,
including request validation and response formatting for different
types of ingestion (files, directories, Git repositories).
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class FileIngestionRequest(BaseModel):
    """Request model for file-based ingestion."""
    ingest_type: str = Field(..., description="Type of ingestion, should be 'file'")
    collection_name: str = Field(..., description="Target collection name for the ingested data")
    file_paths: List[str] = Field(..., description="List of temporary file paths to process")


class DirectoryZipIngestionRequest(BaseModel):
    """Request model for directory ZIP ingestion."""
    ingest_type: str = Field(..., description="Type of ingestion, should be 'dir_zip'")
    collection_name: str = Field(..., description="Target collection name for the ingested data")
    zip_file_path: str = Field(..., description="Path to the uploaded ZIP file")
    repo_name: Optional[str] = Field(None, description="Optional identifier name for the directory")


class GitIngestionRequest(BaseModel):
    """Request model for Git repository ingestion."""
    ingest_type: str = Field(..., description="Type of ingestion, should be 'git'")
    collection_name: str = Field(..., description="Target collection name for the ingested data")
    git_url: str = Field(..., description="Git repository URL to clone")
    repo_name: str = Field(..., description="Identifier name for the repository")
    git_ref: Optional[str] = Field("HEAD", description="Git branch, tag, or commit to clone")


class IngestionSubmitRequest(BaseModel):
    """Generic ingestion request that can handle different ingestion types."""
    ingest_type: str = Field(..., description="Type of ingestion: 'file', 'dir_zip', or 'git'")
    collection_name: str = Field(..., description="Target collection name for the ingested data")

    # File ingestion specific
    file_paths: Optional[List[str]] = Field(None, description="List of temporary file paths (for file ingestion)")

    # Directory ZIP ingestion specific
    zip_file_path: Optional[str] = Field(None, description="Path to ZIP file (for dir_zip ingestion)")
    repo_name: Optional[str] = Field(None, description="Repository identifier name")

    # Git ingestion specific
    git_url: Optional[str] = Field(None, description="Git repository URL (for git ingestion)")
    git_ref: Optional[str] = Field("HEAD", description="Git branch, tag, or commit")


class IngestionSubmitResponse(BaseModel):
    """Response model for ingestion submission."""
    task_id: str = Field(..., description="Unique identifier for the background ingestion task")
    message: str = Field(..., description="Confirmation message about the submitted task")


class IngestionResult(BaseModel):
    """Model for ingestion task results."""
    status: str = Field(..., description="Final status: 'success', 'error', 'completed_with_some_errors', etc.")
    message: str = Field(..., description="Human-readable description of the result")
    collection_name: Optional[str] = Field(None, description="Target collection name")

    # Statistics (may be N/A for some ingestion types)
    total_files_submitted: Optional[Any] = Field(None, description="Number of files submitted for processing")
    files_processed_successfully: Optional[Any] = Field(None, description="Number of files processed successfully")
    files_with_errors: Optional[Any] = Field(None, description="Number of files that had errors")
    total_chunks_added_to_db: Optional[Any] = Field(None, description="Total number of chunks added to vector store")

    # Error details
    error_messages: Optional[List[str]] = Field([], description="List of error messages if any occurred")

    # Additional metadata
    details: Optional[Dict[str, Any]] = Field(None, description="Additional ingestion-specific details")
