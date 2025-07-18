# llmcore/src/llmcore/task_master/tasks/ingestion.py
"""
Asynchronous data ingestion tasks for the TaskMaster service.

This module contains arq tasks for processing data ingestion requests
including file uploads, directory (ZIP) processing, and Git repository cloning.
These tasks run in the background worker process and report progress through
the task status system.
"""

import asyncio
import json
import logging
import tempfile
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from llmcore.api import LLMCore

logger = logging.getLogger(__name__)

# Import ingestion dependencies with fallback handling
APYKATU_AVAILABLE = False
GITPYTHON_AVAILABLE = False
ApykatuAppConfig = None
IngestionPipeline = None
apykatu_process_file_path_api = None
ApykatuConfigError = None
ApykatuProcessedChunk = None
ApykatuProcessingStats = None
git = None

try:
    from apykatu.pipelines.ingest import IngestionPipeline as ApykatuIngestionPipeline
    from apykatu.api import process_file_path as apykatu_api_process_file_path
    from apykatu.config.models import AppConfig as PyApykatuAppConfig
    from apykatu.config.models import ConfigError as PyApykatuConfigError
    from apykatu.api_models import ProcessedChunk as PyApykatuProcessedChunk
    from apykatu.api_models import ProcessingStats as PyApykatuProcessingStats
    APYKATU_AVAILABLE = True
    ApykatuAppConfig = PyApykatuAppConfig
    IngestionPipeline = ApykatuIngestionPipeline
    apykatu_process_file_path_api = apykatu_api_process_file_path
    ApykatuConfigError = PyApykatuConfigError
    ApykatuProcessedChunk = PyApykatuProcessedChunk
    ApykatuProcessingStats = PyApykatuProcessingStats
except ImportError:
    logger.warning("Apykatu library not found. Ingestion features will be disabled.")

try:
    import git as pygit
    GITPYTHON_AVAILABLE = True
    git = pygit
except ImportError:
    logger.warning("GitPython library not found. Git ingestion will be disabled.")


def _get_apykatu_config_for_ingestion(llmcore_instance: LLMCore, collection_name: str) -> Optional[Any]:
    """
    Prepares ApykatuAppConfig for an ingestion run.

    Args:
        llmcore_instance: The LLMCore instance to use for configuration
        collection_name: The target collection name for this ingestion task

    Returns:
        An ApykatuAppConfig object configured for the ingestion, or None if configuration fails
    """
    if not llmcore_instance or not llmcore_instance.config:
        logger.error("LLMCore instance or its config not available for Apykatu config preparation.")
        return None

    if not APYKATU_AVAILABLE or ApykatuAppConfig is None or ApykatuConfigError is None:
        logger.error("Apykatu library or its core components not available for config preparation.")
        return None

    apykatu_settings_from_llmcore_raw = llmcore_instance.config.get('apykatu', {})
    apykatu_settings_from_llmcore: Dict[str, Any]

    if hasattr(apykatu_settings_from_llmcore_raw, 'as_dict'):
        apykatu_settings_from_llmcore = apykatu_settings_from_llmcore_raw.as_dict()
    elif isinstance(apykatu_settings_from_llmcore_raw, dict):
        apykatu_settings_from_llmcore = apykatu_settings_from_llmcore_raw
    else:
        logger.error(f"LLMCore's 'apykatu' config section is not a dictionary or Confy object. Type: {type(apykatu_settings_from_llmcore_raw)}")
        apykatu_settings_from_llmcore = {}

    logger.debug(f"Apykatu settings from LLMCore for ingestion: {apykatu_settings_from_llmcore}")

    try:
        from apykatu.config.confy_loader import load_app_config_with_confy as load_apykatu_config

        final_apykatu_config_tuple = load_apykatu_config(
            cli_config_file_path=None,
            cli_overrides=apykatu_settings_from_llmcore
        )
        final_apykatu_config: Any = final_apykatu_config_tuple[0]

    except ApykatuConfigError as e_conf:
        logger.error(f"ApykatuConfigError during Apykatu config loading: {e_conf}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading Apykatu config: {e}", exc_info=True)
        return None

    # Override DB path and collection name from LLMCore's main vector store config
    llmcore_vector_db_path = llmcore_instance.config.get("storage.vector.path")
    if llmcore_vector_db_path:
        final_apykatu_config.database.path = Path(llmcore_vector_db_path).expanduser().resolve()
    else:
        logger.warning("LLMCore's storage.vector.path is not set. Apykatu will use its default DB path.")

    final_apykatu_config.database.collection_name = collection_name
    logger.info(f"Apykatu config prepared for ingestion. Collection: '{collection_name}', DB Path: '{final_apykatu_config.database.path}'")
    return final_apykatu_config


async def ingest_data_task(ctx, ingestion_params: dict):
    """
    Main ingestion task that processes different types of data ingestion requests.

    Args:
        ctx: arq context containing shared resources like the LLMCore instance
        ingestion_params: Dictionary containing ingestion parameters

    Returns:
        Dictionary with ingestion results and statistics
    """
    llmcore: LLMCore = ctx['llmcore_instance']
    ingest_type = ingestion_params.get('ingest_type')
    collection_name = ingestion_params.get('collection_name')

    logger.info(f"Starting ingestion task: type={ingest_type}, collection={collection_name}")

    if not APYKATU_AVAILABLE:
        error_msg = "Apykatu library not available for ingestion processing"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": error_msg,
            "files_processed": 0,
            "chunks_added": 0,
            "error_messages": [error_msg]
        }

    # Prepare Apykatu configuration
    apykatu_cfg = _get_apykatu_config_for_ingestion(llmcore, collection_name)
    if not apykatu_cfg:
        error_msg = "Failed to prepare Apykatu configuration"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": error_msg,
            "files_processed": 0,
            "chunks_added": 0,
            "error_messages": [error_msg]
        }

    try:
        if ingest_type == "file":
            return await _process_file_ingestion(llmcore, ingestion_params, apykatu_cfg)
        elif ingest_type == "dir_zip":
            return await _process_dir_zip_ingestion(llmcore, ingestion_params, apykatu_cfg)
        elif ingest_type == "git":
            return await _process_git_ingestion(llmcore, ingestion_params, apykatu_cfg)
        else:
            error_msg = f"Unsupported ingestion type: {ingest_type}"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg,
                "files_processed": 0,
                "chunks_added": 0,
                "error_messages": [error_msg]
            }

    except Exception as e:
        error_msg = f"Unexpected error during {ingest_type} ingestion: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "status": "error",
            "message": error_msg,
            "files_processed": 0,
            "chunks_added": 0,
            "error_messages": [error_msg]
        }


async def _process_file_ingestion(llmcore: LLMCore, params: dict, apykatu_cfg: Any) -> dict:
    """Process individual file uploads for ingestion."""
    file_paths = params.get('file_paths', [])
    collection_name = params.get('collection_name')

    if not file_paths:
        return {
            "status": "error",
            "message": "No file paths provided for file ingestion",
            "files_processed": 0,
            "chunks_added": 0,
            "error_messages": ["No file paths provided"]
        }

    total_files = len(file_paths)
    overall_chunks_added = 0
    files_processed_successfully = 0
    files_with_errors = 0
    error_messages = []

    logger.info(f"Processing {total_files} files for collection '{collection_name}'")

    for i, file_path_str in enumerate(file_paths):
        file_path = Path(file_path_str)
        filename = file_path.name

        try:
            if not file_path.exists():
                error_msg = f"File not found: {filename}"
                error_messages.append(error_msg)
                files_with_errors += 1
                continue

            logger.info(f"Processing file ({i+1}/{total_files}): '{filename}'")

            # Process file with Apykatu
            processed_chunks, api_stats_obj = await apykatu_process_file_path_api(
                file_path=file_path,
                config=apykatu_cfg,
                generate_embeddings=True
            )

            if api_stats_obj.error_messages:
                error_msg = f"File '{filename}': " + "; ".join(api_stats_obj.error_messages)
                error_messages.append(error_msg)
                files_with_errors += 1
                logger.warning(f"Errors processing file '{filename}': {error_msg}")
                continue

            if processed_chunks:
                docs_for_llmcore = []
                for pc in processed_chunks:
                    if pc.embedding_data and pc.embedding_data.get("vector"):
                        meta_to_store = pc.metadata_from_apykatu.model_dump() if hasattr(pc.metadata_from_apykatu, 'model_dump') else pc.metadata_from_apykatu
                        docs_for_llmcore.append({
                            "id": pc.semantiscan_chunk_id,
                            "content": pc.content_text,
                            "embedding": pc.embedding_data["vector"],
                            "metadata": meta_to_store
                        })

                if docs_for_llmcore:
                    added_ids = await llmcore.add_documents_to_vector_store(
                        documents=docs_for_llmcore,
                        collection_name=collection_name
                    )
                    chunks_this_file = len(added_ids)
                    overall_chunks_added += chunks_this_file
                    files_processed_successfully += 1
                    logger.info(f"Successfully added {chunks_this_file} chunks from file '{filename}'")
                else:
                    error_msg = f"File '{filename}': No processable chunks with embeddings found"
                    error_messages.append(error_msg)
                    files_with_errors += 1
            else:
                error_msg = f"File '{filename}': No chunks produced by Apykatu"
                error_messages.append(error_msg)
                files_with_errors += 1

        except Exception as e:
            error_msg = f"File '{filename}': {str(e)}"
            error_messages.append(error_msg)
            files_with_errors += 1
            logger.error(f"Error processing file '{filename}': {e}", exc_info=True)
        finally:
            # Clean up temporary file if it exists
            try:
                if file_path.exists():
                    file_path.unlink()
            except OSError as e_unlink:
                logger.warning(f"Could not delete temporary file {file_path}: {e_unlink}")

    # Determine overall status
    if files_with_errors == 0:
        status = "success"
    elif files_processed_successfully > 0:
        status = "completed_with_some_errors"
    else:
        status = "completed_with_all_errors"

    return {
        "status": status,
        "message": f"File ingestion completed for collection '{collection_name}'",
        "total_files_submitted": total_files,
        "files_processed_successfully": files_processed_successfully,
        "files_with_errors": files_with_errors,
        "total_chunks_added_to_db": overall_chunks_added,
        "collection_name": collection_name,
        "error_messages": error_messages
    }


async def _process_dir_zip_ingestion(llmcore: LLMCore, params: dict, apykatu_cfg: Any) -> dict:
    """Process directory ZIP file ingestion."""
    zip_file_path = params.get('zip_file_path')
    collection_name = params.get('collection_name')
    repo_name = params.get('repo_name', 'uploaded_directory')

    if not zip_file_path or not Path(zip_file_path).exists():
        return {
            "status": "error",
            "message": "ZIP file not found or not provided",
            "files_processed_successfully": "N/A",
            "total_chunks_added_to_db": "N/A",
            "error_messages": ["ZIP file not found"]
        }

    with tempfile.TemporaryDirectory(prefix="llmcore_ingest_dir_zip_") as temp_dir_str:
        temp_dir_path = Path(temp_dir_str)

        try:
            # Extract ZIP file
            zip_path = Path(zip_file_path)
            extracted_dir_path = temp_dir_path / "unzipped_content"
            extracted_dir_path.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extracted_dir_path)

            logger.info(f"Extracted ZIP '{zip_path.name}' to '{extracted_dir_path}'. Starting Apykatu pipeline.")

            # Run Apykatu ingestion pipeline
            pipeline = IngestionPipeline(config=apykatu_cfg, progress_context=None)
            await pipeline.run(
                repo_path=extracted_dir_path,
                repo_name=repo_name,
                git_ref="HEAD",
                mode='snapshot'
            )

            return {
                "status": "success",
                "message": f"Directory (ZIP) '{zip_path.name}' ingestion completed for collection '{collection_name}'",
                "files_processed_successfully": "N/A (directory)",
                "total_chunks_added_to_db": "N/A (directory)",
                "collection_name": collection_name,
                "error_messages": []
            }

        except Exception as e:
            error_msg = f"Error during directory ZIP ingestion: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "status": "error",
                "message": error_msg,
                "files_processed_successfully": "N/A",
                "total_chunks_added_to_db": "N/A",
                "error_messages": [error_msg]
            }
        finally:
            # Clean up uploaded ZIP file
            try:
                if Path(zip_file_path).exists():
                    Path(zip_file_path).unlink()
            except OSError as e:
                logger.warning(f"Could not delete temporary ZIP file {zip_file_path}: {e}")


async def _process_git_ingestion(llmcore: LLMCore, params: dict, apykatu_cfg: Any) -> dict:
    """Process Git repository ingestion."""
    if not GITPYTHON_AVAILABLE:
        return {
            "status": "error",
            "message": "GitPython library not available for Git ingestion",
            "files_processed_successfully": "N/A",
            "total_chunks_added_to_db": "N/A",
            "error_messages": ["GitPython library not available"]
        }

    git_url = params.get('git_url')
    collection_name = params.get('collection_name')
    repo_name = params.get('repo_name')
    git_ref = params.get('git_ref', 'HEAD')

    if not git_url or not repo_name:
        return {
            "status": "error",
            "message": "Missing Git URL or repository identifier",
            "files_processed_successfully": "N/A",
            "total_chunks_added_to_db": "N/A",
            "error_messages": ["Missing Git URL or repository identifier"]
        }

    with tempfile.TemporaryDirectory(prefix="llmcore_ingest_git_") as temp_dir_str:
        temp_dir_path = Path(temp_dir_str)

        try:
            # Clone repository
            cloned_repo_path = temp_dir_path / "cloned_repo"
            cloned_repo_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"Cloning Git repo from '{git_url}' (ref: {git_ref}) to '{cloned_repo_path}'")

            # Use asyncio.to_thread for the synchronous git operation
            await asyncio.to_thread(
                git.Repo.clone_from,
                git_url,
                str(cloned_repo_path),
                branch=git_ref if git_ref != "HEAD" else None,
                depth=1
            )

            logger.info(f"Git repo '{repo_name}' cloned. Starting Apykatu pipeline.")

            # Run Apykatu ingestion pipeline
            pipeline = IngestionPipeline(config=apykatu_cfg, progress_context=None)
            await pipeline.run(
                repo_path=cloned_repo_path,
                repo_name=repo_name,
                git_ref=git_ref,
                mode='snapshot'
            )

            return {
                "status": "success",
                "message": f"Git repository '{repo_name}' (ref: {git_ref}) ingestion completed for collection '{collection_name}'",
                "files_processed_successfully": "N/A (git)",
                "total_chunks_added_to_db": "N/A (git)",
                "collection_name": collection_name,
                "error_messages": []
            }

        except Exception as e:
            error_msg = f"Error during Git ingestion: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "status": "error",
                "message": error_msg,
                "files_processed_successfully": "N/A",
                "total_chunks_added_to_db": "N/A",
                "error_messages": [error_msg]
            }
