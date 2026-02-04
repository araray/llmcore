# src/llmcore/memory/rag_utils.py
"""
Utility functions for Retrieval Augmented Generation (RAG) within the MemoryManager.

This module contains helper functions responsible for formatting retrieved documents
and rendering the final RAG prompt template, separating the specifics of RAG-based
prompt engineering from the main context assembly logic.
"""

import logging
from typing import Dict, List, Optional

from ..models import ContextDocument

logger = logging.getLogger(__name__)


def render_prompt_template(
    template_content: str,
    rag_context_str: str,
    question_str: str,
    custom_template_values: dict[str, str] | None,
) -> str:
    """
    Renders the loaded prompt template with the provided context and question.

    Args:
        template_content: The string content of the prompt template.
        rag_context_str: The formatted string of retrieved RAG documents.
        question_str: The user's original question.
        custom_template_values: A dictionary of additional key-value pairs
                                for custom template substitution.

    Returns:
        The fully rendered prompt string ready to be sent to the LLM.
    """
    rendered_prompt = template_content
    rendered_prompt = rendered_prompt.replace("{context}", rag_context_str)
    rendered_prompt = rendered_prompt.replace("{question}", question_str)
    if custom_template_values:
        for key, value in custom_template_values.items():
            rendered_prompt = rendered_prompt.replace(f"{{{key}}}", str(value))
    return rendered_prompt


def format_rag_docs_for_context(documents: list[ContextDocument]) -> str:
    """
    Formats a list of retrieved RAG documents into a single string for the prompt.

    Sorts documents by score, and includes metadata like source file and line numbers
    to provide the LLM with rich, traceable context.

    Args:
        documents: A list of ContextDocument objects retrieved from the vector store.

    Returns:
        A formatted string containing the content of all documents, ready to be
        injected into the RAG prompt template.
    """
    if not documents:
        return ""

    # Sort documents by score (lower is better for distance-based metrics like L2)
    sorted_documents = sorted(
        documents, key=lambda d: d.score if d.score is not None else float("inf")
    )

    context_parts = ["--- Retrieved Relevant Documents ---"]
    for i, doc in enumerate(sorted_documents):
        source_info_parts = []
        # Prioritize specific metadata keys for a clean source description
        if doc.metadata and doc.metadata.get("source_file_path_relative"):
            source_info_parts.append(f"File: {doc.metadata.get('source_file_path_relative')}")
        elif doc.metadata and doc.metadata.get("source"):
            source_info_parts.append(f"Source: {doc.metadata.get('source')}")
        else:
            source_info_parts.append(f"DocID: {doc.id[:12]}")  # Fallback to document ID

        if doc.metadata and doc.metadata.get("start_line"):
            source_info_parts.append(f"Line: {doc.metadata.get('start_line')}")

        score_info = f"(Score: {doc.score:.4f})" if doc.score is not None else ""
        header = f"Context Document {i + 1}: [{', '.join(source_info_parts)}] {score_info}"

        # Clean up content for better presentation in the prompt
        content_snippet = " ".join(doc.content.splitlines()).strip()

        context_parts.append(f"\n{header}\n{content_snippet}")

    context_parts.append("\n--- End Retrieved Documents ---")
    return "\n".join(context_parts)
