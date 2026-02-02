"""
Unit tests for llmcore.memory.rag_utils module.

Tests the RAG utility functions for template rendering and document formatting.
"""

import pytest
from llmcore.models import ContextDocument
from llmcore.memory import rag_utils


class TestRenderPromptTemplate:
    """Tests for render_prompt_template function."""

    def test_render_prompt_template_simple(self):
        """Test basic template with {context} and {question} placeholders."""
        template = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        context = "Machine learning is a subset of AI."
        question = "What is machine learning?"
        
        result = rag_utils.render_prompt_template(template, context, question, None)
        
        assert "Machine learning is a subset of AI." in result
        assert "What is machine learning?" in result
        assert "{context}" not in result
        assert "{question}" not in result

    def test_render_prompt_template_custom_placeholders(self):
        """Test template with custom template values."""
        template = "Context: {context}\nQuestion: {question}\nInstructions: {instructions}"
        context = "Python is a programming language."
        question = "Tell me about Python?"
        custom_values = {"instructions": "Be concise"}
        
        result = rag_utils.render_prompt_template(template, context, question, custom_values)
        
        assert "Python is a programming language." in result
        assert "Tell me about Python?" in result
        assert "Be concise" in result
        assert "{instructions}" not in result

    def test_render_prompt_template_empty_context(self):
        """Test template with empty context."""
        template = "Context: {context}\nQuestion: {question}"
        context = ""
        question = "What is AI?"
        
        result = rag_utils.render_prompt_template(template, context, question, None)
        
        assert "Context: \nQuestion: What is AI?" in result

    def test_render_prompt_template_empty_question(self):
        """Test template with empty question."""
        template = "Context: {context}\nQuestion: {question}"
        context = "AI helps with automation."
        question = ""
        
        result = rag_utils.render_prompt_template(template, context, question, None)
        
        assert "Context: AI helps with automation.\nQuestion: " in result

    def test_render_prompt_template_empty_both(self):
        """Test template with both context and question empty."""
        template = "Context: {context}\nQuestion: {question}"
        context = ""
        question = ""
        
        result = rag_utils.render_prompt_template(template, context, question, None)
        
        assert result == "Context: \nQuestion: "

    def test_render_prompt_template_no_placeholders(self):
        """Test template with no placeholders."""
        template = "This is a static template with no variables."
        context = "Some context"
        question = "Some question"
        
        result = rag_utils.render_prompt_template(template, context, question, None)
        
        # Template should be returned unchanged
        assert result == template

    def test_render_prompt_template_multiline(self):
        """Test template with newlines and formatting."""
        template = """System: You are a helpful assistant.

Context:
{context}

User Question:
{question}

Assistant Response:"""
        context = "Relevant background information here."
        question = "What is your expertise?"
        
        result = rag_utils.render_prompt_template(template, context, question, None)
        
        assert "System: You are a helpful assistant." in result
        assert "Relevant background information here." in result
        assert "What is your expertise?" in result
        assert "Assistant Response:" in result

    def test_render_prompt_template_special_characters(self):
        """Test template with special characters in values."""
        template = "Context: {context}\n\nQ&A: {question}"
        context = "Data & ML (machine learning)"
        question = "What's the difference?"
        
        result = rag_utils.render_prompt_template(template, context, question, None)
        
        assert "Data & ML (machine learning)" in result
        assert "What's the difference?" in result

    def test_render_prompt_template_with_newlines_in_content(self):
        """Test template when context or question contain newlines."""
        template = "Context:\n{context}\n\nQuestion:\n{question}"
        context = "Line 1\nLine 2\nLine 3"
        question = "Part 1\nPart 2"
        
        result = rag_utils.render_prompt_template(template, context, question, None)
        
        assert "Line 1\nLine 2\nLine 3" in result
        assert "Part 1\nPart 2" in result

    def test_render_prompt_template_with_custom_values(self):
        """Test template with multiple custom template values."""
        template = "System: {system}\nContext: {context}\nQuestion: {question}"
        context = "Some context"
        question = "Some question"
        custom_values = {"system": "You are helpful"}
        
        result = rag_utils.render_prompt_template(template, context, question, custom_values)
        
        assert "You are helpful" in result
        assert "{system}" not in result


class TestFormatRagDocsForContext:
    """Tests for format_rag_docs_for_context function."""

    def test_format_rag_docs_empty_list(self):
        """Test formatting empty document list."""
        documents = []
        
        result = rag_utils.format_rag_docs_for_context(documents)
        
        # Should return empty string
        assert result == ""

    def test_format_rag_docs_single_document(self):
        """Test formatting a single document."""
        documents = [
            ContextDocument(
                id="doc1",
                content="Machine learning is a field of AI.",
                metadata={"source": "article.pdf"},
                score=0.95,
            )
        ]
        
        result = rag_utils.format_rag_docs_for_context(documents)
        
        assert isinstance(result, str)
        assert "Machine learning is a field of AI." in result
        assert "Retrieved Relevant Documents" in result

    def test_format_rag_docs_multiple_documents(self):
        """Test formatting multiple documents."""
        documents = [
            ContextDocument(
                id="doc1",
                content="Machine learning basics.",
                metadata={"source": "doc1.pdf"},
                score=0.95,
            ),
            ContextDocument(
                id="doc2",
                content="Deep learning advanced.",
                metadata={"source": "doc2.pdf"},
                score=0.87,
            ),
        ]
        
        result = rag_utils.format_rag_docs_for_context(documents)
        
        assert "Machine learning basics." in result
        assert "Deep learning advanced." in result
        assert "Context Document 1:" in result
        assert "Context Document 2:" in result

    def test_format_rag_docs_with_metadata(self):
        """Test formatting documents with various metadata."""
        documents = [
            ContextDocument(
                id="doc1",
                content="Supervised learning requires labels.",
                metadata={
                    "source": "textbook.pdf",
                    "start_line": 42,
                },
                score=0.93,
            )
        ]
        
        result = rag_utils.format_rag_docs_for_context(documents)
        
        assert "Supervised learning requires labels." in result
        assert len(result) > len("Supervised learning requires labels.")

    def test_format_rag_docs_score_sorting(self):
        """Test that documents are sorted by score (lower distance first)."""
        documents = [
            ContextDocument(
                id="doc3",
                content="Low relevance",
                metadata={"source": "low.pdf"},
                score=0.95,  # Higher distance (worse match)
            ),
            ContextDocument(
                id="doc1",
                content="High relevance",
                metadata={"source": "high.pdf"},
                score=0.65,  # Lower distance (better match)
            ),
            ContextDocument(
                id="doc2",
                content="Medium relevance",
                metadata={"source": "med.pdf"},
                score=0.80,  # Medium distance
            ),
        ]
        
        result = rag_utils.format_rag_docs_for_context(documents)
        
        # Lower score (better match) should appear first
        pos_high = result.find("High relevance")
        pos_med = result.find("Medium relevance")
        pos_low = result.find("Low relevance")
        
        assert pos_high < pos_med < pos_low

    def test_format_rag_docs_with_multiline_content(self):
        """Test formatting documents with multiline content."""
        documents = [
            ContextDocument(
                id="doc1",
                content="Line 1\nLine 2\nLine 3\nLine 4",
                metadata={"source": "multiline.pdf"},
                score=0.90,
            )
        ]
        
        result = rag_utils.format_rag_docs_for_context(documents)
        
        # Content should be joined into single line
        assert "Line 1 Line 2 Line 3 Line 4" in result or "Line 1" in result

    def test_format_rag_docs_with_source_file_path(self):
        """Test formatting with source_file_path_relative metadata."""
        documents = [
            ContextDocument(
                id="doc1",
                content="Some content",
                metadata={"source_file_path_relative": "docs/guide.md"},
                score=0.92,
            )
        ]
        
        result = rag_utils.format_rag_docs_for_context(documents)
        
        assert "docs/guide.md" in result or "File:" in result

    def test_format_rag_docs_with_line_numbers(self):
        """Test formatting with line number metadata."""
        documents = [
            ContextDocument(
                id="doc1",
                content="Code snippet",
                metadata={"source": "code.py", "start_line": 123},
                score=0.88,
            )
        ]
        
        result = rag_utils.format_rag_docs_for_context(documents)
        
        assert "123" in result or "Line:" in result

    def test_format_rag_docs_none_score(self):
        """Test formatting documents with None score."""
        documents = [
            ContextDocument(
                id="doc1",
                content="Content without score",
                metadata={"source": "noscore.pdf"},
                score=None,
            )
        ]
        
        result = rag_utils.format_rag_docs_for_context(documents)
        
        assert isinstance(result, str)
        assert "Content without score" in result

    def test_format_rag_docs_unicode_content(self):
        """Test formatting documents with unicode content."""
        documents = [
            ContextDocument(
                id="doc1",
                content="Introduction to AI & ML - 中文 support 🚀",
                metadata={"source": "unicode.pdf"},
                score=0.91,
            )
        ]
        
        result = rag_utils.format_rag_docs_for_context(documents)
        
        assert "AI & ML" in result
        assert "中文" in result or "support" in result


class TestRagUtilsEdgeCases:
    """Test edge cases and error scenarios."""

    def test_format_rag_docs_with_empty_metadata(self):
        """Test handling of documents with empty metadata."""
        documents = [
            ContextDocument(
                id="doc1",
                content="Content with empty metadata",
                metadata={},
                score=0.85,
            )
        ]
        
        result = rag_utils.format_rag_docs_for_context(documents)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert "doc1" in result  # Should fall back to document ID

    def test_format_rag_docs_very_long_content(self):
        """Test handling documents with very long content."""
        long_content = "This is a long document. " * 100
        documents = [
            ContextDocument(
                id="doc1",
                content=long_content,
                metadata={"source": "long.pdf"},
                score=0.80,
            )
        ]
        
        result = rag_utils.format_rag_docs_for_context(documents)
        
        assert len(result) > 0
        assert "This is a long document." in result

    def test_render_template_with_only_custom_values(self):
        """Test rendering with custom values and no standard placeholders."""
        template = "Task: {task}\nInstructions: {instructions}"
        context = "ignored"
        question = "also ignored"
        custom_values = {"task": "Analyze", "instructions": "Be brief"}
        
        result = rag_utils.render_prompt_template(template, context, question, custom_values)
        
        assert "Analyze" in result
        assert "Be brief" in result
        assert "{task}" not in result
        assert "{instructions}" not in result
