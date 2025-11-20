# External RAG Engine Integration Guide

## Overview

LLMCore now provides first-class support for external Retrieval Augmented Generation (RAG) engines like `semantiscan`. This guide explains how to integrate your RAG engine with LLMCore as the LLM backend.

**Version**: 0.24.0+
 **Target Audience**: Developers building RAG pipelines with LLMCore

------

## Table of Contents

1. [Integration Philosophy](#integration-philosophy)
2. [Quick Start](#quick-start)
3. [Integration Patterns](#integration-patterns)
4. [API Reference](#api-reference)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

------

## Integration Philosophy

### The External RAG Pattern

When integrating an external RAG engine with LLMCore:

1. **Your RAG engine** controls document retrieval, chunking, and ranking
2. **Your RAG engine** constructs the final prompt with context
3. **LLMCore** handles LLM provider abstraction, session management, and response generation

**Critical Rule**: Always set `enable_rag=False` when calling LLMCore from an external RAG engine to prevent double-RAG scenarios.

### Why This Approach?

- **Separation of concerns**: RAG logic stays in specialized engines, LLM logic in LLMCore
- **Flexibility**: Full control over RAG algorithms and prompt construction
- **Performance**: No redundant retrieval operations
- **Composability**: Mix and match RAG engines with different LLM providers

------

## Quick Start

### Basic Integration (5 minutes)

```python
from llmcore.api import LLMCore
from my_rag_engine import retrieve_documents, format_context

# 1. Initialize LLMCore
llm = await LLMCore.create()

# 2. User query
query = "What is the purpose of async context managers?"

# 3. Your RAG engine retrieves documents
relevant_docs = await retrieve_documents(query, top_k=5)

# 4. Your RAG engine constructs the prompt
context = format_context(relevant_docs)
full_prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""

# 5. Call LLMCore with constructed prompt
response = await llm.chat(
    message=full_prompt,
    enable_rag=False,  # CRITICAL: Prevent double-RAG
    stream=False
)

print(response)
```

------

## Integration Patterns

LLMCore supports three integration patterns, ordered by complexity:

### Pattern 1: Fully-Constructed Prompts (Recommended)

**Best for**: Simple integrations, maximum control over prompt structure

```python
# Your RAG engine builds the complete prompt
query = "How do I configure LLMCore?"
retrieved_context = retrieve_and_format_documents(query)

# Construct full prompt exactly as you want it
full_prompt = f"""You are an AI assistant helping with LLMCore configuration.

Context from documentation:
{retrieved_context}

User question: {query}

Provide a detailed answer based only on the context above."""

# Pass to LLMCore
response = await llm.chat(
    message=full_prompt,
    enable_rag=False,  # Never forget this!
    stream=False
)
```

**Pros**:

- Complete control over prompt format
- Simplest to implement
- Works with any LLM provider

**Cons**:

- Manual token management if needed

------

### Pattern 2: Structured Context (explicitly_staged_items)

**Best for**: Complex contexts, automatic token management, metadata preservation

```python
from llmcore.models import ContextItem, ContextItemType

# Convert your retrieved documents to ContextItems
context_items = [
    ContextItem(
        id=f"doc_{i}",
        type=ContextItemType.RAG_SNIPPET,
        content=doc.content,
        source_id=doc.source_path,
        metadata={
            "score": doc.relevance_score,
            "chunk_id": doc.chunk_id,
            "rag_engine": "semantiscan"
        }
    )
    for i, doc in enumerate(retrieved_docs)
]

# Pass structured context to LLMCore
response = await llm.chat(
    message=query,  # Just the query, not full prompt
    explicitly_staged_items=context_items,
    enable_rag=False,
    stream=False
)
```

**Pros**:

- LLMCore handles token counting and truncation
- Preserves document metadata
- Better for debugging and observability

**Cons**:

- Slightly more complex
- Less control over exact prompt format

------

### Pattern 3: Context Preview (Token Estimation)

**Best for**: Cost estimation, debugging, UI displays

```python
# Preview what would be sent to the LLM without making API call
preview = await llm.preview_context_for_chat(
    current_user_query=query,
    explicitly_staged_items=context_items,
    enable_rag=False
)

# Check token counts before committing
if preview['final_token_count'] > 8000:
    print(f"Warning: Large context ({preview['final_token_count']} tokens)")
    # Maybe reduce context or warn user
else:
    # Proceed with actual chat
    response = await llm.chat(
        message=query,
        explicitly_staged_items=context_items,
        enable_rag=False
    )
```

**Use cases**:

- Estimate costs before generation
- Validate prompt construction
- Debug context issues
- Show token usage in UI

------

## API Reference

### Enhanced `chat()` Method

```python
async def chat(
    self,
    message: str,
    *,
    # Standard parameters
    session_id: Optional[str] = None,
    system_message: Optional[str] = None,
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None,
    stream: bool = False,
    save_session: bool = True,
    
    # Internal RAG (set to False for external RAG)
    enable_rag: bool = False,
    rag_retrieval_k: Optional[int] = None,
    rag_collection_name: Optional[str] = None,
    rag_metadata_filter: Optional[Dict[str, Any]] = None,
    
    # NEW: External RAG support
    active_context_item_ids: Optional[List[str]] = None,
    explicitly_staged_items: Optional[List[Union[Message, ContextItem]]] = None,
    prompt_template_values: Optional[Dict[str, str]] = None,
    
    # Tool calling and provider-specific
    tools: Optional[List[Tool]] = None,
    tool_choice: Optional[str] = None,
    **provider_kwargs
) -> Union[str, AsyncGenerator[str, None]]:
```

### Key Parameters for External RAG

| Parameter                 | Type                                | Purpose                                                      |
| ------------------------- | ----------------------------------- | ------------------------------------------------------------ |
| `message`                 | `str`                               | Your constructed prompt or just the query (depending on pattern) |
| `enable_rag`              | `bool`                              | **MUST be False** for external RAG to prevent double-RAG     |
| `explicitly_staged_items` | `List[Union[Message, ContextItem]]` | Structured context from your retrieval                       |
| `active_context_item_ids` | `List[str]`                         | IDs of items in session workspace to include                 |
| `prompt_template_values`  | `Dict[str, str]`                    | Custom values for prompt template placeholders               |

### Preview Context Method

```python
async def preview_context_for_chat(
    self,
    current_user_query: str,
    *,
    session_id: Optional[str] = None,
    system_message: Optional[str] = None,
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None,
    active_context_item_ids: Optional[List[str]] = None,
    explicitly_staged_items: Optional[List[Union[Message, ContextItem]]] = None,
    enable_rag: bool = False,
    rag_retrieval_k: Optional[int] = None,
    rag_collection_name: Optional[str] = None,
    rag_metadata_filter: Optional[Dict[str, Any]] = None,
    prompt_template_values: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
```

**Returns**:

```python
{
    "prepared_messages": List[Message],  # Exact messages to be sent
    "final_token_count": int,            # Total token count
    "max_tokens_for_model": int,         # Model's context limit
    "truncation_actions_taken": dict,    # Details of any truncation
    "rag_documents_used": List[...],     # If enable_rag=True
    "rendered_rag_template_content": str # If RAG template used
}
```

------

## Best Practices

### 1. Always Disable Internal RAG

```python
# ❌ BAD: Will cause double-RAG
response = await llm.chat(
    message=prompt_with_your_context,
    enable_rag=True  # DON'T DO THIS!
)

# ✅ GOOD: External RAG only
response = await llm.chat(
    message=prompt_with_your_context,
    enable_rag=False
)
```

### 2. Use Sessions for Conversations

```python
# Maintain conversation history across queries
session_id = f"user_{user_id}_conversation"

for query in user_queries:
    # Retrieve context for this query
    context = retrieve_documents(query)
    prompt = build_prompt(context, query)
    
    # Session preserves history
    response = await llm.chat(
        message=prompt,
        session_id=session_id,  # Reuse same session
        enable_rag=False,
        save_session=True
    )
```

### 3. Preview Before Generation (For Production)

```python
# Estimate tokens first
preview = await llm.preview_context_for_chat(
    current_user_query=query,
    explicitly_staged_items=context_items,
    enable_rag=False
)

# Check if within limits
if preview['final_token_count'] > model_context_limit * 0.9:
    # Reduce context
    context_items = context_items[:len(context_items)//2]

# Then generate
response = await llm.chat(
    message=query,
    explicitly_staged_items=context_items,
    enable_rag=False
)
```

### 4. Handle Streaming Properly

```python
# For real-time UIs
response_stream = await llm.chat(
    message=prompt,
    enable_rag=False,
    stream=True
)

async for chunk in response_stream:
    # Send to UI, websocket, etc.
    await send_to_client(chunk)
```

### 5. Use Protocol for Type Safety

```python
from llmcore.api import LLMCoreProtocol

class MyRAGPipeline:
    def __init__(self, llm: LLMCoreProtocol):
        # Type checker ensures llm has required methods
        self.llm = llm
    
    async def query(self, question: str) -> str:
        context = self.retrieve(question)
        prompt = self.build_prompt(context, question)
        return await self.llm.chat(
            message=prompt,
            enable_rag=False
        )
```

------

## Complete Example: Semantiscan Integration

Here's how `semantiscan` integrates with LLMCore:

```python
# semantiscan/pipelines/query.py

from typing import Dict, Any, List
from llmcore.api import LLMCore, LLMCoreProtocol
from llmcore.models import ContextItem, ContextItemType

class QueryPipeline:
    """
    Semantiscan query pipeline using LLMCore as LLM backend.
    """
    
    def __init__(self, config: Dict[str, Any], llmcore_instance: LLMCoreProtocol):
        self.config = config
        self.llm = llmcore_instance
        self.retriever = self._initialize_retriever(config)
    
    async def run(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Execute RAG query using semantiscan retrieval + LLMCore generation.
        
        Args:
            query: User's question
            session_id: Optional session for conversation history
            top_k: Number of documents to retrieve
            
        Returns:
            Dict containing answer and source metadata
        """
        
        # Step 1: Retrieve relevant documents (semantiscan logic)
        retrieved_docs = await self.retriever.retrieve(
            query=query,
            top_k=top_k
        )
        
        # Step 2: Format context (semantiscan logic)
        context_str = self._format_documents(retrieved_docs)
        
        # Step 3: Build prompt (semantiscan logic)
        full_prompt = self._build_rag_prompt(
            context=context_str,
            question=query
        )
        
        # Step 4: Call LLMCore (critical integration point)
        try:
            response = await self.llm.chat(
                message=full_prompt,
                session_id=session_id,
                enable_rag=False,  # CRITICAL: semantiscan handles RAG
                save_session=bool(session_id),
                stream=False
            )
        except Exception as e:
            logger.error(f"LLMCore chat failed: {e}")
            return {
                "answer": None,
                "error": str(e),
                "sources": []
            }
        
        # Step 5: Return formatted result
        return {
            "answer": response,
            "sources": [
                {
                    "path": doc.source_path,
                    "content": doc.content,
                    "score": doc.score
                }
                for doc in retrieved_docs
            ],
            "query": query,
            "session_id": session_id
        }
    
    def _format_documents(self, docs: List[Any]) -> str:
        """Format retrieved documents into context string."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            formatted.append(
                f"Document {i} (Score: {doc.score:.3f}):\n"
                f"Source: {doc.source_path}\n"
                f"{doc.content}\n"
            )
        return "\n---\n".join(formatted)
    
    def _build_rag_prompt(self, context: str, question: str) -> str:
        """Build the final RAG prompt."""
        template = self.config.get(
            'prompt_template',
            """You are an AI assistant helping developers understand code.

Context from codebase:
{context}

Question: {question}

Provide a detailed answer based ONLY on the context above. If the answer
is not in the context, say so clearly."""
        )
        
        return template.format(context=context, question=question)


# Usage in application
async def main():
    # Initialize LLMCore
    llm = await LLMCore.create(
        config_file_path="~/.config/llmcore/config.toml"
    )
    
    # Initialize semantiscan with LLMCore
    config = load_semantiscan_config()
    pipeline = QueryPipeline(config=config, llmcore_instance=llm)
    
    # Run query
    result = await pipeline.run(
        query="How does the authentication system work?",
        session_id="user_123_session",
        top_k=5
    )
    
    print(f"Answer: {result['answer']}")
    print(f"\nSources: {len(result['sources'])} documents used")
```

------

## Troubleshooting

### Issue: Double-RAG Symptoms

**Symptoms**:

- Responses include "according to the context" twice
- Very slow response times
- Redundant or conflicting information

**Solution**:

```python
# Verify enable_rag is False
response = await llm.chat(
    message=prompt,
    enable_rag=False  # ← Check this is explicitly set
)
```

### Issue: Context Too Large

**Symptoms**:

- `ContextLengthError` exceptions
- Truncated or incomplete responses

**Solutions**:

1. **Preview first**:

```python
preview = await llm.preview_context_for_chat(...)
if preview['final_token_count'] > limit:
    # Reduce context
    pass
```

1. **Use structured items** (LLMCore handles truncation):

```python
response = await llm.chat(
    message=query,
    explicitly_staged_items=context_items,  # Auto-truncated if needed
    enable_rag=False
)
```

1. **Reduce retrieved documents**:

```python
# Retrieve fewer documents
docs = retrieve_documents(query, top_k=3)  # Instead of 10
```

### Issue: Session Not Persisting

**Symptoms**:

- Conversation history lost between queries
- Each query treated as new conversation

**Solution**:

```python
# Ensure session_id is reused and save_session=True
response = await llm.chat(
    message=prompt,
    session_id="persistent_session_id",  # ← Reuse same ID
    save_session=True,  # ← Enable saving
    enable_rag=False
)
```

### Issue: Type Errors with explicitly_staged_items

**Symptoms**:

- TypeError when passing context items
- Pydantic validation errors

**Solution**:

```python
from llmcore.models import ContextItem, ContextItemType

# Ensure items are proper types
items = [
    ContextItem(  # Use ContextItem, not dict
        id=str(i),
        type=ContextItemType.RAG_SNIPPET,  # Use enum
        content=doc.content,
        source_id=doc.path
    )
    for i, doc in enumerate(docs)
]
```

------

## Additional Resources

- **LLMCore API Documentation**: See `docs/API.md`
- **Example Implementations**: See `examples/external_rag_example.py`
- **Test Suite**: See `tests/api/test_external_rag_integration.py`
- **Semantiscan Integration**: See `docs/SEMANTISCAN_INTEGRATION.md`

------

