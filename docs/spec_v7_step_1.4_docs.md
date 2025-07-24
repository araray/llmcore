## **Rationale Block**

**Pre-state**: The v0.21.0 codebase had provider-specific logic scattered throughout, hardcoded model capabilities, and no unified interface for tool-calling. Each provider handled tools differently, and context length management relied on static fallback values.

**Limitation**: This created technical debt, made adding new providers difficult, and prevented dynamic adaptation to model capabilities. Tool-calling required provider-specific code, limiting the platform's flexibility and robustness.

**Decision Path**: Implemented the standardized interface as specified in the task dossier:
1. Added abstract `get_models_details()` method to `BaseProvider` for dynamic model discovery
2. Enhanced `chat_completion()` signatures across all providers to accept `tools` and `tool_choice` parameters
3. Updated main `LLMCore.chat()` method to expose unified tool-calling interface
4. Added `get_models_details_for_provider()` public method for external access to model introspection
5. Enhanced `MemoryManager` to use precise context lengths from dynamic model discovery

**Post-state**: The platform now provides a truly standardized interface. Developers can use `llm.chat(tools=[...])` consistently across all providers. The `MemoryManager` fetches exact context lengths dynamically, eliminating hardcoded fallbacks and preventing `ContextLengthError` exceptions.

## **Concise Diff Summary**

- **External Behavior**: Added `tools` and `tool_choice` parameters to `LLMCore.chat()` method for unified tool-calling across all providers
- **New Public Method**: `get_models_details_for_provider()` exposes dynamic model capability discovery
- **Enhanced Context Management**: `MemoryManager` now uses precise context lengths from model introspection rather than static fallbacks
- **Provider Interface**: All providers now implement `get_models_details()` and accept standardized tool parameters

## **Reversion Path**
To revert: Remove `tools`/`tool_choice` parameters from `chat()` method, remove `get_models_details()` implementations, and restore `MemoryManager` to use `get_max_context_length()` directly.

---

## **Commit Message**

```text
feat(core): implement standardized LLM interface with dynamic model discovery

* **Why** – Abstract provider-specific logic and eliminate hardcoded model capabilities; enable unified tool-calling interface across all providers
* **What** – Added get_models_details() abstract method to BaseProvider; enhanced chat_completion() signatures for unified tool support; updated LLMCore.chat() with tools/tool_choice parameters; added get_models_details_for_provider() public method; enhanced MemoryManager to use dynamic model introspection
* **Impact** – Developers can now use llm.chat(tools=[...]) consistently across providers; MemoryManager fetches precise context lengths eliminating ContextLengthError from incorrect hardcoded values; platform becomes more robust and adaptable
* **Risk** – Comprehensive test coverage added for all new methods; backward compatible (new parameters are optional); reversion path is straightforward parameter removal

Refs: spec7_step-1.4
```

## **Documentation Update**

```markdown
# Dynamic Model Discovery & Unified Tool-Calling

LLMCore v0.21.0 introduces two major standardization features that make the platform more robust and adaptable:

## Dynamic Model Introspection

The platform now dynamically discovers model capabilities at runtime:

```python
# Get detailed information about all models from a provider
models = await llm.get_models_details_for_provider("openai")
for model in models:
    print(f"Model: {model.id}")
    print(f"Context Length: {model.context_length}")
    print(f"Supports Tools: {model.supports_tools}")
    print(f"Supports Streaming: {model.supports_streaming}")
```

The `MemoryManager` automatically uses these precise context lengths, eliminating hardcoded fallbacks and preventing context length errors.

## Unified Tool-Calling

Use the same tool-calling interface across all providers:

```python
from llmcore.models import Tool

# Define a tool
weather_tool = Tool(
    name="get_weather",
    description="Get current weather for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"}
        },
        "required": ["location"]
    }
)

# Use with any provider (OpenAI, Anthropic, Gemini, Ollama)
response = await llm.chat(
    "What's the weather in Tokyo?",
    tools=[weather_tool],
    tool_choice="auto"
)
```

The platform automatically translates the standardized `Tool` objects into each provider's native format, providing a consistent developer experience regardless of the underlying model.
