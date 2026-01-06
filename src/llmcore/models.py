# src/llmcore/models.py
"""
Core data models for the LLMCore library.

This module defines the Pydantic models used to represent fundamental
data structures such as messages, roles, chat sessions, context documents,
context items, and context presets. It also includes models for the
unified tool-calling interface and dynamic provider introspection.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class Role(str, Enum):
    """
    Enumeration of possible roles in a conversation.
    These roles define the origin or type of a message.
    """
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool" # Added for tool results

    @classmethod
    def _missing_(cls, value: object): # type: ignore[misc]
        """
        Handles case-insensitive matching and common aliases for roles.
        For example, "Agent" or "AGENT" will be mapped to Role.ASSISTANT.
        """
        if isinstance(value, str):
            lower_value = value.lower()
            if lower_value == "agent":
                return cls.ASSISTANT
            for member in cls:
                if member.value == lower_value:
                    return member
        return None


class Message(BaseModel):
    """
    Represents a single message within a chat session.

    Attributes:
        id: A unique identifier for the message.
        session_id: The identifier of the chat session this message belongs to.
        role: The role of the entity that produced the message.
        content: The textual content of the message.
        timestamp: The date and time when the message was created or recorded.
        tool_call_id: For messages with role 'tool', the ID of the tool call this is a response to.
        tokens: An optional count of tokens for the message content.
        metadata: An optional dictionary for storing additional, unstructured information.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the message.")
    session_id: Optional[str] = Field(default=None, description="Identifier of the chat session this message belongs to.")
    role: Role = Field(description="The role of the message sender (system, user, assistant, or tool).")
    content: str = Field(description="The textual content of the message.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of when the message was created (UTC).")
    tool_call_id: Optional[str] = Field(default=None, description="For role 'tool', the ID of the corresponding tool call.")
    tokens: Optional[int] = Field(default=None, description="Optional token count for the message content.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional dictionary for additional message metadata.")

    class Config:
        """Pydantic model configuration."""
        use_enum_values = True
        validate_assignment = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat().replace('+00:00', 'Z')
        }

    @field_validator('timestamp', mode='before')
    @classmethod
    def ensure_utc_timestamp(cls, v: Any) -> datetime:
        """Ensure the timestamp is timezone-aware and in UTC if naive."""
        if isinstance(v, str):
            try:
                if v.endswith('Z'):
                    v_parsed = datetime.fromisoformat(v[:-1] + '+00:00')
                else:
                    v_parsed = datetime.fromisoformat(v)
            except ValueError:
                for fmt in ("%Y-%m-%d %H:%M:%S.%f%z", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
                    try:
                        v_parsed = datetime.strptime(v, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError(f"Invalid datetime format: {v}")

            if v_parsed.tzinfo is None:
                return v_parsed.replace(tzinfo=timezone.utc)
            return v_parsed.astimezone(timezone.utc)
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        if v is None:
            return datetime.now(timezone.utc)
        return v


class ContextItemType(str, Enum):
    """
    Enumeration of types for items that can be part of the LLM context pool or a saved preset.
    """
    HISTORY_MESSAGE = "history_message"
    USER_TEXT = "user_text"
    USER_FILE = "user_file"
    RAG_SNIPPET = "rag_snippet"
    PRESET_TEXT_CONTENT = "preset_text_content"
    PRESET_FILE_REFERENCE = "preset_file_reference"
    PRESET_RAG_CONTENT = "preset_rag_content"

    @classmethod
    def _missing_(cls, value: object): # type: ignore[misc]
        if isinstance(value, str):
            lower_value = value.lower()
            for member in cls:
                if member.value == lower_value:
                    return member
        return None


class ContextItem(BaseModel):
    """
    Represents an individual item that can be part of the LLM's context pool,
    typically managed within a ChatSession's `context_items` list (workspace).
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: ContextItemType
    source_id: Optional[str] = None
    content: str
    tokens: Optional[int] = None
    original_tokens: Optional[int] = None
    is_truncated: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        use_enum_values = True
        validate_assignment = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat().replace('+00:00', 'Z')
        }


class EpisodeType(str, Enum):
    """Enumeration of possible event types in an agent's episodic memory."""
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    USER_INTERACTION = "user_interaction"
    AGENT_REFLECTION = "agent_reflection"

    @classmethod
    def _missing_(cls, value: object): # type: ignore[misc]
        if isinstance(value, str):
            lower_value = value.lower()
            for member in cls:
                if member.value == lower_value:
                    return member
        return None


class Episode(BaseModel):
    """Represents a single event in an agent's experience log (Episodic Memory)."""
    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the episode.")
    session_id: str = Field(description="The session this episode belongs to.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of the event (UTC).")
    event_type: EpisodeType = Field(description="The type of event that occurred.")
    data: Dict[str, Any] = Field(description="A JSON blob containing the structured data of the event.")

    class Config:
        use_enum_values = True
        validate_assignment = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat().replace('+00:00', 'Z')
        }

    @field_validator('timestamp', mode='before')
    @classmethod
    def ensure_utc_timestamp(cls, v: Any) -> datetime:
        """Ensure the timestamp is timezone-aware and in UTC if naive."""
        if isinstance(v, str):
            try:
                if v.endswith('Z'):
                    v_parsed = datetime.fromisoformat(v[:-1] + '+00:00')
                else:
                    v_parsed = datetime.fromisoformat(v)
            except ValueError:
                for fmt in ("%Y-%m-%d %H:%M:%S.%f%z", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
                    try:
                        v_parsed = datetime.strptime(v, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError(f"Invalid datetime format: {v}")

            if v_parsed.tzinfo is None:
                return v_parsed.replace(tzinfo=timezone.utc)
            return v_parsed.astimezone(timezone.utc)
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        if v is None:
            return datetime.now(timezone.utc)
        return v


class ChatSession(BaseModel):
    """
    Represents a single conversation or chat session.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    messages: List[Message] = Field(default_factory=list)
    context_items: List[ContextItem] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        validate_assignment = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat().replace('+00:00', 'Z')
        }

    def add_message(self, message_content: str, role: Role, session_id_override: Optional[str] = None) -> Message:
        new_message = Message(content=message_content, role=role, session_id=session_id_override or self.id)
        self.messages.append(new_message)
        self.updated_at = datetime.now(timezone.utc)
        return new_message

    def add_context_item(self, item: ContextItem) -> None:
        self.context_items = [ci for ci in self.context_items if ci.id != item.id]
        self.context_items.append(item)
        self.context_items.sort(key=lambda x: x.timestamp)
        self.updated_at = datetime.now(timezone.utc)

    def remove_context_item(self, item_id: str) -> bool:
        initial_len = len(self.context_items)
        self.context_items = [ci for ci in self.context_items if ci.id != item_id]
        if len(self.context_items) < initial_len:
            self.updated_at = datetime.now(timezone.utc)
            return True
        return False

    def get_context_item(self, item_id: str) -> Optional[ContextItem]:
        for item in self.context_items:
            if item.id == item_id:
                return item
        return None


class ContextDocument(BaseModel):
    """
    Represents a document used for context, typically in Retrieval Augmented Generation (RAG).
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None

    class Config:
        validate_assignment = True


class ContextPreparationDetails(BaseModel):
    """
    Structured output from ContextManager.prepare_context.
    """
    prepared_messages: List[Message]
    final_token_count: int
    max_tokens_for_model: int
    rag_documents_used: Optional[List[ContextDocument]] = None
    rendered_rag_template_content: Optional[str] = None
    truncation_actions_taken: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        validate_assignment = True


class ContextPresetItem(BaseModel):
    """
    Represents an item within a saved ContextPreset.
    """
    item_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: ContextItemType
    content: Optional[str] = None
    source_identifier: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True
        validate_assignment = True


class ContextPreset(BaseModel):
    """
    Represents a named, saved collection of context items (a "Context Preset").
    """
    name: str
    description: Optional[str] = None
    items: List[ContextPresetItem] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        validate_assignment = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat().replace('+00:00', 'Z')
        }

    @model_validator(mode='before')
    @classmethod
    def ensure_name_is_valid_identifier(cls, data: Any) -> Any:
        if isinstance(data, dict):
            name = data.get('name')
            if name and not isinstance(name, str):
                raise ValueError("Preset name must be a string.")
            if name and (not name.strip() or any(c in name for c in ['/', '\\', ':', '*', '?', '"', '<', '>', '|'])):
                raise ValueError(f"Preset name '{name}' contains invalid characters or is empty.")
        return data

# --- Models for spec5.md ---

class ModelDetails(BaseModel):
    """
    Represents detailed information about a specific LLM model, discovered dynamically.

    Attributes:
        id: The unique identifier for the model (e.g., "gpt-4o").
        context_length: The maximum context window size in tokens.
        supports_streaming: Flag indicating if the model supports streaming responses.
        supports_tools: Flag indicating if the model supports tool/function calling.
        provider_name: The name of the provider this model belongs to.
        metadata: A dictionary for any other provider-specific metadata.
    """
    id: str = Field(description="The unique identifier for the model.")
    context_length: int = Field(description="The maximum context window size in tokens.")
    supports_streaming: bool = Field(default=True, description="Indicates if the model supports streaming responses.")
    supports_tools: bool = Field(default=False, description="Indicates if the model supports tool/function calling.")
    provider_name: str = Field(description="The name of the provider this model belongs to.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Provider-specific metadata.")


class Tool(BaseModel):
    """
    Represents a function that can be called by an LLM, defined in a provider-agnostic way.

    Attributes:
        name: The name of the tool/function.
        description: A description of what the tool does, used by the LLM to decide when to call it.
        parameters: A dictionary representing the JSON Schema for the tool's input parameters.
    """
    name: str = Field(description="The name of the tool/function.")
    description: str = Field(description="A description of what the tool does.")
    parameters: Dict[str, Any] = Field(description="A JSON Schema object defining the tool's input parameters.")


class ToolCall(BaseModel):
    """
    Represents a request from the LLM to execute a specific tool with provided arguments.

    Attributes:
        id: A unique identifier for this specific tool call, used to match it with a ToolResult.
        name: The name of the tool to be executed.
        arguments: A dictionary of arguments for the tool, as generated by the LLM.
    """
    id: str = Field(description="Unique identifier for this specific tool call.")
    name: str = Field(description="The name of the tool to be executed.")
    arguments: Dict[str, Any] = Field(description="A dictionary of arguments for the tool, generated by the LLM.")


class ToolResult(BaseModel):
    """
    Represents the output from the execution of a tool, to be sent back to the LLM.

    Attributes:
        tool_call_id: The ID of the ToolCall this result corresponds to.
        content: The string representation of the tool's output.
        is_error: Whether the tool execution resulted in an error.
    """
    tool_call_id: str = Field(description="The ID of the ToolCall this result corresponds to.")
    content: str = Field(description="The string representation of the tool's output.")
    is_error: bool = Field(default=False, description="Whether the tool execution resulted in an error.")


# --- Models for Phase 4: Agentic Loop ---

class AgentState(BaseModel):
    """
    Represents the agent's "Working Memory" or "scratchpad" for a specific task.
    This model holds the transient, short-term cognitive state required for the
    agent's reasoning loop.

    Attributes:
        goal: The high-level objective the agent is trying to achieve.
        plan: A list of strings representing the decomposed steps the agent intends to take.
        current_plan_step_index: Index of the current step being executed in the plan.
        plan_steps_status: List tracking the status of each plan step ('pending', 'completed', 'failed').
        history_of_thoughts: A log of the agent's internal reasoning steps ("Thoughts").
        observations: A dictionary mapping tool calls or actions to their observed results.
        scratchpad: A free-form text field for intermediate reasoning or notes.
    """
    goal: str = Field(description="The high-level objective for the agent.")
    plan: List[str] = Field(default_factory=list, description="The decomposed plan of sub-tasks.")
    current_plan_step_index: int = Field(default=0, description="Index of the current plan step being executed.")
    plan_steps_status: List[str] = Field(default_factory=list, description="Status of each plan step ('pending', 'completed', 'failed').")
    history_of_thoughts: List[str] = Field(default_factory=list, description="A chronological log of the agent's internal 'Thoughts'.")
    observations: Dict[str, Any] = Field(default_factory=dict, description="A mapping of actions to their observed results.")
    scratchpad: str = Field(default="", description="A transient workspace for intermediate reasoning.")

    class Config:
        validate_assignment = True


class AgentTask(BaseModel):
    """
    Represents an asynchronous agentic task being managed by the TaskMaster service.
    This model tracks the lifecycle and state of a long-running agent operation.

    UPDATED: Added HITL workflow support with pending_action_data and approval_prompt fields.

    Attributes:
        task_id: A unique identifier for the agent task.
        status: The current status of the task (e.g., PENDING, RUNNING, SUCCESS, FAILURE, PENDING_APPROVAL).
        goal: The original goal provided by the user.
        agent_state: The current working memory (AgentState) of the agent performing the task.
        pending_action_data: JSON representation of the ToolCall awaiting human approval (HITL workflow).
        approval_prompt: The question/prompt for the human operator (HITL workflow).
        created_at: The timestamp when the task was created.
        updated_at: The timestamp when the task was last updated.
    """
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the agent task.")
    status: str = Field(default="PENDING", description="The current status of the task.")
    goal: str = Field(description="The original high-level goal for the task.")
    agent_state: AgentState = Field(description="The agent's current working memory state.")
    pending_action_data: Optional[Dict[str, Any]] = Field(default=None, description="JSON representation of the ToolCall awaiting approval (HITL).")
    approval_prompt: Optional[str] = Field(default=None, description="The question for the human operator (HITL workflow).")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of task creation (UTC).")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of last task update (UTC).")

    class Config:
        validate_assignment = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat().replace('+00:00', 'Z')
        }
