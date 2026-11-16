# src/llmcore/providers/openai_responses.py
"""OpenAI Responses API wire format, as pure functions.

``POST /responses`` is OpenAI's newer API and the native one for its reasoning
models. llmcore's providers parse the Chat Completions shape everywhere: every
``extract_*`` method, the session layer and the streaming wrappers. So this
module converts in both directions, and nothing downstream has to learn a
second shape:

- :func:`build_responses_request` turns llmcore messages, tools and parameters
  into a Responses request body.
- :func:`normalize_response` turns a Responses response into a Chat
  Completions dict.
- :class:`ResponsesStreamNormalizer` turns Responses stream events into Chat
  Completions chunks.

The module holds no client and no transport, so any provider that speaks
Responses can use it. Command Code's gateway is the first.

Conventions it adds on top of the Chat Completions shape:

- ``message["reasoning"]``: the reasoning summary text, when the model gave one.
- ``message["reasoning_items"]``: the raw ``reasoning`` output items, including
  ``encrypted_content``. Put them in the next assistant message's
  ``metadata["reasoning_items"]`` and they are sent back ahead of that turn,
  which keeps a reasoning model's chain intact across a tool loop without
  server-side state (``store: false``).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from ..models import Message, Tool

#: Chat-style keyword -> Responses field, for parameters that only differ by name.
_RENAMED: dict[str, str] = {
    "max_tokens": "max_output_tokens",
    "max_completion_tokens": "max_output_tokens",
}

#: Parameters passed through unchanged.
PASSTHROUGH_PARAMS: tuple[str, ...] = (
    "max_output_tokens",
    "temperature",
    "top_p",
    "parallel_tool_calls",
    "store",
    "include",
    "metadata",
    "user",
    "service_tier",
    "truncation",
    "prompt_cache_key",
    "prompt_cache_retention",
    "previous_response_id",
    "max_tool_calls",
    "top_logprobs",
    "safety_identifier",
)


class ResponsesFailed(Exception):
    """A response came back with ``status: "failed"`` or an ``error`` event.

    Carries the payload's ``error`` object, so the provider can map it onto its
    own exception type.
    """

    def __init__(self, error: Mapping[str, Any] | None, partial: str | None = None) -> None:
        self.error = dict(error or {})
        self.partial = partial
        super().__init__(self.error.get("message") or "response failed")


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


def _image_part(img: Any) -> dict[str, Any] | None:
    """One ``inline_images`` entry (URL string, or dict) as an ``input_image`` part."""
    if isinstance(img, str):
        return {"type": "input_image", "image_url": img}
    if not isinstance(img, Mapping):
        return None
    if img.get("url"):
        part: dict[str, Any] = {"type": "input_image", "image_url": str(img["url"])}
    elif img.get("data"):
        media = img.get("media_type") or img.get("mime_type") or "image/png"
        part = {"type": "input_image", "image_url": f"data:{media};base64,{img['data']}"}
    elif img.get("file_id"):
        part = {"type": "input_image", "file_id": str(img["file_id"])}
    else:
        return None
    if img.get("detail"):
        part["detail"] = img["detail"]
    return part


def _user_content(msg: Message) -> Any:
    metadata = msg.metadata or {}
    if "content_parts" in metadata:
        return list(metadata["content_parts"])
    images = [p for p in (_image_part(i) for i in metadata.get("inline_images", [])) if p]
    if not images:
        return msg.content
    parts: list[dict[str, Any]] = []
    if msg.content:
        parts.append({"type": "input_text", "text": msg.content})
    return parts + images


def _call_arguments(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    return json.dumps(raw if raw is not None else {})


def _assistant_items(msg: Message) -> list[dict[str, Any]]:
    """An assistant turn as Responses input items: reasoning, text, calls."""
    metadata = msg.metadata or {}
    items: list[dict[str, Any]] = [
        dict(item) for item in metadata.get("reasoning_items") or [] if isinstance(item, Mapping)
    ]
    if msg.content:
        items.append({"role": "assistant", "content": msg.content})
    for call in getattr(msg, "tool_calls", None) or metadata.get("tool_calls") or []:
        if not isinstance(call, Mapping):
            continue
        func = call.get("function") if isinstance(call.get("function"), Mapping) else call
        items.append(
            {
                "type": "function_call",
                "call_id": str(call.get("id") or call.get("call_id") or ""),
                "name": str(func.get("name") or ""),
                "arguments": _call_arguments(func.get("arguments")),
            }
        )
    return items


def messages_to_input(messages: Iterable[Message]) -> tuple[str | None, list[dict[str, Any]]]:
    """Split llmcore messages into Responses ``instructions`` and ``input`` items.

    System messages become ``instructions`` (joined in order). Tool results
    become ``function_call_output`` items keyed by their ``tool_call_id``.
    """
    instructions: list[str] = []
    items: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        if role == "system":
            if msg.content:
                instructions.append(msg.content)
        elif role == "user":
            items.append({"role": "user", "content": _user_content(msg)})
        elif role == "assistant":
            items.extend(_assistant_items(msg))
        elif role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": str(msg.tool_call_id or ""),
                    "output": msg.content or "",
                }
            )
    return ("\n\n".join(instructions) or None), items


def tools_to_responses(tools: Iterable[Tool]) -> list[dict[str, Any]]:
    """llmcore tools as Responses function tools (flat, not nested in ``function``)."""
    out: list[dict[str, Any]] = []
    for tool in tools:
        spec = tool.model_dump() if hasattr(tool, "model_dump") else dict(tool)
        out.append(
            {
                "type": "function",
                "name": spec.get("name"),
                "description": spec.get("description") or "",
                "parameters": spec.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return out


def tool_choice_to_responses(choice: Any) -> Any:
    """Chat-style ``tool_choice`` (string, tool name, or dict) for Responses."""
    if choice is None:
        return None
    if isinstance(choice, Mapping):
        func = choice.get("function")
        if isinstance(func, Mapping) and func.get("name"):
            return {"type": "function", "name": func["name"]}
        return dict(choice)
    value = str(choice)
    if value in ("auto", "none", "required"):
        return value
    if value == "any":
        return "required"
    return {"type": "function", "name": value}


def response_format_to_text(fmt: Mapping[str, Any]) -> dict[str, Any] | None:
    """Chat ``response_format`` as the Responses ``text.format`` object."""
    kind = fmt.get("type")
    if kind == "json_schema":
        schema = fmt.get("json_schema") or {}
        out: dict[str, Any] = {
            "type": "json_schema",
            "name": schema.get("name") or "response",
            "schema": schema.get("schema") or {},
        }
        if "strict" in schema:
            out["strict"] = schema["strict"]
        if schema.get("description"):
            out["description"] = schema["description"]
        return {"format": out}
    if kind in ("json_object", "text"):
        return {"format": {"type": kind}}
    return None


def build_responses_request(
    model: str,
    messages: Iterable[Message],
    *,
    stream: bool = False,
    tools: Iterable[Tool] | None = None,
    tool_choice: Any = None,
    extra_tools: Iterable[Mapping[str, Any]] | None = None,
    **params: Any,
) -> dict[str, Any]:
    """A complete ``POST /responses`` body.

    ``params`` accepts Chat Completions spellings where they differ
    (``max_tokens``, ``response_format``, ``reasoning_effort``), llmcore's
    canonical ``effort``, and the Responses fields themselves (``reasoning``,
    ``text``, ``include``, …), which win when both are given.

    Defaults: ``store`` is false, so nothing is kept server-side, and encrypted
    reasoning is requested whenever reasoning is in play, so the chain can be
    sent back on the next turn.
    """
    instructions, items = messages_to_input(messages)
    body: dict[str, Any] = {"model": model, "input": items, "stream": stream}
    if instructions:
        body["instructions"] = params.pop("instructions", None) or instructions
    elif params.get("instructions"):
        body["instructions"] = params.pop("instructions")

    all_tools = tools_to_responses(tools or []) + [dict(t) for t in extra_tools or []]
    if all_tools:
        body["tools"] = all_tools
    choice = tool_choice_to_responses(tool_choice)
    if choice is not None:
        body["tool_choice"] = choice

    for old, new in _RENAMED.items():
        if old in params:
            value = params.pop(old)
            body.setdefault(new, value)

    reasoning = dict(params.pop("reasoning", None) or {})
    effort = params.pop("effort", None) or params.pop("reasoning_effort", None)
    if effort and "effort" not in reasoning:
        reasoning["effort"] = str(effort).strip().lower()
    summary = params.pop("reasoning_summary", None)
    if summary and "summary" not in reasoning:
        reasoning["summary"] = summary
    if reasoning:
        body["reasoning"] = reasoning

    text = dict(params.pop("text", None) or {})
    fmt = params.pop("response_format", None)
    if isinstance(fmt, Mapping) and "format" not in text:
        converted = response_format_to_text(fmt)
        if converted:
            text.update(converted)
    verbosity = params.pop("verbosity", None)
    if verbosity and "verbosity" not in text:
        text["verbosity"] = verbosity
    if text:
        body["text"] = text

    for key in PASSTHROUGH_PARAMS:
        if key in params:
            body[key] = params.pop(key)
    body.update(params)  # anything else is the caller's business

    body.setdefault("store", False)
    if body.get("store") is False and "reasoning" in body:
        include = list(body.get("include") or [])
        if "reasoning.encrypted_content" not in include:
            include.append("reasoning.encrypted_content")
        body["include"] = include
    return body


# ---------------------------------------------------------------------------
# Responses -> Chat Completions shape
# ---------------------------------------------------------------------------


def normalize_usage(usage: Mapping[str, Any] | None) -> dict[str, Any]:
    """Responses usage, with the Chat Completions keys llmcore reads added."""
    if not usage:
        return {}
    out = dict(usage)
    inp = int(usage.get("input_tokens") or 0)
    outp = int(usage.get("output_tokens") or 0)
    in_details = usage.get("input_tokens_details") or {}
    out_details = usage.get("output_tokens_details") or {}
    cache_write = in_details.get("cache_write_tokens")
    if cache_write is None and isinstance(in_details.get("cache_write_tokens_details"), Mapping):
        cache_write = sum(
            int(v or 0) for v in in_details["cache_write_tokens_details"].values()
        )
    out["prompt_tokens"] = inp
    out["completion_tokens"] = outp
    out["total_tokens"] = int(usage.get("total_tokens") or inp + outp)
    prompt_details: dict[str, Any] = {"cached_tokens": int(in_details.get("cached_tokens") or 0)}
    if cache_write is not None:
        prompt_details["cache_write_tokens"] = int(cache_write)
    out["prompt_tokens_details"] = prompt_details
    out["completion_tokens_details"] = {
        "reasoning_tokens": int(out_details.get("reasoning_tokens") or 0)
    }
    return out


def _reasoning_text(item: Mapping[str, Any]) -> str:
    parts = [p.get("text", "") for p in item.get("summary") or [] if isinstance(p, Mapping)]
    if not any(parts):
        # Open models on /responses return raw reasoning as `content`.
        parts = [p.get("text", "") for p in item.get("content") or [] if isinstance(p, Mapping)]
    return "".join(parts)


def _finish_reason(response: Mapping[str, Any], has_calls: bool) -> str:
    status = response.get("status")
    if status == "incomplete":
        reason = (response.get("incomplete_details") or {}).get("reason")
        return "content_filter" if reason == "content_filter" else "length"
    return "tool_calls" if has_calls else "stop"


def normalize_response(response: Mapping[str, Any]) -> dict[str, Any]:
    """A Responses response as a Chat Completions dict.

    Raises:
        ResponsesFailed: If the response's ``status`` is ``failed``.
    """
    texts: list[str] = []
    refusals: list[str] = []
    annotations: list[Any] = []
    reasoning_texts: list[str] = []
    reasoning_items: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    other_items: list[dict[str, Any]] = []

    for item in response.get("output") or []:
        if not isinstance(item, Mapping):
            continue
        kind = item.get("type")
        if kind == "message":
            for part in item.get("content") or []:
                if part.get("type") == "output_text":
                    texts.append(part.get("text") or "")
                    annotations.extend(part.get("annotations") or [])
                elif part.get("type") == "refusal":
                    refusals.append(part.get("refusal") or "")
        elif kind == "reasoning":
            reasoning_items.append(dict(item))
            text = _reasoning_text(item)
            if text:
                reasoning_texts.append(text)
        elif kind == "function_call":
            tool_calls.append(
                {
                    "id": item.get("call_id") or item.get("id"),
                    "type": "function",
                    "function": {"name": item.get("name"), "arguments": item.get("arguments") or "{}"},
                }
            )
        else:
            other_items.append(dict(item))

    if response.get("status") == "failed":
        raise ResponsesFailed(response.get("error"), partial="".join(texts) or None)

    message: dict[str, Any] = {"role": "assistant", "content": "".join(texts) or None}
    if not tool_calls and message["content"] is None:
        message["content"] = ""
    if tool_calls:
        message["tool_calls"] = tool_calls
    if refusals:
        message["refusal"] = "".join(refusals)
    if reasoning_texts:
        message["reasoning"] = "\n\n".join(reasoning_texts)
    if reasoning_items:
        message["reasoning_items"] = reasoning_items
    if annotations:
        message["annotations"] = annotations
    if other_items:
        message["output_items"] = other_items

    return {
        "id": response.get("id"),
        "object": "chat.completion",
        "created": response.get("created_at"),
        "model": response.get("model"),
        "choices": [
            {"index": 0, "message": message, "finish_reason": _finish_reason(response, bool(tool_calls))}
        ],
        "usage": normalize_usage(response.get("usage")),
        "response_id": response.get("id"),
        "status": response.get("status"),
    }


class ResponsesStreamNormalizer:
    """Turns Responses stream events into Chat Completions chunks.

    Feed each decoded event to :meth:`feed`. It returns zero or more chunks.
    Text and reasoning arrive as ``delta.content`` / ``delta.reasoning``. Function
    calls arrive as ``delta.tool_calls`` with a stable ``index`` per call. The
    final chunk carries ``finish_reason``, ``usage`` and, under
    ``delta.reasoning_items``, the reasoning items to send back next turn.

    Raises:
        ResponsesFailed: On ``response.failed`` or an ``error`` event.
    """

    def __init__(self) -> None:
        self.response_id: str | None = None
        self.model: str | None = None
        self._tool_index: dict[int, int] = {}
        self._text: list[str] = []

    def _chunk(self, delta: dict[str, Any], finish: str | None = None, **extra: Any) -> dict[str, Any]:
        chunk: dict[str, Any] = {
            "id": self.response_id,
            "object": "chat.completion.chunk",
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        chunk.update(extra)
        return chunk

    def feed(self, event: Mapping[str, Any]) -> list[dict[str, Any]]:
        kind = event.get("type")
        if kind in ("response.created", "response.in_progress"):
            resp = event.get("response") or {}
            self.response_id = resp.get("id") or self.response_id
            self.model = resp.get("model") or self.model
            return []
        if kind == "response.output_text.delta":
            text = event.get("delta") or ""
            self._text.append(text)
            return [self._chunk({"role": "assistant", "content": text})] if text else []
        if kind in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            text = event.get("delta") or ""
            return [self._chunk({"role": "assistant", "reasoning": text})] if text else []
        if kind == "response.refusal.delta":
            return [self._chunk({"role": "assistant", "refusal": event.get("delta") or ""})]
        if kind == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") != "function_call":
                return []
            index = len(self._tool_index)
            self._tool_index[int(event.get("output_index", index))] = index
            return [
                self._chunk(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": index,
                                "id": item.get("call_id") or item.get("id"),
                                "type": "function",
                                "function": {"name": item.get("name"), "arguments": ""},
                            }
                        ],
                    }
                )
            ]
        if kind == "response.function_call_arguments.delta":
            index = self._tool_index.get(int(event.get("output_index", -1)))
            if index is None:
                return []
            return [
                self._chunk(
                    {
                        "role": "assistant",
                        "tool_calls": [{"index": index, "function": {"arguments": event.get("delta") or ""}}],
                    }
                )
            ]
        if kind in ("response.completed", "response.incomplete"):
            resp = event.get("response") or {}
            final = normalize_response(resp)
            message = final["choices"][0]["message"]
            delta: dict[str, Any] = {"role": "assistant", "content": ""}
            if message.get("reasoning_items"):
                delta["reasoning_items"] = message["reasoning_items"]
            return [
                self._chunk(
                    delta,
                    final["choices"][0]["finish_reason"],
                    usage=final["usage"],
                    response_id=resp.get("id"),
                )
            ]
        if kind == "response.failed":
            resp = event.get("response") or {}
            raise ResponsesFailed(resp.get("error"), partial="".join(self._text) or None)
        if kind == "error":
            raise ResponsesFailed(
                {"message": event.get("message"), "code": event.get("code")},
                partial="".join(self._text) or None,
            )
        return []
