import json
from typing import Any, Dict, List, Optional, Union, Literal, Tuple

import openai
from functools import lru_cache
from ..domain.models import (
    Message,
    SystemContent,
    Tool,
    ToolChoice,
    ContentBlockText,
    ContentBlockImage,
    ContentBlockToolUse,
    ContentBlockToolResult,
    ContentBlock,
    MessagesResponse,
    Usage,
)
from ..config import SUPPORT_DEVELOPER_MESSAGE_MODELS, MessageRoles, Settings
from ..logging import warning, error, LogRecord, LogEvent


StopReasonType = Optional[
    Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", "error"]
]


@lru_cache(maxsize=512)
def _serialize_tool_result_content_for_openai_cached(key: Tuple[str, str]) -> str:
    """Cached helper function to serialize Anthropic tool result content into OpenAI format.

    Takes a tuple key containing serialized JSON content and returns a string suitable
    for OpenAI's tool response format. Handles both text and structured content.

    Args:
        key: Tuple containing (content_json, version_identifier)

    Returns:
        String representation of the tool result content
    """
    content_json, _ = key
    items = json.loads(content_json)
    parts = []
    for item in items:
        if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
            parts.append(str(item["text"]))
        else:
            parts.append(json.dumps(item))
    return "\n".join(parts)


def _serialize_tool_result_content_for_openai(
    anthropic_tool_result_content: object,
    request_id: Optional[str],
    log_context: Dict,
) -> str:
    """
    Serializes Anthropic tool result content (which can be complex) into a single string
    as expected by OpenAI for the 'content' field of a 'tool' role message.
    """
    if isinstance(anthropic_tool_result_content, str):
        return anthropic_tool_result_content

    if isinstance(anthropic_tool_result_content, list):
        try:
            key = (
                json.dumps(
                    anthropic_tool_result_content, sort_keys=True, separators=(",", ":")
                ),
                "1",
            )
            result_str = _serialize_tool_result_content_for_openai_cached(key)
        except TypeError:
            processed_parts = []
            for item in anthropic_tool_result_content:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "text"
                    and "text" in item
                ):
                    processed_parts.append(str(item["text"]))
                else:
                    try:
                        processed_parts.append(json.dumps(item))
                    except TypeError:
                        processed_parts.append(
                            f"<unserializable_item type='{type(item).__name__}'>"
                        )
            result_str = "\n".join(processed_parts)
        return result_str

    # At this point, content should be either str or list per schema; any other type indicates malformed input.
    actual_type = type(anthropic_tool_result_content).__name__
    error(
        LogRecord(
            event=LogEvent.TOOL_RESULT_SERIALIZATION_FAILURE.value,
            message="Unsupported tool result content type; expected str or list.",
            request_id=request_id,
            data={**log_context, "actual_type": actual_type},
        )
    )
    raise TypeError(f"Unsupported tool result content type: {actual_type}")


def convert_anthropic_to_openai_messages(
    anthropic_messages: List[Message],
    anthropic_system: Optional[Union[str, List[SystemContent]]] = None,
    request_id: Optional[str] = None,
    target_model: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> List[Dict[str, Any]]:
    """Convert Anthropic messages and optional system prompt into the
    OpenAI Chat Completions message format.

    Parameters
    ----------
    anthropic_messages
        Sequence of Message objects provided by the Anthropic client.
    anthropic_system
        Optional system prompt content (text or blocks).
    request_id
        Correlator for structured logging.
    target_model
        Resolved OpenAI model name; affects role mapping.
    """
    openai_messages: List[Dict[str, Any]] = []

    system_text_content = ""
    if isinstance(anthropic_system, str):
        system_text_content = anthropic_system
    elif isinstance(anthropic_system, list):
        system_texts = [
            block.text
            for block in anthropic_system
            if isinstance(block, SystemContent) and block.type == "text"
        ]
        if len(system_texts) < len(anthropic_system):
            warning(
                LogRecord(
                    event=LogEvent.SYSTEM_PROMPT_ADJUSTED.value,
                    message="Non-text content blocks in Anthropic system prompt were ignored.",
                    request_id=request_id,
                )
            )
        system_text_content = "\n".join(system_texts)

    # Handle system content and UTF-8 enforcement together to avoid duplicate developer messages
    supports_developer_role = (
        target_model and target_model in SUPPORT_DEVELOPER_MESSAGE_MODELS
    )

    if system_text_content and supports_developer_role:
        # Combine system content with UTF-8 enforcement for models that support developer role
        utf8_enforcement_message = "\n\n" + settings.UTF8_ENFORCEMENT_MESSAGE
        combined_content = system_text_content + utf8_enforcement_message
        openai_messages.append(
            {"role": MessageRoles.Developer.value, "content": combined_content}
        )
    elif system_text_content:
        # Use system role for models that don't support developer role
        openai_messages.append(
            {"role": MessageRoles.System.value, "content": system_text_content}
        )
    elif supports_developer_role:
        # Only UTF-8 enforcement message needed
        openai_messages.append(
            {"role": MessageRoles.Developer.value, "content": settings.UTF8_ENFORCEMENT_MESSAGE}
        )

    for i, msg in enumerate(anthropic_messages):
        role = msg.role
        content = msg.content

        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            openai_parts_for_user_message = []
            assistant_tool_calls = []
            text_content_for_assistant = []

            if not content and role == "user":
                openai_messages.append({"role": "user", "content": ""})
                continue
            if not content and role == "assistant":
                openai_messages.append({"role": "assistant", "content": ""})
                continue

            for block_idx, block in enumerate(content):
                block_log_ctx = {
                    "anthropic_message_index": i,
                    "block_index": block_idx,
                    "block_type": block.type,
                }

                if isinstance(block, ContentBlockText):
                    if role == "user":
                        openai_parts_for_user_message.append(
                            {"type": "text", "text": block.text}
                        )
                    elif role == "assistant":
                        text_content_for_assistant.append(block.text)

                elif isinstance(block, ContentBlockImage) and role == "user":
                    if block.source.type == "base64":
                        openai_parts_for_user_message.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{block.source.media_type};base64,{block.source.data}"
                                },
                            }
                        )
                    else:
                        warning(
                            LogRecord(
                                event=LogEvent.IMAGE_FORMAT_UNSUPPORTED.value,
                                message=f"Image block with source type '{block.source.type}' (expected 'base64') ignored in user message {i}.",
                                request_id=request_id,
                                data=block_log_ctx,
                            )
                        )

                elif isinstance(block, ContentBlockToolUse) and role == "assistant":
                    try:
                        args_str = json.dumps(block.input)
                    except Exception as e:
                        error(
                            LogRecord(
                                event=LogEvent.TOOL_INPUT_SERIALIZATION_FAILURE.value,
                                message=f"Failed to serialize tool input for tool '{block.name}'. Using empty JSON.",
                                request_id=request_id,
                                data={
                                    **block_log_ctx,
                                    "tool_id": block.id,
                                    "tool_name": block.name,
                                },
                            ),
                            exc=e,
                        )
                        args_str = "{}"

                    assistant_tool_calls.append(
                        {
                            "id": block.id,
                            "type": "function",
                            "function": {"name": block.name, "arguments": args_str},
                        }
                    )

                elif isinstance(block, ContentBlockToolResult) and role == "user":
                    serialized_content = _serialize_tool_result_content_for_openai(
                        block.content, request_id, block_log_ctx
                    )
                    openai_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.tool_use_id,
                            "content": serialized_content,
                        }
                    )

            if role == "user" and openai_parts_for_user_message:
                is_multimodal = any(
                    part["type"] == "image_url"
                    for part in openai_parts_for_user_message
                )
                if is_multimodal or len(openai_parts_for_user_message) > 1:
                    openai_messages.append(
                        {"role": "user", "content": openai_parts_for_user_message}
                    )
                elif (
                    len(openai_parts_for_user_message) == 1
                    and openai_parts_for_user_message[0]["type"] == "text"
                ):
                    openai_messages.append(
                        {
                            "role": "user",
                            "content": openai_parts_for_user_message[0]["text"],
                        }
                    )
                elif not openai_parts_for_user_message:
                    openai_messages.append({"role": "user", "content": ""})

            if role == "assistant":
                assistant_text = "\n".join(filter(None, text_content_for_assistant))
                if assistant_text:
                    openai_messages.append(
                        {"role": "assistant", "content": assistant_text}
                    )

                if assistant_tool_calls:
                    if (
                        openai_messages
                        and openai_messages[-1]["role"] == "assistant"
                        and openai_messages[-1].get("content")
                    ):
                        openai_messages.append(
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": assistant_tool_calls,
                            }
                        )

                    elif (
                        openai_messages
                        and openai_messages[-1]["role"] == "assistant"
                        and not openai_messages[-1].get("tool_calls")
                    ):
                        openai_messages[-1]["tool_calls"] = assistant_tool_calls
                        openai_messages[-1]["content"] = None
                    else:
                        openai_messages.append(
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": assistant_tool_calls,
                            }
                        )

    final_openai_messages = []
    for msg_dict in openai_messages:
        if (
            msg_dict.get("role") == "assistant"
            and msg_dict.get("tool_calls")
            and msg_dict.get("content") is not None
        ):
            warning(
                LogRecord(
                    event=LogEvent.MESSAGE_FORMAT_NORMALIZED.value,
                    message="Corrected assistant message with tool_calls to have content: None.",
                    request_id=request_id,
                    data={"original_content": msg_dict["content"]},
                )
            )
            msg_dict["content"] = None
        final_openai_messages.append(msg_dict)

    return final_openai_messages


@lru_cache(maxsize=256)
def _tools_cache(key: Tuple[Tuple[str, str, str], ...]) -> List[Dict[str, Any]]:
    """Cached helper function to convert Anthropic tools to OpenAI format.

    Takes a tuple of tool information and returns a list of OpenAI tool definitions.
    This is used to avoid repeated serialization of the same tool definitions.

    Args:
        key: Tuple of (name, description, schema_json) tuples for each tool

    Returns:
        List of OpenAI tool definitions
    """
    tools = []
    for name, desc, schema_json in key:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": json.loads(schema_json),
                },
            }
        )
    return tools


def convert_anthropic_tools_to_openai(
    anthropic_tools: Optional[List[Tool]],
    settings: Optional[object] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Convert a list of Anthropic Tool objects into the JSON schema
    expected by OpenAI as the ``tools`` parameter.

    Each Anthropic tool is mapped to an OpenAI function tool definition.
    """
    if settings is not None and not getattr(settings, "cache_converters_enabled", True):
        # Bypass cache when disabled
        if not anthropic_tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.input_schema,
                },
            }
            for t in anthropic_tools
        ]
    if not anthropic_tools:
        return None
    try:
        key = tuple(
            (
                t.name,
                t.description or "",
                json.dumps(t.input_schema, sort_keys=True, separators=(",", ":")),
            )
            for t in anthropic_tools
        )
        return _tools_cache(key)
    except TypeError:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.input_schema,
                },
            }
            for t in anthropic_tools
        ]


@lru_cache(maxsize=256)
def _tool_choice_cache(
    key: Tuple[str, Optional[str]],
) -> Optional[Union[str, Dict[str, Any]]]:
    """Cached helper function to convert Anthropic tool choice to OpenAI format.

    Takes a tuple of tool choice information and returns the equivalent OpenAI format.
    Handles 'auto', 'any', and specific tool choices.

    Args:
        key: Tuple containing (tool_choice_type, tool_name)

    Returns:
        OpenAI equivalent of the tool choice, either as a string or dict
    """
    typ, name = key
    if typ == "auto":
        return "auto"
    if typ == "any":
        return "auto"
    if typ == "tool" and name:
        return {"type": "function", "function": {"name": name}}
    return "auto"


def convert_anthropic_tool_choice_to_openai(
    anthropic_choice: Optional[ToolChoice],
    request_id: Optional[str] = None,
    settings: Optional[object] = None,
) -> Optional[Union[str, Dict[str, Any]]]:
    """Translate an Anthropic ``tool_choice`` object into the value
    accepted by OpenAI Chat Completions.

    Returns either ``"auto"`` or a ``{"type": "function", "function": {…}}``
    mapping, falling back to ``"auto"`` for unsupported cases.
    """
    if settings is not None and not getattr(settings, "cache_converters_enabled", True):
        # Compute without cache
        if not anthropic_choice:
            return None
        if anthropic_choice.type == "auto":
            return "auto"
        if anthropic_choice.type == "any":
            warning(
                LogRecord(
                    event=LogEvent.TOOL_CHOICE_UNSUPPORTED.value,
                    message="Anthropic tool_choice 'any' mapped to 'auto' (cache bypass).",
                    request_id=request_id,
                )
            )
            return "auto"
        if anthropic_choice.type == "tool" and anthropic_choice.name:
            return {"type": "function", "function": {"name": anthropic_choice.name}}
        return "auto"
    if not anthropic_choice:
        return None
    if anthropic_choice.type == "any":
        warning(
            LogRecord(
                event=LogEvent.TOOL_CHOICE_UNSUPPORTED.value,
                message="Anthropic tool_choice type 'any' mapped to OpenAI 'auto'. Exact behavior might differ (OpenAI 'auto' allows no tool use).",
                request_id=request_id,
                data={"anthropic_tool_choice": anthropic_choice.model_dump()},
            )
        )
    return _tool_choice_cache((anthropic_choice.type, anthropic_choice.name))


def convert_openai_to_anthropic_response(
    openai_response: openai.types.chat.ChatCompletion,
    original_anthropic_model_name: str,
    request_id: Optional[str] = None,
) -> MessagesResponse:
    """Convert an OpenAI ChatCompletion response into an Anthropic
    ``MessagesResponse`` instance.

    Parameters
    ----------
    openai_response
        The response object returned from ``create_chat_completion``.
    original_anthropic_model_name
        The model name originally requested by the Anthropic client.
    request_id
        Correlator for structured logging.
    """
    anthropic_content: List[ContentBlock] = []
    anthropic_stop_reason: StopReasonType = None

    stop_reason_map: Dict[Optional[str], StopReasonType] = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "content_filter": "stop_sequence",
        None: "end_turn",
    }

    if openai_response.choices:
        choice = openai_response.choices[0]
        message = choice.message
        finish_reason = choice.finish_reason

        anthropic_stop_reason = stop_reason_map.get(finish_reason, "end_turn")

        if message.content:
            anthropic_content.append(
                ContentBlockText(type="text", text=message.content)
            )

        if message.tool_calls:
            for call in message.tool_calls:
                if call.type == "function":
                    tool_input_dict: Dict[str, Any] = {}
                    try:
                        parsed_input = json.loads(call.function.arguments)
                        if isinstance(parsed_input, dict):
                            tool_input_dict = parsed_input
                        else:
                            tool_input_dict = {"value": parsed_input}
                            warning(
                                LogRecord(
                                    event=LogEvent.TOOL_ARGS_TYPE_MISMATCH.value,
                                    message=f"OpenAI tool arguments for '{call.function.name}' parsed to non-dict type '{type(parsed_input).__name__}'. Wrapped in 'value'.",
                                    request_id=request_id,
                                    data={
                                        "tool_name": call.function.name,
                                        "tool_id": call.id,
                                    },
                                )
                            )
                    except json.JSONDecodeError as e:
                        error(
                            LogRecord(
                                event=LogEvent.TOOL_ARGS_PARSE_FAILURE.value,
                                message=f"Failed to parse JSON arguments for tool '{call.function.name}'. Storing raw string.",
                                request_id=request_id,
                                data={
                                    "tool_name": call.function.name,
                                    "tool_id": call.id,
                                    "raw_args": call.function.arguments,
                                },
                            ),
                            exc=e,
                        )
                        tool_input_dict = {
                            "error_parsing_arguments": call.function.arguments
                        }

                    anthropic_content.append(
                        ContentBlockToolUse(
                            type="tool_use",
                            id=call.id,
                            name=call.function.name,
                            input=tool_input_dict,
                        )
                    )
            if finish_reason == "tool_calls":
                anthropic_stop_reason = "tool_use"

    if not anthropic_content:
        anthropic_content.append(ContentBlockText(type="text", text=""))

    usage = openai_response.usage
    anthropic_usage = Usage(
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
    )

    response_id = (
        f"msg_{openai_response.id}"
        if openai_response.id
        else f"msg_{request_id}_completed"
    )

    return MessagesResponse(
        id=response_id,
        type="message",
        role="assistant",
        model=original_anthropic_model_name,
        content=anthropic_content,
        stop_reason=anthropic_stop_reason,
        usage=anthropic_usage,
    )
