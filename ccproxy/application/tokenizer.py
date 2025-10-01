import json
import tiktoken
import time
import hashlib
import asyncio
from typing import Dict, List, Optional, Union, Tuple, Protocol, Any

from .thread_pool import asyncify

from ..domain.models import (
    Message,
    SystemContent,
    Tool,
    ContentBlockText,
    ContentBlockImage,
    ContentBlockToolUse,
    ContentBlockToolResult,
    ContentBlockThinking,
    ContentBlockRedactedThinking,
)
from ..logging import warning, debug, LogRecord, LogEvent
from ccproxy.config import Settings, TruncationConfig


class TokenEncoder(Protocol):
    """Protocol for token encoders that can encode text into tokens."""

    def encode(self, text: str) -> List[int]: ...


_token_encoder_cache: Dict[str, TokenEncoder] = {}
_token_count_cache: Dict[str, Tuple[int, float]] = {}
_token_count_lru_order: List[str] = []
_token_count_hits = 0
_token_count_misses = 0
_token_lock = asyncio.Lock()


def get_token_encoder(
    model_name: str = "gpt-4", request_id: Optional[str] = None
) -> TokenEncoder:
    """Retrieves or caches a tiktoken encoder for the specified model.

    Args:
        model_name: The name of the model for which to get the encoder. Defaults to 'gpt-4'.
        request_id: Optional request identifier for logging. Defaults to None.

    Returns:
        TokenEncoder: The encoder instance for the specified model.
    """

    cache_key = model_name
    if cache_key not in _token_encoder_cache:
        try:
            _token_encoder_cache[cache_key] = tiktoken.encoding_for_model(model_name)
        except Exception:
            try:
                _token_encoder_cache[cache_key] = tiktoken.get_encoding("cl100k_base")
                warning(
                    LogRecord(
                        event=LogEvent.TOKEN_ENCODER_LOAD_FAILED.value,
                        message=f"Could not load tiktoken encoder for '{model_name}', using 'cl100k_base'. Token counts may be approximate.",
                        request_id=request_id,
                        data={"model_tried": model_name},
                    )
                )
            except Exception as e_cl:
                warning(
                    LogRecord(
                        event=LogEvent.TOKEN_ENCODER_LOAD_FAILED.value,
                        message="Failed to load any tiktoken encoder. Token counting will be inaccurate.",
                        request_id=request_id,
                    ),
                    exc=e_cl,
                )

                class DummyEncoder:
                    def encode(self, text: str) -> List[int]:
                        return list(range(len(text) // 4))

                _token_encoder_cache[cache_key] = DummyEncoder()
    return _token_encoder_cache[cache_key]


async def _stable_hash_for_token_inputs(
    messages: List[Message],
    system: Optional[Union[str, List[SystemContent]]],
    model_name: str,
    tools: Optional[List[Tool]],
) -> str:
    """Generates a stable hash key for token count caching.

    Creates a deterministic hash based on the normalized representation of all
    input parameters to serve as a cache key for token counting results.

    Args:
        messages: The list of conversation messages.
        system: Optional system instructions (string or structured content).
        model_name: The name of the model being used.
        tools: Optional list of tool definitions.

    Returns:
        str: SHA-256 hash of the normalized input parameters.
    """
    payload = {
        "model": model_name,
        "messages": [m.model_dump(exclude_unset=True) for m in messages],
        "system": system
        if isinstance(system, str)
        else [s.model_dump(exclude_unset=True) for s in (system or [])],
        "tools": [t.model_dump(exclude_unset=True) for t in (tools or [])],
    }
    # Offload JSON serialization to thread pool for large payloads
    json_dumps_async = asyncify(json.dumps)
    j = await json_dumps_async(payload, sort_keys=True, separators=(",", ":"))

    # Hash computation is also CPU-intensive for large strings
    hash_compute_async = asyncify(
        lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()
    )
    return await hash_compute_async(j)


def _truncate_text(
    text: str,
    config: TruncationConfig,
    model_name: str,
    request_id: Optional[str] = None,
) -> str:
    """Truncate text chunks while preserving semantic meaning where possible."""
    if len(text) <= config.min_tokens:
        return text

    # Preserve beginning for context
    return text[: config.min_tokens]


async def truncate_request(
    messages: List[Message],
    system: Optional[Union[str, List[SystemContent]]],
    model_name: str,
    limit: int,
    config: TruncationConfig,
    request_id: Optional[str] = None,
) -> Tuple[List[Message], Optional[Union[str, List[SystemContent]]]]:
    """Truncate request content to fit within token limit."""

    # Clone to avoid modifying originals
    truncated_messages = messages.copy()
    truncated_system = system

    while (
        await count_tokens_for_anthropic_request(
            truncated_messages, truncated_system, model_name, None, request_id
        )
        > limit
    ):
        if not truncated_messages:
            break

        if config.strategy == "oldest_first":
            # Remove oldest non-system message
            for i, msg in enumerate(truncated_messages):
                if msg.role != "system":
                    truncated_messages.pop(i)
                    break
        elif config.strategy == "newest_first":
            # Remove newest message
            truncated_messages.pop()
        elif config.strategy == "system_priority" and truncated_system:
            # Keep system message, truncate messages
            truncated_messages.pop(0)
        else:
            # Fallback to oldest first
            truncated_messages.pop(0)

    return truncated_messages, truncated_system


async def count_tokens_for_anthropic_request(
    messages: List[Message],
    system: Optional[Union[str, List[SystemContent]]],
    model_name: str,
    tools: Optional[List[Tool]] = None,
    request_id: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> int:
    """Calculate the total number of tokens required for an Anthropic API request.

    This function counts tokens across messages, system content, and tool definitions,
    using the appropriate tokenizer for the specified model. Token counts are cached
    when enabled via settings for performance.

    Args:
        messages: List of conversation messages to process.
        system: Optional system instructions (string or structured content).
        model_name: Name of the model to determine tokenizer.
        tools: Optional list of tool definitions to include in count.
        request_id: Optional request identifier for logging purposes.
        settings: Optional application settings to control caching behavior.

    Returns:
        int: Total estimated token count for the input request.
    """
    use_cache = True
    ttl_s = 300
    max_entries = 2048
    if settings is not None:
        use_cache = settings.cache_token_counts_enabled
        ttl_s = int(settings.cache_token_counts_ttl_s)
        max_entries = int(settings.cache_token_counts_max)

    if use_cache:
        key = await _stable_hash_for_token_inputs(messages, system, model_name, tools)
        now = time.time()
        async with _token_lock:
            if key in _token_count_cache:
                count, ts = _token_count_cache[key]
                if now - ts <= ttl_s:
                    global _token_count_hits
                    _token_count_hits += 1
                    debug(
                        LogRecord(
                            LogEvent.TOKEN_COUNT.value,
                            "Token count cache hit",
                            request_id,
                            {"key": key[:8], "age_s": round(now - ts, 3)},
                        )
                    )
                    if key in _token_count_lru_order:
                        _token_count_lru_order.remove(key)
                    _token_count_lru_order.append(key)
                    return count
                else:
                    # expired; evict
                    _token_count_cache.pop(key, None)
                    try:
                        _token_count_lru_order.remove(key)
                    except ValueError:
                        pass
            # miss path after lock
        global _token_count_misses
        _token_count_misses += 1

    enc = get_token_encoder(model_name, request_id)

    # Create async encode function
    encode_async = asyncify(enc.encode)

    # Helper function to encode text blocks
    async def encode_text(text: str) -> int:
        tokens = await encode_async(text)
        return len(tokens)

    # Process all encoding tasks
    encoding_tasks = []
    fixed_tokens = 0

    # System prompt encoding
    if isinstance(system, str):
        encoding_tasks.append(("system", encode_text(system)))
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, SystemContent) and block.type == "text":
                encoding_tasks.append(("system_block", encode_text(block.text)))

    # Message encoding
    for msg in messages:
        fixed_tokens += 4  # Message structure overhead
        if msg.role:
            encoding_tasks.append(("role", encode_text(msg.role)))

        if isinstance(msg.content, str):
            encoding_tasks.append(("content", encode_text(msg.content)))
        elif isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, ContentBlockText):
                    encoding_tasks.append(("text_block", encode_text(block.text)))
                elif isinstance(block, ContentBlockImage):
                    fixed_tokens += 768  # Fixed token count for images
                elif isinstance(block, ContentBlockToolUse):
                    encoding_tasks.append(("tool_name", encode_text(block.name)))
                    try:
                        # Serialize tool input asynchronously
                        json_dumps_async = asyncify(json.dumps)
                        input_str = await json_dumps_async(block.input)
                        encoding_tasks.append(("tool_input", encode_text(input_str)))
                    except Exception:
                        warning(
                            LogRecord(
                                event=LogEvent.TOOL_INPUT_SERIALIZATION_FAILURE.value,
                                message="Failed to serialize tool input for token counting.",
                                data={"tool_name": block.name},
                                request_id=request_id,
                            )
                        )
                elif isinstance(block, ContentBlockToolResult):
                    try:
                        content_str = ""
                        if isinstance(block.content, str):
                            content_str = block.content
                        elif isinstance(block.content, list):
                            # Build content string asynchronously
                            json_dumps_async = asyncify(json.dumps)
                            for item in block.content:
                                if (
                                    isinstance(item, dict)
                                    and item.get("type") == "text"
                                ):
                                    content_str += item.get("text", "")
                                else:
                                    content_str += await json_dumps_async(item)
                        else:
                            json_dumps_async = asyncify(json.dumps)
                            content_str = await json_dumps_async(block.content)
                        encoding_tasks.append(("tool_result", encode_text(content_str)))
                    except Exception:
                        warning(
                            LogRecord(
                                event=LogEvent.TOOL_RESULT_SERIALIZATION_FAILURE.value,
                                message="Failed to serialize tool result for token counting.",
                                request_id=request_id,
                            )
                        )
                elif isinstance(block, ContentBlockThinking):
                    encoding_tasks.append(("thinking", encode_text(block.thinking)))
                elif isinstance(block, ContentBlockRedactedThinking):
                    # Redacted thinking blocks don't contribute to visible tokens
                    # but should still be counted as they represent computation
                    fixed_tokens += 100  # Placeholder token count for redacted thinking

    # Execute encoding tasks in parallel for better performance
    total_tokens = fixed_tokens

    if encoding_tasks:
        # Process encoding tasks in parallel using task group
        # Execute the coroutines directly since they're already created
        task_results = await asyncio.gather(*[coro for _, coro in encoding_tasks])

        # Sum up all token counts
        for result in task_results:
            total_tokens += result

    # Process tool definitions
    if tools:
        fixed_tokens += 2  # Tools structure overhead
        tool_tasks = []

        for tool in tools:
            tool_tasks.append(("tool_name", encode_text(tool.name)))
            if tool.description:
                tool_tasks.append(("tool_desc", encode_text(tool.description)))
            try:
                json_dumps_async = asyncify(json.dumps)
                schema_str = await json_dumps_async(tool.input_schema)
                tool_tasks.append(("tool_schema", encode_text(schema_str)))
            except Exception:
                warning(
                    LogRecord(
                        event=LogEvent.TOOL_INPUT_SERIALIZATION_FAILURE.value,
                        message="Failed to serialize tool schema for token counting.",
                        data={"tool_name": tool.name},
                        request_id=request_id,
                    )
                )

        # Process tool encoding tasks in parallel
        if tool_tasks:
            # Execute the coroutines directly
            tool_results = await asyncio.gather(*[coro for _, coro in tool_tasks])

            # Add tool token counts
            for result in tool_results:
                total_tokens += result
    debug(
        LogRecord(
            event=LogEvent.TOKEN_COUNT.value,
            message=f"Estimated {total_tokens} input tokens for model {model_name}",
            data={"model": model_name, "token_count": total_tokens},
            request_id=request_id,
        )
    )

    if use_cache:
        key = await _stable_hash_for_token_inputs(messages, system, model_name, tools)
        now = time.time()
        async with _token_lock:
            _token_count_cache[key] = (total_tokens, now)
            if key in _token_count_lru_order:
                _token_count_lru_order.remove(key)
            _token_count_lru_order.append(key)
            while len(_token_count_lru_order) > max_entries:
                evict_key = _token_count_lru_order.pop(0)
                _token_count_cache.pop(evict_key, None)

    return total_tokens


async def count_tokens_for_openai_request(
    messages: List[Dict[str, Union[str, List]]],
    model_name: str = "gpt-4",
    tools: Optional[List[Dict[str, Any]]] = None,
    request_id: Optional[str] = None,
) -> int:
    """Calculate the total number of tokens for an OpenAI API request.

    Args:
        messages: List of OpenAI message dictionaries
        model_name: The OpenAI model name for accurate encoding
        tools: Optional list of tool/function definitions
        request_id: Optional request identifier for logging

    Returns:
        int: Estimated total token count
    """
    try:
        enc = get_token_encoder(model_name, request_id)

        # Create async encode function
        encode_async = asyncify(enc.encode)
        json_dumps_async = asyncify(json.dumps)

        # Helper function to encode text
        async def encode_text(text: str) -> int:
            tokens = await encode_async(text)
            return len(tokens)

        # Collect all encoding tasks
        encoding_tasks = []
        fixed_tokens = 0

        # Count tokens in messages
        for message in messages:
            # Count role tokens
            role = message.get("role", "")
            if role:
                encoding_tasks.append(("role", encode_text(role)))

            # Count content tokens
            content = message.get("content", "")
            if isinstance(content, str):
                encoding_tasks.append(("content", encode_text(content)))
            elif isinstance(content, list):
                # Handle multi-part content (e.g., text + images)
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        encoding_tasks.append(
                            ("text_part", encode_text(part.get("text", "")))
                        )
                    elif isinstance(part, dict) and part.get("type") == "image_url":
                        # Images have fixed token cost
                        fixed_tokens += 85  # Base cost for image

            # Count tool/function call tokens
            if "tool_calls" in message:
                for tool_call in message.get("tool_calls", []):
                    if "function" in tool_call:
                        func = tool_call["function"]
                        encoding_tasks.append(
                            ("func_name", encode_text(func.get("name", "")))
                        )
                        if "arguments" in func:
                            encoding_tasks.append(
                                ("func_args", encode_text(func.get("arguments", "")))
                            )

            # Legacy function_call support
            if "function_call" in message:
                func = message["function_call"]
                encoding_tasks.append(
                    ("legacy_func_name", encode_text(func.get("name", "")))
                )
                if "arguments" in func:
                    encoding_tasks.append(
                        ("legacy_func_args", encode_text(func.get("arguments", "")))
                    )

        # Count tokens in tools/functions
        if tools:
            for tool in tools:
                # Tool type and function wrapper
                fixed_tokens += 10  # Overhead for tool structure

                if "function" in tool:
                    func = tool["function"]
                    encoding_tasks.append(
                        ("tool_func_name", encode_text(func.get("name", "")))
                    )
                    if "description" in func:
                        encoding_tasks.append(
                            ("tool_desc", encode_text(func.get("description", "")))
                        )
                    if "parameters" in func:
                        params_str = await json_dumps_async(func["parameters"])
                        encoding_tasks.append(("tool_params", encode_text(params_str)))

        # Add message structure overhead (~3 tokens per message)
        fixed_tokens += len(messages) * 3

        # Execute all encoding tasks in parallel
        total_tokens = fixed_tokens

        if encoding_tasks:
            # Execute the coroutines directly
            task_results = await asyncio.gather(*[coro for _, coro in encoding_tasks])

            # Sum up all token counts
            for result in task_results:
                total_tokens += result

        debug(
            LogRecord(
                event=LogEvent.TOKEN_COUNT.value,
                message=f"Counted {total_tokens} OpenAI tokens for model {model_name}",
                data={"model": model_name, "token_count": total_tokens},
                request_id=request_id,
            )
        )

        return total_tokens

    except Exception as e:
        warning(
            LogRecord(
                event=LogEvent.TOKEN_COUNT.value,
                message=f"Failed to count OpenAI tokens accurately, using rough estimate: {e}",
                request_id=request_id,
            )
        )
        # Fallback to rough estimate
        total_chars = 0
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
        return max(1, total_chars // 4)  # ~4 chars per token
