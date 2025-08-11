import json
import tiktoken
import time
import hashlib
import threading
from typing import Dict, List, Optional, Union, Tuple

from ..domain.models import Message, SystemContent, Tool, ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult, ContentBlockThinking, ContentBlockRedactedThinking
from ..logging import warning, debug, LogRecord, LogEvent
from ..config import Settings


_token_encoder_cache: Dict[str, tiktoken.Encoding] = {}
_token_count_cache: Dict[str, Tuple[int, float]] = {}
_token_count_lru_order: List[str] = []
_token_count_hits = 0
_token_count_misses = 0
_token_lock = threading.Lock()


def get_token_encoder(
    model_name: str = "gpt-4", request_id: Optional[str] = None
) -> tiktoken.Encoding:
    """Gets a tiktoken encoder, caching it for performance."""

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


def _stable_hash_for_token_inputs(messages: List[Message], system: Optional[Union[str, List[SystemContent]]], model_name: str, tools: Optional[List[Tool]]) -> str:
    payload = {
        "model": model_name,
        "messages": [m.model_dump(exclude_unset=True) for m in messages],
        "system": system if isinstance(system, str) else [s.model_dump(exclude_unset=True) for s in (system or [])],
        "tools": [t.model_dump(exclude_unset=True) for t in (tools or [])],
    }
    j = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(j.encode("utf-8")).hexdigest()


def count_tokens_for_anthropic_request(
    messages: List[Message],
    system: Optional[Union[str, List[SystemContent]]],
    model_name: str,
    tools: Optional[List[Tool]] = None,
    request_id: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> int:
    use_cache = True
    ttl_s = 300
    max_entries = 2048
    if settings is not None:
        use_cache = settings.cache_token_counts_enabled
        ttl_s = int(settings.cache_token_counts_ttl_s)
        max_entries = int(settings.cache_token_counts_max)

    if use_cache:
        key = _stable_hash_for_token_inputs(messages, system, model_name, tools)
        now = time.time()
        with _token_lock:
            if key in _token_count_cache:
                count, ts = _token_count_cache[key]
                if now - ts <= ttl_s:
                    global _token_count_hits
                    _token_count_hits += 1
                    debug(LogRecord(LogEvent.TOKEN_COUNT.value, "Token count cache hit", request_id, {"key": key[:8], "age_s": round(now - ts, 3)}))
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
    total_tokens = 0

    if isinstance(system, str):
        total_tokens += len(enc.encode(system))
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, SystemContent) and block.type == "text":
                total_tokens += len(enc.encode(block.text))

    for msg in messages:
        total_tokens += 4
        if msg.role:
            total_tokens += len(enc.encode(msg.role))

        if isinstance(msg.content, str):
            total_tokens += len(enc.encode(msg.content))
        elif isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, ContentBlockText):
                    total_tokens += len(enc.encode(block.text))
                elif isinstance(block, ContentBlockImage):
                    total_tokens += 768
                elif isinstance(block, ContentBlockToolUse):
                    total_tokens += len(enc.encode(block.name))
                    try:
                        input_str = json.dumps(block.input)
                        total_tokens += len(enc.encode(input_str))
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
                            for item in block.content:
                                if (
                                    isinstance(item, dict)
                                    and item.get("type") == "text"
                                ):
                                    content_str += item.get("text", "")
                                else:
                                    content_str += json.dumps(item)
                        else:
                            content_str = json.dumps(block.content)
                        total_tokens += len(enc.encode(content_str))
                    except Exception:
                        warning(
                            LogRecord(
                                event=LogEvent.TOOL_RESULT_SERIALIZATION_FAILURE.value,
                                message="Failed to serialize tool result for token counting.",
                                request_id=request_id,
                            )
                        )
                elif isinstance(block, ContentBlockThinking):
                    total_tokens += len(enc.encode(block.thinking))
                elif isinstance(block, ContentBlockRedactedThinking):
                    # Redacted thinking blocks don't contribute to visible tokens
                    # but should still be counted as they represent computation
                    total_tokens += 100  # Placeholder token count for redacted thinking

    if tools:
        total_tokens += 2
        for tool in tools:
            total_tokens += len(enc.encode(tool.name))
            if tool.description:
                total_tokens += len(enc.encode(tool.description))
            try:
                schema_str = json.dumps(tool.input_schema)
                total_tokens += len(enc.encode(schema_str))
            except Exception:
                warning(
                    LogRecord(
                        event=LogEvent.TOOL_INPUT_SERIALIZATION_FAILURE.value,
                        message="Failed to serialize tool schema for token counting.",
                        data={"tool_name": tool.name},
                        request_id=request_id,
                    )
                )
    debug(
        LogRecord(
            event=LogEvent.TOKEN_COUNT.value,
            message=f"Estimated {total_tokens} input tokens for model {model_name}",
            data={"model": model_name, "token_count": total_tokens},
            request_id=request_id,
        )
    )

    if use_cache:
        key = _stable_hash_for_token_inputs(messages, system, model_name, tools)
        now = time.time()
        with _token_lock:
            _token_count_cache[key] = (total_tokens, now)
            if key in _token_count_lru_order:
                _token_count_lru_order.remove(key)
            _token_count_lru_order.append(key)
            while len(_token_count_lru_order) > max_entries:
                evict_key = _token_count_lru_order.pop(0)
                _token_count_cache.pop(evict_key, None)

    return total_tokens
