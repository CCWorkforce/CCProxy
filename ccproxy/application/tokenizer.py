import json
import tiktoken
import time
import hashlib
import anyio
from anyio.abc import Lock as AnyioLock
from dataclasses import dataclass, field
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
from ..logging import warning, debug, LogRecord, LogEvent, info
from ccproxy.config import Settings, TruncationConfig
from .._cython import CYTHON_ENABLED

# Try to import Cython-optimized functions
if CYTHON_ENABLED:
    try:
        from .._cython.lru_ops import (
            get_shard_index,
            is_expired,
            calculate_hit_rate,
            calculate_max_per_shard,
        )
        from .._cython.cache_keys import (
            compute_sha256_hex_from_str,
        )
        from .._cython.json_ops import (
            json_dumps_sorted,
            json_dumps_compact,
        )

        _USING_CYTHON = True
    except ImportError:
        _USING_CYTHON = False
else:
    _USING_CYTHON = False

# Fallback to pure Python implementations if Cython not available
if not _USING_CYTHON:

    def get_shard_index(key: str, num_shards: int) -> int:
        """Get shard index for a key using consistent hashing."""
        return hash(key) % num_shards

    def is_expired(timestamp: float, current_time: float, ttl_seconds: float) -> bool:
        """Check if a timestamp has expired based on TTL."""
        return (current_time - timestamp) > ttl_seconds

    def calculate_hit_rate(hits: int, total: int) -> float:
        """Calculate hit rate percentage."""
        return (hits / total * 100) if total > 0 else 0.0

    def calculate_max_per_shard(max_entries: int, num_shards: int) -> int:
        """Calculate maximum entries per shard."""
        return max(1, max_entries // num_shards)

    def compute_sha256_hex_from_str(text: str) -> str:
        """Compute SHA256 hash of string and return hexadecimal digest."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def json_dumps_sorted(obj: Any) -> str:
        """JSON serialization with sorted keys for cache consistency."""
        return json.dumps(
            obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )

    def json_dumps_compact(obj: Any) -> str:
        """Compact JSON serialization with minimal separators."""
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


class TokenEncoder(Protocol):
    """Protocol for token encoders that can encode text into tokens."""

    def encode(self, text: str) -> List[int]: ...


@dataclass
class CacheEntry:
    """Cache entry for token counts with timestamp for TTL management.

    Preserves count and insertion time for LRU eviction and TTL expiration checks.
    """

    count: int
    timestamp: float


@dataclass
class TokenCacheShard:
    """Single shard of the token cache with its own lock."""

    lock: AnyioLock = field(default_factory=lambda: anyio.Lock())
    cache: Dict[str, CacheEntry] = field(default_factory=dict)
    lru_order: List[str] = field(default_factory=list)
    last_cleanup_time: float = field(default_factory=lambda: time.time())


# Global cache for token encoders (doesn't need sharding - small and rarely accessed)
_token_encoder_cache: Dict[str, TokenEncoder] = {}

# Sharded cache for token counts
_token_cache_shards: List[TokenCacheShard] = []
_num_shards: int = 16
_shards_initialized: bool = False

# Global statistics (shared across all shards)
_token_count_hits = 0
_token_count_misses = 0

# Background cleanup task state
_cleanup_task: Optional[anyio.abc.CancelScope] = None
_cleanup_enabled: bool = False


def _ensure_shards_initialized(num_shards: int = 16) -> None:
    """Initialize cache shards on first use or when shard count changes."""
    global _token_cache_shards, _shards_initialized, _num_shards
    if not _shards_initialized or _num_shards != num_shards:
        _num_shards = num_shards
        _token_cache_shards = [TokenCacheShard() for _ in range(_num_shards)]
        _shards_initialized = True
        info(
            LogRecord(
                event=LogEvent.TOKEN_COUNT.value,
                message=f"Initialized {_num_shards} token cache shards",
                data={"num_shards": _num_shards},
            )
        )


def _get_shard_for_key(key: str) -> TokenCacheShard:
    """Select shard using consistent hashing."""
    shard_index = get_shard_index(key, _num_shards)
    return _token_cache_shards[shard_index]


def _batch_cleanup_expired_entries(
    shard: TokenCacheShard, current_time: float, ttl_seconds: float
) -> int:
    """Remove all expired entries from a shard in one pass.

    This is more efficient than removing expired entries one at a time during lookups.
    Should be called while holding the shard lock.

    Args:
        shard: The cache shard to clean up
        current_time: Current timestamp
        ttl_seconds: TTL in seconds for cache entries

    Returns:
        Number of entries removed
    """
    removed_count = 0
    keys_to_remove = []

    # Identify all expired entries
    for key, entry in shard.cache.items():
        if is_expired(entry.timestamp, current_time, ttl_seconds):
            keys_to_remove.append(key)

    # Remove expired entries
    for key in keys_to_remove:
        shard.cache.pop(key, None)
        try:
            shard.lru_order.remove(key)
        except ValueError:
            pass
        removed_count += 1

    if removed_count > 0:
        shard.last_cleanup_time = current_time

    return removed_count


def _should_trigger_cleanup(
    shard: TokenCacheShard,
    max_per_shard: int,
    cleanup_threshold_multiplier: float = 1.5,
) -> bool:
    """Determine if cleanup should be triggered based on shard size.

    Uses lazy cleanup strategy: only trigger when cache exceeds threshold.
    Default threshold is 150% of max_per_shard to reduce cleanup frequency.

    Args:
        shard: The cache shard to check
        max_per_shard: Maximum entries per shard
        cleanup_threshold_multiplier: Multiplier for cleanup threshold (default 1.5 = 150%)

    Returns:
        True if cleanup should be triggered
    """
    threshold = int(max_per_shard * cleanup_threshold_multiplier)
    return len(shard.cache) > threshold


def get_token_encoder(
    model_name: str = "gpt-4", request_id: Optional[str] = None
) -> TokenEncoder:
    """Retrieve and cache the tokenizer encoder for the given model.

    Falls back to the ``cl100k_base`` encoder when the requested model is
    unknown. Encoders are cached per model to avoid repeated construction cost.
    """
    if model_name in _token_encoder_cache:
        return _token_encoder_cache[model_name]

    try:
        encoder = tiktoken.encoding_for_model(model_name)
    except KeyError:
        warning(
            LogRecord(
                event=LogEvent.TOKEN_COUNT.value,
                message=(
                    "Unknown model %s, falling back to cl100k_base encoder" % model_name
                ),
                request_id=request_id,
            )
        )
        encoder = tiktoken.get_encoding("cl100k_base")
    except Exception:
        warning(
            LogRecord(
                event=LogEvent.TOKEN_COUNT.value,
                message=(
                    "Failed to load encoder for %s, using cl100k_base fallback"
                    % model_name
                ),
                request_id=request_id,
            )
        )
        encoder = tiktoken.get_encoding("cl100k_base")

    _token_encoder_cache[model_name] = encoder
    return encoder


async def _stable_hash_for_token_inputs(
    messages: List[Message],
    system: Optional[Union[str, List[SystemContent]]],
    model_name: str,
    tools: Optional[List[Tool]],
) -> str:
    """Generate a stable hash key for caching token counts."""
    payload = {
        "model": model_name,
        "messages": [m.model_dump(exclude_unset=True) for m in messages],
        "system": system
        if isinstance(system, str)
        else [s.model_dump(exclude_unset=True) for s in (system or [])],
        "tools": [t.model_dump(exclude_unset=True) for t in (tools or [])],
    }
    # Use Cython-optimized JSON serialization
    json_dumps_async = asyncify(json_dumps_sorted)
    serialized = await json_dumps_async(payload)
    # Use Cython-optimized hashing
    hash_compute_async = asyncify(compute_sha256_hex_from_str)
    return await hash_compute_async(serialized)


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

    **Token Cache TTL Configuration:**

    The token count cache uses a Time-To-Live (TTL) mechanism to balance performance,
    privacy, and memory usage. The default configuration is:

    - **TTL: 300 seconds (5 minutes)** - Cache entries expire after this duration
    - **Max Entries: 2048** - Maximum number of cached token counts

    **Trade-offs and Considerations:**

    1. **Privacy Implications:**
       - Shorter TTL (60-300s): Minimizes retention of hashed request patterns
       - Longer TTL (600-3600s): Request patterns stay in memory longer
       - Recommendation: Use 60s TTL for high-privacy environments (e.g., healthcare, finance)

    2. **Memory Usage:**
       - Each cache entry uses ~200 bytes (hash key + count + timestamp)
       - 2048 entries ≈ 400KB base memory
       - Recommendation: Reduce max_entries to 512 for memory-constrained deployments

    3. **Performance Impact:**
       - Cache hits save 5-50ms per request (tiktoken encoding overhead)
       - Hit rate typically 60-80% for production workloads with repeated patterns
       - Miss penalty is minimal (one-time encoding cost)

    4. **Recommended Configurations by Use Case:**

       **High-Privacy (Healthcare/Finance):**
       - cache_token_counts_ttl_s=60
       - cache_token_counts_max=512

       **Standard Production:**
       - cache_token_counts_ttl_s=300 (default)
       - cache_token_counts_max=2048 (default)

       **High-Performance (Internal Tools):**
       - cache_token_counts_ttl_s=3600
       - cache_token_counts_max=8192

       **Development/Testing:**
       - cache_token_counts_enabled=false (disable caching)

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
    ttl_s = 300  # Default: 5 minutes - balances performance with privacy
    max_entries = 2048  # Default: ~400KB memory footprint
    num_shards = 16  # Default number of shards
    if settings is not None:
        use_cache = settings.cache_token_counts_enabled
        ttl_s = int(settings.cache_token_counts_ttl_s)
        max_entries = int(settings.cache_token_counts_max)
        num_shards = int(settings.tokenizer_cache_shards)

    if use_cache:
        # Ensure shards are initialized with the correct number only when caching is enabled
        _ensure_shards_initialized(num_shards)
        key = await _stable_hash_for_token_inputs(messages, system, model_name, tools)
        now = time.time()

        # Get the appropriate shard for this key
        shard = _get_shard_for_key(key)

        async with shard.lock:
            if key in shard.cache:
                entry = shard.cache[key]
                # Use Cython-optimized expiry check
                if not is_expired(entry.timestamp, now, ttl_s):
                    global _token_count_hits
                    _token_count_hits += 1
                    debug(
                        LogRecord(
                            LogEvent.TOKEN_COUNT.value,
                            "Token count cache hit",
                            request_id,
                            {
                                "key": key[:8],
                                "age_s": round(now - entry.timestamp, 3),
                                "shard_id": get_shard_index(key, _num_shards),
                            },
                        )
                    )
                    # Update LRU position on hit (move to end)
                    if key in shard.lru_order:
                        shard.lru_order.remove(key)
                    shard.lru_order.append(key)
                    return entry.count
                else:
                    # Entry is expired - trigger batch cleanup of all expired entries
                    # This is more efficient than cleaning one at a time
                    removed_count = _batch_cleanup_expired_entries(shard, now, ttl_s)
                    if removed_count > 0:
                        debug(
                            LogRecord(
                                LogEvent.TOKEN_COUNT.value,
                                f"Batch cleaned {removed_count} expired entries from shard",
                                request_id,
                                {
                                    "removed_count": removed_count,
                                    "shard_id": get_shard_index(key, _num_shards),
                                },
                            )
                        )
            # miss path after lock
        global _token_count_misses
        _token_count_misses += 1

    enc = get_token_encoder(model_name, request_id)

    # Create async encode function
    encode_async = asyncify(enc.encode)

    # Helper function to encode text
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
                        # Serialize tool input asynchronously with Cython-optimized JSON
                        json_dumps_async = asyncify(json_dumps_compact)
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
                            # Build content string asynchronously with Cython-optimized JSON
                            json_dumps_async = asyncify(json_dumps_compact)
                            for item in block.content:
                                if (
                                    isinstance(item, dict)
                                    and item.get("type") == "text"
                                ):
                                    content_str += item.get("text", "")
                                else:
                                    content_str += await json_dumps_async(item)
                        else:
                            json_dumps_async = asyncify(json_dumps_compact)
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

    # Initialize total_tokens with fixed overhead
    total_tokens = fixed_tokens

    # Process tool definitions
    if tools:
        total_tokens += 2  # Tools structure overhead
        tool_tasks = []

        for tool in tools:
            tool_tasks.append(("tool_name", encode_text(tool.name)))
            if tool.description:
                tool_tasks.append(("tool_desc", encode_text(tool.description)))
            try:
                # Use Cython-optimized JSON serialization
                json_dumps_async = asyncify(json_dumps_compact)
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
            # Execute the coroutines using anyio task group
            tool_results = []
            async with anyio.create_task_group() as tg:

                async def run_tool_task(coro, idx):
                    result = await coro
                    tool_results.append((idx, result))

                for idx, (_, coro) in enumerate(tool_tasks):
                    tg.start_soon(run_tool_task, coro, idx)

            # Sort results by original index and add tool token counts
            tool_results.sort(key=lambda x: x[0])
            for _, result in tool_results:
                total_tokens += result

    # Process all encoding tasks in parallel
    if encoding_tasks:
        # Execute the coroutines using anyio task group
        task_results = []
        async with anyio.create_task_group() as tg:

            async def run_encode_task(coro, idx):
                result = await coro
                task_results.append((idx, result))

            for idx, (_, coro) in enumerate(encoding_tasks):
                tg.start_soon(run_encode_task, coro, idx)

        # Sort results by original index and sum up token counts
        task_results.sort(key=lambda x: x[0])
        for _, result in task_results:
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

        # Get the appropriate shard for this key
        shard = _get_shard_for_key(key)

        async with shard.lock:
            if key in shard.lru_order:
                shard.lru_order.remove(key)
            shard.lru_order.append(key)
            # Store using CacheEntry for structured timestamp management
            shard.cache[key] = CacheEntry(total_tokens, now)

            # Lazy cleanup strategy: only trigger cleanup when cache exceeds 150% of max size
            # This reduces cleanup frequency and improves performance
            max_per_shard = calculate_max_per_shard(max_entries, _num_shards)

            if _should_trigger_cleanup(shard, max_per_shard, cleanup_threshold_multiplier=1.5):
                # First, try to clean up expired entries
                removed_count = _batch_cleanup_expired_entries(shard, now, ttl_s)

                if removed_count > 0:
                    debug(
                        LogRecord(
                            LogEvent.TOKEN_COUNT.value,
                            f"Lazy cleanup removed {removed_count} expired entries",
                            request_id,
                            {
                                "removed_count": removed_count,
                                "shard_size": len(shard.cache),
                                "shard_id": get_shard_index(key, _num_shards),
                            },
                        )
                    )

                # If still over capacity after expiry cleanup, evict oldest entries via LRU
                while len(shard.lru_order) > max_per_shard:
                    evict_key = shard.lru_order.pop(0)
                    shard.cache.pop(evict_key, None)

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
        # Use Cython-optimized JSON serialization
        json_dumps_async = asyncify(json_dumps_compact)

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
            # Execute the coroutines using anyio task group
            task_results = []
            async with anyio.create_task_group() as tg:

                async def run_encode_task(coro, idx):
                    result = await coro
                    task_results.append((idx, result))

                for idx, (_, coro) in enumerate(encoding_tasks):
                    tg.start_soon(run_encode_task, coro, idx)

            # Sort results by original index and sum up token counts
            task_results.sort(key=lambda x: x[0])
            for _, result in task_results:
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


def get_token_cache_stats() -> Dict[str, Any]:
    """Get aggregated statistics across all cache shards.

    Returns:
        Dictionary containing cache statistics including:
        - total_entries: Total cached entries across all shards
        - hits: Total cache hits
        - misses: Total cache misses
        - hit_rate: Cache hit rate percentage
        - shard_distribution: Entry count per shard
        - most_loaded_shard: ID of shard with most entries
        - least_loaded_shard: ID of shard with least entries
    """
    if not _shards_initialized:
        return {
            "total_entries": 0,
            "hits": _token_count_hits,
            "misses": _token_count_misses,
            "hit_rate": 0.0,
            "shard_distribution": [],
            "num_shards": 0,
            "initialized": False,
            "cleanup_enabled": _cleanup_enabled,
        }

    shard_sizes = []
    total_entries = 0

    for i, shard in enumerate(_token_cache_shards):
        shard_size = len(shard.cache)
        shard_sizes.append(shard_size)
        total_entries += shard_size

    total_requests = _token_count_hits + _token_count_misses
    # Use Cython-optimized hit rate calculation
    hit_rate = calculate_hit_rate(_token_count_hits, total_requests)

    return {
        "total_entries": total_entries,
        "hits": _token_count_hits,
        "misses": _token_count_misses,
        "hit_rate": hit_rate,
        "shard_distribution": shard_sizes,
        "num_shards": _num_shards,
        "most_loaded_shard": shard_sizes.index(max(shard_sizes)) if shard_sizes else -1,
        "least_loaded_shard": shard_sizes.index(min(shard_sizes))
        if shard_sizes
        else -1,
        "avg_entries_per_shard": total_entries / _num_shards if _num_shards > 0 else 0,
        "initialized": True,
        "cleanup_enabled": _cleanup_enabled,
    }


async def periodic_cache_cleanup(
    interval_seconds: int = 60, ttl_seconds: float = 300
) -> None:
    """Background task that periodically cleans up expired entries from all shards.

    This function should be run as a background task during application startup.
    It will continuously run until cancelled, cleaning up expired entries from all
    cache shards at regular intervals.

    Args:
        interval_seconds: Time between cleanup runs (default: 60 seconds)
        ttl_seconds: TTL for cache entries (default: 300 seconds / 5 minutes)
    """
    global _cleanup_enabled
    _cleanup_enabled = True

    info(
        LogRecord(
            event=LogEvent.TOKEN_COUNT.value,
            message=f"Started periodic token cache cleanup (interval={interval_seconds}s, ttl={ttl_seconds}s)",
            data={"interval_seconds": interval_seconds, "ttl_seconds": ttl_seconds},
        )
    )

    try:
        while True:
            await anyio.sleep(interval_seconds)

            if not _shards_initialized:
                continue

            total_removed = 0
            now = time.time()

            # Clean up each shard
            for i, shard in enumerate(_token_cache_shards):
                async with shard.lock:
                    removed_count = _batch_cleanup_expired_entries(shard, now, ttl_seconds)
                    total_removed += removed_count

            if total_removed > 0:
                info(
                    LogRecord(
                        event=LogEvent.TOKEN_COUNT.value,
                        message=f"Periodic cleanup removed {total_removed} expired entries across {_num_shards} shards",
                        data={
                            "removed_count": total_removed,
                            "num_shards": _num_shards,
                            "avg_per_shard": total_removed / _num_shards if _num_shards > 0 else 0,
                        },
                    )
                )
    except anyio.get_cancelled_exc_class():
        _cleanup_enabled = False
        info(
            LogRecord(
                event=LogEvent.TOKEN_COUNT.value,
                message="Periodic token cache cleanup task cancelled",
            )
        )
        raise
