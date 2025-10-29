import subprocess
import sys
import time

import anyio
import pytest
from ccproxy.application.tokenizer import (
    truncate_request,
    count_tokens_for_anthropic_request,
)
from ccproxy.domain.models import Message, ContentBlockText
from ccproxy.config import TruncationConfig, Settings  # type: ignore[attr-defined]
from typing import Any


def create_test_messages(count: int, role: str = "user") -> list[Message]:
    # Repeat text to ensure it exceeds token limits
    long_text = (
        "This is a longer message content that should push token count higher. " * 20
    )
    return [
        Message(
            role=role,
            content=[ContentBlockText(type="text", text=f"{long_text} Message {i}")],
        )
        for i in range(count)
    ]


@pytest.mark.anyio
async def test_truncate_oldest_first() -> None:
    messages = create_test_messages(5)
    system = "System prompt"
    config = TruncationConfig()

    truncated_msgs, truncated_system = await truncate_request(
        messages,
        system,
        "gpt-4",
        150,
        config,
        request_id="test",  # type: ignore[arg-type]
    )
    new_tokens = await count_tokens_for_anthropic_request(
        truncated_msgs, truncated_system, "gpt-4", None, None
    )

    assert new_tokens <= 150
    # Check that some messages were removed
    assert len(truncated_msgs) < len(messages)


@pytest.mark.anyio
async def test_truncate_newest_first() -> None:
    messages = create_test_messages(5)
    system = "System prompt"

    # Create config with newest_first strategy
    config = TruncationConfig()
    config.strategy = "newest_first"

    truncated_msgs, truncated_system = await truncate_request(
        messages,
        system,
        "gpt-4",
        150,
        config,
        request_id="test",  # type: ignore[arg-type]
    )
    new_tokens = await count_tokens_for_anthropic_request(
        truncated_msgs, truncated_system, "gpt-4", None, None
    )

    assert new_tokens <= 150
    # Check that some messages were removed
    assert len(truncated_msgs) < len(messages)


@pytest.mark.anyio
async def test_truncate_system_priority() -> None:
    messages = create_test_messages(5)
    system = "System prompt"

    # Create config with system_priority strategy
    config = TruncationConfig()
    config.strategy = "system_priority"

    truncated_msgs, truncated_system = await truncate_request(
        messages,
        system,
        "gpt-4",
        150,
        config,
        request_id="test",  # type: ignore[arg-type]
    )
    new_tokens = await count_tokens_for_anthropic_request(
        truncated_msgs, truncated_system, "gpt-4", None, None
    )

    assert new_tokens <= 150
    assert truncated_system == "System prompt"


@pytest.mark.anyio
async def test_truncate_below_limit() -> None:
    messages = create_test_messages(2)
    system = "System prompt"
    config = TruncationConfig()

    truncated_msgs, truncated_system = await truncate_request(
        messages,
        system,
        "gpt-4",
        10000,
        config,  # type: ignore[arg-type]
    )

    assert len(truncated_msgs) == 2
    assert truncated_system == system


def test_tokenizer_import_without_event_loop() -> None:
    """Importing tokenizer in a fresh interpreter should not require a loop."""
    result = subprocess.run(
        [sys.executable, "-c", "import ccproxy.application.tokenizer"],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        pytest.fail(
            "Importing ccproxy.application.tokenizer raised unexpectedly without an event loop:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


# === New Sharded Cache Tests ===


@pytest.fixture
async def clean_cache_state() -> Any:
    """Fixture to reset cache state between tests."""
    import ccproxy.application.tokenizer as tokenizer

    # Reset global state before test
    tokenizer._token_cache_shards = []
    tokenizer._shards_initialized = False
    tokenizer._token_count_hits = 0
    tokenizer._token_count_misses = 0
    tokenizer._num_shards = 16  # Reset to default
    yield
    # Clean up after test
    tokenizer._token_cache_shards = []
    tokenizer._shards_initialized = False
    tokenizer._num_shards = 16  # Reset to default


@pytest.mark.anyio
async def test_sharded_cache_initialization(clean_cache_state: Any) -> None:
    """Verify shards initialize correctly."""
    import ccproxy.application.tokenizer as tokenizer

    # Initialize with 8 shards
    tokenizer._ensure_shards_initialized(num_shards=8)

    assert len(tokenizer._token_cache_shards) == 8
    assert tokenizer._num_shards == 8
    assert tokenizer._shards_initialized is True

    # Verify each shard has independent lock/cache/lru
    for shard in tokenizer._token_cache_shards:
        assert hasattr(shard, "lock")
        assert hasattr(shard, "cache")
        assert hasattr(shard, "lru_order")
        assert isinstance(shard.cache, dict)
        assert isinstance(shard.lru_order, list)


@pytest.mark.anyio
async def test_shard_distribution(clean_cache_state: Any) -> None:
    """Verify keys distribute reasonably across shards."""
    import ccproxy.application.tokenizer as tokenizer

    # Initialize with 16 shards
    tokenizer._ensure_shards_initialized(num_shards=16)

    # Generate 1000 unique keys and track distribution
    shard_counts = [0] * 16
    for i in range(1000):
        key = f"test_key_{i}"
        shard = tokenizer._get_shard_for_key(key)
        shard_index = tokenizer._token_cache_shards.index(shard)
        shard_counts[shard_index] += 1

    # Each shard should have roughly 1000/16 = 62.5 keys
    expected_per_shard = 1000 / 16
    min_acceptable = expected_per_shard * 0.5  # 50% below expected
    max_acceptable = expected_per_shard * 1.5  # 50% above expected

    for count in shard_counts:
        assert min_acceptable <= count <= max_acceptable, (
            f"Shard distribution uneven: {count} not in range [{min_acceptable}, {max_acceptable}]"
        )


@pytest.mark.anyio
async def test_concurrent_shard_access(clean_cache_state: Any) -> None:
    """Verify different shards don't block each other."""

    # Create settings with sharding enabled
    settings = Settings(
        openai_api_key="test-key",
        big_model_name="gpt-4",
        small_model_name="gpt-3.5-turbo",
        cache_token_counts_enabled=True,
        TOKENIZER_CACHE_SHARDS=16,
    )

    # Create 100 unique message sets
    messages_list = []
    for i in range(100):
        messages_list.append(
            [
                Message(
                    role="user",
                    content=f"Unique test message {i} with different content",
                )
            ]
        )

    # Measure concurrent execution time
    start_time = time.time()

    # Run tasks concurrently using anyio task groups
    results: list[Any] = [None] * len(messages_list)
    async def run_task(idx: int, msgs: list[Message]) -> None:
        results[idx] = await count_tokens_for_anthropic_request(
            messages=msgs,
            system=None,
            model_name="gpt-4",
            tools=None,
            request_id=f"req_{idx}",
            settings=settings,
        )

    async with anyio.create_task_group() as tg:
        for i, msgs in enumerate(messages_list):
            tg.start_soon(run_task, i, msgs)

    concurrent_time = time.time() - start_time

    # Verify all tasks completed
    assert len(results) == 100
    assert all(r > 0 for r in results)

    # Second run should be faster due to cache hits
    start_time = time.time()

    # Run cached tasks concurrently using anyio task groups
    cached_results: list[Any] = [None] * len(messages_list)
    async def run_cached_task(idx: int, msgs: list[Message]) -> None:
        cached_results[idx] = await count_tokens_for_anthropic_request(
            messages=msgs,
            system=None,
            model_name="gpt-4",
            tools=None,
            request_id=f"req_cached_{idx}",
            settings=settings,
        )

    async with anyio.create_task_group() as tg:
        for i, msgs in enumerate(messages_list):
            tg.start_soon(run_cached_task, i, msgs)

    cached_time = time.time() - start_time

    # Cached run should be significantly faster
    assert cached_time < concurrent_time * 0.5  # At least 2x faster
    assert cached_results == results  # Same token counts


@pytest.mark.anyio
async def test_shard_lru_eviction(clean_cache_state: Any) -> None:
    """Verify LRU eviction works correctly within individual shards."""
    import ccproxy.application.tokenizer as tokenizer

    # Create settings with small cache size for easy testing
    settings = Settings(
        openai_api_key="test-key",
        big_model_name="gpt-4",
        small_model_name="gpt-3.5-turbo",
        cache_token_counts_enabled=True,
        cache_token_counts_max=32,  # 32 total entries
        TOKENIZER_CACHE_SHARDS=4,  # 4 shards = 8 entries per shard
    )

    # Fill one specific shard beyond capacity
    # Find messages that hash to shard 0
    shard_0_messages = []
    for i in range(100):
        test_msg = [Message(role="user", content=f"Message {i}")]
        # Use a deterministic way to check which shard this would go to
        import hashlib
        import json

        payload = {
            "model": "gpt-4",
            "messages": [m.model_dump(exclude_unset=True) for m in test_msg],
            "system": None,
            "tools": [],
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        if hash(key) % 4 == 0:  # This will go to shard 0
            shard_0_messages.append(test_msg)
            if len(shard_0_messages) >= 12:  # More than shard capacity
                break

    # Add messages to cache
    for i, msgs in enumerate(shard_0_messages):
        await count_tokens_for_anthropic_request(
            messages=msgs,
            system=None,
            model_name="gpt-4",
            tools=None,
            request_id=f"evict_test_{i}",
            settings=settings,
        )

    # Check that shard 0 has at most 8 entries (32 total / 4 shards)
    tokenizer._ensure_shards_initialized(4)
    shard_0 = tokenizer._token_cache_shards[0]
    assert len(shard_0.cache) <= 8, (
        f"Shard 0 has {len(shard_0.cache)} entries, expected <= 8"
    )

    # Verify other shards are not affected
    for i in range(1, 4):
        shard = tokenizer._token_cache_shards[i]
        # Other shards might have some entries but shouldn't be full
        assert len(shard.cache) <= 8


@pytest.mark.anyio
async def test_cache_stats_aggregation(clean_cache_state: Any) -> None:
    """Verify cache statistics are correctly aggregated across shards."""
    import ccproxy.application.tokenizer as tokenizer

    # Initially, stats should show uninitialized state
    stats = tokenizer.get_token_cache_stats()
    assert stats["initialized"] is False
    assert stats["total_entries"] == 0

    # Initialize and add some entries
    settings = Settings(
        openai_api_key="test-key",
        big_model_name="gpt-4",
        small_model_name="gpt-3.5-turbo",
        cache_token_counts_enabled=True,
        TOKENIZER_CACHE_SHARDS=4,
    )

    # Add 20 unique messages
    for i in range(20):
        await count_tokens_for_anthropic_request(
            messages=[Message(role="user", content=f"Message {i}")],
            system=None,
            model_name="gpt-4",
            tools=None,
            request_id=f"stats_test_{i}",
            settings=settings,
        )

    # Get aggregated stats
    stats = tokenizer.get_token_cache_stats()
    assert stats["initialized"] is True
    assert stats["total_entries"] == 20
    assert stats["num_shards"] == 4
    assert len(stats["shard_distribution"]) == 4
    assert sum(stats["shard_distribution"]) == 20
    assert stats["misses"] == 20  # All were cache misses

    # Access same messages again to generate hits
    for i in range(10):
        await count_tokens_for_anthropic_request(
            messages=[Message(role="user", content=f"Message {i}")],
            system=None,
            model_name="gpt-4",
            tools=None,
            request_id=f"stats_hit_test_{i}",
            settings=settings,
        )

    stats = tokenizer.get_token_cache_stats()
    assert stats["hits"] == 10
    assert stats["hit_rate"] > 0


@pytest.mark.anyio
async def test_backward_compatibility_single_shard(clean_cache_state: Any) -> None:
    """Verify setting shards=1 works like the old single-lock system."""
    settings = Settings(
        openai_api_key="test-key",
        big_model_name="gpt-4",
        small_model_name="gpt-3.5-turbo",
        cache_token_counts_enabled=True,
        TOKENIZER_CACHE_SHARDS=1,  # Single shard = old behavior
    )

    # Add multiple messages
    messages_list = [[Message(role="user", content=f"Message {i}")] for i in range(10)]

    # All should go to the single shard
    for i, msgs in enumerate(messages_list):
        await count_tokens_for_anthropic_request(
            messages=msgs,
            system=None,
            model_name="gpt-4",
            tools=None,
            request_id=f"single_shard_{i}",
            settings=settings,
        )

    import ccproxy.application.tokenizer as tokenizer

    assert tokenizer._num_shards == 1
    assert len(tokenizer._token_cache_shards) == 1
    assert len(tokenizer._token_cache_shards[0].cache) == 10


# === Additional Coverage Tests ===


@pytest.mark.anyio
async def test_get_token_encoder_unknown_model(clean_cache_state: Any) -> None:
    """Test get_token_encoder with unknown model falls back to cl100k_base."""
    from ccproxy.application.tokenizer import get_token_encoder

    encoder = get_token_encoder("unknown-model-12345", request_id="test")
    assert encoder is not None

    # Should be able to encode text
    tokens = encoder.encode("test")
    assert len(tokens) > 0


@pytest.mark.anyio
async def test_count_tokens_with_images(clean_cache_state: Any) -> None:
    """Test token counting with image content blocks."""
    from ccproxy.domain.models import ContentBlockImage

    messages = [
        Message(
            role="user",
            content=[
                ContentBlockText(type="text", text="What's in this image?"),
                ContentBlockImage(
                    type="image",
                    source={
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "abc123",
                    },
                ),
            ],
        )
    ]

    settings = Settings(
        openai_api_key="test-key",
        big_model_name="gpt-4",
        small_model_name="gpt-3.5-turbo",
        cache_token_counts_enabled=False,
    )

    tokens = await count_tokens_for_anthropic_request(
        messages, None, "gpt-4", None, "test", settings
    )

    # Should include text tokens + 768 for image
    assert tokens > 768


@pytest.mark.anyio
async def test_count_tokens_with_tool_use(clean_cache_state: Any) -> None:
    """Test token counting with tool use blocks."""
    from ccproxy.domain.models import ContentBlockToolUse

    messages = [
        Message(
            role="assistant",
            content=[
                ContentBlockToolUse(
                    type="tool_use",
                    id="tool_1",
                    name="search",
                    input={"query": "test query"},
                ),
            ],
        )
    ]

    settings = Settings(
        openai_api_key="test-key",
        big_model_name="gpt-4",
        small_model_name="gpt-3.5-turbo",
        cache_token_counts_enabled=False,
    )

    tokens = await count_tokens_for_anthropic_request(
        messages, None, "gpt-4", None, "test", settings
    )

    assert tokens > 0


@pytest.mark.anyio
async def test_count_tokens_with_tool_result(clean_cache_state: Any) -> None:
    """Test token counting with tool result blocks."""
    from ccproxy.domain.models import ContentBlockToolResult

    messages = [
        Message(
            role="user",
            content=[
                ContentBlockToolResult(  # type: ignore[call-arg]
                    # type="tool_result",
                    tool_use_id="tool_1",
                    content="Search results here",
                ),
            ],
        )
    ]

    settings = Settings(
        openai_api_key="test-key",
        big_model_name="gpt-4",
        small_model_name="gpt-3.5-turbo",
        cache_token_counts_enabled=False,
    )

    tokens = await count_tokens_for_anthropic_request(
        messages, None, "gpt-4", None, "test", settings
    )

    assert tokens > 0


@pytest.mark.anyio
async def test_count_tokens_with_thinking_block(clean_cache_state: Any) -> None:
    """Test token counting with thinking blocks."""
    from ccproxy.domain.models import ContentBlockThinking

    messages = [
        Message(
            role="assistant",
            content=[
                ContentBlockThinking(
                    type="thinking", thinking="Let me think about this..."
                ),
            ],
        )
    ]

    settings = Settings(
        openai_api_key="test-key",
        big_model_name="gpt-4",
        small_model_name="gpt-3.5-turbo",
        cache_token_counts_enabled=False,
    )

    tokens = await count_tokens_for_anthropic_request(
        messages, None, "gpt-4", None, "test", settings
    )

    assert tokens > 0


@pytest.mark.anyio
async def test_count_tokens_with_redacted_thinking(clean_cache_state: Any) -> None:
    """Test token counting with redacted thinking blocks."""
    from ccproxy.domain.models import ContentBlockRedactedThinking

    messages = [
        Message(
            role="assistant",
            content=[
                ContentBlockRedactedThinking(type="redacted_thinking", data="hidden"),
            ],
        )
    ]

    settings = Settings(
        openai_api_key="test-key",
        big_model_name="gpt-4",
        small_model_name="gpt-3.5-turbo",
        cache_token_counts_enabled=False,
    )

    tokens = await count_tokens_for_anthropic_request(
        messages, None, "gpt-4", None, "test", settings
    )

    # Should add fixed 100 tokens for redacted thinking
    assert tokens >= 100


@pytest.mark.anyio
async def test_count_tokens_with_tools(clean_cache_state: Any) -> None:
    """Test token counting with tool definitions."""
    from ccproxy.domain.models import Tool

    messages = [Message(role="user", content="Use the search tool")]

    tools = [
        Tool(
            name="search",
            description="Search for information",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        )
    ]

    settings = Settings(
        openai_api_key="test-key",
        big_model_name="gpt-4",
        small_model_name="gpt-3.5-turbo",
        cache_token_counts_enabled=False,
    )

    tokens = await count_tokens_for_anthropic_request(
        messages, None, "gpt-4", tools, "test", settings
    )

    # Should include tool definition tokens
    assert tokens > 0


@pytest.mark.anyio
async def test_count_tokens_with_system_content_list(clean_cache_state: Any) -> None:
    """Test token counting with system content as list."""
    from ccproxy.domain.models import SystemContent

    messages = [Message(role="user", content="Hello")]
    system = [
        SystemContent(type="text", text="You are a helpful assistant"),
        SystemContent(type="text", text="Answer concisely"),
    ]

    settings = Settings(
        openai_api_key="test-key",
        big_model_name="gpt-4",
        small_model_name="gpt-3.5-turbo",
        cache_token_counts_enabled=False,
    )

    tokens = await count_tokens_for_anthropic_request(
        messages, system, "gpt-4", None, "test", settings
    )

    assert tokens > 0


@pytest.mark.anyio
async def test_cache_ttl_expiration(clean_cache_state: Any) -> None:
    """Test that cache entries expire after TTL."""

    settings = Settings(
        openai_api_key="test-key",
        big_model_name="gpt-4",
        small_model_name="gpt-3.5-turbo",
        cache_token_counts_enabled=True,
        cache_token_counts_ttl_s=1,  # 1 second TTL
        TOKENIZER_CACHE_SHARDS=2,
    )

    messages = [Message(role="user", content="Test message")]

    # First call - cache miss
    tokens1 = await count_tokens_for_anthropic_request(
        messages, None, "gpt-4", None, "test1", settings
    )

    # Second call immediately - cache hit
    tokens2 = await count_tokens_for_anthropic_request(
        messages, None, "gpt-4", None, "test2", settings
    )

    assert tokens1 == tokens2

    # Wait for TTL to expire
    await anyio.sleep(1.5)

    # Third call after TTL - should be cache miss (entry expired)
    tokens3 = await count_tokens_for_anthropic_request(
        messages, None, "gpt-4", None, "test3", settings
    )

    assert tokens3 == tokens1


@pytest.mark.anyio
async def test_count_tokens_for_openai_request(clean_cache_state: Any) -> None:
    """Test count_tokens_for_openai_request function."""
    from ccproxy.application.tokenizer import count_tokens_for_openai_request

    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
    ]

    tokens = await count_tokens_for_openai_request(messages, "gpt-4", None, "test")  # type: ignore[arg-type]

    assert tokens > 0


@pytest.mark.anyio
async def test_count_tokens_for_openai_with_tools(clean_cache_state: Any) -> None:
    """Test OpenAI token counting with tools."""
    from ccproxy.application.tokenizer import count_tokens_for_openai_request

    messages = [{"role": "user", "content": "Search for something"}]

    tools = [
        {
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {}},
            }
        }
    ]

    tokens = await count_tokens_for_openai_request(messages, "gpt-4", tools, "test")  # type: ignore[arg-type]

    assert tokens > 0


@pytest.mark.anyio
async def test_count_tokens_for_openai_with_tool_calls(clean_cache_state: Any) -> None:
    """Test OpenAI token counting with tool calls in messages."""
    from ccproxy.application.tokenizer import count_tokens_for_openai_request

    messages = [
        {"role": "user", "content": "Search for something"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "test"}',
                    }
                }
            ],
        },
    ]

    tokens = await count_tokens_for_openai_request(messages, "gpt-4", None, "test")  # type: ignore[arg-type]

    assert tokens > 0


@pytest.mark.anyio
async def test_count_tokens_for_openai_with_multipart_content(
    clean_cache_state: Any,
) -> None:
    """Test OpenAI token counting with multipart content (text + image)."""
    from ccproxy.application.tokenizer import count_tokens_for_openai_request

    messages = [
        {
            "role": "user",
            "content": [
                # {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.png"},
                },
            ],
        }
    ]

    tokens = await count_tokens_for_openai_request(messages, "gpt-4", None, "test")  # type: ignore[arg-type]

    # Should include text tokens + 85 for image
    assert tokens > 85


@pytest.mark.anyio
async def test_get_token_cache_stats_uninitialized(clean_cache_state: Any) -> None:
    """Test get_token_cache_stats when cache is not initialized."""
    from ccproxy.application.tokenizer import get_token_cache_stats

    stats = get_token_cache_stats()

    assert stats["initialized"] is False
    assert stats["total_entries"] == 0
    assert stats["num_shards"] == 0


@pytest.mark.anyio
async def test_cache_with_string_content(clean_cache_state: Any) -> None:
    """Test token counting with messages that have string content."""
    messages = [Message(role="user", content="Simple string content")]

    settings = Settings(
        openai_api_key="test-key",
        big_model_name="gpt-4",
        small_model_name="gpt-3.5-turbo",
        cache_token_counts_enabled=False,
    )

    tokens = await count_tokens_for_anthropic_request(
        messages, None, "gpt-4", None, "test", settings
    )

    assert tokens > 0


@pytest.mark.anyio
async def test_tool_result_with_list_content(clean_cache_state: Any) -> None:
    """Test tool result with list content."""
    from ccproxy.domain.models import ContentBlockToolResult

    messages = [
        Message(
            role="user",
            content=[
                ContentBlockToolResult(
                    type="tool_result",
                    tool_use_id="tool_1",
                    content=[
                        {"type": "text", "text": "Result 1"},
                        {"type": "text", "text": "Result 2"},
                    ],
                ),
            ],
        )
    ]

    settings = Settings(
        openai_api_key="test-key",
        big_model_name="gpt-4",
        small_model_name="gpt-3.5-turbo",
        cache_token_counts_enabled=False,
    )

    tokens = await count_tokens_for_anthropic_request(
        messages, None, "gpt-4", None, "test", settings
    )

    assert tokens > 0
