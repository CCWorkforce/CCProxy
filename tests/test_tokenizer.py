import subprocess
import sys
import asyncio
import time

import pytest
from ccproxy.application.tokenizer import (
    truncate_request,
    count_tokens_for_anthropic_request,
)
from ccproxy.domain.models import Message, ContentBlockText
from ccproxy.config import TruncationConfig, Settings


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
async def test_truncate_oldest_first():
    messages = create_test_messages(5)
    system = "System prompt"
    config = TruncationConfig()

    truncated_msgs, truncated_system = await truncate_request(
        messages, system, "gpt-4", 150, config, request_id="test"
    )
    new_tokens = await count_tokens_for_anthropic_request(
        truncated_msgs, truncated_system, "gpt-4", None, None
    )

    assert new_tokens <= 150
    # Check that some messages were removed
    assert len(truncated_msgs) < len(messages)


@pytest.mark.anyio
async def test_truncate_newest_first():
    messages = create_test_messages(5)
    system = "System prompt"

    # Create config with newest_first strategy
    config = TruncationConfig()
    config.strategy = "newest_first"

    truncated_msgs, truncated_system = await truncate_request(
        messages, system, "gpt-4", 150, config, request_id="test"
    )
    new_tokens = await count_tokens_for_anthropic_request(
        truncated_msgs, truncated_system, "gpt-4", None, None
    )

    assert new_tokens <= 150
    # Check that some messages were removed
    assert len(truncated_msgs) < len(messages)


@pytest.mark.anyio
async def test_truncate_system_priority():
    messages = create_test_messages(5)
    system = "System prompt"

    # Create config with system_priority strategy
    config = TruncationConfig()
    config.strategy = "system_priority"

    truncated_msgs, truncated_system = await truncate_request(
        messages, system, "gpt-4", 150, config, request_id="test"
    )
    new_tokens = await count_tokens_for_anthropic_request(
        truncated_msgs, truncated_system, "gpt-4", None, None
    )

    assert new_tokens <= 150
    assert truncated_system == "System prompt"


@pytest.mark.anyio
async def test_truncate_below_limit():
    messages = create_test_messages(2)
    system = "System prompt"
    config = TruncationConfig()

    truncated_msgs, truncated_system = await truncate_request(
        messages, system, "gpt-4", 10000, config
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
async def clean_cache_state():
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
async def test_sharded_cache_initialization(clean_cache_state):
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
async def test_shard_distribution(clean_cache_state):
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
async def test_concurrent_shard_access(clean_cache_state):
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
    tasks = [
        count_tokens_for_anthropic_request(
            messages=msgs,
            system=None,
            model_name="gpt-4",
            tools=None,
            request_id=f"req_{i}",
            settings=settings,
        )
        for i, msgs in enumerate(messages_list)
    ]
    results = await asyncio.gather(*tasks)
    concurrent_time = time.time() - start_time

    # Verify all tasks completed
    assert len(results) == 100
    assert all(r > 0 for r in results)

    # Second run should be faster due to cache hits
    start_time = time.time()
    tasks = [
        count_tokens_for_anthropic_request(
            messages=msgs,
            system=None,
            model_name="gpt-4",
            tools=None,
            request_id=f"req_cached_{i}",
            settings=settings,
        )
        for i, msgs in enumerate(messages_list)
    ]
    cached_results = await asyncio.gather(*tasks)
    cached_time = time.time() - start_time

    # Cached run should be significantly faster
    assert cached_time < concurrent_time * 0.5  # At least 2x faster
    assert cached_results == results  # Same token counts


@pytest.mark.anyio
async def test_shard_lru_eviction(clean_cache_state):
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
async def test_cache_stats_aggregation(clean_cache_state):
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
async def test_backward_compatibility_single_shard(clean_cache_state):
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
