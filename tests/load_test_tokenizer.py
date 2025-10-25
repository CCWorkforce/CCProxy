"""Load test for tokenizer sharded cache to measure lock contention improvements."""

import time
import statistics
from typing import Tuple, Any
import anyio

from ccproxy.application.tokenizer import (
    count_tokens_for_anthropic_request,
    get_token_cache_stats,
)
from ccproxy.domain.models import Message
from ccproxy.config import Settings


async def benchmark_cache_contention(
    num_requests: int = 100, num_shards: int = 16, cache_enabled: bool = True
) -> Tuple[float, float, dict]:  # type: ignore[type-arg]
    """Measure cache performance under high concurrency.

    Args:
        num_requests: Number of concurrent requests to test
        num_shards: Number of cache shards to use
        cache_enabled: Whether to enable caching

    Returns:
        Tuple of (first_run_time, cached_run_time, statistics)
    """
    # Create settings with specified configuration
    settings = Settings(
        openai_api_key="test-key",
        big_model_name="gpt-4",
        small_model_name="gpt-3.5-turbo",
        cache_token_counts_enabled=cache_enabled,
        TOKENIZER_CACHE_SHARDS=num_shards,
        cache_token_counts_max=2048,
    )

    # Create unique message sets to avoid immediate cache hits
    message_sets = []
    for i in range(num_requests):
        # Vary the content to ensure different cache keys
        long_text = f"This is test message {i} with unique content. " * 10
        message_sets.append(
            [Message(role="user", content=f"{long_text} Message ID: {i}")]
        )

    # First run - populate cache
    print(
        f"\nRunning {num_requests} concurrent requests (shards={num_shards}, cache={cache_enabled})..."
    )
    start = time.time()

    # Run tasks concurrently using anyio task groups
    results: list[Any] = [None] * len(message_sets)
    async def run_task(idx: int, msgs: list) -> None:  # type: ignore[type-arg]
        results[idx] = await count_tokens_for_anthropic_request(
            messages=msgs,
            system=None,
            model_name="gpt-4",
            tools=None,
            request_id=f"req_{idx}",
            settings=settings,
        )

    async with anyio.create_task_group() as tg:
        for i, msgs in enumerate(message_sets):
            tg.start_soon(run_task, i, msgs)

    first_run_time = time.time() - start

    print(f"First run completed in {first_run_time:.3f}s")
    print(f"Throughput: {len(results) / first_run_time:.0f} ops/sec")

    # Second run - should hit cache if enabled
    print("\nRunning cached requests...")
    start = time.time()

    # Run cached tasks concurrently using anyio task groups
    cached_results: list[Any] = [None] * len(message_sets)
    async def run_cached_task(idx: int, msgs: list) -> None:  # type: ignore[type-arg]
        cached_results[idx] = await count_tokens_for_anthropic_request(
            messages=msgs,
            system=None,
            model_name="gpt-4",
            tools=None,
            request_id=f"req_cached_{idx}",
            settings=settings,
        )

    async with anyio.create_task_group() as tg:
        for i, msgs in enumerate(message_sets):
            tg.start_soon(run_cached_task, i, msgs)

    cached_run_time = time.time() - start

    print(f"Cached run completed in {cached_run_time:.3f}s")
    print(f"Cache hit throughput: {len(cached_results) / cached_run_time:.0f} ops/sec")

    # Get final cache statistics
    stats_final = get_token_cache_stats()

    return first_run_time, cached_run_time, stats_final


async def compare_shard_configurations() -> None:
    """Compare performance across different shard configurations."""
    print("=" * 60)
    print("Token Cache Sharding Performance Test")
    print("=" * 60)

    configurations = [
        (1, "Single Shard (Original)"),
        (4, "4 Shards"),
        (8, "8 Shards"),
        (16, "16 Shards (Default)"),
        (32, "32 Shards"),
    ]

    results = []

    for num_shards, description in configurations:
        print(f"\nTesting: {description}")
        print("-" * 40)

        # Reset cache state between tests
        import ccproxy.application.tokenizer as tokenizer

        tokenizer._token_cache_shards = []
        tokenizer._shards_initialized = False
        tokenizer._token_count_hits = 0
        tokenizer._token_count_misses = 0

        first_time, cached_time, stats = await benchmark_cache_contention(
            num_requests=100, num_shards=num_shards, cache_enabled=True
        )

        speedup = first_time / cached_time if cached_time > 0 else 0
        results.append(
            {
                "shards": num_shards,
                "description": description,
                "first_run": first_time,
                "cached_run": cached_time,
                "speedup": speedup,
                "hit_rate": stats.get("hit_rate", 0),
                "distribution": stats.get("shard_distribution", []),
            }
        )

        print("\nCache Statistics:")
        print(f"  Hit rate: {stats.get('hit_rate', 0):.1f}%")
        print(f"  Total entries: {stats.get('total_entries', 0)}")
        if stats.get("shard_distribution"):
            distribution = stats["shard_distribution"]
            print(
                f"  Shard distribution: min={min(distribution)}, max={max(distribution)}, "
                f"avg={statistics.mean(distribution):.1f}"
            )

    # Print summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(
        f"{'Shards':<10} {'First Run':<12} {'Cached Run':<12} {'Speedup':<10} {'Hit Rate':<10}"
    )
    print("-" * 60)

    for result in results:
        print(
            f"{result['shards']:<10} "
            f"{result['first_run']:<12.3f} "
            f"{result['cached_run']:<12.3f} "
            f"{result['speedup']:<10.1f}x "
            f"{result['hit_rate']:<10.1f}%"
        )

    # Find optimal configuration
    best_cached = min(results, key=lambda x: x["cached_run"])
    print(
        f"\nBest cached performance: {best_cached['description']} "
        f"({best_cached['cached_run']:.3f}s)"
    )

    # Calculate improvement over single shard
    single_shard = next((r for r in results if r["shards"] == 1), None)
    if single_shard and best_cached["shards"] != 1:
        improvement = (
            (single_shard["cached_run"] - best_cached["cached_run"])
            / single_shard["cached_run"]
            * 100
        )
        print(f"Improvement over single shard: {improvement:.1f}%")


async def stress_test_high_concurrency() -> None:
    """Test with very high concurrency to stress the sharding system."""
    print("\n" + "=" * 60)
    print("HIGH CONCURRENCY STRESS TEST")
    print("=" * 60)

    concurrency_levels = [50, 100, 200, 500]

    for level in concurrency_levels:
        print(f"\nTesting with {level} concurrent requests...")

        # Reset cache
        import ccproxy.application.tokenizer as tokenizer

        tokenizer._token_cache_shards = []
        tokenizer._shards_initialized = False
        tokenizer._token_count_hits = 0
        tokenizer._token_count_misses = 0

        try:
            first_time, cached_time, stats = await benchmark_cache_contention(
                num_requests=level, num_shards=16, cache_enabled=True
            )

            print(f"  Results: first={first_time:.3f}s, cached={cached_time:.3f}s")
            print(f"  Throughput: {level / cached_time:.0f} ops/sec (cached)")

        except Exception as e:
            print(f"  Failed with error: {e}")


async def main() -> None:
    """Run all performance tests."""
    # Compare different shard configurations
    await compare_shard_configurations()

    # Run high concurrency stress test
    await stress_test_high_concurrency()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    anyio.run(main)
