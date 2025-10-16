"""Benchmark for Cython-optimized JSON operations."""

import json
import time

CYTHON_AVAILABLE = False
cython_json_ops = None

try:
    from ccproxy._cython import json_ops as cython_json_ops

    if cython_json_ops is not None:
        CYTHON_AVAILABLE = True
        # Verify the module has the expected functions
        if not hasattr(cython_json_ops, "json_dumps_compact"):
            print(
                f"DEBUG: json_ops imported but missing functions: {dir(cython_json_ops)}"
            )
            CYTHON_AVAILABLE = False
except (ImportError, AttributeError) as e:
    print(f"DEBUG: Import failed with error: {e}")

# Test data
SMALL_DICT = {"message": "Hello", "status": "ok", "count": 42}
LARGE_DICT = {
    "messages": [{"role": "user", "content": "Hello" * 100} for _ in range(10)],
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 1000,
    "tools": [{"name": f"tool_{i}", "description": "Test tool" * 50} for i in range(5)],
}


def benchmark_json_dumps_compact(iterations=10000):
    """Benchmark compact JSON serialization."""
    print("\n=== JSON Dumps Compact ===")

    # Baseline: Pure Python
    start = time.time()
    for _ in range(iterations):
        json.dumps(SMALL_DICT, ensure_ascii=False, separators=(",", ":"))
    baseline_time = time.time() - start

    if CYTHON_AVAILABLE:
        # Cython optimized
        start = time.time()
        for _ in range(iterations):
            cython_json_ops.json_dumps_compact(SMALL_DICT)
        cython_time = time.time() - start

        improvement = (baseline_time - cython_time) / baseline_time * 100
        print(
            f"  Baseline:  {baseline_time:.4f}s ({iterations / baseline_time:.0f} ops/sec)"
        )
        print(
            f"  Cython:    {cython_time:.4f}s ({iterations / cython_time:.0f} ops/sec)"
        )
        print(f"  Speedup:   {baseline_time / cython_time:.2f}x ({improvement:+.1f}%)")
    else:
        print(
            f"  Baseline:  {baseline_time:.4f}s ({iterations / baseline_time:.0f} ops/sec)"
        )
        print("  Cython:    NOT AVAILABLE")


def benchmark_json_dumps_sorted(iterations=5000):
    """Benchmark sorted JSON serialization for cache keys."""
    print("\n=== JSON Dumps Sorted (Cache Keys) ===")

    # Baseline: Pure Python
    start = time.time()
    for _ in range(iterations):
        json.dumps(
            LARGE_DICT, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
    baseline_time = time.time() - start

    if CYTHON_AVAILABLE:
        # Cython optimized
        start = time.time()
        for _ in range(iterations):
            cython_json_ops.json_dumps_sorted(LARGE_DICT)
        cython_time = time.time() - start

        improvement = (baseline_time - cython_time) / baseline_time * 100
        print(
            f"  Baseline:  {baseline_time:.4f}s ({iterations / baseline_time:.0f} ops/sec)"
        )
        print(
            f"  Cython:    {cython_time:.4f}s ({iterations / cython_time:.0f} ops/sec)"
        )
        print(f"  Speedup:   {baseline_time / cython_time:.2f}x ({improvement:+.1f}%)")
    else:
        print(
            f"  Baseline:  {baseline_time:.4f}s ({iterations / baseline_time:.0f} ops/sec)"
        )
        print("  Cython:    NOT AVAILABLE")


def benchmark_estimate_json_size(iterations=50000):
    """Benchmark fast JSON size estimation."""
    print("\n=== JSON Size Estimation ===")

    # Baseline: Full serialization then length
    start = time.time()
    for _ in range(iterations):
        len(json.dumps(LARGE_DICT))
    baseline_time = time.time() - start

    if CYTHON_AVAILABLE:
        # Cython fast estimation
        start = time.time()
        for _ in range(iterations):
            cython_json_ops.estimate_json_size(LARGE_DICT)
        cython_time = time.time() - start

        improvement = (baseline_time - cython_time) / baseline_time * 100
        print(
            f"  Baseline (full serialization):  {baseline_time:.4f}s ({iterations / baseline_time:.0f} ops/sec)"
        )
        print(
            f"  Cython (estimation):            {cython_time:.4f}s ({iterations / cython_time:.0f} ops/sec)"
        )
        print(f"  Speedup:   {baseline_time / cython_time:.2f}x ({improvement:+.1f}%)")
    else:
        print(
            f"  Baseline:  {baseline_time:.4f}s ({iterations / baseline_time:.0f} ops/sec)"
        )
        print("  Cython:    NOT AVAILABLE")


if __name__ == "__main__":
    print("=" * 60)
    print("JSON Operations Performance Benchmark")
    print("=" * 60)

    if not CYTHON_AVAILABLE:
        print("\n⚠️  Cython modules not available - run 'uv pip install -e .' to build")

    benchmark_json_dumps_compact()
    benchmark_json_dumps_sorted()
    benchmark_estimate_json_size()

    print("\n" + "=" * 60)
    print("Benchmark completed!")
    print("=" * 60)
