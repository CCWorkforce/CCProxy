"""Benchmark for Cython-optimized streaming state and SSE event operations."""

import json
import time

CYTHON_AVAILABLE = False
cython_stream_state = None

try:
    from ccproxy._cython import stream_state as cython_stream_state

    if cython_stream_state is not None:
        CYTHON_AVAILABLE = True
        if not hasattr(cython_stream_state, "build_sse_event"):
            print(
                f"DEBUG: stream_state imported but missing functions: {dir(cython_stream_state)}"
            )
            CYTHON_AVAILABLE = False
except (ImportError, AttributeError) as e:
    print(f"DEBUG: Import failed with error: {e}")

# Test data
SIMPLE_EVENT_DATA = {"type": "ping", "timestamp": "2025-10-16T00:00:00Z"}
CONTENT_BLOCK_START_DATA = {
    "type": "content_block_start",
    "index": 0,
    "content": {"type": "text", "text": "Hello, world!"},
}
CONTENT_BLOCK_DELTA_DATA = {
    "type": "content_block_delta",
    "index": 0,
    "delta": {"type": "text_delta", "text": " How can I help you?"},
}
MESSAGE_START_DATA = {
    "type": "message_start",
    "message": {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet-20241022",
        "usage": {"input_tokens": 100, "output_tokens": 0},
    },
}


def build_sse_event_python(event_type: str, data_dict: dict) -> str:
    """Pure Python baseline for SSE event formatting."""
    if not event_type:
        event_type = "message"

    try:
        data_json = json.dumps(data_dict, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        data_json = "{}"

    return f"event: {event_type}\ndata: {data_json}\n\n"


def benchmark_build_sse_event(iterations=50000):
    """Benchmark SSE event building."""
    print("\n=== Build SSE Event ===")

    # Baseline: Pure Python
    start = time.time()
    for _ in range(iterations):
        build_sse_event_python("content_block_delta", CONTENT_BLOCK_DELTA_DATA)
    baseline_time = time.time() - start

    if CYTHON_AVAILABLE:
        # Cython optimized
        start = time.time()
        for _ in range(iterations):
            cython_stream_state.build_sse_event(
                "content_block_delta", CONTENT_BLOCK_DELTA_DATA
            )
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


def benchmark_format_content_block_start(iterations=25000):
    """Benchmark content_block_start formatting."""
    print("\n=== Format Content Block Start ===")

    # Baseline: Pure Python
    start = time.time()
    for _ in range(iterations):
        event_data = {
            "type": "content_block_start",
            "index": 0,
            "content": {"type": "text", "text": "Hello"},
        }
        build_sse_event_python("content_block_start", event_data)
    baseline_time = time.time() - start

    if CYTHON_AVAILABLE:
        # Cython optimized
        start = time.time()
        for _ in range(iterations):
            cython_stream_state.format_content_block_start(
                0, "text", {"type": "text", "text": "Hello"}
            )
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


def benchmark_format_content_block_delta(iterations=100000):
    """Benchmark content_block_delta formatting (hot path)."""
    print("\n=== Format Content Block Delta (Hot Path) ===")

    # Baseline: Pure Python
    start = time.time()
    for _ in range(iterations):
        event_data = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "word"},
        }
        build_sse_event_python("content_block_delta", event_data)
    baseline_time = time.time() - start

    if CYTHON_AVAILABLE:
        # Cython optimized
        start = time.time()
        for _ in range(iterations):
            cython_stream_state.format_content_block_delta(
                0, {"type": "text_delta", "text": "word"}
            )
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


def benchmark_format_message_start(iterations=10000):
    """Benchmark message_start formatting."""
    print("\n=== Format Message Start ===")

    # Baseline: Pure Python
    start = time.time()
    for _ in range(iterations):
        event_data = {
            "type": "message_start",
            "message": {
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet-20241022",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 100, "output_tokens": 0},
            },
        }
        build_sse_event_python("message_start", event_data)
    baseline_time = time.time() - start

    if CYTHON_AVAILABLE:
        # Cython optimized
        start = time.time()
        for _ in range(iterations):
            cython_stream_state.format_message_start(
                "msg_123", "claude-3-5-sonnet-20241022", 100
            )
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


def benchmark_batch_format_events(iterations=5000):
    """Benchmark batch event formatting."""
    print("\n=== Batch Format Events ===")

    # Test data
    event_specs = [
        ("message_start", MESSAGE_START_DATA),
        ("content_block_start", CONTENT_BLOCK_START_DATA),
        ("content_block_delta", CONTENT_BLOCK_DELTA_DATA),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_stop", {"type": "message_stop"}),
    ]

    # Baseline: Pure Python
    start = time.time()
    for _ in range(iterations):
        results = []
        for event_type, data in event_specs:
            results.append(build_sse_event_python(event_type, data))
    baseline_time = time.time() - start

    if CYTHON_AVAILABLE:
        # Cython optimized
        start = time.time()
        for _ in range(iterations):
            cython_stream_state.batch_format_events(event_specs)
        cython_time = time.time() - start

        improvement = (baseline_time - cython_time) / baseline_time * 100
        print(
            f"  Baseline:  {baseline_time:.4f}s ({iterations / baseline_time:.0f} batches/sec)"
        )
        print(
            f"  Cython:    {cython_time:.4f}s ({iterations / cython_time:.0f} batches/sec)"
        )
        print(f"  Speedup:   {baseline_time / cython_time:.2f}x ({improvement:+.1f}%)")
    else:
        print(
            f"  Baseline:  {baseline_time:.4f}s ({iterations / baseline_time:.0f} batches/sec)"
        )
        print("  Cython:    NOT AVAILABLE")


if __name__ == "__main__":
    print("=" * 60)
    print("Streaming State & SSE Event Performance Benchmark")
    print("=" * 60)

    if not CYTHON_AVAILABLE:
        print("\n⚠️  Cython modules not available - run 'uv pip install -e .' to build")

    benchmark_build_sse_event()
    benchmark_format_content_block_start()
    benchmark_format_content_block_delta()
    benchmark_format_message_start()
    benchmark_batch_format_events()

    print("\n" + "=" * 60)
    print("Benchmark completed!")
    print("=" * 60)
