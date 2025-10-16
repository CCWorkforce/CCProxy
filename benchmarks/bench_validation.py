"""Benchmark for Cython-optimized validation operations."""

import json
import time

CYTHON_AVAILABLE = False
cython_validation = None

try:
    from ccproxy._cython import validation as cython_validation

    if cython_validation is not None:
        CYTHON_AVAILABLE = True
        if not hasattr(cython_validation, "validate_content_blocks"):
            print(
                f"DEBUG: validation imported but missing functions: {dir(cython_validation)}"
            )
            CYTHON_AVAILABLE = False
except (ImportError, AttributeError) as e:
    print(f"DEBUG: Import failed with error: {e}")

# Test data
VALID_TEXT_BLOCKS = [
    {"type": "text", "text": "Hello, world!"},
    {"type": "text", "text": "How can I help you today?"},
]

VALID_MIXED_BLOCKS = [
    {"type": "text", "text": "Here's an image:"},
    {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0K..."},
    },
    {"type": "text", "text": "And a tool use:"},
    {
        "type": "tool_use",
        "id": "tool_123",
        "name": "calculator",
        "input": {"expression": "2+2"},
    },
]

INVALID_BLOCKS = [
    {"type": "text"},  # Missing 'text' field
    {"text": "Hello"},  # Missing 'type' field
]

VALID_MESSAGE = {"role": "user", "content": "Hello, assistant!"}

VALID_MESSAGE_WITH_BLOCKS = {"role": "assistant", "content": VALID_TEXT_BLOCKS}

VALID_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather in a location",
    "input_schema": {
        "type": "object",
        "properties": {"location": {"type": "string", "description": "City name"}},
        "required": ["location"],
    },
}

COMPLEX_DATA = {
    "messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
    ],
    "tools": [VALID_TOOL],
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
}


def validate_content_blocks_python(blocks):
    """Pure Python baseline for content block validation."""
    if not blocks or not isinstance(blocks, list):
        return (False, "Content blocks must be a non-empty list")

    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            return (False, f"Block at index {index} is not a dictionary")

        if "type" not in block:
            return (False, f"Block at index {index} missing required 'type' field")

        block_type = block.get("type")

        if block_type == "text":
            if "text" not in block or not isinstance(block["text"], str):
                return (False, f"Text block at index {index} invalid")
        elif block_type == "image":
            if "source" not in block or not isinstance(block["source"], dict):
                return (False, f"Image block at index {index} invalid")
        elif block_type == "tool_use":
            if "id" not in block or "name" not in block or "input" not in block:
                return (
                    False,
                    f"Tool use block at index {index} missing required fields",
                )

    return (True, "")


def check_json_serializable_python(obj):
    """Pure Python baseline for JSON serializability check."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return True

    if isinstance(obj, (list, tuple)):
        return all(check_json_serializable_python(item) for item in obj)

    if isinstance(obj, dict):
        return all(
            isinstance(k, str) and check_json_serializable_python(v)
            for k, v in obj.items()
        )

    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError, OverflowError):
        return False


def benchmark_validate_content_blocks(iterations=25000):
    """Benchmark content block validation."""
    print("\n=== Validate Content Blocks ===")

    # Baseline: Pure Python
    start = time.time()
    for _ in range(iterations):
        validate_content_blocks_python(VALID_MIXED_BLOCKS)
    baseline_time = time.time() - start

    if CYTHON_AVAILABLE:
        # Cython optimized
        start = time.time()
        for _ in range(iterations):
            cython_validation.validate_content_blocks(VALID_MIXED_BLOCKS)
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


def benchmark_check_json_serializable(iterations=50000):
    """Benchmark JSON serializability check."""
    print("\n=== Check JSON Serializable ===")

    # Baseline: Pure Python
    start = time.time()
    for _ in range(iterations):
        check_json_serializable_python(COMPLEX_DATA)
    baseline_time = time.time() - start

    if CYTHON_AVAILABLE:
        # Cython optimized
        start = time.time()
        for _ in range(iterations):
            cython_validation.check_json_serializable(COMPLEX_DATA)
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


def benchmark_validate_message_structure(iterations=30000):
    """Benchmark message structure validation."""
    print("\n=== Validate Message Structure ===")

    # Baseline: Pure Python
    def validate_message_python(message):
        if not message or not isinstance(message, dict):
            return (False, "Message must be a non-empty dictionary")

        if "role" not in message:
            return (False, "Message missing 'role' field")

        if message["role"] not in ("user", "assistant"):
            return (False, f"Invalid role: {message['role']}")

        if "content" not in message:
            return (False, "Message missing 'content' field")

        content = message["content"]
        if isinstance(content, str):
            if not content:
                return (False, "Content cannot be empty string")
        elif isinstance(content, list):
            is_valid, error = validate_content_blocks_python(content)
            if not is_valid:
                return (False, f"Invalid content blocks: {error}")
        else:
            return (False, "Content must be string or list")

        return (True, "")

    start = time.time()
    for _ in range(iterations):
        validate_message_python(VALID_MESSAGE_WITH_BLOCKS)
    baseline_time = time.time() - start

    if CYTHON_AVAILABLE:
        # Cython optimized
        start = time.time()
        for _ in range(iterations):
            cython_validation.validate_message_structure(VALID_MESSAGE_WITH_BLOCKS)
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


def benchmark_validate_tool_structure(iterations=40000):
    """Benchmark tool structure validation."""
    print("\n=== Validate Tool Structure ===")

    # Baseline: Pure Python
    def validate_tool_python(tool):
        if not tool or not isinstance(tool, dict):
            return (False, "Tool must be a non-empty dictionary")

        if "name" not in tool:
            return (False, "Tool missing 'name' field")

        if not isinstance(tool["name"], str) or not tool["name"]:
            return (False, "Tool name must be non-empty string")

        if "input_schema" not in tool:
            return (False, "Tool missing 'input_schema' field")

        schema = tool["input_schema"]
        if not isinstance(schema, dict):
            return (False, "input_schema must be a dictionary")

        if schema.get("type") != "object":
            return (False, "input_schema type must be 'object'")

        return (True, "")

    start = time.time()
    for _ in range(iterations):
        validate_tool_python(VALID_TOOL)
    baseline_time = time.time() - start

    if CYTHON_AVAILABLE:
        # Cython optimized
        start = time.time()
        for _ in range(iterations):
            cython_validation.validate_tool_structure(VALID_TOOL)
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


def benchmark_estimate_object_complexity(iterations=20000):
    """Benchmark object complexity estimation."""
    print("\n=== Estimate Object Complexity ===")

    # Baseline: Pure Python
    def estimate_complexity_python(obj):
        complexity = 1

        if obj is None or isinstance(obj, (bool, int, float, str)):
            return 1

        if isinstance(obj, dict):
            complexity += len(obj)
            for value in obj.values():
                complexity += estimate_complexity_python(value)

        elif isinstance(obj, (list, tuple)):
            complexity += len(obj)
            for item in obj:
                complexity += estimate_complexity_python(item)

        return complexity

    start = time.time()
    for _ in range(iterations):
        estimate_complexity_python(COMPLEX_DATA)
    baseline_time = time.time() - start

    if CYTHON_AVAILABLE:
        # Cython optimized
        start = time.time()
        for _ in range(iterations):
            cython_validation.estimate_object_complexity(COMPLEX_DATA)
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


def benchmark_is_safe_for_cache(iterations=15000):
    """Benchmark comprehensive cache safety check."""
    print("\n=== Is Safe For Cache (Combined Check) ===")

    # Baseline: Pure Python (combining checks)
    def is_safe_for_cache_python(data, max_complexity=10000):
        if not data:
            return False

        # Check JSON serializability
        if not check_json_serializable_python(data):
            return False

        # Check complexity
        complexity = estimate_complexity_python(data)
        if complexity > max_complexity:
            return False

        # Check for sensitive fields
        sensitive_keys = ["api_key", "token", "password", "secret", "auth"]
        for key in data.keys():
            key_lower = key.lower()
            for sensitive_key in sensitive_keys:
                if sensitive_key in key_lower:
                    return False

        return True

    def estimate_complexity_python(obj):
        complexity = 1
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return 1
        if isinstance(obj, dict):
            complexity += len(obj)
            for value in obj.values():
                complexity += estimate_complexity_python(value)
        elif isinstance(obj, (list, tuple)):
            complexity += len(obj)
            for item in obj:
                complexity += estimate_complexity_python(item)
        return complexity

    start = time.time()
    for _ in range(iterations):
        is_safe_for_cache_python(COMPLEX_DATA)
    baseline_time = time.time() - start

    if CYTHON_AVAILABLE:
        # Cython optimized
        start = time.time()
        for _ in range(iterations):
            cython_validation.is_safe_for_cache(COMPLEX_DATA)
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


if __name__ == "__main__":
    print("=" * 60)
    print("Validation Operations Performance Benchmark")
    print("=" * 60)

    if not CYTHON_AVAILABLE:
        print("\n⚠️  Cython modules not available - run 'uv pip install -e .' to build")

    benchmark_validate_content_blocks()
    benchmark_check_json_serializable()
    benchmark_validate_message_structure()
    benchmark_validate_tool_structure()
    benchmark_estimate_object_complexity()
    benchmark_is_safe_for_cache()

    print("\n" + "=" * 60)
    print("Benchmark completed!")
    print("=" * 60)
