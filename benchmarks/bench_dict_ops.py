"""Benchmark for Cython-optimized dictionary operations."""

import time
from typing import Any, List, Dict


CYTHON_AVAILABLE = False
cython_dict_ops = None

try:
    from ccproxy._cython import dict_ops as cython_dict_ops

    if cython_dict_ops is not None:
        CYTHON_AVAILABLE = True  # type: ignore[unreachable]
        if not hasattr(cython_dict_ops, "recursive_redact"):
            print(
                f"DEBUG: dict_ops imported but missing functions: {dir(cython_dict_ops)}"
            )
            CYTHON_AVAILABLE = False
except (ImportError, AttributeError) as e:
    print(f"DEBUG: Import failed with error: {e}")

# Test data
SIMPLE_DICT = {"name": "alice", "password": "secret123", "email": "alice@example.com"}
NESTED_DICT = {
    "user": {
        "name": "bob",
        "credentials": {
            "api_key": "sk-1234567890abcdef",
            "password": "hunter2",
            "token": "abc123",
        },
        "profile": {"age": 30, "city": "New York"},
    },
    "settings": {"theme": "dark", "auth_token": "xyz789"},
}
COMPLEX_DICT_WITH_LISTS = {
    "users": [
        {"name": "user1", "password": "pass1", "data": {"api_key": "key1"}},
        {"name": "user2", "password": "pass2", "data": {"api_key": "key2"}},
        {"name": "user3", "password": "pass3", "data": {"api_key": "key3"}},
    ],
    "config": {"secret": "config_secret", "public": "public_value"},
}

SENSITIVE_KEYS = ["password", "api_key", "token", "secret", "auth", "credentials"]


def recursive_redact_python(
    data: dict[str, Any], sensitive_keys: List[Any]
) -> dict[str, Any]:
    """Pure Python baseline for recursive redaction."""
    result = {}
    sensitive_set = {k.lower() for k in sensitive_keys}

    for key, value in data.items():
        key_lower = key.lower()

        if key_lower in sensitive_set:
            result[key] = "[REDACTED]"
        elif isinstance(value, dict):
            result[key] = recursive_redact_python(value, sensitive_keys)  # type: ignore[assignment]
        elif isinstance(value, list):
            result[key] = [  # type: ignore[assignment]
                recursive_redact_python(item, sensitive_keys)
                if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            result[key] = value

    return result


def recursive_filter_none_python(data: Dict[str, Any]) -> Dict[str, Any]:
    """Pure Python baseline for filtering None values."""
    result = {}

    for key, value in data.items():
        if value is None:
            continue
        elif isinstance(value, dict):
            filtered = recursive_filter_none_python(value)
            if filtered:
                result[key] = filtered
        elif isinstance(value, list):
            filtered_list = [
                recursive_filter_none_python(item) if isinstance(item, dict) else item
                for item in value
                if item is not None
            ]
            if filtered_list:
                result[key] = filtered_list  # type: ignore[assignment]
        else:
            result[key] = value

    return result


def benchmark_recursive_redact(iterations=10000) -> None:  # type: ignore[no-untyped-def]
    """Benchmark recursive redaction on nested dictionaries."""
    print("\n=== Recursive Redact (Nested Dict[str, Any]) ===")

    # Baseline: Pure Python
    start = time.time()
    for _ in range(iterations):
        recursive_redact_python(NESTED_DICT, SENSITIVE_KEYS)
    baseline_time = time.time() - start

    if CYTHON_AVAILABLE:
        # Cython optimized
        start = time.time()
        for _ in range(iterations):
            if cython_dict_ops is not None:
                if cython_dict_ops is not None:  # type: ignore[unreachable]
                    if cython_dict_ops is not None:
                        if cython_dict_ops is not None:
                            cython_dict_ops.recursive_redact(
                                NESTED_DICT, SENSITIVE_KEYS
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


def benchmark_recursive_redact_with_lists(iterations=5000) -> None:  # type: ignore[no-untyped-def]
    """Benchmark recursive redaction with nested lists."""
    print("\n=== Recursive Redact (With Lists) ===")

    # Baseline: Pure Python
    start = time.time()
    for _ in range(iterations):
        recursive_redact_python(COMPLEX_DICT_WITH_LISTS, SENSITIVE_KEYS)
    baseline_time = time.time() - start

    if CYTHON_AVAILABLE:
        # Cython optimized
        start = time.time()
        for _ in range(iterations):
            if cython_dict_ops is not None:
                if cython_dict_ops is not None:  # type: ignore[unreachable]
                    if cython_dict_ops is not None:
                        if cython_dict_ops is not None:
                            cython_dict_ops.recursive_redact(
                                COMPLEX_DICT_WITH_LISTS, SENSITIVE_KEYS
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


def benchmark_recursive_filter_none(iterations=15000) -> None:  # type: ignore[no-untyped-def]
    """Benchmark filtering None values from nested dictionaries."""
    print("\n=== Recursive Filter None ===")

    # Test data with None values
    test_data = {
        "a": 1,
        "b": None,
        "c": {"d": 2, "e": None, "f": {"g": 3, "h": None}},
        "i": [1, None, {"j": 4, "k": None}],
        "l": None,
    }

    # Baseline: Pure Python
    start = time.time()
    for _ in range(iterations):
        recursive_filter_none_python(test_data)
    baseline_time = time.time() - start

    if CYTHON_AVAILABLE:
        # Cython optimized
        start = time.time()
        for _ in range(iterations):
            if cython_dict_ops is not None:
                if cython_dict_ops is not None:  # type: ignore[unreachable]
                    if cython_dict_ops is not None:
                        if cython_dict_ops is not None:
                            cython_dict_ops.recursive_filter_none(test_data)
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


def benchmark_deep_merge_dicts(iterations=20000) -> Any:  # type: ignore[no-untyped-def]
    """Benchmark deep dictionary merging."""
    print("\n=== Deep Merge Dicts ===")

    base = {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}
    update = {"b": {"d": 30, "f": 40}, "g": 5}

    # Baseline: Pure Python
    def deep_merge_python(base_dict, update_dict) -> Any:  # type: ignore[no-untyped-def]
        result = base_dict.copy()
        for key, value in update_dict.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = deep_merge_python(result[key], value)
            else:
                result[key] = value
        return result

    start = time.time()
    for _ in range(iterations):
        deep_merge_python(base, update)
    baseline_time = time.time() - start

    if CYTHON_AVAILABLE:
        # Cython optimized
        start = time.time()
        for _ in range(iterations):
            if cython_dict_ops is not None:
                if cython_dict_ops is not None:  # type: ignore[unreachable]
                    if cython_dict_ops is not None:
                        if cython_dict_ops is not None:
                            cython_dict_ops.deep_merge_dicts(base, update)
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


def benchmark_count_nested_keys(iterations=15000) -> Any:  # type: ignore[no-untyped-def]
    """Benchmark counting nested dictionary keys."""
    print("\n=== Count Nested Keys ===")

    # Baseline: Pure Python
    def count_nested_keys_python(data) -> Any:  # type: ignore[no-untyped-def]
        if not isinstance(data, dict):
            return 0
        count = len(data)
        for value in data.values():
            if isinstance(value, dict):
                count += count_nested_keys_python(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        count += count_nested_keys_python(item)
        return count

    start = time.time()
    for _ in range(iterations):
        count_nested_keys_python(NESTED_DICT)
    baseline_time = time.time() - start

    if CYTHON_AVAILABLE:
        # Cython optimized
        start = time.time()
        for _ in range(iterations):
            if cython_dict_ops is not None:
                if cython_dict_ops is not None:  # type: ignore[unreachable]
                    if cython_dict_ops is not None:
                        if cython_dict_ops is not None:
                            cython_dict_ops.count_nested_keys(NESTED_DICT)
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


def benchmark_sanitize_for_logging(iterations=5000) -> Any:  # type: ignore[no-untyped-def]
    """Benchmark comprehensive sanitization for logging."""
    print("\n=== Sanitize For Logging (Combined Operations) ===")

    # Baseline: Pure Python (combining operations)
    def sanitize_python(data, sensitive_keys, max_length=1000) -> Any:  # type: ignore[no-untyped-def]
        result = {}
        sensitive_set = {k.lower() for k in sensitive_keys}

        for key, value in data.items():
            if value is None:
                continue

            key_lower = key.lower()

            if key_lower in sensitive_set:
                result[key] = "[REDACTED]"
            elif isinstance(value, str):
                if len(value) > max_length:
                    result[key] = value[:max_length] + "...[truncated]"
                else:
                    result[key] = value
            elif isinstance(value, dict):
                sanitized = sanitize_python(value, sensitive_keys, max_length)
                if sanitized:
                    result[key] = sanitized
            else:
                result[key] = value

        return result

    start = time.time()
    for _ in range(iterations):
        sanitize_python(NESTED_DICT, SENSITIVE_KEYS, 1000)
    baseline_time = time.time() - start

    if CYTHON_AVAILABLE:
        # Cython optimized
        start = time.time()
        for _ in range(iterations):
            if cython_dict_ops is not None:
                if cython_dict_ops is not None:  # type: ignore[unreachable]
                    if cython_dict_ops is not None:
                        if cython_dict_ops is not None:
                            cython_dict_ops.sanitize_for_logging(
                                NESTED_DICT, SENSITIVE_KEYS, 1000
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


if __name__ == "__main__":
    print("=" * 60)
    print("Dictionary Operations Performance Benchmark")
    print("=" * 60)

    if not CYTHON_AVAILABLE:
        print("\n⚠️  Cython modules not available - run 'uv pip install -e .' to build")

    benchmark_recursive_redact()
    benchmark_recursive_redact_with_lists()
    benchmark_recursive_filter_none()
    benchmark_deep_merge_dicts()
    benchmark_count_nested_keys()
    benchmark_sanitize_for_logging()

    print("\n" + "=" * 60)
    print("Benchmark completed!")
    print("=" * 60)
