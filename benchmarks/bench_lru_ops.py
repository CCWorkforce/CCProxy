"""Benchmarks for LRU cache operations."""

from collections import OrderedDict
from ccproxy.application.request_validator import RequestValidator


class TestLRUCacheOperations:
    """Benchmark LRU cache management operations."""

    def test_cache_hit(self, benchmark):
        """Benchmark cache hit performance."""
        validator = RequestValidator(cache_size=1000)

        # Pre-populate cache
        request_body = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        validator.validate_request(request_body)

        # Benchmark cache hit
        result = benchmark(validator.validate_request, request_body)
        assert result is not None

    def test_cache_miss_and_validation(self, benchmark):
        """Benchmark cache miss with Pydantic validation."""
        validator = RequestValidator(cache_size=1000)

        def validate_new_request():
            import time

            # Create unique request each time to ensure cache miss
            request_body = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": f"Message at {time.time_ns()}"}
                ],
            }
            return validator.validate_request(request_body)

        result = benchmark(validate_new_request)
        assert result is not None

    def test_lru_eviction(self, benchmark):
        """Benchmark LRU eviction when cache is full."""
        validator = RequestValidator(cache_size=10)

        # Pre-populate cache to capacity
        for i in range(10):
            request_body = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": f"Message {i}"}],
            }
            validator.validate_request(request_body)

        # Benchmark insertion that triggers eviction
        def insert_with_eviction():
            request_body = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "New message"}],
            }
            return validator.validate_request(request_body)

        result = benchmark(insert_with_eviction)
        assert result is not None

    def test_ordered_dict_operations(self, benchmark):
        """Benchmark raw OrderedDict operations (baseline)."""

        def ordered_dict_ops():
            cache = OrderedDict()

            # Populate
            for i in range(100):
                cache[f"key_{i}"] = f"value_{i}"

            # Access and move to end (LRU update)
            for i in range(0, 100, 10):
                key = f"key_{i}"
                if key in cache:
                    cache.move_to_end(key)

            # Evict oldest
            if len(cache) > 90:
                for _ in range(10):
                    cache.popitem(last=False)

            return len(cache)

        result = benchmark(ordered_dict_ops)
        assert result == 90


class TestCacheKeyGeneration:
    """Benchmark cache key generation (hashing)."""

    def test_hash_generation_small_request(self, benchmark):
        """Benchmark hash generation for small requests."""
        import json
        import hashlib

        request_body = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
        }

        def generate_hash():
            request_json = json.dumps(request_body, sort_keys=True)
            return hashlib.sha256(request_json.encode()).hexdigest()

        result = benchmark(generate_hash)
        assert len(result) == 64

    def test_hash_generation_large_request(self, benchmark):
        """Benchmark hash generation for large requests (with tools)."""
        import json
        import hashlib

        request_body = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 4096,
            "messages": [
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_123",
                            "name": "get_weather",
                            "input": {"location": "San Francisco", "unit": "celsius"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_123",
                            "content": "The weather is 18°C and sunny.",
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                }
            ],
        }

        def generate_hash():
            request_json = json.dumps(request_body, sort_keys=True)
            return hashlib.sha256(request_json.encode()).hexdigest()

        result = benchmark(generate_hash)
        assert len(result) == 64
