"""Tests for request validator with LRU cache."""

import pytest
from typing import Any
from ccproxy.application.request_validator import RequestValidator, request_validator


class TestRequestValidator:
    """Test RequestValidator class."""

    def test_init_default_cache_size(self) -> None:
        """Test validator initialization with default cache size."""
        validator = RequestValidator()
        assert validator._cache_size == 10000
        assert len(validator._validation_cache) == 0
        assert validator._cache_hits == 0
        assert validator._cache_misses == 0

    def test_init_custom_cache_size(self) -> None:
        """Test validator initialization with custom cache size."""
        validator = RequestValidator(cache_size=100)
        assert validator._cache_size == 100

    def test_validate_request_simple(self) -> None:
        """Test validating a simple request."""
        validator = RequestValidator()
        request_data = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 1024,
        }

        result = validator.validate_request(request_data, "test-req-1")

        assert result.model == "claude-3-opus-20240229"
        assert len(result.messages) == 1
        assert result.messages[0].role == "user"
        assert result.max_tokens == 1024

    def test_cache_hit_on_duplicate_request(self) -> None:
        """Test that duplicate requests hit cache."""
        validator = RequestValidator()
        request_data = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 1024,
        }

        # First request - cache miss
        result1 = validator.validate_request(request_data, "test-req-1")
        stats1 = validator.get_cache_stats()
        assert stats1["cache_misses"] == 1
        assert stats1["cache_hits"] == 0

        # Second identical request - cache hit
        result2 = validator.validate_request(request_data, "test-req-2")
        stats2 = validator.get_cache_stats()
        assert stats2["cache_hits"] == 1
        assert stats2["cache_misses"] == 1

        # Results should be the same object (from cache)
        assert result1 is result2

    def test_different_requests_cache_separately(self) -> None:
        """Test that different requests are cached separately."""
        validator = RequestValidator()

        request1 = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 1024,
        }

        request2 = {
            "model": "claude-3-haiku-20240307",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 512,
        }

        result1 = validator.validate_request(request1)
        result2 = validator.validate_request(request2)

        assert result1.model == "claude-3-opus-20240229"
        assert result2.model == "claude-3-haiku-20240307"
        assert result1 is not result2

        stats = validator.get_cache_stats()
        assert stats["cache_size"] == 2
        assert stats["cache_misses"] == 2

    def test_request_hash_generation(self) -> None:
        """Test that request hashing is consistent."""
        validator = RequestValidator()

        request_data = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 1024,
        }

        import json

        request_json = json.dumps(request_data, sort_keys=True)
        hash1 = validator._get_request_hash(request_json)
        hash2 = validator._get_request_hash(request_json)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 produces 64 hex characters

    def test_request_hash_different_for_different_requests(self) -> None:
        """Test that different requests produce different hashes."""
        validator = RequestValidator()

        import json

        request1_json = json.dumps({"content": "Hello"}, sort_keys=True)
        request2_json = json.dumps({"content": "World"}, sort_keys=True)

        hash1 = validator._get_request_hash(request1_json)
        hash2 = validator._get_request_hash(request2_json)

        assert hash1 != hash2

    def test_request_hash_same_for_reordered_keys(self) -> None:
        """Test that key order doesn't affect hash due to sort_keys."""
        validator = RequestValidator()

        import json

        # Same data, different key order
        data1 = {"model": "claude", "max_tokens": 100}
        data2 = {"max_tokens": 100, "model": "claude"}

        json1 = json.dumps(data1, sort_keys=True)
        json2 = json.dumps(data2, sort_keys=True)

        hash1 = validator._get_request_hash(json1)
        hash2 = validator._get_request_hash(json2)

        assert hash1 == hash2

    def test_lru_cache_eviction(self) -> None:
        """Test that LRU cache evicts least recently used items."""
        validator = RequestValidator(cache_size=3)

        # Add 3 requests to fill cache
        for i in range(3):
            request = {
                "model": "claude-3-opus-20240229",
                "messages": [{"role": "user", "content": f"Message {i}"}],
                "max_tokens": 1024,
            }
            validator.validate_request(request)

        stats = validator.get_cache_stats()
        assert stats["cache_size"] == 3

        # Access request 1 and 2 to make them recently used
        request1 = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Message 1"}],
            "max_tokens": 1024,
        }
        request2 = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Message 2"}],
            "max_tokens": 1024,
        }

        validator.validate_request(request1)  # Cache hit
        validator.validate_request(request2)  # Cache hit

        # Add a new request (should evict request 0, the least recently used)
        request_new = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "New message"}],
            "max_tokens": 1024,
        }
        validator.validate_request(request_new)

        # Cache should still be size 3
        assert validator.get_cache_stats()["cache_size"] == 3

        # Request 0 should now be a cache miss (was evicted)
        request0 = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Message 0"}],
            "max_tokens": 1024,
        }
        validator.validate_request(request0)

        # This should be a cache miss (evicted) followed by re-adding
        stats = validator.get_cache_stats()
        assert stats["cache_misses"] == 5  # 3 initial + 1 new + 1 re-add

    def test_get_cache_stats_structure(self) -> Any:
        """Test cache stats return correct structure."""
        validator = RequestValidator(cache_size=100)

        stats = validator.get_cache_stats()

        assert "cache_size" in stats
        assert "max_cache_size" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "hit_rate" in stats
        assert "total_requests" in stats

        assert stats["max_cache_size"] == 100
        assert stats["cache_size"] == 0
        assert stats["hit_rate"] == 0.0

    def test_hit_rate_calculation(self) -> None:
        """Test that hit rate is calculated correctly."""
        validator = RequestValidator()

        request = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 1024,
        }

        # First request - miss
        validator.validate_request(request)
        stats = validator.get_cache_stats()
        assert stats["hit_rate"] == 0.0

        # Next 3 requests - hits
        for _ in range(3):
            validator.validate_request(request)

        stats = validator.get_cache_stats()
        assert stats["total_requests"] == 4
        assert stats["cache_hits"] == 3
        assert stats["cache_misses"] == 1
        assert stats["hit_rate"] == 0.75

    def test_validation_error_not_cached(self) -> None:
        """Test that validation errors are not cached."""
        validator = RequestValidator()

        invalid_request = {
            "model": "claude-3-opus-20240229",
            # Missing required 'messages' field
            "max_tokens": 1024,
        }

        with pytest.raises(Exception):  # Pydantic validation error
            validator.validate_request(invalid_request)

        # Cache should be empty
        assert validator.get_cache_stats()["cache_size"] == 0
        assert validator.get_cache_stats()["cache_misses"] == 1

    def test_global_validator_instance(self) -> None:
        """Test that global validator instance exists and is configured."""
        assert isinstance(request_validator, RequestValidator)
        assert request_validator._cache_size == 10000

    def test_multiple_cache_hits(self) -> None:
        """Test multiple cache hits update LRU order."""
        validator = RequestValidator(cache_size=5)

        request = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 1024,
        }

        # Initial request
        validator.validate_request(request)

        # Multiple hits
        for _ in range(10):
            validator.validate_request(request)

        stats = validator.get_cache_stats()
        assert stats["cache_hits"] == 10
        assert stats["cache_misses"] == 1
        assert stats["total_requests"] == 11

    def test_request_id_optional(self) -> None:
        """Test that request_id is optional."""
        validator = RequestValidator()

        request = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 1024,
        }

        # Without request_id
        result = validator.validate_request(request)
        assert result is not None

    def test_complex_request_validation(self) -> None:
        """Test validation of complex request with tools and system prompt."""
        validator = RequestValidator()

        request = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "max_tokens": 1024,
            "system": "You are a helpful assistant.",
            "temperature": 0.7,
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "input_schema": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                }
            ],
        }

        result = validator.validate_request(request)
        assert result.model == "claude-3-opus-20240229"
        assert result.system == "You are a helpful assistant."
        assert result.temperature == 0.7
        assert len(result.tools) == 1  # type: ignore[arg-type]
        assert result.tools[0].name == "get_weather"  # type: ignore[index]

    def test_cache_with_streaming_parameter(self) -> None:
        """Test caching works with streaming parameter."""
        validator = RequestValidator()

        request1 = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 1024,
            "stream": True,
        }

        request2 = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 1024,
            "stream": False,
        }

        result1 = validator.validate_request(request1)
        result2 = validator.validate_request(request2)

        # Different requests due to different stream value
        assert result1 is not result2
        assert validator.get_cache_stats()["cache_size"] == 2
