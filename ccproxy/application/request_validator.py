"""Request validation utilities for performance optimization."""

from typing import Dict, Any, Optional
from functools import lru_cache
from collections import OrderedDict
import hashlib
import json

from ..domain.models import MessagesRequest
from ..logging import debug, LogRecord, LogEvent


class RequestValidator:
    """Validates and caches request validation results."""

    def __init__(self, cache_size: int = 10000):
        self._cache_size = cache_size
        # Use OrderedDict for efficient LRU cache implementation
        self._validation_cache: OrderedDict[str, MessagesRequest] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

    @lru_cache(maxsize=10000)
    def _get_request_hash(self, request_json: str) -> str:
        """Generate a hash for request caching."""
        return hashlib.md5(request_json.encode()).hexdigest()

    def validate_request(self, raw_body: Dict[str, Any], request_id: Optional[str] = None) -> MessagesRequest:
        """Validate request with caching for repeated requests."""
        request_json = json.dumps(raw_body, sort_keys=True)
        request_hash = self._get_request_hash(request_json)

        # Check if we've seen this exact request before
        if request_hash in self._validation_cache:
            self._cache_hits += 1
            # Move to end (most recently used)
            self._validation_cache.move_to_end(request_hash)
            debug(LogRecord(
                LogEvent.ANTHROPIC_REQUEST.value,
                "Using cached validation result",
                request_id,
                {"cache_hit": True, "hit_rate": self._cache_hits / (self._cache_hits + self._cache_misses)}
            ))
            return self._validation_cache[request_hash]

        # Validate the request
        self._cache_misses += 1
        validated_request = MessagesRequest.model_validate(raw_body)

        # Cache successful validation with LRU eviction
        if len(self._validation_cache) >= self._cache_size:
            # Remove least recently used (first item)
            self._validation_cache.popitem(last=False)
        self._validation_cache[request_hash] = validated_request

        return validated_request

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._cache_hits + self._cache_misses
        return {
            "cache_size": len(self._validation_cache),
            "max_cache_size": self._cache_size,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": self._cache_hits / max(1, total_requests),
            "total_requests": total_requests
        }


# Global validator instance with maximized cache
request_validator = RequestValidator(cache_size=10000)
