"""Data models for the cache module."""

import sys
import time
from dataclasses import dataclass, field

from ...domain.models import MessagesResponse


@dataclass
class CachedResponse:
    """Represents a cached response with metadata."""

    response: MessagesResponse
    request_hash: str
    timestamp: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0

    def __post_init__(self):
        """Calculate size after initialization."""
        if self.size_bytes == 0:
            # Estimate size of the cached response
            self.size_bytes = sys.getsizeof(self.response.model_dump_json())

    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if this cached response has expired."""
        return time.time() - self.timestamp > ttl_seconds

    def update_access(self):
        """Update access count and timestamp."""
        self.access_count += 1
        self.last_accessed = time.time()
