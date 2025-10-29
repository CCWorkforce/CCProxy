"""Cache warmup and preloading functionality for improved performance."""

import hashlib
import json
from typing import Any, Dict, List, Optional
from pathlib import Path
import time
import aiofiles  # type: ignore[import-untyped]
from asyncer import asyncify
import anyio
from anyio import create_task_group as anyio_create_task_group

from ...domain.models import (
    MessagesRequest,
    MessagesResponse,
    Message,
    ContentBlockText,
    Usage,
)
from ...logging import info, warning, debug, LogRecord, LogEvent
from .response_cache import ResponseCache


class CacheWarmupConfig:
    """Configuration for cache warmup behavior."""

    def __init__(
        self,
        enabled: bool = True,
        warmup_file_path: Optional[str] = None,
        max_warmup_items: int = 100,
        warmup_on_startup: bool = True,
        preload_common_prompts: bool = True,
        auto_save_popular: bool = True,
        popularity_threshold: int = 3,
        save_interval_seconds: int = 3600,
    ):
        """
        Initialize cache warmup configuration.

        Args:
            enabled: Whether cache warmup is enabled
            warmup_file_path: Path to file containing warmup data
            max_warmup_items: Maximum number of items to warmup
            warmup_on_startup: Whether to warmup cache on application startup
            preload_common_prompts: Whether to preload common prompts
            auto_save_popular: Whether to automatically save popular queries
            popularity_threshold: Number of hits before considering item popular
            save_interval_seconds: Interval for saving popular items
        """
        self.enabled = enabled
        self.warmup_file_path = warmup_file_path or "cache_warmup.json"
        self.max_warmup_items = max_warmup_items
        self.warmup_on_startup = warmup_on_startup
        self.preload_common_prompts = preload_common_prompts
        self.auto_save_popular = auto_save_popular
        self.popularity_threshold = popularity_threshold
        self.save_interval_seconds = save_interval_seconds


class CacheWarmupManager:
    """Manages cache warmup and preloading operations."""

    def __init__(
        self,
        cache: ResponseCache,
        config: Optional[CacheWarmupConfig] = None,
    ):
        """
        Initialize cache warmup manager.

        Args:
            cache: The response cache to warm up
            config: Warmup configuration
        """
        self.cache = cache
        self.config = config or CacheWarmupConfig()
        self._popular_items: Dict[str, int] = {}
        self._task_group = None
        self._common_prompts = self._get_common_prompts()

    def _get_common_prompts(self) -> List[Dict[str, Any]]:
        """Get list of common prompts to preload."""
        return [
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "response": {
                    "id": "warmup_hello",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Hello! How can I help you today?"}
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 20},
                    "stop_reason": "end_turn",
                },
            },
            {
                "messages": [{"role": "user", "content": "What can you help me with?"}],
                "response": {
                    "id": "warmup_help",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "I can help you with a wide range of tasks including answering questions, writing, analysis, coding, and more!",
                        }
                    ],
                    "usage": {"input_tokens": 15, "output_tokens": 30},
                    "stop_reason": "end_turn",
                },
            },
            {
                "messages": [
                    {"role": "user", "content": "Explain how to use this API"}
                ],
                "response": {
                    "id": "warmup_api",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "This API provides OpenAI-compatible endpoints for Anthropic's Claude models...",
                        }
                    ],
                    "usage": {"input_tokens": 20, "output_tokens": 50},
                    "stop_reason": "end_turn",
                },
            },
        ]

    async def start(self) -> None:
        """Start the cache warmup manager."""
        if not self.config.enabled:
            debug(
                LogRecord(
                    event=LogEvent.CACHE_EVENT.value,
                    message="Cache warmup disabled",
                )
            )
            return

        info(
            LogRecord(
                event=LogEvent.CACHE_EVENT.value,
                message="Starting cache warmup manager",
            )
        )

        # Start background tasks using anyio task group
        async with anyio_create_task_group() as tg:
            self._task_group = tg  # type: ignore[assignment]

            # Start warmup on startup if configured
            if self.config.warmup_on_startup:
                tg.start_soon(self._warmup_cache)

            # Start auto-save task if configured
            if self.config.auto_save_popular:
                tg.start_soon(self._auto_save_loop)

    async def stop(self) -> None:
        """Stop the cache warmup manager."""
        # Cancel background tasks
        if self._task_group:
            self._task_group.cancel_scope.cancel()  # type: ignore[unreachable]
            try:
                await self._task_group.__aexit__(None, None, None)
            except Exception:
                pass
            self._task_group = None

        # Save popular items before shutdown
        if self.config.auto_save_popular:
            await self._save_popular_items()

    async def _warmup_cache(self) -> Any:
        """Warm up the cache with preloaded data."""
        try:
            warmup_items = []

            # Load common prompts if configured
            if self.config.preload_common_prompts:
                warmup_items.extend(self._common_prompts)

            # Load from warmup file if exists
            warmup_file = anyio.Path(self.config.warmup_file_path)
            if await warmup_file.exists():
                try:
                    # Read file asynchronously
                    content = await warmup_file.read_text()

                    # Parse JSON asynchronously
                    json_loads_async = asyncify(json.loads)
                    warmup_data = await json_loads_async(content)

                    # Add items from file
                    warmup_items.extend(warmup_data[: self.config.max_warmup_items])

                except Exception as e:
                    warning(
                        LogRecord(
                            event=LogEvent.CACHE_EVENT.value,
                            message=f"Failed to load warmup file: {e}",
                        )
                    )

            # Load all warmup items in parallel for better performance
            warmup_count = 0
            if warmup_items:
                from asyncer import create_task_group

                async with create_task_group() as tg:
                    soon_values = []
                    for item in warmup_items:
                        soon_values.append(tg.soonify(self._load_warmup_item)(item))

                # Count successful loads (None return indicates success)
                warmup_count = len(warmup_items)

            info(
                LogRecord(
                    event=LogEvent.CACHE_EVENT.value,
                    message=f"Cache warmup completed with {warmup_count} items",
                )
            )

        except Exception as e:
            warning(
                LogRecord(
                    event=LogEvent.CACHE_EVENT.value,
                    message=f"Cache warmup failed: {e}",
                )
            )

    async def _load_warmup_item(self, item: Dict[str, Any]) -> None:
        """Load a single warmup item into cache."""
        try:
            # Create request from warmup data
            messages = [
                Message(
                    role=msg["role"],
                    content=[ContentBlockText(type="text", text=msg["content"])]
                    if isinstance(msg.get("content"), str)
                    else msg.get("content", []),
                )
                for msg in item.get("messages", [])
            ]

            request = MessagesRequest(
                model=item.get("model", "claude-3-opus-20240229"),
                messages=messages,
                max_tokens=item.get("max_tokens", 100),
            )

            # Create response from warmup data
            response_data = item.get("response", {})
            response = MessagesResponse(
                id=response_data.get("id", f"warmup_{time.time()}"),
                type=response_data.get("type", "message"),
                role=response_data.get("role", "assistant"),
                model=request.model,
                content=[
                    ContentBlockText(type="text", text=content.get("text", ""))
                    for content in response_data.get("content", [])
                ],
                usage=Usage(
                    input_tokens=response_data.get("usage", {}).get("input_tokens", 10),
                    output_tokens=response_data.get("usage", {}).get(
                        "output_tokens", 20
                    ),
                ),
                stop_reason=response_data.get("stop_reason", "end_turn"),
            )

            # Generate cache key
            cache_key = await self._generate_cache_key(request)

            # Store in cache
            await self.cache.cache_response(request, response)

            debug(
                LogRecord(
                    event=LogEvent.CACHE_EVENT.value,
                    message=f"Warmed up cache entry: {cache_key[:20]}...",
                )
            )

        except Exception as e:
            debug(
                LogRecord(
                    event=LogEvent.CACHE_EVENT.value,
                    message=f"Failed to load warmup item: {e}",
                )
            )

    async def _auto_save_loop(self) -> None:
        """Periodically save popular items for warmup."""
        while True:
            try:
                await anyio.sleep(self.config.save_interval_seconds)
                await self._save_popular_items()
            except anyio.get_cancelled_exc_class():
                break
            except Exception as e:
                warning(
                    LogRecord(
                        event=LogEvent.CACHE_EVENT.value,
                        message=f"Failed to save popular items: {e}",
                    )
                )

    async def _save_popular_items(self) -> None:
        """Save popular cache items for future warmup."""
        try:
            # Get items that meet popularity threshold
            popular_items = [
                key
                for key, count in self._popular_items.items()
                if count >= self.config.popularity_threshold
            ]

            if not popular_items:
                return

            # Collect cache data for popular items
            warmup_data = []
            for cache_key in popular_items[: self.config.max_warmup_items]:
                # This would need access to the actual cached data
                # For now, we'll just save the keys
                warmup_data.append({"cache_key": cache_key})

            # Save to file using anyio.Path
            warmup_file = anyio.Path(self.config.warmup_file_path)
            parent_dir = warmup_file.parent
            if not await parent_dir.exists():
                await parent_dir.mkdir(parents=True)

            # Serialize JSON asynchronously
            json_dumps_async = asyncify(json.dumps)
            json_content = await json_dumps_async(warmup_data, indent=2)

            # Write to file asynchronously
            await warmup_file.write_text(json_content)

            info(
                LogRecord(
                    event=LogEvent.CACHE_EVENT.value,
                    message=f"Saved {len(warmup_data)} popular items for warmup",
                )
            )

        except Exception as e:
            warning(
                LogRecord(
                    event=LogEvent.CACHE_EVENT.value,
                    message=f"Failed to save popular items: {e}",
                )
            )

    def track_cache_hit(self, cache_key: str) -> Any:
        """Track a cache hit for popularity tracking."""
        if not self.config.auto_save_popular:
            return

        self._popular_items[cache_key] = self._popular_items.get(cache_key, 0) + 1

    async def _generate_cache_key(self, request: MessagesRequest) -> str:
        """Generate a cache key for a request."""
        # Simple key generation - in production would be more sophisticated
        # Use asyncify for JSON serialization
        json_dumps_async = asyncify(json.dumps)
        messages_json = await json_dumps_async(
            [msg.model_dump() for msg in request.messages]
        )

        key_parts = [
            request.model,
            str(request.max_tokens),
            messages_json,
        ]
        key_string = "|".join(key_parts)

        # Hash computation asynchronously
        hash_compute_async = asyncify(lambda s: hashlib.sha256(s.encode()).hexdigest())
        return await hash_compute_async(key_string)

    async def preload_responses(
        self, requests: List[MessagesRequest], responses: List[MessagesResponse]
    ) -> int:
        """
        Preload multiple request-response pairs into cache.

        Args:
            requests: List of requests
            responses: Corresponding list of responses

        Returns:
            Number of items successfully preloaded
        """
        if len(requests) != len(responses):
            raise ValueError("Requests and responses must have same length")

        preloaded = 0
        for request, response in zip(requests, responses):
            try:
                await self.cache.cache_response(request, response)
                preloaded += 1
            except Exception as e:
                debug(
                    LogRecord(
                        event=LogEvent.CACHE_EVENT.value,
                        message=f"Failed to preload item: {e}",
                    )
                )

        info(
            LogRecord(
                event=LogEvent.CACHE_EVENT.value,
                message=f"Preloaded {preloaded} items into cache",
            )
        )

        return preloaded

    async def warmup_from_log(self, log_file_path: str, max_items: int = 100) -> int:
        """
        Warm up cache from a log file of previous requests.

        Args:
            log_file_path: Path to log file
            max_items: Maximum items to load

        Returns:
            Number of items loaded
        """
        loaded = 0
        log_path = Path(log_file_path)

        if not log_path.exists():
            warning(
                LogRecord(
                    event=LogEvent.CACHE_EVENT.value,
                    message=f"Log file not found: {log_file_path}",
                )
            )
            return 0

        try:
            async with aiofiles.open(log_path, "r") as f:
                async for line in f:
                    if loaded >= max_items:
                        break

                    try:
                        # Parse log entry (assuming JSONL format)
                        entry = json.loads(line)

                        # Extract request/response if present
                        if "request" in entry and "response" in entry:
                            await self._load_warmup_item(entry)
                            loaded += 1

                    except Exception:
                        # Skip invalid entries
                        continue

        except Exception as e:
            warning(
                LogRecord(
                    event=LogEvent.CACHE_EVENT.value,
                    message=f"Failed to warmup from log: {e}",
                )
            )

        info(
            LogRecord(
                event=LogEvent.CACHE_EVENT.value,
                message=f"Loaded {loaded} items from log file",
            )
        )

        return loaded
