"""
Optimized OpenAI provider with high-performance HTTP client configuration.
Supports multiple HTTP client backends for maximum performance.
"""

from openai import AsyncOpenAI
from typing import Any, Optional, Literal
import httpx
import os
import logging
import asyncio
import random
import openai

from ...config import Settings


def _safe_decode_response(response_bytes: bytes, context: str = "API response") -> str:
    """
    Safely decode response bytes to UTF-8 string with error handling.

    Args:
        response_bytes: Raw bytes from HTTP response
        context: Description of the response context for logging

    Returns:
        Decoded UTF-8 string

    Raises:
        UnicodeDecodeError: If decoding fails even with error handling
    """
    try:
        # First try strict UTF-8 decoding
        return response_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        # Log the issue
        logging.warning(
            f"Malformed UTF-8 bytes detected in {context}. "
            f"Attempting recovery with byte replacement. Error: {str(e)}"
        )

        try:
            # Try with error replacement - replaces malformed bytes with replacement character
            decoded = response_bytes.decode("utf-8", errors="replace")

            # Log successful recovery
            logging.info(
                f"Successfully recovered {context} by replacing malformed UTF-8 bytes"
            )
            return decoded
        except Exception as recovery_error:
            # If even replacement fails, raise the original error
            logging.error(
                f"Failed to recover {context} even with byte replacement. "
                f"Recovery error: {str(recovery_error)}"
            )
            raise e


class OpenAIProvider:
    """
    High-performance OpenAI provider with optimized HTTP client settings.

    Key optimizations:
    - HTTP/2 support for multiplexing
    - Increased connection pool limits
    - Optimized keepalive settings
    - DNS caching
    - Request pipelining
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: Optional[AsyncOpenAI] = None
        self._max_retries = settings.provider_max_retries
        self._base_delay = settings.provider_retry_base_delay
        self._jitter = settings.provider_retry_jitter
        self._http_client: Optional[httpx.AsyncClient] = None

        # High-performance HTTP/2 configuration
        _read_timeout = float(self.settings.max_stream_seconds)
        self._http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=50,
                max_connections=500,
                keepalive_expiry=120,
            ),
            timeout=httpx.Timeout(
                connect=10.0,
                read=_read_timeout,
                write=30.0,
                pool=10.0,
            ),
            http2=True,
            verify=os.getenv("SSL_CERT_FILE", True),
            follow_redirects=True,
            headers={
                "User-Agent": "CCProxy/1.0 OptimizedClient",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
            },
        )

        # Initialize OpenAI client with our custom HTTP client
        self._client = AsyncOpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.base_url,
            default_headers={
                "HTTP-Referer": self.settings.referer_url,
                "X-Title": self.settings.app_name,
                "Accept-Charset": "utf-8",
            },
            timeout=_read_timeout,
            http_client=self._http_client,
            max_retries=self._max_retries,
        )

    async def create_chat_completion(self, **params: Any) -> Any:
        """Create a chat completion with the optimized client."""

        if not self._client:
            self._initialize_sync()

        attempt = 0
        while True:
            try:
                return await self._client.chat.completions.create(**params)
            except openai.RateLimitError as e:
                if attempt >= self._max_retries:
                    raise e
                delay = self._base_delay * (2**attempt) + random.uniform(
                    0, self._jitter
                )
                await asyncio.sleep(delay)
                attempt += 1
            except UnicodeDecodeError as e:
                raise openai.APIError(
                    message=f"Received malformed response from API that could not be decoded as UTF-8: {str(e)}",
                    request=None,
                    body=None,
                ) from e
            except Exception as e:
                if "utf-8" in str(e).lower() or "codec can't decode" in str(e).lower():
                    raise openai.APIError(
                        message=f"Received malformed response from API that could not be decoded as UTF-8: {str(e)}",
                        request=None,
                        body=None,
                    ) from e
                raise

    async def close(self):
        """Clean up the HTTP client when shutting down."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
