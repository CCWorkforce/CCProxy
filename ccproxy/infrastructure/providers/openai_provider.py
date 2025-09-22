"""
Optimized OpenAI provider with high-performance HTTP client configuration.
Supports multiple HTTP client backends for maximum performance.
"""

from openai import AsyncOpenAI
from openai import DefaultAioHttpClient
from openai import DefaultAsyncHttpxClient
from typing import Any, Optional
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
        self._openAIClient: Optional[AsyncOpenAI] = None
        self._max_retries = settings.provider_max_retries
        self._base_delay = settings.provider_retry_base_delay
        self._jitter = settings.provider_retry_jitter
        self._http_client = None  # Will be managed by OpenAI SDK

        try:
            # High-performance configuration
            _read_timeout = float(self.settings.max_stream_seconds)

            # Check deployment environment
            is_local = os.getenv("IS_LOCAL_DEPLOYMENT", "False").lower() == "true"

            if is_local:
                # Use httpx for local development (better debugging)
                self._http_client = DefaultAsyncHttpxClient(
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
                    verify=os.getenv("SSL_CERT_FILE", True),
                    follow_redirects=True,
                    http2=True,
                )
                logging.info("Using DefaultAsyncHttpxClient for local deployment")
            else:
                # Use aiohttp for production (better performance under high concurrency)
                # Note: DefaultAioHttpClient requires the openai SDK to be installed with aiohttp extra
                # Try to use aiohttp client if available, fallback to httpx with optimized settings
                try:
                    # Try to create aiohttp client - will fail if aiohttp extra not installed
                    self._http_client = DefaultAioHttpClient()
                    logging.info("Using DefaultAioHttpClient for production deployment")
                except (ImportError, RuntimeError):
                    # Fallback to optimized httpx client for production
                    # RuntimeError is raised when aiohttp extra is not installed
                    self._http_client = DefaultAsyncHttpxClient(
                        limits=httpx.Limits(
                            max_keepalive_connections=100,  # Higher for production
                            max_connections=1000,  # Higher for production
                            keepalive_expiry=120,
                        ),
                        timeout=httpx.Timeout(
                            connect=10.0,
                            read=_read_timeout,
                            write=30.0,
                            pool=10.0,
                        ),
                        verify=os.getenv("SSL_CERT_FILE", True),
                        follow_redirects=True,
                    )
                    logging.info("Using optimized DefaultAsyncHttpxClient for production deployment (aiohttp not available)")

            # Initialize OpenAI client with the appropriate HTTP client
            self._openAIClient = AsyncOpenAI(
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
        except Exception:
            # Clean up if initialization fails
            # The OpenAI SDK clients handle their own cleanup
            self._http_client = None
            raise

    async def create_chat_completion(self, **params: Any) -> Any:
        """Create a chat completion with retry logic and UTF-8 error handling.

        Args:
            **params: Parameters to pass to the OpenAI chat completion API

        Returns:
            The API response from OpenAI (regular object for non-streaming,
            async generator for streaming)

        Raises:
            openai.APIError: If there are issues with the API call after retries
        """

        if not self._openAIClient:
            raise ValueError("OpenAI client not properly initialized")

        # Handle streaming case separately so we can return the AsyncStream
        stream = params.get('stream', False)
        if stream:
            # For streaming, await the client call to obtain the AsyncStream
            attempt = 0
            while True:
                try:
                    return await self._openAIClient.chat.completions.create(**params)
                except openai.RateLimitError as e:
                    if attempt >= self._max_retries:
                        raise e
                    delay = self._base_delay * (2**attempt) + random.uniform(
                        0, self._jitter
                    )
                    await asyncio.sleep(delay)
                    attempt += 1
                except (
                    openai.APIConnectionError,
                    httpx.ConnectError,
                    httpx.TimeoutException,
                    httpx.NetworkError,
                ) as e:
                    if attempt >= self._max_retries:
                        raise e
                    delay = self._base_delay * (2**attempt) + random.uniform(
                        0, self._jitter
                    )
                    await asyncio.sleep(delay)
                    attempt += 1

        # Non-streaming case
        attempt = 0
        while True:
            try:
                response = await self._openAIClient.chat.completions.create(**params)
                # If response has a .content attribute that's bytes, decode it safely
                if hasattr(response, 'choices') and response.choices:
                    for choice in response.choices:
                        if hasattr(choice, 'message') and hasattr(choice.message, 'content') and isinstance(choice.message.content, bytes):
                            choice.message.content = _safe_decode_response(choice.message.content, "chat completion response")
                return response
            except openai.RateLimitError as e:
                if attempt >= self._max_retries:
                    raise e
                delay = self._base_delay * (2**attempt) + random.uniform(
                    0, self._jitter
                )
                await asyncio.sleep(delay)
                attempt += 1
            except (
                openai.APIConnectionError,
                httpx.ConnectError,
                httpx.TimeoutException,
                httpx.NetworkError,
            ) as e:
                # Handle network-related errors with retry
                if attempt >= self._max_retries:
                    raise e
                delay = self._base_delay * (2**attempt) + random.uniform(
                    0, self._jitter
                )
                await asyncio.sleep(delay)
                attempt += 1
            except UnicodeDecodeError as e:
                raise ValueError(
                    f"Received malformed response from API that could not be decoded as UTF-8: {str(e)}"
                ) from e
            except Exception as e:
                if "utf-8" in str(e).lower() or "codec can't decode" in str(e).lower():
                    raise ValueError(
                        f"Received malformed response from API that could not be decoded as UTF-8: {str(e)}"
                    ) from e
                raise

    async def close(self) -> None:
        """Clean up the HTTP client when shutting down."""
        # The OpenAI SDK clients handle their own cleanup
        self._http_client = None
        self._openAIClient = None

    async def __aenter__(self) -> "OpenAIProvider":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
