"""
Optimized OpenAI provider with high-performance HTTP client configuration.
Supports multiple HTTP client backends for maximum performance.
"""

from openai import AsyncOpenAI
from typing import Any, Optional, Literal
import httpx
import os
import logging

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
        return response_bytes.decode('utf-8')
    except UnicodeDecodeError as e:
        # Log the issue
        logging.warning(
            f"Malformed UTF-8 bytes detected in {context}. "
            f"Attempting recovery with byte replacement. Error: {str(e)}"
        )

        try:
            # Try with error replacement - replaces malformed bytes with replacement character
            decoded = response_bytes.decode('utf-8', errors='replace')

            # Log successful recovery
            logging.info(f"Successfully recovered {context} by replacing malformed UTF-8 bytes")
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

    def __init__(
        self,
        settings: Settings,
        backend: Literal["httpx", "httpx-http2", "aiohttp"] = "httpx-http2"
    ):
        self.settings = settings
        self.backend = backend
        self._client: Optional[AsyncOpenAI] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._aiohttp_session: Optional[Any] = None

        # Initialize immediately for backward compatibility
        self._initialize_sync()

    def _initialize_sync(self):
        """Initialize the HTTP client synchronously for backward compatibility."""
        if self.backend == "aiohttp":
            # For aiohttp, we'll initialize lazily in async methods
            return

        # High-performance HTTP/2 configuration (default)
        _read_timeout = float(self.settings.max_stream_seconds)
        if self.backend == "httpx-http2":
            self._http_client = httpx.AsyncClient(
                limits=httpx.Limits(
                    # Connection pool settings
                    max_keepalive_connections=50,  # Increased from 10
                    max_connections=500,  # Increased from 100
                    keepalive_expiry=120,  # Increased from 30 seconds
                ),
                timeout=httpx.Timeout(
                    # Timeout settings
                    connect=10.0,  # Connection timeout
                    read=_read_timeout,    # Read timeout
                    write=30.0,    # Write timeout
                    pool=10.0,     # Connection pool timeout
                ),
                # Enable HTTP/2 for better performance
                http2=True,
                # Verify SSL but allow custom CA bundle
                verify=os.getenv("SSL_CERT_FILE", True),
                # Follow redirects
                follow_redirects=True,
                # Custom headers for all requests
                headers={
                    "User-Agent": "CCProxy/1.0 OptimizedClient",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Connection": "keep-alive",
                },
            )
        else:
            # Standard HTTP/1.1 configuration (fallback)
            self._http_client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_keepalive_connections=30,
                    max_connections=200,
                    keepalive_expiry=60,
                ),
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=_read_timeout,
                    write=30.0,
                    pool=10.0,
                ),
                http2=False,
                verify=os.getenv("SSL_CERT_FILE", True),
                follow_redirects=True,
            )

        # Initialize OpenAI client with our custom HTTP client
        self._client = AsyncOpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.base_url,
            default_headers={
                "HTTP-Referer": self.settings.referer_url,
                "X-Title": self.settings.app_name,
                "Accept-Charset": "utf-8",  # Explicitly request UTF-8 responses
            },
            timeout=_read_timeout,
            http_client=self._http_client,
            # Retry configuration
            max_retries=3,
        )

    async def _initialize_aiohttp(self):
        """Initialize aiohttp session for alternative backend."""
        if self._aiohttp_session:
            return

        try:
            import aiohttp
        except ImportError:
            raise ImportError(
                "aiohttp is not installed. Install it with: pip install aiohttp aiodns"
            )

        # Create optimized connector
        connector = aiohttp.TCPConnector(
            limit=500,  # Total connection pool limit
            limit_per_host=100,  # Per-host connection limit
            ttl_dns_cache=300,  # DNS cache timeout (5 minutes)
            enable_cleanup_closed=True,
            force_close=False,
            keepalive_timeout=120,
            use_dns_cache=True,
        )

        # Create session with optimized settings
        timeout = aiohttp.ClientTimeout(
            total=_read_timeout,
            connect=10,
            sock_connect=10,
            sock_read=180,
        )

        self._aiohttp_session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.settings.openai_api_key}",
                "Content-Type": "application/json",
                "User-Agent": "CCProxy/1.0 AIOHTTPClient",
                "Accept-Charset": "utf-8",  # Explicitly request UTF-8 responses
            },
            connector_owner=True,
            raise_for_status=False,
            read_bufsize=65536,  # 64KB read buffer
        )

    async def create_chat_completion(self, **params: Any) -> Any:
        """Create a chat completion with the optimized client."""

        # Use aiohttp backend if selected
        if self.backend == "aiohttp":
            return await self._create_chat_completion_aiohttp(**params)

        # Use httpx backend (default)
        if not self._client:
            self._initialize_sync()

        try:
            return await self._client.chat.completions.create(**params)
        except UnicodeDecodeError as e:
            # Convert UnicodeDecodeError to a more specific OpenAI API error
            import openai
            raise openai.APIError(
                message=f"Received malformed response from API that could not be decoded as UTF-8: {str(e)}",
                request=None,
                body=None
            ) from e
        except Exception as e:
            # Handle other potential encoding issues from httpx/openai client
            if "utf-8" in str(e).lower() or "codec can't decode" in str(e).lower():
                import openai
                raise openai.APIError(
                    message=f"Received malformed response from API that could not be decoded as UTF-8: {str(e)}",
                    request=None,
                    body=None
                ) from e
            raise

    async def _create_chat_completion_aiohttp(self, **params: Any) -> Any:
        """Create a chat completion using aiohttp backend."""
        await self._initialize_aiohttp()

        import json

        url = f"{self.settings.base_url}/chat/completions"

        # Handle streaming separately
        if params.get("stream", False):
            return await self._create_chat_completion_stream_aiohttp(**params)

        try:
            async with self._aiohttp_session.post(url, json=params) as response:
                if response.status != 200:
                    error_data = await response.text()
                    raise Exception(f"API error: {response.status} - {error_data}")

                # Read response with comprehensive error handling for malformed content
                try:
                    # First try to read response text with UTF-8 handling
                    response_text = None
                    try:
                        response_text = await response.text(encoding='utf-8')
                    except UnicodeDecodeError as e:
                        # Use safe decoding utility for malformed bytes
                        try:
                            response_bytes = await response.read()
                            response_text = _safe_decode_response(response_bytes, "non-streaming API response")
                        except UnicodeDecodeError:
                            # Convert UnicodeDecodeError to OpenAI API error
                            import openai
                            raise openai.APIError(
                                message=f"Received malformed response from API that could not be decoded as UTF-8: {str(e)}",
                                request=None,
                                body=None
                            ) from e

                    # Parse JSON from the cleaned text
                    data = json.loads(response_text)

                except json.JSONDecodeError as e:
                    # Handle invalid JSON response
                    import openai
                    raise openai.APIError(
                        message=f"Received invalid JSON response from API: {str(e)}",
                        request=None,
                        body=None
                    ) from e
                except UnicodeDecodeError as e:
                    # Convert UnicodeDecodeError to a more specific OpenAI API error
                    import openai
                    raise openai.APIError(
                        message=f"Received malformed response from API that could not be decoded as UTF-8: {str(e)}",
                        request=None,
                        body=None
                    ) from e

                # Convert to OpenAI response format
                from openai.types.chat import ChatCompletion
                return ChatCompletion(**data)
        except UnicodeDecodeError as e:
            # Also catch UnicodeDecodeError at the outer level for any other potential sources
            import openai
            raise openai.APIError(
                message=f"Received malformed response from API that could not be decoded as UTF-8: {str(e)}",
                request=None,
                body=None
            ) from e

    async def _create_chat_completion_stream_aiohttp(self, **params: Any):
        """Create a streaming chat completion using aiohttp backend."""
        import json

        url = f"{self.settings.base_url}/chat/completions"
        params["stream"] = True

        try:
            async with self._aiohttp_session.post(url, json=params) as response:
                if response.status != 200:
                    error_data = await response.text()
                    raise Exception(f"API error: {response.status} - {error_data}")

                async for line in response.content:
                    if line:
                        try:
                            line = line.decode('utf-8').strip()
                        except UnicodeDecodeError as e:
                            # Use safe decoding utility for malformed bytes
                            try:
                                line = _safe_decode_response(line, "streaming response line").strip()
                            except UnicodeDecodeError:
                                # Convert UnicodeDecodeError to a more specific OpenAI API error
                                import openai
                                raise openai.APIError(
                                    message=f"Received malformed streaming response that could not be decoded as UTF-8: {str(e)}",
                                    request=None,
                                    body=None
                                ) from e

                        if line.startswith("data: "):
                            data = line[6:]
                            if data != "[DONE]":
                                try:
                                    from openai.types.chat import ChatCompletionChunk
                                    yield ChatCompletionChunk(**json.loads(data))
                                except json.JSONDecodeError as e:
                                    # Handle invalid JSON in streaming response
                                    import openai
                                    raise openai.APIError(
                                        message=f"Received invalid JSON in streaming response: {str(e)}",
                                        request=None,
                                        body=None
                                    ) from e
        except UnicodeDecodeError as e:
            # Also catch UnicodeDecodeError at the outer level for any other potential sources
            import openai
            raise openai.APIError(
                message=f"Received malformed streaming response from API that could not be decoded as UTF-8: {str(e)}",
                request=None,
                body=None
            ) from e

    async def close(self):
        """Clean up the HTTP client when shutting down."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
            self._client = None

        if self._aiohttp_session:
            await self._aiohttp_session.close()
            self._aiohttp_session = None

    async def __aenter__(self):
        """Async context manager entry."""
        if self.backend == "aiohttp":
            await self._initialize_aiohttp()
        elif not self._client:
            self._initialize_sync()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
