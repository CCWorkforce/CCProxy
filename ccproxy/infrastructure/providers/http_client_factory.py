"""
HTTP client factory for provider implementations.
Handles configuration and initialization of high-performance HTTP clients.
"""

import logging
import os
from typing import Any, Dict, Union

import httpx
from openai import DefaultAioHttpClient, DefaultAsyncHttpxClient

from ...config import Settings


class HttpClientFactory:
    """Factory for creating configured HTTP clients for OpenAI SDK."""

    @staticmethod
    def create_client(
        settings: Settings,
    ) -> Union[DefaultAsyncHttpxClient, DefaultAioHttpClient]:
        """
        Create an optimized HTTP client based on deployment environment.

        Args:
            settings: Application settings

        Returns:
            Configured HTTP client (httpx or aiohttp based)
        """
        # Check deployment environment
        is_local = os.getenv("IS_LOCAL_DEPLOYMENT", "False").lower() == "true"

        if is_local:
            return HttpClientFactory._create_local_client(settings)
        else:
            return HttpClientFactory._create_production_client(settings)

    @staticmethod
    def _create_local_client(
        settings: Settings,
    ) -> DefaultAsyncHttpxClient:
        """
        Create HTTP client optimized for local development.

        Args:
            settings: Application settings

        Returns:
            Configured httpx client for local use
        """
        # Calculate timeout from settings
        read_timeout = float(settings.max_stream_seconds)

        # Use local-specific defaults if not overridden
        max_keepalive = settings.pool_max_keepalive_connections
        max_connections = settings.pool_max_connections

        http_client_kwargs = {
            "limits": httpx.Limits(
                max_keepalive_connections=max_keepalive,
                max_connections=max_connections,
                keepalive_expiry=settings.pool_keepalive_expiry,
            ),
            "timeout": httpx.Timeout(
                connect=settings.http_connect_timeout,
                read=read_timeout,
                write=settings.http_write_timeout,
                pool=settings.http_pool_timeout,
            ),
            "verify": os.getenv("SSL_CERT_FILE", True),
            "follow_redirects": True,
        }

        try:
            # Try with HTTP/2
            client = DefaultAsyncHttpxClient(**http_client_kwargs, http2=True)
            logging.info(
                "Using DefaultAsyncHttpxClient with HTTP/2 for local deployment"
            )
            return client
        except ImportError:
            # Fallback without HTTP/2
            client = DefaultAsyncHttpxClient(**http_client_kwargs)
            logging.info(
                "Using DefaultAsyncHttpxClient (HTTP/1.1) for local deployment"
            )
            return client

    @staticmethod
    def _create_production_client(
        settings: Settings,
    ) -> Union[DefaultAioHttpClient, DefaultAsyncHttpxClient]:
        """
        Create HTTP client optimized for production.

        Args:
            settings: Application settings

        Returns:
            Configured client (aiohttp preferred, httpx fallback)
        """
        # Try to use aiohttp client if available, fallback to httpx
        try:
            # Try to create aiohttp client - will fail if aiohttp extra not installed
            client = DefaultAioHttpClient()
            logging.info("Using DefaultAioHttpClient for production deployment")
            return client
        except (ImportError, RuntimeError):
            # RuntimeError is raised when aiohttp extra is not installed
            # Fallback to optimized httpx client for production
            return HttpClientFactory._create_optimized_httpx_client(settings)

    @staticmethod
    def _create_optimized_httpx_client(
        settings: Settings,
    ) -> DefaultAsyncHttpxClient:
        """
        Create an optimized httpx client for production use.

        Args:
            settings: Application settings

        Returns:
            Optimized httpx client
        """
        # Calculate timeout from settings
        read_timeout = float(settings.max_stream_seconds)

        # Use production-optimized values with higher limits
        max_keepalive = max(settings.pool_max_keepalive_connections, 100)
        max_connections = min(
            settings.pool_max_connections, 300
        )  # Cap at reasonable limit

        http_client_kwargs = {
            "limits": httpx.Limits(
                max_keepalive_connections=max_keepalive,
                max_connections=max_connections,
                keepalive_expiry=settings.pool_keepalive_expiry,
            ),
            "timeout": httpx.Timeout(
                connect=settings.http_connect_timeout,
                read=read_timeout,
                write=settings.http_write_timeout,
                pool=settings.http_pool_timeout,
            ),
            "verify": os.getenv("SSL_CERT_FILE", True),
            "follow_redirects": True,
        }

        try:
            # Try with HTTP/2
            client = DefaultAsyncHttpxClient(**http_client_kwargs, http2=True)
            logging.info(
                "Using optimized DefaultAsyncHttpxClient with HTTP/2 for production. "
                "For better performance, install aiohttp: pip install 'openai[aiohttp]'"
            )
            return client
        except ImportError:
            # Fallback without HTTP/2
            client = DefaultAsyncHttpxClient(**http_client_kwargs)
            logging.info(
                "Using optimized DefaultAsyncHttpxClient (HTTP/1.1) for production. "
                "Install h2 for HTTP/2: pip install 'httpx[http2]'. "
                "For better performance, install aiohttp: pip install 'openai[aiohttp]'"
            )
            return client

    @staticmethod
    async def close_client(
        client: Union[DefaultAsyncHttpxClient, DefaultAioHttpClient, None],
    ) -> None:
        """
        Properly close an HTTP client to avoid resource leaks.

        Args:
            client: HTTP client to close
        """
        if not client:
            return

        try:
            if hasattr(client, "aclose"):
                # For httpx-based clients
                await client.aclose()
            elif hasattr(client, "_session") and hasattr(client._session, "close"):
                # For aiohttp-based clients
                await client._session.close()
        except Exception as e:
            logging.warning(f"Error closing HTTP client: {e}")


class HttpClientConfig:
    """Configuration helper for HTTP client parameters."""

    @staticmethod
    def get_default_headers(settings: Settings) -> Dict[str, str]:
        """
        Get default headers for HTTP requests.

        Args:
            settings: Application settings

        Returns:
            Dictionary of default headers
        """
        return {
            "HTTP-Referer": settings.referer_url,
            "X-Title": settings.app_name,
            "Accept-Charset": "utf-8",
        }

    @staticmethod
    def get_client_info() -> Dict[str, Any]:
        """
        Get information about available HTTP clients.

        Returns:
            Dictionary with client availability information
        """
        info = {
            "httpx_available": True,  # Always available as dependency
            "aiohttp_available": False,
            "http2_available": False,
        }

        # Check for aiohttp
        try:
            import aiohttp  # noqa: F401

            info["aiohttp_available"] = True
        except ImportError:
            pass

        # Check for HTTP/2 support
        try:
            import h2  # noqa: F401

            info["http2_available"] = True
        except ImportError:
            pass

        return info

    @staticmethod
    def log_client_configuration(
        client: Union[DefaultAsyncHttpxClient, DefaultAioHttpClient],
        settings: Settings,
    ) -> None:
        """
        Log HTTP client configuration details.

        Args:
            client: The configured HTTP client
            settings: Application settings
        """
        client_type = type(client).__name__
        is_local = os.getenv("IS_LOCAL_DEPLOYMENT", "False").lower() == "true"
        environment = "local" if is_local else "production"

        config_info = {
            "client_type": client_type,
            "environment": environment,
            "max_stream_seconds": settings.max_stream_seconds,
            "pool_max_keepalive": settings.pool_max_keepalive_connections,
            "pool_max_connections": settings.pool_max_connections,
        }

        # Add client-specific info
        if hasattr(client, "_limits"):
            # httpx client
            config_info["http2_enabled"] = getattr(client, "_http2", False)

        logging.info(f"HTTP client configuration: {config_info}")
