"""
HTTP client factory for provider implementations.
Handles configuration and initialization of high-performance HTTP clients.
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Union

import httpx
from openai import DefaultAioHttpClient, DefaultAsyncHttpxClient

from ...config import Settings


class Environment(Enum):
    """Deployment environment types."""

    LOCAL = "local"
    PRODUCTION = "production"


@dataclass
class ConnectionLimits:
    """Connection pool configuration."""

    max_keepalive: int
    max_connections: int
    keepalive_expiry: float

    @classmethod
    def for_environment(
        cls, settings: Settings, env: Environment
    ) -> "ConnectionLimits":
        """Create connection limits based on environment."""
        if env == Environment.LOCAL:
            return cls(
                max_keepalive=settings.pool_max_keepalive_connections,
                max_connections=settings.pool_max_connections,
                keepalive_expiry=settings.pool_keepalive_expiry,
            )
        else:  # Production
            return cls(
                max_keepalive=max(settings.pool_max_keepalive_connections, 100),
                max_connections=min(settings.pool_max_connections, 300),
                keepalive_expiry=settings.pool_keepalive_expiry,
            )


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
        environment = HttpClientFactory._get_environment()

        if environment == Environment.PRODUCTION:
            # Try aiohttp first for production
            client = HttpClientFactory._try_create_aiohttp_client()
            if client:
                return client

        # Use httpx for local or as fallback for production
        return HttpClientFactory._create_httpx_client(settings, environment)

    @staticmethod
    def _get_environment() -> Environment:
        """Determine current deployment environment."""
        is_local = os.getenv("IS_LOCAL_DEPLOYMENT", "False").lower() == "true"
        return Environment.LOCAL if is_local else Environment.PRODUCTION

    @staticmethod
    def _try_create_aiohttp_client() -> Optional[DefaultAioHttpClient]:
        """
        Try to create an aiohttp client for production use.

        Returns:
            Configured aiohttp client or None if not available
        """
        try:
            client = DefaultAioHttpClient(http2=True)
            logging.info("Using DefaultAioHttpClient for production deployment")
            return client
        except (ImportError, RuntimeError):
            # aiohttp extra not installed or initialization failed
            return None

    @staticmethod
    def _create_httpx_client(
        settings: Settings, environment: Environment
    ) -> DefaultAsyncHttpxClient:
        """
        Create an httpx client with environment-specific configuration.

        Args:
            settings: Application settings
            environment: Deployment environment

        Returns:
            Configured httpx client
        """
        limits = ConnectionLimits.for_environment(settings, environment)

        http_client_kwargs = HttpClientFactory._build_httpx_config(settings, limits)

        return HttpClientFactory._create_with_http2_fallback(
            http_client_kwargs, environment
        )

    @staticmethod
    def _build_httpx_config(
        settings: Settings, limits: ConnectionLimits
    ) -> Dict[str, Any]:
        """Build httpx client configuration."""
        read_timeout = float(settings.max_stream_seconds)

        return {
            "limits": httpx.Limits(
                max_keepalive_connections=limits.max_keepalive,
                max_connections=limits.max_connections,
                keepalive_expiry=limits.keepalive_expiry,
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

    @staticmethod
    def _create_with_http2_fallback(
        http_client_kwargs: Dict[str, Any], environment: Environment
    ) -> DefaultAsyncHttpxClient:
        """
        Create httpx client with HTTP/2 support, falling back to HTTP/1.1.

        Args:
            http_client_kwargs: Base client configuration
            environment: Deployment environment

        Returns:
            Configured httpx client
        """
        env_name = environment.value

        try:
            # Try with HTTP/2
            client = DefaultAsyncHttpxClient(**http_client_kwargs, http2=True)
            logging.info(
                f"Using DefaultAsyncHttpxClient with HTTP/2 for {env_name} deployment"
            )
            return client
        except ImportError:
            # Fallback without HTTP/2
            client = DefaultAsyncHttpxClient(**http_client_kwargs)

            if environment == Environment.PRODUCTION:
                logging.info(
                    f"Using DefaultAsyncHttpxClient (HTTP/1.1) for {env_name}. "
                    "Install h2 for HTTP/2: pip install 'httpx[http2]'. "
                    "For better performance, install aiohttp: pip install 'openai[aiohttp]'"
                )
            else:
                logging.info(
                    f"Using DefaultAsyncHttpxClient (HTTP/1.1) for {env_name} deployment"
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
        environment = HttpClientFactory._get_environment()

        config_info = {
            "client_type": client_type,
            "environment": environment.value,
            "max_stream_seconds": settings.max_stream_seconds,
            "pool_max_keepalive": settings.pool_max_keepalive_connections,
            "pool_max_connections": settings.pool_max_connections,
        }

        # Add client-specific info
        if hasattr(client, "_limits"):
            # httpx client
            config_info["http2_enabled"] = getattr(client, "_http2", False)

        logging.info(f"HTTP client configuration: {config_info}")
