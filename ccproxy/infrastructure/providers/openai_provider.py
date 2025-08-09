from openai import AsyncOpenAI
from typing import Any
import httpx

from ...config import Settings


class OpenAIProvider:
    def __init__(self, settings: Settings):
        # Configure connection pooling for better performance
        http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=100,
                keepalive_expiry=30,
            ),
            timeout=httpx.Timeout(180.0),
        )
        
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.base_url,
            default_headers={
                "HTTP-Referer": settings.referer_url,
                "X-Title": settings.app_name,
            },
            timeout=180.0,
            http_client=http_client,
        )

    async def create_chat_completion(self, **params: Any) -> Any:
        return await self._client.chat.completions.create(**params)
    
    async def close(self):
        """Clean up the HTTP client when shutting down."""
        if hasattr(self._client, '_client'):
            await self._client._client.close()
