import openai
from typing import Any

from ...config import Settings


class OpenAIProvider:
    def __init__(self, settings: Settings):
        self._client = openai.AsyncClient(
            api_key=settings.openai_api_key,
            base_url=settings.base_url,
            default_headers={
                "HTTP-Referer": settings.referer_url,
                "X-Title": settings.app_name,
            },
            timeout=180.0,
        )

    async def create_chat_completion(self, **params: Any) -> Any:
        return await self._client.chat.completions.create(**params)
