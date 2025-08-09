from typing import Any
from typing_extensions import Protocol


class ChatProvider(Protocol):
    async def create_chat_completion(self, **params: Any) -> Any: ...
