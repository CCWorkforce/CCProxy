import os
from openai import AsyncOpenAI, DefaultAioHttpClient

openai_client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                http_client=DefaultAioHttpClient(http2=True),
            )
