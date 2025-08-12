import pytest
from unittest.mock import AsyncMock, MagicMock

from ccproxy.interfaces.http.streaming import (
    handle_anthropic_streaming_response_from_openai_stream,
)


@pytest.mark.asyncio
async def test_streaming_thinking_content():
    # Setup mock OpenAI stream
    mock_stream = AsyncMock()
    mock_stream.__aiter__.return_value = [
        MagicMock(
            choices=[
                MagicMock(
                    delta=MagicMock(content="think END_THINKING", tool_calls=None),
                    finish_reason=None,
                )
            ]
        )
    ]

    # Invoke streaming handler
    events = []
    async for event in handle_anthropic_streaming_response_from_openai_stream(
        openai_stream=mock_stream,
        original_anthropic_model_name="claude-3-opus",
        estimated_input_tokens=10,
        request_id="test123",
        start_time_mono=0.0,
        thinking_enabled=True,
    ):
        events.append(event)

    # Verify thinking block generated
    assert any("content_block_start" in e and "thinking" in e for e in events)


@pytest.mark.asyncio
async def test_streaming_text_content():
    # Setup mock OpenAI stream with text
    mock_stream = AsyncMock()
    mock_stream.__aiter__.return_value = [
        MagicMock(
            choices=[
                MagicMock(
                    delta=MagicMock(content="Hello", tool_calls=None),
                    finish_reason="stop",
                )
            ]
        )
    ]

    events = []
    async for event in handle_anthropic_streaming_response_from_openai_stream(
        openai_stream=mock_stream,
        original_anthropic_model_name="claude-3-opus",
        estimated_input_tokens=10,
        request_id="test123",
        start_time_mono=0.0,
        thinking_enabled=False,
    ):
        events.append(event)

    assert any("text" in e for e in events)
