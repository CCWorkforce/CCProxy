import pytest
from ccproxy.application.tokenizer import (
    truncate_request,
    count_tokens_for_anthropic_request,
)
from ccproxy.domain.models import Message, ContentBlockText
from ccproxy.config import TruncationConfig


def create_test_messages(count: int, role: str = "user") -> list[Message]:
    # Repeat text to ensure it exceeds token limits
    long_text = (
        "This is a longer message content that should push token count higher. " * 20
    )
    return [
        Message(
            role=role,
            content=[ContentBlockText(type="text", text=f"{long_text} Message {i}")],
        )
        for i in range(count)
    ]


@pytest.mark.asyncio
async def test_truncate_oldest_first():
    messages = create_test_messages(5)
    system = "System prompt"
    config = TruncationConfig()

    truncated_msgs, truncated_system = await truncate_request(
        messages, system, "gpt-4", 150, config, request_id="test"
    )
    new_tokens = await count_tokens_for_anthropic_request(
        truncated_msgs, truncated_system, "gpt-4", None, None
    )

    assert new_tokens <= 150
    # Check that some messages were removed
    assert len(truncated_msgs) < len(messages)


@pytest.mark.asyncio
async def test_truncate_newest_first():
    messages = create_test_messages(5)
    system = "System prompt"

    # Create config with newest_first strategy
    config = TruncationConfig()
    config.strategy = "newest_first"

    truncated_msgs, truncated_system = await truncate_request(
        messages, system, "gpt-4", 150, config, request_id="test"
    )
    new_tokens = await count_tokens_for_anthropic_request(
        truncated_msgs, truncated_system, "gpt-4", None, None
    )

    assert new_tokens <= 150
    # Check that some messages were removed
    assert len(truncated_msgs) < len(messages)


@pytest.mark.asyncio
async def test_truncate_system_priority():
    messages = create_test_messages(5)
    system = "System prompt"

    # Create config with system_priority strategy
    config = TruncationConfig()
    config.strategy = "system_priority"

    truncated_msgs, truncated_system = await truncate_request(
        messages, system, "gpt-4", 150, config, request_id="test"
    )
    new_tokens = await count_tokens_for_anthropic_request(
        truncated_msgs, truncated_system, "gpt-4", None, None
    )

    assert new_tokens <= 150
    assert truncated_system == "System prompt"


@pytest.mark.asyncio
async def test_truncate_below_limit():
    messages = create_test_messages(2)
    system = "System prompt"
    config = TruncationConfig()

    truncated_msgs, truncated_system = await truncate_request(
        messages, system, "gpt-4", 10000, config
    )

    assert len(truncated_msgs) == 2
    assert truncated_system == system
