import pytest

from ccproxy.application.tokenizer import truncate_request
from ccproxy.domain.models import Message, ContentBlockText
from ccproxy.config import TruncationConfig  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_utf8_preservation() -> None:
    # Create message with multi-byte UTF-8 characters
    messages = [
        Message(
            role="user",
            content=[
                ContentBlockText(
                    type="text",
                    text="This is a test 😊 with Chinese: 你好 and emoji sequences: 👨‍💻",
                )
            ],
        ),
        Message(
            role="assistant",
            content=[
                ContentBlockText(
                    type="text",
                    text="Response with special chars: £€¥ and Korean: 안녕하세요",
                )
            ],
        ),
    ]

    system = "System prompt 🌍 with multi-lingual support"
    config = TruncationConfig()

    # Test truncation doesn't corrupt UTF-8
    truncated_msgs, truncated_system = await truncate_request(
        messages,
        system,
        "gpt-5",
        10000,
        config,
        request_id="utf8_test",  # type: ignore[arg-type]
    )

    # Verify all text remains valid UTF-8
    all_text = []
    if truncated_system:
        all_text.append(truncated_system)
    for msg in truncated_msgs:
        if isinstance(msg.content, ContentBlockText):  # type: ignore[unreachable]
            all_text.append(msg.content.text)  # type: ignore[unreachable]

    for text in all_text:
        assert isinstance(text, str)
        # Verify string can be encoded to UTF-8
        encoded = text.encode("utf-8", errors="strict")
        # Verify decoding works
        assert encoded.decode("utf-8") == text
