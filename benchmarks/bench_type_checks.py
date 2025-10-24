"""Benchmarks for type checking operations."""

from typing import Any

from ccproxy.application.type_utils import (  # type: ignore[attr-defined]
    is_text_block,
    is_image_block,
    is_tool_use_block,
    is_tool_result_block,
    ContentBlockDispatcher,
)
from ccproxy.domain.models import (
    ContentBlockText,
    ContentBlockImage,
    ContentBlockToolUse,
    ContentBlockToolResult,
)


# Sample data for benchmarks
TEXT_BLOCK = ContentBlockText(type="text", text="Hello, world!")
IMAGE_BLOCK = ContentBlockImage(
    type="image",
    source={
        "type": "base64",
        "media_type": "image/png",
        "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
    },
)
TOOL_USE_BLOCK = ContentBlockToolUse(
    type="tool_use",
    id="toolu_123",
    name="get_weather",
    input={"location": "San Francisco", "unit": "celsius"},
)
TOOL_RESULT_BLOCK = ContentBlockToolResult(
    type="tool_result",
    tool_use_id="toolu_123",
    content="The weather in San Francisco is 18°C and sunny.",
)


class TestTypeCheckPerformance:
    """Benchmark type checking functions."""

    def test_is_text_block(self, benchmark) -> None:  # type: ignore[no-untyped-def]
        """Benchmark is_text_block() calls."""
        result = benchmark(is_text_block, TEXT_BLOCK)
        assert result is True

    def test_is_image_block(self, benchmark) -> None:  # type: ignore[no-untyped-def]
        """Benchmark is_image_block() calls."""
        result = benchmark(is_image_block, IMAGE_BLOCK)
        assert result is True

    def test_is_tool_use_block(self: Any, benchmark: Any) -> Any:
        """Benchmark is_tool_use_block() calls."""
        result = benchmark(is_tool_use_block, TOOL_USE_BLOCK)
        assert result is True

    def test_is_tool_result_block(self: Any, benchmark: Any) -> Any:
        """Benchmark is_tool_result_block() calls."""
        result = benchmark(is_tool_result_block, TOOL_RESULT_BLOCK)
        assert result is True

    def test_multiple_type_checks(self: Any, benchmark: Any) -> Any:
        """Benchmark multiple type checks in sequence (realistic usage)."""

        def check_all_types() -> Any:
            blocks = [TEXT_BLOCK, IMAGE_BLOCK, TOOL_USE_BLOCK, TOOL_RESULT_BLOCK]
            results = []
            for block in blocks:
                if is_text_block(block):
                    results.append("text")
                elif is_image_block(block):
                    results.append("image")
                elif is_tool_use_block(block):
                    results.append("tool_use")
                elif is_tool_result_block(block):
                    results.append("tool_result")
            return results

        result = benchmark(check_all_types)
        assert len(result) == 4


class TestDispatcherPerformance:
    """Benchmark ContentBlockDispatcher operations."""

    def test_dispatcher_text(self: Any, benchmark: Any) -> Any:
        """Benchmark dispatcher for text blocks."""

        def text_handler(block) -> Any:  # type: ignore[no-untyped-def]
            return f"text: {block.text[:10]}"

        dispatcher = ContentBlockDispatcher()
        dispatcher.register_text_handler(text_handler)  # type: ignore[attr-defined]

        result = benchmark(dispatcher.dispatch, TEXT_BLOCK)
        assert result.startswith("text:")

    def test_dispatcher_image(self: Any, benchmark: Any) -> Any:
        """Benchmark dispatcher for image blocks."""

        def image_handler(block) -> Any:  # type: ignore[no-untyped-def]
            return f"image: {block.source['type']}"

        dispatcher = ContentBlockDispatcher()
        dispatcher.register_image_handler(image_handler)  # type: ignore[attr-defined]

        result = benchmark(dispatcher.dispatch, IMAGE_BLOCK)
        assert result.startswith("image:")

    def test_dispatcher_tool_use(self: Any, benchmark: Any) -> Any:
        """Benchmark dispatcher for tool use blocks."""

        def tool_use_handler(block) -> Any:  # type: ignore[no-untyped-def]
            return f"tool: {block.name}"

        dispatcher = ContentBlockDispatcher()
        dispatcher.register_tool_use_handler(tool_use_handler)  # type: ignore[attr-defined]

        result = benchmark(dispatcher.dispatch, TOOL_USE_BLOCK)
        assert result.startswith("tool:")

    def test_dispatcher_multiple_blocks(self: Any, benchmark: Any) -> Any:
        """Benchmark dispatcher for multiple blocks (realistic usage)."""

        def text_handler(block) -> Any:  # type: ignore[no-untyped-def]
            return f"text: {len(block.text)}"

        def image_handler(block) -> Any:  # type: ignore[no-untyped-def]
            return f"image: {block.source['type']}"

        def tool_use_handler(block) -> Any:  # type: ignore[no-untyped-def]
            return f"tool: {block.name}"

        def tool_result_handler(block) -> Any:  # type: ignore[no-untyped-def]
            return f"result: {block.tool_use_id}"

        dispatcher = ContentBlockDispatcher()
        dispatcher.register_text_handler(text_handler)  # type: ignore[attr-defined]
        dispatcher.register_image_handler(image_handler)  # type: ignore[attr-defined]
        dispatcher.register_tool_use_handler(tool_use_handler)  # type: ignore[attr-defined]
        dispatcher.register_tool_result_handler(tool_result_handler)  # type: ignore[attr-defined]

        def dispatch_all() -> Any:
            blocks = [TEXT_BLOCK, IMAGE_BLOCK, TOOL_USE_BLOCK, TOOL_RESULT_BLOCK]
            return [dispatcher.dispatch(block) for block in blocks]  # type: ignore[call-arg, arg-type]

        result = benchmark(dispatch_all)
        assert len(result) == 4
