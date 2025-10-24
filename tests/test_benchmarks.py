import tracemalloc
import gc
import pytest
from time import monotonic
import anyio

from ccproxy.interfaces.http.streaming import StreamProcessor


@pytest.mark.benchmark
def test_stream_processor_memory() -> None:
    tracemalloc.start()
    gc.collect()
    snapshot1 = tracemalloc.take_snapshot()

    class MockEncoder:
        def encode(self, text: str) -> list[int]:
            return [1]

    enc_mock = MockEncoder()
    processor = StreamProcessor(
        enc=enc_mock, request_id="bench", thinking_enabled=False
    )
    for _ in range(100):
        anyio.run(processor.process_text_content, "chunk")

    gc.set_debug(gc.DEBUG_SAVEALL)
    start_time = monotonic()

    processor = StreamProcessor(enc_mock, "bench", False)

    async def process_all_chunks() -> None:
        for i in range(1000):
            await processor.process_text_content(f"chunk_{i}")

    anyio.run(process_all_chunks)

    duration = monotonic() - start_time
    snapshot2 = tracemalloc.take_snapshot()
    stats = snapshot2.compare_to(snapshot1, "lineno")

    print("\nSTREAM PROCESSOR BENCHMARK")
    print("---------------------")
    print(f"Total duration: {duration:.2f}s")
    print(f"Allocations: {sum(stat.size_diff for stat in stats[:5]):,d} bytes")
    print("Top allocation:")
    for stat in stats[:3]:
        print(f"  {stat.traceback.format()[-1]}: {stat.size_diff:,d} bytes")

    assert duration < 0.15, f"Performance regression: {duration:.2f}s > 0.15s"
    assert sum(stat.size_diff for stat in stats[:5]) < 2_500_000, "Memory regression"
