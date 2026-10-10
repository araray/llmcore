"""gpu.ai's /usage can serve the same cursor and page again (seen live 2026-10-10):
the report must count it once, not 200 times."""

import asyncio

from llmcore.runtimes.gpuai import GpuAiRuntime


class _Repeating:
    def __init__(self):
        self.calls = 0

    async def __call__(self, method, path, params=None, **kw):
        self.calls += 1
        return {"data": [{"bucket_start": "2026-10-10T00:00:00Z", "instance_id": "gpu-1",
                          "gpu_seconds": 990, "cost_cents": 11}],
                "next_cursor": "same"}


def test_a_repeated_page_is_counted_once():
    rt = GpuAiRuntime(api_key="x")
    fake = _Repeating()
    rt._call = fake
    rep = asyncio.run(rt.usage())
    assert rep.total_cost_usd == 0.11
    assert rep.by_instance() == {"gpu-1": 0.11}
    assert fake.calls <= 2
