"""Benchmark the local LLM server with Wazir's real system prompt.

Measures time-to-first-token (TTFT) and decode speed, cold and warm
(warm = the server has the system prompt cached), and at 1/2/4 parallel
requests to see how latency degrades with concurrent callers.

Usage:
  uv run python scripts/bench_llm.py
  uv run python scripts/bench_llm.py --url http://127.0.0.1:8000/v3   # e.g. OpenVINO Model Server
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import settings  # noqa: E402
from agent.prompts import build_system_prompt  # noqa: E402

QUESTIONS = [
    "Hi, do you have any BMW SUVs?",
    "How much is the Land Cruiser?",
    "Can I test drive the X3 tomorrow at five at Motor City?",
    "What's the mileage on the MINI?",
]


async def one_request(client: httpx.AsyncClient, url: str, model: str, question: str) -> dict:
    body = {
        "model": model,
        "stream": True,
        "max_tokens": 60,
        "temperature": 0.4,
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": question},
        ],
    }
    t0 = time.perf_counter()
    first = None
    chunks = 0
    text = ""
    async with client.stream("POST", f"{url}/chat/completions", json=body) as r:
        if r.status_code != 200:
            detail = (await r.aread()).decode(errors="replace")[:500]
            raise SystemExit(f"LLM server returned {r.status_code} for {url}/chat/completions:\n{detail}")
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            delta = json.loads(line[6:])["choices"][0]["delta"].get("content") or ""
            if delta:
                first = first or time.perf_counter()
                chunks += 1
                text += delta
    end = time.perf_counter()
    decode_tps = (chunks - 1) / (end - first) if first and chunks > 1 else 0.0
    return {"ttft_ms": (first - t0) * 1000 if first else None, "tok_s": decode_tps, "text": text.strip()}


async def check_server(client: httpx.AsyncClient, url: str):
    try:
        r = await client.get(f"{url}/models")
    except httpx.ConnectError:
        raise SystemExit(f"Nothing is listening at {url}. Start scripts\\start_llm.ps1 first.")
    print(f"GET {url}/models -> {r.status_code}: {r.text[:300]}\n")
    if r.status_code != 200:
        raise SystemExit(
            "That doesn't look like llama.cpp. Another program may be using port 8080 -\n"
            "run:  netstat -ano | findstr :8080   then   tasklist /FI \"PID eq <pid>\""
        )


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=settings.llm_base_url)
    ap.add_argument("--model", default=settings.llm_model)
    args = ap.parse_args()

    async with httpx.AsyncClient(timeout=120) as client:
        await check_server(client, args.url)
        cold = await one_request(client, args.url, args.model, QUESTIONS[0])
        print(f"cold (first request, prompt not cached): TTFT {cold['ttft_ms']:.0f} ms")
        print(f"  reply: {cold['text']!r}\n")

        # llama.cpp caches per slot; fill every slot once (the agent does the
        # same at startup) so the table shows steady-state latency.
        await asyncio.gather(
            *(one_request(client, args.url, args.model, QUESTIONS[0]) for _ in range(settings.llm_slots))
        )

        print(f"{'parallel':>8} {'TTFT p50':>10} {'TTFT max':>10} {'tok/s each':>11}")
        for n in (1, 2, 4):
            results = []
            for _ in range(3):
                batch = await asyncio.gather(
                    *(one_request(client, args.url, args.model, QUESTIONS[i % len(QUESTIONS)]) for i in range(n))
                )
                results.extend(batch)
            ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"] is not None]
            tps = [r["tok_s"] for r in results if r["tok_s"]]
            print(f"{n:>8} {statistics.median(ttfts):>10.0f} {max(ttfts):>10.0f} {statistics.mean(tps):>11.1f}")

    print("\nSpeech needs only ~4 tokens/s, but TTFT adds directly to response latency.")
    print("Target: warm TTFT under ~250 ms at 1 parallel request.")


if __name__ == "__main__":
    asyncio.run(main())
