"""Summarize logs/turns.jsonl into per-stage p50 / p95 latency.

Usage:
  uv run python scripts/latency_report.py               # all turns
  uv run python scripts/latency_report.py --last 20     # most recent 20 turns
  uv run python scripts/latency_report.py --file other.jsonl

Paste the output back to the developer: it's everything needed to judge latency.
"""

import argparse
import json
from pathlib import Path

STAGES = [
    ("vad_ms", "VAD silence window"),
    ("turn_detect_ms", "Turn detect + STT"),
    ("stt_ms", "  (STT alone)"),
    ("smart_turn_ms", "  (Smart-Turn model)"),
    ("llm_ttft_ms", "LLM first token"),
    ("tts_first_audio_ms", "TTS first audio"),
    ("transport_ms", "Send to WebRTC"),
    ("total_ms", "SERVER TOTAL"),
    ("client_ms", "HEARD IN BROWSER"),
]


def pct(values: list[float], p: float) -> float:
    values = sorted(values)
    k = (len(values) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="logs/turns.jsonl")
    ap.add_argument("--last", type=int, default=0)
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"{path} not found - have a conversation first.")

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    turns = [r for r in records if r.get("event") == "turn"]
    if args.last:
        turns = turns[-args.last :]

    # Attach browser-measured latency to its turn.
    client = {(r["session"], r["turn"]): r["client_ms"] for r in records if r.get("event") == "client_latency"}
    for t in turns:
        t["client_ms"] = client.get((t["session"], t["turn"]))

    barge_ins = sum(1 for r in records if r.get("event") == "barge_in")
    sessions = len({t["session"] for t in turns})
    print(f"{len(turns)} turns across {sessions} sessions, {barge_ins} barge-ins\n")
    print(f"{'stage':<22}{'p50':>8}{'p95':>8}{'max':>8}{'n':>6}")
    print("-" * 52)
    for key, label in STAGES:
        values = [t[key] for t in turns if t.get(key) is not None]
        if not values:
            print(f"{label:<22}{'-':>8}{'-':>8}{'-':>8}{0:>6}")
            continue
        print(f"{label:<22}{pct(values, 50):>8.0f}{pct(values, 95):>8.0f}{max(values):>8.0f}{len(values):>6}")

    totals = [t["total_ms"] for t in turns if t.get("total_ms") is not None]
    if totals:
        under = sum(1 for v in totals if v < 1000)
        print(f"\nServer total under 1 s: {under}/{len(totals)} turns ({100 * under / len(totals):.0f}%)")
    tool_turns = [t for t in turns if t.get("tool")]
    if tool_turns:
        waits = [t["tool_wait_ms"] for t in tool_turns if t.get("tool_wait_ms") is not None]
        if waits:
            print(f"Tool-call turns: {len(tool_turns)}, extra LLM round trip p50 {pct(waits, 50):.0f} ms")


if __name__ == "__main__":
    main()
