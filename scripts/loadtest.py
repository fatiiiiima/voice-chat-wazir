"""Simulate N callers talking to Wazir over real WebRTC and measure latency.

Each simulated caller:
  1. connects like the browser does (WebRTC + data channel),
  2. waits for the greeting to finish,
  3. speaks each utterance (synthesized with a male Kokoro voice),
  4. measures "heard latency": end of its own speech -> first agent audio received,
  5. waits for the agent to finish, then says the next utterance.

Usage:
  uv run python scripts/loadtest.py                    # 1 caller
  uv run python scripts/loadtest.py --sessions 4       # 4 callers at once
  uv run python scripts/loadtest.py --sessions 4 --stagger 0.5

"Heard" here skips the browser's jitter buffer, so real browsers add ~20-60 ms.
"""

import argparse
import asyncio
import fractions
import json
import statistics
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from av import AudioFrame

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RATE = 48000
FRAME = 960  # 20 ms
DEFAULT_UTTERANCES = [
    "Hi, do you have any BMW SUVs?",
    "What's the mileage on the X3?",
    "Can I test drive it on Tuesday at five pm at Motor City?",
]


def synthesize(texts: list[str], voice: str) -> list[np.ndarray]:
    """Kokoro -> int16 mono @ 48 kHz."""
    from kokoro_onnx import Kokoro
    from pipecat.services.kokoro.tts import KOKORO_CACHE_DIR, _ensure_model_files

    model, voices = KOKORO_CACHE_DIR / "kokoro-v1.0.onnx", KOKORO_CACHE_DIR / "voices-v1.0.bin"
    _ensure_model_files(model, voices)
    kokoro = Kokoro(str(model), str(voices))
    out = []
    for text in texts:
        samples, sr = kokoro.create(text, voice=voice, lang="en-us")
        idx = np.arange(0, len(samples), sr / RATE)  # simple linear resample to 48 kHz
        resampled = np.interp(idx, np.arange(len(samples)), samples)
        out.append((resampled * 32767).astype(np.int16))
    return out


class MicTrack(MediaStreamTrack):
    """Outgoing audio: silence, except while an utterance is playing."""

    kind = "audio"

    def __init__(self):
        super().__init__()
        self._pts = 0
        self._start = None
        self._pending: np.ndarray | None = None
        self._pos = 0
        self.speech_done = asyncio.Event()
        self.speech_end_time = 0.0

    def say(self, samples: np.ndarray):
        self._pending, self._pos = samples, 0
        self.speech_done.clear()

    async def recv(self):
        if self._start is None:
            self._start = time.time()
        wait = self._start + self._pts / RATE - time.time()
        if wait > 0:
            await asyncio.sleep(wait)
        chunk = np.zeros(FRAME, dtype=np.int16)
        if self._pending is not None:
            part = self._pending[self._pos : self._pos + FRAME]
            chunk[: len(part)] = part
            self._pos += FRAME
            if self._pos >= len(self._pending):
                self._pending = None
                self.speech_end_time = time.time()
                self.speech_done.set()
        frame = AudioFrame.from_ndarray(chunk.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = RATE
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, RATE)
        self._pts += FRAME
        return frame


class Caller:
    def __init__(self, idx: int, url: str, utterances: list[np.ndarray]):
        self.idx = idx
        self.url = url
        self.utterances = utterances
        self.mic = MicTrack()
        self.pc = RTCPeerConnection()
        self.loud = False
        self.last_loud = time.time()
        self.onset = asyncio.Event()
        self.onset_time = 0.0
        self.listen_from = 0.0
        self.results: list[dict] = []
        self.server_turns: list[dict] = []

    async def _consume(self, track):
        while True:
            try:
                frame = await track.recv()
            except Exception:
                return
            pcm = frame.to_ndarray().astype(np.float32) / 32768.0
            level = float(np.sqrt(np.mean(pcm**2)))
            now = time.time()
            if level > 0.005:
                if not self.loud and now - self.last_loud > 0.3 and now >= self.listen_from:
                    self.onset_time = now
                    self.onset.set()
                self.loud = True
                self.last_loud = now
            else:
                self.loud = False

    async def _wait_agent_quiet(self, quiet_secs=2.5, timeout=60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if time.time() - self.last_loud > quiet_secs:
                return
            await asyncio.sleep(0.05)

    async def run(self):
        self.pc.addTransceiver(self.mic, direction="sendrecv")
        self.pc.addTransceiver("video", direction="sendrecv")
        dc = self.pc.createDataChannel("chat", ordered=True)

        @dc.on("message")
        def on_message(msg):
            if isinstance(msg, str) and msg.startswith("{"):
                m = json.loads(msg)
                if m.get("type") == "wazir-turn":
                    self.server_turns.append(m)

        @self.pc.on("track")
        def on_track(track):
            if track.kind == "audio":
                asyncio.ensure_future(self._consume(track))

        await self.pc.setLocalDescription(await self.pc.createOffer())
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"{self.url}/api/offer",
                json={"sdp": self.pc.localDescription.sdp, "type": self.pc.localDescription.type},
            )
            r.raise_for_status()
            answer = r.json()
        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))

        # Greeting
        await asyncio.wait_for(self.onset.wait(), timeout=30)
        await self._wait_agent_quiet()

        for i, samples in enumerate(self.utterances):
            self.listen_from = float("inf")
            self.mic.say(samples)
            await self.mic.speech_done.wait()
            # Only agent audio that starts after we stop talking counts as the reply.
            self.onset.clear()
            self.listen_from = self.mic.speech_end_time
            try:
                await asyncio.wait_for(self.onset.wait(), timeout=15)
                heard = round((self.onset_time - self.mic.speech_end_time) * 1000)
            except asyncio.TimeoutError:
                heard = None
            self.results.append({"caller": self.idx, "utterance": i, "heard_ms": heard})
            print(f"caller {self.idx} utterance {i}: heard {heard} ms", flush=True)
            await self._wait_agent_quiet()

        await asyncio.sleep(0.5)
        await self.pc.close()


def pct(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7860")
    ap.add_argument("--sessions", type=int, default=1)
    ap.add_argument("--stagger", type=float, default=1.0, help="seconds between caller starts")
    ap.add_argument("--voice", default="am_adam")
    ap.add_argument("--utterance", action="append", help="repeatable; defaults to a short booking chat")
    args = ap.parse_args()

    texts = args.utterance or DEFAULT_UTTERANCES
    print(f"Synthesizing {len(texts)} utterances...", flush=True)
    audio = synthesize(texts, args.voice)

    callers = [Caller(i, args.url, audio) for i in range(args.sessions)]

    async def start(c: Caller, delay: float):
        await asyncio.sleep(delay)
        try:
            await c.run()
        except Exception as e:
            print(f"caller {c.idx} failed: {e!r}", flush=True)

    await asyncio.gather(*(start(c, i * args.stagger) for i, c in enumerate(callers)))

    heard = [r["heard_ms"] for c in callers for r in c.results if r["heard_ms"] is not None]
    missed = sum(1 for c in callers for r in c.results if r["heard_ms"] is None)
    server = [t["total_ms"] for c in callers for t in c.server_turns if t.get("total_ms") is not None]
    print("\n=== Summary ===")
    print(f"sessions={args.sessions} turns={len(heard)} no_reply={missed}")
    if heard:
        print(f"heard latency  p50={pct(heard, 50)} ms  p95={pct(heard, 95)} ms  max={max(heard)} ms  mean={statistics.mean(heard):.0f} ms")
    if server:
        print(f"server total   p50={pct(server, 50)} ms  p95={pct(server, 95)} ms")
    print("Per-stage breakdown: uv run python scripts/latency_report.py")


if __name__ == "__main__":
    asyncio.run(main())
