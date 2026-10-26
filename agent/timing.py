"""Per-turn latency logging.

Watches frames flowing through the pipeline and, for every user turn, records
when each stage happened. All durations are measured from the moment the user
actually stopped talking (VAD stop time minus the VAD silence window), which
is what the user perceives as "waiting for a reply".

    speech_end ─ vad ─► vad_stop ─ turn ─► turn_end ─ llm ─► first token ─ tts ─► first audio ─ out ─► bot speaking
                         (STT runs in parallel with turn detection inside "turn")

Each finished turn is appended as one JSON line to logs/turns.jsonl and sent
to the browser so the widget can show it.
"""

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    FunctionCallInProgressFrame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSTextFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TurnMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed

SendFn = Callable[[dict[str, Any]], Awaitable[None]]


def _ms(a: float | None, b: float | None) -> int | None:
    if a is None or b is None:
        return None
    return round((b - a) * 1000)


class TurnTimingObserver(BaseObserver):
    def __init__(self, session_id: str, log_dir: Path, send: SendFn | None = None):
        super().__init__()
        self._session_id = session_id
        self._log_file = log_dir / "turns.jsonl"
        self._log_file.parent.mkdir(parents=True, exist_ok=True)
        self._send = send
        self._turn_no = 0
        self._turn = self._new_turn()
        self._bot_speaking = False
        self._last_logged_turn: int | None = None
        self._last_bot_started: float | None = None
        self._last_barge_in = 0.0
        self._seen: set[int] = set()

    @staticmethod
    def _new_turn() -> dict[str, Any]:
        return {"text": []}

    # ------------------------------------------------------------------ frames

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        if not isinstance(
            frame,
            (
                VADUserStartedSpeakingFrame,
                VADUserStoppedSpeakingFrame,
                TranscriptionFrame,
                UserStoppedSpeakingFrame,
                LLMFullResponseStartFrame,
                LLMTextFrame,
                FunctionCallInProgressFrame,
                TTSAudioRawFrame,
                TTSTextFrame,
                BotStartedSpeakingFrame,
                BotStoppedSpeakingFrame,
                InterruptionFrame,
                MetricsFrame,
            ),
        ):
            return
        # A frame is observed once per hop; only the first sighting matters.
        if frame.id in self._seen:
            return
        self._seen.add(frame.id)
        if len(self._seen) > 5000:
            self._seen.clear()

        now = time.time()
        t = self._turn

        if isinstance(frame, VADUserStartedSpeakingFrame):
            if t.get("turn_end") and not t.get("bot_started"):
                # User spoke again before we answered: the pending reply is dropped.
                self._write({"event": "turn_superseded", "turn": self._turn_no})
                self._turn = t = self._new_turn()
            t.setdefault("speech_start", now)

        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            t["vad_stop"] = now
            t["vad_stop_secs"] = frame.stop_secs
            t["speech_end"] = frame.timestamp - frame.stop_secs

        elif isinstance(frame, TranscriptionFrame):
            t["stt_final"] = now
            t["text"].append(frame.text)
            self._emit({"type": "wazir-user", "text": frame.text})

        elif isinstance(frame, MetricsFrame):
            for d in frame.data:
                if isinstance(d, TurnMetricsData):
                    t["smart_turn_ms"] = round(d.e2e_processing_time_ms)

        elif isinstance(frame, UserStoppedSpeakingFrame):
            if t.get("speech_end") is not None:
                t["turn_end"] = now

        elif isinstance(frame, LLMFullResponseStartFrame):
            if t.get("turn_end"):
                t.setdefault("llm_start", now)

        elif isinstance(frame, LLMTextFrame):
            if t.get("llm_start"):
                t.setdefault("llm_first_token", now)
                if len(t.setdefault("reply", "")) < 80:
                    t["reply"] += frame.text

        elif isinstance(frame, FunctionCallInProgressFrame):
            t["tool"] = frame.function_name
            t.setdefault("tool_start", now)

        elif isinstance(frame, TTSAudioRawFrame):
            if t.get("llm_first_token"):
                t.setdefault("tts_first_audio", now)

        elif isinstance(frame, TTSTextFrame):
            self._emit({"type": "wazir-bot", "text": frame.text})

        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            self._last_bot_started = now
            if t.get("turn_end") and not t.get("bot_started"):
                t["bot_started"] = now
                self._finish_turn()

        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False

        elif isinstance(frame, InterruptionFrame):
            # One interruption is broadcast as two frames (upstream + downstream).
            recent = now - self._last_barge_in < 0.5
            if self._bot_speaking and self._last_logged_turn is not None and not recent:
                self._last_barge_in = now
                self._write(
                    {
                        "event": "barge_in",
                        "turn": self._last_logged_turn,
                        "after_bot_start_ms": _ms(self._last_bot_started, now),
                    }
                )
                self._emit({"type": "wazir-barge-in"})

    # ----------------------------------------------------------------- results

    def _finish_turn(self):
        t = self._turn
        self._turn_no += 1
        speech_end = t["speech_end"]
        record = {
            "event": "turn",
            "turn": self._turn_no,
            "user_text": " ".join(t["text"]),
            # Additive stages: these sum to total_ms.
            "vad_ms": _ms(speech_end, t.get("vad_stop")),
            "turn_detect_ms": _ms(t.get("vad_stop"), t.get("turn_end")),
            "llm_ttft_ms": _ms(t.get("turn_end"), t.get("llm_first_token")),
            "tts_first_audio_ms": _ms(t.get("llm_first_token"), t.get("tts_first_audio")),
            "transport_ms": _ms(t.get("tts_first_audio"), t.get("bot_started")),
            "total_ms": _ms(speech_end, t.get("bot_started")),
            # Informational: STT and the Smart-Turn model both run inside turn_detect_ms.
            "stt_ms": _ms(t.get("vad_stop"), t.get("stt_final")),
            "smart_turn_ms": t.get("smart_turn_ms"),
            # Start of the reply: a short opener ("Sure! ...") means fast first audio.
            "reply_start": (t.get("reply") or "").strip()[:80],
            "tool": t.get("tool"),
            "tool_wait_ms": _ms(t.get("tool_start"), t.get("llm_first_token")) if t.get("tool") else None,
        }
        self._last_logged_turn = self._turn_no
        self._write(record)
        self._emit({"type": "wazir-turn", **record})
        logger.info(
            f"[turn {self._turn_no}] total={record['total_ms']}ms "
            f"vad={record['vad_ms']} turn={record['turn_detect_ms']} "
            f"(stt={record['stt_ms']} smart_turn={record['smart_turn_ms']}) "
            f"llm_ttft={record['llm_ttft_ms']} tts={record['tts_first_audio_ms']} "
            f"out={record['transport_ms']} tool={record['tool']}"
        )
        self._turn = self._new_turn()

    def add_client_latency(self, client_ms: float, turn: int | None = None):
        """Latency as heard in the browser (server total + playout + network)."""
        turn = turn if turn is not None else self._last_logged_turn
        self._write({"event": "client_latency", "turn": turn, "client_ms": round(client_ms)})

    def _write(self, record: dict[str, Any]):
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "session": self._session_id,
            **record,
        }
        with self._log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _emit(self, message: dict[str, Any]):
        if self._send:
            # Don't block frame delivery on the data channel.
            asyncio.get_running_loop().create_task(self._send(message))
