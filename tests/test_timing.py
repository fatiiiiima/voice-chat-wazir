import json
import time

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection

from agent.timing import TurnTimingObserver


async def push(observer, frame, at, monkeypatch):
    monkeypatch.setattr(time, "time", lambda: at)
    await observer.on_push_frame(
        FramePushed(source=None, destination=None, frame=frame, direction=FrameDirection.DOWNSTREAM, timestamp=0)
    )


def records(log_dir):
    return [json.loads(line) for line in (log_dir / "turns.jsonl").read_text().splitlines()]


async def test_turn_breakdown_sums_to_total(tmp_path, monkeypatch):
    obs = TurnTimingObserver("s1", tmp_path)
    t0 = 1000.0
    vad_stop = VADUserStoppedSpeakingFrame(stop_secs=0.2, timestamp=t0 + 0.2)
    audio = TTSAudioRawFrame(audio=b"\0\0", sample_rate=24000, num_channels=1)
    steps = [
        (VADUserStartedSpeakingFrame(), t0 - 2.0),
        (vad_stop, t0 + 0.2),
        (TranscriptionFrame("hi there", "u", "ts"), t0 + 0.3),
        (UserStoppedSpeakingFrame(), t0 + 0.35),
        (LLMFullResponseStartFrame(), t0 + 0.36),
        (LLMTextFrame("Sure!"), t0 + 0.55),
        (audio, t0 + 0.60),
        (audio, t0 + 0.65),  # second chunk must not move first-audio time
        (BotStartedSpeakingFrame(), t0 + 0.62),
    ]
    for frame, at in steps:
        await push(obs, frame, at, monkeypatch)

    (rec,) = records(tmp_path)
    assert rec["user_text"] == "hi there"
    assert rec["vad_ms"] == 200
    assert rec["stt_ms"] == 100
    assert rec["turn_detect_ms"] == 150
    assert rec["llm_ttft_ms"] == 200
    assert rec["tts_first_audio_ms"] == 50
    assert rec["transport_ms"] == 20
    assert rec["total_ms"] == 620
    parts = ["vad_ms", "turn_detect_ms", "llm_ttft_ms", "tts_first_audio_ms", "transport_ms"]
    assert sum(rec[p] for p in parts) == rec["total_ms"]


async def test_greeting_is_not_logged_and_barge_in_is(tmp_path, monkeypatch):
    obs = TurnTimingObserver("s1", tmp_path)
    # Greeting: bot speaks with no user turn -> nothing logged.
    await push(obs, BotStartedSpeakingFrame(), 1.0, monkeypatch)
    await push(obs, BotStoppedSpeakingFrame(), 3.0, monkeypatch)
    # User talks while bot is silent: interruption frame is not a barge-in.
    await push(obs, VADUserStartedSpeakingFrame(), 4.0, monkeypatch)
    await push(obs, InterruptionFrame(), 4.0, monkeypatch)
    assert not (tmp_path / "turns.jsonl").exists()

    await push(obs, VADUserStoppedSpeakingFrame(stop_secs=0.2, timestamp=5.2), 5.2, monkeypatch)
    await push(obs, UserStoppedSpeakingFrame(), 5.3, monkeypatch)
    await push(obs, LLMFullResponseStartFrame(), 5.3, monkeypatch)
    await push(obs, LLMTextFrame("Sure!"), 5.5, monkeypatch)
    await push(obs, BotStartedSpeakingFrame(), 5.6, monkeypatch)
    # User interrupts while bot is talking.
    await push(obs, VADUserStartedSpeakingFrame(), 6.6, monkeypatch)
    await push(obs, InterruptionFrame(), 6.6, monkeypatch)

    turn, barge = records(tmp_path)
    assert turn["event"] == "turn" and turn["total_ms"] == 600
    assert barge["event"] == "barge_in" and barge["turn"] == 1
    assert barge["after_bot_start_ms"] == 1000
