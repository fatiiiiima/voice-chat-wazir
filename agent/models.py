"""Load the local speech models once per process and share them across sessions.

Loading per session would add seconds to every connection and multiply RAM use.
ONNX Runtime sessions are safe to call from several threads at once.
"""

import os
import time
from functools import lru_cache

import onnxruntime as ort

from loguru import logger

import pipecat.services.kokoro.tts as kokoro_tts_module
from agent.config import settings

_original_kokoro = kokoro_tts_module.Kokoro


def _session_options(threads: int, share_of_cores: float) -> ort.SessionOptions:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads or max(2, int((os.cpu_count() or 4) * share_of_cores))
    # By default idle ORT threads busy-spin, burning CPU that other sessions'
    # VAD / turn detection / TTS need. Sleep instead.
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return opts


@lru_cache(maxsize=2)
def _shared_kokoro(model_path: str, voices_path: str):
    t0 = time.perf_counter()
    opts = _session_options(settings.tts_threads, 0.4)
    session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
    model = _original_kokoro.from_session(session, voices_path)
    logger.info(
        f"Kokoro TTS loaded in {time.perf_counter() - t0:.1f}s ({opts.intra_op_num_threads} threads)"
    )
    return model


# KokoroTTSService builds a new Kokoro() in its __init__; route that through the cache.
kokoro_tts_module.Kokoro = _shared_kokoro


@lru_cache(maxsize=1)
def get_stt_model():
    import onnx_asr

    t0 = time.perf_counter()
    opts = _session_options(settings.stt_threads, 0.25)
    model = onnx_asr.load_model(
        settings.stt_model,
        quantization=settings.stt_quantization or None,
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    logger.info(
        f"STT model {settings.stt_model} loaded in {time.perf_counter() - t0:.1f}s "
        f"({opts.intra_op_num_threads} threads)"
    )
    return model


def preload() -> None:
    """Load (and on first run, download) every model before accepting calls."""
    from agent.bot import make_tts

    get_stt_model()
    make_tts()  # constructs the shared Kokoro instance
