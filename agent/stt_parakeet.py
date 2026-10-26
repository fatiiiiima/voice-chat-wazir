"""Parakeet TDT 0.6B (English) speech-to-text via onnx-asr, running on CPU.

Pipecat hands us each utterance as soon as the VAD sees the user stop, and we
transcribe it in a worker thread. Parakeet transcribes a few seconds of audio
in well under 200 ms on a modern laptop CPU.
"""

import asyncio
from collections.abc import AsyncGenerator

import numpy as np
from loguru import logger

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

from agent.config import settings
from agent.models import get_stt_model

SAMPLE_RATE = 16000


class ParakeetSTTService(SegmentedSTTService):
    @property
    def wants_wav_segments(self) -> bool:
        return False

    def __init__(self, **kwargs):
        # Parakeet needs far less end padding than Whisper-style models.
        kwargs.setdefault("trailing_silence_secs", 0.2)
        super().__init__(
            sample_rate=SAMPLE_RATE,
            settings=STTSettings(model=settings.stt_model, language=Language.EN),
            **kwargs,
        )
        self._model = get_stt_model()

    def can_generate_metrics(self) -> bool:
        return True

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        await self.start_processing_metrics()
        waveform = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        text = await asyncio.to_thread(self._model.recognize, waveform, sample_rate=SAMPLE_RATE)
        text = (text or "").strip()
        await self.stop_processing_metrics()
        if text:
            logger.debug(f"Transcription: [{text}]")
            yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), Language.EN)
