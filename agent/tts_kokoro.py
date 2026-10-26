"""Kokoro TTS with pre-rendered opening phrases.

Wazir is prompted to open every reply with one short phrase from OPENERS
("Sure!", "Got it.", ...). We synthesize those (and the fixed greeting) once
at startup, so the first audio of most replies costs ~0 ms instead of a full
Kokoro inference.
The rest of the reply is synthesized normally while the opener plays.
"""

import re
from collections.abc import AsyncGenerator

import numpy as np
from loguru import logger

from pipecat.frames.frames import Frame, TTSAudioRawFrame
from pipecat.services.kokoro.tts import KokoroTTSService

OPENERS = [
    "Sure!",
    "Of course!",
    "Got it.",
    "Great choice.",
    "Absolutely!",
    "Perfect.",
    "No problem.",
    "Okay.",
    "Great question.",
    "Thank you!",
]

KOKORO_SAMPLE_RATE = 24000

# (voice, speed) -> {normalized phrase: int16 PCM bytes at 24 kHz}
_phrase_cache: dict[tuple[str, float], dict[str, bytes]] = {}


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z ]+", "", text.lower()).strip()


class CachedKokoroTTSService(KokoroTTSService):
    def __init__(self, *, extra_phrases: list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        self._extra_phrases = extra_phrases or []
        key = (self._settings.voice, float(self._settings.speed))
        if key not in _phrase_cache:
            _phrase_cache[key] = self._render_openers()
        self._phrases = _phrase_cache[key]

    def _render_openers(self) -> dict[str, bytes]:
        cache = {}
        # Fixed text (like the greeting) may reach us one sentence at a time.
        sentences = [x for p in self._extra_phrases for x in re.split(r"(?<=[.!?])\s+", p)]
        for phrase in OPENERS + self._extra_phrases + sentences:
            samples, sr = self._kokoro.create(
                phrase, voice=self._settings.voice, speed=float(self._settings.speed), lang="en-us"
            )
            assert sr == KOKORO_SAMPLE_RATE
            cache[_normalize(phrase)] = (samples * 32767).astype(np.int16).tobytes()
        logger.info(f"Pre-rendered {len(cache)} fixed phrases for voice {self._settings.voice}")
        return cache

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        audio = self._phrases.get(_normalize(text))
        if audio is None:
            async for frame in super().run_tts(text, context_id):
                yield frame
            return
        await self.stop_ttfb_metrics()
        if self.sample_rate != KOKORO_SAMPLE_RATE:
            audio = await self._resampler.resample(audio, KOKORO_SAMPLE_RATE, self.sample_rate)
        yield TTSAudioRawFrame(
            audio=audio, sample_rate=self.sample_rate, num_channels=1, context_id=context_id
        )
