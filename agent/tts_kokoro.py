"""Kokoro TTS tuned for low latency and correct pronunciation.

- Pre-rendered phrases: Wazir is prompted to open every reply with one short
  phrase from OPENERS ("Sure!", "Got it.", ...). We synthesize those, the
  greeting, and the tool-call filler once at startup, so their audio costs ~0 ms.
- Early first chunk: the first piece of each reply is also cut at a comma once
  it has a few words, so speech starts before a long first sentence is complete.
- Pronunciation: "AGMC" is spelled out and phone numbers / booking references
  are read digit by digit, without the LLM having to format them.
"""

import re
from collections.abc import AsyncGenerator, AsyncIterator

import numpy as np
from loguru import logger
from num2words import num2words

from pipecat.frames.frames import Frame, TTSAudioRawFrame
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.utils.text.base_text_aggregator import Aggregation, TextType
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

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

# Said right before the booking tool runs, to cover the LLM writing the tool call.
TOOL_FILLER = "One moment while I book that."

KOKORO_SAMPLE_RATE = 24000

# (voice, speed) -> {normalized phrase: int16 PCM bytes at 24 kHz}
_phrase_cache: dict[tuple[str, float], dict[str, bytes]] = {}


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z ]+", "", text.lower()).strip()


def _spell_digits(match: re.Match) -> str:
    raw = match.group(0)
    digits = re.sub(r"\D", "", raw)
    if sum(c.isdigit() for c in raw) < 9:
        return raw  # not a phone number
    prefix = "plus, " if raw.startswith("+") else ""
    if len(digits) == 10:
        groups = [digits[:3], digits[3:6], digits[6:]]  # 050 263 5478
    elif len(digits) == 12 and digits.startswith("971"):
        groups = [digits[:3], digits[3:5], digits[5:8], digits[8:]]  # 971 50 263 5478
    else:
        groups = [digits[i : i + 3] for i in range(0, len(digits), 3)]
        if len(groups) > 1 and len(groups[-1]) == 1:
            groups[-2:] = [groups[-2] + groups[-1]]
    return prefix + ", ".join(" ".join(g) for g in groups)


def speakable(text: str) -> str:
    """Rewrite text so Kokoro pronounces it the way a person at AGMC would."""
    # "AGMC" alone is read as one garbled word; hyphens make Kokoro say the letters.
    text = re.sub(r"\bAGMC\b", "A-G-M-C", text)
    # Booking references like TD-45TML: one character at a time.
    text = re.sub(
        r"\b([A-Z]{2})-([A-Z0-9]{4,8})\b",
        lambda m: " ".join(m.group(1)) + ", " + " ".join(m.group(2)),
        text,
    )
    # Phone numbers (9+ digits, any spacing): digit by digit, in groups.
    text = re.sub(r"\+?\d[\d \-]{7,}\d", _spell_digits, text)
    # Model years the way car people say them: 2021 -> "twenty twenty-one".
    text = re.sub(r"\b20([1-3]\d)\b", lambda m: "twenty " + num2words(int(m.group(1))), text)
    return text


class EarlyClauseAggregator(SimpleTextAggregator):
    """Sentence aggregation, except the first chunk of a reply may end at a comma.

    At ~19 tokens/s, waiting for a full 20-word first sentence costs about a
    second before synthesis can even start. Cutting the first chunk at a
    comma once it has MIN_WORDS words gets audio going sooner; later
    sentences keep normal sentence-level prosody.
    """

    MIN_WORDS = 3

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._emitted = False

    async def aggregate(self, text: str) -> AsyncIterator[Aggregation]:
        async for aggregation in super().aggregate(text):
            self._emitted = True
            yield aggregation
        if self._emitted or self._aggregation_type == TextType.TOKEN:
            return
        # Only punctuation followed by a space, so "145,000" is never split.
        cuts = [m.start() for m in re.finditer(r"[,;:]\s", self._text)]
        cut = cuts[-1] if cuts else -1
        if cut > 0 and len(self._text[:cut].split()) >= self.MIN_WORDS:
            clause, self._text = self._text[: cut + 1], self._text[cut + 1 :]
            self._emitted = True
            yield Aggregation(text=clause.strip(), type=TextType.SENTENCE)

    async def flush(self) -> Aggregation | None:
        self._emitted = False
        return await super().flush()

    async def handle_interruption(self):
        self._emitted = False
        await super().handle_interruption()

    async def reset(self):
        self._emitted = False
        await super().reset()


class CachedKokoroTTSService(KokoroTTSService):
    def __init__(self, *, extra_phrases: list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        self._text_aggregator = EarlyClauseAggregator(
            aggregation_type=self._text_aggregation_mode,
            language=self._text_aggregator.language,
        )
        self._extra_phrases = (extra_phrases or []) + [TOOL_FILLER]
        key = (self._settings.voice, float(self._settings.speed))
        if key not in _phrase_cache:
            _phrase_cache[key] = self._render_phrases()
        self._phrases = _phrase_cache[key]

    def _render_phrases(self) -> dict[str, bytes]:
        cache = {}
        # Fixed text (like the greeting) may reach us one sentence at a time.
        sentences = [x for p in self._extra_phrases for x in re.split(r"(?<=[.!?])\s+", p)]
        for phrase in OPENERS + self._extra_phrases + sentences:
            samples, sr = self._kokoro.create(
                speakable(phrase),
                voice=self._settings.voice,
                speed=float(self._settings.speed),
                lang="en-us",
            )
            assert sr == KOKORO_SAMPLE_RATE
            cache[_normalize(phrase)] = (samples * 32767).astype(np.int16).tobytes()
        logger.info(f"Pre-rendered {len(cache)} fixed phrases for voice {self._settings.voice}")
        return cache

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        audio = self._phrases.get(_normalize(text))
        if audio is None:
            async for frame in super().run_tts(speakable(text), context_id):
                yield frame
            return
        await self.stop_ttfb_metrics()
        if self.sample_rate != KOKORO_SAMPLE_RATE:
            audio = await self._resampler.resample(audio, KOKORO_SAMPLE_RATE, self.sample_rate)
        yield TTSAudioRawFrame(
            audio=audio, sample_rate=self.sample_rate, num_channels=1, context_id=context_id
        )
