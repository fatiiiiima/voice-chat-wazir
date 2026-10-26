"""Settings loaded from environment variables (see .env.example)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)


def _get(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    llm_base_url: str = _get("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
    llm_model: str = _get("LLM_MODEL", "qwen3-4b-instruct")
    llm_api_key: str = _get("LLM_API_KEY", "local")
    llm_temperature: float = float(_get("LLM_TEMPERATURE", "0.4"))
    llm_max_tokens: int = int(_get("LLM_MAX_TOKENS", "200"))

    stt_model: str = _get("STT_MODEL", "nemo-parakeet-tdt-0.6b-v2")
    stt_quantization: str = _get("STT_QUANTIZATION", "int8")

    tts_voice: str = _get("TTS_VOICE", "af_heart")
    tts_speed: float = float(_get("TTS_SPEED", "1.05"))

    # Cap ONNX Runtime threads per model so one session's TTS can't starve
    # another session's turn detection. 0 = auto (fraction of CPU cores).
    tts_threads: int = int(_get("TTS_THREADS", "0"))
    stt_threads: int = int(_get("STT_THREADS", "0"))

    vad_stop_secs: float = float(_get("VAD_STOP_SECS", "0.2"))

    host: str = _get("HOST", "127.0.0.1")
    port: int = int(_get("PORT", "7860"))
    log_dir: Path = Path(_get("LOG_DIR", "logs"))


settings = Settings()
