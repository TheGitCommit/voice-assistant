import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (parent of client/)
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)


@dataclass(frozen=True)
class ServerConfig:
    url: str


@dataclass(frozen=True)
class AudioCaptureConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 320  # 20ms at 16kHz
    queue_maxsize: int = 20
    dtype: str = "float32"


@dataclass(frozen=True)
class AudioPlaybackConfig:
    sample_rate: int = 24000  # Kokoro TTS default (Piper uses 22050)
    channels: int = 1
    dtype: str = "int16"  # Server sends PCM16


@dataclass(frozen=True)
class VADConfig:
    aggressiveness: int = 2  # 0=least, 3=most aggressive
    frame_duration_ms: int = 20
    silence_limit_frames: int = 10  # ~200ms of silence ends utterance


@dataclass(frozen=True)
class WakeWordConfig:
    model_name: str = "alexa"  # Pre-trained wake word model name
    threshold: float = 0.5  # Detection confidence threshold (0.0 to 1.0)
    activation_delay_ms: int = 500  # Grace period after wake word detection


@dataclass(frozen=True)
class ClientConfig:
    server: ServerConfig
    capture: AudioCaptureConfig
    playback: AudioPlaybackConfig
    vad: VADConfig
    wake_word: WakeWordConfig


DEFAULT_CONFIG = ClientConfig(
    server=ServerConfig(
        url=os.getenv("SERVER_URL", "ws://localhost:8000/ws/audio"),
    ),
    capture=AudioCaptureConfig(),
    playback=AudioPlaybackConfig(),
    vad=VADConfig(),
    wake_word=WakeWordConfig(),
)
