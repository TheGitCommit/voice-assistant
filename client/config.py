import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


@dataclass(frozen=True)
class ServerConfig:
    """Server connection configuration."""

    url: str


@dataclass(frozen=True)
class AudioCaptureConfig:
    """Microphone capture configuration."""

    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 320  # 20ms at 16kHz
    queue_maxsize: int = 20
    dtype: str = "float32"


@dataclass(frozen=True)
class AudioPlaybackConfig:
    """Speaker playback configuration."""

    sample_rate: int = 22050  # Piper TTS default output rate
    channels: int = 1
    dtype: str = "int16"  # Server sends PCM16


@dataclass(frozen=True)
class VADConfig:
    """Voice Activity Detection configuration."""

    # WebRTC VAD aggressiveness: 0 (least) to 3 (most aggressive)
    aggressiveness: int = 2
    frame_duration_ms: int = 20
    silence_limit_frames: int = 10  # ~200ms of silence ends utterance


@dataclass(frozen=True)
class ClientConfig:
    """Complete client configuration."""

    server: ServerConfig
    capture: AudioCaptureConfig
    playback: AudioPlaybackConfig
    vad: VADConfig


DEFAULT_CONFIG = ClientConfig(
    server=ServerConfig(
        url=os.getenv("SERVER_URL", "ws://localhost:8000/ws/audio"),
    ),
    capture=AudioCaptureConfig(),
    playback=AudioPlaybackConfig(),
    vad=VADConfig(),
)