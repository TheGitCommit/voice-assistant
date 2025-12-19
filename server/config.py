"""
Centralized configuration for the voice assistant server.
All paths, ports, and tunable parameters live here.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LlamaConfig:
    """Configuration for the Llama.cpp server."""

    exe_path: str
    model_path: str
    host: str = "0.0.0.0"
    port: int = 8080
    gpu_layers: int = 99
    context_size: int = 4096
    startup_delay_seconds: float = 10.0
    request_timeout_seconds: float = 60.0

    @property
    def endpoint_url(self) -> str:
        return f"http://localhost:{self.port}/v1/chat/completions"


@dataclass(frozen=True)
class PiperConfig:
    """Configuration for Piper TTS."""

    exe_path: str
    model_path: str

    @property
    def model_config_path(self) -> str:
        return f"{self.model_path}.json"


@dataclass(frozen=True)
class WhisperConfig:
    """Configuration for Whisper STT."""

    model_size: str = "small.en"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 5


@dataclass(frozen=True)
class VADConfig:
    """Voice Activity Detection parameters."""

    # Energy thresholds (RMS amplitude)
    silence_threshold: float = 0.001
    speech_threshold: float = 0.002

    # Timing parameters
    silence_frames_required: int = 15  # ~300ms at 20ms frames
    min_utterance_seconds: float = 0.5
    max_utterance_seconds: float = 10.0
    noise_buffer_clear_seconds: float = 1.0


@dataclass(frozen=True)
class AudioConfig:
    """Audio processing parameters."""

    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "float32"  # numpy dtype string
    bytes_per_sample: int = 4  # float32


@dataclass(frozen=True)
class WebSocketConfig:
    """WebSocket server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    audio_queue_maxsize: int = 100
    event_queue_maxsize: int = 50
    heartbeat_interval_seconds: float = 30.0


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    rate_limit_seconds: float = 5.0


CONFIG = {
    "llama": LlamaConfig(
        exe_path=r"path_to_server.exe",
        model_path=r"path_to_model",
    ),
    "piper": PiperConfig(
        exe_path=r"/\server\piper\piper.exe",
        model_path=r"/\server\piper\models\en_US-amy-medium.onnx",
    ),
    "whisper": WhisperConfig(),
    "vad": VADConfig(),
    "audio": AudioConfig(),
    "websocket": WebSocketConfig(),
    "logging": LoggingConfig(),
}
