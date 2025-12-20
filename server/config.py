import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


@dataclass(frozen=True)
class LlamaConfig:
    exe_path: str
    model_path: str

    host: str = "0.0.0.0"
    port: int = 8080

    gpu_layers: int = -1
    context_size: int = 8192

    max_history_turns: int = 10

    threads: int = 12
    threads_batch: int = 12
    batch_size: int = 2048
    ubatch_size: int = 512
    parallel: int = 1

    mlock: bool = True
    no_mmap: bool = True

    startup_delay_seconds: float = 10.0
    request_timeout_seconds: float = 60.0

    @property
    def endpoint_url(self) -> str:
        return f"http://localhost:{self.port}/v1/chat/completions"


@dataclass(frozen=True)
class PiperConfig:
    exe_path: str
    model_path: str

    @property
    def model_config_path(self) -> str:
        return f"{self.model_path}.json"


@dataclass(frozen=True)
class WhisperConfig:
    model_size: str = "small.en"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 5


@dataclass(frozen=True)
class VADConfig:
    speech_threshold: float = 0.45
    silence_threshold: float = 0.35
    silence_frames_required: int = 10  # ~320ms at 32ms/frame
    min_utterance_seconds: float = 0.5
    max_utterance_seconds: float = 12.0
    noise_buffer_clear_seconds: float = 1.0


@dataclass(frozen=True)
class WebSocketConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    audio_queue_maxsize: int = 200
    event_queue_maxsize: int = 200

    heartbeat_interval_seconds: float = 30.0


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "float32"
    bytes_per_sample: int = 4


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    rate_limit_seconds: float = 5.0


CONFIG = {
    "llama": LlamaConfig(
        exe_path=os.getenv("LLAMA_EXE_PATH", r"path_to_server.exe"),
        model_path=os.getenv("LLAMA_MODEL_PATH", r"path_to_model"),
    ),
    "piper": PiperConfig(
        exe_path=os.getenv("PIPER_EXE_PATH", r".\server\piper\piper.exe"),
        model_path=os.getenv(
            "PIPER_MODEL_PATH", r".\server\piper\models\en_US-amy-medium.onnx"
        ),
    ),
    "whisper": WhisperConfig(),
    "vad": VADConfig(),
    "audio": AudioConfig(),
    "websocket": WebSocketConfig(),
    "logging": LoggingConfig(),
}
