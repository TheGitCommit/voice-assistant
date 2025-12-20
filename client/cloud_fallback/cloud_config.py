import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


@dataclass(frozen=True)
class DeepgramConfig:
    api_key: str
    stt_model: str = "nova-2"
    stt_language: str = "en-US"
    tts_model: str = "aura-asteria-en"
    tts_encoding: str = "linear16"
    tts_sample_rate: int = 24000


@dataclass(frozen=True)
class DeepSeekConfig:
    api_key: str
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    max_tokens: int = 1000
    temperature: float = 0.7


@dataclass(frozen=True)
class CloudConfig:
    deepgram: DeepgramConfig
    deepseek: DeepSeekConfig
    enable_fallback: bool = True
    server_health_check_timeout: float = 5.0
    api_request_timeout: float = 30.0


CLOUD_CONFIG = CloudConfig(
    deepgram=DeepgramConfig(
        api_key=os.getenv("DEEPGRAM_API_KEY", ""),
    ),
    deepseek=DeepSeekConfig(
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
    ),
    enable_fallback=os.getenv("ENABLE_CLOUD_FALLBACK", "true").lower() == "true",
)


def is_cloud_configured() -> bool:
    return bool(
        CLOUD_CONFIG.deepgram.api_key
        and CLOUD_CONFIG.deepseek.api_key
        and CLOUD_CONFIG.enable_fallback
    )
