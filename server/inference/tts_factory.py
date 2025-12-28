"""TTS Factory for seamless provider switching.

Usage:
    from server.inference.tts_factory import create_tts
    from server.config import CONFIG
    
    tts = create_tts(CONFIG)  # Returns appropriate TTS based on config
    audio = await tts.synthesize("Hello world")
    
To switch providers, set TTS_PROVIDER environment variable:
    TTS_PROVIDER=piper   # Use Piper TTS (default)
    TTS_PROVIDER=kokoro  # Use Kokoro-82M TTS
"""

import logging
from typing import Union

from server.config import CONFIG, PiperConfig, KokoroConfig, TTSConfig
from server.inference.tts_base import BaseTTS

logger = logging.getLogger(__name__)


def create_tts(config: dict = None) -> BaseTTS:
    """Create TTS instance based on configuration.

    Args:
        config: Configuration dict (defaults to CONFIG if None)

    Returns:
        BaseTTS implementation (PiperTTS or KokoroTTS)

    Raises:
        ValueError: If provider is unknown
        ImportError: If provider dependencies are missing
    """
    if config is None:
        config = CONFIG

    tts_config: TTSConfig = config.get("tts", TTSConfig())
    provider = tts_config.provider.lower()

    logger.info("Creating TTS provider: %s", provider)

    if provider == "piper":
        from server.inference.piper_tts import PiperTTS

        piper_config: PiperConfig = config["piper"]
        return PiperTTS(piper_config)

    elif provider == "kokoro":
        from server.inference.kokoro_tts import KokoroTTS

        kokoro_config: KokoroConfig = config["kokoro"]
        return KokoroTTS(kokoro_config)

    else:
        available = ["piper", "kokoro"]
        raise ValueError(f"Unknown TTS provider: '{provider}'. Available: {available}")
