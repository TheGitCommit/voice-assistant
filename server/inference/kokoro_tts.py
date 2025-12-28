"""Kokoro-82M TTS implementation using the official kokoro library.

Requires: 
    pip install kokoro>=0.9.4 soundfile
    Windows: Install espeak-ng from https://github.com/espeak-ng/espeak-ng/releases

Reference: https://huggingface.co/hexgrad/Kokoro-82M
"""

import asyncio
import logging
import time
import os
from typing import Optional
import numpy as np

from server.config import KokoroConfig
from server.inference.tts_base import BaseTTS

try:
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    if not hasattr(EspeakWrapper, "set_data_path"):

        def _dummy_set_data_path(path):
            pass

        EspeakWrapper.set_data_path = _dummy_set_data_path
except ImportError:
    pass

logger = logging.getLogger(__name__)


class KokoroTTS(BaseTTS):
    """Kokoro-82M TTS using the official kokoro library.

    Features:
    - High quality neural TTS (82M parameters)
    - Multiple voices and languages
    - 24kHz output sample rate
    - Streaming generation support

    Voices (American English 'a'):
        af_heart, af_alloy, af_aoede, af_bella, af_jessica, af_kore, af_nicole,
        af_nova, af_river, af_sarah, af_sky, am_adam, am_echo, am_eric, am_fenrir,
        am_liam, am_michael, am_onyx, am_puck, am_santa

    See: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
    """

    _shared_pipeline = None

    def __init__(self, config: KokoroConfig):
        self.config = config

        logger.info(
            "KokoroTTS configured: voice=%s speed=%.1f",
            config.voice,
            config.speed,
        )

    @classmethod
    async def initialize(cls, config: KokoroConfig) -> None:
        """Initialize the shared Kokoro pipeline at startup.

        This should be called once at server startup before accepting connections.
        The pipeline is shared across all KokoroTTS instances.

        Args:
            config: KokoroConfig with model settings
        """
        if cls._shared_pipeline is not None:
            logger.info("Kokoro pipeline already initialized")
            return

        try:
            logger.info("Initializing Kokoro TTS pipeline...")
            start = time.perf_counter()

            # Import kokoro
            try:
                from kokoro import KPipeline
            except ImportError as e:
                raise ImportError(
                    "kokoro not installed. Install with: pip install kokoro>=0.9.4\n"
                    "Also install espeak-ng: https://github.com/espeak-ng/espeak-ng/releases"
                ) from e

            # Initialize pipeline in thread pool
            # lang_code: 'a' = American English, 'b' = British English
            loop = asyncio.get_event_loop()
            cls._shared_pipeline = await loop.run_in_executor(
                None, lambda: KPipeline(lang_code="a")
            )

            duration = time.perf_counter() - start
            logger.info("Kokoro pipeline initialized in %.2fs", duration)

        except Exception as e:
            logger.exception("Failed to initialize Kokoro: %s", e)
            raise

    @property
    def name(self) -> str:
        return "kokoro"

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate  # 24000Hz

    async def synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize text to raw PCM int16 audio bytes."""
        if not text or not text.strip():
            return None

        if KokoroTTS._shared_pipeline is None:
            logger.error(
                "Kokoro pipeline not initialized. Call KokoroTTS.initialize() at startup."
            )
            return None

        start_time = time.perf_counter()

        try:
            loop = asyncio.get_event_loop()

            # Run synthesis in thread pool
            # The pipeline returns a generator of (graphemes, phonemes, audio) tuples
            def _synthesize():
                all_audio = []
                generator = KokoroTTS._shared_pipeline(
                    text,
                    voice=self.config.voice,
                    speed=self.config.speed,
                )
                for gs, ps, audio in generator:
                    all_audio.append(audio)

                if not all_audio:
                    return None

                # Concatenate all audio chunks
                return np.concatenate(all_audio)

            samples = await loop.run_in_executor(None, _synthesize)

            if samples is None:
                logger.warning("Kokoro produced no audio")
                return None

            # Convert float32 samples to int16 PCM bytes
            # Kokoro returns float32 in range [-1, 1]
            audio_int16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()

            duration = time.perf_counter() - start_time
            audio_duration = len(samples) / self.sample_rate

            logger.info(
                "Kokoro synthesized: text_len=%d audio=%.2fs in %.3fs (%.1fx realtime)",
                len(text),
                audio_duration,
                duration,
                audio_duration / duration if duration > 0 else 0,
            )

            return audio_bytes

        except Exception as e:
            logger.exception("Kokoro synthesis failed: %s", e)
            return None
