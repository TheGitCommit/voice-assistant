import asyncio
import concurrent.futures
import logging
import time

import numpy as np
from faster_whisper import WhisperModel

from server.config import WhisperConfig

logger = logging.getLogger(__name__)

# Thread pool for CPU-bound transcription
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)


class WhisperSTT:
    _shared_model = None

    def __init__(self, config: WhisperConfig):
        self.config = config

        if WhisperSTT._shared_model is None:
            logger.info(
                "Loading Whisper model: model=%s device=%s compute_type=%s",
                config.model_size,
                config.device,
                config.compute_type,
            )
            WhisperSTT._shared_model = WhisperModel(
                config.model_size,
                device=config.device,
                compute_type=config.compute_type,
            )
            logger.info("Whisper model loaded successfully")
        else:
            logger.info("Using cached Whisper model")

        self.model = WhisperSTT._shared_model

    def transcribe(self, audio: np.ndarray) -> str:
        """Synchronous transcription (blocks the calling thread)."""
        start_time = time.perf_counter()
        try:
            if self.model is None:
                logger.error("Whisper model is None!")
                return ""

            segments, info = self.model.transcribe(
                audio,
                beam_size=self.config.beam_size,
            )

            text = " ".join(seg.text for seg in segments).strip()
            duration = time.perf_counter() - start_time

            logger.info(
                "Transcription complete: text_len=%d language=%s probability=%.2f latency=%.3fs",
                len(text),
                info.language,
                info.language_probability,
                duration,
            )

            return text

        except Exception:
            logger.exception("Transcription failed")
            return ""

    async def transcribe_async(self, audio: np.ndarray) -> str:
        """Async transcription - runs in thread pool to avoid blocking event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self.transcribe, audio)


# Note: transcribe_streaming() was removed - it was defined but never called.
# The method was misleadingly named as it still waited for full transcription.
# See REFACTOR_LOG.md for details.
