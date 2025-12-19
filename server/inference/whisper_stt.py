"""
Speech-to-Text using Faster Whisper.
"""

import logging
from typing import Any

import numpy as np
from faster_whisper import WhisperModel

from server.config import WhisperConfig

logger = logging.getLogger(__name__)


class WhisperSTT:
    """Wrapper around Faster Whisper for speech transcription."""

    def __init__(self, config: WhisperConfig):
        self.config = config
        logger.info(
            "Loading Whisper model: model=%s device=%s compute_type=%s",
            config.model_size,
            config.device,
            config.compute_type,
        )
        self.model = WhisperModel(
            config.model_size,
            device=config.device,
            compute_type=config.compute_type,
        )
        logger.info("Whisper model loaded successfully")

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Float32 numpy array of audio samples

        Returns:
            Transcribed text (empty string if transcription fails)
        """
        try:
            segments, info = self.model.transcribe(
                audio,
                beam_size=self.config.beam_size,
            )

            # Consume generator and join segments
            text = " ".join(seg.text for seg in segments).strip()

            logger.info(
                "Transcription complete: text_len=%d language=%s probability=%.2f",
                len(text),
                info.language,
                info.language_probability,
            )

            return text

        except Exception:
            logger.exception("Transcription failed")
            return ""
