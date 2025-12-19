"""
Text-to-Speech using Piper.
"""

import logging
import os
import subprocess
from typing import Optional

from server.config import PiperConfig

logger = logging.getLogger(__name__)


class PiperTTS:
    """Wrapper around Piper TTS for speech synthesis."""

    def __init__(self, config: PiperConfig):
        self.config = config
        self._validate_installation()
        logger.info("Piper TTS initialized: model=%s", config.model_path)

    def _validate_installation(self) -> None:
        """Validate that Piper executable and model files exist."""
        if not os.path.exists(self.config.exe_path):
            raise FileNotFoundError(
                f"Piper executable not found: {self.config.exe_path}"
            )

        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Piper model not found: {self.config.model_path}")

        if not os.path.exists(self.config.model_config_path):
            raise FileNotFoundError(
                f"Piper model config not found: {self.config.model_config_path}"
            )

    def synthesize(self, text: str) -> Optional[bytes]:
        """
        Synthesize text to raw PCM audio.

        Args:
            text: Text to synthesize

        Returns:
            Raw PCM audio bytes, or None if synthesis fails or text is empty
        """
        if not text or not text.strip():
            return None

        command = [
            self.config.exe_path,
            "--model",
            self.config.model_path,
            "--output_raw",
        ]

        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )

            audio_bytes, stderr = process.communicate(
                input=text.encode("utf-8"), timeout=30
            )

            if process.returncode != 0:
                error_msg = stderr.decode("utf-8", errors="replace").strip()
                logger.error("Piper synthesis failed: %s", error_msg)
                return None

            if not audio_bytes:
                logger.warning("Piper produced empty audio for text: %s", text[:100])
                return None

            logger.info(
                "Synthesized %d bytes of audio from %d chars",
                len(audio_bytes),
                len(text),
            )
            return audio_bytes

        except subprocess.TimeoutExpired:
            logger.error("Piper synthesis timed out for text: %s", text[:100])
            process.kill()
            return None
        except Exception:
            logger.exception("Piper synthesis error")
            return None
