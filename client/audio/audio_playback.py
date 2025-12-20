import logging
import threading
from collections import deque

import numpy as np
import sounddevice as sd

from client.config import AudioPlaybackConfig

logger = logging.getLogger(__name__)


class AudioPlayback:
    """Plays PCM audio to speakers via sounddevice."""

    def __init__(self, config: AudioPlaybackConfig):
        self.config = config
        self._stream = sd.OutputStream(
            samplerate=config.sample_rate,
            channels=config.channels,
            dtype=config.dtype,
        )
        self._playing = False
        self._lock = threading.Lock()

        logger.info(
            "AudioPlayback initialized: rate=%d channels=%d",
            config.sample_rate,
            config.channels,
        )

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._playing

    def start(self) -> None:
        logger.info("Starting audio playback")
        self._stream.start()
        logger.info("Audio playback started")

    def play(self, audio_bytes: bytes) -> None:
        if len(audio_bytes) % 2 != 0:
            raise ValueError(
                f"Audio data must be 16-bit aligned, got {len(audio_bytes)} bytes"
            )

        pcm_frames = np.frombuffer(audio_bytes, dtype=np.int16)

        logger.debug("Playing %d audio frames", len(pcm_frames))

        with self._lock:
            self._playing = True

        try:
            self._stream.write(pcm_frames)
        except Exception:
            logger.exception("Error writing to audio stream")
        finally:
            with self._lock:
                self._playing = False

    def stop_playback(self) -> None:
        """Stop current playback and clear buffer."""
        with self._lock:
            self._playing = False
        try:
            self._stream.stop()
            self._stream.start()
            logger.info("Playback stopped and buffer cleared")
        except Exception:
            logger.exception("Error stopping playback")

    def close(self) -> None:
        logger.info("Closing audio playback")

        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            logger.exception("Error closing audio stream")

        logger.info("Audio playback closed")

    def is_active(self) -> bool:
        return self._stream.active
