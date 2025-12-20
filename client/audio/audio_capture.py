import logging
import queue
from typing import Optional

import numpy as np
import sounddevice as sd

from client.config import AudioCaptureConfig

logger = logging.getLogger(__name__)


class AudioCapture:
    """
    Represents an audio capturing utility that uses an input stream to capture
    audio data according to a specified configuration. This class provides
    methods for starting and stopping audio capture, reading audio data, and
    checking if the audio stream is running. It manages a queue to store
    audio data chunks and handles streaming operations efficiently.

    :ivar config: Configuration object for audio capture settings, including
        sample rate, channels, chunk size, and queue maximum size.
    :type config: AudioCaptureConfig
    """

    def __init__(self, config: AudioCaptureConfig):
        self.config = config
        self._stream: Optional[sd.InputStream] = None
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=config.queue_maxsize)

        logger.info(
            "AudioCapture initialized: rate=%d channels=%d chunk_size=%d",
            config.sample_rate,
            config.channels,
            config.chunk_size,
        )

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            logger.warning("Audio capture status: %s", status)
            return

        # Flatten and copy to avoid issues with internal buffers
        chunk = indata.reshape(-1).copy()

        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            # Drop frame if queue is full (prevents blocking audio thread)
            logger.debug("Audio queue full, dropping frame")

    def start(self) -> None:
        if self._stream is not None:
            logger.warning("AudioCapture already started")
            return

        logger.info("Starting audio capture")

        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype=self.config.dtype,
            blocksize=self.config.chunk_size,
            callback=self._audio_callback,
        )
        self._stream.start()

        logger.info("Audio capture started")

    def read(self, timeout: Optional[float] = None) -> np.ndarray:

        return self._queue.get(timeout=timeout)

    def stop(self) -> None:
        if self._stream is None:
            return

        logger.info("Stopping audio capture")

        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            logger.exception("Error stopping audio stream")
        finally:
            self._stream = None

        logger.info("Audio capture stopped")

    def is_running(self) -> bool:
        return self._stream is not None and self._stream.active
