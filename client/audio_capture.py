"""
Audio capture from microphone using sounddevice.
"""

import logging
import queue
from typing import Optional

import numpy as np
import sounddevice as sd

from client.config import AudioCaptureConfig

logger = logging.getLogger(__name__)


class AudioCapture:
    """
    Handles capturing audio data from a microphone and provides functionality
    to retrieve audio chunks for further processing.

    This class interfaces with the underlying sound device to stream audio
    data into a queue for asynchronous consumption. It allows for starting,
    stopping, and checking the running status of audio capture, as well as
    reading captured audio chunks.

    :ivar config: Configuration for audio capture, such as sample rate, number
        of channels, chunk size, and data type.
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
        """
        Callback invoked by sounddevice when audio is available.

        Runs in a separate thread - must be lightweight and non-blocking.
        """
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
        """Start capturing audio from microphone."""
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
        """
        Read next audio chunk from queue.

        Args:
            timeout: Maximum time to wait for data (None = block forever)

        Returns:
            Audio chunk as float32 numpy array

        Raises:
            queue.Empty: If timeout expires with no data
        """
        return self._queue.get(timeout=timeout)

    def stop(self) -> None:
        """Stop capturing audio and release resources."""
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
        """Check if capture is currently running."""
        return self._stream is not None and self._stream.active
