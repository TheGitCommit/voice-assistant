"""
Audio playback to speaker using sounddevice.
"""

import logging

import numpy as np
import sounddevice as sd

from client.config import AudioPlaybackConfig

logger = logging.getLogger(__name__)


class AudioPlayback:
    """
    Handles audio playback functionality, providing methods to start, play, and close audio streams.

    This class utilizes the given audio configuration to initialize an output audio stream.
    It ensures proper playback by managing audio frames, handling errors, and logging activities.
    The class can play audio data from provided byte streams, check the stream's active status,
    and securely close the stream when no longer needed.

    :ivar config: Configuration settings for audio playback, including sample rate, channels, and dtype.
    :type config: AudioPlaybackConfig
    """

    def __init__(self, config: AudioPlaybackConfig):
        self.config = config
        self._stream = sd.OutputStream(
            samplerate=config.sample_rate,
            channels=config.channels,
            dtype=config.dtype,
        )

        logger.info(
            "AudioPlayback initialized: rate=%d channels=%d",
            config.sample_rate,
            config.channels,
        )

    def start(self) -> None:
        """Start the output stream."""
        logger.info("Starting audio playback")
        self._stream.start()
        logger.info("Audio playback started")

    def play(self, audio_bytes: bytes) -> None:
        """
        Plays audio from the provided byte stream. The method ensures the given audio data is
        16-bit sample aligned. It converts the input audio bytes to PCM frames and writes them
        to the designated audio stream. Logs the frame count before playing audio.

        :param audio_bytes: A bytes object containing audio data aligned to 16-bit samples.
        :type audio_bytes: bytes
        :return: None
        :rtype: None
        :raises ValueError: If the audio data length is not aligned to 16-bit samples.
        :raises Exception: If there is an error writing to the audio stream.
        """
        # Validate alignment
        if len(audio_bytes) % 2 != 0:
            raise ValueError(
                f"Audio data must be aligned to 16-bit samples, got {len(audio_bytes)} bytes"
            )

        # Convert bytes to numpy array
        pcm_frames = np.frombuffer(audio_bytes, dtype=np.int16)

        logger.debug("Playing %d audio frames", len(pcm_frames))

        try:
            self._stream.write(pcm_frames)
        except Exception:
            logger.exception("Error writing to audio stream")

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
