import asyncio
import logging
import queue
import threading
from collections import deque
from typing import Optional

import numpy as np
import sounddevice as sd

from client.config.config import AudioCaptureConfig

logger = logging.getLogger(__name__)

# Pre-roll duration in seconds (gives hardware time to stabilize)
PREROLL_DURATION = 1.5


class AudioCapture:
    """Captures microphone audio into a queue for async consumption.

    Plain English:
    -------------
    This records audio from your microphone and puts it in a queue for processing.

    The 1.5-second Pre-roll Buffer:
    When you start talking, microphones often miss the first syllable because:
    - Hardware needs time to "wake up"
    - VAD needs a few frames to detect speech

    Solution: We buffer 1.5 seconds of audio BEFORE sending anything to VAD.
    When speech is detected, we include this buffer so "Hello" stays "Hello",
    not "ello".

    What happens if this is removed?
    --------------------------------
    No audio capture. The assistant is deaf.
    """

    def __init__(self, config: AudioCaptureConfig):
        self.config = config
        self._stream: Optional[sd.InputStream] = None
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=config.queue_maxsize)

        # Pre-roll buffer: collects audio for PREROLL_DURATION before feeding to VAD
        self._preroll_samples = int(config.sample_rate * PREROLL_DURATION)
        self._preroll_buffer: deque[np.ndarray] = deque()
        self._preroll_collected = 0
        self._preroll_complete = False
        self._preroll_event = (
            threading.Event()
        )  # Thread-safe event for cross-thread signaling

        logger.info(
            "AudioCapture initialized: rate=%d channels=%d chunk_size=%d preroll=%.1fs",
            config.sample_rate,
            config.channels,
            config.chunk_size,
            PREROLL_DURATION,
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

        chunk = indata.reshape(-1).copy()

        # Pre-roll phase: collect audio but don't feed to queue yet
        if not self._preroll_complete:
            self._preroll_buffer.append(chunk)
            self._preroll_collected += len(chunk)

            if self._preroll_collected >= self._preroll_samples:
                self._preroll_complete = True
                logger.info(
                    "Pre-roll complete: collected %.2fs of audio",
                    self._preroll_collected / self.config.sample_rate,
                )
                # Flush pre-roll buffer to queue
                while self._preroll_buffer:
                    buffered_chunk = self._preroll_buffer.popleft()
                    try:
                        self._queue.put_nowait(buffered_chunk)
                    except queue.Full:
                        pass  # Drop oldest if queue is full
                # Signal that pre-roll is done (thread-safe)
                self._preroll_event.set()
            return

        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            logger.debug("Audio queue full, dropping frame")

    def start(self) -> None:
        if self._stream is not None:
            logger.warning("AudioCapture already started")
            return

        logger.info("Starting audio capture (pre-roll phase)")

        # Reset pre-roll state
        self._preroll_buffer.clear()
        self._preroll_collected = 0
        self._preroll_complete = False
        self._preroll_event.clear()

        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype=self.config.dtype,
            blocksize=self.config.chunk_size,
            callback=self._audio_callback,
        )
        self._stream.start()

        logger.info("Audio capture started, waiting for pre-roll...")

    async def wait_for_preroll(self, timeout: float = 3.0) -> bool:
        """Wait for pre-roll buffer to fill. Returns True if successful.

        Uses threading.Event with async polling for cross-thread compatibility.
        """
        if self._preroll_complete:
            return True

        # Poll the threading.Event in an async-friendly way
        loop = asyncio.get_event_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, self._preroll_event.wait, timeout),
                timeout=timeout + 0.5,  # Small buffer for executor overhead
            )
            return self._preroll_complete
        except asyncio.TimeoutError:
            logger.warning("Pre-roll timeout after %.1fs", timeout)
            self._preroll_complete = True  # Continue anyway
            return False

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

    @property
    def preroll_complete(self) -> bool:
        return self._preroll_complete
