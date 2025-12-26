import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class AudioFeedback:
    """Generates simple feedback tones for UI states."""

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self._cache: dict[str, bytes] = {}
        self._generate_tones()

    def _generate_tones(self) -> None:
        """Pre-generate common feedback tones."""
        # Listening beep: short high tone
        self._cache["listening"] = self._generate_tone(
            frequency=880, duration=0.1, volume=0.3
        )

        # Processing beep: two quick tones
        tone1 = self._generate_tone(frequency=660, duration=0.08, volume=0.25)
        silence = self._generate_silence(0.05)
        tone2 = self._generate_tone(frequency=880, duration=0.08, volume=0.25)
        self._cache["processing"] = tone1 + silence + tone2

        # Ready/done beep: ascending tones
        tone1 = self._generate_tone(frequency=440, duration=0.1, volume=0.2)
        tone2 = self._generate_tone(frequency=660, duration=0.1, volume=0.2)
        tone3 = self._generate_tone(frequency=880, duration=0.15, volume=0.25)
        self._cache["ready"] = tone1 + tone2 + tone3

        # Error beep: low descending tone
        self._cache["error"] = self._generate_tone(
            frequency=220, duration=0.3, volume=0.3
        )

        # Wake word detected: short ascending chirp
        tone1 = self._generate_tone(frequency=523, duration=0.05, volume=0.25)
        tone2 = self._generate_tone(frequency=659, duration=0.05, volume=0.25)
        self._cache["wake_word"] = tone1 + tone2

        logger.debug("Audio feedback tones generated")

    def _generate_tone(
        self, frequency: float, duration: float, volume: float = 0.5
    ) -> bytes:
        """Generate a sine wave tone as PCM16 bytes."""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        wave = np.sin(2 * np.pi * frequency * t)

        # Apply fade in/out to prevent clicks
        fade_samples = int(self.sample_rate * 0.01)
        if fade_samples > 0 and len(wave) > fade_samples * 2:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            wave[:fade_samples] *= fade_in
            wave[-fade_samples:] *= fade_out

        # Convert to int16
        pcm = (wave * volume * 32767).astype(np.int16)
        return pcm.tobytes()

    def _generate_silence(self, duration: float) -> bytes:
        """Generate silence as PCM16 bytes."""
        samples = int(self.sample_rate * duration)
        return np.zeros(samples, dtype=np.int16).tobytes()

    def get_tone(self, name: str) -> Optional[bytes]:
        """Get a pre-generated tone by name."""
        return self._cache.get(name)

    @property
    def listening_tone(self) -> bytes:
        return self._cache["listening"]

    @property
    def processing_tone(self) -> bytes:
        return self._cache["processing"]

    @property
    def ready_tone(self) -> bytes:
        return self._cache["ready"]

    @property
    def error_tone(self) -> bytes:
        return self._cache["error"]
