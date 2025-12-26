"""Base TTS interface for swappable TTS providers."""

import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class BaseTTS(ABC):
    """Abstract base class for TTS providers.

    All TTS implementations must provide:
    - async synthesize(text) -> Optional[bytes]: Convert text to audio bytes
    - sample_rate property: Output sample rate in Hz
    - name property: Provider name for logging
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'piper', 'kokoro')."""
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Output sample rate in Hz."""
        ...

    @abstractmethod
    async def synthesize(self, text: str) -> Optional[bytes]:
        """Synthesize text to raw audio bytes.

        Args:
            text: Text to synthesize

        Returns:
            Raw audio bytes (PCM int16 mono) or None on failure
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, sample_rate={self.sample_rate})"
