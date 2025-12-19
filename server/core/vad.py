"""
Energy-based Voice Activity Detection (VAD).
"""

import logging
from enum import Enum
from typing import Optional

import numpy as np

from server.config import VADConfig, AudioConfig

logger = logging.getLogger(__name__)


class VADState(Enum):
    """
    Represents the various states of a Voice Activity Detection (VAD) system.

    The VADState Enum captures the state transitions in a typical VAD system,
    including the idle state when no speech is detected, the active speech state,
    and the silence phase that follows detected speech.

    """

    IDLE = "idle"  # No speech detected, buffer empty or contains only noise
    SPEECH = "speech"  # Currently receiving speech
    SILENCE_AFTER_SPEECH = "silence_after_speech"  # Silence detected after speech


class VoiceActivityDetector:
    """
    Energy-based VAD that segments audio into utterances.

    Uses RMS energy thresholds and silence duration to detect speech boundaries.
    """

    def __init__(self, vad_config: VADConfig, audio_config: AudioConfig):
        self.vad_config = vad_config
        self.audio_config = audio_config

        self._state = VADState.IDLE
        self._silence_frame_count = 0
        self._buffer = bytearray()

        logger.info(
            "VAD initialized: silence_thresh=%.4f speech_thresh=%.4f",
            vad_config.silence_threshold,
            vad_config.speech_threshold,
        )

    def _calculate_energy(self, audio_chunk: bytes) -> float:
        """Calculate RMS energy of audio chunk."""
        pcm = np.frombuffer(audio_chunk, dtype=np.float32)
        return float(np.sqrt(np.mean(pcm**2)))

    def _get_buffer_duration(self) -> float:
        """Get duration of buffered audio in seconds."""
        num_samples = len(self._buffer) // self.audio_config.bytes_per_sample
        return num_samples / self.audio_config.sample_rate

    def process_chunk(self, audio_chunk: bytes) -> Optional[bytes]:
        """
        Process an audio chunk and return complete utterance if ready.

        Args:
            audio_chunk: Raw PCM audio bytes (float32)

        Returns:
            Complete utterance bytes if end detected, None otherwise
        """
        energy = self._calculate_energy(audio_chunk)
        self._buffer.extend(audio_chunk)
        duration = self._get_buffer_duration()

        # State machine for speech detection
        if energy >= self.vad_config.speech_threshold:
            # Definite speech detected
            if self._state != VADState.SPEECH:
                logger.debug("Speech START: energy=%.4f", energy)
                self._state = VADState.SPEECH
            self._silence_frame_count = 0

        elif energy < self.vad_config.silence_threshold:
            # Definite silence detected
            if self._state == VADState.SPEECH:
                self._silence_frame_count += 1
                self._state = VADState.SILENCE_AFTER_SPEECH
            elif self._state == VADState.IDLE:
                # Clear noise buffer if it's been accumulating too long
                if duration > self.vad_config.noise_buffer_clear_seconds:
                    logger.debug("Clearing noise buffer: duration=%.2fs", duration)
                    self._buffer.clear()
                    self._silence_frame_count = 0

        else:
            # Ambiguous energy level - maintain current state
            if self._state == VADState.SPEECH:
                # Reset silence counter if we're in speech
                self._silence_frame_count = 0

        # Check for utterance completion conditions
        utterance_complete = False

        if self._state == VADState.SPEECH:
            if duration >= self.vad_config.max_utterance_seconds:
                logger.debug("Max duration reached: %.2fs", duration)
                utterance_complete = True

        if self._state == VADState.SILENCE_AFTER_SPEECH:
            if self._silence_frame_count >= self.vad_config.silence_frames_required:
                logger.debug(
                    "Silence detected after speech: frames=%d",
                    self._silence_frame_count,
                )
                utterance_complete = True

        # Extract and return utterance if complete
        if utterance_complete:
            if duration < self.vad_config.min_utterance_seconds:
                logger.debug(
                    "Utterance too short: %.2fs < %.2fs, discarding",
                    duration,
                    self.vad_config.min_utterance_seconds,
                )
                self._reset()
                return None

            logger.info(
                "Utterance complete: duration=%.2fs bytes=%d",
                duration,
                len(self._buffer),
            )
            utterance = bytes(self._buffer)
            self._reset()
            return utterance

        return None

    def _reset(self) -> None:
        """Reset VAD state for next utterance."""
        self._buffer.clear()
        self._silence_frame_count = 0
        self._state = VADState.IDLE

    def get_state_debug_info(self) -> dict:
        """Get current state for debugging/logging."""
        return {
            "state": self._state.value,
            "silence_frames": self._silence_frame_count,
            "buffer_duration": self._get_buffer_duration(),
            "buffer_bytes": len(self._buffer),
        }
