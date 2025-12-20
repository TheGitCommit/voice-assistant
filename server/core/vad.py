import logging
from enum import Enum
from typing import Optional
import numpy as np
import torch
from silero_vad import load_silero_vad
from server.config import VADConfig, AudioConfig

logger = logging.getLogger(__name__)


class VADState(Enum):
    IDLE = "idle"
    SPEECH = "speech"
    SILENCE_AFTER_SPEECH = "silence_after_speech"


class VoiceActivityDetector:
    """Silero VAD with 512-sample windowing."""

    def __init__(self, vad_config: VADConfig, audio_config: AudioConfig):
        self.vad_config = vad_config
        self.audio_config = audio_config

        self.model = load_silero_vad(onnx=True)

        self._state = VADState.IDLE
        self._silence_frame_count = 0
        self._buffer = bytearray()
        self._streaming_buffer = bytearray()

        self.WINDOW_SIZE_SAMPLES = 512  # Silero v5 @ 16kHz
        self.BYTES_PER_WINDOW = (
            self.WINDOW_SIZE_SAMPLES * self.audio_config.bytes_per_sample
        )

        logger.info("Silero VAD initialized: Fixed 512-sample windowing.")

    def _get_speech_probability(self, audio_chunk: bytes) -> float:
        audio_np = np.frombuffer(audio_chunk, dtype=np.float32).copy()
        tensor_audio = torch.from_numpy(audio_np)
        speech_prob = self.model(tensor_audio, self.audio_config.sample_rate).item()
        return speech_prob

    def process_chunk(self, audio_chunk: bytes) -> Optional[bytes]:
        """Returns complete utterance when speech ends, else None."""
        self._streaming_buffer.extend(audio_chunk)

        while len(self._streaming_buffer) >= self.BYTES_PER_WINDOW:
            window = bytes(self._streaming_buffer[: self.BYTES_PER_WINDOW])
            self._streaming_buffer = self._streaming_buffer[self.BYTES_PER_WINDOW :]

            speech_prob = self._get_speech_probability(window)

            if speech_prob >= self.vad_config.speech_threshold:
                if self._state == VADState.IDLE:
                    logger.debug("Speech START: prob=%.4f", speech_prob)
                    max_pre_roll = self.BYTES_PER_WINDOW * 6
                    if len(self._buffer) > max_pre_roll:
                        self._buffer = self._buffer[-max_pre_roll:]

                self._state = VADState.SPEECH
                self._silence_frame_count = 0
                self._buffer.extend(window)

            else:
                if self._state == VADState.SPEECH:
                    self._silence_frame_count = 1
                    self._state = VADState.SILENCE_AFTER_SPEECH
                    self._buffer.extend(window)

                elif self._state == VADState.SILENCE_AFTER_SPEECH:
                    self._silence_frame_count += 1
                    self._buffer.extend(window)

                elif self._state == VADState.IDLE:
                    max_pre_roll = self.BYTES_PER_WINDOW * 6
                    self._buffer.extend(window)
                    if len(self._buffer) > max_pre_roll:
                        self._buffer = self._buffer[-max_pre_roll:]

            duration = self._get_buffer_duration()

            if self._state == VADState.SILENCE_AFTER_SPEECH:
                if self._silence_frame_count >= self.vad_config.silence_frames_required:
                    if duration >= self.vad_config.min_utterance_seconds:
                        return self._finalize_utterance(duration)
                    else:
                        logger.debug("Utterance too short (%.2fs), resetting", duration)
                        self._reset()

            if duration >= self.vad_config.max_utterance_seconds:
                logger.warning("Max utterance duration reached, forcing finalization")
                return self._finalize_utterance(duration)

        return None

    def _finalize_utterance(self, duration: float) -> bytes:
        logger.info(
            "Utterance complete: duration=%.2fs bytes=%d", duration, len(self._buffer)
        )
        utterance = bytes(self._buffer)
        self._reset()
        return utterance

    def _get_buffer_duration(self) -> float:
        return (
            len(self._buffer) / self.audio_config.bytes_per_sample
        ) / self.audio_config.sample_rate

    def _reset(self) -> None:
        self._buffer.clear()
        self._streaming_buffer.clear()
        self._silence_frame_count = 0
        self._state = VADState.IDLE
        logger.debug("VAD state reset to IDLE")

    def get_state_debug_info(self) -> dict:
        return {
            "state": self._state.value,
            "silence_frames": self._silence_frame_count,
            "buffer_duration": self._get_buffer_duration(),
        }
