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
    """Silero VAD with 512-sample windowing and Dynamic Echo Suppression.

    Plain English:
    -------------
    This component listens to audio and decides: "Is the user talking or silent?"

    It works like a state machine with 3 states:
    - IDLE: No one is talking. We keep a small buffer just in case.
    - SPEECH: User is talking. We collect all their audio.
    - SILENCE_AFTER_SPEECH: User paused. If silence continues, we're done.

    When enough silence is detected after speech, we "commit" the utterance
    and send it for transcription.

    What happens if this is removed?
    --------------------------------
    The system cannot detect when you start/stop speaking. Either:
    - It sends audio constantly (wasting resources)
    - It never knows when to transcribe (nothing works)

    Key features:
    - 500ms pre-roll: Captures the start of words before VAD detected speech
    - Echo suppression: Raises threshold during TTS so assistant's voice is ignored
    - Max duration cutoff: Prevents infinite recordings
    """

    def __init__(self, vad_config: VADConfig, audio_config: AudioConfig):
        self.vad_config = vad_config
        self.audio_config = audio_config

        # Load Silero VAD v5 ONNX model
        self.model = load_silero_vad(onnx=True)

        self._state = VADState.IDLE
        self._silence_frame_count = 0
        self._buffer = bytearray()
        self._streaming_buffer = bytearray()

        self.WINDOW_SIZE_SAMPLES = 512  # Fixed for Silero v5 @ 16kHz
        self.BYTES_PER_WINDOW = (
            self.WINDOW_SIZE_SAMPLES * self.audio_config.bytes_per_sample
        )

        logger.info("Silero VAD initialized: Fixed 512-sample windowing.")

    def _get_speech_probability(self, audio_chunk: bytes) -> float:
        audio_np = np.frombuffer(audio_chunk, dtype=np.float32).copy()
        tensor_audio = torch.from_numpy(audio_np)
        speech_prob = self.model(tensor_audio, self.audio_config.sample_rate).item()
        return speech_prob

    def process_chunk(
        self, audio_chunk: bytes, is_tts_active: bool = False
    ) -> Optional[bytes]:
        """Returns complete utterance when speech ends, else None.

        Args:
            audio_chunk: Raw PCM audio bytes.
            is_tts_active: True if assistant is currently speaking.
        """
        self._streaming_buffer.extend(audio_chunk)

        # 1. ACOUSTIC ECHO SUPPRESSION: Raise threshold while assistant is talking
        active_threshold = self.vad_config.speech_threshold
        if is_tts_active:
            # Multiply threshold to make VAD more conservative during playback
            active_threshold = min(0.9, active_threshold * 1.5)

        while len(self._streaming_buffer) >= self.BYTES_PER_WINDOW:
            window = bytes(self._streaming_buffer[: self.BYTES_PER_WINDOW])
            self._streaming_buffer = self._streaming_buffer[self.BYTES_PER_WINDOW :]

            speech_prob = self._get_speech_probability(window)

            if speech_prob >= active_threshold:
                if self._state == VADState.IDLE:
                    logger.debug(
                        "Speech START (prob=%.4f, tts_active=%s)",
                        speech_prob,
                        is_tts_active,
                    )
                    # PRE-ROLL: Keep audio before speech detection to avoid clipping first word
                    # Use 500ms pre-roll to capture consonants/plosives at word start
                    max_pre_roll = int(
                        self.audio_config.sample_rate
                        * 0.5
                        * self.audio_config.bytes_per_sample
                    )
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
                    # Keep a rolling buffer for pre-roll even when idle
                    max_pre_roll = int(
                        self.audio_config.sample_rate
                        * 0.5
                        * self.audio_config.bytes_per_sample
                    )
                    self._buffer.extend(window)
                    if len(self._buffer) > max_pre_roll:
                        self._buffer = self._buffer[-max_pre_roll:]

            duration = self._get_buffer_duration()

            # Finalize utterance on silence duration or max length
            if self._state == VADState.SILENCE_AFTER_SPEECH:
                if self._silence_frame_count >= self.vad_config.silence_frames_required:
                    if duration >= self.vad_config.min_utterance_seconds:
                        return self._finalize_utterance(duration)
                    else:
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

    def reset(self) -> None:
        """Public reset method for external callers."""
        self._reset()
        logger.debug("VAD state reset to IDLE")

    def get_state_debug_info(self) -> dict:
        return {
            "state": self._state.value,
            "silence_frames": self._silence_frame_count,
            "buffer_duration": self._get_buffer_duration(),
        }
