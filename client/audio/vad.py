import logging

import numpy as np
import webrtcvad

from client.config import VADConfig

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """
    VoiceActivityDetector class is responsible for detecting voice activity from audio frames.

    This class uses WebRTC Voice Activity Detection (VAD) to determine if a given audio
    frame contains speech or not. It processes audio signals in small chunks (frames) and maintains
    a state of whether the current context is within speech or silence. It also provides a reset
    mechanism to reinitialize state and an additional method to check the current voice activity status.

    :ivar config: The configuration for the VAD behavior, including aggressiveness and silence
        processing limits.
    :type config: VADConfig
    :ivar sample_rate: The sample rate of the audio input in Hz. Supported values are 8000, 16000,
        32000, and 48000.
    :type sample_rate: int
    """
    def __init__(self, config: VADConfig, sample_rate: int):
        self.config = config
        self.sample_rate = sample_rate

        # WebRTC VAD only supports specific sample rates
        if sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError(
                f"VAD requires sample rate of 8/16/32/48kHz, got {sample_rate}"
            )

        self._vad = webrtcvad.Vad(config.aggressiveness)
        self._in_speech = False
        self._silence_frame_count = 0

        logger.info(
            "VAD initialized: aggressiveness=%d sample_rate=%d",
            config.aggressiveness,
            sample_rate,
        )

    def process_frame(self, audio_float32: np.ndarray) -> bool:
        # Convert float32 to int16 for WebRTC VAD
        pcm_int16 = self._float32_to_int16(audio_float32)

        # Check if frame contains speech
        is_speech = self._vad.is_speech(pcm_int16, self.sample_rate)

        # Update state machine
        if is_speech:
            if not self._in_speech:
                logger.debug("Speech START detected")
            self._in_speech = True
            self._silence_frame_count = 0
        else:
            # Only count silence if we were in speech
            if self._in_speech:
                self._silence_frame_count += 1

                if self._silence_frame_count >= self.config.silence_limit_frames:
                    logger.debug("Speech END detected (silence)")
                    self._in_speech = False
                    self._silence_frame_count = 0

        return self._in_speech

    @staticmethod
    def _float32_to_int16(audio: np.ndarray) -> bytes:
        # Clip to valid range and convert to int16
        clipped = np.clip(audio, -1.0, 1.0)
        pcm_int16 = (clipped * 32767).astype(np.int16)
        return pcm_int16.tobytes()

    def is_in_speech(self) -> bool:
        return self._in_speech

    def reset(self) -> None:
        logger.debug("VAD state reset")
        self._in_speech = False
        self._silence_frame_count = 0
