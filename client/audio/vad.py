import logging

import numpy as np
import webrtcvad

from client.config import VADConfig

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """WebRTC-based VAD for detecting speech in audio frames."""

    def __init__(self, config: VADConfig, sample_rate: int):
        self.config = config
        self.sample_rate = sample_rate

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
        pcm_int16 = self._float32_to_int16(audio_float32)
        is_speech = self._vad.is_speech(pcm_int16, self.sample_rate)

        if is_speech:
            if not self._in_speech:
                logger.debug("Speech START detected")
            self._in_speech = True
            self._silence_frame_count = 0
        else:
            if self._in_speech:
                self._silence_frame_count += 1

                if self._silence_frame_count >= self.config.silence_limit_frames:
                    logger.debug("Speech END detected (silence)")
                    self._in_speech = False
                    self._silence_frame_count = 0

        return self._in_speech

    @staticmethod
    def _float32_to_int16(audio: np.ndarray) -> bytes:
        clipped = np.clip(audio, -1.0, 1.0)
        pcm_int16 = (clipped * 32767).astype(np.int16)
        return pcm_int16.tobytes()

    def is_in_speech(self) -> bool:
        return self._in_speech

    def reset(self) -> None:
        logger.debug("VAD state reset")
        self._in_speech = False
        self._silence_frame_count = 0
