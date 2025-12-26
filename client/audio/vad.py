import logging

import numpy as np
import webrtcvad

from client.config.config import VADConfig

logger = logging.getLogger(__name__)

# Echo suppression: increase energy threshold by this factor when TTS is playing
ECHO_SUPPRESSION_FACTOR = 1.25


class VoiceActivityDetector:
    """WebRTC-based VAD for detecting speech in audio frames.

    Plain English:
    -------------
    This is the client-side "ears" that detect when you're talking.
    It uses WebRTC's VAD (the same tech used in video calls) to distinguish
    speech from background noise.

    Echo Suppression:
    When the assistant is speaking, we raise the detection threshold by 25%.
    This prevents the assistant's own voice (coming from speakers) from being
    detected as YOUR voice.

    Note: This is a SIMPLE client-side VAD. The server has a more sophisticated
    Silero VAD for actual utterance detection.

    What happens if this is removed?
    --------------------------------
    The client can't detect speech, so it can't do echo suppression properly.
    The assistant would constantly interrupt itself.
    """

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

        # Echo suppression state
        self._tts_active = False
        self._base_energy_threshold = getattr(config, "energy_threshold", 0.01)

        logger.info(
            "VAD initialized: aggressiveness=%d sample_rate=%d echo_suppression=%.0f%%",
            config.aggressiveness,
            sample_rate,
            (ECHO_SUPPRESSION_FACTOR - 1) * 100,
        )

    def process_frame(self, audio_float32: np.ndarray) -> bool:
        # Echo suppression: apply higher threshold when TTS is active
        if self._tts_active:
            energy = self._calculate_energy(audio_float32)
            effective_threshold = self._base_energy_threshold * ECHO_SUPPRESSION_FACTOR

            # If energy is below the raised threshold, don't even check VAD
            if energy < effective_threshold:
                # Still count silence if we were in speech
                if self._in_speech:
                    self._silence_frame_count += 1
                    if self._silence_frame_count >= self.config.silence_limit_frames:
                        logger.debug("Speech END detected (silence during TTS)")
                        self._in_speech = False
                        self._silence_frame_count = 0
                return self._in_speech

        pcm_int16 = self._float32_to_int16(audio_float32)
        is_speech = self._vad.is_speech(pcm_int16, self.sample_rate)

        if is_speech:
            if not self._in_speech:
                logger.debug(
                    "Speech START detected%s",
                    " (during TTS)" if self._tts_active else "",
                )
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
        self._tts_active = False

    def set_tts_active(self, active: bool) -> None:
        """Set TTS playback state for echo suppression."""
        if active != self._tts_active:
            self._tts_active = active
            logger.debug("Echo suppression: TTS %s", "active" if active else "inactive")

    @staticmethod
    def _calculate_energy(audio: np.ndarray) -> float:
        """Calculate RMS energy of audio frame."""
        return float(np.sqrt(np.mean(audio**2)))
