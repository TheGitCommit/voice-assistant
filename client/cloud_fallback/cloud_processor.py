import asyncio
import logging
import queue
import re
from enum import Enum

import numpy as np

from client.cloud_fallback.cloud_config import CLOUD_CONFIG
from client.cloud_fallback.deepgram_stt import DeepgramSTT
from client.cloud_fallback.deepgram_tts import DeepgramTTS
from client.cloud_fallback.deepseek_llm import DeepSeekLLM
from client.audio.audio_capture import AudioCapture
from client.audio.audio_playback import AudioPlayback
from client.audio.vad import VoiceActivityDetector
from client.audio.wake_word import WakeWordDetector
from client.audio.feedback import AudioFeedback
from client.config.config import ClientConfig

logger = logging.getLogger(__name__)


class CloudState(Enum):
    """Cloud processor state machine states."""

    WAITING_FOR_WAKE = "waiting_for_wake"
    WAKE_DETECTED = "wake_detected"
    LISTENING = "listening"


class CloudAudioProcessor:
    """Cloud fallback: Deepgram STT/TTS + DeepSeek LLM."""

    def __init__(self, config: ClientConfig):
        self.config = config

        self.capture = AudioCapture(config.capture)
        self.playback = AudioPlayback(config.playback)
        self.vad = VoiceActivityDetector(config.vad, config.capture.sample_rate)
        self.wake_word = WakeWordDetector(config.wake_word, config.capture.sample_rate)
        self.feedback = AudioFeedback(config.playback.sample_rate)

        self.stt = DeepgramSTT(CLOUD_CONFIG.deepgram)
        self.tts = DeepgramTTS(CLOUD_CONFIG.deepgram)
        self.llm = DeepSeekLLM(CLOUD_CONFIG.deepseek)

        self._running = False
        self._state = CloudState.WAITING_FOR_WAKE
        self._buffer = bytearray()
        self._was_in_speech = False  # Track previous speech state

        logger.info("CloudAudioProcessor initialized (wake-word gated mode)")

    async def run(self) -> None:
        logger.info("=== Running in CLOUD MODE ===")
        logger.info("STT/TTS: Deepgram | LLM: DeepSeek")

        self._running = True

        try:
            self.capture.start()
            self.playback.start()

            # Wait for pre-roll buffer to fill
            await self.capture.wait_for_preroll()
            logger.info("Pre-roll complete, audio capture ready")
            logger.info("Waiting for wake word: %s", self.config.wake_word.model_name)
            print(f"Waiting for wake word: {self.config.wake_word.model_name}")

            while self._running:
                try:
                    chunk = self.capture.read(timeout=1.0)

                    if self._state == CloudState.WAITING_FOR_WAKE:
                        # Listen for wake word
                        if self.wake_word.process_frame(chunk):
                            logger.info("Wake word detected! Starting to listen...")
                            self._state = CloudState.WAKE_DETECTED
                            # Play feedback tone
                            try:
                                tone = self.feedback.get_tone("listening")
                                if tone:
                                    self.playback.play(tone)
                            except Exception:
                                pass  # Ignore feedback errors
                            # Brief activation delay
                            await asyncio.sleep(
                                self.config.wake_word.activation_delay_ms / 1000.0
                            )
                            self._state = CloudState.LISTENING
                            self._buffer.clear()
                            self._was_in_speech = False
                            logger.info("Now listening for speech...")
                            print("Speak now (cloud mode - Deepgram + DeepSeek)")

                    elif self._state == CloudState.LISTENING:
                        # Process audio with VAD-based utterance detection
                        self._buffer.extend(chunk.tobytes())
                        is_speech = self.vad.process_frame(chunk)

                        # Only process when transitioning from speech to silence
                        # This ensures we capture complete utterances, not silence/noise
                        if (
                            self._was_in_speech
                            and not is_speech
                            and len(self._buffer) > 0
                        ):
                            num_samples = len(self._buffer) // 4
                            duration = num_samples / self.config.capture.sample_rate

                            if duration >= 0.5:
                                utterance = bytes(self._buffer)
                                self._buffer.clear()
                                asyncio.create_task(self._process_utterance(utterance))
                            else:
                                # Utterance too short, likely false positive
                                logger.debug(
                                    "Dropping short utterance: %.2fs", duration
                                )
                                self._buffer.clear()
                        elif not is_speech:
                            # Reset buffer if we're in silence and weren't in speech
                            # This prevents accumulation of silence/noise
                            if len(self._buffer) > 0 and not self._was_in_speech:
                                # Only keep a small amount of pre-roll audio (max 0.5s)
                                max_pre_roll_samples = int(
                                    0.5 * self.config.capture.sample_rate
                                )
                                max_pre_roll_bytes = (
                                    max_pre_roll_samples * 4
                                )  # float32 = 4 bytes
                                if len(self._buffer) > max_pre_roll_bytes:
                                    # Keep only the most recent pre-roll audio
                                    self._buffer = self._buffer[-max_pre_roll_bytes:]

                        self._was_in_speech = is_speech

                except queue.Empty:
                    # Timeout is normal - no audio available yet, continue polling
                    pass

                await asyncio.sleep(0.001)

        except asyncio.CancelledError:
            logger.info("Cloud processor cancelled")
            raise
        except Exception:
            logger.exception("Cloud processor error")
            raise
        finally:
            await self._cleanup()

    async def _process_utterance(self, utterance: bytes) -> None:
        try:
            pcm = np.frombuffer(utterance, dtype=np.float32)
            duration = len(pcm) / self.config.capture.sample_rate

            logger.info("Processing utterance: %.2fs", duration)

            transcript = await self.stt.transcribe(pcm, self.config.capture.sample_rate)

            if not transcript:
                logger.warning("Empty transcription")
                return

            print(f"You: {transcript}")

            response = await self.llm.get_completion(transcript)

            if not response:
                return

            clean_text = re.sub(r"[*#_`~-]", "", response)

            if len(clean_text) > 2000:
                logger.warning("Truncating response to 2000 chars for TTS")
                clean_text = clean_text[:1997] + "..."

            print(f"Assistant: {response}")

            audio = await self.tts.synthesize(clean_text)

            if audio:
                self.playback.play(audio)
            else:
                logger.warning("TTS failed")

            # Return to wake word listening after response complete
            self._state = CloudState.WAITING_FOR_WAKE
            self.wake_word.reset()
            self._buffer.clear()
            self._was_in_speech = False
            logger.info("Returning to wake word listening")
            print(f"Waiting for wake word: {self.config.wake_word.model_name}")

        except Exception:
            logger.exception("Error processing utterance")
            # Return to wake word listening on error
            self._state = CloudState.WAITING_FOR_WAKE
            self.wake_word.reset()
            self._buffer.clear()
            self._was_in_speech = False

    async def _cleanup(self) -> None:
        logger.info("Cleaning up cloud processor")

        self._running = False

        try:
            self.capture.stop()
        except Exception:
            logger.exception("Error stopping capture")

        try:
            self.playback.close()
        except Exception:
            logger.exception("Error closing playback")

        try:
            self.wake_word.close()
        except Exception:
            logger.exception("Error closing wake word detector")

    def stop(self) -> None:
        self._running = False
