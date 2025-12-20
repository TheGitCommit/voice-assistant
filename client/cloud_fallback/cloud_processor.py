"""
Cloud-based audio processing pipeline (no local server).
Direct REST API calls to Deepgram and DeepSeek.
CLIENT-SIDE MODULE.
"""

import asyncio
import logging
import re

import numpy as np

from client.cloud_fallback.cloud_config import CLOUD_CONFIG
from client.cloud_fallback.deepgram_stt import DeepgramSTT
from client.cloud_fallback.deepgram_tts import DeepgramTTS
from client.cloud_fallback.deepseek_llm import DeepSeekLLM
from client.audio.audio_capture import AudioCapture
from client.audio.audio_playback import AudioPlayback
from client.audio.vad import VoiceActivityDetector
from client.config import ClientConfig

logger = logging.getLogger(__name__)


class CloudAudioProcessor:
    """
    The CloudAudioProcessor class facilitates processing audio via cloud-based STT (Speech-to-Text),
    TTS (Text-to-Speech), and LLM (Large Language Model) services. It acts as a central controller for
    managing audio capture, playback, and processing audio chunks in real-time.

    This class leverages cloud service APIs for speech transcription, generating text responses,
    and synthesizing speech playback. It maintains the audio processing flow, such as managing audio
    buffers and leveraging voice activity detection (VAD) for identifying and processing speech events.

    :ivar config: Configuration object containing parameters for audio capture, playback, VAD, and
        cloud services.
    :type config: ClientConfig
    :ivar capture: Component for capturing audio data from a microphone input.
    :type capture: AudioCapture
    :ivar playback: Component for playing synthesized audio responses.
    :type playback: AudioPlayback
    :ivar vad: Component for detecting the presence of speech in real-time audio.
    :type vad: VoiceActivityDetector
    :ivar stt: Client instance for performing speech-to-text transcription using the configured cloud API.
    :type stt: DeepgramSTT
    :ivar tts: Client instance for performing text-to-speech synthesis using the configured cloud API.
    :type tts: DeepgramTTS
    :ivar llm: Client instance for interacting with a large language model to generate responses.
    :type llm: DeepSeekLLM
    """

    def __init__(self, config: ClientConfig):
        self.config = config

        # Audio components
        self.capture = AudioCapture(config.capture)
        self.playback = AudioPlayback(config.playback)
        self.vad = VoiceActivityDetector(config.vad, config.capture.sample_rate)

        # Cloud API clients
        self.stt = DeepgramSTT(CLOUD_CONFIG.deepgram)
        self.tts = DeepgramTTS(CLOUD_CONFIG.deepgram)
        self.llm = DeepSeekLLM(CLOUD_CONFIG.deepseek)

        # State
        self._running = False
        self._buffer = bytearray()

        logger.info("CloudAudioProcessor initialized")

    async def run(self) -> None:
        logger.info("=== Running in CLOUD MODE ===")
        logger.info("STT/TTS: Deepgram | LLM: DeepSeek")

        self._running = True

        try:
            self.capture.start()
            self.playback.start()

            logger.info("Microphone active (cloud mode)...")
            print("Speak now (cloud mode - Deepgram + DeepSeek)")

            # Process audio chunks
            while self._running:
                # Read audio chunk
                chunk = self.capture.read(timeout=1.0)

                # Accumulate for VAD processing
                self._buffer.extend(chunk.tobytes())

                # Check VAD state
                is_speech = self.vad.process_frame(chunk)

                # If speech ended, process the utterance
                if not is_speech and len(self._buffer) > 0:
                    # Get duration
                    num_samples = len(self._buffer) // 4  # float32 = 4 bytes
                    duration = num_samples / self.config.capture.sample_rate

                    # Only process if long enough
                    if duration >= 0.5:  # Min 500ms
                        utterance = bytes(self._buffer)
                        self._buffer.clear()

                        # Process in background
                        asyncio.create_task(self._process_utterance(utterance))
                    else:
                        self._buffer.clear()

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

            # STT through deepgram
            transcript = await self.stt.transcribe(pcm, self.config.capture.sample_rate)

            if not transcript:
                logger.warning("Empty transcription")
                return

            print(f"You: {transcript}")

            response = await self.llm.get_completion(transcript)

            if not response:
                return

            # strip markdown so TTS doesn't read it
            clean_text = re.sub(r"[*#_`~-]", "", response)

            # truncate to 2000 chars
            if len(clean_text) > 2000:
                logger.warning("Truncating response to 2000 chars for TTS")
                clean_text = clean_text[:1997] + "..."

            print(f"Assistant: {response}")

            audio = await self.tts.synthesize(clean_text)

            if audio:
                self.playback.play(audio)
            else:
                logger.warning("TTS failed")

        except Exception:
            logger.exception("Error processing utterance")

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

    def stop(self) -> None:
        self._running = False
