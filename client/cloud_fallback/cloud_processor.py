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
    """Cloud fallback: Deepgram STT/TTS + DeepSeek LLM."""

    def __init__(self, config: ClientConfig):
        self.config = config

        self.capture = AudioCapture(config.capture)
        self.playback = AudioPlayback(config.playback)
        self.vad = VoiceActivityDetector(config.vad, config.capture.sample_rate)

        self.stt = DeepgramSTT(CLOUD_CONFIG.deepgram)
        self.tts = DeepgramTTS(CLOUD_CONFIG.deepgram)
        self.llm = DeepSeekLLM(CLOUD_CONFIG.deepseek)

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

            while self._running:
                chunk = self.capture.read(timeout=1.0)
                self._buffer.extend(chunk.tobytes())
                is_speech = self.vad.process_frame(chunk)

                if not is_speech and len(self._buffer) > 0:
                    num_samples = len(self._buffer) // 4
                    duration = num_samples / self.config.capture.sample_rate

                    if duration >= 0.5:
                        utterance = bytes(self._buffer)
                        self._buffer.clear()
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
