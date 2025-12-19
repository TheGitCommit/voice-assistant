"""
Audio processing pipeline: VAD -> STT -> LLM -> TTS.
"""

import asyncio
import logging
from typing import TYPE_CHECKING

import numpy as np

from server.config import CONFIG
from server.core.vad import VoiceActivityDetector
from server.inference.whisper_stt import WhisperSTT
from server.inference.piper_tts import PiperTTS
from server.inference.llm_client import LLMClient
from server.utils.logging_utils import RateLimitedLogger

if TYPE_CHECKING:
    from server.networking.websocket_connection import WebSocketConnection

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Processes audio and handles interactions for a given WebSocket connection.

    The AudioProcessor class integrates various components, including Voice Activity
    Detection (VAD), Speech-to-Text (STT), Large Language Model (LLM), and Text-to-Speech
    (TTS). It processes incoming audio streams in an asynchronous manner, detects complete
    utterances through VAD, and passes these utterances through the STT -> LLM -> TTS
    pipeline to generate and send responses back through the WebSocket connection. The
    class also supports direct text input for text-only clients or debugging purposes.

    :ivar connection: Active WebSocket connection instance for audio processing.
    :type connection: WebSocketConnection
    """

    def __init__(self, connection: "WebSocketConnection"):
        self.connection = connection

        # Initialize components
        self.vad = VoiceActivityDetector(CONFIG["vad"], CONFIG["audio"])
        self.stt = WhisperSTT(CONFIG["whisper"])
        self.tts = PiperTTS(CONFIG["piper"])
        self.llm = LLMClient(CONFIG["llama"])

        # Rate-limited logger for periodic status updates
        self._rl_logger = RateLimitedLogger(
            logger, CONFIG["logging"].rate_limit_seconds
        )

        self._chunk_count = 0

        logger.info(
            "AudioProcessor initialized for connection %s", connection.connection_id
        )

    async def run(self) -> None:
        """
        Main processing loop.

        Consumes audio chunks, performs VAD, and triggers STT/LLM/TTS pipeline
        when complete utterances are detected.
        """
        logger.info("[%s] AudioProcessor started", self.connection.connection_id)

        try:
            while True:
                # Get audio chunk from queue
                chunk = await self.connection.audio_queue.get()
                self._chunk_count += 1

                # Process through VAD
                utterance = self.vad.process_chunk(chunk)

                # Periodic debug logging
                if self._chunk_count % 50 == 0:
                    energy = self._calculate_energy(chunk)
                    vad_info = self.vad.get_state_debug_info()
                    self._rl_logger.info(
                        "audio_status",
                        "[%s] Audio status: chunks=%d energy=%.4f vad_state=%s buffer=%.2fs",
                        self.connection.connection_id,
                        self._chunk_count,
                        energy,
                        vad_info["state"],
                        vad_info["buffer_duration"],
                    )

                # If complete utterance detected, process it
                if utterance:
                    await self._process_utterance(utterance)

        except asyncio.CancelledError:
            logger.info("[%s] AudioProcessor cancelled", self.connection.connection_id)
            raise
        except Exception:
            logger.exception(
                "[%s] AudioProcessor fatal error", self.connection.connection_id
            )
            raise

    async def _process_utterance(self, utterance: bytes) -> None:
        """
        Process a complete utterance through STT -> LLM -> TTS pipeline.

        Runs in background task to avoid blocking audio queue processing.
        """
        asyncio.create_task(self._pipeline_task(utterance))

    async def _pipeline_task(self, utterance: bytes) -> None:
        """
        Execute the full processing pipeline for an utterance.

        Args:
            utterance: Complete audio utterance as PCM bytes
        """
        try:
            conn_id = self.connection.connection_id

            # Convert bytes to numpy array for STT
            pcm = np.frombuffer(utterance, dtype=np.float32)
            duration = len(pcm) / CONFIG["audio"].sample_rate

            logger.info(
                "[%s] Processing utterance: duration=%.2fs samples=%d",
                conn_id,
                duration,
                len(pcm),
            )

            # Speech-to-Text
            transcript = self.stt.transcribe(pcm)

            if not transcript:
                logger.warning("[%s] Empty transcription, skipping pipeline", conn_id)
                return

            logger.info("[%s] Transcription: %s", conn_id, transcript)

            # Send transcription to client
            await self.connection.send_event(
                {
                    "type": "transcription",
                    "text": transcript,
                }
            )

            # LLM inference
            response = await self.llm.get_completion(transcript)

            if not response:
                logger.warning("[%s] Empty LLM response", conn_id)
                return

            logger.info("[%s] LLM response: %s", conn_id, response[:100])

            # Send LLM response to client
            await self.connection.send_event(
                {
                    "type": "llm_response",
                    "text": response,
                }
            )

            # Text-to-Speech
            audio_bytes = self.tts.synthesize(response)

            if audio_bytes:
                await self.connection.send_audio(audio_bytes)
            else:
                logger.warning("[%s] TTS synthesis failed or empty", conn_id)

        except Exception:
            logger.exception(
                "[%s] Pipeline processing failed",
                self.connection.connection_id,
            )

    async def handle_text_message(self, text: str) -> None:
        """
        Handle direct text input (bypass STT).

        Used for debugging or text-only clients.
        """
        if not text or not text.strip():
            return

        logger.info("[%s] Text input: %s", self.connection.connection_id, text)

        # Skip STT, go directly to LLM
        response = await self.llm.get_completion(text)

        if not response:
            return

        await self.connection.send_event(
            {
                "type": "llm_response",
                "text": response,
            }
        )

        audio_bytes = self.tts.synthesize(response)
        if audio_bytes:
            await self.connection.send_audio(audio_bytes)

    @staticmethod
    def _calculate_energy(chunk: bytes) -> float:
        """Calculate RMS energy for logging."""
        pcm = np.frombuffer(chunk, dtype=np.float32)
        return float(np.sqrt(np.mean(pcm**2)))
