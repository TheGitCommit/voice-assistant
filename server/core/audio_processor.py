import asyncio
import logging
import time
import re
from typing import TYPE_CHECKING

import numpy as np

from server.config import CONFIG
from server.core.vad import VoiceActivityDetector
from server.inference.whisper_stt import WhisperSTT
from server.inference.piper_tts import PiperTTS
from server.inference.llm_client import LLMClient
from server.utils.logging_utils import RateLimitedLogger
from server.utils.timing import TimingStats

if TYPE_CHECKING:
    from server.networking.websocket_connection import WebSocketConnection

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Pipeline: VAD → STT → LLM (streaming) → TTS → audio out."""

    def __init__(self, connection: "WebSocketConnection", session_id: str = None):
        self.connection = connection

        self.vad = VoiceActivityDetector(CONFIG["vad"], CONFIG["audio"])
        self.stt = WhisperSTT(CONFIG["whisper"])
        self.tts = PiperTTS(CONFIG["piper"])

        # Use provided session_id or connection_id for persistence
        self._session_id = session_id or connection.connection_id
        self.llm = LLMClient(CONFIG["llama"], session_id=self._session_id)

        self._rl_logger = RateLimitedLogger(
            logger, CONFIG["logging"].rate_limit_seconds
        )
        self.timing_stats = TimingStats()

        self._chunk_count = 0
        self._pipeline_running = False
        self._interrupted = False
        self._tts_active = False

        logger.info(
            "AudioProcessor initialized for connection %s (session: %s)",
            connection.connection_id,
            self._session_id,
        )

    def load_session(self, session_id: str) -> bool:
        """Load a previous conversation session."""
        self._session_id = session_id
        return self.llm.load_history(session_id)

    def save_session(self) -> bool:
        """Save current conversation to disk."""
        return self.llm.save_history()

    @property
    def session_id(self) -> str:
        return self._session_id

    def interrupt(self) -> None:
        """Signal to cancel the current pipeline (barge-in)."""
        if self._pipeline_running:
            self._interrupted = True
            logger.info(
                "[%s] Pipeline interrupted (barge-in)", self.connection.connection_id
            )

    @property
    def is_tts_active(self) -> bool:
        return self._tts_active

    async def run(self) -> None:
        logger.info("[%s] AudioProcessor started", self.connection.connection_id)

        try:
            while True:
                chunk = await self.connection.audio_queue.get()
                self._chunk_count += 1

                utterance = self.vad.process_chunk(chunk)

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
        if self._pipeline_running:
            self.interrupt()
            await asyncio.sleep(0.1)

        asyncio.create_task(self._pipeline_task(utterance))

    async def _pipeline_task(self, utterance: bytes) -> None:
        pipeline_start = time.perf_counter()
        conn_id = self.connection.connection_id
        self._pipeline_running = True
        self._interrupted = False

        try:
            pcm = np.frombuffer(utterance, dtype=np.float32)
            audio_duration = len(pcm) / CONFIG["audio"].sample_rate

            logger.info(
                "[%s] Processing utterance: duration=%.2fs samples=%d",
                conn_id,
                audio_duration,
                len(pcm),
            )

            if self._interrupted:
                return

            stt_start = time.perf_counter()
            transcript = await self.stt.transcribe_async(pcm)
            stt_latency = time.perf_counter() - stt_start
            self.timing_stats.record("stt", stt_latency)

            if self._interrupted:
                return

            if not transcript:
                logger.warning("[%s] Empty transcription", conn_id)
                return

            logger.info(
                "[%s] Transcription: %s (%.3fs)", conn_id, transcript, stt_latency
            )
            await self.connection.send_event(
                {"type": "transcription", "text": transcript}
            )

            await self._streaming_llm_tts_pipeline(transcript, conn_id, pipeline_start)

        except Exception:
            logger.exception("[%s] Pipeline failed", conn_id)
        finally:
            self._pipeline_running = False
            self._tts_active = False
            self._interrupted = False

    async def handle_text_message(self, text: str) -> None:
        """Direct text input, bypasses STT."""
        if not text or not text.strip():
            return

        logger.info("[%s] Text input: %s", self.connection.connection_id, text)

        await self.connection.send_event({"type": "transcription", "text": text})
        await self._streaming_llm_tts_pipeline(
            text, self.connection.connection_id, time.perf_counter()
        )

    async def _streaming_llm_tts_pipeline(
        self, input_text: str, conn_id: str, pipeline_start: float
    ) -> None:
        llm_start = time.perf_counter()
        accumulated_text = ""
        pending_sentence = ""

        await self.connection.send_event({"type": "tts_start"})
        self._tts_active = True

        # TTS prefetch queue: (sentence, synthesis_task)
        pending_audio_task = None
        pending_audio_sentence = None

        async def send_pending_audio():
            """Send any pending audio that was prefetched."""
            nonlocal pending_audio_task, pending_audio_sentence
            if pending_audio_task is not None:
                audio_chunk = await pending_audio_task
                if audio_chunk and not self._interrupted:
                    await self.connection.send_audio(audio_chunk)
                    logger.debug(
                        "[%s] Streamed: %s (%d bytes)",
                        conn_id,
                        pending_audio_sentence,
                        len(audio_chunk),
                    )
                pending_audio_task = None
                pending_audio_sentence = None

        try:
            async for chunk in self.llm.stream_completion(input_text):
                if self._interrupted:
                    logger.info(
                        "[%s] Pipeline interrupted during LLM streaming", conn_id
                    )
                    await self.connection.send_event({"type": "tts_stop"})
                    return

                if not chunk:
                    continue

                accumulated_text += chunk
                pending_sentence += chunk

                await self.connection.send_event(
                    {
                        "type": "partial_llm_response",
                        "text": chunk,
                        "full_text": accumulated_text,
                    }
                )

                while True:
                    if self._interrupted:
                        break

                    match = re.search(r"([.!?]|[,])\s+", pending_sentence)
                    if not match:
                        break

                    end_pos = match.end()
                    sentence = pending_sentence[:end_pos].strip()
                    pending_sentence = pending_sentence[end_pos:]

                    if sentence:
                        # Send any previously prefetched audio
                        await send_pending_audio()

                        # Start synthesizing this sentence (prefetch)
                        pending_audio_task = asyncio.create_task(
                            self.tts.synthesize(sentence + " ")
                        )
                        pending_audio_sentence = sentence

            # Send any remaining prefetched audio
            await send_pending_audio()

            # Handle final incomplete sentence
            if pending_sentence.strip() and not self._interrupted:
                audio_chunk = await self.tts.synthesize(pending_sentence)
                if audio_chunk and not self._interrupted:
                    await self.connection.send_audio(audio_chunk)
                    logger.debug(
                        "[%s] Streamed final: %s (%d bytes)",
                        conn_id,
                        pending_sentence,
                        len(audio_chunk),
                    )

            await self.connection.send_event({"type": "tts_stop"})

            if not self._interrupted:
                await self.connection.send_event(
                    {"type": "llm_response", "text": accumulated_text}
                )

                llm_latency = time.perf_counter() - llm_start
                e2e_latency = time.perf_counter() - pipeline_start
                self.timing_stats.record("llm", llm_latency)
                self.timing_stats.record("end_to_end", e2e_latency)

                logger.info(
                    "[%s] Streaming complete: %d chars in %.3fs",
                    conn_id,
                    len(accumulated_text),
                    e2e_latency,
                )

        except Exception:
            logger.exception("[%s] Streaming failed", conn_id)
        finally:
            self._tts_active = False

    @staticmethod
    def _calculate_energy(chunk: bytes) -> float:
        pcm = np.frombuffer(chunk, dtype=np.float32)
        return float(np.sqrt(np.mean(pcm**2)))
