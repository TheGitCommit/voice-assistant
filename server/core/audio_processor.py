import asyncio
import logging
import re
from typing import TYPE_CHECKING, Optional

import numpy as np

from server.config import CONFIG
from server.core.vad import VoiceActivityDetector
from server.inference.whisper_stt import WhisperSTT
from server.inference.tts_factory import create_tts
from server.inference.llm_client import LLMClient
from server.rag.rag_router import augment_with_rag
from server.utils.logging_utils import RateLimitedLogger
from server.utils.timing import TimingStats
from server.utils.latency_monitor import get_latency_monitor, RequestTracker

if TYPE_CHECKING:
    from server.networking.websocket_connection import WebSocketConnection

logger = logging.getLogger(__name__)

# Keywords that trigger barge-in via STT
BARGE_IN_KEYWORDS = {"stop", "pause", "shut up", "cancel", "quiet", "enough", "wait"}


class AudioProcessor:
    """Pipeline: VAD → STT → LLM (streaming) → TTS → audio out with semantic barge-in.

    Plain English:
    -------------
    This is the "brain" that orchestrates the entire voice assistant pipeline:

    1. Audio comes in from the client (your microphone)
    2. VAD detects when you've finished speaking
    3. Whisper transcribes your speech to text
    4. The LLM generates a response (streaming, word by word)
    5. As clauses complete, TTS synthesizes audio
    6. Audio is streamed back to the client (your speakers)

    The "Waterfall" model:
    - We don't wait for the full LLM response before starting TTS
    - As soon as we have a complete clause (ends with comma or period), we synthesize
    - This means you hear audio ~1-2s after speaking, not 5-6s

    Semantic Barge-in:
    - If you speak while the assistant is talking, we transcribe your speech
    - If you said "stop", "cancel", etc., we interrupt immediately
    - If you said something else, we queue it for after the response

    What happens if this is removed?
    --------------------------------
    The voice assistant stops working entirely. This is the central coordinator.
    """

    def __init__(self, connection: "WebSocketConnection", session_id: str = None):
        self.connection = connection
        self.vad = VoiceActivityDetector(CONFIG["vad"], CONFIG["audio"])
        self.stt = WhisperSTT(CONFIG["whisper"])
        self.tts = create_tts(CONFIG)  # Uses TTS_PROVIDER env var (piper/kokoro)

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
        self._current_pipeline_task: Optional[asyncio.Task] = None
        self._bargein_buffer: list[bytes] = []

    def interrupt(self, reason: str = "barge-in") -> None:
        """Cancel current pipeline and signal client to stop playback."""
        if self._pipeline_running:
            self._interrupted = True
            logger.info(
                "[%s] Pipeline interrupted (%s)", self.connection.connection_id, reason
            )
            if self._current_pipeline_task and not self._current_pipeline_task.done():
                self._current_pipeline_task.cancel()

            # Immediately notify client to kill local audio buffers
            asyncio.create_task(self.connection.send_event({"type": "tts_stop"}))

    async def run(self) -> None:
        try:
            while True:
                chunk = await self.connection.audio_queue.get()
                self._chunk_count += 1

                # Tell VAD if we are speaking to raise threshold (Echo Suppression)
                utterance = self.vad.process_chunk(
                    chunk, is_tts_active=self._tts_active
                )

                if utterance:
                    if self._tts_active:
                        # Semantic check: Transcription-based barge-in
                        await self._handle_bargein_speech(utterance)
                    else:
                        await self._process_utterance(utterance)

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[%s] AudioProcessor error", self.connection.connection_id)
            raise

    async def _handle_bargein_speech(self, utterance: bytes) -> None:
        """Check for stop keywords during assistant playback."""
        pcm = np.frombuffer(utterance, dtype=np.float32)
        transcript = await self.stt.transcribe_async(pcm)

        if transcript:
            transcript_lower = transcript.lower().strip()
            if any(kw in transcript_lower for kw in BARGE_IN_KEYWORDS):
                logger.info(
                    "[%s] Semantic barge-in keyword found in: '%s'",
                    self.connection.connection_id,
                    transcript,
                )
                self.interrupt(reason="keyword")
            else:
                # User is talking but didn't say stop; queue for after TTS
                self._bargein_buffer.append(utterance)

    async def _streaming_llm_tts_pipeline(
        self, input_text: str, conn_id: str, tracker: RequestTracker
    ) -> None:
        """Clause-level splitting for faster perceived speed (TTFA)."""
        sentence_buffer = ""
        # Aggressive split: includes commas to start TTS faster
        split_pattern = re.compile(r"([.!?]|,)\s+")

        self._tts_active = True
        await self.connection.send_event({"type": "tts_start"})

        first_token_recorded = False
        full_response = ""
        tts_tasks = []

        try:
            async for chunk in self.llm.stream_completion(input_text):
                if self._interrupted:
                    break

                # Record first token arrival (TTFT - Time To First Token)
                if not first_token_recorded and chunk.strip():
                    tracker.record_llm_first_token()
                    first_token_recorded = True

                full_response += chunk
                sentence_buffer += chunk
                # Partial text UI update
                await self.connection.send_event(
                    {"type": "partial_llm_response", "text": chunk}
                )

                # Process split boundaries
                while not self._interrupted:
                    match = split_pattern.search(sentence_buffer)
                    if not match:
                        break

                    clause = sentence_buffer[: match.end()].strip()
                    sentence_buffer = sentence_buffer[match.end() :]

                    if len(clause.split()) > 3:  # Avoid synthesizing tiny fragments
                        task = asyncio.create_task(
                            self._synthesize_and_send(clause, tracker)
                        )
                        tts_tasks.append(task)

            if sentence_buffer.strip() and not self._interrupted:
                task = asyncio.create_task(
                    self._synthesize_and_send(sentence_buffer.strip(), tracker)
                )
                tts_tasks.append(task)

            # Record LLM completion metrics
            # Estimate token count: ~4 chars per token (rough estimate)
            estimated_tokens = len(full_response) // 4 if full_response else 0
            tracker.record_llm_complete(
                response_length=len(full_response),
                token_count=estimated_tokens,
            )

            # Wait for all TTS tasks to complete
            if tts_tasks:
                await asyncio.gather(*tts_tasks, return_exceptions=True)

            # Record TTS completion
            tracker.record_tts_complete()

        finally:
            self._tts_active = False
            await self.connection.send_event({"type": "tts_stop"})

    async def _synthesize_and_send(self, text: str, tracker: RequestTracker):
        audio = await self.tts.synthesize(text)
        if audio and not self._interrupted:
            # Only record first audio once (first TTS chunk)
            if tracker.metrics.tts_first_audio == 0:
                tracker.record_tts_first_audio()
            await self.connection.send_audio(audio)

    async def _process_utterance(self, utterance: bytes) -> None:
        """Full pipeline: STT → LLM → TTS."""
        if self._pipeline_running:
            logger.warning(
                "[%s] Pipeline already running, queuing utterance",
                self.connection.connection_id,
            )
            self._bargein_buffer.append(utterance)
            return

        self._pipeline_running = True
        self._interrupted = False

        conn_id = self.connection.connection_id
        tracker = get_latency_monitor().start_request(conn_id)

        try:
            # STT
            pcm = np.frombuffer(utterance, dtype=np.float32)
            duration = len(pcm) / CONFIG["audio"].sample_rate

            logger.info("[%s] Processing utterance: %.2fs", conn_id, duration)

            transcript = await self.stt.transcribe_async(pcm)
            tracker.record_stt_complete(
                transcript_length=len(transcript) if transcript else 0,
                audio_duration=duration,
            )

            if not transcript:
                logger.warning("[%s] Empty transcription", conn_id)
                return

            logger.info("[%s] User: %s", conn_id, transcript)
            await self.connection.send_event(
                {"type": "transcription", "text": transcript}
            )

            # RAG: Check if we need to augment with knowledge base context
            if CONFIG["rag"].enabled:
                rag_context, was_augmented = await augment_with_rag(
                    transcript, force=False
                )
                if was_augmented:
                    logger.info(
                        "[%s] RAG activated: retrieved context (length=%d)",
                        conn_id,
                        len(rag_context),
                    )
                    # Prepend context to the query
                    augmented_query = f"{rag_context}\n\nUser: {transcript}"
                else:
                    augmented_query = transcript
            else:
                augmented_query = transcript

            # LLM + TTS pipeline
            self._current_pipeline_task = asyncio.create_task(
                self._streaming_llm_tts_pipeline(augmented_query, conn_id, tracker)
            )
            await self._current_pipeline_task

        except asyncio.CancelledError:
            logger.info("[%s] Pipeline cancelled", conn_id)
        except Exception:
            logger.exception("[%s] Pipeline error", conn_id)
        finally:
            self._pipeline_running = False
            self._current_pipeline_task = None
            await tracker.finish()

            # Process any queued barge-in utterances
            if self._bargein_buffer:
                queued = self._bargein_buffer.pop(0)
                asyncio.create_task(self._process_utterance(queued))

    def load_session(self, session_id: str) -> bool:
        """Load a previous conversation session."""
        self._session_id = session_id
        return self.llm.load_history(session_id)

    def save_session(self) -> bool:
        """Save the current conversation session."""
        return self.llm.save_history()

    @property
    def session_id(self) -> str:
        """Get the current session ID."""
        return self._session_id

    async def handle_text_message(self, text: str) -> None:
        """Handle a text message from the client (typed input)."""
        if not text or not text.strip():
            return

        if self._pipeline_running:
            logger.warning(
                "[%s] Pipeline running, ignoring text message",
                self.connection.connection_id,
            )
            return

        self._pipeline_running = True
        self._interrupted = False

        conn_id = self.connection.connection_id
        tracker = get_latency_monitor().start_request(conn_id)

        try:
            logger.info("[%s] User (text): %s", conn_id, text)
            await self.connection.send_event({"type": "transcription", "text": text})

            # RAG: Check if we need to augment with knowledge base context
            if CONFIG["rag"].enabled:
                rag_context, was_augmented = await augment_with_rag(text, force=False)
                if was_augmented:
                    logger.info(
                        "[%s] RAG activated: retrieved context (length=%d)",
                        conn_id,
                        len(rag_context),
                    )
                    # Prepend context to the query
                    augmented_query = f"{rag_context}\n\nUser: {text}"
                else:
                    augmented_query = text
            else:
                augmented_query = text

            # LLM + TTS pipeline
            self._current_pipeline_task = asyncio.create_task(
                self._streaming_llm_tts_pipeline(augmented_query, conn_id, tracker)
            )
            await self._current_pipeline_task

        except asyncio.CancelledError:
            logger.info("[%s] Text pipeline cancelled", conn_id)
        except Exception:
            logger.exception("[%s] Text pipeline error", conn_id)
        finally:
            self._pipeline_running = False
            self._current_pipeline_task = None
            await tracker.finish()
