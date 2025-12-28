"""
WebSocket client that streams audio to server and plays responses.

This version uses wake word detection to gate audio streaming.
"""

import asyncio
import json
import logging
from enum import Enum

import websockets

from client.config.config import ClientConfig
from client.audio.audio_capture import AudioCapture
from client.audio.audio_playback import AudioPlayback
from client.audio.vad import VoiceActivityDetector
from client.audio.feedback import AudioFeedback
from client.audio.wake_word import WakeWordDetector

logger = logging.getLogger(__name__)

# Note: Barge-in is now handled server-side via keyword detection in transcription


class ClientState(Enum):
    """Client state machine states."""

    WAITING_FOR_WAKE = "waiting_for_wake"
    WAKE_DETECTED = "wake_detected"
    STREAMING = "streaming"


class VoiceAssistantClient:
    """WebSocket client that streams audio to server and plays responses."""

    def __init__(self, config: ClientConfig):
        self.config = config

        self.capture = AudioCapture(config.capture)
        self.playback = AudioPlayback(config.playback)
        self.vad = VoiceActivityDetector(config.vad, config.capture.sample_rate)
        self.wake_word = WakeWordDetector(config.wake_word, config.capture.sample_rate)
        self.feedback = AudioFeedback(config.playback.sample_rate)

        self._websocket = None
        self._running = False
        self._state = ClientState.WAITING_FOR_WAKE
        self._tts_active = False
        self._interrupt_sent = False
        self._feedback_enabled = True

        logger.info("VoiceAssistantClient initialized (wake-word gated mode)")

    async def run(self) -> None:
        logger.info("Connecting to server: %s", self.config.server.url)

        try:
            async with websockets.connect(
                self.config.server.url,
                max_size=None,
                ping_interval=60,
                ping_timeout=30,
            ) as websocket:
                self._websocket = websocket
                self._running = True

                logger.info("Connected to server")

                self.capture.start()
                self.playback.start()
                # Don't send hello immediately - wait for wake word first
                # Hello will be sent when streaming starts

                send_task = asyncio.create_task(self._send_loop())
                recv_task = asyncio.create_task(self._receive_loop())

                done, pending = await asyncio.wait(
                    [send_task, recv_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in done:
                    if task.exception():
                        logger.error("Task failed: %s", task.exception())

                for task in pending:
                    task.cancel()

                await asyncio.gather(*pending, return_exceptions=True)

        except websockets.exceptions.WebSocketException as e:
            logger.error("WebSocket error: %s", e)
        except Exception:
            logger.exception("Unexpected error in client")
        finally:
            await self._cleanup()

    async def _send_hello(self) -> None:
        if not self._websocket:
            return

        hello = {
            "type": "hello",
            "sample_rate": self.config.capture.sample_rate,
            "channels": self.config.capture.channels,
        }

        await self._websocket.send(json.dumps(hello))
        logger.info("Sent hello message")

    async def _send_loop(self) -> None:
        logger.info("Send loop started (wake-word gated mode)")

        # Wait for pre-roll buffer to fill (gives hardware time to stabilize)
        await self.capture.wait_for_preroll()
        logger.info("Pre-roll complete, audio capture ready")
        logger.info("Waiting for wake word: %s", self.config.wake_word.model_name)

        try:
            chunk_count = 0
            while self._running and self._websocket:
                chunk = self.capture.read(timeout=1.0)
                chunk_count += 1

                if self._state == ClientState.WAITING_FOR_WAKE:
                    # Log periodically that we're receiving audio while waiting
                    if chunk_count == 1:
                        logger.info(
                            "First audio chunk received while waiting for wake word: "
                            "shape=%s dtype=%s samples=%d",
                            chunk.shape,
                            chunk.dtype,
                            len(chunk),
                        )
                    elif (
                        chunk_count % 100 == 0
                    ):  # Log every ~2 seconds (100 chunks * 20ms)
                        logger.debug(
                            "Still waiting for wake word: received %d chunks, "
                            "chunk_shape=%s chunk_samples=%d",
                            chunk_count,
                            chunk.shape,
                            len(chunk),
                        )

                    # Listen for wake word
                    if self.wake_word.process_frame(chunk):
                        logger.info("Wake word detected! Starting audio stream...")
                        self._state = ClientState.WAKE_DETECTED
                        # Play feedback tone
                        self._play_feedback("listening")
                        # Brief activation delay before starting stream
                        await asyncio.sleep(
                            self.config.wake_word.activation_delay_ms / 1000.0
                        )
                        self._state = ClientState.STREAMING
                        # Send hello message when starting to stream
                        await self._send_hello()

                elif self._state == ClientState.STREAMING:
                    # Stream audio to server
                    # Update VAD with current TTS state for echo suppression
                    self.vad.set_tts_active(self._tts_active)

                    # Process frame for VAD (updates internal state, used for echo suppression)
                    _ = self.vad.process_frame(chunk)
                    await self._websocket.send(chunk.tobytes())

                await asyncio.sleep(0.001)

        except asyncio.CancelledError:
            logger.info("Send loop cancelled")
            raise
        except Exception:
            logger.exception("Send loop error")
            raise

    async def _send_interrupt(self) -> None:
        """Send interrupt signal to server (barge-in)."""
        if self._websocket and not self._interrupt_sent:
            self._interrupt_sent = True
            await self._websocket.send(json.dumps({"type": "interrupt"}))
            self.playback.stop_playback()
            logger.info("Sent interrupt (barge-in)")

    def _stop_playback_from_server(self) -> None:
        """Called when server sends playback_stop event (keyword-based barge-in)."""
        self.playback.stop_playback()
        self._tts_active = False
        logger.info("Playback stopped by server (keyword barge-in)")

    async def _receive_loop(self) -> None:
        logger.info("Receive loop started")

        try:
            while self._running and self._websocket:
                try:
                    message = await self._websocket.recv()

                    if isinstance(message, bytes):
                        self._handle_audio(message)
                    elif isinstance(message, str):
                        self._handle_json_message(message)

                except websockets.exceptions.ConnectionClosedOK:
                    logger.info("Connection closed normally")
                    break
                except websockets.exceptions.ConnectionClosedError as e:
                    logger.warning("Connection closed with error: %s", e)
                    break
                except RuntimeError as e:
                    # Handle "cannot call recv while another coroutine is waiting"
                    if "cannot call recv" in str(e):
                        logger.warning("WebSocket recv conflict, retrying...")
                        await asyncio.sleep(0.1)
                        continue
                    raise

        except asyncio.CancelledError:
            logger.info("Receive loop cancelled")
            raise
        except Exception:
            logger.exception("Receive loop error")
            raise

    def _handle_audio(self, audio_bytes: bytes) -> None:
        logger.debug("Received %d bytes of audio", len(audio_bytes))

        try:
            self.playback.play(audio_bytes)
        except Exception:
            logger.exception("Error playing audio")

    def _handle_json_message(self, message: str) -> None:
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "transcription":
                text = data.get("text", "")
                print(f"You: {text}")
                logger.info("Transcription: %s", text)
                self._play_feedback("listening")

            elif msg_type == "llm_response":
                text = data.get("text", "")
                print(f"Assistant: {text}")
                logger.info("LLM response received")

            elif msg_type == "tts_start":
                self._tts_active = True
                self._interrupt_sent = False
                logger.debug("TTS started")

            elif msg_type == "tts_stop":
                self._tts_active = False
                self._interrupt_sent = False
                logger.debug("TTS stopped")
                # Return to wake word listening after response complete
                self._state = ClientState.WAITING_FOR_WAKE
                self.wake_word.reset()
                logger.info("Returning to wake word listening")

            elif msg_type == "playback_stop":
                # Server detected a barge-in keyword in user speech
                self._stop_playback_from_server()

            else:
                logger.debug("Unknown message type: %s", msg_type)

        except json.JSONDecodeError:
            logger.warning("Received invalid JSON: %s", message[:100])
        except Exception:
            logger.exception("Error handling JSON message")

    def _play_feedback(self, tone_name: str) -> None:
        """Play a feedback tone if enabled."""
        if not self._feedback_enabled:
            return
        try:
            tone = self.feedback.get_tone(tone_name)
            if tone:
                self.playback.play(tone)
        except Exception:
            logger.debug("Failed to play feedback tone: %s", tone_name)

    async def _cleanup(self) -> None:
        logger.info("Cleaning up client resources")

        self._running = False

        try:
            self.capture.stop()
        except Exception:
            logger.exception("Error stopping audio capture")

        try:
            self.playback.close()
        except Exception:
            logger.exception("Error closing audio playback")

        try:
            self.wake_word.close()
        except Exception:
            logger.exception("Error closing wake word detector")

        if self._websocket:
            try:
                await self._websocket.close()
            except Exception:
                logger.exception("Error closing WebSocket")

        logger.info("Client cleanup complete")

    def stop(self) -> None:
        logger.info("Stop requested")
        self._running = False
