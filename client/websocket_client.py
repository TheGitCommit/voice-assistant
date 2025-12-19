"""
WebSocket client for voice assistant communication.
"""

import asyncio
import json
import logging
from typing import Optional

import websockets

from client.config import ClientConfig
from client.audio_capture import AudioCapture
from client.audio_playback import AudioPlayback
from client.vad import VoiceActivityDetector

logger = logging.getLogger(__name__)


class VoiceAssistantClient:
    """
    WebSocket client for voice assistant.

    Manages bidirectional audio streaming:
    - Sends microphone audio to server
    - Receives and plays TTS audio responses
    - Handles JSON control messages
    """

    def __init__(self, config: ClientConfig):
        self.config = config

        # Audio components
        self.capture = AudioCapture(config.capture)
        self.playback = AudioPlayback(config.playback)
        self.vad = VoiceActivityDetector(config.vad, config.capture.sample_rate)

        # Connection state
        self._websocket = None
        self._running = False

        logger.info("VoiceAssistantClient initialized")

    async def run(self) -> None:
        """
        Main client loop.

        Connects to server and runs send/receive tasks until error or shutdown.
        """
        logger.info("Connecting to server: %s", self.config.server.url)

        try:
            async with websockets.connect(
                self.config.server.url,
                max_size=None,  # No message size limit
                ping_interval=20,  # Keep connection alive
                ping_timeout=10,
            ) as websocket:
                self._websocket = websocket
                self._running = True

                logger.info("Connected to server")

                # Start audio I/O
                self.capture.start()
                self.playback.start()

                # Send initial hello message
                await self._send_hello()

                # Run send/receive tasks concurrently
                send_task = asyncio.create_task(self._send_loop())
                recv_task = asyncio.create_task(self._receive_loop())

                # Wait for either task to complete (error or shutdown)
                done, pending = await asyncio.wait(
                    [send_task, recv_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Log task completion
                for task in done:
                    if task.exception():
                        logger.error("Task failed: %s", task.exception())

                # Cancel remaining tasks
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
        """Send initial hello message to server."""
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
        """
        Continuously send microphone audio to server.

        Reads from audio capture queue and sends as binary WebSocket messages.
        """
        logger.info("Send loop started (microphone active)")

        try:
            while self._running and self._websocket:
                # Read audio chunk from microphone (blocks until available)
                chunk = self.capture.read(timeout=1.0)

                # Update VAD state (for logging/feedback)
                self.vad.process_frame(chunk)

                # Send to server as binary message
                await self._websocket.send(chunk.tobytes())

                # Small sleep to reduce CPU usage
                await asyncio.sleep(0.001)

        except asyncio.CancelledError:
            logger.info("Send loop cancelled")
            raise
        except Exception:
            logger.exception("Send loop error")
            raise

    async def _receive_loop(self) -> None:
        """
        Continuously receive messages from server.

        Handles both binary audio data and JSON control messages.
        """
        logger.info("Receive loop started")

        try:
            while self._running and self._websocket:
                message = await self._websocket.recv()

                if isinstance(message, bytes):
                    # Binary audio data from TTS
                    self._handle_audio(message)
                elif isinstance(message, str):
                    # JSON control message
                    self._handle_json_message(message)

        except asyncio.CancelledError:
            logger.info("Receive loop cancelled")
            raise
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed by server")
        except Exception:
            logger.exception("Receive loop error")
            raise

    def _handle_audio(self, audio_bytes: bytes) -> None:
        """
        Handle received audio data.

        Args:
            audio_bytes: PCM16LE audio from server TTS
        """
        logger.debug("Received %d bytes of audio", len(audio_bytes))

        try:
            self.playback.play(audio_bytes)
        except Exception:
            logger.exception("Error playing audio")

    def _handle_json_message(self, message: str) -> None:
        """
        Handle received JSON control message.

        Args:
            message: JSON string from server
        """
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "transcription":
                text = data.get("text", "")
                print(f"📝 You: {text}")
                logger.info("Transcription: %s", text)

            elif msg_type == "llm_response":
                text = data.get("text", "")
                print(f"🤖 Assistant: {text}")
                logger.info("LLM response received")

            else:
                logger.debug("Unknown message type: %s", msg_type)

        except json.JSONDecodeError:
            logger.warning("Received invalid JSON: %s", message[:100])
        except Exception:
            logger.exception("Error handling JSON message")

    async def _cleanup(self) -> None:
        """Clean up resources."""
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

        if self._websocket:
            try:
                await self._websocket.close()
            except Exception:
                logger.exception("Error closing WebSocket")

        logger.info("Client cleanup complete")

    def stop(self) -> None:
        """Request client shutdown (call from outside async context)."""
        logger.info("Stop requested")
        self._running = False
