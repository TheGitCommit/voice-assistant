"""
WebSocket connection management and message handling.
"""

import asyncio
import json
import logging
import uuid
from typing import Any, Union

from fastapi import WebSocket

from server.config import CONFIG
from server.utils.logging_utils import RateLimitedLogger

logger = logging.getLogger(__name__)


class WebSocketConnection:
    """
    Manage a WebSocket connection for asynchronous communication.

    Handles communication with a WebSocket client, including sending and
    receiving JSON events, raw audio data, and maintaining connection
    statistics. The class utilizes asyncio queues for managing communication
    tasks and provides rate-limited logging for better observability.

    :ivar websocket: The WebSocket instance for communication with the client.
    :type websocket: WebSocket
    :ivar connection_id: Unique identifier for the connection.
    :type connection_id: str
    :ivar audio_queue: Queue for storing incoming audio data from the client.
    :type audio_queue: asyncio.Queue[bytes]
    :ivar event_queue: Queue for storing outgoing events (JSON or audio data)
        to be sent to the client.
    :type event_queue: asyncio.Queue[Union[dict[str, Any], bytes]]
    """

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.connection_id = uuid.uuid4().hex[:8]

        # Queues for async communication
        ws_config = CONFIG["websocket"]
        self.audio_queue: asyncio.Queue[bytes] = asyncio.Queue(
            maxsize=ws_config.audio_queue_maxsize
        )
        self.event_queue: asyncio.Queue[Union[dict[str, Any], bytes]] = asyncio.Queue(
            maxsize=ws_config.event_queue_maxsize
        )

        # Statistics
        self._stats = {
            "audio_chunks_received": 0,
            "audio_bytes_received": 0,
            "text_messages_received": 0,
            "json_events_sent": 0,
            "audio_bytes_sent": 0,
        }

        # Rate-limited logger
        self._rl_logger = RateLimitedLogger(
            logger, CONFIG["logging"].rate_limit_seconds
        )

        logger.info("[%s] Connection created", self.connection_id)

    async def send_event(self, event: dict[str, Any]) -> None:
        """
        Queue a JSON event to be sent to the client.

        Args:
            event: Dictionary to be sent as JSON
        """
        await self.event_queue.put(event)

    async def send_audio(self, audio_bytes: bytes) -> None:
        """
        Queue audio bytes to be sent to the client.

        Args:
            audio_bytes: Raw audio data
        """
        await self.event_queue.put(audio_bytes)

    async def send_loop(self) -> None:
        """
        Continuously send queued messages to the client.

        Handles both JSON events and binary audio data.
        """
        try:
            logger.debug("[%s] Send loop started", self.connection_id)

            while True:
                message = await self.event_queue.get()

                if isinstance(message, dict):
                    await self.websocket.send_json(message)
                    self._stats["json_events_sent"] += 1

                    self._rl_logger.info(
                        "send_json",
                        "[%s] Sent JSON event: type=%s queue_depth=%d",
                        self.connection_id,
                        message.get("type"),
                        self.event_queue.qsize(),
                    )

                elif isinstance(message, (bytes, bytearray, memoryview)):
                    await self.websocket.send_bytes(bytes(message))
                    self._stats["audio_bytes_sent"] += len(message)

                    self._rl_logger.info(
                        "send_audio",
                        "[%s] Sent audio: bytes=%d queue_depth=%d",
                        self.connection_id,
                        len(message),
                        self.event_queue.qsize(),
                    )

        except asyncio.CancelledError:
            logger.debug("[%s] Send loop cancelled", self.connection_id)
            raise
        except Exception:
            logger.exception("[%s] Send loop error", self.connection_id)
            raise

    async def receive_loop(self, on_text_message=None) -> None:
        """
        Continuously receive messages from the client.

        Args:
            on_text_message: Optional callback for handling text messages
        """
        try:
            logger.debug("[%s] Receive loop started", self.connection_id)

            while True:
                msg = await self.websocket.receive()
                msg_type = msg.get("type")

                # Handle audio data (binary)
                if msg.get("bytes") is not None:
                    chunk = msg["bytes"]
                    self._stats["audio_chunks_received"] += 1
                    self._stats["audio_bytes_received"] += len(chunk)

                    self._rl_logger.info(
                        "audio_received",
                        "[%s] Audio received: chunks=%d total_bytes=%d queue=%d/%d",
                        self.connection_id,
                        self._stats["audio_chunks_received"],
                        self._stats["audio_bytes_received"],
                        self.audio_queue.qsize(),
                        CONFIG["websocket"].audio_queue_maxsize,
                    )

                    await self.audio_queue.put(chunk)
                    continue

                # Handle text/JSON messages
                if msg.get("text") is not None:
                    self._stats["text_messages_received"] += 1
                    text = msg["text"]

                    try:
                        data = json.loads(text)
                        await self._handle_json_message(data, on_text_message)
                    except json.JSONDecodeError:
                        logger.warning(
                            "[%s] Received invalid JSON (len=%d): %s",
                            self.connection_id,
                            len(text),
                            text[:100],
                        )
                    continue

                # Unexpected message type
                if msg_type != "websocket.receive":
                    self._rl_logger.warning(
                        "unexpected_msg",
                        "[%s] Unexpected message type: %s",
                        self.connection_id,
                        msg_type,
                    )

        except asyncio.CancelledError:
            logger.debug("[%s] Receive loop cancelled", self.connection_id)
            raise
        except Exception:
            logger.exception("[%s] Receive loop error", self.connection_id)
            raise

    async def _handle_json_message(
        self,
        data: dict[str, Any],
        on_text_message=None,
    ) -> None:
        """Handle parsed JSON messages from client."""
        msg_type = data.get("type")

        if msg_type == "hello":
            logger.info(
                "[%s] Client hello: sample_rate=%s channels=%s",
                self.connection_id,
                data.get("sample_rate"),
                data.get("channels"),
            )

        elif msg_type == "test_question" and on_text_message:
            # Debug feature: direct text input bypassing STT
            question_text = data.get("text", "")
            if question_text:
                # Echo transcription back to client
                await self.send_event(
                    {
                        "type": "transcription",
                        "text": question_text,
                    }
                )
                # Trigger text processing callback
                asyncio.create_task(on_text_message(question_text))
        else:
            self._rl_logger.info(
                "json_message",
                "[%s] Received JSON message: type=%s",
                self.connection_id,
                msg_type,
            )

    def get_stats(self) -> dict[str, Any]:
        """Get connection statistics."""
        return self._stats.copy()
