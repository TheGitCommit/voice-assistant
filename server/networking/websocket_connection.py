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
    """Manages async send/receive queues for a WebSocket."""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.connection_id = uuid.uuid4().hex[:8]

        ws_config = CONFIG["websocket"]
        self.audio_queue: asyncio.Queue[bytes] = asyncio.Queue(
            maxsize=ws_config.audio_queue_maxsize
        )
        self.event_queue: asyncio.Queue[Union[dict[str, Any], bytes]] = asyncio.Queue(
            maxsize=ws_config.event_queue_maxsize
        )

        self._stats = {
            "audio_chunks_received": 0,
            "audio_bytes_received": 0,
            "text_messages_received": 0,
            "json_events_sent": 0,
            "audio_bytes_sent": 0,
        }

        self._rl_logger = RateLimitedLogger(
            logger, CONFIG["logging"].rate_limit_seconds
        )

        logger.info("[%s] Connection created", self.connection_id)

    async def send_event(self, event: dict[str, Any]) -> None:
        await self.event_queue.put(event)

    async def send_audio(self, audio_bytes: bytes) -> None:
        await self.event_queue.put(audio_bytes)

    async def send_loop(self) -> None:
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

    async def receive_loop(
        self,
        on_text_message=None,
        on_interrupt=None,
    ) -> None:
        try:
            logger.debug("[%s] Receive loop started", self.connection_id)

            while True:
                try:
                    msg = await self.websocket.receive()
                except RuntimeError as e:
                    # Handle disconnect during receive
                    if "disconnect" in str(e).lower() or "closed" in str(e).lower():
                        logger.info(
                            "[%s] WebSocket disconnected during receive",
                            self.connection_id,
                        )
                        break
                    raise

                msg_type = msg.get("type")

                # Handle disconnect message type
                if msg_type == "websocket.disconnect":
                    logger.info("[%s] Received disconnect message", self.connection_id)
                    break

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

                if msg.get("text") is not None:
                    self._stats["text_messages_received"] += 1
                    text = msg["text"]

                    try:
                        data = json.loads(text)
                        await self._handle_json_message(
                            data, on_text_message, on_interrupt
                        )
                    except json.JSONDecodeError:
                        logger.warning(
                            "[%s] Received invalid JSON (len=%d): %s",
                            self.connection_id,
                            len(text),
                            text[:100],
                        )
                    continue

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
        on_interrupt=None,
    ) -> None:
        msg_type = data.get("type")

        if msg_type == "hello":
            logger.info(
                "[%s] Client hello: sample_rate=%s channels=%s",
                self.connection_id,
                data.get("sample_rate"),
                data.get("channels"),
            )

        elif msg_type == "interrupt" and on_interrupt:
            logger.info("[%s] Interrupt received (barge-in)", self.connection_id)
            on_interrupt()

        elif msg_type == "test_question" and on_text_message:
            question_text = data.get("text", "")
            if question_text:
                await self.send_event({"type": "transcription", "text": question_text})
                asyncio.create_task(on_text_message(question_text))
        else:
            self._rl_logger.info(
                "json_message",
                "[%s] Received JSON message: type=%s",
                self.connection_id,
                msg_type,
            )

    def get_stats(self) -> dict[str, Any]:
        return self._stats.copy()
