"""
WebSocket endpoint for audio streaming.
"""

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from server.networking.websocket_connection import WebSocketConnection
from server.core.audio_processor import AudioProcessor

router = APIRouter()
logger = logging.getLogger(__name__)


@router.websocket("/ws/audio")
async def audio_stream_endpoint(websocket: WebSocket):
    """
    Handles a WebSocket connection for streaming audio, allowing clients to interact
    via text messages and receive processed audio data in real-time.

    This function coordinates the interaction flow, including accepting the WebSocket
    connection, managing concurrent tasks for sending, receiving, and processing
    data, and handling cleanup upon completion. It ensures fault tolerance by
    handling task exceptions and WebSocket disconnections.

    :param websocket: The WebSocket connection object used to communicate with the client.
    :type websocket: WebSocket
    :return: None
    """
    await websocket.accept()

    # Create connection and processor
    connection = WebSocketConnection(websocket)
    processor = AudioProcessor(connection)

    logger.info("[%s] Client connected", connection.connection_id)

    # Create tasks for concurrent processing
    tasks = {
        asyncio.create_task(
            connection.send_loop(),
            name=f"send-{connection.connection_id}",
        ),
        asyncio.create_task(
            connection.receive_loop(
                on_text_message=processor.handle_text_message,
            ),
            name=f"recv-{connection.connection_id}",
        ),
        asyncio.create_task(
            processor.run(),
            name=f"proc-{connection.connection_id}",
        ),
    }

    try:
        # Wait for any task to complete (usually means error or disconnect)
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        # Log any task failures
        for task in done:
            exc = task.exception()
            if exc and not isinstance(exc, asyncio.CancelledError):
                logger.error(
                    "[%s] Task %s failed: %s",
                    connection.connection_id,
                    task.get_name(),
                    exc,
                )

    except WebSocketDisconnect:
        logger.info("[%s] Client disconnected", connection.connection_id)
    except Exception:
        logger.exception(
            "[%s] Unexpected error in connection handler", connection.connection_id
        )
    finally:
        # Clean up all tasks
        logger.info("[%s] Cleaning up connection", connection.connection_id)

        for task in tasks:
            if not task.done():
                task.cancel()

        # Wait for cancellation to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        # Log final statistics including performance metrics
        stats = connection.get_stats()
        perf_stats = processor.timing_stats.get_all_stats()
        
        logger.info(
            "[%s] Connection closed: stats=%s",
            connection.connection_id,
            stats,
        )
        
        if perf_stats:
            logger.info(
                "[%s] Performance summary: %s",
                connection.connection_id,
                perf_stats,
            )
        
        # Cleanup LLM client
        await processor.llm.close()
