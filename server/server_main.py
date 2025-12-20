"""
Voice Assistant Server - Main Entry Point

Coordinates startup/shutdown of:
- Llama.cpp inference server
- FastAPI WebSocket server
- Audio processing pipeline
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.config import CONFIG
from server.utils.logging_utils import setup_logging
from server.inference.llama_process_manager import LlamaProcessManager
from server.networking.websocket_server import router as ws_router

# Initialize logging
setup_logging(CONFIG["logging"].level)
logger = logging.getLogger(__name__)

# Initialize Llama manager
llama_manager = LlamaProcessManager(CONFIG["llama"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous context manager handling the lifespan of a FastAPI application.

    This function manages the startup and shutdown lifecycle of the Voice Assistant
    Server using an external Llama inference server. It ensures the server is started
    and ready before connections are allowed and handles clean shutdown processes.

    :param app: The FastAPI application for which lifespan is being managed.
    :type app: FastAPI
    :return: Yields control back to allow application server lifecycle management.
    :rtype: AsyncGenerator[None, None]
    """
    logger.info("=== Starting Voice Assistant Server ===")

    # Start Llama inference server
    try:
        llama_manager.start()

        # Wait for Llama to load model and be ready
        logger.info(
            "Waiting %.1fs for Llama model to load...",
            CONFIG["llama"].startup_delay_seconds,
        )
        await asyncio.sleep(CONFIG["llama"].startup_delay_seconds)

        if not llama_manager.is_running():
            raise RuntimeError("Llama server not running after startup delay")

        logger.info("Llama server ready")

    except Exception:
        logger.exception("Failed to start Llama server")
        raise

    logger.info("Voice Assistant Server ready for connections")

    yield

    # Shutdown
    logger.info("=== Shutting down Voice Assistant Server ===")
    llama_manager.stop()
    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Voice Assistant Server",
    lifespan=lifespan,
)

# Register WebSocket router
app.include_router(ws_router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "llama_running": llama_manager.is_running(),
    }


@app.get("/metrics")
async def metrics():
    """
    Performance metrics endpoint.
    
    Returns aggregate performance statistics (if available).
    Note: Currently returns placeholder structure. In a production system,
    this would aggregate metrics from all active connections.
    """
    return {
        "note": "Metrics endpoint - performance stats are logged per connection",
        "llama_running": llama_manager.is_running(),
    }


def main():
    """Run the application."""
    ws_config = CONFIG["websocket"]

    logger.info(
        "Starting server: host=%s port=%d",
        ws_config.host,
        ws_config.port,
    )

    uvicorn.run(
        app,
        host=ws_config.host,
        port=ws_config.port,
        log_config=None,  # Use our logging config
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception:
        logger.exception("Fatal error")
        sys.exit(1)
