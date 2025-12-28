import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.config import CONFIG
from server.utils.logging_utils import setup_logging
from server.utils.latency_monitor import get_latency_monitor, get_metrics
from server.inference.llama_process_manager import LlamaProcessManager
from server.networking.websocket_server import router as ws_router

setup_logging(CONFIG["logging"].level)
logger = logging.getLogger(__name__)

llama_manager = LlamaProcessManager(CONFIG["llama"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage server lifecycle and initialize components."""
    logger.info("=== Starting Sovereign Voice Assistant Server ===")

    monitor_task = None

    try:
        # 1. Initialize Latency Monitor
        logger.info("Initializing latency monitor...")
        get_latency_monitor()

        # 2. Start Llama server
        llama_manager.start()
        logger.info(
            "Waiting %.1fs for model to load...",
            CONFIG["llama"].startup_delay_seconds,
        )
        await asyncio.sleep(CONFIG["llama"].startup_delay_seconds)

        if not llama_manager.is_running():
            raise RuntimeError("Llama server not running after startup delay")

        logger.info("Llama server ready")

        # 3. Initialize TTS provider if Kokoro
        tts_provider = CONFIG["tts"].provider.lower()
        if tts_provider == "kokoro":
            from server.inference.kokoro_tts import KokoroTTS

            logger.info("Initializing Kokoro TTS...")
            await KokoroTTS.initialize(CONFIG["kokoro"])
            logger.info("Kokoro TTS ready")

        # Start health monitor in background
        monitor_task = asyncio.create_task(
            llama_manager.monitor_loop(check_interval=30.0)
        )

    except Exception:
        logger.exception("Failed to start server")
        raise

    logger.info("=" * 50)
    logger.info("Sovereign Voice Assistant Server ready")
    logger.info("=" * 50)

    yield

    logger.info("=== Shutting down Voice Assistant Server ===")

    if monitor_task:
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

    llama_manager.stop()
    logger.info("Shutdown complete")


app = FastAPI(title="Voice Assistant Server", lifespan=lifespan)
app.include_router(ws_router)


@app.get("/health")
async def health_check():
    llama_healthy = await llama_manager.health_check()
    return {
        "status": "healthy" if llama_healthy else "degraded",
        "llama_running": llama_manager.is_running(),
        "llama_healthy": llama_healthy,
    }


@app.get("/metrics")
async def metrics():
    """Get latency and performance metrics."""
    return JSONResponse(content=get_metrics())


def main():
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
        log_config=None,
        ws_ping_interval=60.0,  # Ping every 60s
        ws_ping_timeout=30.0,  # Wait 30s for pong
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception:
        logger.exception("Fatal error")
        sys.exit(1)
