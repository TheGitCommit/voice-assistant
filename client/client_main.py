import asyncio
import logging
import signal
import sys
from pathlib import Path
from dataclasses import replace

import httpx

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from client.config import DEFAULT_CONFIG
from client.cloud_fallback.cloud_config import is_cloud_configured
from client.websocket_client import VoiceAssistantClient
from client.cloud_fallback.cloud_processor import CloudAudioProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


async def check_server_health(url: str, timeout: float = 5.0) -> bool:
    try:
        http_url = url.replace("ws://", "http://").replace("wss://", "https://")
        base_url = http_url.rsplit("/ws/", 1)[0]
        health_url = f"{base_url}/health"

        logger.debug("Checking server health: %s", health_url)

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(health_url)

            if response.status_code == 200:
                data = response.json()
                logger.debug("Server health: %s", data)
                return data.get("status") == "healthy"

            return False

    except httpx.ConnectError:
        return False
    except httpx.TimeoutException:
        return False
    except Exception:
        return False


async def main() -> None:
    logger.info("===Voice Assistant Client Starting===")

    stop_event = asyncio.Event()
    current_client = None

    def signal_handler(sig, frame):
        logger.info("Received interrupt signal - Stopping...")
        stop_event.set()
        if current_client:
            current_client.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    while not stop_event.is_set():

        logger.info("Evaluating connection mode...")

        server_available = await check_server_health(
            DEFAULT_CONFIG.server.url, timeout=2.0
        )

        run_config = DEFAULT_CONFIG

        if server_available:
            logger.info("Local server FOUND. Mode: LOCAL")
            mode = "local"

        elif is_cloud_configured():
            logger.warning("Local server UNREACHABLE. Mode: CLOUD")
            mode = "cloud"

            new_playback = replace(run_config.playback, sample_rate=24000)
            run_config = replace(run_config, playback=new_playback)
            logger.info("Updated playback rate to 24000Hz for Cloud TTS")

        else:
            logger.error("No Server + No Cloud Keys. Retrying in 5s...")
            await asyncio.sleep(5)
            continue

        try:
            if mode == "local":
                client = VoiceAssistantClient(run_config)
            else:
                client = CloudAudioProcessor(run_config)

            current_client = client

            logger.info(
                "Starting %s Client (Capture: %dHz | Playback: %dHz)",
                mode.upper(),
                run_config.capture.sample_rate,
                run_config.playback.sample_rate,
            )

            await client.run()

        except Exception as e:
            logger.error(f"Client crashed or disconnected: {e}")
        finally:
            current_client = None

        if not stop_event.is_set():
            logger.warning("Connection lost. Re-evaluating in 2 seconds...")
            await asyncio.sleep(2)

    logger.info("=== Voice Assistant Client Stopped ===")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
