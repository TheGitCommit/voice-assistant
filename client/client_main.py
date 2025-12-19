"""
Raspberry Pi Voice Assistant Client - Main Entry Point
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from client.config import DEFAULT_CONFIG
from client.websocket_client import VoiceAssistantClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


async def main() -> None:
    """Main application entry point."""
    logger.info("=== Voice Assistant Client Starting ===")
    logger.info("Server: %s", DEFAULT_CONFIG.server.url)
    logger.info(
        "Capture: %dHz, %d channels",
        DEFAULT_CONFIG.capture.sample_rate,
        DEFAULT_CONFIG.capture.channels,
    )
    logger.info(
        "Playback: %dHz, %d channels",
        DEFAULT_CONFIG.playback.sample_rate,
        DEFAULT_CONFIG.playback.channels,
    )

    # Create client
    client = VoiceAssistantClient(DEFAULT_CONFIG)

    # Handle Ctrl+C gracefully
    loop = asyncio.get_running_loop()

    def signal_handler(sig, frame):
        logger.info("Received interrupt signal")
        client.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run client
    try:
        await client.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt")
    except Exception:
        logger.exception("Fatal error")
        sys.exit(1)
    finally:
        logger.info("=== Voice Assistant Client Stopped ===")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass  # Already handled in main()
