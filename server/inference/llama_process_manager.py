"""
Manages the Llama.cpp server process lifecycle.
"""

import logging
import os
import subprocess
import time
from typing import Optional

from server.config import LlamaConfig

logger = logging.getLogger(__name__)


class LlamaProcessManager:
    """
    Manages the Llama.cpp server as a subprocess.

    Handles startup, monitoring, and graceful shutdown of the inference server.
    """

    def __init__(self, config: LlamaConfig):
        self.config = config
        self._process: Optional[subprocess.Popen] = None
        self._validate_installation()

    def _validate_installation(self) -> None:
        """Validate that required files exist."""
        if not os.path.exists(self.config.exe_path):
            raise FileNotFoundError(
                f"Llama executable not found: {self.config.exe_path}"
            )

        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Llama model not found: {self.config.model_path}")

    def _build_command(self) -> list[str]:
        """Build command line arguments for llama-server."""
        return [
            self.config.exe_path,
            "-m",
            self.config.model_path,
            "-ngl",
            str(self.config.gpu_layers),
            "-c",
            str(self.config.context_size),
            "--port",
            str(self.config.port),
            "--host",
            self.config.host,
        ]

    def start(self) -> None:
        """Start the Llama server process."""
        if self._process is not None:
            logger.warning("Llama server already running")
            return

        model_name = os.path.basename(self.config.model_path)
        logger.info(
            "Starting Llama server: model=%s port=%d gpu_layers=%d",
            model_name,
            self.config.port,
            self.config.gpu_layers,
        )

        cmd = self._build_command()

        try:
            self._process = subprocess.Popen(
                cmd,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )

            # Brief wait to check for immediate failure
            time.sleep(2)

            if self._process.poll() is not None:
                logger.error("Llama server failed to start (immediate exit)")
                self._process = None
                raise RuntimeError("Llama server process exited immediately")

            logger.info("Llama server process started (PID: %d)", self._process.pid)

        except Exception:
            logger.exception("Failed to start Llama server")
            self._process = None
            raise

    def stop(self) -> None:
        """Stop the Llama server process gracefully."""
        if self._process is None:
            return

        logger.info("Stopping Llama server (PID: %d)", self._process.pid)

        try:
            self._process.terminate()

            # Wait for graceful shutdown
            try:
                self._process.wait(timeout=5)
                logger.info("Llama server stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("Llama server didn't stop gracefully, killing")
                self._process.kill()
                self._process.wait()

        except Exception:
            logger.exception("Error stopping Llama server")
        finally:
            self._process = None

    def is_running(self) -> bool:
        """Check if the Llama server process is running."""
        return self._process is not None and self._process.poll() is None
