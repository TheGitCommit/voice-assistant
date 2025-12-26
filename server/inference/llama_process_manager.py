import asyncio
import logging
import os
import subprocess
import time
from typing import Optional

import httpx

from server.config import LlamaConfig

logger = logging.getLogger(__name__)


class LlamaProcessManager:
    """Manages Llama.cpp server subprocess lifecycle with auto-recovery."""

    def __init__(self, config: LlamaConfig):
        self.config = config
        self._process: Optional[subprocess.Popen] = None
        self._restart_count = 0
        self._max_restarts = 5
        self._last_restart_time = 0.0
        self._validate_installation()

    def _validate_installation(self) -> None:
        if not os.path.exists(self.config.exe_path):
            raise FileNotFoundError(
                f"Llama executable not found: {self.config.exe_path}"
            )

        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Llama model not found: {self.config.model_path}")

        # Check for split model files
        model_path = self.config.model_path
        if "-00001-of-" in model_path or "-0001-of-" in model_path:
            # This is a split model, check if all parts exist
            import re

            base_match = re.search(r"(.+)-(\d{4,5})-of-(\d{4,5})\.gguf$", model_path)
            if base_match:
                base_name, part_num, total_parts_str = base_match.groups()
                total_parts_int = int(total_parts_str)
                part_num_int = int(part_num)

                # Determine the format width from the original filename
                format_width = len(part_num)

                missing_parts = []
                for i in range(1, total_parts_int + 1):
                    if format_width == 5:
                        part_file = f"{base_name}-{i:05d}-of-{total_parts_int:05d}.gguf"
                    else:
                        part_file = f"{base_name}-{i:04d}-of-{total_parts_int:04d}.gguf"

                    if not os.path.exists(part_file):
                        missing_parts.append(part_file)

                if missing_parts:
                    logger.warning(
                        "Split model detected but some parts are missing:\n"
                        "  Model: %s\n"
                        "  Missing parts: %s\n"
                        "  Ensure all %d parts are in the same directory.",
                        model_path,
                        ", ".join(missing_parts),
                        total_parts_int,
                    )

    def _build_command(self) -> list[str]:
        cmd = [
            self.config.exe_path,
            "-m",
            self.config.model_path,
            "-ngl",
            str(self.config.gpu_layers),
            "-c",
            str(self.config.context_size),
            "--threads",
            str(self.config.threads),
            "--threads-batch",
            str(self.config.threads_batch),
            "--batch-size",
            str(self.config.batch_size),
            "--ubatch-size",
            str(self.config.ubatch_size),
            "--parallel",
            str(self.config.parallel),
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
        ]

        if self.config.mlock:
            cmd.append("--mlock")

        if self.config.no_mmap:
            cmd.append("--no-mmap")

        return cmd

    def start(self) -> None:
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

        # Verify model file exists
        if not os.path.exists(self.config.model_path):
            error_msg = f"Model file not found: {self.config.model_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        cmd = self._build_command()
        logger.debug("Llama command: %s", " ".join(cmd))

        try:
            # Capture stderr to see actual error messages
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0,
            )

            time.sleep(2)

            if self._process.poll() is not None:
                # Process exited, read error output
                stdout, stderr = self._process.communicate(timeout=1)
                error_output = stderr.decode("utf-8", errors="ignore") if stderr else ""
                stdout_output = (
                    stdout.decode("utf-8", errors="ignore") if stdout else ""
                )

                logger.error("Llama server failed to start (immediate exit)")
                logger.error("Exit code: %d", self._process.returncode)
                if error_output:
                    logger.error("Stderr: %s", error_output.strip())
                if stdout_output:
                    logger.error("Stdout: %s", stdout_output.strip())

                # Provide helpful error message
                if (
                    "not found" in error_output.lower()
                    or "cannot open" in error_output.lower()
                ):
                    error_msg = f"Model file not found or cannot be opened: {self.config.model_path}"
                elif (
                    "split" in error_output.lower()
                    or "00001-of-00002" in self.config.model_path
                ):
                    error_msg = (
                        f"Split model detected. Ensure all split files are in the same directory.\n"
                        f"Model path: {self.config.model_path}\n"
                        f"Expected files: qwen2.5-7b-instruct-q6_k-00001-of-00002.gguf and qwen2.5-7b-instruct-q6_k-00002-of-00002.gguf"
                    )
                elif error_output:
                    error_msg = f"Llama server error: {error_output.strip()}"
                else:
                    error_msg = "Llama server process exited immediately (check model file and system resources)"

                self._process = None
                raise RuntimeError(error_msg)

            logger.info("Llama server process started (PID: %d)", self._process.pid)

        except subprocess.TimeoutExpired:
            # Process is still running, which is good
            logger.info("Llama server process started (PID: %d)", self._process.pid)
        except Exception:
            logger.exception("Failed to start Llama server")
            if self._process:
                try:
                    stdout, stderr = self._process.communicate(timeout=1)
                    if stderr:
                        logger.error(
                            "Stderr: %s", stderr.decode("utf-8", errors="ignore")
                        )
                except:
                    pass
                self._process = None
            raise

    def stop(self) -> None:
        if self._process is None:
            return

        logger.info("Stopping Llama server (PID: %d)", self._process.pid)

        try:
            self._process.terminate()

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
        return self._process is not None and self._process.poll() is None

    async def health_check(self) -> bool:
        """Check if Llama server is responding to HTTP requests."""
        if not self.is_running():
            return False

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"http://localhost:{self.config.port}/health"
                )
                return response.status_code == 200
        except Exception:
            return False

    def restart(self) -> bool:
        """Restart the Llama server. Returns True if successful."""
        now = time.time()

        # Reset restart count if it's been a while since last restart
        if now - self._last_restart_time > 300:  # 5 minutes
            self._restart_count = 0

        if self._restart_count >= self._max_restarts:
            logger.error(
                "Max restart attempts (%d) reached, not restarting",
                self._max_restarts,
            )
            return False

        logger.warning(
            "Restarting Llama server (attempt %d/%d)",
            self._restart_count + 1,
            self._max_restarts,
        )

        self.stop()
        time.sleep(2)

        try:
            self.start()
            self._restart_count += 1
            self._last_restart_time = now
            return True
        except Exception:
            logger.exception("Failed to restart Llama server")
            return False

    async def ensure_running(self) -> bool:
        """Ensure Llama is running, restart if needed. Returns True if healthy."""
        if await self.health_check():
            return True

        if not self.is_running():
            logger.warning("Llama server not running, attempting restart")
            if self.restart():
                await asyncio.sleep(self.config.startup_delay_seconds)
                return await self.health_check()

        return False

    async def monitor_loop(self, check_interval: float = 30.0) -> None:
        """Background task that monitors Llama health and auto-restarts."""
        logger.info("Llama health monitor started (interval: %.0fs)", check_interval)

        while True:
            await asyncio.sleep(check_interval)

            if not await self.health_check():
                logger.warning("Llama health check failed")
                await self.ensure_running()
            else:
                logger.debug("Llama health check passed")
