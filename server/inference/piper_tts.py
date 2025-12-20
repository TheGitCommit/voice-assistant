import asyncio
import logging
import os
import subprocess
import time
from typing import Optional

from server.config import PiperConfig

logger = logging.getLogger(__name__)


class PiperTTS:
    def __init__(self, config: PiperConfig):
        self.config = config
        self._validate_installation()
        logger.info("Piper TTS initialized: model=%s", config.model_path)

    def _validate_installation(self) -> None:
        if not os.path.exists(self.config.exe_path):
            raise FileNotFoundError(
                f"Piper executable not found: {self.config.exe_path}"
            )

        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Piper model not found: {self.config.model_path}")

        if not os.path.exists(self.config.model_config_path):
            raise FileNotFoundError(
                f"Piper model config not found: {self.config.model_config_path}"
            )

    async def synthesize(self, text: str, max_retries: int = 2) -> Optional[bytes]:
        if not text or not text.strip():
            return None

        args = [
            "--model",
            self.config.model_path,
            "--output_raw",
        ]

        start_time = time.perf_counter()
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                process = await asyncio.create_subprocess_exec(
                    self.config.exe_path,
                    *args,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                stdout, stderr = await process.communicate(input=text.encode("utf-8"))

                duration = time.perf_counter() - start_time

                if process.returncode != 0:
                    last_error = stderr.decode()
                    if attempt < max_retries:
                        logger.warning(
                            "Piper failed (attempt %d/%d): %s. Retrying...",
                            attempt,
                            max_retries,
                            last_error[:100],
                        )
                        await asyncio.sleep(0.5)
                        continue
                    logger.error(
                        "Piper failed after %d attempts: %s", max_retries, last_error
                    )
                    return None

                if not stdout:
                    return None

                logger.info("Synthesized %d bytes in %.3fs", len(stdout), duration)
                return stdout

            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    logger.warning(
                        "Piper error (attempt %d/%d): %s. Retrying...",
                        attempt,
                        max_retries,
                        e,
                    )
                    await asyncio.sleep(0.5)
                else:
                    logger.exception(
                        "Piper synthesis error after %d attempts", max_retries
                    )

        return None
