import logging
import time
from typing import Any


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class RateLimitedLogger:
    """Logs at most once per interval per key to prevent spam."""

    def __init__(self, logger: logging.Logger, interval_seconds: float = 5.0):
        self._logger = logger
        self._interval = float(interval_seconds)
        self._last_log_time: dict[str, float] = {}

    def _should_log(self, key: str) -> bool:
        now = time.monotonic()
        last = self._last_log_time.get(key, 0.0)
        if now - last >= self._interval:
            self._last_log_time[key] = now
            return True
        return False

    def debug(self, key: str, msg: str, *args: Any, **kwargs: Any) -> None:
        if self._should_log(key):
            self._logger.debug(msg, *args, **kwargs)

    def info(self, key: str, msg: str, *args: Any, **kwargs: Any) -> None:
        if self._should_log(key):
            self._logger.info(msg, *args, **kwargs)

    def warning(self, key: str, msg: str, *args: Any, **kwargs: Any) -> None:
        if self._should_log(key):
            self._logger.warning(msg, *args, **kwargs)

    def error(self, key: str, msg: str, *args: Any, **kwargs: Any) -> None:
        if self._should_log(key):
            self._logger.error(msg, *args, **kwargs)
