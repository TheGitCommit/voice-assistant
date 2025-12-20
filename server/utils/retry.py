import asyncio
import logging
import functools
from typing import Callable, Type, Tuple, Optional

logger = logging.getLogger(__name__)


def retry_async(
    max_attempts: int = 3,
    delay: float = 0.5,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """
    Async retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exception types to catch
        on_retry: Optional callback(exception, attempt) called on each retry
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(
                            "%s failed after %d attempts: %s",
                            func.__name__,
                            max_attempts,
                            e,
                        )
                        raise

                    if on_retry:
                        on_retry(e, attempt)

                    logger.warning(
                        "%s attempt %d/%d failed: %s. Retrying in %.1fs",
                        func.__name__,
                        attempt,
                        max_attempts,
                        e,
                        current_delay,
                    )

                    await asyncio.sleep(current_delay)
                    current_delay *= backoff

            return None

        return wrapper

    return decorator


def retry_sync(
    max_attempts: int = 3,
    delay: float = 0.5,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """Sync retry decorator with exponential backoff."""
    import time

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(
                            "%s failed after %d attempts: %s",
                            func.__name__,
                            max_attempts,
                            e,
                        )
                        raise

                    logger.warning(
                        "%s attempt %d/%d failed: %s. Retrying in %.1fs",
                        func.__name__,
                        attempt,
                        max_attempts,
                        e,
                        current_delay,
                    )

                    time.sleep(current_delay)
                    current_delay *= backoff

            return None

        return wrapper

    return decorator
