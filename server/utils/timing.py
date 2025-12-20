import time
from contextlib import contextmanager
from typing import Dict, Optional


class TimingStats:
    """
    Tracks and computes statistics for operation timings.

    This class is used to record the duration of various operations and calculate
    statistics such as the minimum, maximum, mean, and total time for each operation.
    It provides mechanisms to retrieve statistics for specific operations or all
    operations and to reset the collected data.

    :ivar timings: A dictionary that stores lists of recorded durations for each operation.
    :type timings: Dict[str, list[float]]
    :ivar counts: A dictionary that stores the count of recorded durations for each operation.
    :type counts: Dict[str, int]
    """

    def __init__(self):
        self._timings: Dict[str, list[float]] = {}
        self._counts: Dict[str, int] = {}

    def record(self, operation: str, duration: float) -> None:
        """Record a timing measurement."""
        if operation not in self._timings:
            self._timings[operation] = []
            self._counts[operation] = 0
        self._timings[operation].append(duration)
        self._counts[operation] += 1

    def get_stats(self, operation: str) -> Optional[Dict[str, float]]:
        """Get statistics for a specific operation."""
        if operation not in self._timings or not self._timings[operation]:
            return None

        timings = self._timings[operation]
        return {
            "count": self._counts[operation],
            "min": min(timings),
            "max": max(timings),
            "mean": sum(timings) / len(timings),
            "total": sum(timings),
        }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all recorded operations."""
        return {
            op: self.get_stats(op) for op in self._timings.keys() if self.get_stats(op)
        }

    def reset(self) -> None:
        """Reset all statistics."""
        self._timings.clear()
        self._counts.clear()


@contextmanager
def measure_time(operation: str, stats: Optional[TimingStats] = None):
    """
    Context manager to measure the execution time of a code block and optionally
    record it into a timing statistics object.

    This context manager calculates the time taken for the execution of the enclosed
    code block. If a `TimingStats` object is provided, it records the measured
    duration against a specified operation name.

    :param operation: Name of the operation being timed.
    :param stats: Optional TimingStats object for recording the operation's duration.
    :type stats: TimingStats or None
    :return: None
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        if stats:
            stats.record(operation, duration)


def format_duration(seconds: float) -> str:
    """
    Formats a duration given in seconds into a human-readable string with an appropriate unit. The
    function selects seconds (s), milliseconds (ms), or microseconds (µs) based on the given value
    and includes formatted precision.

    :param seconds: The duration in seconds to format.
    :type seconds: float
    :return: A string representing the formatted duration.
    :rtype: str
    """
    if seconds >= 1.0:
        return f"{seconds:.3f}s"
    elif seconds >= 0.001:
        return f"{seconds * 1000:.1f}ms"
    else:
        return f"{seconds * 1000000:.1f}µs"
