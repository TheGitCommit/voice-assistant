import time
from contextlib import contextmanager
from typing import Dict, Optional


class TimingStats:
    """Tracks min/max/mean/total timing for named operations."""

    def __init__(self):
        self._timings: Dict[str, list[float]] = {}
        self._counts: Dict[str, int] = {}

    def record(self, operation: str, duration: float) -> None:
        if operation not in self._timings:
            self._timings[operation] = []
            self._counts[operation] = 0
        self._timings[operation].append(duration)
        self._counts[operation] += 1

    def get_stats(self, operation: str) -> Optional[Dict[str, float]]:
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
        return {
            op: self.get_stats(op) for op in self._timings.keys() if self.get_stats(op)
        }

    def reset(self) -> None:
        self._timings.clear()
        self._counts.clear()


@contextmanager
def measure_time(operation: str, stats: Optional[TimingStats] = None):
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        if stats:
            stats.record(operation, duration)


def format_duration(seconds: float) -> str:
    if seconds >= 1.0:
        return f"{seconds:.3f}s"
    elif seconds >= 0.001:
        return f"{seconds * 1000:.1f}ms"
    else:
        return f"{seconds * 1000000:.1f}µs"
