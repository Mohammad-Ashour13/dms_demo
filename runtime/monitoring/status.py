from __future__ import annotations

import copy
import threading
import time
from collections import defaultdict, deque
from typing import Any


class StageTimingRegistry:
    def __init__(self, max_samples=300):
        self.samples: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=max(1, int(max_samples)))
        )
        self.lock = threading.Lock()

    def record(self, stage: str, milliseconds: float) -> None:
        with self.lock:
            self.samples[str(stage)].append(float(milliseconds))

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        position = (len(ordered) - 1) * percentile
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    def snapshot(self) -> dict[str, dict[str, float | int | None]]:
        with self.lock:
            copied = {key: list(values) for key, values in self.samples.items()}
        return {
            key: {
                "samples": len(values),
                "p50_ms": self._percentile(values, 0.50),
                "p95_ms": self._percentile(values, 0.95),
            }
            for key, values in copied.items()
        }


class RuntimeStatusStore:
    """Thread-safe dashboard-status-v1 snapshot outside safety contracts."""

    SCHEMA_VERSION = "dashboard-status-v1"

    def __init__(self):
        self.lock = threading.Lock()
        self.value: dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "updated_monotonic_sec": time.monotonic(),
            "runtime": {"status": "STARTING"},
        }

    def update(self, **sections: Any) -> dict[str, Any]:
        with self.lock:
            self.value.update(copy.deepcopy(sections))
            self.value["updated_monotonic_sec"] = time.monotonic()
            return copy.deepcopy(self.value)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return copy.deepcopy(self.value)
