from __future__ import annotations

from collections import deque

from dms_final_system.shared.contracts import FaceSignal


class FaceSignalBuffer:
    def __init__(self, retention_sec: float = 65.0):
        self.retention_sec = retention_sec
        self.items: deque[FaceSignal] = deque()

    def append(self, signal: FaceSignal) -> None:
        self.items.append(signal)
        cutoff = signal.monotonic_sec - self.retention_sec
        while self.items and self.items[0].monotonic_sec < cutoff:
            self.items.popleft()

    def window(self, start_sec: float, end_sec: float, quality_min: float = 0.0) -> list[FaceSignal]:
        return [
            item for item in self.items
            if start_sec <= item.monotonic_sec < end_sec
            and item.face_detected and item.face_quality >= quality_min
        ]

    def clear(self) -> None:
        self.items.clear()
