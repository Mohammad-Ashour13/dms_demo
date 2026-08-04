from __future__ import annotations

import queue

from dms_final_system.shared.contracts import EvidenceEvent


class EvidenceBus:
    """Thread-safe extension point for phone/smoking/eating/distraction detectors."""

    def __init__(self, maxsize: int = 256):
        self.queue: queue.Queue[EvidenceEvent] = queue.Queue(maxsize=maxsize)

    def publish(self, event: EvidenceEvent) -> None:
        try:
            self.queue.put_nowait(event)
        except queue.Full:
            # Stale external evidence is safer to drop than to delay current decisions.
            pass

    def drain(self) -> list[EvidenceEvent]:
        items = []
        while True:
            try:
                items.append(self.queue.get_nowait())
            except queue.Empty:
                return items
