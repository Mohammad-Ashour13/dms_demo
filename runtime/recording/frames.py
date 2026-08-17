from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EncodedFrame:
    frame_id: int
    monotonic_sec: float
    utc_timestamp: str
    jpeg: bytes

