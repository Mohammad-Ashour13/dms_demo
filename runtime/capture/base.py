from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from dms_final_system.shared.contracts import FramePacket


@runtime_checkable
class CameraCapture(Protocol):
    """Stable live/replay capture contract used by the composition root."""

    dropped_frames: int

    def start(self) -> "CameraCapture": ...

    def read(self, timeout: float = 1.0) -> FramePacket: ...

    def close(self) -> None: ...

    @property
    def actual_configuration(self) -> dict[str, Any]: ...
