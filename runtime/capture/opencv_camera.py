from __future__ import annotations

import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

from dms_final_system.shared.contracts import FramePacket


class LatestFrameCapture:
    """Bounded camera reader; stale AI frames are dropped instead of accumulated."""

    def __init__(self, source=0, width=640, height=480, fps=15.0, queue_size=2, frame_sink=None):
        self.source, self.width, self.height, self.fps = source, width, height, fps
        self.queue: queue.Queue[FramePacket] = queue.Queue(maxsize=queue_size)
        self.capture = None
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.dropped_frames = 0
        self.captured_frames = 0
        self.frame_sink = frame_sink

    def start(self) -> "LatestFrameCapture":
        self.capture = cv2.VideoCapture(self.source)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera source {self.source!r}")
        self.thread = threading.Thread(target=self._reader, name="camera-capture", daemon=True)
        self.thread.start()
        return self

    @property
    def actual_configuration(self) -> dict:
        if self.capture is None:
            return {
                "backend": "opencv",
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
            }
        return {
            "backend": "opencv",
            "width": int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or self.width),
            "height": int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.height),
            "fps": float(self.capture.get(cv2.CAP_PROP_FPS) or self.fps),
            "source": self.source,
        }

    def _reader(self) -> None:
        frame_id = 0
        interval = 1.0 / max(self.fps, 1.0)
        next_due = time.monotonic()
        while not self.stop_event.is_set():
            ok, frame = self.capture.read()
            now = time.monotonic()
            if not ok:
                self.stop_event.set()
                break
            packet = FramePacket(
                frame_id=frame_id,
                utc_timestamp=datetime.now(timezone.utc).isoformat(),
                monotonic_sec=now,
                frame=frame,
            )
            frame_id += 1
            self.captured_frames = frame_id
            if self.frame_sink is not None:
                self.frame_sink(packet)
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                    self.dropped_frames += 1
                except queue.Empty:
                    pass
            self.queue.put_nowait(packet)
            next_due += interval
            delay = next_due - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -interval:
                next_due = time.monotonic()

    def read(self, timeout: float = 1.0) -> FramePacket:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError("No camera frame received") from exc

    def close(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.capture:
            self.capture.release()


class ReplayCapture:
    """Video-file source using a synthetic monotonic timeline for reproducible replay."""

    def __init__(self, path: Path, target_fps: float | None = None):
        self.path = Path(path)
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open replay video: {path}")
        native = self.capture.get(cv2.CAP_PROP_FPS)
        self.fps = target_fps or (native if native > 0 else 15.0)
        self.frame_id = 0
        self.captured_frames = 0
        self.started = time.monotonic()

    def read(self, timeout: float = 1.0) -> FramePacket:
        ok, frame = self.capture.read()
        if not ok:
            raise EOFError
        monotonic_sec = self.started + self.frame_id / self.fps
        packet = FramePacket(
            frame_id=self.frame_id,
            utc_timestamp=datetime.now(timezone.utc).isoformat(),
            monotonic_sec=monotonic_sec,
            frame=frame,
        )
        self.frame_id += 1
        self.captured_frames = self.frame_id
        return packet

    @property
    def dropped_frames(self) -> int:
        return 0

    @property
    def actual_configuration(self) -> dict:
        return {
            "backend": "replay",
            "path": str(self.path),
            "width": int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            "fps": float(self.fps),
        }

    def close(self) -> None:
        self.capture.release()
