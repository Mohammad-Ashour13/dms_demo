from __future__ import annotations

import queue
import threading
from collections.abc import Callable

import cv2

from dms_final_system.shared.contracts import FramePacket

from .frames import EncodedFrame


class CompressedFrameHub:
    """JPEG-encode each camera frame once and fan out immutable packets."""

    def __init__(self, telemetry, *, jpeg_quality=75, queue_size=64):
        self.telemetry = telemetry
        self.jpeg_quality = int(jpeg_quality)
        self.queue: queue.Queue[FramePacket | None] = queue.Queue(maxsize=max(1, int(queue_size)))
        self.subscribers: list[Callable[[EncodedFrame], None]] = []
        self.dropped_frames = 0
        self.encoded_frames = 0
        self.encode_failures = 0
        self.worker = threading.Thread(target=self._run, name="compressed-frame-hub", daemon=True)
        self.worker.start()

    def subscribe(self, callback: Callable[[EncodedFrame], None]) -> None:
        self.subscribers.append(callback)

    def push(self, packet: FramePacket) -> None:
        try:
            self.queue.put_nowait(packet)
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.queue.task_done()
                self.dropped_frames += 1
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(packet)
            except queue.Full:
                self.dropped_frames += 1

    def _run(self) -> None:
        while True:
            packet = self.queue.get()
            if packet is None:
                self.queue.task_done()
                return
            try:
                ok, encoded = cv2.imencode(
                    ".jpg",
                    packet.frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                )
                if not ok:
                    raise RuntimeError("OpenCV JPEG encoder returned failure")
                frame = EncodedFrame(
                    packet.frame_id,
                    packet.monotonic_sec,
                    packet.utc_timestamp,
                    encoded.tobytes(),
                )
                self.encoded_frames += 1
                for callback in tuple(self.subscribers):
                    try:
                        callback(frame)
                    except Exception as exc:
                        self.telemetry.emit(
                            "FrameHub",
                            "subscriber_failed",
                            {"subscriber": repr(callback), "error": repr(exc)},
                            monotonic_sec=packet.monotonic_sec,
                            frame_id=packet.frame_id,
                            level="ERROR",
                        )
            except Exception as exc:
                self.encode_failures += 1
                self.telemetry.emit(
                    "FrameHub",
                    "jpeg_encode_failed",
                    {"error": repr(exc)},
                    monotonic_sec=packet.monotonic_sec,
                    frame_id=packet.frame_id,
                    level="ERROR",
                )
            finally:
                self.queue.task_done()

    def close(self) -> None:
        self.queue.join()
        self.queue.put(None)
        self.queue.join()
        self.worker.join(timeout=5.0)
        if self.worker.is_alive():
            raise RuntimeError("Compressed frame hub did not stop cleanly")

