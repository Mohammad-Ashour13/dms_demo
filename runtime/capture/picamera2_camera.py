from __future__ import annotations

import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable

from dms_final_system.shared.contracts import FramePacket


class Picamera2Capture:
    """Headless, bounded CSI-camera reader built on Raspberry Pi's libcamera stack."""

    def __init__(
        self,
        camera_num=0,
        width=640,
        height=480,
        fps=15.0,
        queue_size=2,
        frame_sink: Callable[[FramePacket], None] | None = None,
        pixel_format="RGB888",
    ):
        self.camera_num = int(camera_num)
        self.width, self.height, self.fps = int(width), int(height), float(fps)
        self.pixel_format = str(pixel_format)
        self.queue: queue.Queue[FramePacket] = queue.Queue(maxsize=max(1, int(queue_size)))
        self.frame_sink = frame_sink
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.camera = None
        self.dropped_frames = 0
        self.captured_frames = 0
        self.capture_errors = 0
        self.last_error = ""
        self._actual_configuration: dict[str, Any] = {}
        self.configuration_lock = threading.Lock()

    def start(self) -> "Picamera2Capture":
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "Picamera2 is required for CSI camera capture. On Raspberry Pi OS Lite run: "
                "sudo apt install -y python3-picamera2 --no-install-recommends"
            ) from exc
        frame_duration_us = max(1, int(round(1_000_000.0 / max(self.fps, 1.0))))
        self.camera = Picamera2(self.camera_num)
        configuration = self.camera.create_video_configuration(
            main={
                "size": (self.width, self.height),
                "format": self.pixel_format,
            },
            controls={"FrameDurationLimits": (frame_duration_us, frame_duration_us)},
            buffer_count=4,
        )
        self.camera.configure(configuration)
        applied = self.camera.camera_configuration()
        main = dict(applied.get("main") or {})
        actual_configuration = {
            "backend": "picamera2",
            "camera_num": self.camera_num,
            "requested_width": self.width,
            "requested_height": self.height,
            "requested_fps": self.fps,
            "width": int(main.get("size", (self.width, self.height))[0]),
            "height": int(main.get("size", (self.width, self.height))[1]),
            "format": str(main.get("format", self.pixel_format)),
            "fps": self.fps,
            "frame_duration_limits_us": [frame_duration_us, frame_duration_us],
            "actual_frame_duration_us": None,
        }
        with self.configuration_lock:
            self._actual_configuration = actual_configuration
        self.camera.start(show_preview=False)
        self.thread = threading.Thread(target=self._reader, name="picamera2-capture", daemon=True)
        self.thread.start()
        return self

    def _reader(self) -> None:
        frame_id = 0
        assert self.camera is not None
        while not self.stop_event.is_set():
            request = None
            try:
                request = self.camera.capture_request()
                # Own the pixels before releasing libcamera's request buffer.
                frame = request.make_array("main").copy()
                metadata_reader = getattr(request, "get_metadata", None)
                metadata = (metadata_reader() if callable(metadata_reader) else {}) or {}
                actual_duration = metadata.get("FrameDuration")
                if actual_duration:
                    with self.configuration_lock:
                        self._actual_configuration["actual_frame_duration_us"] = int(
                            actual_duration
                        )
                        self._actual_configuration["fps"] = 1_000_000.0 / float(
                            actual_duration
                        )
                now = time.monotonic()
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
            except Exception as exc:
                self.capture_errors += 1
                self.last_error = repr(exc)
                if not self.stop_event.is_set():
                    self.stop_event.set()
            finally:
                if request is not None:
                    request.release()

    def read(self, timeout: float = 1.0) -> FramePacket:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty as exc:
            if self.stop_event.is_set() and self.capture_errors:
                raise RuntimeError(
                    f"Picamera2 capture stopped after a camera error: {self.last_error}"
                ) from exc
            raise TimeoutError("No Picamera2 frame received") from exc

    @property
    def actual_configuration(self) -> dict[str, Any]:
        with self.configuration_lock:
            return dict(self._actual_configuration)

    def close(self) -> None:
        self.stop_event.set()
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:
                pass
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.camera is not None:
            try:
                self.camera.close()
            finally:
                self.camera = None
