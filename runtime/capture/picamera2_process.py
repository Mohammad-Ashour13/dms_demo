from __future__ import annotations

import importlib.util
import json
import os
import queue
import struct
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Callable

import numpy as np

from dms_final_system.shared.contracts import FramePacket


READY_MAGIC = b"DMS1"
READY_HEADER = struct.Struct("<4sIId")
FRAME_HEADER = struct.Struct("<QdI")


def _worker_path() -> Path:
    return Path(__file__).with_name("picamera2_process_worker.py")


def _clean_worker_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        environment.pop(name, None)
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def in_process_picamera2_available() -> bool:
    if importlib.util.find_spec("picamera2") is None:
        return False
    try:
        from picamera2 import Picamera2  # noqa: F401
    except Exception:
        return False
    return True


def probe_system_picamera2(system_python: str, timeout: float = 10.0) -> list[dict]:
    result = subprocess.run(
        [str(system_python), str(_worker_path()), "--probe"],
        check=False,
        capture_output=True,
        text=True,
        timeout=max(1.0, float(timeout)),
        env=_clean_worker_environment(),
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
        raise RuntimeError(
            f"System Picamera2 probe failed with exit code {result.returncode}: {detail}"
        )
    try:
        cameras = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("System Picamera2 probe returned invalid JSON") from exc
    if not isinstance(cameras, list):
        raise RuntimeError("System Picamera2 probe did not return a camera list")
    return cameras


class Picamera2ProcessCapture:
    """CSI capture through the OS Python when its libcamera ABI differs from AI Python."""

    def __init__(
        self,
        camera_num=0,
        width=640,
        height=480,
        fps=15.0,
        queue_size=2,
        frame_sink: Callable[[FramePacket], None] | None = None,
        pixel_format="BGR888",
        system_python="/usr/bin/python3",
        startup_timeout_sec=15.0,
    ):
        self.camera_num = int(camera_num)
        self.width, self.height, self.fps = int(width), int(height), float(fps)
        self.pixel_format = str(pixel_format)
        self.system_python = str(system_python)
        self.startup_timeout_sec = max(1.0, float(startup_timeout_sec))
        self.queue: queue.Queue[FramePacket] = queue.Queue(maxsize=max(1, int(queue_size)))
        self.frame_sink = frame_sink
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.process: subprocess.Popen | None = None
        self.dropped_frames = 0
        self.captured_frames = 0
        self.capture_errors = 0
        self.last_error = ""
        self._actual_configuration: dict[str, Any] = {}
        self.configuration_lock = threading.Lock()
        self._last_capture_timestamp: float | None = None

    @staticmethod
    def _read_exact(stream: BinaryIO, length: int) -> bytes:
        chunks: list[bytes] = []
        remaining = int(length)
        while remaining:
            chunk = stream.read(remaining)
            if not chunk:
                raise EOFError("Picamera2 worker closed its frame stream")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def start(self) -> "Picamera2ProcessCapture":
        command = [
            self.system_python,
            str(_worker_path()),
            "--camera-num",
            str(self.camera_num),
            "--width",
            str(self.width),
            "--height",
            str(self.height),
            "--fps",
            str(self.fps),
            "--pixel-format",
            self.pixel_format,
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=None,
            env=_clean_worker_environment(),
            bufsize=0,
        )
        self.thread = threading.Thread(
            target=self._reader,
            name="picamera2-system-process",
            daemon=True,
        )
        self.thread.start()
        if not self.ready_event.wait(self.startup_timeout_sec):
            self.close()
            detail = self.last_error or "worker startup timed out"
            raise RuntimeError(f"Could not start system Picamera2 worker: {detail}")
        if self.capture_errors:
            self.close()
            raise RuntimeError(f"Could not start system Picamera2 worker: {self.last_error}")
        return self

    def _reader(self) -> None:
        try:
            assert self.process is not None and self.process.stdout is not None
            ready = self._read_exact(self.process.stdout, READY_HEADER.size)
            magic, width, height, fps = READY_HEADER.unpack(ready)
            if magic != READY_MAGIC:
                raise RuntimeError("Picamera2 worker protocol mismatch")
            expected_bytes = int(width) * int(height) * 3
            with self.configuration_lock:
                self._actual_configuration = {
                    "backend": "picamera2_system_process",
                    "camera_num": self.camera_num,
                    "system_python": self.system_python,
                    "requested_width": self.width,
                    "requested_height": self.height,
                    "requested_fps": self.fps,
                    "width": int(width),
                    "height": int(height),
                    "format": self.pixel_format,
                    "fps": float(fps),
                    "actual_frame_duration_us": None,
                }
            self.ready_event.set()
            while not self.stop_event.is_set():
                header = self._read_exact(self.process.stdout, FRAME_HEADER.size)
                frame_id, capture_timestamp, payload_bytes = FRAME_HEADER.unpack(header)
                if payload_bytes != expected_bytes:
                    raise RuntimeError(
                        f"Picamera2 worker sent {payload_bytes} bytes; expected {expected_bytes}"
                    )
                payload = self._read_exact(self.process.stdout, payload_bytes)
                frame = np.frombuffer(payload, dtype=np.uint8).reshape(
                    int(height), int(width), 3
                )
                previous = self._last_capture_timestamp
                self._last_capture_timestamp = float(capture_timestamp)
                if previous is not None and capture_timestamp > previous:
                    duration = float(capture_timestamp) - previous
                    with self.configuration_lock:
                        self._actual_configuration["actual_frame_duration_us"] = int(
                            duration * 1_000_000
                        )
                        self._actual_configuration["fps"] = 1.0 / duration
                packet = FramePacket(
                    frame_id=int(frame_id),
                    utc_timestamp=datetime.now(timezone.utc).isoformat(),
                    monotonic_sec=float(capture_timestamp),
                    frame=frame,
                )
                self.captured_frames = int(frame_id) + 1
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
            if not self.stop_event.is_set():
                self.capture_errors += 1
                self.last_error = repr(exc)
        finally:
            self.ready_event.set()
            self.stop_event.set()

    def read(self, timeout: float = 1.0) -> FramePacket:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty as exc:
            if self.stop_event.is_set() and self.capture_errors:
                raise RuntimeError(
                    f"System Picamera2 capture stopped: {self.last_error}"
                ) from exc
            raise TimeoutError("No system Picamera2 frame received") from exc

    @property
    def actual_configuration(self) -> dict[str, Any]:
        with self.configuration_lock:
            return dict(self._actual_configuration)

    def close(self) -> None:
        self.stop_event.set()
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=2.0)
        if process is not None and process.stdout is not None:
            process.stdout.close()
        self.process = None
