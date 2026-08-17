from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

from dms_final_system.runtime.capture.picamera2_camera import Picamera2Capture
from dms_final_system.runtime.capture.picamera2_process import Picamera2ProcessCapture
import dms_final_system.runtime.capture.picamera2_process as process_capture_module


class _Request:
    def __init__(self):
        self.released = False

    def make_array(self, _name):
        return np.zeros((48, 64, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class _Camera:
    instances = []

    def __init__(self, number):
        self.number = number
        self.controls = None
        self.closed = False
        self.__class__.instances.append(self)

    def create_video_configuration(self, **kwargs):
        self.controls = kwargs["controls"]
        return {"main": kwargs["main"]}

    def configure(self, configuration):
        self.configuration = configuration

    def camera_configuration(self):
        return {"main": {"size": (64, 48), "format": "RGB888"}}

    def start(self, show_preview=False):
        assert show_preview is False

    def capture_request(self):
        return _Request()

    def stop(self):
        pass

    def close(self):
        self.closed = True


def test_picamera2_adapter_applies_fixed_headless_configuration(monkeypatch):
    module = types.ModuleType("picamera2")
    module.Picamera2 = _Camera
    monkeypatch.setitem(sys.modules, "picamera2", module)
    capture = Picamera2Capture(
        camera_num=1,
        width=64,
        height=48,
        fps=15.0,
        queue_size=1,
        pixel_format="RGB888",
    ).start()
    try:
        packet = capture.read(timeout=1.0)
        assert packet.frame.shape == (48, 64, 3)
        assert packet.monotonic_sec > 0
        assert capture.actual_configuration == {
            "backend": "picamera2",
            "camera_num": 1,
            "requested_width": 64,
            "requested_height": 48,
            "requested_fps": 15.0,
            "width": 64,
            "height": 48,
            "format": "RGB888",
            "fps": 15.0,
            "frame_duration_limits_us": [66667, 66667],
            "actual_frame_duration_us": None,
        }
        assert _Camera.instances[-1].controls == {
            "FrameDurationLimits": (66667, 66667)
        }
    finally:
        capture.close()
    assert _Camera.instances[-1].closed


def test_system_python_picamera2_process_uses_bounded_binary_frames(
    monkeypatch, tmp_path
):
    worker = tmp_path / "fake_camera_worker.py"
    worker.write_text(
        """
import struct
import sys
import time

ready = struct.Struct("<4sIId")
frame = struct.Struct("<QdI")
output = sys.stdout.buffer
output.write(ready.pack(b"DMS1", 8, 6, 15.0))
output.flush()
payload = bytes([17]) * (8 * 6 * 3)
for frame_id in range(4):
    output.write(frame.pack(frame_id, time.monotonic(), len(payload)))
    output.write(payload)
    output.flush()
    time.sleep(0.01)
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(process_capture_module, "_worker_path", lambda: Path(worker))
    capture = Picamera2ProcessCapture(
        width=8,
        height=6,
        fps=15.0,
        queue_size=1,
        system_python=sys.executable,
    ).start()
    try:
        packet = capture.read(timeout=1.0)
        assert packet.frame.shape == (6, 8, 3)
        assert np.all(packet.frame == 17)
        assert packet.monotonic_sec > 0
        assert capture.actual_configuration["backend"] == "picamera2_system_process"
        assert capture.actual_configuration["system_python"] == sys.executable
    finally:
        capture.close()
