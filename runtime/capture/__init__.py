from .base import CameraCapture
from .opencv_camera import LatestFrameCapture, ReplayCapture
from .picamera2_camera import Picamera2Capture
from .picamera2_process import Picamera2ProcessCapture, in_process_picamera2_available


def create_live_capture(config, *, frame_sink=None):
    backend = str(getattr(config, "backend", "opencv")).lower()
    if backend == "picamera2_auto":
        backend = "picamera2" if in_process_picamera2_available() else "picamera2_process"
    if backend == "picamera2":
        return Picamera2Capture(
            getattr(config, "camera_num", 0),
            config.width,
            config.height,
            config.fps,
            config.ai_queue_size,
            frame_sink,
            getattr(config, "pixel_format", "RGB888"),
        ).start()
    if backend == "picamera2_process":
        return Picamera2ProcessCapture(
            getattr(config, "camera_num", 0),
            config.width,
            config.height,
            config.fps,
            config.ai_queue_size,
            frame_sink,
            getattr(config, "pixel_format", "RGB888"),
            getattr(config, "system_python", "/usr/bin/python3"),
            getattr(config, "startup_timeout_sec", 15.0),
        ).start()
    if backend == "opencv":
        return LatestFrameCapture(
            config.source,
            config.width,
            config.height,
            config.fps,
            config.ai_queue_size,
            frame_sink,
        ).start()
    raise ValueError(f"Unsupported camera backend: {backend}")


__all__ = [
    "CameraCapture",
    "LatestFrameCapture",
    "Picamera2Capture",
    "Picamera2ProcessCapture",
    "ReplayCapture",
    "create_live_capture",
]
