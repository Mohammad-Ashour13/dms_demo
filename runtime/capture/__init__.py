from .base import CameraCapture
from .opencv_camera import LatestFrameCapture, ReplayCapture
from .picamera2_camera import Picamera2Capture


def create_live_capture(config, *, frame_sink=None):
    backend = str(getattr(config, "backend", "opencv")).lower()
    if backend == "picamera2":
        return Picamera2Capture(
            getattr(config, "camera_num", 0),
            config.width,
            config.height,
            config.fps,
            config.ai_queue_size,
            frame_sink,
            getattr(config, "pixel_format", "BGR888"),
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
    "ReplayCapture",
    "create_live_capture",
]
