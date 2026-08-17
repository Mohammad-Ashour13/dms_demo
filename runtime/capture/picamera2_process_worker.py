"""Standalone Picamera2 worker for mixed-Python Raspberry Pi installations.

This file intentionally imports no DMS modules.  It is executed by the Raspberry
Pi OS system Python so its compiled libcamera binding always matches the OS.
Frames are sent as fixed-size OpenCV-compatible BGR packets to the AI runtime
over stdout. Picamera2's ``RGB888`` format has that in-memory byte order, so the
production profile avoids a per-frame channel conversion.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time

import numpy as np
from picamera2 import Picamera2


READY_MAGIC = b"DMS1"
READY_HEADER = struct.Struct("<4sIId")
FRAME_HEADER = struct.Struct("<QdI")


def _write_all(output, data) -> None:
    view = memoryview(data).cast("B")
    while view:
        written = output.write(view)
        if not written:
            raise BrokenPipeError("Picamera2 frame stream closed")
        view = view[written:]


def _probe() -> int:
    print(json.dumps(Picamera2.global_camera_info(), separators=(",", ":")), flush=True)
    return 0


def _stream(args: argparse.Namespace) -> int:
    frame_duration_us = max(1, int(round(1_000_000.0 / max(args.fps, 1.0))))
    camera = Picamera2(args.camera_num)
    request = None
    try:
        configuration = camera.create_video_configuration(
            main={"size": (args.width, args.height), "format": args.pixel_format},
            controls={"FrameDurationLimits": (frame_duration_us, frame_duration_us)},
            buffer_count=4,
        )
        camera.configure(configuration)
        applied = camera.camera_configuration()
        main = dict(applied.get("main") or {})
        width, height = main.get("size", (args.width, args.height))
        width, height = int(width), int(height)
        camera.start(show_preview=False)

        output = sys.stdout.buffer
        _write_all(output, READY_HEADER.pack(READY_MAGIC, width, height, float(args.fps)))
        output.flush()
        expected_bytes = width * height * 3
        frame_id = 0
        while True:
            request = camera.capture_request()
            try:
                frame = request.make_array("main")
                if (
                    frame.dtype != np.uint8
                    or frame.ndim != 3
                    or frame.shape != (height, width, 3)
                ):
                    raise RuntimeError(
                        f"Unexpected Picamera2 frame {frame.shape}/{frame.dtype}; "
                        f"expected {(height, width, 3)}/uint8"
                    )
                frame = np.ascontiguousarray(frame)
                payload = memoryview(frame).cast("B")
                if payload.nbytes != expected_bytes:
                    raise RuntimeError(
                        f"Unexpected Picamera2 payload size {payload.nbytes}; "
                        f"expected {expected_bytes}"
                    )
                _write_all(
                    output,
                    FRAME_HEADER.pack(frame_id, time.monotonic(), payload.nbytes),
                )
                _write_all(output, payload)
                output.flush()
                frame_id += 1
            finally:
                request.release()
                request = None
    except BrokenPipeError:
        return 0
    finally:
        if request is not None:
            request.release()
        try:
            camera.stop()
        except Exception:
            pass
        camera.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--camera-num", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--pixel-format", default="RGB888")
    args = parser.parse_args()
    return _probe() if args.probe else _stream(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Picamera2 worker failed: {exc!r}", file=sys.stderr, flush=True)
        raise
