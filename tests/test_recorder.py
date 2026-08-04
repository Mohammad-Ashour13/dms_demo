from __future__ import annotations

import json

import cv2
import numpy as np

from dms_final_system.runtime.recording import RollingIncidentRecorder
from dms_final_system.shared.contracts import FramePacket


class FakeTelemetry:
    def __init__(self):
        self.records = []

    def emit(self, stage, event, payload=None, **kwargs):
        self.records.append((stage, event, payload or {}, kwargs))

    def export_range(self, path, start_sec, end_sec, incident_id):
        path.write_text(json.dumps({"start": start_sec, "end": end_sec, "incident_id": incident_id}) + "\n")


def test_recorder_creates_openable_video_and_linked_json(tmp_path):
    telemetry = FakeTelemetry()
    recorder = RollingIncidentRecorder(
        tmp_path / "outbox", "session", telemetry,
        pre_alert_sec=0.4, post_alert_sec=0.2, max_clip_sec=2.0,
        fps=10.0, queue_size=32,
    )
    for index in range(8):
        frame = np.full((24, 32, 3), index * 20, dtype=np.uint8)
        recorder.push(FramePacket(index, f"utc-{index}", index / 10.0, frame))
    recorder.queue.join()
    incident_id = recorder.trigger(
        0.7, "DROWSY", ["YAWNING"], ["MODEL_ABOVE_DROWSY_ENTER"], 0.9,
        {"model_version": "m1", "feature_version": "f1", "fusion_version": "u1"},
    )
    for index in range(8, 12):
        frame = np.full((24, 32, 3), index * 10, dtype=np.uint8)
        recorder.push(FramePacket(index, f"utc-{index}", index / 10.0, frame))
    recorder.queue.join()
    recorder.close()
    directory = tmp_path / "outbox" / incident_id
    payload = json.loads((directory / "incident.json").read_text())
    assert payload["incident_id"] == incident_id
    assert payload["video_sha256"]
    capture = cv2.VideoCapture(str(directory / "video.mp4"))
    ok, _ = capture.read()
    capture.release()
    assert ok
