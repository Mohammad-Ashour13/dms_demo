from __future__ import annotations

import hashlib
import json
import os
import subprocess

import cv2
import numpy as np

from dms_final_system.runtime.recording import CompressedFrameHub, RollingIncidentRecorder
from dms_final_system.shared.contracts import FramePacket


class FakeTelemetry:
    def __init__(self):
        self.records = []

    def emit(self, stage, event, payload=None, **kwargs):
        self.records.append((stage, event, payload or {}, kwargs))

    def export_range(self, path, start_sec, end_sec, incident_id):
        path.write_text(
            json.dumps({"start": start_sec, "end": end_sec, "incident_id": incident_id})
            + "\n",
            encoding="utf-8",
        )


def _packet(index: int, fps=15.0) -> FramePacket:
    frame = np.full((48, 64, 3), index % 255, dtype=np.uint8)
    return FramePacket(index, f"utc-{index}", index / fps, frame)


def test_shared_jpeg_copy_mux_creates_fixed_atomic_incident(tmp_path, monkeypatch):
    telemetry = FakeTelemetry()
    recorder = RollingIncidentRecorder(
        tmp_path / "outbox",
        "session",
        telemetry,
        pre_alert_sec=5.0,
        post_alert_sec=5.0,
        max_clip_sec=10.0,
        fps=15.0,
        queue_size=256,
        require_copy_mux=True,
        reserve_free_bytes=0,
    )
    hub = CompressedFrameHub(telemetry, jpeg_quality=75, queue_size=256)
    delivered = []
    hub.subscribe(recorder.push_encoded)
    hub.subscribe(delivered.append)

    real_encode = cv2.imencode
    encode_calls = 0

    def counted_encode(*args, **kwargs):
        nonlocal encode_calls
        encode_calls += 1
        return real_encode(*args, **kwargs)

    monkeypatch.setattr(cv2, "imencode", counted_encode)
    monkeypatch.setattr(
        cv2,
        "imdecode",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("copy-mux path must not decode JPEG pixels")
        ),
    )

    for index in range(76):
        hub.push(_packet(index))
    hub.queue.join()
    recorder.queue.join()
    incident_id = recorder.trigger(
        5.0,
        "DROWSY",
        [],
        ["MODEL_ABOVE_DROWSY_ENTER"],
        0.9,
        {"model_version": "m1", "feature_version": "f1", "fusion_version": "u1"},
        trigger_kinds=["DROWSY"],
    )
    for index in range(76, 151):
        hub.push(_packet(index))
    hub.close()
    recorder.close()

    directory = tmp_path / "outbox" / incident_id
    payload = json.loads((directory / "incident.json").read_text(encoding="utf-8"))
    video = directory / "video.mp4"
    assert encode_calls == 151
    assert len(delivered) == 151
    assert payload["encoder_backend"] == "ffmpeg_mjpeg_copy"
    assert payload["pixel_reencode"] is False
    assert payload["actual_start_monotonic_sec"] == 0.0
    assert payload["actual_end_monotonic_sec"] == 10.0
    assert payload["actual_duration_sec"] == 10.0
    assert payload["trigger_kinds"] == ["DROWSY"]
    assert payload["video_sha256"] == hashlib.sha256(video.read_bytes()).hexdigest()
    assert (directory / "video.mp4.sha256").read_text().startswith(payload["video_sha256"])
    assert not list((tmp_path / "outbox").glob(".*.partial"))

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,duration",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["codec_name"] == "mjpeg"
    assert abs(float(stream["duration"]) - 10.0) <= 0.5


def test_hub_fans_out_the_same_immutable_encoded_packet(tmp_path):
    telemetry = FakeTelemetry()
    left, right = [], []
    hub = CompressedFrameHub(telemetry, queue_size=4)
    hub.subscribe(left.append)
    hub.subscribe(right.append)
    hub.push(_packet(1))
    hub.close()
    assert len(left) == len(right) == 1
    assert left[0] is right[0]
    assert left[0].jpeg is right[0].jpeg


def test_oldest_finalized_incident_is_deleted_but_partial_is_preserved(tmp_path, monkeypatch):
    outbox = tmp_path / "outbox"
    outbox.mkdir()
    oldest = outbox / "incident-1"
    newest = outbox / "incident-2"
    partial = outbox / ".active.partial"
    for directory in (oldest, newest, partial):
        directory.mkdir()
    (oldest / "incident.json").write_text("{}", encoding="utf-8")
    (newest / "incident.json").write_text("{}", encoding="utf-8")
    os.utime(oldest, (1, 1))
    os.utime(newest, (2, 2))
    telemetry = FakeTelemetry()
    recorder = RollingIncidentRecorder(
        outbox,
        "session",
        telemetry,
        reserve_free_bytes=100,
        delete_oldest_when_full=True,
    )

    class Usage:
        total = 1000
        used = 950

        @property
        def free(self):
            return 200 if not oldest.exists() else 50

    monkeypatch.setattr("dms_final_system.runtime.recording.rolling.shutil.disk_usage", lambda _p: Usage())
    recorder._ensure_storage_reserve()
    recorder.close()
    assert not oldest.exists()
    assert newest.exists()
    assert partial.exists() or any(path.name.startswith(".active.quarantine") for path in outbox.iterdir())
    assert any(event == "incident_deleted_for_space" for _, event, _, _ in telemetry.records)


def test_complete_partial_incident_is_checksum_verified_and_recovered(tmp_path):
    outbox = tmp_path / "outbox"
    partial = outbox / ".incident-recover.partial"
    partial.mkdir(parents=True)
    video = partial / "video.mp4"
    video.write_bytes(b"valid-video-payload")
    checksum = hashlib.sha256(video.read_bytes()).hexdigest()
    (partial / "video.mp4.sha256").write_text(
        f"{checksum}  video.mp4\n", encoding="utf-8"
    )
    (partial / "incident.json").write_text(
        json.dumps({"incident_id": "incident-recover"}), encoding="utf-8"
    )
    (partial / "telemetry.jsonl").write_text("{}\n", encoding="utf-8")
    telemetry = FakeTelemetry()
    recorder = RollingIncidentRecorder(
        outbox,
        "session",
        telemetry,
        reserve_free_bytes=0,
    )
    recorder.close()
    recovered = outbox / "incident-recover"
    assert recovered.is_dir()
    assert not partial.exists()
    assert recorder.incident_count == 1
    assert any(event == "partial_recovered" for _, event, _, _ in telemetry.records)
