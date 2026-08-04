from __future__ import annotations

import copy
import csv
import json

import cv2
import numpy as np
import pandas as pd

from dms_final_system.runtime.evaluation.evaluate import aggregate_reports, evaluate_session
from dms_final_system.runtime.evaluation.session import EvaluationSessionRecorder
from dms_final_system.shared.contracts import FramePacket


class FakeTelemetry:
    def __init__(self):
        self.records = []

    def emit(self, stage, event, payload=None, **kwargs):
        self.records.append((stage, event, payload or {}, kwargs))


def test_full_session_recorder_writes_video_timestamps_and_manifest(tmp_path):
    telemetry = FakeTelemetry()
    recorder = EvaluationSessionRecorder(
        tmp_path, "session-1", telemetry, fps=10.0, queue_size=16,
        metadata={"deployment_mode": "SHADOW"},
    )
    for index in range(6):
        recorder.push(
            FramePacket(
                index, f"utc-{index}", 100.0 + index / 10.0,
                np.full((24, 32, 3), index * 20, dtype=np.uint8),
            )
        )
    result = recorder.close()
    session = tmp_path / "session-1"
    assert result["status"] == "READY_FOR_ANNOTATION"
    assert result["written_frames"] == 6
    assert result["video_sha256"]
    assert len(pd.read_csv(session / "frame_timestamps.csv")) == 6
    assert list(pd.read_csv(session / "annotations.csv").columns) == [
        "session_id", "start_sec", "end_sec", "label", "confidence", "notes"
    ]
    capture = cv2.VideoCapture(str(session / "session.mp4"))
    ok, _ = capture.read()
    capture.release()
    assert ok


def _write_synthetic_session(session):
    session.mkdir()
    manifest = {
        "session_id": "synthetic-1", "deployment_mode": "SHADOW",
        "status": "READY_FOR_ANNOTATION",
        "model_version": "model-1", "dropped_frames": 0,
        "evaluation_config": {
            "sampling_interval_sec": 0.5,
            "transition_tolerance_sec": 1.0,
            "alert_merge_gap_sec": 2.0,
            "minimum_overlap_sec": 0.5,
            "primary_positive_states": ["DROWSY", "CRITICAL"],
        },
    }
    (session / "session_manifest.json").write_text(json.dumps(manifest))
    pd.DataFrame(
        {
            "video_frame_index": range(21), "frame_id": range(21),
            "video_sec": np.arange(0.0, 10.5, 0.5),
            "monotonic_sec": np.arange(100.0, 110.5, 0.5),
            "utc_timestamp": [f"utc-{i}" for i in range(21)],
        }
    ).to_csv(session / "frame_timestamps.csv", index=False)
    annotations = [
        ["synthetic-1", 0.0, 3.0, "AWAKE", 1.0, ""],
        ["synthetic-1", 3.0, 6.0, "DROWSY_SIMULATED", 1.0, ""],
        ["synthetic-1", 6.0, 10.1, "AWAKE", 1.0, ""],
    ]
    with (session / "annotations.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["session_id", "start_sec", "end_sec", "label", "confidence", "notes"])
        writer.writerows(annotations)
    records = []
    for timestamp, state in ((100.0, "NORMAL"), (103.0, "DROWSY"), (106.0, "NORMAL")):
        records.append(
            {
                "monotonic_sec": timestamp, "stage": "Fusion", "event": "decision",
                "payload": {"driver_state": state},
            }
        )
    records.append(
        {
            "monotonic_sec": 100.1, "stage": "Calibration", "event": "status",
            "payload": {"status": "READY"},
        }
    )
    for timestamp in np.arange(102.0, 110.0, 0.5):
        records.append(
            {
                "monotonic_sec": float(timestamp), "stage": "Window",
                "event": "feature_snapshot", "payload": {"valid": True},
            }
        )
        records.append(
            {
                "monotonic_sec": float(timestamp), "stage": "Events",
                "event": "snapshot", "payload": {
                    "eye_signal_valid": True, "evidence": [],
                },
            }
        )
    (session / "telemetry.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in records)
    )


def test_evaluator_builds_per_session_and_aggregate_acceptance(tmp_path):
    session = tmp_path / "synthetic"
    _write_synthetic_session(session)
    report = evaluate_session(session)
    assert report["timeline_metrics"]["fp"] == 0
    assert report["episode_metrics"]["precision"] == 1.0
    assert report["episode_metrics"]["recall"] == 1.0
    assert report["acceptance_gate"]["status"] == "SESSION_PASS"
    assert (session / "evaluation_report.md").is_file()
    reports = []
    for index in range(3):
        item = copy.deepcopy(report)
        item["session_id"] = f"synthetic-{index}"
        item["duration_sec"] = 900.0
        item["timeline_metrics"].update({"tn": 1200, "fp": 0})
        item["quality"].update(
            {
                "awake_samples": 1200,
                "awake_normal_samples": 1200,
                "awake_normal_ratio": 1.0,
                "valid_eye_observation_ratio": 1.0,
            }
        )
        item["episode_metrics"].update(
            {
                "ground_truth_episodes": 7,
                "predicted_episodes": 7,
                "true_positive_episodes": 7,
                "false_positive_episodes": 0,
                "false_negative_episodes": 0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "delays_sec": [0.5] * 7,
            }
        )
        for event in item["event_metrics"]:
            count = 10 if event["kind"] == "BLINK" else 4
            event.update(
                {
                    "ground_truth_events": count,
                    "detected_events": count,
                    "tp": count,
                    "fp": 0,
                    "fn": 0,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                }
            )
        reports.append(item)
    aggregate = aggregate_reports(reports, tmp_path / "aggregate")
    assert aggregate["acceptance_gate"]["status"] == "PERSONAL_PILOT_PASS"
    assert (tmp_path / "aggregate" / "personal_acceptance_report.md").is_file()
