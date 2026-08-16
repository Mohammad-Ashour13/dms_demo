from __future__ import annotations

import json
import os

from dms_final_system.runtime.telemetry import StructuredTelemetry


def test_structured_log_and_incident_export_share_ids(tmp_path):
    telemetry = StructuredTelemetry(tmp_path / "logs", "session-1")
    telemetry.emit("Model", "prediction", {"p": 0.8}, monotonic_sec=10.0, frame_id=3, window_id="w-1")
    telemetry.emit("Fusion", "decision", {"state": "DROWSY"}, monotonic_sec=10.1, frame_id=3, window_id="w-1")
    telemetry.close()
    records = [json.loads(line) for line in telemetry.path.read_text().splitlines()]
    assert records[0]["session_id"] == "session-1"
    assert records[0]["window_id"] == "w-1"


def test_normal_mode_throttles_high_volume_records_but_keeps_changes(tmp_path):
    telemetry = StructuredTelemetry(tmp_path / "logs", "session-1", mode="NORMAL")
    telemetry.emit("Fusion", "decision", {"state": "NORMAL"}, monotonic_sec=10.0)
    telemetry.emit("Fusion", "decision", {"state": "NORMAL"}, monotonic_sec=11.0)
    telemetry.emit(
        "Fusion", "decision", {"state": "DROWSY", "changed": True},
        monotonic_sec=11.1,
    )
    telemetry.emit(
        "Fusion", "decision", {"state": "CRITICAL"}, monotonic_sec=11.2,
        level="WARNING",
    )
    telemetry.close()
    records = [json.loads(line) for line in telemetry.path.read_text().splitlines()]
    assert [record["payload"]["state"] for record in records] == [
        "NORMAL", "DROWSY", "CRITICAL",
    ]


def test_old_session_logs_are_bounded_on_startup(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    for index in range(5):
        path = log_dir / f"old-{index}.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        os.utime(path, (index + 1, index + 1))

    telemetry = StructuredTelemetry(log_dir, "current", backups=2)
    telemetry.close()

    assert sorted(path.name for path in log_dir.glob("*.jsonl")) == [
        "current.jsonl", "old-3.jsonl", "old-4.jsonl",
    ]
    assert telemetry.pruned_log_files == 3
