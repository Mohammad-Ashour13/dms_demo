from __future__ import annotations

import json

from dms_final_system.runtime.telemetry import StructuredTelemetry


def test_structured_log_and_incident_export_share_ids(tmp_path):
    telemetry = StructuredTelemetry(tmp_path / "logs", "session-1")
    telemetry.emit("Model", "prediction", {"p": 0.8}, monotonic_sec=10.0, frame_id=3, window_id="w-1")
    telemetry.emit("Fusion", "decision", {"state": "DROWSY"}, monotonic_sec=10.1, frame_id=3, window_id="w-1")
    telemetry.close()
    records = [json.loads(line) for line in telemetry.path.read_text().splitlines()]
    assert records[0]["session_id"] == "session-1"
    assert records[0]["window_id"] == "w-1"
