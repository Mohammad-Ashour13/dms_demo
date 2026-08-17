from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from dms_final_system.runtime.config import SeatbeltDetectorConfig, load_config
from dms_final_system.runtime.integration import EvidenceBus
from dms_final_system.runtime.objects import SeatbeltDetector
from dms_final_system.runtime.preflight import run_preflight
from dms_final_system.shared.contracts import FramePacket


class _Telemetry:
    def __init__(self):
        self.records = []

    def emit(self, stage, event, payload=None, **kwargs):
        self.records.append((stage, event, payload or {}, kwargs))


class _Backend:
    def __init__(self, no_seatbelt=0.95):
        self.no_seatbelt = no_seatbelt

    def predict(self, _frame):
        return {
            "no_seatbelt": self.no_seatbelt,
            "seat_belt": 1.0 - self.no_seatbelt,
        }


def _config(**overrides):
    values = {
        "enabled": True,
        "required": True,
        "inference_interval_sec": 0.01,
        "activation_persistence_sec": 0.0,
        "clear_persistence_sec": 0.0,
        "evidence_refresh_sec": 0.01,
        "evidence_ttl_sec": 0.1,
        "queue_size": 1,
    }
    values.update(overrides)
    return SeatbeltDetectorConfig(**values)


def _submit_and_wait(detector, frame_id=1, timestamp=1.0):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detector.submit(
        FramePacket(frame_id, "utc", timestamp, frame),
        (240.0, 80.0, 400.0, 260.0),
    )
    deadline = time.monotonic() + 1.0
    while detector.snapshot().frame_id != frame_id and time.monotonic() < deadline:
        time.sleep(0.01)
    return detector.snapshot()


def test_shadow_mode_surfaces_persistent_missing_belt_without_evidence():
    telemetry = _Telemetry()
    evidence = EvidenceBus()
    detector = SeatbeltDetector(_config(shadow_mode=True), evidence, telemetry, backend=_Backend())
    try:
        snapshot = _submit_and_wait(detector)
        assert snapshot.health == "READY"
        assert snapshot.active_violations == ["SEATBELT_MISSING"]
        assert snapshot.shadow_mode is True
        assert evidence.drain() == []
        assert any(record[1] == "shadow_violation" for record in telemetry.records)
    finally:
        detector.close()


def test_non_shadow_mode_publishes_only_seatbelt_missing_evidence():
    telemetry = _Telemetry()
    evidence = EvidenceBus()
    detector = SeatbeltDetector(_config(shadow_mode=False), evidence, telemetry, backend=_Backend())
    try:
        _submit_and_wait(detector)
        events = evidence.drain()
        assert len(events) == 1
        assert events[0].kind == "SEATBELT_MISSING"
        assert events[0].details["seatbelt_probability"] == pytest.approx(0.05)
        assert events[0].details["shadow_mode"] is False
    finally:
        detector.close()


def test_seatbelt_prediction_clears_after_low_no_seatbelt_score():
    backend = _Backend(0.95)
    detector = SeatbeltDetector(_config(shadow_mode=True), EvidenceBus(), _Telemetry(), backend=backend)
    try:
        assert _submit_and_wait(detector, 1, 1.0).active_violations == ["SEATBELT_MISSING"]
        backend.no_seatbelt = 0.05
        assert _submit_and_wait(detector, 2, 2.0).active_violations == []
    finally:
        detector.close()


def test_pi5_profile_enables_the_seatbelt_classifier_in_shadow_mode():
    config = load_config("configs/runtime.raspberry_pi5.json")
    assert config.seatbelt_detector.enabled
    assert config.seatbelt_detector.required
    assert config.seatbelt_detector.shadow_mode
    assert config.seatbelt_detector.image_size == 224
    assert config.seatbelt_detector.ncnn_num_threads == 1
    assert config.ncnn_num_threads == 2


def test_laptop_alarm_profile_enables_the_seatbelt_classifier_in_shadow_mode():
    config = load_config("configs/runtime.laptop_alarm_demo.json")
    assert config.seatbelt_detector.enabled
    assert config.seatbelt_detector.required
    assert config.seatbelt_detector.shadow_mode
    assert config.seatbelt_detector.image_size == 224
    assert config.seatbelt_detector.ncnn_num_threads == 1


def test_preflight_verifies_the_checked_in_seatbelt_source_and_ncnn_artifacts():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/runtime.raspberry_pi5.json")
    config.behavior_detector.enabled = False
    config.recorder.enabled = False
    result = run_preflight(
        config,
        lambda value: Path(value)
        if Path(value).is_absolute()
        else root / value,
        replay=True,
    )
    assert set(result["seatbelt_ncnn_artifacts_sha256"]) == {
        "model.ncnn.bin",
        "model.ncnn.param",
    }
