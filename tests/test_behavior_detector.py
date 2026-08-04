from __future__ import annotations

import time

import numpy as np

from dms_final_system.runtime.config import BehaviorDetectorConfig
from dms_final_system.runtime.fusion import FusionStateMachine
from dms_final_system.runtime.integration import EvidenceBus
from dms_final_system.runtime.objects import TemporalBehaviorFilter, YoloBehaviorDetector
from dms_final_system.shared.contracts import (
    AlarmLevel,
    DriverState,
    EvidenceEvent,
    ObjectDetection,
    FramePacket,
)


MAPPING = {
    "phone": "PHONE_USE",
    "cigarette": "SMOKING",
    "drink_or_food": "EATING",
}
THRESHOLDS = {"phone": 0.40, "cigarette": 0.30, "drink_or_food": 0.35}


def _detection(label: str, confidence=0.9):
    return ObjectDetection(0, label, confidence, (1.0, 2.0, 30.0, 40.0))


def _filter(**overrides):
    values = dict(
        window_sec=1.5,
        minimum_samples=3,
        activation_ratio=0.60,
        activation_persistence_sec=0.50,
        clear_persistence_sec=2.0,
        evidence_refresh_sec=1.0,
        evidence_ttl_sec=2.5,
        model_version="test-model",
    )
    values.update(overrides)
    return TemporalBehaviorFilter(MAPPING, THRESHOLDS, **values)


def test_one_frame_detection_does_not_create_a_violation():
    temporal = _filter()
    events = []
    for index, detections in enumerate(([_detection("phone")], [], [])):
        active, _, emitted = temporal.update(index * 0.5, detections, frame_id=index)
        events.extend(emitted)
    assert active == []
    assert events == []


def test_stable_phone_detection_maps_to_phone_use_once_then_refreshes():
    temporal = _filter()
    events = []
    for index in range(5):
        active, ratios, emitted = temporal.update(
            index * 0.5, [_detection("phone", 0.88)], frame_id=index
        )
        events.extend(emitted)
    assert active == ["PHONE_USE"]
    assert ratios["phone"] == 1.0
    assert [(event.kind, event.details["trigger"]) for event in events] == [
        ("PHONE_USE", "ONSET"),
        ("PHONE_USE", "REFRESH"),
    ]
    assert all(event.source == "luthfi-yolo11n-v1" for event in events)


def test_class_specific_threshold_is_applied_before_temporal_activation():
    temporal = _filter()
    for index in range(6):
        active, _, events = temporal.update(
            index * 0.5,
            [_detection("phone", 0.39), _detection("cigarette", 0.31)],
            frame_id=index,
        )
    assert active == ["SMOKING"]
    assert all(event.kind == "SMOKING" for event in events)


def test_active_behavior_clears_two_seconds_after_last_detection():
    temporal = _filter()
    for index in range(3):
        active, _, _ = temporal.update(index * 0.5, [_detection("drink_or_food")], frame_id=index)
    assert active == ["EATING"]
    for timestamp in (1.5, 2.0, 2.5):
        active, _, _ = temporal.update(timestamp, [], frame_id=int(timestamp * 10))
    assert active == ["EATING"]
    active, _, _ = temporal.update(3.0, [], frame_id=30)
    assert active == []


def test_external_object_violation_does_not_turn_normal_driver_drowsy():
    fusion = FusionStateMachine(normal_persistence_sec=0.0, transition_cooldown_sec=0.0)
    fusion.add_evidence([
        EvidenceEvent("PHONE_USE", 1.0, 0.9, 2.5, "test-yolo")
    ])
    snapshot = {
        "eye_signal_valid": True,
        "eye_evidence_trustworthy": True,
        "eye_observation_status": "VALID",
        "eye_state": "OPEN",
        "current_closure_sec": 0.0,
        "blink_count_60s": 0,
        "yawn_count_60s": 0,
        "decision_perclos_30s": None,
        "decision_valid_observation_sec_30s": 0.0,
        "decision_perclos_coverage_30s": 0.0,
    }
    decision = fusion.update(1.0, None, snapshot, True)
    assert decision.driver_state == DriverState.NORMAL
    assert decision.violations == ["PHONE_USE"]
    assert decision.alarm_level == AlarmLevel.WARNING
    assert "VIOLATION_ALARM_ACTIVE" in decision.reason_codes


class _Telemetry:
    def __init__(self):
        self.records = []

    def emit(self, stage, event, payload=None, **kwargs):
        self.records.append((stage, event, payload or {}, kwargs))


class _Backend:
    names = {
        0: "cigarette",
        1: "closed_eyes",
        2: "drink_or_food",
        3: "hand_near_head",
        4: "inattentive_gaze",
        5: "open_mouth",
        6: "phone",
    }

    def predict(self, frame, **kwargs):
        return [_detection("phone", 0.9)]


def test_downloaded_model_manifest_checksum_and_async_contract_are_valid():
    config = BehaviorDetectorConfig(
        enabled=True,
        required=True,
        model_path="dms_final_system/models/driver_behavior/luthfi_yolo11n/best_yolo11n.pt",
        manifest_path="dms_final_system/models/driver_behavior/luthfi_yolo11n/model_manifest.json",
        inference_interval_sec=0.01,
    )
    telemetry = _Telemetry()
    detector = YoloBehaviorDetector(config, EvidenceBus(), telemetry, backend=_Backend())
    try:
        detector.submit(FramePacket(7, "utc", 1.0, np.zeros((48, 64, 3), dtype=np.uint8)))
        deadline = time.monotonic() + 1.0
        while detector.snapshot().frame_id != 7 and time.monotonic() < deadline:
            time.sleep(0.01)
        snapshot = detector.snapshot()
        assert snapshot.health == "READY"
        assert snapshot.frame_id == 7
        assert snapshot.model_version == "luthfi-driver-inattention-yolo11n-7cb52d3"
        assert snapshot.detections[0].label == "phone"
    finally:
        detector.close()
