from __future__ import annotations

import time

import numpy as np

from dms_final_system.runtime.config import BehaviorDetectorConfig
from dms_final_system.runtime.fusion import FusionStateMachine
from dms_final_system.runtime.integration import EvidenceBus
from dms_final_system.runtime.objects import (
    TemporalBehaviorFilter,
    YoloBehaviorDetector,
    resolve_execution_mode,
    resolve_yolo_backend,
)
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


class _EmptyBackend(_Backend):
    def predict(self, frame, **kwargs):
        return []


def _detector_config(tmp_path, **overrides):
    model = tmp_path / "dummy.pt"
    model.write_bytes(b"dummy")
    manifest = tmp_path / "model_manifest.json"
    manifest.write_text(
        """{
  "model_version": "test-yolo",
  "source_classes": ["cigarette", "closed_eyes", "drink_or_food", "hand_near_head", "inattentive_gaze", "open_mouth", "phone"]
}"""
    )
    values = dict(
        enabled=True,
        required=True,
        model_path=str(model),
        manifest_path=str(manifest),
        inference_interval_sec=0.01,
    )
    values.update(overrides)
    return BehaviorDetectorConfig(**values)


def test_backend_auto_resolver_infers_runtime_from_model_path(tmp_path):
    ncnn_dir = tmp_path / "best_yolo11n_ncnn_model"
    ncnn_dir.mkdir()
    assert resolve_yolo_backend(ncnn_dir, "auto") == "ncnn"
    assert resolve_yolo_backend("driver.onnx", "auto") == "onnx"
    assert resolve_yolo_backend("driver.pt", "auto") == "pytorch"
    assert resolve_yolo_backend("ignored", "ultralytics") == "pytorch"
    assert resolve_execution_mode("ncnn", "auto") == "thread"
    assert resolve_execution_mode("onnx", "auto") == "thread"
    assert resolve_execution_mode("pytorch", "auto") == "process"
    assert resolve_execution_mode("ncnn", "process") == "process"


def test_static_roi_crop_preserves_full_frame_offsets():
    config = BehaviorDetectorConfig(
        enabled=False,
        roi_mode="static",
        driver_roi=[0.50, 0.25, 1.0, 1.0],
    )
    detector = YoloBehaviorDetector(config, EvidenceBus(), _Telemetry())
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    crop, offset_x, offset_y, info = detector._crop(frame)
    assert crop.shape[:2] == (60, 60)
    assert (offset_x, offset_y) == (60, 20)
    assert info["crop_xyxy"] == [60, 20, 120, 80]


def test_face_roi_crop_contains_face_and_is_clamped_to_frame():
    config = BehaviorDetectorConfig(
        enabled=False,
        roi_mode="face",
        roi_scale=2.0,
        roi_min_size=40,
    )
    detector = YoloBehaviorDetector(config, EvidenceBus(), _Telemetry())
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    crop, offset_x, offset_y, info = detector._crop(frame, (60.0, 30.0, 100.0, 70.0))
    x1, y1, x2, y2 = info["crop_xyxy"]
    assert info["mode"] == "face"
    assert x1 <= 60 <= 100 <= x2
    assert y1 <= 30 <= 70 <= y2
    assert crop.shape[:2] == (y2 - y1, x2 - x1)
    assert (offset_x, offset_y) == (x1, y1)
    assert 0 <= x1 < x2 <= 160
    assert 0 <= y1 < y2 <= 100


def test_face_roi_falls_back_to_static_when_face_missing():
    config = BehaviorDetectorConfig(
        enabled=False,
        roi_mode="face",
        driver_roi=[0.0, 0.0, 0.5, 1.0],
        roi_fallback_static_on_no_face=True,
    )
    detector = YoloBehaviorDetector(config, EvidenceBus(), _Telemetry())
    frame = np.zeros((40, 80, 3), dtype=np.uint8)
    crop, offset_x, offset_y, info = detector._crop(frame, None)
    assert crop.shape[:2] == (40, 40)
    assert (offset_x, offset_y) == (0, 0)
    assert info["fallback_reason"] == "no_face"


def test_adaptive_interval_grows_and_shrinks_with_latency_budget(tmp_path):
    telemetry = _Telemetry()
    detector = YoloBehaviorDetector(
        _detector_config(
            tmp_path,
            adaptive_interval=True,
            latency_budget_ms=10.0,
            max_inference_interval_sec=0.05,
        ),
        EvidenceBus(),
        telemetry,
        backend=_EmptyBackend(),
    )
    try:
        base = detector.effective_interval_sec
        detector._update_effective_interval(20.0, 1.0, 1)
        assert detector.effective_interval_sec > base
        detector._update_effective_interval(1.0, 2.0, 2)
        assert detector.effective_interval_sec == base
        changes = [record for record in telemetry.records if record[1] == "interval_changed"]
        assert len(changes) == 2
    finally:
        detector.close()


def test_context_detection_holds_context_long_enough_for_temporal_samples(tmp_path):
    detector = YoloBehaviorDetector(
        _detector_config(
            tmp_path,
            roi_mode="face",
            context_roi_interval_sec=2.0,
            adaptive_interval=False,
        ),
        EvidenceBus(),
        _Telemetry(),
        backend=_Backend(),
    )
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    try:
        for frame_id, timestamp in enumerate((1.0, 1.5, 2.0, 2.5), start=1):
            detector.submit(
                FramePacket(frame_id, "utc", timestamp, frame),
                (60, 30, 100, 70),
            )
            deadline = time.monotonic() + 1.0
            while (
                detector.snapshot().frame_id != frame_id
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            expected = "context_reacquisition" if frame_id == 1 else "context_followup"
            assert detector.last_crop_info["mode"] == expected
        assert detector.snapshot().active_behaviors == ["PHONE_USE"]
    finally:
        detector.close()


def test_face_roi_periodically_reacquires_driver_context(tmp_path):
    detector = YoloBehaviorDetector(
        _detector_config(
            tmp_path,
            roi_mode="face",
            context_roi_interval_sec=1.0,
            adaptive_interval=False,
        ),
        EvidenceBus(),
        _Telemetry(),
        backend=_EmptyBackend(),
    )
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    try:
        detector.submit(FramePacket(1, "utc", 1.0, frame), (60, 30, 100, 70))
        deadline = time.monotonic() + 1.0
        while detector.snapshot().frame_id != 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert detector.last_crop_info["mode"] == "context_reacquisition"

        detector.submit(FramePacket(2, "utc", 1.1, frame), (60, 30, 100, 70))
        deadline = time.monotonic() + 1.0
        while detector.snapshot().frame_id != 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert detector.last_crop_info["mode"] == "face"

        detector.submit(FramePacket(3, "utc", 2.1, frame), (60, 30, 100, 70))
        deadline = time.monotonic() + 1.0
        while detector.snapshot().frame_id != 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert detector.last_crop_info["mode"] == "context_reacquisition"
    finally:
        detector.close()


def test_downloaded_model_manifest_checksum_and_async_contract_are_valid():
    config = BehaviorDetectorConfig(
        enabled=True,
        required=True,
        model_path="models/driver_behavior/luthfi_yolo11n/best_yolo11n.pt",
        manifest_path="models/driver_behavior/luthfi_yolo11n/model_manifest.json",
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
