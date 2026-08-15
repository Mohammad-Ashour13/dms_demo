from __future__ import annotations

import json
import queue
from types import SimpleNamespace

import pytest

from dms_final_system.runtime.app import (
    _eye_signal_blockers,
    _normalize_behavior_detections,
    _normalize_face_bbox,
)
from dms_final_system.runtime.config import PowerConfig, load_config
from dms_final_system.runtime.monitoring import (
    RuntimeStatusStore,
    SafeModeController,
    StageTimingRegistry,
    parse_throttled_value,
)
from dms_final_system.runtime.recording import IncidentTriggerPolicy


def _decision(state="NORMAL", violations=()):
    return SimpleNamespace(
        driver_state=SimpleNamespace(value=state),
        violations=list(violations),
    )


def test_pi5_profile_has_fixed_camera_recording_dashboard_and_power_contracts():
    config = load_config("configs/runtime.raspberry_pi5.json")
    assert config.camera.backend == "picamera2_auto"
    assert (config.camera.width, config.camera.height, config.camera.fps) == (640, 480, 15.0)
    # Picamera2 RGB888 is BGR-compatible in memory, as expected by OpenCV.
    assert config.camera.pixel_format == "RGB888"
    # 320 is the Pi compromise selected after 256px live CSI frames made
    # eyelid/head-pose landmarks too unstable to count blinks reliably.
    assert config.perception.process_width == 320
    assert config.behavior_detector.inference_interval_sec == 0.5
    assert config.behavior_detector.max_inference_interval_sec == 0.75
    assert config.recorder.pre_alert_sec == config.recorder.post_alert_sec == 5.0
    assert config.recorder.require_copy_mux is True
    assert config.recorder.reserve_free_bytes == 2 * 1024**3
    assert config.dashboard.enabled and config.dashboard.port == 8080


def test_eye_signal_blockers_explain_dashboard_quality_and_pose_rejection():
    config = SimpleNamespace(min_eye_signal_quality=0.45, max_eye_pose_deg=25.0)
    signal = SimpleNamespace(
        face_detected=True,
        eye_resolution_valid=True,
        eye_signal_quality=0.35,
        pitch=3.0,
        yaw=28.0,
        roll=1.0,
        event_left_ear=0.25,
        event_right_ear=0.24,
        eye_signal_valid=False,
    )
    assert _eye_signal_blockers(signal, config) == [
        "LOW_EYE_QUALITY",
        "HEAD_POSE_OUT_OF_RANGE",
    ]


def test_eye_signal_blockers_are_empty_for_valid_eye_frame():
    config = SimpleNamespace(min_eye_signal_quality=0.45, max_eye_pose_deg=25.0)
    signal = SimpleNamespace(
        face_detected=True,
        eye_resolution_valid=True,
        eye_signal_quality=0.80,
        pitch=3.0,
        yaw=2.0,
        roll=1.0,
        event_left_ear=0.25,
        event_right_ear=0.24,
        eye_signal_valid=True,
    )
    assert _eye_signal_blockers(signal, config) == []


def test_face_bbox_is_normalized_for_browser_overlay_and_hidden_without_face():
    assert _normalize_face_bbox(True, (64.0, 48.0, 320.0, 240.0), 640, 480) == [
        0.1,
        0.1,
        0.5,
        0.5,
    ]
    assert _normalize_face_bbox(False, (64.0, 48.0, 320.0, 240.0), 640, 480) is None
    assert _normalize_face_bbox(True, (20.0, 20.0, 10.0, 10.0), 640, 480) is None


def test_behavior_detections_are_bounded_and_normalized_for_dashboard_overlay():
    detections = [
        SimpleNamespace(label="phone", confidence=0.7, xyxy=(64, 48, 320, 240)),
        SimpleNamespace(label="drink_or_food", confidence=0.9, xyxy=(0, 0, 640, 480)),
    ]
    assert _normalize_behavior_detections(detections, 640, 480, limit=1) == [
        {
            "label": "drink_or_food",
            "confidence": 0.9,
            "bbox_normalized": [0.0, 0.0, 1.0, 1.0],
        }
    ]


def test_invalid_yolo_temporal_sampling_contract_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "behavior_detector": {
                    "inference_interval_sec": 0.5,
                    "max_inference_interval_sec": 0.76,
                    "temporal_window_sec": 1.5,
                    "minimum_samples": 3,
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cannot supply minimum_samples"):
        load_config(path)


def test_raw_behavior_threshold_cannot_bypass_activation_thresholds(tmp_path):
    path = tmp_path / "bad-raw-threshold.json"
    path.write_text(
        json.dumps(
            {
                "behavior_detector": {
                    "raw_confidence_threshold": 0.5,
                    "class_thresholds": {
                        "phone": 0.4,
                        "cigarette": 0.3,
                        "drink_or_food": 0.35,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="raw_confidence_threshold"):
        load_config(path)


def test_incident_policy_suppresses_yawn_only_warning_and_rearms_after_clear():
    policy = IncidentTriggerPolicy(
        ["FATIGUE_WARNING", "DROWSY", "CRITICAL"],
        ["PHONE_USE", "SMOKING", "EATING"],
        suppress_yawn_only=True,
        episode_clear_sec=2.0,
    )
    yawn = policy.evaluate(1.0, _decision("FATIGUE_WARNING", ["YAWNING"]))
    assert not yawn.qualifies and not yawn.onset
    phone = policy.evaluate(2.0, _decision("NORMAL", ["PHONE_USE"]))
    assert phone.qualifies and phone.onset and phone.trigger_kinds == ("PHONE_USE",)
    overlap = policy.evaluate(3.0, _decision("DROWSY", ["PHONE_USE"]))
    assert overlap.qualifies and not overlap.onset
    policy.evaluate(4.0, _decision())
    policy.evaluate(6.1, _decision())
    next_episode = policy.evaluate(7.0, _decision("DROWSY"))
    assert next_episode.onset


def test_current_and_historical_throttling_flags_are_distinct():
    flags = parse_throttled_value("throttled=0x50005")
    assert flags["undervoltage_current"]
    assert flags["throttled_current"]
    assert flags["undervoltage_occurred"]
    assert flags["throttled_occurred"]
    assert not flags["frequency_capped_current"]


def test_safe_mode_enters_immediately_and_requires_cool_hysteresis():
    controller = SafeModeController(PowerConfig(recovery_sec=60.0))
    entered = controller.update(
        10.0,
        {"temperature_c": 70.0, "throttled": {"undervoltage_current": True}},
    )
    assert entered.enabled and entered.changed and entered.reason == "UNDERVOLTAGE"
    waiting = controller.update(
        20.0,
        {"temperature_c": 71.0, "throttled": {"undervoltage_current": False}},
    )
    assert waiting.enabled and not waiting.changed
    still_waiting = controller.update(
        79.0,
        {"temperature_c": 71.0, "throttled": {"undervoltage_current": False}},
    )
    assert still_waiting.enabled
    recovered = controller.update(
        80.1,
        {"temperature_c": 71.0, "throttled": {"undervoltage_current": False}},
    )
    assert not recovered.enabled and recovered.changed


def test_dashboard_status_schema_and_stage_percentiles():
    status = RuntimeStatusStore().update(runtime={"status": "RUNNING"})
    assert status["schema_version"] == "dashboard-status-v1"
    registry = StageTimingRegistry()
    for value in (1.0, 2.0, 3.0, 4.0, 5.0):
        registry.record("fusion", value)
    snapshot = registry.snapshot()["fusion"]
    assert snapshot["p50_ms"] == 3.0
    assert snapshot["p95_ms"] == pytest.approx(4.8)


def test_latest_only_queue_replaces_stale_value():
    from dms_final_system.runtime.dashboard.process import DashboardProcess

    target = queue.Queue(maxsize=1)
    assert DashboardProcess._put_latest(target, {"value": 1})
    assert DashboardProcess._put_latest(target, {"value": 2})
    assert target.get_nowait() == {"value": 2}
