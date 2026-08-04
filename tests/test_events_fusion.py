from __future__ import annotations

from dms_final_system.runtime.events import EventEngine
from dms_final_system.runtime.fusion import FusionStateMachine
from dms_final_system.shared.contracts import (
    DriverState,
    EvidenceEvent,
    FaceSignal,
    ModelPrediction,
)


def _face(
    t: float,
    event_relative: float,
    *,
    valid=True,
    left_relative: float | None = None,
    right_relative: float | None = None,
    binocular_consistent=True,
    gaze_x=0.0,
    gaze_y=0.0,
) -> FaceSignal:
    left_relative = event_relative if left_relative is None else left_relative
    right_relative = event_relative if right_relative is None else right_relative
    left_event_ear = 0.30 * left_relative
    right_event_ear = 0.30 * right_relative
    event_ear = (left_event_ear + right_event_ear) / 2
    return FaceSignal(
        int(t * 1000), "utc", t, True, 1.0,
        left_ear=0.2, right_ear=0.2, ear=0.2, relative_ear=1.0,
        event_left_ear=left_event_ear,
        event_right_ear=right_event_ear,
        event_ear=event_ear,
        event_left_relative_ear=left_relative,
        event_right_relative_ear=right_relative,
        event_relative_ear=(left_relative + right_relative) / 2,
        eye_signal_valid=valid,
        eye_signal_quality=1.0 if valid else 0.0,
        binocular_consistent=binocular_consistent,
        gaze_x=gaze_x,
        gaze_y=gaze_y,
        mar=0.1,
    )


def _empty_events(
    t: float,
    *,
    eye_state="OPEN",
    valid=True,
    closure=0.0,
    perclos=None,
    observed=0.0,
    coverage=None,
    observation_status=None,
    gap=0.0,
    yaw_count=0,
    blink_count=0,
    strong_closure=None,
    sustained_partial=False,
    decision_perclos_marker=False,
):
    if coverage is None:
        coverage = min(1.0, observed / 30.0)
    result = {
        "monotonic_sec": t,
        "eye_state": eye_state,
        "eye_signal_valid": valid,
        "current_closure_sec": closure,
        "perclos_30s": perclos,
        "valid_observation_sec_30s": observed,
        "perclos_coverage_30s": coverage,
        "blink_count_60s": blink_count,
        "yawn_count_60s": yaw_count,
        "eye_observation_status": observation_status or ("VALID" if valid else "LOST"),
        "eye_signal_gap_sec": gap,
        "eye_evidence_trustworthy": valid,
        "strong_closure_valid": closure >= 1.5 if strong_closure is None else strong_closure,
        "sustained_partial_closure": sustained_partial,
    }
    if decision_perclos_marker:
        result.update({
            "decision_perclos_30s": perclos,
            "decision_valid_observation_sec_30s": observed,
            "decision_perclos_coverage_30s": coverage,
        })
    return result


def test_one_frame_blink_at_15_fps_is_counted_once():
    engine = EventEngine()
    dt = 1.0 / 15.0
    engine.update(_face(0.0, 1.0), 0.1, 0.0)
    engine.update(_face(dt, 0.4), 0.1, 0.0)
    result = engine.update(_face(2 * dt, 1.0), 0.1, 0.0)
    assert result.blink_count_60s == 1
    assert [event.kind for event in result.evidence] == ["BLINK"]
    result = engine.update(_face(3 * dt, 1.0), 0.1, 0.0)
    assert result.blink_count_60s == 1
    assert not result.evidence


def test_five_blinks_are_not_merged_or_duplicated():
    engine = EventEngine()
    dt = 1.0 / 15.0
    t = 0.0
    engine.update(_face(t, 1.0), 0.1, 0.0)
    emitted = 0
    for _ in range(5):
        t += dt
        engine.update(_face(t, 0.4), 0.1, 0.0)
        t += dt
        result = engine.update(_face(t, 1.0), 0.1, 0.0)
        emitted += sum(event.kind == "BLINK" for event in result.evidence)
        t += dt
        engine.update(_face(t, 1.0), 0.1, 0.0)
    assert result.blink_count_60s == 5
    assert emitted == 5


def test_two_eyes_may_close_one_frame_apart_and_count_one_blink():
    engine = EventEngine(binocular_close_sync_sec=0.15)
    dt = 1.0 / 15.0
    engine.update(_face(0.0, 1.0), 0.1, 0.0)
    engine.update(
        _face(dt, 0.7, left_relative=0.4, right_relative=1.0, binocular_consistent=False),
        0.1,
        0.0,
    )
    engine.update(_face(2 * dt, 0.4), 0.1, 0.0)
    result = engine.update(_face(3 * dt, 1.0), 0.1, 0.0)
    assert result.blink_count_60s == 1
    assert [event.kind for event in result.evidence] == ["BLINK"]


def test_one_eye_closure_is_not_a_blink_or_critical_closure():
    engine = EventEngine()
    dt = 1.0 / 15.0
    engine.update(_face(0.0, 1.0), 0.1, 0.0)
    middle = engine.update(
        _face(dt, 0.7, left_relative=0.4, right_relative=1.0, binocular_consistent=False),
        0.1,
        0.0,
    )
    result = engine.update(_face(2 * dt, 1.0), 0.1, 0.0)
    assert middle.eye_state == "OPEN"
    assert middle.current_closure_sec == 0.0
    assert result.blink_count_60s == 0
    assert not result.evidence


def test_extreme_gaze_cannot_start_a_false_eye_closure():
    engine = EventEngine(max_eye_gaze_offset=0.20)
    engine.update(_face(0.0, 1.0), 0.1, 0.0)
    for index in range(1, 31):
        result = engine.update(
            _face(index / 15.0, 0.5, gaze_x=-0.35),
            0.1,
            0.0,
        )
    assert result.eye_state == "OPEN"
    assert result.current_closure_sec == 0.0
    assert not result.closure_measurement_valid
    assert not result.evidence


def test_strong_bilateral_closure_ignores_unreliable_iris_coordinates():
    """Regression: real closed eyes produced bad iris gaze at 325-365s."""
    engine = EventEngine(prolonged_closure_sec=1.5)
    engine.update(_face(0.0, 1.0), 0.1, 0.0)
    emitted = []
    for index in range(1, 31):
        result = engine.update(
            _face(index / 15.0, 0.2, gaze_x=0.85, gaze_y=0.85),
            0.1,
            0.0,
        )
        emitted.extend(result.evidence)
    assert result.eye_state == "CLOSED"
    assert result.strong_closure_valid
    assert any(event.kind == "PROLONGED_EYE_CLOSURE" for event in emitted)


def test_short_invalid_gap_is_grace_and_does_not_invent_blink():
    engine = EventEngine(max_eye_signal_gap_sec=0.30)
    engine.update(_face(0.0, 1.0), 0.1, 0.0)
    grace = engine.update(_face(0.10, 1.0, valid=False), 0.1, 0.0)
    result = engine.update(_face(0.20, 1.0), 0.1, 0.0)
    assert grace.eye_observation_status == "GRACE"
    assert grace.eye_state == "OPEN"
    assert grace.eye_evidence_trustworthy
    assert not grace.raw_eye_signal_valid
    assert result.blink_count_60s == 0


def test_invalid_eye_samples_do_not_raise_perclos():
    engine = EventEngine(perclos_min_observation_sec=0.1)
    for index in range(40):
        result = engine.update(
            _face(index / 15.0, 0.3, valid=False), 0.1, 0.0
        )
    assert result.eye_state == "UNKNOWN"
    assert result.valid_observation_sec_30s == 0.0
    assert result.perclos_30s is None


def test_perclos_is_time_weighted_with_irregular_samples():
    engine = EventEngine(perclos_min_observation_sec=0.1, max_eye_signal_gap_sec=2.0)
    engine.update(_face(0.0, 1.0), 0.1, 0.0)
    engine.update(_face(1.0, 0.4), 0.1, 0.0)
    engine.update(_face(1.5, 0.4), 0.1, 0.0)
    result = engine.update(_face(2.0, 1.0), 0.1, 0.0)
    # 0-1 was open; 1-2 was closed. Sample counts would give a different value.
    assert result.valid_observation_sec_30s == 2.0
    assert result.perclos_30s == 0.5


def test_critical_eye_closure_bypasses_model_but_requires_trusted_eye():
    engine = EventEngine(prolonged_closure_sec=1.5)
    engine.update(_face(0.0, 1.0), 0.1, 0.0)
    engine.update(_face(0.1, 0.2), 0.1, 0.0)
    critical_events = None
    emitted = []
    for index in range(2, 26):
        critical_events = engine.update(_face(index / 15.0, 0.2), 0.1, 0.0)
        emitted.extend(critical_events.evidence)
    assert critical_events is not None
    assert any(event.kind == "PROLONGED_EYE_CLOSURE" for event in emitted)
    fusion = FusionStateMachine()
    fusion.add_evidence(emitted)
    decision = fusion.update(
        critical_events.monotonic_sec,
        None,
        critical_events,
        True,
        model_contribution_enabled=False,
        eye_evidence_trustworthy=True,
    )
    assert decision.driver_state == DriverState.CRITICAL


def test_experimental_model_alone_stays_model_risk_pending():
    fusion = FusionStateMachine(
        warning_persistence_sec=2.0,
        transition_cooldown_sec=0.0,
        require_evidence_for_drowsy=True,
    )
    metrics = _empty_events(0.0)
    for index, timestamp in enumerate((0.0, 1.0, 2.1, 3.1)):
        prediction = ModelPrediction(
            f"w{index}", timestamp, 0.99, 0.99, 0.5, True, "experimental"
        )
        decision = fusion.update(
            timestamp, prediction, metrics, True,
            model_contribution_enabled=True,
            eye_evidence_trustworthy=True,
        )
    assert decision.driver_state == DriverState.NORMAL
    assert decision.model_risk_pending
    assert "MODEL_RISK_PENDING" in decision.reason_codes


def test_too_small_eye_resolution_becomes_unknown_not_normal():
    fusion = FusionStateMachine(
        quality_loss_persistence_sec=0.5,
        transition_cooldown_sec=0.0,
    )
    metrics = _empty_events(
        1.0, valid=False, observation_status="LOST", gap=1.0
    )
    metrics.update({
        "driver_distance_status": "TOO_FAR",
        "eye_resolution_valid": False,
        "face_width_ratio": 0.17,
        "interocular_distance_px": 45.0,
        "minimum_eye_width_px": 20.0,
    })
    decision = fusion.update(
        1.0,
        None,
        metrics,
        True,
        eye_evidence_trustworthy=False,
    )
    assert decision.driver_state == DriverState.UNKNOWN
    assert "DRIVER_TOO_FAR" in decision.reason_codes


def test_normal_requires_three_seconds_below_exit_threshold():
    fusion = FusionStateMachine(normal_persistence_sec=3.0, transition_cooldown_sec=0.0)
    metrics = _empty_events(0.0)
    for index, timestamp in enumerate((0.0, 2.0)):
        low = ModelPrediction(f"w{index}", timestamp, 0.1, 0.1, 0.5, False, "m")
        decision = fusion.update(timestamp, low, metrics, True)
        assert decision.driver_state == DriverState.UNKNOWN
    low = ModelPrediction("w2", 3.1, 0.1, 0.1, 0.5, False, "m")
    decision = fusion.update(3.1, low, metrics, True)
    assert decision.driver_state == DriverState.NORMAL


def test_model_plus_recent_physiology_can_reach_drowsy():
    fusion = FusionStateMachine(drowsy_persistence_sec=1.5, transition_cooldown_sec=0.0)
    fusion.add_evidence([EvidenceEvent("LONG_BLINK", 0.0, 1.0, 5.0, "test", 0.9)])
    metrics = _empty_events(0.0)
    for index, timestamp in enumerate((0.0, 0.8, 1.6)):
        prediction = ModelPrediction(f"w{index}", timestamp, 0.9, 0.9, 0.5, True, "m")
        decision = fusion.update(
            timestamp, prediction, metrics, True,
            model_contribution_enabled=True,
            eye_evidence_trustworthy=True,
        )
    assert decision.driver_state == DriverState.DROWSY


def test_model_plus_current_strong_closure_reaches_drowsy_before_critical():
    fusion = FusionStateMachine(
        precritical_drowsy_closure_sec=0.5,
        transition_cooldown_sec=0.0,
    )
    metrics = _empty_events(
        1.0,
        eye_state="CLOSED",
        closure=0.6,
        strong_closure=False,
    )
    metrics.update({
        # Current iris/gaze may be invalid after the already-trusted bilateral
        # closure started; that must not erase the pre-critical DROWSY stage.
        "closure_measurement_valid": False,
        "binocular_consistent": True,
        "event_left_relative_ear": 0.20,
        "event_right_relative_ear": 0.22,
    })
    prediction = ModelPrediction("w0", 1.0, 0.9, 0.9, 0.5, True, "m")
    decision = fusion.update(1.0, prediction, metrics, True)
    assert decision.driver_state == DriverState.DROWSY
    assert "CURRENT_STRONG_BILATERAL_CLOSURE" in decision.reason_codes


def test_critical_recovers_after_three_seconds_of_continuous_open_eye():
    fusion = FusionStateMachine(recovery_persistence_sec=3.0, transition_cooldown_sec=0.0)
    closure = _empty_events(0.0, eye_state="CLOSED", closure=1.5)
    # A historical event may remain in telemetry/violations, but open eyes start
    # recovery immediately rather than extending CRITICAL by its TTL.
    fusion.add_evidence([EvidenceEvent("PROLONGED_EYE_CLOSURE", 0.0, 1.0, 5.0, "test", 1.5)])
    assert fusion.update(0.0, None, closure, True).driver_state == DriverState.CRITICAL
    for timestamp in (1.0, 2.0, 4.1):
        decision = fusion.update(
            timestamp, None, _empty_events(timestamp), True,
            model_contribution_enabled=False,
            eye_evidence_trustworthy=True,
        )
    assert decision.driver_state == DriverState.NORMAL
    assert decision.state_entry_reason[0] == "CRITICAL_RECOVERY_COMPLETE"


def test_short_gap_never_changes_normal_and_long_loss_requires_half_second():
    fusion = FusionStateMachine(
        normal_persistence_sec=0.0,
        quality_loss_persistence_sec=0.5,
        transition_cooldown_sec=0.0,
    )
    decision = fusion.update(0.0, None, _empty_events(0.0), True)
    assert decision.driver_state == DriverState.NORMAL
    grace = _empty_events(
        1.0, valid=True, observation_status="GRACE", gap=0.2
    )
    decision = fusion.update(1.0, None, grace, True, eye_evidence_trustworthy=True)
    assert decision.driver_state == DriverState.NORMAL
    pending = _empty_events(
        1.2, valid=False, observation_status="LOST", gap=0.4
    )
    decision = fusion.update(1.2, None, pending, True, eye_evidence_trustworthy=False)
    assert decision.driver_state == DriverState.NORMAL
    lost = _empty_events(
        1.4, valid=False, observation_status="LOST", gap=0.6
    )
    decision = fusion.update(1.4, None, lost, True, eye_evidence_trustworthy=False)
    assert decision.driver_state == DriverState.UNKNOWN


def test_critical_is_held_across_grace_and_lost_signal():
    fusion = FusionStateMachine(transition_cooldown_sec=0.0)
    closed = _empty_events(0.0, eye_state="CLOSED", closure=1.5)
    assert fusion.update(0.0, None, closed, True).driver_state == DriverState.CRITICAL
    grace = _empty_events(
        0.2, eye_state="CLOSED", closure=1.7,
        observation_status="GRACE", gap=0.2,
    )
    decision = fusion.update(0.2, None, grace, True, eye_evidence_trustworthy=True)
    assert decision.driver_state == DriverState.CRITICAL
    lost = _empty_events(
        0.8, eye_state="UNKNOWN", valid=False,
        observation_status="LOST", gap=0.8,
    )
    decision = fusion.update(0.8, None, lost, True, eye_evidence_trustworthy=False)
    assert decision.driver_state == DriverState.CRITICAL
    assert decision.reason_codes == ["CRITICAL_EYE_SIGNAL_LOST"]


def test_post_critical_high_perclos_becomes_recovery_warning():
    fusion = FusionStateMachine(recovery_persistence_sec=3.0, transition_cooldown_sec=0.0)
    closed = _empty_events(0.0, eye_state="CLOSED", closure=1.5)
    assert fusion.update(0.0, None, closed, True).driver_state == DriverState.CRITICAL
    for timestamp in (1.0, 2.0, 4.1):
        decision = fusion.update(
            timestamp,
            None,
            _empty_events(timestamp, perclos=0.4, observed=30.0, coverage=1.0),
            True,
        )
    assert decision.driver_state == DriverState.FATIGUE_WARNING
    assert decision.recovery_status == "POST_CRITICAL_FATIGUE"
    assert decision.state_entry_reason[0] == "CRITICAL_RECOVERY_COMPLETE"


def test_consumed_decision_perclos_cannot_restore_drowsy_after_recovery():
    fusion = FusionStateMachine(recovery_persistence_sec=3.0, transition_cooldown_sec=0.0)
    closed = _empty_events(0.0, eye_state="CLOSED", closure=1.5)
    assert fusion.update(0.0, None, closed, True).driver_state == DriverState.CRITICAL
    for timestamp in (1.0, 2.0, 4.1):
        metrics = _empty_events(timestamp, perclos=0.4, observed=30.0, coverage=1.0)
        metrics.update({
            "raw_perclos_30s": 0.4,
            "decision_perclos_30s": None,
            "decision_valid_observation_sec_30s": 0.0,
            "decision_perclos_coverage_30s": 0.0,
        })
        decision = fusion.update(timestamp, None, metrics, True)
    assert decision.driver_state == DriverState.NORMAL


def test_weak_bilateral_eye_reduction_cannot_emit_strong_closure():
    engine = EventEngine(prolonged_closure_sec=1.5)
    engine.update(_face(0.0, 1.0), 0.1, 0.0)
    emitted = []
    for index in range(1, 31):
        result = engine.update(_face(index / 15.0, 0.4), 0.1, 0.0)
        emitted.extend(result.evidence)
    assert result.eye_state == "CLOSED"
    assert not result.strong_closure_valid
    assert not any(item.kind == "PROLONGED_EYE_CLOSURE" for item in emitted)


def test_open_acknowledgement_consumes_decision_perclos_only():
    engine = EventEngine(prolonged_closure_sec=1.5, perclos_min_observation_sec=0.1)
    engine.update(_face(0.0, 1.0), 0.1, 0.0)
    for index in range(1, 31):
        result = engine.update(_face(index / 15.0, 0.2), 0.1, 0.0)
    assert result.strong_closure_valid
    for index in range(31, 41):
        result = engine.update(_face(index / 15.0, 1.0), 0.1, 0.0)
    assert result.eye_open_trust_sec >= 0.5
    assert result.raw_perclos_30s is not None and result.raw_perclos_30s > 0
    assert result.decision_perclos_30s is None


def test_yawn_and_repeated_yawns_are_violation_alarms_not_fatigue_state():
    fusion = FusionStateMachine(normal_persistence_sec=0.0, transition_cooldown_sec=0.0)
    fusion.add_evidence([EvidenceEvent("YAWNING", 0.0, 1.0, 8.0, "test", 2.0)])
    decision = fusion.update(
        0.0,
        None,
        _empty_events(0.0, yaw_count=2),
        True,
    )
    assert decision.driver_state == DriverState.NORMAL
    assert decision.alarm_level.value == "WARNING"
    assert set(decision.violations) == {"YAWNING", "REPEATED_YAWNS"}


def test_recent_yawn_cannot_turn_partial_eye_support_into_drowsy():
    fusion = FusionStateMachine(
        drowsy_persistence_sec=0.0,
        warning_persistence_sec=0.0,
        transition_cooldown_sec=0.0,
    )
    fusion.add_evidence([EvidenceEvent("YAWNING", 0.0, 1.0, 8.0, "test", 3.0)])
    prediction = ModelPrediction("w0", 1.0, 0.9, 0.9, 0.5, True, "m")
    decision = fusion.update(
        1.0,
        prediction,
        _empty_events(1.0, sustained_partial=True),
        True,
    )
    assert decision.driver_state == DriverState.FATIGUE_WARNING
    assert decision.driver_state != DriverState.DROWSY


def test_untrusted_long_blink_cannot_support_model_drowsy():
    fusion = FusionStateMachine(
        drowsy_persistence_sec=0.0,
        normal_persistence_sec=0.0,
        transition_cooldown_sec=0.0,
    )
    fusion.add_evidence([
        EvidenceEvent(
            "LONG_BLINK", 0.0, 0.35, 2.0, "test", 1.0,
            details={"physiology_trustworthy": False},
        )
    ])
    prediction = ModelPrediction("w0", 0.0, 0.9, 0.9, 0.5, True, "m")
    decision = fusion.update(0.0, prediction, _empty_events(0.0), True)
    assert decision.driver_state == DriverState.NORMAL
    assert decision.model_risk_pending
