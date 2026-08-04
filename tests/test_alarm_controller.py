from __future__ import annotations

import time
from types import SimpleNamespace

from dms_final_system.runtime.alarm import AlarmController, AlarmOutput
from dms_final_system.runtime.config import AlarmConfig
from dms_final_system.shared.contracts import AlarmLevel, DriverState, FusionDecision


class FakeTelemetry:
    def __init__(self):
        self.records = []

    def emit(self, stage, event, payload=None, **kwargs):
        self.records.append((stage, event, payload or {}, kwargs))


class FakeAudio(AlarmOutput):
    def __init__(self, available=True, fail=False):
        self.available = available
        self.fail = fail
        self.played = []
        self.stop_calls = 0

    def probe(self):
        return self.available, "fake"

    def play(self, command, stop_event):
        self.played.append(command)
        if self.fail:
            raise RuntimeError("synthetic audio failure")
        return "PREEMPTED" if stop_event.is_set() else "COMPLETED"

    def stop(self):
        self.stop_calls += 1


def _decision(t, state):
    return FusionDecision(
        t, DriverState.NORMAL, state, [],
        AlarmLevel.CRITICAL if state == DriverState.CRITICAL else AlarmLevel.WARNING,
        [f"TEST_{state.value}"], None, True,
    )


def _eyes(*, state="CLOSED", open_sec=0.0, trustworthy=True):
    return SimpleNamespace(
        eye_state=state,
        eye_open_trust_sec=open_sec,
        eye_evidence_trustworthy=trustworthy,
    )


def _controller(**overrides):
    values = dict(enabled=True, mode="LOG_ONLY", startup_self_test=False)
    values.update(overrides)
    telemetry = FakeTelemetry()
    output = FakeAudio()
    return AlarmController(AlarmConfig(**values), telemetry, output), telemetry, output


def test_many_critical_frames_create_one_onset_not_per_frame():
    controller, _, _ = _controller()
    try:
        for index in range(100):
            controller.update(index * 0.02, _decision(index * 0.02, DriverState.CRITICAL), _eyes())
        assert [(item.pattern, item.trigger) for item in controller.command_history] == [("CRITICAL", "ONSET")]
    finally:
        controller.close()


def test_drowsy_25_seconds_has_onset_and_two_reminders():
    controller, _, _ = _controller()
    try:
        for timestamp in range(26):
            controller.update(float(timestamp), _decision(timestamp, DriverState.DROWSY), _eyes())
        assert [(item.pattern, item.trigger) for item in controller.command_history] == [
            ("DROWSY", "ONSET"),
            ("DROWSY", "REMINDER"),
            ("DROWSY", "REMINDER"),
        ]
    finally:
        controller.close()


def test_critical_reminds_every_four_seconds_then_increases_cadence():
    controller, _, _ = _controller()
    try:
        for timestamp in (0.0, 4.0, 8.0, 12.0, 16.0, 18.5):
            controller.update(timestamp, _decision(timestamp, DriverState.CRITICAL), _eyes())
        assert [item.trigger for item in controller.command_history] == [
            "ONSET", "REMINDER", "REMINDER", "REMINDER", "REMINDER", "REMINDER"
        ]
    finally:
        controller.close()


def test_trusted_open_acknowledges_critical_and_reclosure_resumes():
    controller, telemetry, _ = _controller()
    try:
        controller.update(0.0, _decision(0.0, DriverState.CRITICAL), _eyes())
        status = controller.update(
            0.5, _decision(0.5, DriverState.CRITICAL),
            _eyes(state="OPEN", open_sec=0.5),
        )
        assert status.acknowledged
        controller.update(5.0, _decision(5.0, DriverState.CRITICAL), _eyes(state="OPEN", open_sec=5.0))
        controller.update(5.1, _decision(5.1, DriverState.CRITICAL), _eyes())
        assert [item.trigger for item in controller.command_history] == ["ONSET", "RESUME"]
        assert any(event == "alarm_acknowledged" for _, event, _, _ in telemetry.records)
    finally:
        controller.close()


def test_warning_unknown_and_yawn_violation_never_issue_audio_command():
    controller, _, _ = _controller()
    try:
        for index, state in enumerate((DriverState.FATIGUE_WARNING, DriverState.UNKNOWN, DriverState.NORMAL)):
            decision = _decision(float(index), state)
            decision.violations = ["YAWNING", "REPEATED_YAWNS"]
            controller.update(float(index), decision, _eyes(state="OPEN", open_sec=1.0))
        assert controller.command_history == []
    finally:
        controller.close()


def test_replay_source_can_log_commands_without_playing_audio():
    telemetry, output = FakeTelemetry(), FakeAudio()
    config = AlarmConfig(enabled=True, mode="LOCAL", startup_self_test=False)
    controller = AlarmController(config, telemetry, output, audible=False)
    try:
        controller.update(0.0, _decision(0.0, DriverState.DROWSY), _eyes())
        time.sleep(0.02)
        assert len(controller.command_history) == 1
        assert output.played == []
        assert any(event == "alarm_suppressed" for _, event, _, _ in telemetry.records)
    finally:
        controller.close()


def test_critical_preempts_lower_priority_output():
    telemetry, output = FakeTelemetry(), FakeAudio()
    config = AlarmConfig(enabled=True, mode="LOCAL", startup_self_test=False)
    controller = AlarmController(config, telemetry, output, audible=True)
    try:
        controller.update(0.0, _decision(0.0, DriverState.DROWSY), _eyes())
        controller.update(0.1, _decision(0.1, DriverState.CRITICAL), _eyes())
        controller.queue.join()
        assert output.stop_calls >= 1
        assert [item.pattern for item in controller.command_history] == ["DROWSY", "CRITICAL"]
    finally:
        controller.close()


def test_audio_failure_is_exposed_in_status_and_telemetry():
    telemetry, output = FakeTelemetry(), FakeAudio(fail=True)
    config = AlarmConfig(enabled=True, mode="LOCAL", startup_self_test=False)
    controller = AlarmController(config, telemetry, output, audible=True)
    try:
        controller.update(0.0, _decision(0.0, DriverState.CRITICAL), _eyes())
        controller.queue.join()
        assert controller.snapshot().backend_health == "FAILED"
        assert any(event == "alarm_output_failed" for _, event, _, _ in telemetry.records)
    finally:
        controller.close()


def test_short_drowsy_gap_suppresses_second_onset_but_keeps_reminders():
    controller, _, _ = _controller()
    try:
        controller.update(0.0, _decision(0.0, DriverState.DROWSY), _eyes())
        controller.update(1.0, _decision(1.0, DriverState.NORMAL), _eyes(state="OPEN", open_sec=1.0))
        controller.update(3.0, _decision(3.0, DriverState.DROWSY), _eyes())
        controller.update(13.0, _decision(13.0, DriverState.DROWSY), _eyes())
        assert [item.trigger for item in controller.command_history] == ["ONSET", "REMINDER"]
    finally:
        controller.close()
