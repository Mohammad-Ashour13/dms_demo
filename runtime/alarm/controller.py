from __future__ import annotations

import queue
import threading
import time
import uuid
from collections import deque

from dms_final_system.shared.contracts import AlarmCommand, AlarmStatus, DriverState
from dms_final_system.runtime.alarm.outputs import AlarmOutput, NullAlarmOutput


class AlarmController:
    """Edge- and timer-driven audible policy, deliberately separate from Fusion."""

    PRIORITY = {"DROWSY": 20, "DROWSY_ESCALATED": 10, "CRITICAL": 0, "STARTUP": 30}

    def __init__(self, config, telemetry, output: AlarmOutput | None = None, *, audible=True):
        self.config = config
        self.telemetry = telemetry
        self.audible = bool(config.enabled and config.mode == "LOCAL" and audible)
        self.output = output or NullAlarmOutput()
        self.queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=config.queue_size)
        self.stop_event = threading.Event()
        self.play_stop = threading.Event()
        self.lock = threading.RLock()
        self.sequence = 0
        self.status = AlarmStatus(backend_health="DISABLED")
        self.active_state = DriverState.UNKNOWN
        self.episode_id: str | None = None
        self.episode_state: DriverState | None = None
        self.episode_started_at: float | None = None
        self.episode_ended_at: float | None = None
        self.next_reminder: float | None = None
        self.acknowledged = False
        self.ack_silence_until: float | None = None
        self.drowsy_onsets: deque[float] = deque()
        self.command_history: list[AlarmCommand] = []
        self.worker = threading.Thread(target=self._run, name="alarm-output", daemon=True)
        ok, detail = self.output.probe()
        if self.audible and not ok and config.required:
            raise RuntimeError(f"Required alarm audio output is unavailable: {detail}")
        self.audible = bool(self.audible and ok)
        self.status.backend_health = "READY" if self.audible else ("LOG_ONLY" if config.enabled else "DISABLED")
        self.status.last_error = "" if ok else detail
        self.telemetry.emit(
            "Alarm", "output_probe",
            {"audible": self.audible, "mode": config.mode, "backend": detail, "required": config.required},
            level="INFO" if ok or not config.required else "ERROR",
        )
        if self.audible and config.startup_self_test:
            command = AlarmCommand(
                "alarm-startup-self-test", "alarm-startup-self-test",
                time.monotonic(), DriverState.NORMAL, "STARTUP", "ONSET", 1,
                ["AUDIO_STARTUP_SELF_TEST"], None,
            )
            try:
                self.telemetry.emit("Alarm", "audio_self_test_started", command.to_dict())
                self.output.play(command, threading.Event())
                self.telemetry.emit("Alarm", "audio_self_test_passed", command.to_dict())
            except Exception as exc:
                self.status.backend_health = "FAILED"
                self.status.last_error = repr(exc)
                self.telemetry.emit(
                    "Alarm", "audio_self_test_failed",
                    {**command.to_dict(), "error": repr(exc)}, level="ERROR",
                )
                self.audible = False
                if config.required:
                    self.output.close()
                    raise RuntimeError(f"Required alarm audio self-test failed: {exc}") from exc
        self.worker.start()

    def _emit(self, event: str, payload: dict, command: AlarmCommand | None = None, level="INFO") -> None:
        self.telemetry.emit(
            "Alarm", event, payload,
            monotonic_sec=command.monotonic_sec if command else None,
            incident_id=command.incident_id if command else None,
            level=level,
        )

    def _new_episode(self, now: float, state: DriverState) -> None:
        self.episode_id = f"alarm-{uuid.uuid4().hex[:10]}"
        self.episode_state = state
        self.episode_started_at = now
        self.episode_ended_at = None
        self.acknowledged = False
        self.ack_silence_until = None
        self.status.command_count = 0
        self.status.reminder_count = 0
        self.status.escalation_count = 0
        if state == DriverState.DROWSY:
            self.drowsy_onsets.append(now)
            while self.drowsy_onsets and self.drowsy_onsets[0] < now - self.config.escalation_window_sec:
                self.drowsy_onsets.popleft()
        self._emit("alarm_episode_started", {"episode_id": self.episode_id, "driver_state": state.value})

    def _issue(self, now, state, pattern, trigger, reasons, incident_id, episode_id=None) -> AlarmCommand:
        if episode_id is not None:
            self.episode_id = episode_id
        if self.episode_id is None:
            self._new_episode(now, state)
        command = AlarmCommand(
            alarm_id=f"alarm-command-{uuid.uuid4().hex[:10]}",
            episode_id=self.episode_id or "startup",
            monotonic_sec=float(now),
            driver_state=state,
            pattern=pattern,
            trigger=trigger,
            recurrence_count=len(self.drowsy_onsets) if state == DriverState.DROWSY else 1,
            reason_codes=list(reasons),
            incident_id=incident_id,
        )
        self._emit("alarm_command_issued", command.to_dict(), command)
        self.command_history.append(command)
        self.status.command_count += 1
        if trigger == "REMINDER":
            self.status.reminder_count += 1
        elif trigger == "ESCALATION":
            self.status.escalation_count += 1
        if not self.audible:
            self._emit("alarm_suppressed", {**command.to_dict(), "reason": "LOG_ONLY_OR_SOURCE_BLOCKED"}, command)
            return command
        if pattern == "CRITICAL":
            self._cancel_pending("PREEMPTED_BY_CRITICAL")
            self.play_stop.set()
            self.output.stop()
        self.sequence += 1
        try:
            self.queue.put_nowait((self.PRIORITY[pattern], self.sequence, command))
        except queue.Full:
            self._emit("alarm_suppressed", {**command.to_dict(), "reason": "OUTPUT_QUEUE_FULL"}, command, "WARNING")
        return command

    def _cancel_pending(self, reason: str) -> None:
        while True:
            try:
                _, _, pending = self.queue.get_nowait()
            except queue.Empty:
                break
            self._emit(
                "alarm_suppressed",
                {**pending.to_dict(), "reason": reason},
                pending,
            )
            self.queue.task_done()

    def _is_escalated_drowsy(self, now: float) -> bool:
        recurrent = len(self.drowsy_onsets) >= self.config.escalation_episode_count
        continuous = self.episode_started_at is not None and now - self.episode_started_at >= self.config.escalation_continuous_sec
        return bool(recurrent or continuous)

    def update(self, now, decision, event_snapshot, *, incident_id=None) -> AlarmStatus:
        if not self.config.enabled or self.config.mode == "OFF":
            return self.snapshot()
        state = decision.driver_state
        open_trust = float(getattr(event_snapshot, "eye_open_trust_sec", 0.0) or 0.0)
        reliable_open = bool(
            event_snapshot is not None
            and getattr(event_snapshot, "eye_state", "UNKNOWN") == "OPEN"
            and getattr(event_snapshot, "eye_evidence_trustworthy", False)
            and open_trust >= self.config.acknowledge_open_sec
        )

        if state not in {DriverState.DROWSY, DriverState.CRITICAL}:
            if self.active_state in {DriverState.DROWSY, DriverState.CRITICAL}:
                self.episode_ended_at = now
                self.play_stop.set()
                self.output.stop()
                self._cancel_pending("EPISODE_ENDED")
                self._emit("alarm_episode_ended", {"episode_id": self.episode_id, "driver_state": self.active_state.value})
            self.active_state = state
            self.next_reminder = None
            self.acknowledged = False
            self.status.active_pattern = "NONE"
            self.status.acknowledged = False
            self.status.next_reminder_sec = None
            return self.snapshot()

        entering = state != self.active_state
        if state == DriverState.CRITICAL:
            if entering:
                self._new_episode(now, state)
                self._issue(now, state, "CRITICAL", "ONSET", decision.reason_codes, incident_id)
                self.next_reminder = now + self.config.critical_reminder_sec
            if reliable_open and not self.acknowledged:
                self.acknowledged = True
                self.play_stop.set()
                self.output.stop()
                self._cancel_pending("TRUSTED_EYE_OPEN")
                self._emit("alarm_acknowledged", {"episode_id": self.episode_id, "reason": "TRUSTED_EYE_OPEN", "open_sec": open_trust})
            elif self.acknowledged and not reliable_open and getattr(event_snapshot, "eye_state", "UNKNOWN") == "CLOSED":
                self.acknowledged = False
                self._issue(now, state, "CRITICAL", "RESUME", decision.reason_codes, incident_id)
                self.next_reminder = now + self.config.critical_reminder_sec
            elif not self.acknowledged and not entering and self.next_reminder is not None and now >= self.next_reminder:
                self._issue(now, state, "CRITICAL", "REMINDER", decision.reason_codes, incident_id)
                age = now - (
                    self.episode_started_at
                    if self.episode_started_at is not None
                    else now
                )
                interval = (
                    self.config.critical_escalated_reminder_sec
                    if age >= self.config.critical_escalate_after_sec
                    else self.config.critical_reminder_sec
                )
                self.next_reminder = now + interval
            self.status.active_pattern = "CRITICAL"
        else:
            same_recent_episode = bool(
                entering
                and self.active_state != DriverState.CRITICAL
                and self.episode_ended_at is not None
                and now - self.episode_ended_at < self.config.drowsy_rearm_sec
                and self.episode_id is not None
                and self.episode_state == DriverState.DROWSY
            )
            if entering and not same_recent_episode:
                self._new_episode(now, state)
                pattern = "DROWSY_ESCALATED" if self._is_escalated_drowsy(now) else "DROWSY"
                self._issue(now, state, pattern, "ONSET", decision.reason_codes, incident_id)
                interval = self.config.escalation_reminder_sec if pattern == "DROWSY_ESCALATED" else self.config.drowsy_reminder_sec
                self.next_reminder = now + interval
            elif entering and same_recent_episode:
                self._emit("alarm_suppressed", {"episode_id": self.episode_id, "reason": "DROWSY_REARM_GUARD"})
                self.acknowledged = False
                self.next_reminder = now + self.config.drowsy_reminder_sec
            if reliable_open and not self.acknowledged:
                self.acknowledged = True
                self.ack_silence_until = now + self.config.drowsy_ack_silence_sec
                self.play_stop.set()
                self.output.stop()
                self._cancel_pending("TRUSTED_EYE_OPEN")
                self._emit("alarm_acknowledged", {"episode_id": self.episode_id, "reason": "TRUSTED_EYE_OPEN", "open_sec": open_trust})
            if self.acknowledged and self.ack_silence_until is not None and now >= self.ack_silence_until:
                self.acknowledged = False
                self.next_reminder = now
            if not self.acknowledged and self.next_reminder is not None and now >= self.next_reminder:
                escalated = self._is_escalated_drowsy(now)
                pattern = "DROWSY_ESCALATED" if escalated else "DROWSY"
                trigger = "ESCALATION" if escalated else "REMINDER"
                self._issue(now, state, pattern, trigger, decision.reason_codes, incident_id)
                self.next_reminder = now + (
                    self.config.escalation_reminder_sec if escalated else self.config.drowsy_reminder_sec
                )
            self.status.active_pattern = "DROWSY_ESCALATED" if self._is_escalated_drowsy(now) else "DROWSY"

        self.active_state = state
        self.status.acknowledged = self.acknowledged
        self.status.next_reminder_sec = self.next_reminder
        self.status.episode_id = self.episode_id
        return self.snapshot()

    def snapshot(self) -> AlarmStatus:
        with self.lock:
            return AlarmStatus(**self.status.to_dict())

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                _, _, command = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self.play_stop.clear()
            started = time.perf_counter()
            with self.lock:
                self.status.audio_playing = True
            self._emit("audio_play_started", {**command.to_dict(), "command_latency_ms": max(0.0, (time.monotonic() - command.monotonic_sec) * 1000)}, command)
            try:
                outcome = self.output.play(command, self.play_stop)
                self._emit(f"audio_play_{outcome.lower()}", {**command.to_dict(), "worker_elapsed_ms": (time.perf_counter() - started) * 1000}, command)
            except Exception as exc:
                with self.lock:
                    self.status.backend_health = "FAILED"
                    self.status.last_error = repr(exc)
                self._emit("alarm_output_failed", {**command.to_dict(), "error": repr(exc)}, command, "ERROR")
            finally:
                with self.lock:
                    self.status.audio_playing = False
                self.queue.task_done()

    def close(self) -> None:
        self.stop_event.set()
        self.play_stop.set()
        self.output.stop()
        self._cancel_pending("CONTROLLER_SHUTDOWN")
        self.worker.join(timeout=2.0)
        self.output.close()
