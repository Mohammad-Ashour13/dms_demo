from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from dms_final_system.shared.contracts import EvidenceEvent, FaceSignal


@dataclass(slots=True)
class EventEngineSnapshot:
    monotonic_sec: float
    eye_state: str
    current_closure_sec: float
    blink_count_60s: int
    long_blink_count_60s: int
    perclos_30s: float | None
    perclos_60s: float | None
    mouth_state: str
    current_mouth_open_sec: float
    yawn_count_60s: int
    evidence: list[EvidenceEvent]
    eye_signal_valid: bool = False
    eye_signal_quality: float = 0.0
    event_relative_ear: float = 0.0
    event_left_relative_ear: float = 0.0
    event_right_relative_ear: float = 0.0
    valid_observation_sec: float = 0.0
    valid_observation_sec_30s: float = 0.0
    valid_observation_sec_60s: float = 0.0
    perclos_coverage_30s: float = 0.0
    perclos_coverage_60s: float = 0.0
    raw_eye_signal_valid: bool = False
    eye_observation_status: str = "LOST"
    eye_signal_gap_sec: float = 0.0
    binocular_consistent: bool = True
    eye_evidence_trustworthy: bool = False
    closure_measurement_valid: bool = False
    eye_open_trust_sec: float = 0.0
    strong_closure_valid: bool = False
    downward_gaze_confidence: float = 0.0
    downward_gaze_active: bool = False
    sustained_partial_closure: bool = False
    raw_perclos_30s: float | None = None
    decision_perclos_30s: float | None = None
    decision_valid_observation_sec_30s: float = 0.0
    decision_perclos_coverage_30s: float = 0.0
    eye_resolution_valid: bool = True
    driver_distance_status: str = "UNKNOWN"
    face_width_ratio: float = 0.0
    interocular_distance_px: float = 0.0
    minimum_eye_width_px: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class EventEngine:
    """Timestamp-based physiological events, independent from model EAR features."""

    def __init__(
        self,
        eye_close_relative_ear=0.60,
        eye_reopen_relative_ear=0.72,
        blink_min_sec=0.05,
        blink_max_sec=0.80,
        prolonged_closure_sec=1.50,
        max_eye_signal_gap_sec=0.30,
        binocular_close_sync_sec=0.15,
        max_eye_gaze_offset=0.20,
        strong_closure_median_ear=0.30,
        strong_closure_sample_ear=0.40,
        strong_closure_fraction=0.80,
        strong_closure_coverage=0.80,
        strong_closure_downward_confidence=0.25,
        strong_closure_max_downward_fraction=0.50,
        downward_gaze_delta=0.18,
        partial_closure_ear=0.70,
        partial_closure_window_sec=2.0,
        partial_closure_fraction=0.70,
        eye_acknowledge_open_sec=0.50,
        perclos_min_observation_sec=10.0,
        mouth_open_relative_ratio=1.55,
        yawn_min_sec=1.20,
        yawn_max_sec=8.0,
        head_nod_pitch_delta_deg=18.0,
        head_nod_min_sec=0.35,
        **legacy,
    ):
        # Accept old config objects during rolling upgrades, but never use the old
        # single threshold as the reopen threshold.
        if "eye_closed_relative_ear" in legacy:
            eye_close_relative_ear = min(float(legacy["eye_closed_relative_ear"]), 0.60)
        if not 0 < eye_close_relative_ear < eye_reopen_relative_ear:
            raise ValueError("eye thresholds must satisfy 0 < close < reopen")
        self.eye_close_threshold = float(eye_close_relative_ear)
        self.eye_reopen_threshold = float(eye_reopen_relative_ear)
        self.blink_min = float(blink_min_sec)
        self.blink_max = float(blink_max_sec)
        self.prolonged_sec = float(prolonged_closure_sec)
        self.max_eye_gap = float(max_eye_signal_gap_sec)
        self.binocular_close_sync = float(binocular_close_sync_sec)
        self.max_eye_gaze_offset = float(max_eye_gaze_offset)
        self.strong_median_ear = float(strong_closure_median_ear)
        self.strong_sample_ear = float(strong_closure_sample_ear)
        self.strong_fraction = float(strong_closure_fraction)
        self.strong_coverage = float(strong_closure_coverage)
        self.strong_downward_confidence = float(strong_closure_downward_confidence)
        self.strong_max_downward_fraction = float(strong_closure_max_downward_fraction)
        self.downward_gaze_delta = float(downward_gaze_delta)
        self.partial_closure_ear = float(partial_closure_ear)
        self.partial_window = float(partial_closure_window_sec)
        self.partial_fraction = float(partial_closure_fraction)
        self.eye_acknowledge_open = float(eye_acknowledge_open_sec)
        self.perclos_min_observation = float(perclos_min_observation_sec)
        self.mouth_ratio = float(mouth_open_relative_ratio)
        self.yawn_min = float(yawn_min_sec)
        self.yawn_max = float(yawn_max_sec)
        self.nod_delta = float(head_nod_pitch_delta_deg)
        self.nod_min = float(head_nod_min_sec)
        self.reset()

    def reset(self) -> None:
        self.eye_state = "UNKNOWN"
        self.left_eye_state = "UNKNOWN"
        self.right_eye_state = "UNKNOWN"
        self.left_eye_closed_since: float | None = None
        self.right_eye_closed_since: float | None = None
        self.had_reliable_open = False
        self.eye_closed_since: float | None = None
        self.eye_open_since: float | None = None
        self.last_valid_eye_time: float | None = None
        self.invalid_eye_since: float | None = None
        self.mouth_open_since: float | None = None
        self.nod_since: float | None = None
        self.prolonged_emitted = False
        self.risk_episode_active = False
        self.nod_emitted = False
        self.blinks: deque[tuple[float, bool]] = deque()
        self.yawns: deque[float] = deque()
        self.eye_intervals: deque[tuple[float, float, int]] = deque()
        self.decision_eye_intervals: deque[tuple[float, float, int]] = deque()
        self.last_decision_closed = False
        self.decision_epoch_sec: float | None = None
        self.strong_samples: deque[tuple[float, float, bool, float]] = deque()
        self.downward_gaze_since: float | None = None
        self.downward_gaze_emitted = False
        self.frame_intervals: deque[float] = deque(maxlen=30)

    def _prune(self, now: float) -> None:
        while self.blinks and self.blinks[0][0] < now - 60:
            self.blinks.popleft()
        while self.yawns and self.yawns[0] < now - 60:
            self.yawns.popleft()
        while self.eye_intervals and self.eye_intervals[0][1] <= now - 60:
            self.eye_intervals.popleft()
        while self.decision_eye_intervals and self.decision_eye_intervals[0][1] <= now - 60:
            self.decision_eye_intervals.popleft()
        while self.strong_samples and self.strong_samples[0][0] < now - max(self.partial_window, self.prolonged_sec) - 0.5:
            self.strong_samples.popleft()

    def _perclos(self, now: float, period: float, intervals=None) -> tuple[float | None, float, float]:
        start = now - period
        observed = closed = 0.0
        source = self.eye_intervals if intervals is None else intervals
        for interval_start, interval_end, is_closed in source:
            overlap = max(0.0, min(now, interval_end) - max(start, interval_start))
            observed += overlap
            closed += overlap * is_closed
        value = closed / observed if observed >= self.perclos_min_observation else None
        return value, observed, min(1.0, observed / period)

    def _invalidate_eye(self, now: float) -> None:
        if self.invalid_eye_since is None:
            self.invalid_eye_since = now
        self.last_valid_eye_time = None
        if now - self.invalid_eye_since > self.max_eye_gap:
            self.eye_state = "UNKNOWN"
            self.left_eye_state = "UNKNOWN"
            self.right_eye_state = "UNKNOWN"
            self.left_eye_closed_since = None
            self.right_eye_closed_since = None
            self.had_reliable_open = False
            self.eye_closed_since = None
            self.eye_open_since = None
            self.prolonged_emitted = False
            self.last_decision_closed = False

    def _advance_eye(
        self,
        state: str,
        closed_since: float | None,
        score: float,
        now: float,
        allow_close: bool,
    ) -> tuple[str, float | None]:
        if state == "UNKNOWN":
            if score >= self.eye_reopen_threshold:
                return "OPEN", None
            if allow_close and score <= self.eye_close_threshold:
                return "CLOSED", now
            return state, closed_since
        if state == "OPEN" and allow_close and score <= self.eye_close_threshold:
            return "CLOSED", now
        if state == "CLOSED" and score >= self.eye_reopen_threshold:
            return "OPEN", None
        return state, closed_since

    def _update_eye(self, signal: FaceSignal) -> list[EvidenceEvent]:
        now = signal.monotonic_sec
        evidence: list[EvidenceEvent] = []
        if not signal.eye_signal_valid or signal.event_relative_ear <= 0:
            self._invalidate_eye(now)
            signal.left_eye_closed = signal.right_eye_closed = signal.eyes_closed = 0
            return evidence

        if self.invalid_eye_since is not None:
            gap = now - self.invalid_eye_since
            if gap > self.max_eye_gap:
                self.eye_state = "UNKNOWN"
                self.left_eye_state = "UNKNOWN"
                self.right_eye_state = "UNKNOWN"
                self.left_eye_closed_since = None
                self.right_eye_closed_since = None
                self.had_reliable_open = False
                self.eye_closed_since = None
                self.eye_open_since = None
                self.prolonged_emitted = False
            self.invalid_eye_since = None

        if self.last_valid_eye_time is not None:
            dt = now - self.last_valid_eye_time
            if 0 < dt <= self.max_eye_gap:
                self.frame_intervals.append(dt)
                if self.eye_state in {"OPEN", "CLOSED"}:
                    self.eye_intervals.append(
                        (self.last_valid_eye_time, now, int(self.eye_state == "CLOSED"))
                    )
                self.decision_eye_intervals.append(
                    (self.last_valid_eye_time, now, int(self.last_decision_closed))
                )
            elif dt > self.max_eye_gap:
                self.eye_state = "UNKNOWN"
                self.had_reliable_open = False
                self.eye_closed_since = None
                self.eye_open_since = None
                self.prolonged_emitted = False

        signal.downward_gaze_confidence = float(
            np.clip(
                (signal.relative_gaze_y - self.downward_gaze_delta)
                / max(self.downward_gaze_delta, 1e-6),
                0.0,
                1.0,
            )
            if signal.gaze_signal_valid else 0.0
        )
        max_relative_ear = max(
            float(signal.event_left_relative_ear),
            float(signal.event_right_relative_ear),
        )
        strong_bilateral_candidate = bool(
            max_relative_ear <= self.strong_sample_ear
            and signal.binocular_consistent
        )
        downward_gaze_active = bool(
            signal.downward_gaze_confidence >= 0.5
            and max_relative_ear > self.strong_sample_ear
        )
        # Iris position is not geometrically trustworthy once the eyelids are
        # strongly closed. Requiring centred iris coordinates in that state
        # suppressed real 4-9 second closures in the live regression video.
        # For clearly bilateral closure use eyelid geometry; retain the gaze
        # guard for open/partially-open eyes so looking down is not a closure.
        closure_measurement_valid = bool(
            strong_bilateral_candidate
            or (
                abs(signal.gaze_x) <= self.max_eye_gaze_offset
                and abs(signal.gaze_y) <= self.max_eye_gaze_offset
                and not downward_gaze_active
            )
        )
        allow_close = bool(closure_measurement_valid or self.eye_state == "CLOSED")
        self.left_eye_state, self.left_eye_closed_since = self._advance_eye(
            self.left_eye_state,
            self.left_eye_closed_since,
            signal.event_left_relative_ear,
            now,
            allow_close,
        )
        self.right_eye_state, self.right_eye_closed_since = self._advance_eye(
            self.right_eye_state,
            self.right_eye_closed_since,
            signal.event_right_relative_ear,
            now,
            allow_close,
        )
        both_closed = self.left_eye_state == self.right_eye_state == "CLOSED"
        both_open = self.left_eye_state == self.right_eye_state == "OPEN"
        self.last_decision_closed = bool(
            closure_measurement_valid and max_relative_ear <= self.strong_sample_ear
        )
        effective_downward_confidence = (
            0.0 if strong_bilateral_candidate else signal.downward_gaze_confidence
        )
        self.strong_samples.append(
            (now, max_relative_ear, closure_measurement_valid, effective_downward_confidence)
        )

        if both_closed and self.eye_state != "CLOSED" and closure_measurement_valid:
            starts = (self.left_eye_closed_since, self.right_eye_closed_since)
            if all(value is not None for value in starts):
                left_start, right_start = float(starts[0]), float(starts[1])
                if abs(left_start - right_start) <= self.binocular_close_sync:
                    self.eye_state = "CLOSED"
                    # Whole-driver closure starts when the second eye closes.
                    self.eye_closed_since = max(left_start, right_start)
                    self.prolonged_emitted = False
                    self.eye_open_since = None
        elif both_open and self.eye_state == "UNKNOWN":
            self.eye_state = "OPEN"
            self.had_reliable_open = True
            self.eye_open_since = now
        elif both_open and self.eye_state == "CLOSED":
            self.eye_state = "OPEN"
            duration = now - (self.eye_closed_since if self.eye_closed_since is not None else now)
            if self.frame_intervals:
                duration = max(duration, float(np.median(self.frame_intervals)))
            if self.had_reliable_open and duration >= self.blink_min:
                long_blink = duration > self.blink_max
                self.blinks.append((now, long_blink))
                physiology_trustworthy, closure_details = self._closure_quality(now, duration)
                evidence.append(
                    EvidenceEvent(
                        "LONG_BLINK" if long_blink else "BLINK",
                        now,
                        1.0 if physiology_trustworthy or not long_blink else 0.35,
                        2.0,
                        "eye-events-v2.4",
                        duration,
                        details={
                            **closure_details,
                            "physiology_trustworthy": bool(
                                physiology_trustworthy or not long_blink
                            ),
                        },
                    )
                )
            self.had_reliable_open = True
            self.eye_closed_since = None
            self.prolonged_emitted = False
            self.eye_open_since = now

        if both_open and self.eye_state == "OPEN":
            if self.eye_open_since is None:
                self.eye_open_since = now
        elif self.eye_state != "OPEN":
            self.eye_open_since = None

        # A one-eye-only closure is deliberately not promoted to the combined
        # CLOSED state. The last reliable combined state remains available for
        # the short synchronization window without producing a blink/critical.

        if self.eye_state == "CLOSED" and self.eye_closed_since is None:
            self.eye_closed_since = now
        closure_sec = (
            now - self.eye_closed_since
            if self.eye_state == "CLOSED" and self.eye_closed_since is not None
            else 0.0
        )
        strong_closure = self._strong_closure(now, closure_sec)
        if closure_sec >= self.prolonged_sec and strong_closure and not self.prolonged_emitted:
            evidence.append(
                EvidenceEvent(
                    "PROLONGED_EYE_CLOSURE", now, 1.0, 3.0,
                    "eye-events-v2.4", closure_sec,
                    details={"strong_closure_valid": True},
                )
            )
            self.prolonged_emitted = True
            self.risk_episode_active = True

        eye_open_trust = (
            now - self.eye_open_since
            if self.eye_state == "OPEN" and self.eye_open_since is not None
            else 0.0
        )
        if self.risk_episode_active and eye_open_trust >= self.eye_acknowledge_open:
            self.decision_eye_intervals.clear()
            self.decision_epoch_sec = now
            self.last_decision_closed = False
            self.risk_episode_active = False

        if downward_gaze_active:
            if self.downward_gaze_since is None:
                self.downward_gaze_since = now
            if now - self.downward_gaze_since >= 0.30 and not self.downward_gaze_emitted:
                evidence.append(
                    EvidenceEvent(
                        "DISTRACTION", now, signal.downward_gaze_confidence, 2.0,
                        "eye-events-v2.4", details={"subtype": "DOWNWARD_GAZE"},
                    )
                )
                self.downward_gaze_emitted = True
        else:
            self.downward_gaze_since = None
            self.downward_gaze_emitted = False

        signal.left_eye_closed = int(self.left_eye_state == "CLOSED")
        signal.right_eye_closed = int(self.right_eye_state == "CLOSED")
        signal.eyes_closed = int(self.eye_state == "CLOSED")
        self.last_valid_eye_time = now
        return evidence

    def _closure_quality(self, now: float, duration: float) -> tuple[bool, dict]:
        start = now - min(max(duration, self.blink_min), self.partial_window)
        samples = [
            (value, down)
            for timestamp, value, valid, down in self.strong_samples
            if timestamp >= start and valid
        ]
        if not samples:
            return False, {"closure_median_max_ear": None, "downward_fraction": 1.0}
        values = np.asarray([value for value, _ in samples], dtype=float)
        downward = np.asarray([down for _, down in samples], dtype=float)
        median_ear = float(np.median(values))
        downward_fraction = float(
            np.mean(downward >= self.strong_downward_confidence)
        )
        trustworthy = bool(
            median_ear <= self.strong_sample_ear
            and downward_fraction <= self.strong_max_downward_fraction
        )
        return trustworthy, {
            "closure_median_max_ear": median_ear,
            "downward_fraction": downward_fraction,
        }

    def _strong_closure(self, now: float, closure_sec: float) -> bool:
        if closure_sec < self.prolonged_sec:
            return False
        start = now - self.prolonged_sec
        samples = [
            (value, valid, down)
            for timestamp, value, valid, down in self.strong_samples
            if timestamp >= start
        ]
        timed = [
            timestamp for timestamp, _, valid, _ in self.strong_samples
            if timestamp >= start and valid
        ]
        values = np.asarray([value for value, valid, _ in samples if valid], dtype=float)
        downward = np.asarray([down for _, valid, down in samples if valid], dtype=float)
        if len(values) < 2 or not timed:
            return False
        coverage = max(0.0, timed[-1] - timed[0]) / max(self.prolonged_sec, 1e-6)
        return bool(
            coverage >= self.strong_coverage
            and float(np.median(values)) <= self.strong_median_ear
            and float(np.mean(values <= self.strong_sample_ear)) >= self.strong_fraction
            and float(np.mean(downward >= self.strong_downward_confidence))
            < self.strong_max_downward_fraction
        )

    def _sustained_partial_closure(self, now: float) -> bool:
        start = now - self.partial_window
        selected = [
            (timestamp, value, valid, down)
            for timestamp, value, valid, down in self.strong_samples
            if timestamp >= start
        ]
        valid = [(timestamp, value, down) for timestamp, value, ok, down in selected if ok]
        if len(valid) < 2:
            return False
        coverage = (valid[-1][0] - valid[0][0]) / max(self.partial_window, 1e-6)
        fraction = float(np.mean([value <= self.partial_closure_ear for _, value, _ in valid]))
        downward_fraction = float(np.mean([
            down >= self.strong_downward_confidence for _, _, down in valid
        ]))
        return bool(
            coverage >= 0.80
            and fraction >= self.partial_fraction
            and downward_fraction < self.strong_max_downward_fraction
            and valid[-1][2] < self.strong_downward_confidence
        )

    def _snapshot(
        self,
        signal: FaceSignal,
        evidence: list[EvidenceEvent],
        mouth_state: str,
        mouth_duration: float,
    ) -> EventEngineSnapshot:
        now = signal.monotonic_sec
        self._prune(now)
        perclos_30, observed_30, coverage_30 = self._perclos(now, 30.0)
        perclos_60, observed_60, coverage_60 = self._perclos(now, 60.0)
        decision_perclos, decision_observed, decision_coverage = self._perclos(
            now, 30.0, self.decision_eye_intervals
        )
        raw_valid = bool(signal.eye_signal_valid)
        gap_sec = (
            max(0.0, now - self.invalid_eye_since)
            if not raw_valid and self.invalid_eye_since is not None
            else 0.0
        )
        if raw_valid:
            observation_status = "VALID"
        elif gap_sec <= self.max_eye_gap and self.eye_state in {"OPEN", "CLOSED"}:
            observation_status = "GRACE"
        else:
            observation_status = "LOST"
        eye_trustworthy = bool(
            observation_status in {"VALID", "GRACE"}
            and self.eye_state in {"OPEN", "CLOSED"}
        )
        closure = (
            now - self.eye_closed_since
            if eye_trustworthy
            and self.eye_state == "CLOSED"
            and self.eye_closed_since is not None
            else 0.0
        )
        eye_open_trust = (
            now - self.eye_open_since
            if eye_trustworthy and self.eye_state == "OPEN" and self.eye_open_since is not None
            else 0.0
        )
        strong_closure = self._strong_closure(now, closure)
        downward_active = bool(
            signal.downward_gaze_confidence >= 0.5
            and max(signal.event_left_relative_ear, signal.event_right_relative_ear)
            > self.strong_sample_ear
        )
        return EventEngineSnapshot(
            now,
            self.eye_state if eye_trustworthy else "UNKNOWN",
            closure,
            len(self.blinks),
            sum(int(long) for _, long in self.blinks),
            perclos_30,
            perclos_60,
            mouth_state,
            mouth_duration,
            len(self.yawns),
            evidence,
            eye_signal_valid=eye_trustworthy,
            eye_signal_quality=float(signal.eye_signal_quality),
            event_relative_ear=float(signal.event_relative_ear),
            event_left_relative_ear=float(signal.event_left_relative_ear),
            event_right_relative_ear=float(signal.event_right_relative_ear),
            valid_observation_sec=observed_30,
            valid_observation_sec_30s=observed_30,
            valid_observation_sec_60s=observed_60,
            perclos_coverage_30s=coverage_30,
            perclos_coverage_60s=coverage_60,
            raw_eye_signal_valid=raw_valid,
            eye_observation_status=observation_status,
            eye_signal_gap_sec=gap_sec,
            binocular_consistent=bool(signal.binocular_consistent),
            eye_evidence_trustworthy=eye_trustworthy,
            closure_measurement_valid=bool(
                raw_valid
                and abs(signal.gaze_x) <= self.max_eye_gaze_offset
                and abs(signal.gaze_y) <= self.max_eye_gaze_offset
                and not downward_active
            ),
            eye_open_trust_sec=eye_open_trust,
            strong_closure_valid=strong_closure,
            downward_gaze_confidence=float(signal.downward_gaze_confidence),
            downward_gaze_active=downward_active,
            sustained_partial_closure=self._sustained_partial_closure(now),
            raw_perclos_30s=perclos_30,
            decision_perclos_30s=decision_perclos,
            decision_valid_observation_sec_30s=decision_observed,
            decision_perclos_coverage_30s=decision_coverage,
            eye_resolution_valid=bool(signal.eye_resolution_valid),
            driver_distance_status=str(signal.driver_distance_status),
            face_width_ratio=float(signal.face_width_ratio),
            interocular_distance_px=float(signal.interocular_distance_px),
            minimum_eye_width_px=float(
                min(signal.left_eye_width_px, signal.right_eye_width_px)
            ),
        )

    def update(
        self,
        signal: FaceSignal,
        mar_baseline: float,
        pitch_baseline: float,
    ) -> EventEngineSnapshot:
        now = signal.monotonic_sec
        if not signal.face_detected:
            self._invalidate_eye(now)
            self.mouth_open_since = self.nod_since = None
            self.nod_emitted = False
            return self._snapshot(signal, [], "UNKNOWN", 0.0)

        evidence = self._update_eye(signal)

        mouth_threshold = max(0.15, mar_baseline * self.mouth_ratio)
        signal.mouth_open = int(signal.mar >= mouth_threshold)
        signal.yawn_candidate = signal.mouth_open
        mouth_duration = 0.0
        if signal.mouth_open:
            if self.mouth_open_since is None:
                self.mouth_open_since = now
            mouth_duration = now - self.mouth_open_since
        elif self.mouth_open_since is not None:
            duration = now - self.mouth_open_since
            if self.yawn_min <= duration <= self.yawn_max:
                self.yawns.append(now)
                evidence.append(
                    EvidenceEvent(
                        "YAWNING", now, 1.0, 8.0,
                        "event_engine_v2", duration,
                    )
                )
            self.mouth_open_since = None

        nod_active = signal.pitch - pitch_baseline >= self.nod_delta
        if nod_active:
            if self.nod_since is None:
                self.nod_since, self.nod_emitted = now, False
            if now - self.nod_since >= self.nod_min and not self.nod_emitted:
                evidence.append(
                    EvidenceEvent(
                        "HEAD_NOD", now, 0.9, 4.0,
                        "event_engine_v2", now - self.nod_since,
                    )
                )
                self.nod_emitted = True
        else:
            self.nod_since, self.nod_emitted = None, False

        return self._snapshot(
            signal,
            evidence,
            "OPEN" if signal.mouth_open else "CLOSED",
            mouth_duration,
        )
