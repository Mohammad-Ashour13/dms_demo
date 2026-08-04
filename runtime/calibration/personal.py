from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from dms_final_system.shared.contracts import FaceSignal


@dataclass(slots=True)
class CalibrationResult:
    status: str
    progress: float
    good_frames: int
    rejected_frames: int
    ear_baseline: float | None = None
    model_ear_baseline: float | None = None
    event_left_ear_baseline: float | None = None
    event_right_ear_baseline: float | None = None
    event_ear_baseline: float | None = None
    mar_baseline: float | None = None
    pitch_baseline: float | None = None
    yaw_baseline: float | None = None
    roll_baseline: float | None = None
    gaze_y_baseline: float | None = None
    confidence: float = 0.0
    event_ear_cv: float | None = None
    event_eye_asymmetry: float | None = None
    excluded_event_frames: int = 0
    baseline_method: str = "model_median_event_trimmed_median"
    failure_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class PersonalCalibrator:
    def __init__(
        self,
        target_sec=10.0,
        max_sec=15.0,
        minimum_good_frames=75,
        min_quality=0.55,
        max_pose_deviation_deg=25.0,
        event_trim_low_quantile=0.20,
        event_trim_high_quantile=0.95,
        max_event_ear_cv=0.35,
        max_event_eye_asymmetry=0.35,
    ):
        self.target_sec = target_sec
        self.max_sec = max_sec
        self.minimum_good_frames = minimum_good_frames
        self.min_quality = min_quality
        self.max_pose_deviation_deg = max_pose_deviation_deg
        self.event_trim_low_quantile = event_trim_low_quantile
        self.event_trim_high_quantile = event_trim_high_quantile
        self.max_event_ear_cv = max_event_ear_cv
        self.max_event_eye_asymmetry = max_event_eye_asymmetry
        self.started_at: float | None = None
        self.accepted: list[FaceSignal] = []
        self.rejected = 0
        self.result: CalibrationResult | None = None

    @property
    def ready(self) -> bool:
        return self.result is not None and self.result.status == "READY"

    def update(self, signal: FaceSignal) -> CalibrationResult:
        if self.result is not None:
            return self.result
        if self.started_at is None:
            self.started_at = signal.monotonic_sec
        elapsed = signal.monotonic_sec - self.started_at
        pose_ok = max(abs(signal.pitch), abs(signal.yaw), abs(signal.roll)) <= self.max_pose_deviation_deg
        if (
            signal.face_detected
            and signal.face_quality >= self.min_quality
            and signal.eye_signal_valid
            and signal.ear > 0.05
            and signal.event_left_ear > 0
            and signal.event_right_ear > 0
            and pose_ok
        ):
            self.accepted.append(signal)
        else:
            self.rejected += 1
        enough_time = elapsed >= self.target_sec
        enough_frames = len(self.accepted) >= self.minimum_good_frames
        stability_reason = ""
        if enough_time and enough_frames:
            candidate = self._finish("READY")
            if not candidate.failure_reason:
                self.result = candidate
                return self.result
            stability_reason = candidate.failure_reason
        if elapsed >= self.max_sec:
            reason = (
                "insufficient_good_frames"
                if not enough_frames
                else stability_reason or "unstable_event_ear"
            )
            self.result = self._finish("FAILED", reason)
            return self.result
        return CalibrationResult(
            status="CALIBRATING", progress=min(1.0, elapsed / self.target_sec),
            good_frames=len(self.accepted), rejected_frames=self.rejected,
            confidence=min(1.0, len(self.accepted) / max(1, self.minimum_good_frames)),
        )

    def _trimmed_event_values(self) -> tuple[np.ndarray, np.ndarray, int]:
        left = np.asarray([item.event_left_ear for item in self.accepted], dtype=float)
        right = np.asarray([item.event_right_ear for item in self.accepted], dtype=float)
        if not len(left):
            return left, right, 0
        left_low, left_high = np.quantile(
            left, [self.event_trim_low_quantile, self.event_trim_high_quantile]
        )
        right_low, right_high = np.quantile(
            right, [self.event_trim_low_quantile, self.event_trim_high_quantile]
        )
        keep = (
            (left >= left_low) & (left <= left_high)
            & (right >= right_low) & (right <= right_high)
        )
        return left[keep], right[keep], int(len(left) - int(np.sum(keep)))

    def _finish(self, status: str, forced_reason: str = "") -> CalibrationResult:
        if not self.accepted:
            return CalibrationResult(
                status, 1.0, 0, self.rejected, confidence=0.0,
                failure_reason=forced_reason or "no_accepted_frames",
            )
        values = lambda name: np.asarray([getattr(item, name) for item in self.accepted], dtype=float)
        event_left, event_right, excluded = self._trimmed_event_values()
        if not len(event_left):
            return CalibrationResult(
                status, 1.0, len(self.accepted), self.rejected, confidence=0.0,
                excluded_event_frames=excluded,
                failure_reason=forced_reason or "no_stable_event_frames",
            )
        event_mean = (event_left + event_right) / 2.0
        event_cv = float(np.std(event_mean) / max(float(np.mean(event_mean)), 1e-6))
        event_asymmetry = float(
            np.median(np.abs(event_left - event_right) / np.maximum(event_mean, 1e-6))
        )
        stability_reason = ""
        if event_cv > self.max_event_ear_cv:
            stability_reason = "event_ear_cv_above_limit"
        elif event_asymmetry > self.max_event_eye_asymmetry:
            stability_reason = "event_eye_asymmetry_above_limit"
        failure_reason = forced_reason or stability_reason
        if status == "READY" and failure_reason:
            status = "CALIBRATING"
        confidence = min(1.0, len(self.accepted) / max(1, self.minimum_good_frames))
        model_baseline = float(np.median(values("ear")))
        event_left_baseline = float(np.median(event_left))
        event_right_baseline = float(np.median(event_right))
        return CalibrationResult(
            status=status, progress=1.0, good_frames=len(self.accepted), rejected_frames=self.rejected,
            ear_baseline=model_baseline,
            model_ear_baseline=model_baseline,
            event_left_ear_baseline=event_left_baseline,
            event_right_ear_baseline=event_right_baseline,
            event_ear_baseline=float((event_left_baseline + event_right_baseline) / 2.0),
            mar_baseline=float(np.median(values("mar"))),
            pitch_baseline=float(np.median(values("pitch"))),
            yaw_baseline=float(np.median(values("yaw"))),
            roll_baseline=float(np.median(values("roll"))),
            gaze_y_baseline=float(np.median(values("gaze_y"))),
            confidence=confidence if status == "READY" else 0.0,
            event_ear_cv=event_cv,
            event_eye_asymmetry=event_asymmetry,
            excluded_event_frames=excluded,
            failure_reason=failure_reason,
        )

    def apply(self, signal: FaceSignal) -> FaceSignal:
        if not self.ready or not self.result or not self.result.model_ear_baseline:
            signal.relative_ear = 0.0
            signal.event_left_relative_ear = 0.0
            signal.event_right_relative_ear = 0.0
            signal.event_relative_ear = 0.0
            signal.eye_signal_valid = False
            return signal
        signal.relative_ear = float(
            np.clip(signal.ear / self.result.model_ear_baseline, 0.0, 2.0)
        )
        signal.event_left_relative_ear = float(
            np.clip(
                signal.event_left_ear / max(self.result.event_left_ear_baseline or 0.0, 1e-6),
                0.0,
                2.0,
            )
        )
        signal.event_right_relative_ear = float(
            np.clip(
                signal.event_right_ear / max(self.result.event_right_ear_baseline or 0.0, 1e-6),
                0.0,
                2.0,
            )
        )
        signal.event_relative_ear = float(
            (signal.event_left_relative_ear + signal.event_right_relative_ear) / 2.0
        )
        signal.relative_gaze_y = float(
            np.clip(signal.gaze_y - float(self.result.gaze_y_baseline or 0.0), -2.0, 2.0)
        )
        signal.gaze_signal_valid = bool(signal.eye_signal_valid and np.isfinite(signal.gaze_y))
        signal.eye_signal_valid = bool(
            signal.eye_signal_valid
            and np.isfinite(signal.event_relative_ear)
            and signal.event_relative_ear > 0
        )
        return signal
