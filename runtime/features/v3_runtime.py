"""NumPy-only online implementation of the V3 window formulas."""

from __future__ import annotations

import math

import numpy as np

from dms_final_system.shared.contracts import FaceSignal, TemporalFeatureSnapshot
from dms_final_system.shared.feature_contract import (
    BINARY_BASE_SIGNALS,
    CONTINUOUS_BASE_SIGNALS,
    MIN_COVERAGE,
    PHYSICAL_BOUNDS,
    TARGET_SAMPLES,
    WINDOW_SEC,
)
from dms_final_system.runtime.temporal.buffers import FaceSignalBuffer


SMOOTHED_BASE_SIGNALS = {
    "mar", "pitch", "roll", "head_motion", "gaze_x", "gaze_y", "ear",
    "relative_ear", "face_velocity_x", "face_velocity_y",
}
GAZE_VOCABULARY = ("CENTER", "LEFT", "RIGHT", "UP", "DOWN", "UNKNOWN")


def _rolling_mean(values: np.ndarray, window: int = 2) -> np.ndarray:
    if window <= 1:
        return values.copy()
    out = np.empty_like(values, dtype=float)
    # Matches pandas rolling(window=2, min_periods=1, center=True).
    for i in range(len(values)):
        start = max(0, i - (window // 2))
        end = min(len(values), i + ((window - 1) // 2) + 1)
        out[i] = np.mean(values[start:end])
    return out


def _skew(values: np.ndarray) -> float:
    n = len(values)
    if n < 3 or np.ptp(values) <= 1e-12:
        return 0.0
    centered = values - np.mean(values)
    m2, m3 = np.mean(centered ** 2), np.mean(centered ** 3)
    return float(math.sqrt(n * (n - 1)) / (n - 2) * m3 / (m2 ** 1.5))


def _kurtosis(values: np.ndarray) -> float:
    n = len(values)
    if n < 4 or np.ptp(values) <= 1e-12:
        return 0.0
    centered = values - np.mean(values)
    m2, m4 = np.mean(centered ** 2), np.mean(centered ** 4)
    g2 = m4 / (m2 ** 2) - 3.0
    return float((n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6))


def _stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    median = float(np.median(values))
    q25, q75 = float(np.quantile(values, 0.25)), float(np.quantile(values, 0.75))
    mean, std = float(np.mean(values)), float(np.std(values))
    return {
        f"{prefix}_mean": mean, f"{prefix}_std": std,
        f"{prefix}_variance": float(np.var(values)), f"{prefix}_median": median,
        f"{prefix}_min": float(np.min(values)), f"{prefix}_max": float(np.max(values)),
        f"{prefix}_q25": q25, f"{prefix}_q75": q75, f"{prefix}_iqr": q75 - q25,
        f"{prefix}_range": float(np.ptp(values)),
        f"{prefix}_mad": float(np.median(np.abs(values - median))),
        f"{prefix}_coefficient_of_variation": std / abs(mean) if abs(mean) >= 1e-12 else 0.0,
        f"{prefix}_skewness": _skew(values), f"{prefix}_kurtosis": _kurtosis(values),
        f"{prefix}_energy": float(np.mean(values ** 2)),
    }


def _rolling_std_summary(values: np.ndarray, window: int = 3) -> float:
    pieces = []
    for end in range(1, len(values) + 1):
        part = values[max(0, end - window):end]
        pieces.append(float(np.std(part, ddof=1)) if len(part) > 1 else 0.0)
    return float(np.mean(pieces))


def _temporal(values: np.ndarray, time_sec: np.ndarray, prefix: str) -> dict[str, float]:
    slope = float(np.polyfit(time_sec, values, 1)[0]) if len(values) >= 2 else 0.0
    diff = np.diff(values)
    velocity = np.gradient(values, time_sec) if len(values) >= 2 else np.zeros_like(values)
    acceleration = np.gradient(velocity, time_sec) if len(values) >= 2 else np.zeros_like(values)
    rolling_means = [float(np.mean(values[max(0, end - 3):end])) for end in range(1, len(values) + 1)]
    return {
        f"{prefix}_linear_trend": slope, f"{prefix}_slope": slope,
        f"{prefix}_first_last_difference": float(values[-1] - values[0]) if len(values) >= 2 else 0.0,
        f"{prefix}_mean_absolute_change": float(np.mean(np.abs(diff))) if len(diff) else 0.0,
        f"{prefix}_maximum_absolute_change": float(np.max(np.abs(diff))) if len(diff) else 0.0,
        f"{prefix}_rolling_mean": float(np.mean(rolling_means)),
        f"{prefix}_rolling_std": _rolling_std_summary(values),
        f"{prefix}_velocity_mean": float(np.mean(np.abs(velocity))),
        f"{prefix}_velocity_max": float(np.max(np.abs(velocity))),
        f"{prefix}_acceleration_mean": float(np.mean(np.abs(acceleration))),
        f"{prefix}_acceleration_max": float(np.max(np.abs(acceleration))),
    }


def _runs(active: np.ndarray) -> list[tuple[int, int]]:
    found, start = [], None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        elif not value and start is not None:
            found.append((start, index - 1))
            start = None
    if start is not None:
        found.append((start, len(active) - 1))
    return found


def _events(values: np.ndarray, time_sec: np.ndarray, prefix: str) -> dict[str, float]:
    active = values >= 0.5
    runs = _runs(active)
    durations = [float(time_sec[end] - time_sec[start]) if end > start else 0.0 for start, end in runs]
    rising = int(np.sum(active[1:] & ~active[:-1])) if len(active) > 1 else 0
    return {
        f"{prefix}_active_ratio": float(np.mean(active)),
        f"{prefix}_event_count": float(len(runs)),
        f"{prefix}_mean_event_duration": float(np.mean(durations)) if durations else 0.0,
        f"{prefix}_maximum_event_duration": float(np.max(durations)) if durations else 0.0,
        f"{prefix}_minimum_event_duration": float(np.min(durations)) if durations else 0.0,
        f"{prefix}_std_event_duration": float(np.std(durations)) if durations else 0.0,
        f"{prefix}_events_per_second": float(len(runs) / WINDOW_SEC),
        f"{prefix}_first_event_time": float(time_sec[runs[0][0]]) if runs else float("nan"),
        f"{prefix}_last_event_time": float(time_sec[runs[-1][0]]) if runs else float("nan"),
        f"{prefix}_rising_edge_count": float(rising),
    }


class V3RuntimeFeatureBuilder:
    def __init__(self, min_face_quality: float = 0.55, window_sec=WINDOW_SEC, target_samples=TARGET_SAMPLES, min_coverage=MIN_COVERAGE):
        self.buffer = FaceSignalBuffer(max(65.0, window_sec + 1.0))
        self.min_face_quality = min_face_quality
        self.window_sec = window_sec
        self.target_samples = target_samples
        self.min_coverage = min_coverage
        self.sequence_number = 0

    def update(self, signal: FaceSignal) -> None:
        self.buffer.append(signal)

    def build(self, ordered_features: list[str], end_sec: float) -> TemporalFeatureSnapshot:
        start_sec = end_sec - self.window_sec
        samples = self.buffer.window(start_sec, end_sec, self.min_face_quality)
        window_id = f"w-{self.sequence_number:010d}"
        self.sequence_number += 1
        if len(samples) < 2:
            return TemporalFeatureSnapshot(window_id, start_sec, end_sec, 0.0, [], ordered_features, False, "fewer_than_two_good_samples")
        source_t = np.asarray([s.monotonic_sec for s in samples], dtype=float)
        grid_t = np.linspace(start_sec, end_sec, num=self.target_samples, endpoint=False)
        inside = (grid_t >= source_t.min()) & (grid_t <= source_t.max())
        coverage = float(np.mean(inside))
        if coverage < self.min_coverage:
            return TemporalFeatureSnapshot(window_id, start_sec, end_sec, coverage, [], ordered_features, False, "coverage_below_threshold")
        all_features: dict[str, float] = {}
        relative_t = grid_t - grid_t[0]
        for base in CONTINUOUS_BASE_SIGNALS:
            raw = np.asarray([getattr(s, base) for s in samples], dtype=float)
            bounds = PHYSICAL_BOUNDS.get(base)
            if bounds:
                raw = np.clip(raw, *bounds)
            if base in SMOOTHED_BASE_SIGNALS and len(source_t) >= 2:
                positive_dt = np.diff(source_t)
                positive_dt = positive_dt[positive_dt > 0]
                median_dt = float(np.median(positive_dt)) if len(positive_dt) else 1.0 / 15.0
                smooth_samples = max(1, int(round(0.1 / median_dt)))
                raw = _rolling_mean(raw, smooth_samples)
            values = np.interp(grid_t, source_t, raw, left=np.nan, right=np.nan)
            if np.isnan(values).any():
                valid = np.flatnonzero(~np.isnan(values))
                values = np.interp(np.arange(len(values)), valid, values[valid])
            all_features.update(_stats(values, base))
            all_features.update(_temporal(values, relative_t, base))
        for base in BINARY_BASE_SIGNALS:
            raw = np.asarray([getattr(s, base) for s in samples], dtype=float)
            values = np.interp(grid_t, source_t, raw, left=np.nan, right=np.nan)
            if np.isnan(values).any():
                valid = np.flatnonzero(~np.isnan(values))
                values = np.interp(np.arange(len(values)), valid, values[valid])
            all_features.update(_events(values, relative_t, base))
        zones = np.asarray([s.gaze_zone for s in samples], dtype=object)
        right = np.clip(np.searchsorted(source_t, grid_t, side="left"), 0, len(source_t) - 1)
        left = np.clip(right - 1, 0, len(source_t) - 1)
        nearest = np.where(np.abs(grid_t - source_t[left]) <= np.abs(source_t[right] - grid_t), left, right)
        resampled_zones = np.asarray([str(zones[i]).upper() for i in nearest])
        for zone in GAZE_VOCABULARY:
            all_features[f"gaze_zone_{zone}_ratio"] = float(np.mean(resampled_zones == zone))
        missing = [name for name in ordered_features if name not in all_features]
        if missing:
            return TemporalFeatureSnapshot(window_id, start_sec, end_sec, coverage, [], ordered_features, False, f"unsupported_features:{missing[:5]}")
        values = [float(all_features[name]) for name in ordered_features]
        return TemporalFeatureSnapshot(window_id, start_sec, end_sec, coverage, values, list(ordered_features), True)
