from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dms_final_system.runtime.calibration import PersonalCalibrator
from dms_final_system.runtime.perception.mediapipe_face import (
    LEFT_EYE,
    _event_ear,
    _eye_pixel_geometry,
)
from dms_final_system.shared.contracts import FaceSignal


@dataclass
class _Point:
    x: float = 0.0
    y: float = 0.0


def _eye_landmarks(height: float):
    points = [_Point() for _ in range(478)]
    coords = {
        LEFT_EYE[0]: (0.0, 0.0), LEFT_EYE[3]: (1.0, 0.0),
        LEFT_EYE[1]: (0.3, -height), LEFT_EYE[5]: (0.3, height),
        LEFT_EYE[2]: (0.7, -height), LEFT_EYE[4]: (0.7, height),
    }
    for index, (x, y) in coords.items():
        points[index] = _Point(x, y)
    return points


def _signal(t: float, model_ear: float, event_ear: float) -> FaceSignal:
    return FaceSignal(
        int(t * 100), "utc", t, True, 0.95,
        left_ear=model_ear, right_ear=model_ear, ear=model_ear,
        event_left_ear=event_ear, event_right_ear=event_ear,
        event_ear=event_ear, eye_signal_valid=True, eye_signal_quality=0.95,
        mar=0.1, pitch=0.0, yaw=0.0, roll=0.0,
    )


def test_standard_event_ear_separates_open_and_closed_geometry():
    opened = _event_ear(_eye_landmarks(0.20), LEFT_EYE, 1, 1)
    closed = _event_ear(_eye_landmarks(0.02), LEFT_EYE, 1, 1)
    assert opened > closed * 5


def test_eye_pixel_geometry_reports_camera_resolution_not_normalized_ratio():
    points = [_Point() for _ in range(478)]
    # 640x480 image: eye widths 32px and eye centres 80px apart.
    for index, xy in {
        362: (0.60, 0.40), 263: (0.65, 0.40),
        33: (0.475, 0.40), 133: (0.525, 0.40),
    }.items():
        points[index] = _Point(*xy)
    left, right, interocular = _eye_pixel_geometry(points, 640, 480)
    assert np.isclose(left, 32.0)
    assert np.isclose(right, 32.0)
    assert np.isclose(interocular, 80.0)


def test_calibration_uses_model_median_and_trims_blink_outlier():
    calibrator = PersonalCalibrator(
        target_sec=1.0, max_sec=2.0, minimum_good_frames=10,
        max_event_ear_cv=0.35, max_event_eye_asymmetry=0.35,
    )
    model_values = [0.20 + 0.001 * index for index in range(16)]
    event_values = [0.30] * 15 + [0.03]
    for index, (model_ear, event_ear) in enumerate(zip(model_values, event_values)):
        result = calibrator.update(_signal(index / 15.0, model_ear, event_ear))
    assert result.status == "READY"
    assert np.isclose(result.model_ear_baseline, np.median(model_values))
    assert result.model_ear_baseline == result.ear_baseline
    assert np.isclose(result.event_ear_baseline, 0.30)
    assert result.excluded_event_frames > 0
    applied = calibrator.apply(_signal(2.0, result.model_ear_baseline, 0.30))
    assert np.isclose(applied.relative_ear, 1.0)
    assert np.isclose(applied.event_relative_ear, 1.0)
