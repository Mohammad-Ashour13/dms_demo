from __future__ import annotations

import numpy as np

from dms_final_system.runtime.features.v3_runtime import V3RuntimeFeatureBuilder
from dms_final_system.shared.contracts import FaceSignal


def _signal(index: int, relative_ear=1.0, face=True):
    t = index / 15.0
    ear = 0.20 + 0.001 * index
    return FaceSignal(
        index, f"t-{index}", t, face, 1.0 if face else 0.0,
        left_ear=ear, right_ear=ear + 0.01, ear=ear + 0.005,
        relative_ear=relative_ear, mar=0.2, pitch=1.0, yaw=2.0, roll=0.5,
        gaze_x=0.0, gaze_y=0.2, gaze_zone="DOWN",
    )


def test_v3_window_is_30_points_and_ordered_feature_values_are_correct():
    builder = V3RuntimeFeatureBuilder(min_face_quality=0.5)
    for index in range(30):
        builder.update(_signal(index))
    names = ["left_ear_mean", "left_ear_min", "left_ear_max", "gaze_zone_DOWN_ratio"]
    snapshot = builder.build(names, 2.0)
    assert snapshot.valid
    assert snapshot.feature_names == names
    assert snapshot.coverage == 1.0
    expected = np.asarray([0.20 + 0.001 * i for i in range(30)])
    assert np.isclose(snapshot.ordered_features[0], expected.mean())
    assert np.isclose(snapshot.ordered_features[1], expected.min())
    assert np.isclose(snapshot.ordered_features[2], expected.max())
    assert snapshot.ordered_features[3] == 1.0


def test_window_rejects_low_face_coverage():
    builder = V3RuntimeFeatureBuilder(min_face_quality=0.5)
    for index in range(30):
        builder.update(_signal(index, face=index >= 15))
    snapshot = builder.build(["ear_mean"], 2.0)
    assert not snapshot.valid
    assert snapshot.reason == "coverage_below_threshold"
