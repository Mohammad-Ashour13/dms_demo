from __future__ import annotations

import json

from dms_final_system.runtime.model.drift import FeatureDriftMonitor
from dms_final_system.shared.contracts import TemporalFeatureSnapshot


def _snapshot(value_a: float, value_b: float, index: int):
    return TemporalFeatureSnapshot(
        f"w{index}", float(index), float(index + 1), 1.0,
        [value_a, value_b], ["a", "b"], True,
    )


def test_drift_guard_disables_and_recovers_after_configured_streaks(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "feature_reference.json").write_text(
        json.dumps(
            {
                "features": {
                    "a": {"q01": 0.0, "q99": 1.0, "gain_rank": 1},
                    "b": {"q01": 0.0, "q99": 1.0, "gain_rank": 2},
                }
            }
        )
    )
    monitor = FeatureDriftMonitor(
        bundle,
        top_feature_count=2,
        out_of_range_fraction=0.30,
        trigger_consecutive_windows=3,
        recovery_consecutive_windows=3,
    )
    for index in range(3):
        result = monitor.evaluate(_snapshot(2.0, 0.5, index))
    assert result.reason == "MODEL_INPUT_OOD"
    assert not result.model_contribution_enabled
    assert result.changed
    for index in range(3, 6):
        result = monitor.evaluate(_snapshot(0.5, 0.5, index))
    assert result.reason == "IN_DISTRIBUTION"
    assert result.model_contribution_enabled
    assert result.changed


def test_missing_reference_is_non_blocking_but_visible(tmp_path):
    monitor = FeatureDriftMonitor(tmp_path)
    result = monitor.evaluate(_snapshot(100.0, 100.0, 0))
    assert not result.available
    assert result.model_contribution_enabled
    assert result.reason == "REFERENCE_UNAVAILABLE"
