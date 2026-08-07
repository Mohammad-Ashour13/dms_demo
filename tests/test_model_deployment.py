from __future__ import annotations

import json

import pytest

from dms_final_system.runtime.config import load_config
from dms_final_system.runtime.model.bundle import (
    BundleVerificationError,
    assert_deployment_allowed,
)


def test_experimental_bundle_cannot_be_active():
    with pytest.raises(BundleVerificationError, match="restricted to SHADOW"):
        assert_deployment_allowed("EXPERIMENTAL", "ACTIVE")
    assert_deployment_allowed("EXPERIMENTAL", "SHADOW")


def test_evaluation_requires_shadow_mode(tmp_path):
    config = tmp_path / "runtime.json"
    config.write_text(
        json.dumps({"deployment_mode": "ACTIVE", "evaluation": {"enabled": True}})
    )
    with pytest.raises(ValueError, match="must run with deployment_mode=SHADOW"):
        load_config(config)


def test_legacy_single_eye_threshold_is_migrated_safely(tmp_path):
    config = tmp_path / "runtime.json"
    config.write_text(json.dumps({"events": {"eye_closed_relative_ear": 0.72}}))
    loaded = load_config(config)
    assert loaded.events.eye_close_relative_ear == 0.60
    assert loaded.events.eye_reopen_relative_ear == 0.72


def test_behavior_detector_accepts_auto_backend_and_legacy_ultralytics(tmp_path):
    auto_config = tmp_path / "auto.json"
    auto_config.write_text(
        json.dumps(
            {
                "behavior_detector": {
                    "backend": "auto",
                    "execution_mode": "auto",
                    "roi_mode": "face",
                    "adaptive_interval": True,
                }
            }
        )
    )
    loaded = load_config(auto_config)
    assert loaded.behavior_detector.backend == "auto"
    assert loaded.behavior_detector.execution_mode == "auto"
    assert loaded.behavior_detector.roi_mode == "face"
    assert loaded.perception.process_width == 256
    legacy_config = tmp_path / "legacy.json"
    legacy_config.write_text(
        json.dumps({"behavior_detector": {"backend": "ultralytics", "execution_mode": "process"}})
    )
    legacy = load_config(legacy_config)
    assert legacy.behavior_detector.backend == "ultralytics"
    assert legacy.behavior_detector.execution_mode == "process"
