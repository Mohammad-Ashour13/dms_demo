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
