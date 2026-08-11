from __future__ import annotations

import json
from pathlib import Path

import pytest

from dms_final_system.runtime.config import RuntimeConfig, load_config
from dms_final_system.runtime.preflight import run_preflight


def test_preflight_reports_missing_active_descriptor(tmp_path):
    config = RuntimeConfig()
    config.active_model_path = "missing.json"
    config.recorder.enabled = False
    with pytest.raises(RuntimeError, match="Active model descriptor is missing"):
        run_preflight(config, lambda value: tmp_path / value, replay=True)


def test_preflight_rejects_ncnn_image_size_or_checksum_mismatch(tmp_path):
    active = tmp_path / "active.json"
    active.write_text('{"bundle_path":"bundle","deployment_mode":"SHADOW"}', encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    (model / "network.param").write_text("param", encoding="utf-8")
    (model / "network.bin").write_bytes(b"bin")
    (model / "ncnn_export_manifest.json").write_text(
        json.dumps(
            {
                "imgsz": 480,
                "artifacts_sha256": {"network.param": "incorrect"},
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "model_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_classes": ["phone", "cigarette", "drink_or_food"],
            }
        ),
        encoding="utf-8",
    )
    config = RuntimeConfig()
    config.active_model_path = str(active)
    config.recorder.enabled = False
    config.behavior_detector.enabled = True
    config.behavior_detector.model_path = str(model)
    config.behavior_detector.manifest_path = str(manifest)
    config.behavior_detector.image_size = 640
    with pytest.raises(RuntimeError) as error:
        run_preflight(config, lambda value: Path(value), replay=True)
    assert "image size" in str(error.value)
    assert "checksum mismatch" in str(error.value)


def test_pi5_model_and_copy_mux_preflight_pass_for_replay():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/runtime.raspberry_pi5.json")
    result = run_preflight(
        config,
        lambda value: Path(value) if Path(value).is_absolute() else root / value,
        replay=True,
    )
    assert result["ffmpeg_mjpeg_copy_mux"] is True
    assert result["ncnn_image_size"] == 640
    assert set(result["ncnn_artifacts_sha256"]) == {
        "model.ncnn.bin",
        "model.ncnn.param",
    }
