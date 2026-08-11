from __future__ import annotations

import importlib.util
import hashlib
import json
import shutil
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_preflight(config, resolve, *, replay=False) -> dict:
    checks: dict[str, object] = {}
    errors: list[str] = []
    active_model = resolve(config.active_model_path)
    checks["active_model"] = str(active_model)
    if not active_model.is_file():
        errors.append(
            f"Active model descriptor is missing: {active_model}. Run runtime.model.activate first."
        )

    if not replay and config.camera.backend == "picamera2":
        available = importlib.util.find_spec("picamera2") is not None
        checks["picamera2_available"] = available
        if not available:
            errors.append("Picamera2 is unavailable for configured CSI camera backend")
        else:
            try:
                from picamera2 import Picamera2

                cameras = Picamera2.global_camera_info()
                checks["picamera2_cameras"] = cameras
                if int(config.camera.camera_num) >= len(cameras):
                    errors.append(
                        f"Configured CSI camera {config.camera.camera_num} is unavailable"
                    )
            except Exception as exc:
                errors.append(f"Could not enumerate Picamera2 cameras: {exc}")

    recorder_dir = resolve(config.recorder.output_dir)
    recorder_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(recorder_dir).free
    checks["recorder_free_bytes"] = free
    checks["recorder_reserve_bytes"] = config.recorder.reserve_free_bytes
    if config.recorder.enabled and config.recorder.require_copy_mux:
        from dms_final_system.runtime.recording.rolling import RollingIncidentRecorder

        copy_mux_available = RollingIncidentRecorder._probe_copy_mux()
        checks["ffmpeg_mjpeg_copy_mux"] = copy_mux_available
        if not copy_mux_available:
            errors.append("FFmpeg MJPEG-in-MP4 copy mux is required but unavailable")

    detector = config.behavior_detector
    if detector.enabled:
        model_manifest = {}
        model_path = resolve(detector.model_path)
        manifest_path = resolve(detector.manifest_path)
        checks["behavior_model"] = str(model_path)
        checks["behavior_manifest"] = str(manifest_path)
        if not model_path.exists():
            errors.append(f"Behavior model is missing: {model_path}")
        if not manifest_path.is_file():
            errors.append(f"Behavior model manifest is missing: {manifest_path}")
        else:
            try:
                model_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                source_classes = set(model_manifest.get("source_classes") or [])
                required = set(detector.class_mapping)
                if not required <= source_classes:
                    errors.append(
                        f"Behavior manifest lacks required classes: {sorted(required - source_classes)}"
                    )
                if model_manifest.get("enabled_classes") != detector.class_mapping:
                    errors.append(
                        "Behavior manifest enabled_classes does not match configured class_mapping"
                    )
                if model_path.is_file() and model_manifest.get("sha256"):
                    actual = _sha256(model_path)
                    checks["behavior_model_sha256"] = actual
                    if actual != model_manifest["sha256"]:
                        errors.append("Behavior model checksum does not match its manifest")
            except (OSError, ValueError) as exc:
                errors.append(f"Behavior manifest is invalid: {exc}")
        if model_path.is_dir():
            params = list(model_path.glob("*.param"))
            bins = list(model_path.glob("*.bin"))
            export_manifest = model_path / "ncnn_export_manifest.json"
            if not params or not bins:
                errors.append("NCNN directory must contain .param and .bin files")
            if not export_manifest.is_file():
                errors.append("NCNN export manifest is missing")
            else:
                try:
                    payload = json.loads(export_manifest.read_text(encoding="utf-8"))
                    checks["ncnn_image_size"] = payload.get("imgsz")
                    checks["ncnn_exporter_versions"] = payload.get("exporter_versions") or {}
                    if int(payload.get("imgsz", -1)) != int(detector.image_size):
                        errors.append("NCNN export image size does not match detector.image_size")
                    artifact_hashes = payload.get("artifacts_sha256") or {}
                    required_artifacts = {path.name for path in [*params, *bins]}
                    if set(artifact_hashes) != required_artifacts:
                        errors.append("NCNN export manifest checksum coverage is incomplete")
                    actual_artifact_hashes = {}
                    for filename, expected in artifact_hashes.items():
                        artifact = model_path / filename
                        actual = _sha256(artifact) if artifact.is_file() else None
                        actual_artifact_hashes[filename] = actual
                        if actual != expected:
                            errors.append(f"NCNN artifact checksum mismatch: {filename}")
                    checks["ncnn_artifacts_sha256"] = actual_artifact_hashes
                    source_sha = payload.get("source_sha256")
                    if (
                        source_sha
                        and model_manifest.get("sha256")
                        and source_sha != model_manifest["sha256"]
                    ):
                        errors.append("NCNN export source checksum does not match behavior manifest")
                    export_classes = payload.get("source_classes") or []
                    if export_classes != model_manifest.get("source_classes"):
                        errors.append("NCNN export class order does not match behavior manifest")
                except (OSError, ValueError, TypeError) as exc:
                    errors.append(f"NCNN export manifest is invalid: {exc}")
    if errors:
        raise RuntimeError("Preflight failed:\n- " + "\n- ".join(errors))
    return checks
