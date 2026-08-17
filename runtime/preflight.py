from __future__ import annotations

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

    if not replay and config.camera.backend in {
        "picamera2",
        "picamera2_auto",
        "picamera2_process",
    }:
        from dms_final_system.runtime.capture.picamera2_process import (
            in_process_picamera2_available,
            probe_system_picamera2,
        )

        in_process = in_process_picamera2_available()
        use_process = config.camera.backend == "picamera2_process" or (
            config.camera.backend == "picamera2_auto" and not in_process
        )
        checks["picamera2_in_process_available"] = in_process
        checks["picamera2_selected_backend"] = (
            "system_process" if use_process else "in_process"
        )
        if config.camera.backend == "picamera2" and not in_process:
            errors.append("Picamera2 is unavailable for configured in-process CSI backend")
        elif use_process:
            try:
                cameras = probe_system_picamera2(
                    config.camera.system_python,
                    config.camera.startup_timeout_sec,
                )
                checks["picamera2_cameras"] = cameras
                checks["picamera2_system_python"] = config.camera.system_python
                if int(config.camera.camera_num) >= len(cameras):
                    errors.append(
                        f"Configured CSI camera {config.camera.camera_num} is unavailable"
                    )
            except Exception as exc:
                errors.append(f"Could not enumerate system Picamera2 cameras: {exc}")
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
    seatbelt = config.seatbelt_detector
    if seatbelt.enabled:
        model_path = resolve(seatbelt.model_path)
        manifest_path = resolve(seatbelt.manifest_path)
        checks["seatbelt_model"] = str(model_path)
        checks["seatbelt_manifest"] = str(manifest_path)
        if not model_path.is_dir():
            errors.append(f"Seat-belt NCNN model directory is missing: {model_path}")
        if not manifest_path.is_file():
            errors.append(f"Seat-belt model manifest is missing: {manifest_path}")
        elif model_path.is_dir():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("class_names") != ["no_seatbelt", "seat_belt"]:
                    errors.append("Seat-belt manifest must define no_seatbelt and seat_belt in that order")
                if manifest.get("enabled_class") != {"no_seatbelt": "SEATBELT_MISSING"}:
                    errors.append("Seat-belt manifest may enable only no_seatbelt as SEATBELT_MISSING")
                input_spec = manifest.get("input") or {}
                if (
                    int(input_spec.get("width", -1)) != int(seatbelt.image_size)
                    or int(input_spec.get("height", -1)) != int(seatbelt.image_size)
                    or input_spec.get("color_order") != "RGB"
                    or input_spec.get("normalization") != "scale_0_1"
                ):
                    errors.append("Seat-belt model input contract does not match configured image size/RGB normalization")
                ncnn_spec = manifest.get("ncnn") or {}
                if int(ncnn_spec.get("image_size", -1)) != int(seatbelt.image_size):
                    errors.append("Seat-belt NCNN export image size does not match detector.image_size")
                params = list(model_path.glob("*.param"))
                bins = list(model_path.glob("*.bin"))
                if len(params) != 1 or len(bins) != 1:
                    errors.append("Seat-belt NCNN directory must contain exactly one .param and one .bin file")
                artifact_hashes = ncnn_spec.get("artifacts_sha256") or {}
                required_artifacts = {path.name for path in [*params, *bins]}
                if set(artifact_hashes) != required_artifacts:
                    errors.append("Seat-belt NCNN manifest checksum coverage is incomplete")
                actual_hashes = {}
                for filename, expected in artifact_hashes.items():
                    artifact = model_path / filename
                    actual = _sha256(artifact) if artifact.is_file() else None
                    actual_hashes[filename] = actual
                    if actual != expected:
                        errors.append(f"Seat-belt NCNN artifact checksum mismatch: {filename}")
                checks["seatbelt_ncnn_artifacts_sha256"] = actual_hashes
                source_artifact = model_path / str(manifest.get("source_artifact", ""))
                source_sha = manifest.get("source_sha256")
                if not source_artifact.is_file():
                    errors.append("Seat-belt source ONNX artifact is missing")
                elif not source_sha or _sha256(source_artifact) != source_sha:
                    errors.append("Seat-belt source ONNX checksum does not match its manifest")
                try:
                    import ncnn  # noqa: F401
                except ImportError:
                    errors.append("Seat-belt detector requires the ncnn Python package")
            except (OSError, ValueError, TypeError) as exc:
                errors.append(f"Seat-belt model manifest is invalid: {exc}")
    if errors:
        raise RuntimeError("Preflight failed:\n- " + "\n- ".join(errors))
    return checks
