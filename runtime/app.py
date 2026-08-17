"""Composition root only: calculations remain inside their domain modules."""

from __future__ import annotations

import argparse
import os
import time
import uuid
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version as package_version
from functools import lru_cache
from pathlib import Path

from dms_final_system.runtime.config import RuntimeConfig, load_config


SYSTEM_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=8)
def _package_version(name: str) -> str:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return "unavailable"


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else SYSTEM_ROOT / candidate


def _normalize_face_bbox(
    face_detected: bool,
    bbox_xyxy: tuple[float, float, float, float] | None,
    frame_width: int,
    frame_height: int,
) -> list[float] | None:
    """Convert the current pixel bbox to the dashboard's resolution-free contract."""
    if not face_detected or bbox_xyxy is None:
        return None
    x1, y1, x2, y2 = bbox_xyxy
    width = max(int(frame_width), 1)
    height = max(int(frame_height), 1)
    normalized = [
        max(0.0, min(1.0, float(x1) / width)),
        max(0.0, min(1.0, float(y1) / height)),
        max(0.0, min(1.0, float(x2) / width)),
        max(0.0, min(1.0, float(y2) / height)),
    ]
    return normalized if normalized[2] > normalized[0] and normalized[3] > normalized[1] else None


def _normalize_behavior_detections(
    detections,
    frame_width: int,
    frame_height: int,
    *,
    limit: int = 20,
) -> list[dict]:
    """Create a bounded, resolution-free snapshot for browser-side overlays."""
    result = []
    ordered = sorted(
        detections,
        key=lambda detection: float(detection.confidence),
        reverse=True,
    )
    for detection in ordered[: max(0, int(limit))]:
        bbox = _normalize_face_bbox(
            True,
            detection.xyxy,
            frame_width,
            frame_height,
        )
        if bbox is None:
            continue
        result.append(
            {
                "label": str(detection.label),
                "confidence": float(detection.confidence),
                "bbox_normalized": bbox,
            }
        )
    return result


def _eye_signal_blockers(signal, perception_config) -> list[str]:
    """Explain why the current frame cannot contribute to eye events.

    This is dashboard-only diagnostics.  EventEngine remains the authority for
    blink/closure decisions and none of its safety thresholds are bypassed.
    """
    blockers: list[str] = []
    if not bool(signal.face_detected):
        return ["NO_FACE"]
    if not bool(signal.eye_resolution_valid):
        blockers.append("EYES_TOO_SMALL")
    if float(signal.eye_signal_quality) < float(perception_config.min_eye_signal_quality):
        blockers.append("LOW_EYE_QUALITY")
    max_pose = max(abs(float(signal.pitch)), abs(float(signal.yaw)), abs(float(signal.roll)))
    if max_pose > float(perception_config.max_eye_pose_deg):
        blockers.append("HEAD_POSE_OUT_OF_RANGE")
    if not (
        float(signal.event_left_ear) > 0.0
        and float(signal.event_right_ear) > 0.0
    ):
        blockers.append("INVALID_EYE_GEOMETRY")
    if not blockers and not bool(signal.eye_signal_valid):
        blockers.append("EYE_SIGNAL_REJECTED")
    return blockers


def _apply_thread_caps(config: RuntimeConfig) -> None:
    omp_threads = str(int(config.omp_num_threads))
    os.environ["OMP_NUM_THREADS"] = omp_threads
    os.environ["OPENBLAS_NUM_THREADS"] = omp_threads
    os.environ["MKL_NUM_THREADS"] = omp_threads
    os.environ["NUMEXPR_NUM_THREADS"] = omp_threads
    os.environ["NCNN_NUM_THREADS"] = str(int(config.ncnn_num_threads))
    try:
        import cv2

        cv2.setNumThreads(int(config.cv_num_threads))
    except ImportError:
        pass


def _remote_api_snapshot(config: RuntimeConfig, client, source_kind: str) -> dict:
    """Expose why remote delivery is inactive as well as live client health."""
    if client is not None:
        return client.snapshot()

    api = config.safeemax_api
    reasons = []
    if not api.enabled:
        reasons.append("Disabled in runtime configuration")
    if not api.device_id.strip() or api.device_id.upper().startswith("REPLACE_"):
        reasons.append("device ID is not configured")
    if not api.vehicle.strip() or api.vehicle.upper().startswith("REPLACE_"):
        reasons.append("vehicle ID is not configured")
    if source_kind != "camera":
        reasons.append(f"suppressed for {source_kind} input")
    if not reasons:
        reasons.append("Sender did not start")
    reason = "; ".join(reasons)
    return {
        "enabled": False,
        "connection_status": "DISABLED" if not api.enabled else "SUPPRESSED",
        "endpoint_url": str(api.endpoint_url),
        "pending": 0,
        "delivered": 0,
        "duplicates": 0,
        "retries": 0,
        "rejected": 0,
        "queue_overflow": 0,
        "status_reason": reason,
        "last_error": "",
        "last_attempt_utc": None,
        "last_success_utc": None,
        "last_failure_utc": None,
        "last_http_status": None,
        "last_message_id": None,
        "last_message_type": None,
    }


def run(config_path: Path, replay_path: Path | None = None) -> None:
    config = load_config(config_path)
    _apply_thread_caps(config)

    from dms_final_system.runtime.preflight import run_preflight

    preflight = run_preflight(config, _resolve, replay=replay_path is not None)

    from dms_final_system.shared.contracts import AlarmLevel
    from dms_final_system.shared.feature_contract import RUNTIME_FEATURE_VERSION
    from dms_final_system.runtime.calibration import PersonalCalibrator
    from dms_final_system.runtime.alarm import (
        AlarmController,
        GPIOBuzzerOutput,
        NullAlarmOutput,
        PlatformAudioOutput,
    )
    from dms_final_system.runtime.capture import ReplayCapture, create_live_capture
    from dms_final_system.runtime.dashboard import DashboardProcess
    from dms_final_system.runtime.events import EventEngine
    from dms_final_system.runtime.evaluation import EvaluationSessionRecorder
    from dms_final_system.runtime.features import V3RuntimeFeatureBuilder
    from dms_final_system.runtime.fusion import FusionStateMachine
    from dms_final_system.runtime.hmi import VisualHMI
    from dms_final_system.runtime.integration import (
        EvidenceBus,
        SafeemaxDeviceClient,
        SafeemaxDevicePublisher,
        SafeemaxIncidentSink,
        SafeemaxTelemetryBatcher,
    )
    from dms_final_system.runtime.model.bundle import (
        assert_deployment_allowed,
        read_active_model,
        verify_bundle,
    )
    from dms_final_system.runtime.model.predictor import LightGBMRuntimePredictor
    from dms_final_system.runtime.model.drift import FeatureDriftMonitor
    from dms_final_system.runtime.objects import SeatbeltDetector, YoloBehaviorDetector
    from dms_final_system.runtime.perception import MediaPipeFacePerception
    from dms_final_system.runtime.monitoring import (
        RuntimeStatusStore,
        SafeModeController,
        StageTimingRegistry,
        SystemMetricsCollector,
        SystemMetricsSampler,
    )
    from dms_final_system.runtime.recording import (
        CompressedFrameHub,
        IncidentTriggerPolicy,
        RollingIncidentRecorder,
    )
    from dms_final_system.runtime.telemetry import LiveStatus, StructuredTelemetry

    session_id = f"{config.session_name}-{uuid.uuid4().hex[:8]}"
    active_descriptor = read_active_model(_resolve(config.active_model_path))
    bundle = Path(active_descriptor["bundle_dir"])
    bundle_status = verify_bundle(bundle)
    if active_descriptor["deployment_mode"] != config.deployment_mode:
        raise RuntimeError(
            "Deployment mode mismatch: "
            f"active_model={active_descriptor['deployment_mode']} config={config.deployment_mode}"
        )
    assert_deployment_allowed(bundle_status["status"], config.deployment_mode)
    predictor = LightGBMRuntimePredictor(bundle, verify=False)
    golden = predictor.verify_golden_samples()
    evaluation_root = _resolve(config.evaluation.output_dir)
    telemetry_dir = (
        evaluation_root / session_id if config.evaluation.enabled
        else _resolve(config.telemetry.log_dir)
    )
    telemetry = StructuredTelemetry(
        telemetry_dir, session_id, config.telemetry.mode,
        0 if config.evaluation.enabled else config.telemetry.rotate_bytes,
        0 if config.evaluation.enabled else config.telemetry.backups,
        filename="telemetry.jsonl" if config.evaluation.enabled else None,
    )
    live = LiveStatus(config.telemetry.live_status_hz)
    telemetry.emit(
        "Startup", "model_verified",
        {
            **bundle_status,
            **golden,
            "deployment_mode": config.deployment_mode,
            "preflight": preflight,
        },
    )

    perception = MediaPipeFacePerception(
        _resolve(config.perception.face_landmarker_path),
        config.perception.min_detection_confidence,
        config.perception.min_tracking_confidence,
        config.perception.min_eye_signal_quality,
        config.perception.max_eye_pose_deg,
        config.perception.max_eye_asymmetry_ratio,
        config.perception.min_face_width_ratio,
        config.perception.min_interocular_distance_px,
        config.perception.min_eye_width_px,
        process_width=config.perception.process_width,
    )
    calibrator = PersonalCalibrator(
        config.calibration.target_sec, config.calibration.max_sec,
        config.calibration.minimum_good_frames, config.perception.min_face_quality,
        config.calibration.max_pose_deviation_deg,
        config.calibration.event_trim_low_quantile,
        config.calibration.event_trim_high_quantile,
        config.calibration.max_event_ear_cv,
        config.calibration.max_event_eye_asymmetry,
    )
    events = EventEngine(**asdict(config.events))
    feature_builder = V3RuntimeFeatureBuilder(config.perception.min_face_quality)
    fusion_values = asdict(config.fusion)
    fusion_values.pop("version", None)
    fusion = FusionStateMachine(**fusion_values)
    drift = FeatureDriftMonitor(bundle, **asdict(config.drift))
    evidence_bus = EvidenceBus()
    behavior_config = replace(
        config.behavior_detector,
        model_path=str(_resolve(config.behavior_detector.model_path)),
        manifest_path=str(_resolve(config.behavior_detector.manifest_path)),
        ncnn_num_threads=config.ncnn_num_threads,
    )
    seatbelt_config = replace(
        config.seatbelt_detector,
        model_path=str(_resolve(config.seatbelt_detector.model_path)),
        manifest_path=str(_resolve(config.seatbelt_detector.manifest_path)),
    )
    behavior_detector = None
    seatbelt_detector = None
    try:
        behavior_detector = YoloBehaviorDetector(
            behavior_config, evidence_bus, telemetry
        )
        seatbelt_detector = SeatbeltDetector(
            seatbelt_config, evidence_bus, telemetry
        )
    except Exception:
        if seatbelt_detector is not None:
            seatbelt_detector.close()
        if behavior_detector is not None:
            behavior_detector.close()
        perception.close()
        telemetry.close()
        raise
    stage_timings = StageTimingRegistry()
    status_store = RuntimeStatusStore()
    metrics_sampler = SystemMetricsSampler(
        SystemMetricsCollector(_resolve(config.recorder.output_dir)),
        config.power.sample_interval_sec,
    ).start()
    safe_mode_controller = SafeModeController(config.power)
    recorder = RollingIncidentRecorder(
        _resolve(config.recorder.output_dir), session_id, telemetry,
        pre_alert_sec=config.recorder.pre_alert_sec,
        post_alert_sec=config.recorder.post_alert_sec,
        merge_gap_sec=config.recorder.merge_gap_sec,
        max_clip_sec=config.recorder.max_clip_sec,
        fps=config.recorder.fps,
        jpeg_quality=config.recorder.jpeg_quality,
        bitrate=config.recorder.bitrate,
        queue_size=config.recorder.queue_size,
        codec=config.recorder.codec,
        require_copy_mux=config.recorder.require_copy_mux,
        reserve_free_bytes=config.recorder.reserve_free_bytes,
        delete_oldest_when_full=config.recorder.delete_oldest_when_full,
    ) if config.recorder.enabled else None
    incident_policy = IncidentTriggerPolicy(
        config.recorder.trigger_states,
        config.recorder.trigger_violations,
        suppress_yawn_only=config.recorder.suppress_yawn_only,
        episode_clear_sec=config.recorder.episode_clear_sec,
    )
    dashboard = DashboardProcess(
        config.dashboard,
        _resolve(config.recorder.output_dir),
        telemetry,
    ).start()
    frame_hub = (
        CompressedFrameHub(
            telemetry,
            jpeg_quality=config.recorder.jpeg_quality,
            queue_size=config.recorder.queue_size,
        )
        if recorder is not None or dashboard.enabled
        else None
    )
    if frame_hub is not None:
        if recorder is not None:
            frame_hub.subscribe(recorder.push_encoded)
        if dashboard.enabled:
            frame_hub.subscribe(dashboard.publish_preview)
    evaluation = EvaluationSessionRecorder(
        evaluation_root, session_id, telemetry,
        fps=config.evaluation.video_fps,
        queue_size=config.evaluation.queue_size,
        record_video=config.evaluation.record_full_session,
        metadata={
            "deployment_mode": config.deployment_mode,
            "model_version": predictor.model_version,
            "bundle_status": bundle_status["status"],
            "bundle_dir": str(bundle),
            "feature_version": RUNTIME_FEATURE_VERSION,
            "fusion_version": config.fusion.version,
            "eye_event_version": "eye-events-v2.4",
            "drift_reference_available": drift.available,
            "behavior_detector_config": asdict(config.behavior_detector),
            "seatbelt_detector_config": asdict(config.seatbelt_detector),
            "evaluation_config": asdict(config.evaluation),
        },
    ) if config.evaluation.enabled else None

    frame_sinks = []
    if frame_hub is not None:
        frame_sinks.append(frame_hub.push)
    if evaluation is not None:
        frame_sinks.append(evaluation.push)

    def publish_frame(packet) -> None:
        for sink in frame_sinks:
            sink(packet)

    if replay_path:
        capture = ReplayCapture(replay_path, config.camera.fps)
        source_kind = "replay"
    else:
        capture = create_live_capture(
            config.camera,
            frame_sink=publish_frame if frame_sinks else None,
        )
        source_kind = "camera"
    if evaluation:
        evaluation.update_metadata(
            source_kind=source_kind,
            replay_path=str(replay_path) if replay_path else None,
            camera_source=config.camera.source if not replay_path else None,
            camera_configuration=getattr(capture, "actual_configuration", {}),
            alarm_config=asdict(config.alarm),
            hmi_config=asdict(config.hmi),
        )
    audible_source = bool(
        source_kind == "camera"
        or (not config.alarm.live_camera_only and config.alarm.allow_replay_audio)
    )
    alarm_output = NullAlarmOutput()
    if config.alarm.enabled and config.alarm.mode == "LOCAL" and audible_source:
        if config.alarm.output == "gpio":
            alarm_output = GPIOBuzzerOutput(
                config.alarm.gpio_pin,
                buzzer_type=config.alarm.gpio_buzzer_type,
                pwm_frequency_hz=config.alarm.gpio_pwm_frequency_hz,
                master_gain=config.alarm.master_gain,
            )
        else:
            alarm_output = PlatformAudioOutput(
                config.alarm.backend,
                config.alarm.device,
                config.alarm.master_gain,
            )
    try:
        alarm = AlarmController(
            config.alarm, telemetry, alarm_output,
            audible=audible_source,
        )
    except Exception:
        capture.close()
        perception.close()
        behavior_detector.close()
        if frame_hub:
            frame_hub.close()
        if recorder:
            recorder.close()
        if evaluation:
            evaluation.close()
        dashboard.close()
        metrics_sampler.close()
        telemetry.close()
        raise
    safeemax_client = None
    safeemax_publisher = None
    safeemax_telemetry_batcher = None
    if config.safeemax_api.enabled and source_kind == "camera":
        api_config = config.safeemax_api
        try:
            auth_token = (
                os.environ.get(api_config.auth_token_env, "")
                if api_config.auth_token_env else ""
            )
            if api_config.auth_token_env and not auth_token:
                raise ValueError(
                    f"Device API token environment variable {api_config.auth_token_env!r} is empty"
                )
            safeemax_client = SafeemaxDeviceClient(
                api_config.endpoint_url,
                _resolve(api_config.outbox_dir),
                telemetry,
                auth_token=auth_token,
                request_timeout_sec=api_config.request_timeout_sec,
                retry_initial_sec=api_config.retry_initial_sec,
                retry_max_sec=api_config.retry_max_sec,
                queue_size=api_config.queue_size,
            )
            safeemax_publisher = SafeemaxDevicePublisher(
                safeemax_client,
                device_id=api_config.device_id,
                vehicle=api_config.vehicle,
                driver=api_config.driver,
                location=api_config.location,
                battery=api_config.battery,
                event_states=api_config.event_states,
                event_violations=api_config.event_violations,
                status_interval_sec=api_config.status_interval_sec,
            )
            if recorder is not None and api_config.incident_upload_enabled:
                recorder.sink = SafeemaxIncidentSink(safeemax_publisher)
            if api_config.telemetry_upload_enabled:
                safeemax_telemetry_batcher = SafeemaxTelemetryBatcher(
                    telemetry,
                    safeemax_publisher,
                    batch_size=api_config.telemetry_batch_size,
                    flush_interval_sec=api_config.telemetry_flush_interval_sec,
                    queue_size=api_config.telemetry_queue_size,
                )
        except Exception:
            if safeemax_telemetry_batcher:
                safeemax_telemetry_batcher.close()
            if safeemax_client:
                safeemax_client.close()
            alarm.close()
            capture.close()
            perception.close()
            behavior_detector.close()
            seatbelt_detector.close()
            if frame_hub:
                frame_hub.close()
            if recorder:
                recorder.close()
            if evaluation:
                evaluation.close()
            dashboard.close()
            metrics_sampler.close()
            telemetry.close()
            raise
    elif config.safeemax_api.enabled:
        telemetry.emit(
            "SafeemaxAPI",
            "replay_delivery_suppressed",
            {"source": source_kind},
            level="WARNING",
        )
    hmi = VisualHMI(config.hmi, telemetry, enabled=source_kind == "camera")
    telemetry.emit(
        "Startup", "self_test_passed",
        {
            "source": source_kind, "bundle": str(bundle),
            "model_status": bundle_status["status"],
            "deployment_mode": config.deployment_mode,
            "evaluation_enabled": config.evaluation.enabled,
            "safeemax_api_enabled": safeemax_publisher is not None,
        },
    )

    last_inference = last_health = last_stage_log = last_status = float("-inf")
    last_health_sample = None
    health = metrics_sampler.snapshot()
    safe_mode_state = safe_mode_controller.update(time.monotonic(), health)
    last_prediction = None
    last_event_snapshot = None
    last_drift = None
    processed = 0
    wall_started = time.monotonic()
    last_yolo_frame_id = -1
    last_seatbelt_frame_id = -1
    yolo_inferences = 0
    seatbelt_inferences = 0
    try:
        while True:
            try:
                packet = capture.read(timeout=2.0)
            except EOFError:
                break
            if source_kind == "replay":
                publish_frame(packet)
            perception_started = time.perf_counter()
            signal = perception.process(packet)
            perception_ms = (time.perf_counter() - perception_started) * 1000
            stage_timings.record("perception", perception_ms)
            behavior_detector.submit(
                packet,
                face_bbox_xyxy=perception.latest_face_bbox_xyxy,
            )
            seatbelt_detector.submit(
                packet,
                face_bbox_xyxy=perception.latest_face_bbox_xyxy,
            )
            processed += 1
            event_emitted = False
            calibration = calibrator.update(signal)
            if calibrator.ready:
                calibrator.apply(signal)
                last_event_snapshot = events.update(
                    signal,
                    calibrator.result.mar_baseline or 0.1,
                    calibrator.result.pitch_baseline or 0.0,
                )
                feature_builder.update(signal)
                fusion.add_evidence(last_event_snapshot.evidence)
                event_emitted = bool(last_event_snapshot.evidence)
            fusion.add_evidence(evidence_bus.drain())

            if packet.monotonic_sec - last_inference >= config.inference_interval_sec and calibrator.ready:
                last_inference = packet.monotonic_sec
                snapshot = feature_builder.build(predictor.features, packet.monotonic_sec)
                telemetry.emit(
                    "Window", "feature_snapshot",
                    {
                        "coverage": snapshot.coverage, "valid": snapshot.valid,
                        "reason": snapshot.reason,
                        "features": dict(zip(snapshot.feature_names, snapshot.ordered_features)) if config.telemetry.mode.upper() == "DEBUG" and snapshot.valid else None,
                    },
                    monotonic_sec=packet.monotonic_sec, frame_id=packet.frame_id, window_id=snapshot.window_id,
                )
                if snapshot.valid:
                    last_drift = drift.evaluate(snapshot)
                    telemetry.emit(
                        "Model", "feature_drift", last_drift.to_dict(),
                        monotonic_sec=packet.monotonic_sec,
                        frame_id=packet.frame_id,
                        window_id=snapshot.window_id,
                        level="WARNING" if last_drift.reason == "MODEL_INPUT_OOD" else "INFO",
                    )
                    inference_started = time.perf_counter()
                    last_prediction = predictor.predict(snapshot)
                    inference_ms = (time.perf_counter() - inference_started) * 1000
                    stage_timings.record("lightgbm", inference_ms)
                    telemetry.emit(
                        "Model", "prediction",
                        {"raw_probability": last_prediction.raw_score,
                         "calibrated_probability": last_prediction.calibrated_probability,
                         "threshold": last_prediction.threshold,
                         "model_operating_positive": last_prediction.positive,
                         "note": "operating threshold is telemetry; Fusion thresholds drive states",
                         "inference_ms": inference_ms,
                         "model_version": last_prediction.model_version,
                         "model_contribution_enabled": last_drift.model_contribution_enabled,
                         "drift_reason": last_drift.reason},
                        monotonic_sec=packet.monotonic_sec, frame_id=packet.frame_id, window_id=snapshot.window_id,
                    )

            prediction_age = (
                packet.monotonic_sec - last_prediction.monotonic_sec
                if last_prediction is not None else float("inf")
            )
            model_fresh = prediction_age <= max(0.7, 3.0 * config.inference_interval_sec)
            eye_evidence_trustworthy = bool(
                calibrator.ready
                and last_event_snapshot is not None
                and last_event_snapshot.eye_evidence_trustworthy
            )
            # Calibration readiness is a system-level gate. Short face/eye
            # quality gaps are intentionally resolved by EventEngine's
            # VALID/GRACE/LOST state instead of bypassing its grace period here.
            trustworthy = bool(calibrator.ready and last_event_snapshot is not None)
            model_contribution_enabled = bool(
                model_fresh
                and (last_drift is None or last_drift.model_contribution_enabled)
            )
            fusion_started = time.perf_counter()
            decision = fusion.update(
                packet.monotonic_sec,
                last_prediction,
                last_event_snapshot,
                trustworthy,
                model_contribution_enabled=model_contribution_enabled,
                eye_evidence_trustworthy=eye_evidence_trustworthy,
            )
            fusion_ms = (time.perf_counter() - fusion_started) * 1000
            stage_timings.record("fusion", fusion_ms)
            incident_id = None
            alarm_status = alarm.snapshot()
            behavior_snapshot = behavior_detector.snapshot()
            seatbelt_snapshot = seatbelt_detector.snapshot()
            if behavior_snapshot.frame_id != last_yolo_frame_id and behavior_snapshot.frame_id >= 0:
                last_yolo_frame_id = behavior_snapshot.frame_id
                yolo_inferences += 1
                stage_timings.record("yolo", behavior_snapshot.inference_ms)
            if seatbelt_snapshot.frame_id != last_seatbelt_frame_id and seatbelt_snapshot.frame_id >= 0:
                last_seatbelt_frame_id = seatbelt_snapshot.frame_id
                seatbelt_inferences += 1
                stage_timings.record("seatbelt", seatbelt_snapshot.inference_ms)
            incident_policy_result = incident_policy.evaluate(
                packet.monotonic_sec,
                decision,
                recording_active=bool(recorder and recorder.active is not None),
            )
            if recorder and incident_policy_result.qualifies:
                probability = decision.smoothed_probability or 0.0
                incident_reasons = list(
                    dict.fromkeys([*decision.state_entry_reason, *decision.current_reason_codes])
                )
                incident_id = recorder.trigger(
                    packet.monotonic_sec, decision.driver_state.value, decision.violations,
                    incident_reasons, probability,
                    {
                        **(last_event_snapshot.to_dict() if last_event_snapshot else {}),
                        "active_state": decision.driver_state.value,
                        "target_state": (
                            decision.target_state.value
                            if decision.target_state else decision.driver_state.value
                        ),
                        "state_entry_reason": decision.state_entry_reason,
                        "current_reason_codes": decision.current_reason_codes,
                        "eye_observation_status": decision.eye_observation_status,
                        "recovery_status": decision.recovery_status,
                        "model_risk_pending": decision.model_risk_pending,
                        "model_contribution_enabled": decision.model_contribution_enabled,
                        "model_version": predictor.model_version,
                        "feature_version": RUNTIME_FEATURE_VERSION,
                        "fusion_version": config.fusion.version,
                        "alarm_status": alarm_status.to_dict(),
                        "behavior_detector": behavior_snapshot.to_dict(),
                        "seatbelt_detector": seatbelt_snapshot.to_dict(),
                    },
                    start_new=incident_policy_result.onset,
                    trigger_kinds=incident_policy_result.trigger_kinds,
                )
            alarm_status = alarm.update(
                packet.monotonic_sec,
                decision,
                last_event_snapshot,
                incident_id=incident_id,
            )
            if safeemax_publisher is not None:
                try:
                    safeemax_publisher.publish_decision(
                        packet, decision, predictor.model_version
                    )
                except Exception as exc:
                    telemetry.emit(
                        "SafeemaxAPI",
                        "event_queue_failed",
                        {"error": repr(exc)},
                        monotonic_sec=packet.monotonic_sec,
                        frame_id=packet.frame_id,
                        level="ERROR",
                    )
            hmi.push(
                packet.frame,
                decision,
                alarm_status,
                behavior_snapshot,
                face_signal=signal,
            )
            if (
                config.deployment_mode == "SHADOW"
                and (decision.changed or event_emitted)
                and decision.alarm_level in {AlarmLevel.WARNING, AlarmLevel.CRITICAL}
            ):
                telemetry.emit(
                    "Deployment", "shadow_alarm_suppressed",
                    {
                        "driver_state": decision.driver_state.value,
                        "alarm_level": decision.alarm_level.value,
                        "reason_codes": decision.reason_codes,
                    },
                    monotonic_sec=packet.monotonic_sec,
                    frame_id=packet.frame_id,
                    window_id=last_prediction.window_id if last_prediction else None,
                    incident_id=incident_id,
                )

            if (
                decision.changed
                or event_emitted
                or config.telemetry.mode.upper() == "DEBUG"
                or packet.monotonic_sec - last_stage_log >= 0.5
            ):
                last_stage_log = packet.monotonic_sec
                telemetry.emit(
                    "Perception", "face_signal",
                    {"detected": signal.face_detected, "quality": signal.face_quality,
                     "model_ear": signal.ear, "model_relative_ear": signal.relative_ear,
                     "ear": signal.ear, "relative_ear": signal.relative_ear,
                     "event_left_ear": signal.event_left_ear,
                     "event_right_ear": signal.event_right_ear,
                     "event_ear": signal.event_ear,
                     "event_left_relative_ear": signal.event_left_relative_ear,
                     "event_right_relative_ear": signal.event_right_relative_ear,
                     "event_relative_ear": signal.event_relative_ear,
                     "eye_signal_valid": signal.eye_signal_valid,
                     "eye_signal_quality": signal.eye_signal_quality,
                     "face_width_px": signal.face_width_px,
                     "face_height_px": signal.face_height_px,
                     "face_width_ratio": signal.face_width_ratio,
                     "interocular_distance_px": signal.interocular_distance_px,
                     "left_eye_width_px": signal.left_eye_width_px,
                     "right_eye_width_px": signal.right_eye_width_px,
                     "eye_resolution_valid": signal.eye_resolution_valid,
                     "driver_distance_status": signal.driver_distance_status,
                     "binocular_consistent": signal.binocular_consistent,
                     "mar": signal.mar,
                     "pitch": signal.pitch, "yaw": signal.yaw, "roll": signal.roll,
                     "gaze_x": signal.gaze_x, "gaze_y": signal.gaze_y,
                     "latency_ms": perception_ms},
                    monotonic_sec=packet.monotonic_sec, frame_id=packet.frame_id, incident_id=incident_id,
                )
                telemetry.emit(
                    "Calibration", "status", calibration.to_dict(),
                    monotonic_sec=packet.monotonic_sec, frame_id=packet.frame_id, incident_id=incident_id,
                )
                if last_event_snapshot:
                    telemetry.emit(
                        "Events", "snapshot", last_event_snapshot.to_dict(),
                        monotonic_sec=packet.monotonic_sec, frame_id=packet.frame_id, incident_id=incident_id,
                    )
                telemetry.emit(
                    "Fusion", "decision", decision.to_dict(),
                    monotonic_sec=packet.monotonic_sec, frame_id=packet.frame_id,
                    window_id=last_prediction.window_id if last_prediction else None, incident_id=incident_id,
                )
                telemetry.emit(
                    "Fusion", "latency",
                    {"latency_ms": fusion_ms, "prediction_age_sec": prediction_age if last_prediction else None,
                     "model_fresh": model_fresh},
                    monotonic_sec=packet.monotonic_sec, frame_id=packet.frame_id,
                    window_id=last_prediction.window_id if last_prediction else None, incident_id=incident_id,
                )

            elapsed = max(time.monotonic() - wall_started, 1e-6)
            effective_fps = processed / elapsed
            live.show(
                packet.monotonic_sec,
                state=decision.driver_state.value,
                target=decision.target_state.value if decision.target_state else decision.driver_state.value,
                recovery=decision.recovery_status,
                eye_obs=decision.eye_observation_status,
                p=f"{(decision.smoothed_probability or 0):.2f}",
                model_pending=int(decision.model_risk_pending),
                model_on=int(decision.model_contribution_enabled),
                alarm=alarm_status.active_pattern,
                audio=("ACK" if alarm_status.acknowledged else ("PLAY" if alarm_status.audio_playing else alarm_status.backend_health)),
                drift=(last_drift.reason if last_drift else "PENDING"),
                face=f"{signal.face_quality:.2f}",
                eye=(last_event_snapshot.eye_state if last_event_snapshot else "UNKNOWN"),
                event_ear=f"{signal.event_relative_ear:.2f}",
                closure=f"{(last_event_snapshot.current_closure_sec if last_event_snapshot else 0):.2f}",
                perclos=(
                    "N/A" if not last_event_snapshot or last_event_snapshot.perclos_30s is None
                    else f"{last_event_snapshot.perclos_30s:.2f}"
                ),
                blink=last_event_snapshot.blink_count_60s if last_event_snapshot else 0,
                yawn=last_event_snapshot.yawn_count_60s if last_event_snapshot else 0,
                objects=(",".join(behavior_snapshot.active_behaviors) or "NONE"),
                object_ms=f"{behavior_snapshot.inference_ms:.0f}",
                distance=signal.driver_distance_status,
                eye_px=f"{min(signal.left_eye_width_px, signal.right_eye_width_px):.0f}",
                fps=f"{effective_fps:.1f}",
                dropped=getattr(capture, "dropped_frames", 0),
            )
            latest_health = metrics_sampler.snapshot()
            sampled_at = latest_health.get("sampled_monotonic_sec")
            if sampled_at is not None and sampled_at != last_health_sample:
                last_health_sample = sampled_at
                health = latest_health
                safe_mode_state = safe_mode_controller.update(packet.monotonic_sec, health)
                behavior_detector.set_safe_mode(
                    safe_mode_state.enabled,
                    config.power.safe_yolo_interval_sec,
                )
                dashboard.set_safe_mode(safe_mode_state.enabled)
                if safe_mode_state.changed:
                    telemetry.emit(
                        "Power",
                        "safe_mode_changed",
                        {
                            "enabled": safe_mode_state.enabled,
                            "reason": safe_mode_state.reason,
                            "temperature_c": health.get("temperature_c"),
                            "throttled": health.get("throttled"),
                        },
                        monotonic_sec=packet.monotonic_sec,
                        level="WARNING" if safe_mode_state.enabled else "INFO",
                    )

            status_interval = 1.0 / max(config.dashboard.status_hz, 0.1)
            if packet.monotonic_sec - last_status >= status_interval:
                last_status = packet.monotonic_sec
                camera_configuration = getattr(capture, "actual_configuration", {})
                frame_height, frame_width = packet.frame.shape[:2]
                face_bbox_normalized = _normalize_face_bbox(
                    signal.face_detected,
                    perception.latest_face_bbox_xyxy,
                    frame_width,
                    frame_height,
                )
                behavior_detections = _normalize_behavior_detections(
                    behavior_snapshot.detections,
                    frame_width,
                    frame_height,
                )
                status = status_store.update(
                    runtime={
                        "status": "RUNNING",
                        "session_id": session_id,
                        "deployment_mode": config.deployment_mode,
                        "uptime_sec": elapsed,
                    },
                    driver={
                        "state": decision.driver_state.value,
                        "target_state": (
                            decision.target_state.value
                            if decision.target_state else decision.driver_state.value
                        ),
                        "violations": list(decision.violations),
                        "probability": decision.smoothed_probability,
                        "reason_codes": list(decision.reason_codes),
                        "alarm": alarm_status.to_dict(),
                        "face_detected": signal.face_detected,
                        "face_quality": signal.face_quality,
                        "face_bbox_normalized": face_bbox_normalized,
                        "face_bbox_frame_id": perception.latest_face_bbox_frame_id,
                        "driver_distance_status": signal.driver_distance_status,
                        "blink_count_60s": (
                            last_event_snapshot.blink_count_60s
                            if last_event_snapshot else 0
                        ),
                        "eye_state": (
                            last_event_snapshot.eye_state
                            if last_event_snapshot else "CALIBRATING"
                        ),
                        "eye_observation_status": (
                            last_event_snapshot.eye_observation_status
                            if last_event_snapshot else "CALIBRATING"
                        ),
                        "eye_signal_valid": bool(signal.eye_signal_valid),
                        "eye_signal_quality": signal.eye_signal_quality,
                        "eye_signal_quality_threshold": (
                            config.perception.min_eye_signal_quality
                        ),
                        "eye_signal_blockers": _eye_signal_blockers(
                            signal, config.perception
                        ),
                        "perception_process_width": config.perception.process_width,
                        "eye_resolution_valid": bool(signal.eye_resolution_valid),
                        "interocular_distance_px": signal.interocular_distance_px,
                        "minimum_eye_width_px": min(
                            signal.left_eye_width_px, signal.right_eye_width_px
                        ),
                        "pitch": signal.pitch,
                        "yaw": signal.yaw,
                        "roll": signal.roll,
                        "max_eye_pose_deg": config.perception.max_eye_pose_deg,
                        "event_relative_ear": signal.event_relative_ear,
                        "event_left_relative_ear": signal.event_left_relative_ear,
                        "event_right_relative_ear": signal.event_right_relative_ear,
                        "calibration": calibration.to_dict(),
                    },
                    behavior={
                        "frame_id": behavior_snapshot.frame_id,
                        "active_behaviors": list(behavior_snapshot.active_behaviors),
                        "class_ratios": dict(behavior_snapshot.class_ratios),
                        "detections": behavior_detections,
                        "health": behavior_snapshot.health,
                        "last_error": behavior_snapshot.last_error,
                        "raw_confidence_threshold": (
                            config.behavior_detector.raw_confidence_threshold
                        ),
                        "effective_temporal_window_sec": (
                            behavior_detector.effective_temporal_window_sec
                        ),
                        "class_thresholds": dict(
                            config.behavior_detector.class_thresholds
                        ),
                    },
                    camera={
                        "backend": camera_configuration.get("backend", source_kind),
                        "width": camera_configuration.get("width", config.camera.width),
                        "height": camera_configuration.get("height", config.camera.height),
                        "requested_fps": config.camera.fps,
                        "actual_configuration": camera_configuration,
                        "dropped_frames": getattr(capture, "dropped_frames", 0),
                        "capture_errors": getattr(capture, "capture_errors", 0),
                        "last_error": getattr(capture, "last_error", ""),
                    },
                    performance={
                        "effective_fps": effective_fps,
                        "processed_frames": processed,
                        "capture_frames": getattr(capture, "captured_frames", processed),
                        "encoded_frames": frame_hub.encoded_frames if frame_hub else 0,
                        "capture_fps": getattr(capture, "captured_frames", processed) / elapsed,
                        "ai_fps": effective_fps,
                        "yolo_fps": yolo_inferences / elapsed,
                        "seatbelt_fps": seatbelt_inferences / elapsed,
                        "yolo_interval_sec": behavior_detector.effective_interval_sec,
                        "recorder_fps": (
                            frame_hub.encoded_frames / elapsed if frame_hub else 0.0
                        ),
                        "preview_fps_limit": (
                            config.dashboard.safe_preview_fps
                            if safe_mode_state.enabled else config.dashboard.preview_fps
                        ),
                        "preview_fps": dashboard.published_preview / elapsed,
                        "latency": stage_timings.snapshot(),
                        "decision_staleness_sec": prediction_age if last_prediction else None,
                        "queues": {
                            "capture": getattr(getattr(capture, "queue", None), "qsize", lambda: 0)(),
                            "behavior": behavior_detector.queue.qsize(),
                            "seatbelt": seatbelt_detector.queue.qsize(),
                            "frame_hub": frame_hub.queue.qsize() if frame_hub else 0,
                            "recorder": recorder.queue.qsize() if recorder else 0,
                            "evaluation": evaluation.queue.qsize() if evaluation else 0,
                        },
                    },
                    system=health,
                    power={
                        "safe_mode": safe_mode_state.enabled,
                        "reason": safe_mode_state.reason,
                        "entered_monotonic_sec": safe_mode_state.entered_monotonic_sec,
                        "configured_cpu_max_mhz": config.power.cpu_max_mhz,
                        "throttled": health.get("throttled", {}),
                    },
                    models={
                        "drowsiness": predictor.model_version,
                        "drowsiness_backend": f"LightGBM {_package_version('lightgbm')}",
                        "feature_version": RUNTIME_FEATURE_VERSION,
                        "fusion_version": config.fusion.version,
                        "mediapipe": f"MediaPipe {_package_version('mediapipe')} face-landmarker",
                        "behavior": behavior_snapshot.model_version,
                        "behavior_backend": behavior_snapshot.backend,
                        "behavior_health": behavior_snapshot.health,
                        "behavior_runtime_version": (
                            _package_version("ncnn")
                            if behavior_snapshot.backend.startswith("ncnn")
                            else behavior_snapshot.backend
                        ),
                        "ncnn_threads": config.ncnn_num_threads,
                        "behavior_exporter_versions": preflight.get(
                            "ncnn_exporter_versions", {}
                        ),
                        "seatbelt": seatbelt_snapshot.model_version,
                        "seatbelt_backend": seatbelt_snapshot.backend,
                        "seatbelt_health": seatbelt_snapshot.health,
                        "seatbelt_shadow_mode": seatbelt_snapshot.shadow_mode,
                        "commercial_license_status": "BLOCKED_PENDING_SAFEE_APPROVAL",
                        "drift": last_drift.reason if last_drift else "PENDING",
                        "model_contribution_enabled": model_contribution_enabled,
                    },
                    recording={
                        "enabled": recorder is not None,
                        "active_incident": recorder.active.incident_id if recorder and recorder.active else None,
                        "trigger_qualifies": incident_policy_result.qualifies,
                        "trigger_onset": incident_policy_result.onset,
                        "trigger_kinds": list(incident_policy_result.trigger_kinds),
                        "policy_episode_active": incident_policy.episode_active,
                        "policy_highest_priority": incident_policy.highest_priority,
                        "incident_count": recorder.incident_count if recorder else 0,
                        "finalized_incidents": recorder.finalized_incidents if recorder else 0,
                        "deleted_incidents": recorder.deleted_incidents if recorder else 0,
                        "dropped_frames": recorder.dropped_frames if recorder else 0,
                        "finalize_queue_depth": recorder.finalize_queue.qsize() if recorder else 0,
                        "finalize_dropped": recorder.finalize_dropped if recorder else 0,
                        "last_started_incident_id": recorder.last_started_incident_id if recorder else None,
                        "last_finalized_incident_id": recorder.last_finalized_incident_id if recorder else None,
                        "last_error": recorder.last_error if recorder else "",
                        "hub_dropped_frames": frame_hub.dropped_frames if frame_hub else 0,
                        "encoder": "ffmpeg_mjpeg_copy" if recorder else "disabled",
                    },
                    seatbelt={
                        **seatbelt_snapshot.to_dict(),
                        "no_seatbelt_threshold": (
                            config.seatbelt_detector.no_seatbelt_threshold
                        ),
                        "clear_threshold": config.seatbelt_detector.clear_threshold,
                    },
                    dashboard={
                        "healthy": dashboard.healthy,
                        "clients": dashboard.active_clients,
                        "dropped_status": dashboard.dropped_status,
                        "dropped_preview": dashboard.dropped_preview,
                        "published_preview": dashboard.published_preview,
                    },
                    remote_api=_remote_api_snapshot(
                        config, safeemax_client, source_kind
                    ),
                )
                dashboard.publish_status(status)
                if safeemax_publisher is not None:
                    try:
                        safeemax_publisher.publish_status(
                            status,
                            monotonic_sec=packet.monotonic_sec,
                        )
                    except Exception as exc:
                        telemetry.emit(
                            "SafeemaxAPI",
                            "status_queue_failed",
                            {"error": repr(exc)},
                            monotonic_sec=packet.monotonic_sec,
                            level="ERROR",
                        )

            if packet.monotonic_sec - last_health >= 5.0:
                last_health = packet.monotonic_sec
                telemetry.emit(
                    "Health", "status",
                    {**health, "effective_fps": effective_fps,
                     "capture_dropped": getattr(capture, "dropped_frames", 0),
                     "capture_queue_depth": getattr(getattr(capture, "queue", None), "qsize", lambda: 0)(),
                     "recorder_dropped": recorder.dropped_frames if recorder else 0,
                     "recorder_queue_depth": recorder.queue.qsize() if recorder else 0,
                     "evaluation_dropped": evaluation.dropped_frames if evaluation else 0,
                     "evaluation_queue_depth": evaluation.queue.qsize() if evaluation else 0,
                     "prediction_age_sec": prediction_age if last_prediction else None,
                     "model_contribution_enabled": model_contribution_enabled,
                     "drift_reason": last_drift.reason if last_drift else "PENDING",
                     "eye_signal_valid": eye_evidence_trustworthy,
                     "eye_observation_status": decision.eye_observation_status,
                     "recovery_status": decision.recovery_status,
                     "alarm_status": alarm_status.to_dict(),
                     "behavior_detector": behavior_snapshot.to_dict(),
                     "seatbelt_detector": seatbelt_snapshot.to_dict(),
                     "safe_mode": safe_mode_state.enabled,
                     "safe_mode_reason": safe_mode_state.reason,
                     "stage_latencies": stage_timings.snapshot(),
                     "frame_hub_dropped": frame_hub.dropped_frames if frame_hub else 0,
                     "dashboard_healthy": dashboard.healthy,
                     "safeemax_api": _remote_api_snapshot(
                         config, safeemax_client, source_kind
                     ),
                     "telemetry_dropped": telemetry.dropped_records},
                    monotonic_sec=packet.monotonic_sec,
                )
    except KeyboardInterrupt:
        telemetry.emit("Runtime", "keyboard_interrupt")
    finally:
        print()
        capture.close()
        perception.close()
        behavior_detector.close()
        seatbelt_detector.close()
        hmi.close()
        alarm.close()
        if frame_hub:
            frame_hub.close()
        if recorder:
            recorder.close()
        if evaluation:
            evaluation.close()
        dashboard.close()
        metrics_sampler.close()
        if safeemax_telemetry_batcher:
            safeemax_telemetry_batcher.close()
        if safeemax_client:
            safeemax_client.close()
        telemetry.emit("Runtime", "session_finished", {"processed_frames": processed})
        telemetry.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final Raspberry DMS pipeline")
    parser.add_argument("--config", type=Path, default=SYSTEM_ROOT / "configs/runtime.example.json")
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--verify-bundle", type=Path)
    args = parser.parse_args()
    if args.verify_bundle:
        _apply_thread_caps(RuntimeConfig())
        from dms_final_system.runtime.model.bundle import verify_bundle
        from dms_final_system.runtime.model.predictor import LightGBMRuntimePredictor

        info = verify_bundle(args.verify_bundle)
        predictor = LightGBMRuntimePredictor(args.verify_bundle, verify=False)
        print({**info, **predictor.verify_golden_samples()})
        return
    run(args.config, args.replay)


if __name__ == "__main__":
    main()
