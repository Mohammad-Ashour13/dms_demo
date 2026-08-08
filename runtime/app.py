"""Composition root only: calculations remain inside their domain modules."""

from __future__ import annotations

import argparse
import os
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path

from dms_final_system.runtime.config import RuntimeConfig, load_config


SYSTEM_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else SYSTEM_ROOT / candidate


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


def run(config_path: Path, replay_path: Path | None = None) -> None:
    config = load_config(config_path)
    _apply_thread_caps(config)

    from dms_final_system.shared.contracts import AlarmLevel
    from dms_final_system.shared.feature_contract import RUNTIME_FEATURE_VERSION
    from dms_final_system.runtime.calibration import PersonalCalibrator
    from dms_final_system.runtime.alarm import AlarmController, LinuxAudioOutput, NullAlarmOutput
    from dms_final_system.runtime.capture import LatestFrameCapture, ReplayCapture
    from dms_final_system.runtime.events import EventEngine
    from dms_final_system.runtime.evaluation import EvaluationSessionRecorder
    from dms_final_system.runtime.features import V3RuntimeFeatureBuilder
    from dms_final_system.runtime.fusion import FusionStateMachine
    from dms_final_system.runtime.hmi import VisualHMI
    from dms_final_system.runtime.integration import EvidenceBus
    from dms_final_system.runtime.model.bundle import (
        assert_deployment_allowed,
        read_active_model,
        verify_bundle,
    )
    from dms_final_system.runtime.model.predictor import LightGBMRuntimePredictor
    from dms_final_system.runtime.model.drift import FeatureDriftMonitor
    from dms_final_system.runtime.objects import YoloBehaviorDetector
    from dms_final_system.runtime.perception import MediaPipeFacePerception
    from dms_final_system.runtime.recording import RollingIncidentRecorder
    from dms_final_system.runtime.telemetry import LiveStatus, StructuredTelemetry
    from dms_final_system.runtime.telemetry.structured import system_health

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
        {**bundle_status, **golden, "deployment_mode": config.deployment_mode},
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
    try:
        behavior_detector = YoloBehaviorDetector(
            behavior_config, evidence_bus, telemetry
        )
    except Exception:
        perception.close()
        telemetry.close()
        raise
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
    ) if config.recorder.enabled else None
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
            "evaluation_config": asdict(config.evaluation),
        },
    ) if config.evaluation.enabled else None

    frame_sinks = [item.push for item in (recorder, evaluation) if item is not None]

    def publish_frame(packet) -> None:
        for sink in frame_sinks:
            sink(packet)

    if replay_path:
        capture = ReplayCapture(replay_path, config.camera.fps)
        source_kind = "replay"
    else:
        capture = LatestFrameCapture(
            config.camera.source, config.camera.width, config.camera.height,
            config.camera.fps, config.camera.ai_queue_size,
            frame_sink=publish_frame if frame_sinks else None,
        ).start()
        source_kind = "camera"
    if evaluation:
        evaluation.update_metadata(
            source_kind=source_kind,
            replay_path=str(replay_path) if replay_path else None,
            camera_source=config.camera.source if not replay_path else None,
            alarm_config=asdict(config.alarm),
            hmi_config=asdict(config.hmi),
        )
    audible_source = bool(
        source_kind == "camera"
        or (not config.alarm.live_camera_only and config.alarm.allow_replay_audio)
    )
    alarm_output = (
        LinuxAudioOutput(config.alarm.backend, config.alarm.device, config.alarm.master_gain)
        if config.alarm.enabled and config.alarm.mode == "LOCAL" and audible_source
        else NullAlarmOutput()
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
        if recorder:
            recorder.close()
        if evaluation:
            evaluation.close()
        telemetry.close()
        raise
    hmi = VisualHMI(config.hmi, telemetry, enabled=source_kind == "camera")
    telemetry.emit(
        "Startup", "self_test_passed",
        {
            "source": source_kind, "bundle": str(bundle),
            "model_status": bundle_status["status"],
            "deployment_mode": config.deployment_mode,
            "evaluation_enabled": config.evaluation.enabled,
        },
    )

    last_inference = last_health = last_stage_log = float("-inf")
    last_prediction = None
    last_event_snapshot = None
    last_drift = None
    processed = 0
    wall_started = time.monotonic()
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
            behavior_detector.submit(
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
            incident_id = None
            alarm_status = alarm.snapshot()
            behavior_snapshot = behavior_detector.snapshot()
            if recorder and decision.alarm_level in {AlarmLevel.WARNING, AlarmLevel.CRITICAL}:
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
                    },
                )
            alarm_status = alarm.update(
                packet.monotonic_sec,
                decision,
                last_event_snapshot,
                incident_id=incident_id,
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
            if packet.monotonic_sec - last_health >= 5.0:
                last_health = packet.monotonic_sec
                telemetry.emit(
                    "Health", "status",
                    {**system_health(), "effective_fps": effective_fps,
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
        hmi.close()
        alarm.close()
        if recorder:
            recorder.close()
        if evaluation:
            evaluation.close()
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
