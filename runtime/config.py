from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar


@dataclass(slots=True)
class CameraConfig:
    backend: str = "opencv"
    source: int | str = 0
    camera_num: int = 0
    width: int = 640
    height: int = 480
    fps: float = 15.0
    ai_queue_size: int = 2
    # Picamera2's RGB888 memory layout is BGR-compatible with OpenCV.
    pixel_format: str = "RGB888"
    system_python: str = "/usr/bin/python3"
    startup_timeout_sec: float = 15.0


@dataclass(slots=True)
class PerceptionConfig:
    face_landmarker_path: str = "models/mediapipe/face_landmarker.task"
    process_width: int = 256
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    min_face_quality: float = 0.55
    min_eye_signal_quality: float = 0.45
    max_eye_pose_deg: float = 25.0
    max_eye_asymmetry_ratio: float = 0.45
    min_face_width_ratio: float = 0.20
    min_interocular_distance_px: float = 50.0
    min_eye_width_px: float = 24.0


@dataclass(slots=True)
class CalibrationConfig:
    target_sec: float = 10.0
    max_sec: float = 15.0
    minimum_good_frames: int = 75
    max_pose_deviation_deg: float = 25.0
    event_trim_low_quantile: float = 0.20
    event_trim_high_quantile: float = 0.95
    max_event_ear_cv: float = 0.35
    max_event_eye_asymmetry: float = 0.35


@dataclass(slots=True)
class EventConfig:
    eye_close_relative_ear: float = 0.60
    eye_reopen_relative_ear: float = 0.72
    blink_min_sec: float = 0.05
    blink_max_sec: float = 0.80
    prolonged_closure_sec: float = 1.50
    max_eye_signal_gap_sec: float = 0.30
    binocular_close_sync_sec: float = 0.15
    max_eye_gaze_offset: float = 0.20
    strong_closure_median_ear: float = 0.30
    strong_closure_sample_ear: float = 0.40
    strong_closure_fraction: float = 0.80
    strong_closure_coverage: float = 0.80
    strong_closure_downward_confidence: float = 0.25
    strong_closure_max_downward_fraction: float = 0.50
    downward_gaze_delta: float = 0.18
    partial_closure_ear: float = 0.70
    partial_closure_window_sec: float = 2.0
    partial_closure_fraction: float = 0.70
    eye_acknowledge_open_sec: float = 0.50
    perclos_min_observation_sec: float = 10.0
    mouth_open_relative_ratio: float = 1.55
    yawn_min_sec: float = 1.20
    yawn_max_sec: float = 8.0
    head_nod_pitch_delta_deg: float = 18.0
    head_nod_min_sec: float = 0.35


@dataclass(slots=True)
class FusionConfig:
    version: str = "fusion-2.4.0"
    ewma_alpha: float = 0.35
    warning_enter: float = 0.55
    warning_exit: float = 0.42
    drowsy_enter: float = 0.75
    drowsy_exit: float = 0.58
    warning_persistence_sec: float = 2.0
    drowsy_persistence_sec: float = 1.5
    precritical_drowsy_closure_sec: float = 0.50
    normal_persistence_sec: float = 3.0
    recovery_persistence_sec: float = 3.0
    quality_loss_persistence_sec: float = 0.50
    quality_recovery_persistence_sec: float = 1.0
    transition_cooldown_sec: float = 1.0
    evidence_ttl_sec: float = 5.0
    excessive_blinks_per_min: int = 25
    excessive_yawns_per_min: int = 2
    perclos_warning: float = 0.25
    perclos_exit: float = 0.18
    perclos_min_valid_sec: float = 20.0
    perclos_min_coverage: float = 0.67
    require_evidence_for_drowsy: bool = True
    require_evidence_for_warning: bool = True


@dataclass(slots=True)
class DriftConfig:
    enabled: bool = True
    top_feature_count: int = 20
    out_of_range_fraction: float = 0.30
    trigger_consecutive_windows: int = 3
    recovery_consecutive_windows: int = 3


@dataclass(slots=True)
class BehaviorDetectorConfig:
    enabled: bool = False
    required: bool = False
    model_path: str = "models/driver_behavior/luthfi_yolo11n/best_yolo11n_ncnn_model"
    manifest_path: str = "models/driver_behavior/luthfi_yolo11n/model_manifest.json"
    backend: str = "auto"
    execution_mode: str = "auto"
    device: str = "cpu"
    ncnn_num_threads: int = 3
    image_size: int = 640
    iou_threshold: float = 0.45
    # Observable raw boxes may use a lower confidence than safety evidence.
    # class_thresholds remain the only activation/recording thresholds.
    raw_confidence_threshold: float = 0.10
    inference_interval_sec: float = 0.50
    adaptive_interval: bool = True
    latency_budget_ms: float = 250.0
    max_inference_interval_sec: float = 0.75
    queue_size: int = 1
    driver_roi: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])
    roi_mode: str = "face"
    roi_scale: float = 1.6
    roi_min_size: int = 96
    roi_fallback_static_on_no_face: bool = True
    context_roi_interval_sec: float = 2.0
    temporal_window_sec: float = 1.5
    # Scheduling/frame jitter allowance used only when a slower adaptive/safe
    # interval would otherwise make minimum_samples mathematically unreachable.
    temporal_sampling_slack_sec: float = 0.15
    minimum_samples: int = 3
    activation_ratio: float = 0.60
    activation_persistence_sec: float = 0.50
    clear_persistence_sec: float = 2.0
    evidence_refresh_sec: float = 1.0
    evidence_ttl_sec: float = 2.5
    class_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "phone": 0.40,
            "cigarette": 0.30,
            "drink_or_food": 0.35,
        }
    )
    class_mapping: dict[str, str] = field(
        default_factory=lambda: {
            "phone": "PHONE_USE",
            "cigarette": "SMOKING",
            "drink_or_food": "EATING",
        }
    )


@dataclass(slots=True)
class SeatbeltDetectorConfig:
    """Low-rate binary seat-belt classifier for the current driver's torso."""

    enabled: bool = False
    required: bool = False
    model_path: str = "models/seatbelt/risef_yolov11s"
    manifest_path: str = "models/seatbelt/risef_yolov11s/model_manifest.json"
    image_size: int = 224
    ncnn_num_threads: int = 1
    inference_interval_sec: float = 2.0
    queue_size: int = 1
    no_seatbelt_threshold: float = 0.80
    clear_threshold: float = 0.60
    activation_persistence_sec: float = 4.0
    clear_persistence_sec: float = 4.0
    evidence_refresh_sec: float = 2.0
    evidence_ttl_sec: float = 4.5
    roi_width_scale: float = 2.8
    roi_height_scale: float = 3.1
    roi_top_offset: float = -0.15
    roi_min_size: int = 128
    # Shadow mode emits telemetry and dashboard status only.  It never
    # publishes SEATBELT_MISSING into Fusion, alarms, or incident recording.
    shadow_mode: bool = True


@dataclass(slots=True)
class RecorderConfig:
    enabled: bool = True
    output_dir: str = "incidents/outbox"
    pre_alert_sec: float = 10.0
    post_alert_sec: float = 10.0
    merge_gap_sec: float = 5.0
    max_clip_sec: float = 60.0
    fps: float = 15.0
    jpeg_quality: int = 75
    bitrate: str = "1800k"
    queue_size: int = 256
    codec: str = "mjpeg_copy"
    require_copy_mux: bool = False
    reserve_free_bytes: int = 2 * 1024 * 1024 * 1024
    delete_oldest_when_full: bool = True
    trigger_states: list[str] = field(
        default_factory=lambda: ["FATIGUE_WARNING", "DROWSY", "CRITICAL"]
    )
    trigger_violations: list[str] = field(
        default_factory=lambda: ["PHONE_USE", "SMOKING", "EATING"]
    )
    suppress_yawn_only: bool = True
    episode_clear_sec: float = 2.0


@dataclass(slots=True)
class TelemetryConfig:
    log_dir: str = "logs"
    mode: str = "NORMAL"
    rotate_bytes: int = 20 * 1024 * 1024
    backups: int = 5
    live_status_hz: float = 2.0


@dataclass(slots=True)
class EvaluationConfig:
    enabled: bool = False
    output_dir: str = "evaluation_sessions"
    record_full_session: bool = True
    video_fps: float = 15.0
    queue_size: int = 512
    sampling_interval_sec: float = 0.5
    transition_tolerance_sec: float = 1.0
    alert_merge_gap_sec: float = 2.0
    minimum_overlap_sec: float = 0.5
    primary_positive_states: list[str] = field(
        default_factory=lambda: ["DROWSY", "CRITICAL"]
    )


@dataclass(slots=True)
class AlarmConfig:
    enabled: bool = False
    mode: str = "LOG_ONLY"
    allow_in_shadow: bool = True
    live_camera_only: bool = True
    allow_replay_audio: bool = False
    backend: str = "auto"
    device: str = "default"
    output: str = "audio"
    gpio_pin: int = 18
    gpio_buzzer_type: str = "active"
    gpio_pwm_frequency_hz: float = 100.0
    required: bool = False
    startup_self_test: bool = False
    master_gain: float = 0.40
    queue_size: int = 8
    behavior_reminder_sec: float = 15.0
    drowsy_reminder_sec: float = 10.0
    drowsy_rearm_sec: float = 5.0
    drowsy_ack_silence_sec: float = 10.0
    escalation_window_sec: float = 60.0
    escalation_episode_count: int = 3
    escalation_continuous_sec: float = 30.0
    escalation_reminder_sec: float = 7.0
    critical_reminder_sec: float = 4.0
    critical_escalate_after_sec: float = 15.0
    critical_escalated_reminder_sec: float = 2.5
    acknowledge_open_sec: float = 0.50


@dataclass(slots=True)
class HMIConfig:
    enabled: bool = False
    show_camera: bool = True
    critical_flash_hz: float = 2.0
    window_name: str = "DMS Controlled Demo"


@dataclass(slots=True)
class DashboardConfig:
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8080
    status_hz: float = 2.0
    preview_fps: float = 15.0
    safe_preview_fps: float = 5.0
    max_clients: int = 2
    title: str = "Safee Driver Monitoring"


@dataclass(slots=True)
class PowerConfig:
    enabled: bool = True
    sample_interval_sec: float = 1.0
    enter_temperature_c: float = 78.0
    exit_temperature_c: float = 72.0
    recovery_sec: float = 60.0
    safe_yolo_interval_sec: float = 0.75
    cpu_max_mhz: int = 0


@dataclass(slots=True)
class SafeemaxApiConfig:
    """Idempotent, non-blocking delivery to the Safeemax device API."""

    enabled: bool = False
    endpoint_url: str = "http://76.13.131.115:4000/api/device-data"
    auth_token_env: str = ""
    device_id: str = ""
    vehicle: str = ""
    driver: str | None = None
    location: str = ""
    battery: int | None = None
    outbox_dir: str = "incidents/api_outbox"
    request_timeout_sec: float = 5.0
    retry_initial_sec: float = 1.0
    retry_max_sec: float = 30.0
    queue_size: int = 256
    status_interval_sec: float = 5.0
    incident_upload_enabled: bool = True
    telemetry_upload_enabled: bool = True
    telemetry_batch_size: int = 250
    telemetry_flush_interval_sec: float = 15.0
    telemetry_queue_size: int = 2000
    event_states: list[str] = field(
        default_factory=lambda: ["FATIGUE_WARNING", "DROWSY", "CRITICAL"]
    )
    event_violations: list[str] = field(
        default_factory=lambda: ["PHONE_USE", "SMOKING", "EATING", "SEATBELT_MISSING"]
    )


@dataclass(slots=True)
class RuntimeConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    events: EventConfig = field(default_factory=EventConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    behavior_detector: BehaviorDetectorConfig = field(default_factory=BehaviorDetectorConfig)
    seatbelt_detector: SeatbeltDetectorConfig = field(default_factory=SeatbeltDetectorConfig)
    recorder: RecorderConfig = field(default_factory=RecorderConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    alarm: AlarmConfig = field(default_factory=AlarmConfig)
    hmi: HMIConfig = field(default_factory=HMIConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    power: PowerConfig = field(default_factory=PowerConfig)
    safeemax_api: SafeemaxApiConfig = field(default_factory=SafeemaxApiConfig)
    deployment_mode: str = "SHADOW"
    active_model_path: str = "models/drowsiness/active_model.json"
    inference_interval_sec: float = 0.5
    session_name: str = "raspberry-live"
    cv_num_threads: int = 2
    omp_num_threads: int = 2
    ncnn_num_threads: int = 3


T = TypeVar("T")


def _construct(cls: type[T], values: dict[str, Any]) -> T:
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} settings: {unknown}")
    return cls(**values)


def load_config(path: Path) -> RuntimeConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    event_values = raw.get("events")
    if isinstance(event_values, dict) and "eye_closed_relative_ear" in event_values:
        legacy_threshold = float(event_values.pop("eye_closed_relative_ear"))
        event_values.setdefault("eye_close_relative_ear", min(legacy_threshold, 0.60))
        event_values.setdefault("eye_reopen_relative_ear", 0.72)
    known = {f.name for f in fields(RuntimeConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown RuntimeConfig settings: {unknown}")
    nested = {
        "camera": CameraConfig,
        "perception": PerceptionConfig,
        "calibration": CalibrationConfig,
        "events": EventConfig,
        "fusion": FusionConfig,
        "drift": DriftConfig,
        "behavior_detector": BehaviorDetectorConfig,
        "seatbelt_detector": SeatbeltDetectorConfig,
        "recorder": RecorderConfig,
        "telemetry": TelemetryConfig,
        "evaluation": EvaluationConfig,
        "alarm": AlarmConfig,
        "hmi": HMIConfig,
        "dashboard": DashboardConfig,
        "power": PowerConfig,
        "safeemax_api": SafeemaxApiConfig,
    }
    for key, cls in nested.items():
        if key in raw:
            raw[key] = _construct(cls, raw[key])
    config = RuntimeConfig(**raw)
    config.deployment_mode = str(config.deployment_mode).upper()
    if config.deployment_mode not in {"SHADOW", "ACTIVE"}:
        raise ValueError("deployment_mode must be SHADOW or ACTIVE")
    if config.evaluation.enabled and config.deployment_mode != "SHADOW":
        raise ValueError("Evaluation sessions must run with deployment_mode=SHADOW")
    if config.evaluation.sampling_interval_sec <= 0:
        raise ValueError("evaluation.sampling_interval_sec must be positive")
    if config.evaluation.video_fps <= 0 or config.evaluation.queue_size <= 0:
        raise ValueError("evaluation video_fps and queue_size must be positive")
    if not 0 < config.events.eye_close_relative_ear < config.events.eye_reopen_relative_ear:
        raise ValueError("events eye thresholds must satisfy 0 < close < reopen")
    if not 0 < config.events.strong_closure_median_ear < config.events.strong_closure_sample_ear < config.events.eye_reopen_relative_ear:
        raise ValueError("events strong closure thresholds must satisfy 0 < median < sample < reopen")
    if not 0 < config.events.strong_closure_fraction <= 1 or not 0 < config.events.strong_closure_coverage <= 1:
        raise ValueError("events strong closure fraction and coverage must be in (0, 1]")
    if not 0 <= config.events.strong_closure_downward_confidence <= 1 or not 0 <= config.events.strong_closure_max_downward_fraction <= 1:
        raise ValueError("events strong closure downward settings must be between 0 and 1")
    if config.events.blink_min_sec <= 0 or config.events.max_eye_signal_gap_sec <= 0:
        raise ValueError("event timing thresholds must be positive")
    if config.events.binocular_close_sync_sec <= 0:
        raise ValueError("events.binocular_close_sync_sec must be positive")
    if not 0 < config.events.max_eye_gaze_offset <= 1:
        raise ValueError("events.max_eye_gaze_offset must be in (0, 1]")
    if config.perception.process_width < 0:
        raise ValueError("perception.process_width must be zero or positive")
    if not 0 < config.perception.min_face_width_ratio < 1:
        raise ValueError("perception.min_face_width_ratio must be in (0, 1)")
    if min(
        config.perception.min_interocular_distance_px,
        config.perception.min_eye_width_px,
    ) <= 0:
        raise ValueError("perception eye pixel-resolution thresholds must be positive")
    config.camera.backend = str(config.camera.backend).lower()
    if config.camera.backend not in {
        "opencv",
        "picamera2",
        "picamera2_auto",
        "picamera2_process",
    }:
        raise ValueError(
            "camera.backend must be opencv, picamera2, picamera2_auto, or "
            "picamera2_process"
        )
    if min(config.camera.width, config.camera.height, config.camera.ai_queue_size) <= 0 or config.camera.fps <= 0:
        raise ValueError("camera dimensions, fps and queue size must be positive")
    if config.camera.startup_timeout_sec <= 0:
        raise ValueError("camera.startup_timeout_sec must be positive")
    if not 0 <= config.fusion.perclos_exit < config.fusion.perclos_warning <= 1:
        raise ValueError("fusion PERCLOS thresholds must satisfy 0 <= exit < enter <= 1")
    if not 0 < config.fusion.precritical_drowsy_closure_sec < config.events.prolonged_closure_sec:
        raise ValueError(
            "fusion.precritical_drowsy_closure_sec must be positive and below prolonged closure"
        )
    detector = config.behavior_detector
    detector.backend = str(detector.backend).lower()
    detector.execution_mode = str(detector.execution_mode).lower()
    detector.roi_mode = str(detector.roi_mode).lower()
    if detector.backend not in {"auto", "ncnn", "onnx", "pytorch", "ultralytics"}:
        raise ValueError("behavior_detector.backend must be auto, ncnn, onnx, pytorch or ultralytics")
    if detector.execution_mode not in {"auto", "process", "thread"}:
        raise ValueError("behavior_detector.execution_mode must be auto, process or thread")
    if detector.roi_mode not in {"face", "static", "full"}:
        raise ValueError("behavior_detector.roi_mode must be face, static or full")
    if detector.image_size <= 0 or detector.inference_interval_sec <= 0 or detector.queue_size <= 0:
        raise ValueError("behavior detector image size, interval and queue size must be positive")
    if detector.ncnn_num_threads <= 0:
        raise ValueError("behavior_detector.ncnn_num_threads must be positive")
    if detector.latency_budget_ms <= 0 or detector.max_inference_interval_sec < detector.inference_interval_sec:
        raise ValueError("behavior detector adaptive interval settings are invalid")
    if detector.minimum_samples > 1:
        maximum_sampling_interval = detector.temporal_window_sec / (detector.minimum_samples - 1)
        if detector.max_inference_interval_sec > maximum_sampling_interval + 1e-9:
            raise ValueError(
                "behavior_detector.max_inference_interval_sec cannot supply minimum_samples "
                "inside temporal_window_sec"
            )
    if (
        detector.roi_scale < 1.0
        or detector.roi_min_size <= 0
        or detector.context_roi_interval_sec <= 0
    ):
        raise ValueError("behavior_detector ROI scale and min size are invalid")
    if (
        detector.minimum_samples <= 0
        or detector.temporal_window_sec <= 0
        or detector.temporal_sampling_slack_sec < 0
    ):
        raise ValueError("behavior detector temporal settings must be positive")
    if not 0 < detector.activation_ratio <= 1:
        raise ValueError("behavior_detector.activation_ratio must be in (0, 1]")
    if detector.activation_persistence_sec < 0 or detector.clear_persistence_sec < 0:
        raise ValueError("behavior detector persistence settings cannot be negative")
    if detector.evidence_refresh_sec <= 0 or detector.evidence_ttl_sec <= detector.evidence_refresh_sec:
        raise ValueError("behavior detector evidence TTL must be greater than refresh interval")
    if len(detector.driver_roi) != 4 or not all(0.0 <= float(value) <= 1.0 for value in detector.driver_roi):
        raise ValueError("behavior_detector.driver_roi must contain four normalized values")
    x1, y1, x2, y2 = map(float, detector.driver_roi)
    if not x1 < x2 or not y1 < y2:
        raise ValueError("behavior_detector.driver_roi must satisfy x1 < x2 and y1 < y2")
    if set(detector.class_thresholds) != set(detector.class_mapping):
        raise ValueError("behavior detector thresholds and mapping must define the same source classes")
    if any(not 0 < float(value) <= 1 for value in detector.class_thresholds.values()):
        raise ValueError("behavior detector class thresholds must be in (0, 1]")
    if not 0 < detector.raw_confidence_threshold <= min(detector.class_thresholds.values()):
        raise ValueError(
            "behavior_detector.raw_confidence_threshold must be positive and no greater "
            "than the smallest class threshold"
        )
    seatbelt = config.seatbelt_detector
    if (
        seatbelt.image_size <= 0
        or seatbelt.ncnn_num_threads <= 0
        or seatbelt.queue_size <= 0
        or seatbelt.inference_interval_sec <= 0
    ):
        raise ValueError("seatbelt detector image size, threads, interval and queue size must be positive")
    if not 0 < seatbelt.clear_threshold <= seatbelt.no_seatbelt_threshold <= 1:
        raise ValueError("seatbelt detector thresholds must satisfy 0 < clear <= no_seatbelt <= 1")
    if (
        seatbelt.activation_persistence_sec < 0
        or seatbelt.clear_persistence_sec < 0
        or seatbelt.evidence_refresh_sec <= 0
        or seatbelt.evidence_ttl_sec <= seatbelt.evidence_refresh_sec
    ):
        raise ValueError("seatbelt detector temporal settings are invalid")
    if (
        seatbelt.roi_width_scale < 1.0
        or seatbelt.roi_height_scale < 1.0
        or seatbelt.roi_min_size <= 0
    ):
        raise ValueError("seatbelt detector torso ROI settings are invalid")
    if min(config.cv_num_threads, config.omp_num_threads, config.ncnn_num_threads) <= 0:
        raise ValueError("runtime thread caps must be positive")
    recorder = config.recorder
    if recorder.codec not in {"mjpeg_copy", "legacy_h264"}:
        raise ValueError("recorder.codec must be mjpeg_copy or legacy_h264")
    if min(recorder.pre_alert_sec, recorder.post_alert_sec, recorder.max_clip_sec) <= 0:
        raise ValueError("recorder clip timings must be positive")
    if not 1 <= recorder.jpeg_quality <= 100 or recorder.queue_size <= 0:
        raise ValueError("recorder JPEG quality and queue size are invalid")
    if recorder.reserve_free_bytes < 0 or recorder.episode_clear_sec < 0:
        raise ValueError("recorder retention and episode settings cannot be negative")
    known_states = {"FATIGUE_WARNING", "DROWSY", "CRITICAL"}
    known_violations = {"PHONE_USE", "SMOKING", "EATING", "SEATBELT_MISSING"}
    if not set(recorder.trigger_states) <= known_states:
        raise ValueError("recorder.trigger_states contains unsupported states")
    if not set(recorder.trigger_violations) <= known_violations:
        raise ValueError("recorder.trigger_violations contains unsupported violations")
    if config.dashboard.port <= 0 or config.dashboard.status_hz <= 0:
        raise ValueError("dashboard port and status_hz must be positive")
    if min(config.dashboard.preview_fps, config.dashboard.safe_preview_fps, config.dashboard.max_clients) <= 0:
        raise ValueError("dashboard preview settings must be positive")
    if not 0 < config.power.exit_temperature_c < config.power.enter_temperature_c:
        raise ValueError("power temperatures must satisfy 0 < exit < enter")
    if min(config.power.sample_interval_sec, config.power.recovery_sec, config.power.safe_yolo_interval_sec) <= 0:
        raise ValueError("power timing settings must be positive")
    if not (
        detector.inference_interval_sec
        <= config.power.safe_yolo_interval_sec
        <= detector.max_inference_interval_sec
    ):
        raise ValueError(
            "power.safe_yolo_interval_sec must be within the behavior detector interval range"
        )
    if not 0 <= config.fusion.perclos_min_coverage <= 1:
        raise ValueError("fusion.perclos_min_coverage must be between 0 and 1")
    if min(
        config.fusion.quality_loss_persistence_sec,
        config.fusion.quality_recovery_persistence_sec,
    ) <= 0:
        raise ValueError("fusion quality timing thresholds must be positive")
    if not 0.0 <= config.drift.out_of_range_fraction <= 1.0:
        raise ValueError("drift.out_of_range_fraction must be between 0 and 1")
    if min(
        config.drift.top_feature_count,
        config.drift.trigger_consecutive_windows,
        config.drift.recovery_consecutive_windows,
    ) <= 0:
        raise ValueError("drift feature/window counts must be positive")
    api = config.safeemax_api
    api.endpoint_url = str(api.endpoint_url).strip()
    if not api.endpoint_url.startswith(("http://", "https://")):
        raise ValueError("safeemax_api.endpoint_url must use http or https")
    if api.enabled and (not api.device_id.strip() or not api.vehicle.strip()):
        raise ValueError("safeemax_api.device_id and vehicle are required when enabled")
    if min(
        api.request_timeout_sec,
        api.retry_initial_sec,
        api.retry_max_sec,
        api.queue_size,
        api.status_interval_sec,
        api.telemetry_batch_size,
        api.telemetry_flush_interval_sec,
        api.telemetry_queue_size,
    ) <= 0 or api.retry_initial_sec > api.retry_max_sec:
        raise ValueError("safeemax_api timeout, retry, and queue settings are invalid")
    if api.telemetry_batch_size > 500:
        raise ValueError("safeemax_api.telemetry_batch_size must not exceed 500")
    if api.battery is not None and not 0 <= api.battery <= 100:
        raise ValueError("safeemax_api.battery must be between 0 and 100")
    if not set(api.event_states) <= known_states:
        raise ValueError("safeemax_api.event_states contains unsupported states")
    if not set(api.event_violations) <= known_violations:
        raise ValueError("safeemax_api.event_violations contains unsupported violations")
    config.alarm.mode = str(config.alarm.mode).upper()
    config.alarm.output = str(config.alarm.output).lower()
    config.alarm.gpio_buzzer_type = str(config.alarm.gpio_buzzer_type).lower()
    if config.alarm.mode not in {"OFF", "LOG_ONLY", "LOCAL"}:
        raise ValueError("alarm.mode must be OFF, LOG_ONLY or LOCAL")
    if config.alarm.output not in {"audio", "gpio"}:
        raise ValueError("alarm.output must be audio or gpio")
    if config.alarm.gpio_buzzer_type not in {"active", "passive"}:
        raise ValueError("alarm.gpio_buzzer_type must be active or passive")
    if not 0 <= config.alarm.gpio_pin <= 27:
        raise ValueError("alarm.gpio_pin must be a BCM GPIO number between 0 and 27")
    if config.alarm.gpio_pwm_frequency_hz <= 0:
        raise ValueError("alarm.gpio_pwm_frequency_hz must be positive")
    if config.alarm.mode == "LOCAL" and config.deployment_mode == "SHADOW" and not config.alarm.allow_in_shadow:
        raise ValueError("LOCAL alarm in SHADOW requires alarm.allow_in_shadow=true")
    if not 0 <= config.alarm.master_gain <= 1:
        raise ValueError("alarm.master_gain must be between 0 and 1")
    if min(
        config.alarm.queue_size,
        config.alarm.behavior_reminder_sec,
        config.alarm.drowsy_reminder_sec,
        config.alarm.critical_reminder_sec,
        config.alarm.acknowledge_open_sec,
        config.hmi.critical_flash_hz,
    ) <= 0:
        raise ValueError("alarm/HMI timing and queue settings must be positive")
    allowed_states = {"DROWSY", "CRITICAL"}
    if not config.evaluation.primary_positive_states or not set(config.evaluation.primary_positive_states) <= allowed_states:
        raise ValueError("evaluation.primary_positive_states may contain DROWSY and CRITICAL only")
    return config
