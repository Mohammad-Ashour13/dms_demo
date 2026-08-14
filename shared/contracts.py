"""Versioned, serializable contracts between DMS components."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


CONTRACT_VERSION = "2.4.0"


class DriverState(str, Enum):
    UNKNOWN = "UNKNOWN"
    NORMAL = "NORMAL"
    FATIGUE_WARNING = "FATIGUE_WARNING"
    DROWSY = "DROWSY"
    CRITICAL = "CRITICAL"


class AlarmLevel(str, Enum):
    NONE = "NONE"
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(slots=True)
class FramePacket:
    frame_id: int
    utc_timestamp: str
    monotonic_sec: float
    frame: Any = field(repr=False)


@dataclass(slots=True)
class FaceSignal:
    frame_id: int
    utc_timestamp: str
    monotonic_sec: float
    face_detected: bool
    face_quality: float
    left_ear: float = 0.0
    right_ear: float = 0.0
    ear: float = 0.0
    relative_ear: float = 0.0
    # Event EAR is deliberately separate from the legacy/model EAR above.
    # Changing the legacy EAR would silently break the V3/LightGBM contract.
    event_left_ear: float = 0.0
    event_right_ear: float = 0.0
    event_ear: float = 0.0
    event_left_relative_ear: float = 0.0
    event_right_relative_ear: float = 0.0
    event_relative_ear: float = 0.0
    eye_signal_valid: bool = False
    eye_signal_quality: float = 0.0
    # Pixel geometry is a runtime safety gate. Ratios can look numerically
    # plausible even when an eye is represented by only a handful of pixels.
    face_width_px: float = 0.0
    face_height_px: float = 0.0
    face_width_ratio: float = 0.0
    interocular_distance_px: float = 0.0
    left_eye_width_px: float = 0.0
    right_eye_width_px: float = 0.0
    eye_resolution_valid: bool = True
    driver_distance_status: str = "UNKNOWN"
    binocular_consistent: bool = True
    mar: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    gaze_zone: str = "UNKNOWN"
    relative_gaze_y: float = 0.0
    gaze_signal_valid: bool = False
    downward_gaze_confidence: float = 0.0
    face_velocity_x: float = 0.0
    face_velocity_y: float = 0.0
    head_motion: float = 0.0
    left_eye_closed: int = 0
    right_eye_closed: int = 0
    eyes_closed: int = 0
    mouth_open: int = 0
    yawn_candidate: int = 0


@dataclass(slots=True)
class TemporalFeatureSnapshot:
    window_id: str
    start_monotonic_sec: float
    end_monotonic_sec: float
    coverage: float
    ordered_features: list[float]
    feature_names: list[str]
    valid: bool
    reason: str = ""


@dataclass(slots=True)
class ModelPrediction:
    window_id: str
    monotonic_sec: float
    raw_score: float
    calibrated_probability: float
    threshold: float
    positive: bool
    model_version: str


@dataclass(slots=True)
class EvidenceEvent:
    kind: str
    monotonic_sec: float
    confidence: float
    ttl_sec: float
    source: str
    value: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def is_fresh(self, now_sec: float) -> bool:
        return now_sec - self.monotonic_sec <= self.ttl_sec


@dataclass(slots=True)
class ObjectDetection:
    """One detector result in full-frame pixel coordinates."""

    class_id: int
    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]


@dataclass(slots=True)
class BehaviorDetectorSnapshot:
    monotonic_sec: float
    frame_id: int
    model_version: str
    backend: str
    detections: list[ObjectDetection]
    active_behaviors: list[str]
    class_ratios: dict[str, float]
    inference_ms: float
    dropped_frames: int = 0
    health: str = "READY"
    last_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SeatbeltDetectorSnapshot:
    """Serializable state of the low-rate seat-belt classifier."""

    monotonic_sec: float
    frame_id: int
    model_version: str
    backend: str
    no_seatbelt_probability: float | None
    seatbelt_probability: float | None
    active_violations: list[str]
    inference_ms: float
    roi: dict[str, Any] = field(default_factory=dict)
    dropped_frames: int = 0
    health: str = "READY"
    last_error: str = ""
    shadow_mode: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FusionDecision:
    monotonic_sec: float
    previous_state: DriverState
    driver_state: DriverState
    violations: list[str]
    alarm_level: AlarmLevel
    reason_codes: list[str]
    smoothed_probability: float | None
    changed: bool
    target_state: DriverState | None = None
    active_state: DriverState | None = None
    state_entry_reason: list[str] = field(default_factory=list)
    current_reason_codes: list[str] = field(default_factory=list)
    state_age_sec: float = 0.0
    candidate_age_sec: float = 0.0
    model_contribution_enabled: bool = True
    eye_evidence_trustworthy: bool = False
    eye_observation_status: str = "LOST"
    recovery_status: str = "NONE"
    model_risk_pending: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["previous_state"] = self.previous_state.value
        data["driver_state"] = self.driver_state.value
        data["alarm_level"] = self.alarm_level.value
        data["target_state"] = (
            self.target_state.value if self.target_state is not None else self.driver_state.value
        )
        data["active_state"] = (
            self.active_state.value if self.active_state is not None else self.driver_state.value
        )
        return data


@dataclass(slots=True)
class AlarmCommand:
    alarm_id: str
    episode_id: str
    monotonic_sec: float
    driver_state: DriverState
    pattern: str
    trigger: str
    recurrence_count: int
    reason_codes: list[str]
    incident_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["driver_state"] = self.driver_state.value
        return data


@dataclass(slots=True)
class AlarmStatus:
    active_pattern: str = "NONE"
    audio_playing: bool = False
    acknowledged: bool = False
    next_reminder_sec: float | None = None
    backend_health: str = "DISABLED"
    last_error: str = ""
    episode_id: str | None = None
    command_count: int = 0
    reminder_count: int = 0
    escalation_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class IncidentRecord:
    incident_id: str
    session_id: str
    started_utc: str
    ended_utc: str
    highest_state: str
    violations: list[str]
    max_model_probability: float
    reason_codes: list[str]
    model_version: str
    feature_version: str
    fusion_version: str
    video_path: Path
    telemetry_path: Path
    video_sha256: str = ""
    upload_status: str = "PENDING"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["video_path"] = str(self.video_path)
        data["telemetry_path"] = str(self.telemetry_path)
        data["contract_version"] = CONTRACT_VERSION
        return data
