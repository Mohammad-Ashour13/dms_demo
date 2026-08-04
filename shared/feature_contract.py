"""Immutable V3 feature/deployment contract."""

from __future__ import annotations

PIPELINE_VERSION = "3.0.0"
RUNTIME_FEATURE_VERSION = "v3-runtime-1.0.0"
WINDOW_SEC = 2.0
STRIDE_SEC = 0.5
TARGET_SAMPLES = 30
MIN_COVERAGE = 0.80

METADATA_COLUMNS = frozenset(
    {
        "frame_id", "dataset_name", "video_id", "subject_id", "session_id",
        "frame_number", "timestamp_ms", "fps", "label", "window_start_ms",
        "window_end_ms", "ear_baseline", "group_id", "split", "fold",
    }
)
QUALITY_COLUMNS = frozenset(
    {"face_detected", "face_confidence", "face_x", "face_y", "face_w", "face_h"}
)
FORBIDDEN_PREFIXES = ("ds_", "mouth_open_")

# Every base signal here can be produced by runtime/perception/mediapipe_face.py.
DEPLOYABLE_BASE_SIGNALS = (
    "left_ear", "right_ear", "relative_ear", "ear", "mar", "pitch", "yaw",
    "roll", "gaze_x", "gaze_y", "face_velocity_x", "face_velocity_y",
    "head_motion", "left_eye_closed", "right_eye_closed", "eyes_closed",
    "yawn_candidate", "gaze_zone",
)

CONTINUOUS_BASE_SIGNALS = DEPLOYABLE_BASE_SIGNALS[:13]
BINARY_BASE_SIGNALS = DEPLOYABLE_BASE_SIGNALS[13:17]

PHYSICAL_BOUNDS = {
    "left_ear": (0.0, 1.0), "right_ear": (0.0, 1.0), "ear": (0.0, 1.0),
    "relative_ear": (0.0, 2.0), "mar": (0.0, 2.0),
    "pitch": (-90.0, 90.0), "yaw": (-90.0, 90.0), "roll": (-90.0, 90.0),
    "gaze_x": (-1.0, 1.0), "gaze_y": (-1.0, 1.0),
    "face_velocity_x": (-15.0, 15.0), "face_velocity_y": (-15.0, 15.0),
    "head_motion": (0.0, 20_000.0),
}


def base_signal_for_feature(name: str) -> str | None:
    """Resolve an engineered column to its longest matching runtime base signal."""
    if name.startswith("gaze_zone_"):
        return "gaze_zone"
    for base in sorted(DEPLOYABLE_BASE_SIGNALS, key=len, reverse=True):
        if name.startswith(base + "_"):
            return base
    return None


def is_deployable_feature(name: str) -> bool:
    if name in METADATA_COLUMNS or name in QUALITY_COLUMNS:
        return False
    if name.startswith(FORBIDDEN_PREFIXES):
        return False
    return base_signal_for_feature(name) is not None


def audit_features(names: list[str]) -> dict[str, list[str]]:
    return {
        "metadata": [n for n in names if n in METADATA_COLUMNS or n in QUALITY_COLUMNS],
        "forbidden": [n for n in names if n.startswith(FORBIDDEN_PREFIXES)],
        "not_runtime_computable": [n for n in names if not is_deployable_feature(n)],
    }
