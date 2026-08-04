"""Standalone copy of the immutable runtime-computable V3 feature contract.

This file keeps ``training_colab/`` runnable by itself on Google Drive.  Its
version and values must remain identical to ``dms_final_system/shared``.
"""

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

DEPLOYABLE_BASE_SIGNALS = (
    "left_ear", "right_ear", "relative_ear", "ear", "mar", "pitch", "yaw",
    "roll", "gaze_x", "gaze_y", "face_velocity_x", "face_velocity_y",
    "head_motion", "left_eye_closed", "right_eye_closed", "eyes_closed",
    "yawn_candidate", "gaze_zone",
)


def base_signal_for_feature(name: str) -> str | None:
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
