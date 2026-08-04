"""Contracts shared by training, runtime, tests, and integrations."""

from .contracts import (
    BehaviorDetectorSnapshot,
    AlarmCommand,
    AlarmLevel,
    AlarmStatus,
    DriverState,
    EvidenceEvent,
    FaceSignal,
    FramePacket,
    FusionDecision,
    IncidentRecord,
    ModelPrediction,
    ObjectDetection,
    TemporalFeatureSnapshot,
)

__all__ = [
    "AlarmCommand",
    "AlarmLevel",
    "AlarmStatus",
    "BehaviorDetectorSnapshot",
    "DriverState",
    "EvidenceEvent",
    "FaceSignal",
    "FramePacket",
    "FusionDecision",
    "IncidentRecord",
    "ModelPrediction",
    "ObjectDetection",
    "TemporalFeatureSnapshot",
]
