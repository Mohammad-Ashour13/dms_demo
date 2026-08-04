"""Asynchronous YOLO driver-behavior detector with timestamp-based stabilization."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from dms_final_system.shared.contracts import (
    BehaviorDetectorSnapshot,
    EvidenceEvent,
    FramePacket,
    ObjectDetection,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class UltralyticsYoloBackend:
    """Thin wrapper supporting both a YOLO `.pt` file and an exported NCNN directory."""

    def __init__(self, model_path: Path, device="cpu"):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Behavior detection requires ultralytics. Install requirements-laptop.txt "
                "or requirements-raspberry.txt."
            ) from exc
        self.model_path = Path(model_path)
        self.device = str(device)
        self.model = YOLO(str(self.model_path), task="detect")
        names = self.model.names
        if isinstance(names, dict):
            self.names = {int(key): str(value) for key, value in names.items()}
        else:
            self.names = {index: str(value) for index, value in enumerate(names)}

    def predict(self, frame: np.ndarray, *, image_size: int, confidence: float, iou: float):
        result = self.model.predict(
            frame,
            imgsz=int(image_size),
            conf=float(confidence),
            iou=float(iou),
            device=self.device,
            verbose=False,
        )[0]
        detections: list[ObjectDetection] = []
        if result.boxes is None:
            return detections
        class_ids = result.boxes.cls.detach().cpu().tolist()
        confidences = result.boxes.conf.detach().cpu().tolist()
        boxes = result.boxes.xyxy.detach().cpu().tolist()
        for class_id, score, xyxy in zip(class_ids, confidences, boxes):
            integer_id = int(class_id)
            detections.append(
                ObjectDetection(
                    integer_id,
                    self.names.get(integer_id, str(integer_id)),
                    float(score),
                    tuple(float(value) for value in xyxy),
                )
            )
        return detections


def _isolated_yolo_main(model_path, device, requests, responses) -> None:
    """Own all PyTorch/Ultralytics native state inside a spawned process."""
    try:
        backend = UltralyticsYoloBackend(Path(model_path), device)
        responses.put(("READY", backend.names))
    except BaseException as exc:
        responses.put(("STARTUP_ERROR", repr(exc)))
        return

    while True:
        request = requests.get()
        if request is None:
            return
        frame, image_size, confidence, iou = request
        try:
            detections = backend.predict(
                frame,
                image_size=image_size,
                confidence=confidence,
                iou=iou,
            )
            responses.put(
                (
                    "RESULT",
                    [
                        (
                            item.class_id,
                            item.label,
                            item.confidence,
                            tuple(item.xyxy),
                        )
                        for item in detections
                    ],
                )
            )
        except BaseException as exc:
            responses.put(("INFERENCE_ERROR", repr(exc)))


class ProcessIsolatedYoloBackend:
    """YOLO RPC proxy protecting the main DMS process from native crashes.

    MediaPipe may own an EGL context while Ultralytics/PyTorch initializes its
    own native CPU runtime.  Some Linux driver/library combinations terminate
    the interpreter during the first inference.  A spawned process has a clean
    native runtime; if it fails, the parent reports the detector as failed while
    camera, drowsiness inference, Fusion and recording continue to operate.
    """

    def __init__(self, model_path: Path, device="cpu", startup_timeout_sec=45.0):
        self.context = mp.get_context("spawn")
        self.requests = self.context.Queue(maxsize=1)
        self.responses = self.context.Queue(maxsize=1)
        self.process = self.context.Process(
            target=_isolated_yolo_main,
            args=(str(model_path), str(device), self.requests, self.responses),
            name="isolated-yolo-runtime",
            daemon=True,
        )
        self.process.start()
        kind, payload = self._receive(float(startup_timeout_sec))
        if kind != "READY":
            self.close()
            raise RuntimeError(f"Isolated YOLO startup failed: {payload}")
        self.names = {int(key): str(value) for key, value in payload.items()}

    def _receive(self, timeout_sec: float):
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            try:
                return self.responses.get(timeout=min(0.2, max(deadline - time.monotonic(), 0.01)))
            except queue.Empty:
                if not self.process.is_alive():
                    raise RuntimeError(
                        "Isolated YOLO process exited unexpectedly "
                        f"(exitcode={self.process.exitcode})"
                    )
        raise TimeoutError(f"Isolated YOLO did not respond within {timeout_sec:.1f}s")

    def predict(self, frame: np.ndarray, *, image_size: int, confidence: float, iou: float):
        if not self.process.is_alive():
            raise RuntimeError(
                f"Isolated YOLO process is not alive (exitcode={self.process.exitcode})"
            )
        self.requests.put(
            (frame, int(image_size), float(confidence), float(iou)), timeout=1.0
        )
        kind, payload = self._receive(30.0)
        if kind != "RESULT":
            raise RuntimeError(f"Isolated YOLO inference failed: {payload}")
        return [
            ObjectDetection(
                int(class_id), str(label), float(score), tuple(float(v) for v in xyxy)
            )
            for class_id, label, score, xyxy in payload
        ]

    def close(self) -> None:
        process = getattr(self, "process", None)
        if process is None:
            return
        if process.is_alive():
            try:
                self.requests.put(None, timeout=0.5)
            except queue.Full:
                pass
            process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)


class TemporalBehaviorFilter:
    """Turn per-inference detections into stable, refreshable violation evidence."""

    def __init__(
        self,
        class_mapping: dict[str, str],
        class_thresholds: dict[str, float],
        *,
        window_sec=1.5,
        minimum_samples=3,
        activation_ratio=0.60,
        activation_persistence_sec=0.50,
        clear_persistence_sec=2.0,
        evidence_refresh_sec=1.0,
        evidence_ttl_sec=2.5,
        source="luthfi-yolo11n-v1",
        model_version="unknown",
    ):
        self.mapping = {str(key): str(value) for key, value in class_mapping.items()}
        self.thresholds = {str(key): float(value) for key, value in class_thresholds.items()}
        self.window_sec = float(window_sec)
        self.minimum_samples = int(minimum_samples)
        self.activation_ratio = float(activation_ratio)
        self.activation_persistence = float(activation_persistence_sec)
        self.clear_persistence = float(clear_persistence_sec)
        self.refresh_sec = float(evidence_refresh_sec)
        self.ttl_sec = float(evidence_ttl_sec)
        self.source = str(source)
        self.model_version = str(model_version)
        self.history: deque[tuple[float, dict[str, float]]] = deque()
        self.active: set[str] = set()
        self.candidate_since: dict[str, float] = {}
        self.last_seen: dict[str, float] = {}
        self.last_emitted: dict[str, float] = {}

    def reset(self) -> None:
        self.history.clear()
        self.active.clear()
        self.candidate_since.clear()
        self.last_seen.clear()
        self.last_emitted.clear()

    def update(
        self,
        now: float,
        detections: list[ObjectDetection],
        *,
        frame_id: int,
    ) -> tuple[list[str], dict[str, float], list[EvidenceEvent]]:
        current: dict[str, float] = {}
        for detection in detections:
            label = str(detection.label)
            threshold = self.thresholds.get(label)
            if threshold is None or detection.confidence < threshold:
                continue
            current[label] = max(current.get(label, 0.0), float(detection.confidence))
            self.last_seen[label] = float(now)
        self.history.append((float(now), current))
        while self.history and self.history[0][0] < now - self.window_sec:
            self.history.popleft()

        sample_count = len(self.history)
        ratios = {
            label: (
                sum(label in labels for _, labels in self.history) / sample_count
                if sample_count else 0.0
            )
            for label in self.mapping
        }
        evidence: list[EvidenceEvent] = []
        for label, violation in self.mapping.items():
            meets = sample_count >= self.minimum_samples and ratios[label] >= self.activation_ratio
            if meets and label not in self.active:
                positive_times = [timestamp for timestamp, labels in self.history if label in labels]
                since = self.candidate_since.setdefault(
                    label, positive_times[0] if positive_times else float(now)
                )
                if now - since >= self.activation_persistence:
                    self.active.add(label)
                    self.last_emitted[label] = float("-inf")
            elif not meets and label not in self.active:
                self.candidate_since.pop(label, None)

            if label in self.active:
                last_seen = self.last_seen.get(label, float("-inf"))
                if now - last_seen >= self.clear_persistence:
                    self.active.remove(label)
                    self.candidate_since.pop(label, None)
                    self.last_emitted.pop(label, None)
                    continue
                last_emit = self.last_emitted.get(label, float("-inf"))
                # Keep the active state during the clear-persistence grace, but
                # do not manufacture zero-confidence refresh evidence after the
                # temporal window itself is no longer positive.
                if meets and now - last_emit >= self.refresh_sec:
                    trigger = "ONSET" if last_emit == float("-inf") else "REFRESH"
                    confidence = max(
                        (labels.get(label, 0.0) for _, labels in self.history),
                        default=0.0,
                    )
                    evidence.append(
                        EvidenceEvent(
                            violation,
                            float(now),
                            float(confidence),
                            self.ttl_sec,
                            self.source,
                            details={
                                "source_label": label,
                                "temporal_ratio": ratios[label],
                                "samples": sample_count,
                                "trigger": trigger,
                                "frame_id": int(frame_id),
                                "model_version": self.model_version,
                            },
                        )
                    )
                    self.last_emitted[label] = float(now)
        active_violations = sorted(self.mapping[label] for label in self.active)
        return active_violations, ratios, evidence


class YoloBehaviorDetector:
    """Bounded latest-frame worker so object inference never blocks the DMS loop."""

    REQUIRED_SOURCE_CLASSES = {"phone", "cigarette", "drink_or_food"}

    def __init__(self, config, evidence_bus, telemetry, *, backend=None):
        self.config = config
        self.evidence_bus = evidence_bus
        self.telemetry = telemetry
        self.enabled = bool(config.enabled)
        self.queue: queue.Queue[tuple[int, float, np.ndarray] | None] = queue.Queue(
            maxsize=max(1, int(config.queue_size))
        )
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.worker = None
        self.last_submitted = float("-inf")
        self.dropped_frames = 0
        self.backend = None
        self.model_version = "disabled"
        self.backend_name = "disabled"
        self.snapshot_value = BehaviorDetectorSnapshot(
            0.0, -1, self.model_version, self.backend_name, [], [], {}, 0.0,
            health="DISABLED",
        )
        if not self.enabled:
            return
        try:
            model_path = Path(config.model_path)
            manifest_path = Path(config.manifest_path)
            if not model_path.exists():
                raise FileNotFoundError(f"YOLO behavior model not found: {model_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.model_version = str(manifest["model_version"])
            if model_path.is_file() and manifest.get("sha256"):
                actual = _sha256(model_path)
                if actual != manifest["sha256"]:
                    raise RuntimeError(
                        f"YOLO model checksum mismatch: expected={manifest['sha256']} actual={actual}"
                    )
            if backend is not None:
                self.backend = backend
                execution_mode = "injected"
            elif config.execution_mode == "process":
                self.backend = ProcessIsolatedYoloBackend(model_path, config.device)
                execution_mode = "process"
            else:
                self.backend = UltralyticsYoloBackend(model_path, config.device)
                execution_mode = "thread"
            names = {str(value) for value in getattr(self.backend, "names", {}).values()}
            missing = sorted(self.REQUIRED_SOURCE_CLASSES - names)
            if missing:
                raise RuntimeError(f"YOLO model is missing required classes: {missing}; names={sorted(names)}")
            self.backend_name = f"{config.backend}:{execution_mode}"
            self.filter = TemporalBehaviorFilter(
                config.class_mapping,
                config.class_thresholds,
                window_sec=config.temporal_window_sec,
                minimum_samples=config.minimum_samples,
                activation_ratio=config.activation_ratio,
                activation_persistence_sec=config.activation_persistence_sec,
                clear_persistence_sec=config.clear_persistence_sec,
                evidence_refresh_sec=config.evidence_refresh_sec,
                evidence_ttl_sec=config.evidence_ttl_sec,
                model_version=self.model_version,
            )
            self.snapshot_value = BehaviorDetectorSnapshot(
                0.0, -1, self.model_version, self.backend_name, [], [], {}, 0.0,
                health="READY",
            )
            self.worker = threading.Thread(
                target=self._run, name="yolo-behavior-detector", daemon=True
            )
            self.worker.start()
            self.telemetry.emit(
                "BehaviorDetector",
                "startup_ready",
                {
                    "model_version": self.model_version,
                    "backend": self.backend_name,
                    "execution_mode": execution_mode,
                    "model_path": str(model_path),
                    "classes": sorted(names),
                    "interval_sec": config.inference_interval_sec,
                },
            )
        except Exception as exc:
            self.snapshot_value.health = "FAILED"
            self.snapshot_value.last_error = repr(exc)
            self.telemetry.emit(
                "BehaviorDetector", "startup_failed", {"error": repr(exc)}, level="ERROR"
            )
            if config.required:
                raise
            self.enabled = False

    def submit(self, packet: FramePacket) -> None:
        if not self.enabled or self.worker is None:
            return
        if packet.monotonic_sec - self.last_submitted < self.config.inference_interval_sec:
            return
        self.last_submitted = packet.monotonic_sec
        item = (packet.frame_id, packet.monotonic_sec, packet.frame.copy())
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.queue.task_done()
                self.dropped_frames += 1
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(item)
            except queue.Full:
                self.dropped_frames += 1

    def _crop(self, frame: np.ndarray):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = self.config.driver_roi
        px1, py1 = int(round(x1 * width)), int(round(y1 * height))
        px2, py2 = int(round(x2 * width)), int(round(y2 * height))
        return frame[py1:py2, px1:px2], px1, py1

    def _run(self) -> None:
        assert self.backend is not None
        minimum_confidence = min(float(value) for value in self.config.class_thresholds.values())
        while not self.stop_event.is_set():
            try:
                item = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                self.queue.task_done()
                break
            frame_id, timestamp, frame = item
            started = time.perf_counter()
            try:
                crop, offset_x, offset_y = self._crop(frame)
                raw = self.backend.predict(
                    crop,
                    image_size=self.config.image_size,
                    confidence=minimum_confidence,
                    iou=self.config.iou_threshold,
                )
                detections = [
                    ObjectDetection(
                        detection.class_id,
                        detection.label,
                        detection.confidence,
                        (
                            detection.xyxy[0] + offset_x,
                            detection.xyxy[1] + offset_y,
                            detection.xyxy[2] + offset_x,
                            detection.xyxy[3] + offset_y,
                        ),
                    )
                    for detection in raw
                    if detection.label in self.config.class_mapping
                    and detection.confidence >= self.config.class_thresholds[detection.label]
                ]
                active, ratios, events = self.filter.update(
                    timestamp, detections, frame_id=frame_id
                )
                for event in events:
                    self.evidence_bus.publish(event)
                inference_ms = (time.perf_counter() - started) * 1000.0
                snapshot = BehaviorDetectorSnapshot(
                    timestamp,
                    frame_id,
                    self.model_version,
                    self.backend_name,
                    detections,
                    active,
                    ratios,
                    inference_ms,
                    self.dropped_frames,
                    "READY",
                )
                with self.lock:
                    self.snapshot_value = snapshot
                self.telemetry.emit(
                    "BehaviorDetector",
                    "inference",
                    snapshot.to_dict(),
                    monotonic_sec=timestamp,
                    frame_id=frame_id,
                    level="WARNING" if active else "INFO",
                )
                for event in events:
                    self.telemetry.emit(
                        "BehaviorDetector",
                        "evidence_published",
                        {
                            "kind": event.kind,
                            "confidence": event.confidence,
                            "details": event.details,
                        },
                        monotonic_sec=timestamp,
                        frame_id=frame_id,
                    )
            except Exception as exc:
                with self.lock:
                    self.snapshot_value.health = "FAILED"
                    self.snapshot_value.last_error = repr(exc)
                self.telemetry.emit(
                    "BehaviorDetector",
                    "inference_failed",
                    {"error": repr(exc)},
                    monotonic_sec=timestamp,
                    frame_id=frame_id,
                    level="ERROR",
                )
            finally:
                self.queue.task_done()

    def snapshot(self) -> BehaviorDetectorSnapshot:
        with self.lock:
            value = self.snapshot_value
            return BehaviorDetectorSnapshot(
                value.monotonic_sec,
                value.frame_id,
                value.model_version,
                value.backend,
                list(value.detections),
                list(value.active_behaviors),
                dict(value.class_ratios),
                value.inference_ms,
                value.dropped_frames,
                value.health,
                value.last_error,
            )

    def close(self) -> None:
        if self.worker is not None:
            self.stop_event.set()
            try:
                self.queue.put_nowait(None)
            except queue.Full:
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                    self.queue.put_nowait(None)
                except queue.Empty:
                    pass
            self.worker.join(timeout=5.0)
        close_backend = getattr(self.backend, "close", None)
        if callable(close_backend):
            close_backend()
