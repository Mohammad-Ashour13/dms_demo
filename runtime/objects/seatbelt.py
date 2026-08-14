"""Low-rate, NCNN-backed seat-belt classification for a driver-facing camera."""

from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from dms_final_system.runtime.objects.yolo_behavior import NCNN_INFERENCE_LOCK
from dms_final_system.shared.contracts import EvidenceEvent, FramePacket, SeatbeltDetectorSnapshot


NO_SEATBELT_LABEL = "no_seatbelt"
SEATBELT_LABEL = "seat_belt"
SEATBELT_MISSING = "SEATBELT_MISSING"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_ncnn_files(model_path: Path) -> tuple[Path, Path]:
    params = sorted(model_path.glob("*.param"))
    bins = sorted(model_path.glob("*.bin"))
    if len(params) != 1 or len(bins) != 1:
        raise RuntimeError(
            "Seat-belt NCNN model directory must contain exactly one .param and one .bin file: "
            f"{model_path}"
        )
    return params[0], bins[0]


def _parse_ncnn_blob_names(param_path: Path) -> tuple[str, str]:
    lines = [
        line.strip()
        for line in param_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    input_names: list[str] = []
    tops: list[str] = []
    bottoms: set[str] = set()
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            input_count, output_count = int(parts[2]), int(parts[3])
        except ValueError:
            continue
        layer_bottoms = parts[4:4 + input_count]
        layer_tops = parts[4 + input_count:4 + input_count + output_count]
        bottoms.update(layer_bottoms)
        tops.extend(layer_tops)
        if parts[0].lower() == "input":
            input_names.extend(layer_tops)
    output_names = [name for name in tops if name not in bottoms]
    if not input_names or not output_names:
        raise RuntimeError(f"Could not infer NCNN I/O names from {param_path}")
    return input_names[0], output_names[-1]


def _probabilities(output: Any, labels: list[str]) -> dict[str, float]:
    values = np.asarray(output, dtype=np.float32).reshape(-1)
    if len(values) != len(labels):
        raise RuntimeError(
            f"Seat-belt classifier output has {len(values)} values; expected {len(labels)}"
        )
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Seat-belt classifier produced non-finite values")
    if np.all(values >= 0.0) and np.all(values <= 1.0) and abs(float(values.sum()) - 1.0) <= 1e-3:
        normalized = values
    else:
        shifted = values - np.max(values)
        normalized = np.exp(shifted) / np.exp(shifted).sum()
    return {label: float(value) for label, value in zip(labels, normalized)}


class NcnnSeatbeltClassifierBackend:
    """Runtime for the approved 224px RISEF seat-belt ONNX-to-NCNN export."""

    def __init__(self, model_path: Path, class_names: list[str], *, image_size: int, num_threads: int):
        if not Path(model_path).is_dir():
            raise FileNotFoundError(f"Seat-belt NCNN model path is not a directory: {model_path}")
        if class_names != [NO_SEATBELT_LABEL, SEATBELT_LABEL]:
            raise RuntimeError(
                "Seat-belt classifier classes must be ['no_seatbelt', 'seat_belt']; "
                f"got {class_names}"
            )
        try:
            import ncnn
        except ImportError as exc:
            raise RuntimeError("Seat-belt classification requires the ncnn Python package") from exc
        self.ncnn = ncnn
        self.class_names = list(class_names)
        self.image_size = int(image_size)
        param_path, bin_path = _find_ncnn_files(Path(model_path))
        self.input_name, self.output_name = _parse_ncnn_blob_names(param_path)
        self.net = ncnn.Net()
        self.net.opt.num_threads = int(num_threads)
        self.net.opt.use_vulkan_compute = False
        if self.net.load_param(str(param_path)) not in {0, None}:
            raise RuntimeError(f"Failed to load seat-belt NCNN param file: {param_path}")
        if self.net.load_model(str(bin_path)) not in {0, None}:
            raise RuntimeError(f"Failed to load seat-belt NCNN bin file: {bin_path}")

    def predict(self, frame: np.ndarray) -> dict[str, float]:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Seat-belt classifier requires a BGR frame with three channels")
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for seat-belt preprocessing") from exc
        resized = cv2.resize(frame, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        mat = self.ncnn.Mat.from_pixels(
            rgb, self.ncnn.Mat.PixelType.PIXEL_RGB, self.image_size, self.image_size
        )
        mat.substract_mean_normalize([], [1 / 255.0, 1 / 255.0, 1 / 255.0])
        with NCNN_INFERENCE_LOCK:
            extractor = self.net.create_extractor()
            extractor.input(self.input_name, mat)
            result = extractor.extract(self.output_name)
        if isinstance(result, tuple):
            status, output = result
            if status != 0:
                raise RuntimeError(f"Seat-belt NCNN inference failed with status {status}")
        else:
            output = result
        return _probabilities(output, self.class_names)


class SeatbeltDetector:
    """Asynchronous classifier that treats only persistent no-seatbelt as a violation."""

    def __init__(self, config, evidence_bus, telemetry, *, backend=None):
        self.config = config
        self.evidence_bus = evidence_bus
        self.telemetry = telemetry
        self.enabled = bool(config.enabled)
        self.queue: queue.Queue[tuple[int, float, np.ndarray, tuple[float, float, float, float] | None] | None] = queue.Queue(
            maxsize=max(1, int(config.queue_size))
        )
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.worker: threading.Thread | None = None
        self.backend = None
        self.model_version = "disabled"
        self.backend_name = "disabled"
        self.last_submitted = float("-inf")
        self.dropped_frames = 0
        self.no_seatbelt_since: float | None = None
        self.clear_since: float | None = None
        self.last_emitted = float("-inf")
        self.active = False
        self.snapshot_value = SeatbeltDetectorSnapshot(
            0.0, -1, self.model_version, self.backend_name, None, None, [], 0.0,
            health="DISABLED", shadow_mode=bool(config.shadow_mode),
        )
        if not self.enabled:
            return
        try:
            if backend is None:
                model_path = Path(config.model_path)
                manifest_path = Path(config.manifest_path)
                if not model_path.is_dir():
                    raise FileNotFoundError(f"Seat-belt NCNN model directory is missing: {model_path}")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                class_names = [str(value) for value in manifest.get("class_names") or []]
                self.model_version = str(manifest["model_version"])
                self.backend = NcnnSeatbeltClassifierBackend(
                    model_path,
                    class_names,
                    image_size=config.image_size,
                    num_threads=config.ncnn_num_threads,
                )
            else:
                self.model_version = "injected-seatbelt-classifier"
                self.backend = backend
            self.backend_name = "ncnn:thread"
            self.snapshot_value = SeatbeltDetectorSnapshot(
                0.0, -1, self.model_version, self.backend_name, None, None, [], 0.0,
                health="READY", shadow_mode=bool(config.shadow_mode),
            )
            self.worker = threading.Thread(
                target=self._run, name="seatbelt-classifier", daemon=True
            )
            self.worker.start()
            self.telemetry.emit(
                "SeatbeltDetector", "startup_ready",
                {
                    "model_version": self.model_version,
                    "backend": self.backend_name,
                    "interval_sec": float(config.inference_interval_sec),
                    "shadow_mode": bool(config.shadow_mode),
                },
            )
        except Exception as exc:
            self.snapshot_value.health = "FAILED"
            self.snapshot_value.last_error = repr(exc)
            self.telemetry.emit("SeatbeltDetector", "startup_failed", {"error": repr(exc)}, level="ERROR")
            if config.required:
                raise
            self.enabled = False

    def submit(self, packet: FramePacket, face_bbox_xyxy: tuple[float, float, float, float] | None) -> None:
        if not self.enabled or self.worker is None:
            return
        if packet.monotonic_sec - self.last_submitted < float(self.config.inference_interval_sec):
            return
        self.last_submitted = packet.monotonic_sec
        item = (packet.frame_id, packet.monotonic_sec, packet.frame.copy(), face_bbox_xyxy)
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

    def _crop_torso(
        self, frame: np.ndarray, face_bbox_xyxy: tuple[float, float, float, float] | None
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        height, width = frame.shape[:2]
        if face_bbox_xyxy is None:
            return None, {"mode": "skipped", "reason": "no_face"}
        x1, y1, x2, y2 = (float(value) for value in face_bbox_xyxy)
        x1, x2 = sorted((max(0.0, x1), min(float(width), x2)))
        y1, y2 = sorted((max(0.0, y1), min(float(height), y2)))
        face_width, face_height = x2 - x1, y2 - y1
        if face_width <= 1.0 or face_height <= 1.0:
            return None, {"mode": "skipped", "reason": "invalid_face_bbox"}
        crop_width = max(float(self.config.roi_min_size), face_width * float(self.config.roi_width_scale))
        crop_height = max(float(self.config.roi_min_size), face_height * float(self.config.roi_height_scale))
        center_x = (x1 + x2) / 2.0
        crop_x1 = center_x - crop_width / 2.0
        crop_y1 = y1 + face_height * float(self.config.roi_top_offset)
        crop_x2 = crop_x1 + crop_width
        crop_y2 = crop_y1 + crop_height
        crop_x1, crop_y1 = max(0.0, crop_x1), max(0.0, crop_y1)
        crop_x2, crop_y2 = min(float(width), crop_x2), min(float(height), crop_y2)
        px1, py1 = int(np.floor(crop_x1)), int(np.floor(crop_y1))
        px2, py2 = int(np.ceil(crop_x2)), int(np.ceil(crop_y2))
        if px2 - px1 < int(self.config.roi_min_size) or py2 - py1 < int(self.config.roi_min_size):
            return None, {"mode": "skipped", "reason": "torso_roi_too_small"}
        return frame[py1:py2, px1:px2], {
            "mode": "face_anchored_torso",
            "crop_xyxy": [px1, py1, px2, py2],
            "crop_width": px2 - px1,
            "crop_height": py2 - py1,
            "face_bbox_xyxy": [x1, y1, x2, y2],
        }

    def _update_state(self, timestamp: float, no_seatbelt_probability: float) -> bool:
        enter = float(self.config.no_seatbelt_threshold)
        clear = float(self.config.clear_threshold)
        if no_seatbelt_probability >= enter:
            self.clear_since = None
            self.no_seatbelt_since = self.no_seatbelt_since or timestamp
            if timestamp - self.no_seatbelt_since >= float(self.config.activation_persistence_sec):
                self.active = True
        elif self.active and no_seatbelt_probability <= clear:
            self.no_seatbelt_since = None
            self.clear_since = self.clear_since or timestamp
            if timestamp - self.clear_since >= float(self.config.clear_persistence_sec):
                self.active = False
        else:
            if not self.active:
                self.no_seatbelt_since = None
            self.clear_since = None
        return self.active

    def _run(self) -> None:
        assert self.backend is not None
        while not self.stop_event.is_set():
            try:
                item = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                self.queue.task_done()
                return
            frame_id, timestamp, frame, face_bbox_xyxy = item
            try:
                crop, roi = self._crop_torso(frame, face_bbox_xyxy)
                if crop is None:
                    self.telemetry.emit(
                        "SeatbeltDetector", "inference_skipped", {"roi": roi},
                        monotonic_sec=timestamp, frame_id=frame_id,
                    )
                    continue
                started = time.perf_counter()
                probabilities = self.backend.predict(crop)
                inference_ms = (time.perf_counter() - started) * 1000.0
                no_seatbelt_probability = float(probabilities[NO_SEATBELT_LABEL])
                seatbelt_probability = float(probabilities[SEATBELT_LABEL])
                active = self._update_state(timestamp, no_seatbelt_probability)
                active_violations = [SEATBELT_MISSING] if active else []
                snapshot = SeatbeltDetectorSnapshot(
                    timestamp,
                    frame_id,
                    self.model_version,
                    self.backend_name,
                    no_seatbelt_probability,
                    seatbelt_probability,
                    active_violations,
                    inference_ms,
                    roi,
                    self.dropped_frames,
                    "READY",
                    "",
                    bool(self.config.shadow_mode),
                )
                with self.lock:
                    self.snapshot_value = snapshot
                payload = {**snapshot.to_dict(), "probabilities": probabilities}
                self.telemetry.emit(
                    "SeatbeltDetector", "inference", payload,
                    monotonic_sec=timestamp, frame_id=frame_id,
                    level="WARNING" if active else "INFO",
                )
                if active and timestamp - self.last_emitted >= float(self.config.evidence_refresh_sec):
                    self.last_emitted = timestamp
                    details = {
                        "no_seatbelt_probability": no_seatbelt_probability,
                        "seatbelt_probability": seatbelt_probability,
                        "roi": roi,
                        "shadow_mode": bool(self.config.shadow_mode),
                    }
                    if self.config.shadow_mode:
                        self.telemetry.emit(
                            "SeatbeltDetector", "shadow_violation", details,
                            monotonic_sec=timestamp, frame_id=frame_id, level="WARNING",
                        )
                    else:
                        self.evidence_bus.publish(
                            EvidenceEvent(
                                SEATBELT_MISSING,
                                timestamp,
                                no_seatbelt_probability,
                                float(self.config.evidence_ttl_sec),
                                self.model_version,
                                details=details,
                            )
                        )
                        self.telemetry.emit(
                            "SeatbeltDetector", "evidence_published", details,
                            monotonic_sec=timestamp, frame_id=frame_id, level="WARNING",
                        )
            except Exception as exc:
                with self.lock:
                    self.snapshot_value.health = "FAILED"
                    self.snapshot_value.last_error = repr(exc)
                self.telemetry.emit(
                    "SeatbeltDetector", "inference_failed", {"error": repr(exc)},
                    monotonic_sec=timestamp, frame_id=frame_id, level="ERROR",
                )
            finally:
                self.queue.task_done()

    def snapshot(self) -> SeatbeltDetectorSnapshot:
        with self.lock:
            value = self.snapshot_value
            return SeatbeltDetectorSnapshot(
                value.monotonic_sec, value.frame_id, value.model_version, value.backend,
                value.no_seatbelt_probability, value.seatbelt_probability,
                list(value.active_violations), value.inference_ms, dict(value.roi),
                value.dropped_frames, value.health, value.last_error, value.shadow_mode,
            )

    def close(self) -> None:
        if self.worker is None:
            return
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
