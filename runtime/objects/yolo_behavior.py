"""Asynchronous YOLO driver-behavior detector with timestamp-based stabilization."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import platform
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


# NCNN inference releases Python execution into native worker threads.  The
# primary YOLO detector and the low-rate seat-belt classifier share this gate
# so they do not oversubscribe a Raspberry Pi while both are enabled.
NCNN_INFERENCE_LOCK = threading.Lock()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_yolo_backend(model_path: str | Path, backend: str = "auto") -> str:
    """Resolve configured/backend-auto values to runtime backend names."""
    configured = str(backend).lower()
    if configured == "ultralytics":
        return "pytorch"
    if configured in {"ncnn", "onnx", "pytorch"}:
        return configured
    if configured != "auto":
        raise ValueError(f"Unsupported behavior detector backend: {backend}")
    path = Path(model_path)
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name.endswith("_ncnn_model") or path.is_dir():
        return "ncnn"
    if suffix == ".onnx":
        return "onnx"
    if suffix == ".pt":
        return "pytorch"
    raise ValueError(
        "Could not infer behavior detector backend from model path "
        f"{model_path!s}; set behavior_detector.backend explicitly"
    )


def resolve_execution_mode(resolved_backend: str, execution_mode: str = "auto") -> str:
    configured = str(execution_mode).lower()
    if configured in {"process", "thread"}:
        return configured
    if configured != "auto":
        raise ValueError(f"Unsupported behavior detector execution_mode: {execution_mode}")
    return "process" if str(resolved_backend).lower() == "pytorch" else "thread"


def _is_arm_platform() -> bool:
    machine = platform.machine().lower()
    return machine.startswith("arm") or machine in {"aarch64", "arm64"}


def _read_json_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_ncnn_export_image_size(model_path: Path) -> int | None:
    candidates = (
        model_path / "ncnn_export_manifest.json",
        model_path / "metadata.json",
        model_path / "metadata.yaml",
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() == ".json":
            payload = _read_json_if_present(candidate)
            value = payload.get("imgsz") or payload.get("image_size")
            if isinstance(value, list):
                value = value[0] if value else None
            return int(value) if value is not None else None
        text = candidate.read_text(encoding="utf-8")
        for key in ("imgsz", "image_size"):
            for line in text.splitlines():
                if line.strip().startswith(f"{key}:"):
                    value = line.split(":", 1)[1].strip().strip("'\"")
                    if value.startswith("["):
                        value = value.strip("[]").split(",", 1)[0].strip()
                    return int(float(value))
    return None


def _find_ncnn_files(model_path: Path) -> tuple[Path, Path]:
    params = sorted(model_path.glob("*.param"))
    bins = sorted(model_path.glob("*.bin"))
    if not params or not bins:
        raise FileNotFoundError(
            f"NCNN model directory must contain .param and .bin files: {model_path}"
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
        layer_type = parts[0]
        try:
            input_count = int(parts[2])
            output_count = int(parts[3])
        except ValueError:
            continue
        cursor = 4
        layer_bottoms = parts[cursor:cursor + input_count]
        cursor += input_count
        layer_tops = parts[cursor:cursor + output_count]
        bottoms.update(layer_bottoms)
        tops.extend(layer_tops)
        if layer_type.lower() == "input":
            input_names.extend(layer_tops)
    output_names = [name for name in tops if name not in bottoms]
    if not input_names or not output_names:
        raise RuntimeError(f"Could not infer NCNN input/output blob names from {param_path}")
    return input_names[0], output_names[-1]


def _letterbox(frame: np.ndarray, image_size: int) -> tuple[np.ndarray, float, float, float]:
    height, width = frame.shape[:2]
    size = int(image_size)
    scale = min(size / max(height, 1), size / max(width, 1))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for YOLO preprocessing") from exc
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - resized_width) / 2.0
    pad_y = (size - resized_height) / 2.0
    left = int(round(pad_x - 0.1))
    top = int(round(pad_y - 0.1))
    canvas[top:top + resized_height, left:left + resized_width] = resized
    return canvas, scale, float(left), float(top)


def _clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    return boxes


def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    box_area = max(0.0, float((box[2] - box[0]) * (box[3] - box[1])))
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(box_area + areas - intersection, 1e-9)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    if len(boxes) == 0:
        return []
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    while len(order):
        current = int(order[0])
        keep.append(current)
        if len(order) == 1:
            break
        ious = _box_iou(boxes[current], boxes[order[1:]])
        order = order[1:][ious <= float(iou_threshold)]
    return keep


def _postprocess_yolo_output(
    output: Any,
    *,
    image_size: int,
    scale: float,
    pad_x: float,
    pad_y: float,
    original_width: int,
    original_height: int,
    names: dict[int, str],
    confidence: float,
    iou: float,
) -> list[ObjectDetection]:
    array = np.asarray(output, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or not array.size:
        return []
    class_count = len(names)
    if array.shape[0] in {class_count + 4, class_count + 5, class_count + 6} and array.shape[1] > array.shape[0]:
        array = array.T

    boxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    if array.shape[1] == 6:
        first_tail, second_tail = array[:, 4], array[:, 5]
        if np.nanmax(first_tail) <= 1.0:
            scores = first_tail
            class_ids = second_tail.astype(int)
        elif np.nanmax(second_tail) <= 1.0:
            class_ids = first_tail.astype(int)
            scores = second_tail
        else:
            scores = first_tail
            class_ids = second_tail.astype(int)
        boxes = array[:, :4].copy()
    elif array.shape[1] >= class_count + 4:
        raw_boxes = array[:, :4]
        if array.shape[1] >= class_count + 5 and np.nanmax(array[:, 4]) <= 1.0:
            class_scores = array[:, 5:5 + class_count] * array[:, 4:5]
        else:
            class_scores = array[:, 4:4 + class_count]
        class_ids = np.argmax(class_scores, axis=1).astype(int)
        scores = class_scores[np.arange(len(class_scores)), class_ids]
        cx, cy, width, height = raw_boxes.T
        boxes = np.column_stack(
            (cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)
        )
    else:
        return []

    valid = np.isfinite(scores) & (scores >= float(confidence))
    valid &= np.isfinite(boxes).all(axis=1)
    if not np.any(valid):
        return []
    boxes, scores, class_ids = boxes[valid], scores[valid], class_ids[valid]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / max(scale, 1e-9)
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / max(scale, 1e-9)
    boxes = _clip_boxes(boxes, original_width, original_height)
    sizes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    valid = sizes > 1.0
    boxes, scores, class_ids = boxes[valid], scores[valid], class_ids[valid]
    keep = _nms(boxes, scores, float(iou))
    detections: list[ObjectDetection] = []
    for index in keep:
        class_id = int(class_ids[index])
        if class_id not in names:
            continue
        detections.append(
            ObjectDetection(
                class_id,
                names[class_id],
                float(scores[index]),
                tuple(float(value) for value in boxes[index]),
            )
        )
    return detections


class UltralyticsYoloBackend:
    """Thin wrapper around the PyTorch/Ultralytics YOLO runtime."""

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


class NcnnYoloBackend:
    """NCNN runtime for an Ultralytics-exported YOLO detect model."""

    def __init__(
        self,
        model_path: Path,
        class_names: list[str],
        *,
        image_size: int,
        num_threads: int = 3,
    ):
        model_path = Path(model_path)
        if not model_path.is_dir():
            raise FileNotFoundError(f"NCNN model path must be an export directory: {model_path}")
        exported_size = _read_ncnn_export_image_size(model_path)
        if exported_size is None:
            raise RuntimeError(
                "NCNN export image size could not be verified. Re-export with "
                "runtime.objects.export_ncnn so ncnn_export_manifest.json is written."
            )
        if int(exported_size) != int(image_size):
            raise RuntimeError(
                "NCNN image_size mismatch: "
                f"config.behavior_detector.image_size={int(image_size)} but export imgsz={int(exported_size)}. "
                "Re-export NCNN with the configured image_size or update the config."
            )
        try:
            import ncnn
        except ImportError as exc:
            raise RuntimeError(
                "NCNN behavior detection requires the ncnn Python package. "
                "Install requirements-raspberry.txt."
            ) from exc
        self.ncnn = ncnn
        self.model_path = model_path
        self.image_size = int(image_size)
        self.names = {index: str(name) for index, name in enumerate(class_names)}
        param_path, bin_path = _find_ncnn_files(model_path)
        self.input_name, self.output_name = _parse_ncnn_blob_names(param_path)
        self.net = ncnn.Net()
        try:
            self.net.opt.num_threads = int(num_threads)
            self.net.opt.use_vulkan_compute = False
        except AttributeError:
            pass
        status = self.net.load_param(str(param_path))
        if status not in {0, None}:
            raise RuntimeError(f"Failed to load NCNN param file: {param_path}")
        status = self.net.load_model(str(bin_path))
        if status not in {0, None}:
            raise RuntimeError(f"Failed to load NCNN bin file: {bin_path}")

    def predict(self, frame: np.ndarray, *, image_size: int, confidence: float, iou: float):
        if int(image_size) != self.image_size:
            raise RuntimeError(
                "NCNN image_size mismatch during inference: "
                f"configured={int(image_size)} export={self.image_size}"
            )
        original_height, original_width = frame.shape[:2]
        letterboxed, scale, pad_x, pad_y = _letterbox(frame, self.image_size)
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for NCNN preprocessing") from exc
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        mat = self.ncnn.Mat.from_pixels(
            rgb,
            self.ncnn.Mat.PixelType.PIXEL_RGB,
            self.image_size,
            self.image_size,
        )
        mat.substract_mean_normalize([], [1 / 255.0, 1 / 255.0, 1 / 255.0])
        with NCNN_INFERENCE_LOCK:
            extractor = self.net.create_extractor()
            extractor.input(self.input_name, mat)
            result = extractor.extract(self.output_name)
        if isinstance(result, tuple):
            if result[0] != 0:
                raise RuntimeError(f"NCNN inference failed with status {result[0]}")
            output = result[1]
        else:
            output = result
        return _postprocess_yolo_output(
            output,
            image_size=self.image_size,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            original_width=original_width,
            original_height=original_height,
            names=self.names,
            confidence=confidence,
            iou=iou,
        )


class OnnxYoloBackend:
    """Optional ONNX Runtime backend using the same YOLO preprocessing contract."""

    def __init__(
        self,
        model_path: Path,
        class_names: list[str],
        *,
        image_size: int,
        num_threads: int = 3,
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "ONNX behavior detection requires onnxruntime. Install it or use NCNN/PyTorch."
            ) from exc
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        self.model_path = model_path
        self.image_size = int(image_size)
        self.names = {index: str(name) for index, name in enumerate(class_names)}
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(num_threads)
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, frame: np.ndarray, *, image_size: int, confidence: float, iou: float):
        if int(image_size) != self.image_size:
            raise RuntimeError(
                f"ONNX image_size mismatch during inference: configured={int(image_size)} expected={self.image_size}"
            )
        original_height, original_width = frame.shape[:2]
        letterboxed, scale, pad_x, pad_y = _letterbox(frame, self.image_size)
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for ONNX preprocessing") from exc
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
        outputs = self.session.run(None, {self.input_name: tensor})
        return _postprocess_yolo_output(
            outputs[0],
            image_size=self.image_size,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            original_width=original_width,
            original_height=original_height,
            names=self.names,
            confidence=confidence,
            iou=iou,
        )


def _build_yolo_backend(
    backend_name: str,
    model_path: Path,
    device: str,
    class_names: list[str],
    image_size: int,
    ncnn_num_threads: int,
):
    if backend_name == "pytorch":
        return UltralyticsYoloBackend(model_path, device)
    if backend_name == "ncnn":
        return NcnnYoloBackend(
            model_path, class_names, image_size=image_size, num_threads=ncnn_num_threads
        )
    if backend_name == "onnx":
        return OnnxYoloBackend(
            model_path, class_names, image_size=image_size, num_threads=ncnn_num_threads
        )
    raise ValueError(f"Unsupported behavior detector backend: {backend_name}")


def _isolated_yolo_main(
    model_path, device, backend_name, class_names, image_size, ncnn_num_threads, requests, responses
) -> None:
    """Own native detector state inside a spawned process when isolation is requested."""
    try:
        backend = _build_yolo_backend(
            str(backend_name),
            Path(model_path),
            str(device),
            list(class_names or []),
            int(image_size),
            int(ncnn_num_threads),
        )
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
    camera, drowsiness inference, Fusion and recording continue to operate.  The
    same proxy remains available for non-PyTorch backends when explicit process
    isolation is requested.
    """

    def __init__(
        self,
        model_path: Path,
        device="cpu",
        startup_timeout_sec=45.0,
        *,
        backend_name="pytorch",
        class_names: list[str] | None = None,
        image_size: int = 640,
        ncnn_num_threads: int = 3,
    ):
        self.context = mp.get_context("spawn")
        self.requests = self.context.Queue(maxsize=1)
        self.responses = self.context.Queue(maxsize=1)
        self.process = self.context.Process(
            target=_isolated_yolo_main,
            args=(
                str(model_path),
                str(device),
                str(backend_name),
                list(class_names or []),
                int(image_size),
                int(ncnn_num_threads),
                self.requests,
                self.responses,
            ),
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
        self.queue: queue.Queue[
            tuple[int, float, np.ndarray, tuple[float, float, float, float] | None] | None
        ] = queue.Queue(
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
        self.resolved_backend = "disabled"
        self.execution_mode = "disabled"
        self.base_interval_sec = float(config.inference_interval_sec)
        self.effective_interval_sec = self.base_interval_sec
        self.safe_mode_enabled = False
        self.safe_mode_interval_sec = self.base_interval_sec
        self.last_context_roi = float("-inf")
        self.context_hold_until = float("-inf")
        self.last_crop_info: dict[str, Any] = {}
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
            class_names = [str(value) for value in manifest.get("source_classes", [])]
            self.resolved_backend = resolve_yolo_backend(model_path, config.backend)
            if backend is not None:
                self.backend = backend
                execution_mode = "injected"
                self.resolved_backend = "injected"
            else:
                execution_mode = resolve_execution_mode(
                    self.resolved_backend, config.execution_mode
                )
                if self.resolved_backend == "pytorch" and _is_arm_platform():
                    self.telemetry.emit(
                        "BehaviorDetector",
                        "pytorch_backend_on_arm",
                        {
                            "model_path": str(model_path),
                            "message": (
                                "PyTorch/Ultralytics behavior backend is heavy on ARM; "
                                "use an NCNN export for Raspberry Pi 4."
                            ),
                        },
                        level="WARNING",
                    )
                if execution_mode == "process":
                    self.backend = ProcessIsolatedYoloBackend(
                        model_path,
                        config.device,
                        backend_name=self.resolved_backend,
                        class_names=class_names,
                        image_size=config.image_size,
                        ncnn_num_threads=config.ncnn_num_threads,
                    )
                else:
                    self.backend = _build_yolo_backend(
                        self.resolved_backend,
                        model_path,
                        config.device,
                        class_names,
                        config.image_size,
                        config.ncnn_num_threads,
                    )
            self.execution_mode = execution_mode
            names = {str(value) for value in getattr(self.backend, "names", {}).values()}
            missing = sorted(self.REQUIRED_SOURCE_CLASSES - names)
            if missing:
                raise RuntimeError(f"YOLO model is missing required classes: {missing}; names={sorted(names)}")
            self.backend_name = f"{self.resolved_backend}:{execution_mode}"
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
                    "resolved_backend": self.resolved_backend,
                    "execution_mode": execution_mode,
                    "model_path": str(model_path),
                    "classes": sorted(names),
                    "interval_sec": self.effective_interval_sec,
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

    def submit(
        self,
        packet: FramePacket,
        face_bbox_xyxy: tuple[float, float, float, float] | None = None,
    ) -> None:
        if not self.enabled or self.worker is None:
            return
        if packet.monotonic_sec - self.last_submitted < self.effective_interval_sec:
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

    def set_safe_mode(self, enabled: bool, interval_sec: float | None = None) -> None:
        enabled = bool(enabled)
        previous = self.safe_mode_enabled
        self.safe_mode_enabled = enabled
        if interval_sec is not None:
            maximum = float(self.config.max_inference_interval_sec)
            self.safe_mode_interval_sec = min(maximum, max(self.base_interval_sec, float(interval_sec)))
        self.effective_interval_sec = (
            self.safe_mode_interval_sec if enabled else self.base_interval_sec
        )
        if enabled != previous:
            self.telemetry.emit(
                "BehaviorDetector",
                "safe_mode_changed",
                {
                    "enabled": enabled,
                    "effective_interval_sec": self.effective_interval_sec,
                },
            )

    def _static_crop(self, frame: np.ndarray):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = self.config.driver_roi
        px1, py1 = int(round(x1 * width)), int(round(y1 * height))
        px2, py2 = int(round(x2 * width)), int(round(y2 * height))
        return frame[py1:py2, px1:px2], px1, py1, {
            "mode": "static",
            "crop_xyxy": [px1, py1, px2, py2],
            "crop_width": max(0, px2 - px1),
            "crop_height": max(0, py2 - py1),
        }

    def _full_crop(self, frame: np.ndarray, reason: str | None = None):
        height, width = frame.shape[:2]
        info = {
            "mode": "full",
            "crop_xyxy": [0, 0, width, height],
            "crop_width": width,
            "crop_height": height,
        }
        if reason:
            info["reason"] = reason
        return frame, 0, 0, info

    def _face_crop(
        self,
        frame: np.ndarray,
        face_bbox_xyxy: tuple[float, float, float, float] | None,
    ):
        height, width = frame.shape[:2]
        if face_bbox_xyxy is None:
            if self.config.roi_fallback_static_on_no_face:
                crop, x, y, info = self._static_crop(frame)
                info["mode"] = "static"
                info["fallback_reason"] = "no_face"
                return crop, x, y, info
            return self._full_crop(frame, "no_face")
        x1, y1, x2, y2 = (float(value) for value in face_bbox_xyxy)
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        x1, x2 = np.clip([x1, x2], 0.0, float(width))
        y1, y2 = np.clip([y1, y2], 0.0, float(height))
        face_width, face_height = x2 - x1, y2 - y1
        if face_width <= 1.0 or face_height <= 1.0:
            if self.config.roi_fallback_static_on_no_face:
                crop, x, y, info = self._static_crop(frame)
                info["mode"] = "static"
                info["fallback_reason"] = "invalid_face_bbox"
                return crop, x, y, info
            return self._full_crop(frame, "invalid_face_bbox")

        side = max(face_width, face_height) * float(self.config.roi_scale)
        side = max(side, float(self.config.roi_min_size))
        side = min(side, float(max(width, height)))
        center_x = (x1 + x2) / 2.0
        # Bias slightly downward so the crop favors mouth/hand evidence over forehead.
        center_y = (y1 + y2) / 2.0 + 0.10 * face_height
        crop_x1 = center_x - side / 2.0
        crop_y1 = center_y - side / 2.0
        crop_x2 = crop_x1 + side
        crop_y2 = crop_y1 + side
        if crop_x1 < 0:
            crop_x2 -= crop_x1
            crop_x1 = 0.0
        if crop_y1 < 0:
            crop_y2 -= crop_y1
            crop_y1 = 0.0
        if crop_x2 > width:
            crop_x1 -= crop_x2 - width
            crop_x2 = float(width)
        if crop_y2 > height:
            crop_y1 -= crop_y2 - height
            crop_y2 = float(height)
        crop_x1, crop_y1 = max(0.0, crop_x1), max(0.0, crop_y1)
        crop_x2, crop_y2 = min(float(width), crop_x2), min(float(height), crop_y2)
        px1, py1 = int(np.floor(crop_x1)), int(np.floor(crop_y1))
        px2, py2 = int(np.ceil(crop_x2)), int(np.ceil(crop_y2))
        if px2 <= px1 or py2 <= py1:
            return self._full_crop(frame, "empty_face_roi")
        info = {
            "mode": "face",
            "crop_xyxy": [px1, py1, px2, py2],
            "crop_width": px2 - px1,
            "crop_height": py2 - py1,
            "face_bbox_xyxy": [x1, y1, x2, y2],
            "roi_scale": float(self.config.roi_scale),
        }
        return frame[py1:py2, px1:px2], px1, py1, info

    def _crop(
        self,
        frame: np.ndarray,
        face_bbox_xyxy: tuple[float, float, float, float] | None = None,
    ):
        mode = str(self.config.roi_mode).lower()
        if mode == "full":
            return self._full_crop(frame)
        if mode == "static":
            return self._static_crop(frame)
        return self._face_crop(frame, face_bbox_xyxy)

    def _update_effective_interval(self, inference_ms: float, timestamp: float, frame_id: int) -> None:
        if not self.config.adaptive_interval or self.safe_mode_enabled:
            return
        old = float(self.effective_interval_sec)
        budget = float(self.config.latency_budget_ms)
        base = float(self.base_interval_sec)
        maximum = float(self.config.max_inference_interval_sec)
        if inference_ms > budget:
            new = min(maximum, max(old * 1.25, old + base * 0.25))
        elif inference_ms < budget * 0.70 and old > base:
            new = max(base, min(old * 0.85, old - base * 0.25))
        else:
            return
        if abs(new - old) < 1e-4:
            return
        self.effective_interval_sec = float(new)
        self.telemetry.emit(
            "BehaviorDetector",
            "interval_changed",
            {
                "previous_interval_sec": old,
                "effective_interval_sec": self.effective_interval_sec,
                "base_interval_sec": base,
                "max_interval_sec": maximum,
                "latency_budget_ms": budget,
                "inference_ms": float(inference_ms),
            },
            monotonic_sec=timestamp,
            frame_id=frame_id,
        )

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
            frame_id, timestamp, frame, face_bbox_xyxy = item
            started = time.perf_counter()
            try:
                context_interval = float(
                    getattr(self.config, "context_roi_interval_sec", 2.0)
                )
                context_due = timestamp - self.last_context_roi >= context_interval
                context_followup = timestamp <= self.context_hold_until
                if (
                    str(self.config.roi_mode).lower() == "face"
                    and (context_due or context_followup)
                ):
                    crop, offset_x, offset_y, crop_info = self._static_crop(frame)
                    crop_info["mode"] = (
                        "context_reacquisition" if context_due else "context_followup"
                    )
                    crop_info["context_interval_sec"] = context_interval
                    if context_due:
                        self.last_context_roi = timestamp
                else:
                    crop, offset_x, offset_y, crop_info = self._crop(
                        frame, face_bbox_xyxy
                    )
                self.last_crop_info = dict(crop_info)
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
                if (
                    crop_info.get("mode")
                    in {"context_reacquisition", "context_followup"}
                    and detections
                ):
                    self.context_hold_until = max(
                        self.context_hold_until,
                        timestamp + float(self.config.temporal_window_sec),
                    )
                    crop_info["context_hold_until"] = self.context_hold_until
                active, ratios, events = self.filter.update(
                    timestamp, detections, frame_id=frame_id
                )
                for event in events:
                    self.evidence_bus.publish(event)
                inference_ms = (time.perf_counter() - started) * 1000.0
                self._update_effective_interval(inference_ms, timestamp, frame_id)
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
                    {
                        **snapshot.to_dict(),
                        "roi": crop_info,
                        "effective_interval_sec": self.effective_interval_sec,
                    },
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
