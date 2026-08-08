# Raspberry Pi 4 Performance Plan

This is a planning document only. No production runtime code should be changed until the plan is explicitly approved.

## Goal And Guardrails

Goal: make the runtime feasible on a Raspberry Pi 4 without changing decision logic, model behavior, feature semantics, thresholds, event timing, fusion timing, or the LightGBM bundle contract.

Primary rule: all changes reduce runtime cost around inference, frame preparation, scheduling, and resource caps. The drowsiness model, event formulas, fusion thresholds, evidence semantics, and alarm behavior remain unchanged.

## 1. File-By-File Inventory

### `runtime/objects/yolo_behavior.py`

Current behavior:

- `UltralyticsYoloBackend` is the only real in-process backend.
- The class comment mentions exported NCNN directories, but loading always imports `ultralytics.YOLO`.
- `ProcessIsolatedYoloBackend` always wraps `UltralyticsYoloBackend`.
- `YoloBehaviorDetector` selects process vs thread only from `config.execution_mode`.
- `config.backend` is only surfaced in telemetry as a label; it is not a resolver.
- `_crop()` always uses static normalized `driver_roi`.
- `submit()` throttles using fixed `config.inference_interval_sec`.

Proposed behavior:

- Add backend resolution helpers:
  - `backend="auto"` resolves from `model_path`.
  - `*_ncnn_model` or a directory resolves to `ncnn`.
  - `*.onnx` resolves to `onnx`.
  - `*.pt` resolves to `pytorch`.
  - keep legacy `backend="ultralytics"` as an alias for `pytorch` so existing configs continue to load.
- Add `NcnnYoloBackend` next to `UltralyticsYoloBackend`.
  - It imports `ncnn` lazily.
  - It loads the exported NCNN files from the export directory.
  - It exposes the same `names` and `predict(frame, image_size, confidence, iou)` interface.
  - It receives `ncnn_num_threads` from config.
- Add an optional `OnnxYoloBackend` path or a clear optional-dependency failure for `*.onnx`.
  - If implemented with `onnxruntime`, import it lazily and keep it out of the Pi default dependency set unless separately required.
- Keep `UltralyticsYoloBackend` for PyTorch `.pt`.
- Keep `ProcessIsolatedYoloBackend` intact and selectable.
- Add `execution_mode="auto"` handling:
  - resolved `ncnn` or `onnx` defaults to `thread`.
  - resolved `pytorch` defaults to `process`.
  - explicit `process` and `thread` still override.
- Emit a warning when the resolved backend is `pytorch` on ARM/aarch64.
- Validate NCNN export image size at load.
  - Preferred source: a small export manifest written by `export_ncnn.py`, for example `ncnn_export_manifest.json`.
  - Fallback source: Ultralytics export metadata if present.
  - If no reliable export size is available, fail clearly and instruct to re-export with the updated exporter.
  - If `config.image_size` differs from the export image size, fail before starting the worker.
- Add face-driven ROI support:
  - `submit(packet, face_bbox_xyxy=None)` remains backward compatible because the new argument is optional.
  - The queued item carries the face bbox snapshot, avoiding a circular import.
  - `_crop(frame, face_bbox_xyxy)` supports `roi_mode`.
  - `static` preserves current `driver_roi`.
  - `full` uses the full frame.
  - `face` expands the original-frame face bbox by `roi_scale`, makes it square, clamps to frame bounds, enforces `roi_min_size`, and falls back to static when allowed and no usable face bbox exists.
  - The returned `(crop, offset_x, offset_y)` contract stays unchanged so full-frame box remapping is preserved.
- Log effective ROI mode and crop size on each inference telemetry record, and at least on every crop-mode fallback.
- Add adaptive interval state:
  - Track `effective_inference_interval_sec`, initially the configured base interval.
  - When `inference_ms > latency_budget_ms`, increase the effective interval up to `max_inference_interval_sec`.
  - When `inference_ms` is comfortably under budget, decrease toward the configured base.
  - Emit `BehaviorDetector/interval_changed` telemetry on every interval change.
  - Use this only in `submit()` throttling. Evidence timestamps, temporal windows, fusion, event, and alarm timing remain monotonic-second based and unchanged.

### `runtime/objects/export_ncnn.py`

Current behavior:

- Exports a `.pt` model to NCNN with Ultralytics.
- Prints the output directory.
- Does not write runtime-readable export metadata.

Proposed behavior:

- Preserve the existing CLI.
- After export, write an export manifest inside the NCNN directory:
  - `source_model`
  - `source_sha256`
  - `imgsz`
  - `half`
  - `export_format`
  - `created_by`
- The runtime NCNN backend will use this for load-time `image_size` validation.
- No Pi dependency on Ultralytics is introduced; this exporter remains a laptop/export tool.

### `runtime/objects/__init__.py`

Current behavior:

- Exports `ProcessIsolatedYoloBackend`, `TemporalBehaviorFilter`, `UltralyticsYoloBackend`, and `YoloBehaviorDetector`.

Proposed behavior:

- Export `NcnnYoloBackend` and backend resolver helpers if tests or docs need them.
- Keep current exports for compatibility.

### `runtime/perception/mediapipe_face.py`

Current behavior:

- Converts the full-resolution BGR frame to full-resolution RGB.
- Sends the full frame to MediaPipe FaceLandmarker.
- Computes all normalized ratios and pixel safety gates against the same full frame dimensions.
- `_head_pose()` uses full frame width/height.

Proposed behavior:

- Add `process_width` to the constructor.
  - `process_width=256` by default.
  - `process_width=0` preserves native full-resolution behavior.
- If `process_width > 0` and the frame is wider than that, resize the BGR frame to that width with proportional height using `cv2.INTER_AREA`.
- Convert only the processed frame to RGB and send that to MediaPipe.
- Continue to compute these values against the original frame dimensions:
  - `face_width_px`
  - `face_height_px`
  - `interocular_distance_px`
  - `left_eye_width_px`
  - `right_eye_width_px`
  - `eye_resolution_valid`
  - `driver_distance_status`
- Keep `_head_pose(landmarks, width, height)` on original frame dimensions.
- Store the latest original-frame face bbox in an internal side channel, for example:
  - `self.latest_face_bbox_xyxy: tuple[float, float, float, float] | None`
  - `self.latest_face_bbox_frame_id: int | None`
  - `self.latest_face_bbox_monotonic_sec: float | None`
- Do not add a bbox field to `FaceSignal` unless approval explicitly allows a contract extension. The plan avoids changing `shared/contracts.py` semantics.

### `runtime/app.py`

Current behavior:

- Imports runtime modules at module import time, including modules that import OpenCV, MediaPipe, LightGBM, and YOLO wrappers.
- Loads config inside `run()`.
- Submits a frame to `behavior_detector` before running `perception.process(packet)`.
- Does not set OpenCV/OMP thread caps.

Proposed behavior:

- Refactor the composition root so thread caps are applied before heavy imports:
  - keep stdlib and `runtime.config` import at the top.
  - parse config path.
  - load config or minimally read cap keys.
  - set OMP/OpenCV caps.
  - import heavy runtime modules after caps are applied.
- Apply:
  - `OMP_NUM_THREADS`
  - `OPENBLAS_NUM_THREADS`
  - `MKL_NUM_THREADS`
  - `NUMEXPR_NUM_THREADS`
  - `cv2.setNumThreads(config.cv_num_threads)`
- Pass `config.ncnn_num_threads` through the behavior detector config.
- Pass `config.perception.process_width` into `MediaPipeFacePerception`.
- Reorder behavior submission to use same-frame face ROI:
  - run `signal = perception.process(packet)` first.
  - pass `perception.latest_face_bbox_xyxy` into `behavior_detector.submit(packet, face_bbox_xyxy=...)`.
  - continue event, feature, fusion, recorder, alarm, HMI, and telemetry timing exactly as monotonic-second based logic does today.
- Include ROI crop info and adaptive interval state in telemetry metadata, not in decision inputs.

### `runtime/config.py`

Current behavior:

- `PerceptionConfig` has no `process_width`.
- `BehaviorDetectorConfig.backend` defaults to `"ultralytics"` and validation only allows `"ultralytics"`.
- `BehaviorDetectorConfig.execution_mode` defaults to `"process"` and validation allows only `process`/`thread`.
- Static `driver_roi` is always required.
- No adaptive interval keys.
- No thread cap keys.

Proposed behavior:

- Add top-level thread cap keys to `RuntimeConfig`:
  - `cv_num_threads: int = 2`
  - `omp_num_threads: int = 2`
  - `ncnn_num_threads: int = 3`
- Add `PerceptionConfig.process_width: int = 256`.
- Change behavior defaults:
  - `backend: str = "auto"`
  - `execution_mode: str = "auto"`
- Extend behavior config keys:
  - `roi_mode: str = "face"`
  - `roi_scale: float = 1.6`
  - `roi_min_size: int = 96`
  - `roi_fallback_static_on_no_face: bool = True`
  - `adaptive_interval: bool = True`
  - `latency_budget_ms: float = 250.0`
  - `max_inference_interval_sec: float = 1.5`
- Validation updates:
  - `backend` in `{"auto", "ncnn", "onnx", "pytorch", "ultralytics"}`.
  - `execution_mode` in `{"auto", "process", "thread"}`.
  - `roi_mode` in `{"face", "static", "full"}`.
  - `roi_scale >= 1.0`.
  - `roi_min_size > 0`.
  - `max_inference_interval_sec >= inference_interval_sec`.
  - `latency_budget_ms > 0`.
  - all thread caps are positive integers.
  - `process_width >= 0`.

### `configs/runtime.example.json`

Current behavior:

- Behavior detector defaults to the `.pt` artifact.
- `backend` is `"ultralytics"`.
- `execution_mode` is `"process"`.
- No process downscale, dynamic ROI, adaptive interval, or thread cap keys.

Proposed behavior:

- Default `behavior_detector.model_path` to:
  - `models/driver_behavior/luthfi_yolo11n/best_yolo11n_ncnn_model`
- Set:
  - `behavior_detector.backend: "auto"`
  - `behavior_detector.execution_mode: "auto"`
  - `perception.process_width: 256`
  - `behavior_detector.roi_mode: "face"`
  - `behavior_detector.roi_scale: 1.6`
  - `behavior_detector.roi_min_size: 96`
  - `behavior_detector.roi_fallback_static_on_no_face: true`
  - `behavior_detector.adaptive_interval: true`
  - `behavior_detector.latency_budget_ms: 250.0`
  - `behavior_detector.max_inference_interval_sec: 1.5`
  - top-level `cv_num_threads: 2`
  - top-level `omp_num_threads: 2`
  - top-level `ncnn_num_threads: 3`
- Keep `image_size: 640` to avoid changing cigarette recall assumptions without re-measurement.

### `configs/runtime.raspberry_pi4.json`

Current behavior:

- File does not exist.

Proposed behavior:

- Add a Pi-4-oriented profile, instead of changing the existing laptop/replay profiles.
- Use NCNN default path and `backend: "auto"`.
- Use `execution_mode: "auto"` so NCNN resolves to in-process thread mode.
- Use `perception.process_width: 256`.
- Keep `behavior_detector.image_size: 640`.
- Use `behavior_detector.inference_interval_sec: 0.75` as a Pi 4 base interval, with adaptive interval enabled up to `1.5`.
- Keep `queue_size: 1` for behavior detector latest-frame behavior.
- Keep `recorder.enabled: true`; do not disable incident recording.
- Bound recorder/evaluation queues:
  - `recorder.queue_size: 64`
  - `evaluation.queue_size: 128`
- Set:
  - `evaluation.record_full_session: false`
  - `telemetry.mode: "NORMAL"`
  - `hmi.enabled: false` by default for headless Pi runs
  - `alarm.mode: "LOG_ONLY"` unless local audio is explicitly enabled by the operator
- Include thread caps:
  - `cv_num_threads: 2`
  - `omp_num_threads: 2`
  - `ncnn_num_threads: 3`

### `configs/runtime.laptop_alarm_demo.json`

Current behavior:

- Uses `.pt`, `backend: "ultralytics"`, `execution_mode: "process"`.

Proposed behavior:

- Leave unchanged unless tests prove config validation needs an explicit migration.
- Support legacy `backend: "ultralytics"` as a PyTorch alias so this config preserves current behavior.

### `configs/runtime.replay_silent.json`

Current behavior:

- Uses `.pt`, `backend: "ultralytics"`, `execution_mode: "process"`.

Proposed behavior:

- Leave unchanged for replay parity.
- Support legacy `backend: "ultralytics"` as a PyTorch alias.

### `requirements-raspberry.txt`

Current behavior:

- Includes `pandas`.
- Includes `ultralytics`, which brings a heavy PyTorch dependency stack.
- Does not include `ncnn`.

Proposed behavior:

- Remove `pandas`.
- Remove `ultralytics`.
- No explicit `torch` line exists today; do not add one.
- Add `ncnn`.
- Keep NumPy, OpenCV headless, MediaPipe, LightGBM, psutil, and pytest.

### `requirements-laptop.txt`

Current behavior:

- Includes `pandas` and `ultralytics`; Ultralytics supplies the PyTorch-backed export/training path.

Proposed behavior:

- Keep `pandas` and `ultralytics`.
- Leave the laptop file otherwise unchanged unless implementation needs an optional ONNX tooling dependency.

### `runtime/features/v3_runtime.py`

Current behavior:

- Already NumPy-only.
- Comment documents parity with pandas rolling behavior.

Proposed behavior:

- No implementation change expected.
- Keep the pandas-parity comment because it documents a feature-contract equivalence.

### `runtime/evaluation/evaluate.py`

Current behavior:

- Imports pandas for reading `frame_timestamps.csv` and writing several CSV outputs.

Proposed behavior:

- Replace pandas usage with `csv` plus NumPy arrays.
- Preserve all output filenames, column names, ordering, numeric values, and report contents.
- This removes pandas from `runtime/` so Pi installs do not need it for runtime package importability.

### `runtime/model/derive_reference_bundle.py`

Current behavior:

- Imports pandas inside `derive_reference_bundle()` to read a training CSV and compute q01/q05/median/q95/q99/missing-rate feature reference metadata.

Proposed behavior:

- Replace pandas usage with `csv.DictReader` and NumPy numeric conversion.
- Match pandas default quantile interpolation with `np.nanquantile(..., method="linear")`.
- Preserve generated `feature_reference.json` values within floating-point tolerance.
- Keep LightGBM dependency because this is model-bundle tooling, not live inference.

### `tests/test_behavior_detector.py`

Current behavior:

- Tests temporal filtering, fusion independence of object violations, and async detector contract with an injected backend.
- Uses `.pt` path in config and `backend` default behavior.

Proposed behavior:

- Add tests for backend resolver:
  - NCNN directory resolves to `ncnn`.
  - `.onnx` resolves to `onnx`.
  - `.pt` resolves to `pytorch`.
  - legacy `ultralytics` resolves as PyTorch.
- Add tests for `execution_mode="auto"`:
  - NCNN/ONNX -> thread.
  - PyTorch -> process.
- Add tests for static/full/face ROI crop and offset remapping.
- Add tests for adaptive interval growth/shrink telemetry using an injected backend with controlled latency.
- Keep existing temporal behavior tests unchanged.

### `tests/test_eye_calibration.py`

Current behavior:

- Tests pixel geometry with original 640x480 dimensions.

Proposed behavior:

- Add or extend tests to prove perception downscale does not change pixel-gate dimensions:
  - fake landmarks normalized to a 640x480 original frame.
  - process image may be 256px wide.
  - eye widths and interocular distance are still computed as original-frame pixels.
- Keep EAR/MAR/head-pose formulas unchanged.

### `tests/test_model_drift.py`

Current behavior:

- Verifies drift monitor behavior around feature reference ranges.

Proposed behavior:

- No production behavior change expected.
- Use existing tests as a guard that drift thresholds and state transitions were not loosened.

### `tests/test_evaluation.py`

Current behavior:

- Uses pandas in tests to create/read evaluation CSVs.

Proposed behavior:

- Existing test code can remain pandas-based because laptop/dev requirements keep pandas.
- Add value-parity assertions if `runtime/evaluation/evaluate.py` is rewritten from pandas to csv/NumPy.

### `docs/YOLO_BEHAVIOR_INTEGRATION_AR.md`

Current behavior:

- Documents PyTorch/Ultralytics process isolation as recommended.
- Documents NCNN export for Raspberry Pi 5.
- Warns not to reduce `image_size` below 480 without measuring cigarette recall.
- Warns that a wrong ROI can improve FPS while destroying recall.

Proposed behavior:

- Update after implementation approval to document Pi 4 defaults:
  - NCNN as default.
  - `backend: "auto"`.
  - `execution_mode: "auto"`.
  - dynamic ROI modes.
  - adaptive interval.
  - `process_width`.
  - Pi 4 profile.
- Keep the recall warnings prominent.

## 2. New Config Keys, Defaults, And Locations

Top-level `RuntimeConfig` keys:

| Key | Default | Lands in |
|---|---:|---|
| `cv_num_threads` | `2` | dataclass, `runtime.example.json`, `runtime.raspberry_pi4.json` |
| `omp_num_threads` | `2` | dataclass, `runtime.example.json`, `runtime.raspberry_pi4.json` |
| `ncnn_num_threads` | `3` | dataclass, `runtime.example.json`, `runtime.raspberry_pi4.json` |

`perception` keys:

| Key | Default | Lands in |
|---|---:|---|
| `process_width` | `256` | dataclass, `runtime.example.json`, `runtime.raspberry_pi4.json` |

`behavior_detector` keys:

| Key | Default | Lands in |
|---|---:|---|
| `backend` | `"auto"` | dataclass, `runtime.example.json`, `runtime.raspberry_pi4.json` |
| `execution_mode` | `"auto"` | dataclass, `runtime.example.json`, `runtime.raspberry_pi4.json` |
| `roi_mode` | `"face"` | dataclass, `runtime.example.json`, `runtime.raspberry_pi4.json` |
| `roi_scale` | `1.6` | dataclass, `runtime.example.json`, `runtime.raspberry_pi4.json` |
| `roi_min_size` | `96` | dataclass, `runtime.example.json`, `runtime.raspberry_pi4.json` |
| `roi_fallback_static_on_no_face` | `true` | dataclass, `runtime.example.json`, `runtime.raspberry_pi4.json` |
| `adaptive_interval` | `true` | dataclass, `runtime.example.json`, `runtime.raspberry_pi4.json` |
| `latency_budget_ms` | `250.0` | dataclass, `runtime.example.json`, `runtime.raspberry_pi4.json` |
| `max_inference_interval_sec` | `1.5` | dataclass, `runtime.example.json`, `runtime.raspberry_pi4.json` |

Existing keys with changed defaults:

| Key | Current default | Proposed default |
|---|---:|---:|
| `behavior_detector.model_path` | `models/driver_behavior/luthfi_yolo11n/best_yolo11n.pt` | `models/driver_behavior/luthfi_yolo11n/best_yolo11n_ncnn_model` |
| `behavior_detector.backend` | `"ultralytics"` | `"auto"` |
| `behavior_detector.execution_mode` | `"process"` | `"auto"` |

Pi 4 profile-only values:

| Key | Value |
|---|---:|
| `behavior_detector.inference_interval_sec` | `0.75` |
| `behavior_detector.max_inference_interval_sec` | `1.5` |
| `recorder.queue_size` | `64` |
| `evaluation.queue_size` | `128` |
| `evaluation.record_full_session` | `false` |
| `telemetry.mode` | `"NORMAL"` |
| `hmi.enabled` | `false` |

## 3. Risk Analysis

### Change 1: NCNN Default Backend

Risks:

- NCNN `imgsz` mismatch: the exported graph has a fixed input size. If runtime `image_size` differs, detections can be wrong or preprocessing can fail.
- Class-order mismatch: NCNN output class IDs must map to the same names as the `.pt` manifest.
- Optional dependency mismatch: `ncnn` Python API differences can cause startup failure.
- PyTorch fallback on ARM can still consume too much memory if selected.

Mitigations:

- Load-time `image_size` validation against export manifest or export metadata.
- Fail loudly if the export size cannot be verified.
- Validate required classes `phone`, `cigarette`, and `drink_or_food` exactly as today.
- Preserve checksum validation for file artifacts where applicable; for directories, validate export manifest source checksum.
- Emit ARM warning for PyTorch backend.
- Keep `ProcessIsolatedYoloBackend` available and keep auto process isolation for PyTorch.

### Change 2: Perception Downscale

Risks:

- Pixel-gate regression: if pixel geometry is accidentally computed against downscaled dimensions, `interocular_distance_px`, `left_eye_width_px`, and `right_eye_width_px` shrink and can falsely set `driver_distance_status="TOO_FAR"`.
- Feature distribution drift: lower-resolution MediaPipe input may shift landmarks enough to change EAR, MAR, gaze, head pose, velocity, and therefore V3 feature distributions.
- Head-pose regression: using processed dimensions in the camera matrix would alter pitch/yaw/roll.

Mitigations:

- Keep all geometry and `_head_pose()` dimensions as original frame width/height.
- Add a unit test specifically proving pixel geometry remains original-frame pixels under downscale.
- Do not loosen `FeatureDriftMonitor` thresholds.
- Run golden sample verification and drift tests.
- Treat `MODEL_INPUT_OOD` caused by downscale as a task failure, not as a threshold-tuning opportunity.
- Roll back with `perception.process_width: 0`.

### Change 3: Dynamic Face ROI

Risks:

- ROI recall loss, especially cigarette: small objects near the mouth or hand can be cropped out.
- Wrong ROI may improve FPS while hiding violations.
- Face-only crop may miss hands, phone, food, or cigarette.
- No-face fallback may change behavior if it silently switches to full frame or empty crop.

Mitigations:

- Default `image_size` remains 640.
- Default `roi_scale=1.6` and square crop will include the face plus mouth area; implementation should bias crop enough to include lower-face/mouth and nearby hand area.
- Enforce `roi_min_size`.
- Fallback to static ROI when no usable face bbox exists and `roi_fallback_static_on_no_face=true`.
- Log effective crop size, mode, fallback reason, and face bbox age.
- Add tests that crop offsets preserve full-frame detection coordinates.
- Acceptance must include annotated phone/eating/smoking replay or live checks, with particular attention to cigarette recall.
- Roll back with `behavior_detector.roi_mode: "static"` and current `driver_roi: [0.0, 0.0, 1.0, 1.0]`, or `roi_mode: "full"`.

### Change 4: Adaptive Inference Interval

Risks:

- Fewer YOLO samples may delay object violation onset or reduce temporal sample count.
- If adaptive timing leaks into fusion or events, it could alter safety-critical state timing.
- Excessive oscillation could make telemetry noisy and behavior detector latency hard to reason about.

Mitigations:

- Apply adaptive interval only to behavior-detector frame submission.
- Do not change `TemporalBehaviorFilter` thresholds or timestamps.
- Do not change event/fusion/alarm timing.
- Use bounded, gradual interval changes.
- Emit telemetry only on interval changes.
- Roll back with `behavior_detector.adaptive_interval: false`.

### Change 5: Thread Caps

Risks:

- Setting caps too late means native libraries may already have initialized their pools.
- Setting caps too low can reduce throughput on laptop or Pi 5.
- Setting NCNN threads too high can starve capture/perception.

Mitigations:

- Move heavy imports after config cap application in `runtime/app.py`.
- Keep defaults Pi-4 focused and configurable.
- Use `ncnn_num_threads=3` to leave CPU for capture/perception on four Cortex-A72 cores.
- Roll back by increasing caps in config, for example `cv_num_threads: 0` only if implementation explicitly treats zero as OpenCV default. Otherwise set higher positive values.

## 4. Ordered Implementation Sequence And Verification

Step 1: Config schema and examples.

- Update `runtime/config.py`.
- Update `configs/runtime.example.json`.
- Add `configs/runtime.raspberry_pi4.json`.
- Keep existing laptop/replay configs accepted through legacy alias support.

Verification:

```bash
PYTHONPATH=. python3 -m compileall -q dms_final_system
PYTHONPATH=. pytest dms_final_system/tests/test_behavior_detector.py dms_final_system/tests/test_eye_calibration.py
```

Step 2: Thread caps in app startup.

- Refactor `runtime/app.py` import order.
- Apply OMP/OpenCV caps before heavy imports.
- Pass `ncnn_num_threads` through behavior config.

Verification:

```bash
PYTHONPATH=. python3 -m compileall -q dms_final_system
PYTHONPATH=. pytest dms_final_system/tests/test_alarm_controller.py dms_final_system/tests/test_hmi.py dms_final_system/tests/test_recorder.py
```

Step 3: Perception downscale and face bbox side channel.

- Add `process_width`.
- Resize before RGB conversion only.
- Preserve original dimensions for pixel gates and head pose.
- Store latest original-frame bbox internally.

Verification:

```bash
PYTHONPATH=. pytest dms_final_system/tests/test_eye_calibration.py dms_final_system/tests/test_runtime_features.py dms_final_system/tests/test_events_fusion.py
```

Step 4: Dynamic ROI in behavior detector.

- Add optional face bbox argument to `submit()`.
- Add `roi_mode` crop logic.
- Preserve detection offset remapping.
- Emit crop-size telemetry.

Verification:

```bash
PYTHONPATH=. pytest dms_final_system/tests/test_behavior_detector.py dms_final_system/tests/test_events_fusion.py
```

Step 5: Backend resolver, NCNN backend, export manifest, and ARM warning.

- Add resolver and NCNN backend.
- Update exporter to write manifest.
- Validate NCNN `image_size` at load.
- Keep PyTorch process isolation intact.

Verification:

```bash
PYTHONPATH=. python3 -m compileall -q dms_final_system
PYTHONPATH=. pytest dms_final_system/tests/test_behavior_detector.py
PYTHONPATH=. python3 -m dms_final_system.runtime.objects.verify_model --bundle dms_final_system/models/driver_behavior/luthfi_yolo11n
```

Optional laptop export verification after dependencies and export artifact are available:

```bash
PYTHONPATH=. python3 -m dms_final_system.runtime.objects.export_ncnn --model dms_final_system/models/driver_behavior/luthfi_yolo11n/best_yolo11n.pt --imgsz 640
```

Step 6: Adaptive interval.

- Add effective interval state.
- Emit telemetry on interval changes.
- Keep all evidence/fusion/event timestamps unchanged.

Verification:

```bash
PYTHONPATH=. pytest dms_final_system/tests/test_behavior_detector.py dms_final_system/tests/test_telemetry.py
```

Step 7: Remove pandas from runtime and Pi requirements.

- Replace pandas usage in `runtime/evaluation/evaluate.py`.
- Replace pandas usage in `runtime/model/derive_reference_bundle.py`.
- Update requirements files.

Verification:

```bash
PYTHONPATH=. pytest dms_final_system/tests/test_evaluation.py dms_final_system/tests/test_model_deployment.py dms_final_system/tests/test_model_drift.py
```

Step 8: Full regression.

Verification:

```bash
PYTHONPATH=. python3 -m compileall -q dms_final_system
PYTHONPATH=. pytest dms_final_system/tests
```

Step 9: Runtime smoke tests.

Verification:

```bash
PYTHONPATH=. python3 -m dms_final_system.runtime.app --verify-bundle dms_final_system/models/drowsiness/20260731T140605Z_full_runtimefix1
PYTHONPATH=. python3 -m dms_final_system.runtime.app --config dms_final_system/configs/runtime.raspberry_pi4.json --replay /path/to/replay.mp4
```

## 5. Estimated Per-Frame Savings On Cortex-A72 At 1.5GHz

These are estimates, not benchmark claims. Actual values depend on camera driver, thermal state, memory pressure, and NCNN build flags.

| Change | Estimated saving per camera frame | Why |
|---|---:|---|
| NCNN default backend instead of PyTorch/Ultralytics process path | 20-45 ms amortized at 0.5-0.75s interval | NCNN avoids PyTorch runtime overhead and avoids duplicated isolated interpreter RSS; per-inference savings are much larger but amortized over camera frames. |
| Perception downscale to `process_width=256` | 15-35 ms | Smaller BGR->RGB conversion and smaller FaceLandmarker input. |
| Dynamic face ROI for YOLO | 5-20 ms amortized | Smaller object-detector crop reduces resize/preprocess and inference work when the driver occupies a subset of the frame. |
| Adaptive inference interval | 0-25 ms amortized under throttling | Saves work only when YOLO latency exceeds budget; no saving when under budget. |
| OpenCV/OMP/NCNN thread caps | 2-10 ms effective latency reduction | Reduces oversubscription and contention, especially under capture + MediaPipe + recorder load. |
| Removing pandas/Ultralytics from Pi dependency set | 0 ms steady-state per frame | Reduces install size, import risk, memory footprint, and cold-start pressure rather than per-frame compute. |

Expected combined effect on a warm Pi 4: roughly 40-90 ms less average per-frame pressure in the live pipeline, with larger benefits during object-inference frames and thermal throttling.

## 6. Rollback Notes

NCNN backend:

- Set `behavior_detector.model_path` back to `models/driver_behavior/luthfi_yolo11n/best_yolo11n.pt`.
- Set `behavior_detector.backend` to `"pytorch"` or legacy `"ultralytics"`.
- Set `behavior_detector.execution_mode` to `"process"`.

Perception downscale:

- Set `perception.process_width` to `0`.

Dynamic ROI:

- Set `behavior_detector.roi_mode` to `"static"` and keep `driver_roi: [0.0, 0.0, 1.0, 1.0]`.
- Or set `behavior_detector.roi_mode` to `"full"` if static ROI should be bypassed.

Adaptive interval:

- Set `behavior_detector.adaptive_interval` to `false`.
- The fixed interval returns to `behavior_detector.inference_interval_sec`.

Thread caps:

- Increase `cv_num_threads`, `omp_num_threads`, or `ncnn_num_threads` in config.
- If implementation supports zero as "library default", set the relevant cap to `0`; otherwise use the desired positive thread count.

Pandas removal:

- Laptop workflows still use `requirements-laptop.txt`, which keeps pandas.
- If a removed pandas path proves numerically non-parity, restore pandas only in the affected offline tool while keeping it out of live imports, then revisit the NumPy parity code.

Pi 4 profile:

- Use existing `configs/runtime.laptop_alarm_demo.json` or `configs/runtime.replay_silent.json` for previous demo/replay behavior.

## 7. No Behavioral Change Checklist

| Constraint | Verification |
|---|---|
| Do not change `shared/feature_contract.py`. | `git diff -- shared/feature_contract.py` must be empty. |
| Do not change `shared/contracts.py` semantics. | Prefer side-channel face bbox; if file changes at all, diff must show no version/semantic changes and tests that construct contracts must pass. |
| Do not change `RUNTIME_FEATURE_VERSION`. | `git diff -- shared/feature_contract.py` and feature tests. |
| Do not change `CONTRACT_VERSION`. | `git diff -- shared/contracts.py`; no version bump. |
| Do not change EAR/MAR formulas. | `tests/test_eye_calibration.py` plus focused diff review of `_ear`, `_event_ear`, and `_mar`. |
| Do not change head-pose/gaze formulas. | Focused diff review of `_head_pose`, `_iris_offset`, and `_gaze`; downscale code must pass original dimensions. |
| Do not change fusion thresholds. | `git diff -- runtime/fusion/state_machine.py`; `tests/test_events_fusion.py`. |
| Do not change event thresholds. | `git diff -- runtime/events/engine.py`; `tests/test_events_fusion.py`. |
| Do not change LightGBM bundle contract. | `LightGBMRuntimePredictor.verify_golden_samples()` through app verify-bundle command. |
| Golden samples still pass. | `PYTHONPATH=. python3 -m dms_final_system.runtime.app --verify-bundle dms_final_system/models/drowsiness/20260731T140605Z_full_runtimefix1`. |
| `FeatureDriftMonitor` must not be loosened. | `git diff -- runtime/model/drift.py`; `tests/test_model_drift.py`; replay telemetry must not produce new `MODEL_INPUT_OOD` caused by downscale. |
| Object violations do not alter drowsiness state semantics. | Existing `test_external_object_violation_does_not_turn_normal_driver_drowsy`. |
| Behavior detector box remapping stays full-frame. | New ROI offset tests in `tests/test_behavior_detector.py`. |
| Pixel safety gates keep original-frame meaning. | New/extended downscale pixel geometry tests in `tests/test_eye_calibration.py`. |
| Alarm and recorder remain enabled as configured. | `tests/test_alarm_controller.py`, `tests/test_recorder.py`, and Pi profile review. |
| All test modules still pass. | `PYTHONPATH=. pytest dms_final_system/tests`. |
| Package compiles. | `PYTHONPATH=. python3 -m compileall -q dms_final_system`. |

## Stop Point

Stop here and wait for explicit approval before implementing. The next production-code step should not begin until the user replies `approved`.
