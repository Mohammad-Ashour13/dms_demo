from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dms_final_system.shared.contracts import ModelPrediction, TemporalFeatureSnapshot

from .bundle import verify_bundle


class LightGBMRuntimePredictor:
    def __init__(self, bundle_dir: Path, verify: bool = True):
        bundle_dir = Path(bundle_dir).resolve()
        if verify:
            verify_bundle(bundle_dir)
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise RuntimeError("lightgbm is required on Raspberry; install requirements-raspberry.txt") from exc
        self.bundle_dir = bundle_dir
        self.model = lgb.Booster(model_file=str(bundle_dir / "model.txt"))
        self.schema = json.loads((bundle_dir / "feature_schema.json").read_text(encoding="utf-8"))
        self.features = list(self.schema["ordered_features"])
        if self.model.num_feature() != len(self.features):
            raise RuntimeError(
                f"LightGBM feature count {self.model.num_feature()} != schema count {len(self.features)}"
            )
        self.calibration = json.loads((bundle_dir / "calibration.json").read_text(encoding="utf-8"))
        self.operating_point = json.loads((bundle_dir / "operating_point.json").read_text(encoding="utf-8"))
        self.manifest = json.loads((bundle_dir / "training_manifest.json").read_text(encoding="utf-8"))
        self.model_version = self.manifest["run_id"]

    def calibrate(self, raw: float | np.ndarray):
        p = np.clip(np.asarray(raw, dtype=float), 1e-7, 1 - 1e-7)
        logit = np.log(p / (1 - p))
        z = self.calibration["coefficient"] * logit + self.calibration["intercept"]
        result = 1 / (1 + np.exp(-np.clip(z, -50, 50)))
        return float(result) if result.ndim == 0 else result

    def predict(self, snapshot: TemporalFeatureSnapshot) -> ModelPrediction:
        if not snapshot.valid:
            raise ValueError(f"Cannot predict invalid window: {snapshot.reason}")
        if snapshot.feature_names != self.features:
            raise ValueError("Runtime feature order does not match bundle feature_schema.json")
        values = np.asarray(snapshot.ordered_features, dtype=float).reshape(1, -1)
        raw = float(self.model.predict(values)[0])
        calibrated = self.calibrate(raw)
        threshold = float(self.operating_point["threshold"])
        return ModelPrediction(
            snapshot.window_id, snapshot.end_monotonic_sec, raw, calibrated,
            threshold, calibrated >= threshold, self.model_version,
        )

    def verify_golden_samples(self, tolerance: float | None = None) -> dict:
        payload = json.loads((self.bundle_dir / "golden_samples.json").read_text(encoding="utf-8"))
        tolerance = float(payload.get("tolerance", 1e-6) if tolerance is None else tolerance)
        max_raw_error = max_calibrated_error = 0.0
        for sample in payload["samples"]:
            values = np.asarray([[sample["features"][name] for name in self.features]], dtype=float)
            raw = float(self.model.predict(values)[0])
            calibrated = self.calibrate(raw)
            max_raw_error = max(max_raw_error, abs(raw - sample["expected_raw_probability"]))
            max_calibrated_error = max(max_calibrated_error, abs(calibrated - sample["expected_calibrated_probability"]))
        if max_raw_error > tolerance or max_calibrated_error > tolerance:
            raise RuntimeError(
                f"Golden sample parity failed: raw={max_raw_error:.3g}, calibrated={max_calibrated_error:.3g}, tolerance={tolerance}"
            )
        return {"samples": len(payload["samples"]), "max_raw_error": max_raw_error, "max_calibrated_error": max_calibrated_error, "tolerance": tolerance}
