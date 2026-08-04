from __future__ import annotations

from pathlib import Path

import pandas as pd

from .feature_contract import (
    MIN_COVERAGE,
    PIPELINE_VERSION,
    RUNTIME_FEATURE_VERSION,
    STRIDE_SEC,
    TARGET_SAMPLES,
    WINDOW_SEC,
    base_signal_for_feature,
)

from .calibration import PlattCalibration
from .io import sha256_file, write_json


def export_bundle(
    bundle_dir: Path,
    model,
    feature_names: list[str],
    calibration: PlattCalibration,
    threshold: float,
    train_frame: pd.DataFrame,
    manifest: dict,
) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    model_path = bundle_dir / "model.txt"
    model.booster_.save_model(str(model_path))
    write_json(bundle_dir / "calibration.json", calibration.to_dict())
    write_json(
        bundle_dir / "operating_point.json",
        {
            "threshold": float(threshold),
            "selection_source": "calibrated_oof_train_predictions",
            "recall_target": manifest["targets"]["recall"],
            "precision_target": manifest["targets"]["precision"],
            "constraints_met": manifest["oof_constraints_met"],
        },
    )
    write_json(
        bundle_dir / "feature_schema.json",
        {
            "schema_version": "1.0.0",
            "v3_pipeline_version": PIPELINE_VERSION,
            "runtime_feature_version": RUNTIME_FEATURE_VERSION,
            "ordered_features": feature_names,
            "base_signals": sorted({base_signal_for_feature(n) for n in feature_names}),
            "window_sec": WINDOW_SEC,
            "stride_sec": STRIDE_SEC,
            "target_samples": TARGET_SAMPLES,
            "min_coverage": MIN_COVERAGE,
            "motion_units": {
                "face_velocity_x": "face_widths_per_second",
                "face_velocity_y": "face_heights_per_second",
                "head_motion": "degrees_per_second",
            },
            "formula_contract": {
                "resampling": "numpy.linspace(start,end,30,endpoint=False); linear interpolation; nearest categorical",
                "statistics": "V3 population std/variance, quantiles 0.25/0.75, median MAD, bias-corrected scipy-compatible skew/kurtosis, mean-square energy",
                "temporal": "time-aware linear slope, first-last, absolute changes, rolling summaries, numpy-gradient velocity/acceleration",
                "binary_events": "contiguous runs at >=0.5; durations and rates use seconds",
                "smoothing": "0.1-second centered rolling mean on V3-designated continuous base signals before resampling",
            },
            "missing_event_time": "NaN_native_lightgbm",
            "standard_scaler": None,
        },
    )
    write_json(bundle_dir / "training_manifest.json", manifest)

    booster = model.booster_
    booster_names = list(booster.feature_name())
    gains = dict(zip(booster_names, booster.feature_importance(importance_type="gain")))
    total_gain = max(float(sum(gains.values())), 1e-12)
    ranked = sorted(feature_names, key=lambda name: gains.get(name, 0.0), reverse=True)
    rank = {name: index + 1 for index, name in enumerate(ranked)}
    reference_features = {}
    for name in feature_names:
        numeric = pd.to_numeric(train_frame[name], errors="coerce")
        values = numeric.dropna()
        quantiles = values.quantile([0.01, 0.05, 0.50, 0.95, 0.99])
        reference_features[name] = {
            "q01": float(quantiles.loc[0.01]),
            "q05": float(quantiles.loc[0.05]),
            "median": float(quantiles.loc[0.50]),
            "q95": float(quantiles.loc[0.95]),
            "q99": float(quantiles.loc[0.99]),
            "gain": float(gains.get(name, 0.0)),
            "gain_fraction": float(gains.get(name, 0.0) / total_gain),
            "gain_rank": int(rank[name]),
            "missing_rate": float(numeric.isna().mean()),
        }
    write_json(
        bundle_dir / "feature_reference.json",
        {
            "schema_version": "1.0.0",
            "source_train_sha256": manifest.get("data_hashes", {}).get("train_windows"),
            "feature_count": len(feature_names),
            "range_contract": "q01_q99_of_fitting_train",
            "features": reference_features,
        },
    )

    golden = train_frame.sample(min(12, len(train_frame)), random_state=manifest["selected_seed"])
    raw = model.predict_proba(golden[feature_names])[:, 1]
    calibrated = calibration.apply(raw)
    samples = []
    for (_, row), raw_p, cal_p in zip(golden.iterrows(), raw, calibrated):
        samples.append(
            {
                "features": {name: float(row[name]) for name in feature_names},
                "expected_raw_probability": float(raw_p),
                "expected_calibrated_probability": float(cal_p),
            }
        )
    write_json(bundle_dir / "golden_samples.json", {"tolerance": 1e-6, "samples": samples})

    card = "# DMS Drowsiness Model Card\n\n"
    card += f"- Status: **{manifest['bundle_status']}**\n"
    card += f"- Model: LightGBM binary classifier\n"
    card += f"- Features: {len(feature_names)} runtime-computable V3 window features\n"
    card += f"- Window: {WINDOW_SEC}s, {TARGET_SAMPLES} resampled points, update {STRIDE_SEC}s\n"
    card += f"- Calibration: Platt scaling fitted only on OOF training predictions\n"
    card += f"- Threshold: {threshold:.8f}, selected only from OOF training predictions\n"
    card += "- Intended use: one evidence source inside the documented Fusion FSM; never a standalone safety decision.\n"
    card += "- Limitations: video-level weak labels, domain shift, camera/landmark sensitivity. Review per-dataset and LODO metrics.\n"
    (bundle_dir / "MODEL_CARD.md").write_text(card, encoding="utf-8")

    protected = sorted(p for p in bundle_dir.iterdir() if p.name != "checksums.sha256")
    lines = [f"{sha256_file(path)}  {path.name}" for path in protected]
    (bundle_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
