from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def derive_reference_bundle(source: Path, train_csv: Path, output: Path) -> Path:
    """Clone a verified model bundle and add training-range drift metadata."""
    from dms_final_system.runtime.model.bundle import verify_bundle

    source, train_csv, output = Path(source).resolve(), Path(train_csv).resolve(), Path(output).resolve()
    verify_bundle(source)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output bundle is not empty: {output}")
    manifest = json.loads((source / "training_manifest.json").read_text(encoding="utf-8"))
    expected_hash = manifest.get("data_hashes", {}).get("train_windows")
    actual_hash = _sha256(train_csv)
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(
            f"Training CSV hash mismatch: expected={expected_hash}, actual={actual_hash}"
        )

    try:
        import lightgbm as lgb
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy and lightgbm are required to derive the reference bundle") from exc

    schema = json.loads((source / "feature_schema.json").read_text(encoding="utf-8"))
    features = list(schema["ordered_features"])
    columns: dict[str, list[float]] = {name: [] for name in features}
    with train_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = sorted(set(features) - set(reader.fieldnames or []))
        if missing_columns:
            raise ValueError(f"Training CSV is missing required features: {missing_columns[:5]}")
        for row in reader:
            for name in features:
                try:
                    columns[name].append(float(row.get(name, "")))
                except (TypeError, ValueError):
                    columns[name].append(float("nan"))
    booster = lgb.Booster(model_file=str(source / "model.txt"))
    gains = dict(zip(booster.feature_name(), booster.feature_importance(importance_type="gain")))
    total_gain = max(float(sum(gains.values())), 1e-12)
    ranked = sorted(features, key=lambda name: gains.get(name, 0.0), reverse=True)
    ranks = {name: index + 1 for index, name in enumerate(ranked)}
    reference = {}
    for name in features:
        numeric = np.asarray(columns[name], dtype=float)
        valid = numeric[~np.isnan(numeric)]
        quantiles = (
            np.quantile(valid, [0.01, 0.05, 0.50, 0.95, 0.99], method="linear")
            if len(valid)
            else np.asarray([float("nan")] * 5)
        )
        reference[name] = {
            "q01": float(quantiles[0]),
            "q05": float(quantiles[1]),
            "median": float(quantiles[2]),
            "q95": float(quantiles[3]),
            "q99": float(quantiles[4]),
            "gain": float(gains.get(name, 0.0)),
            "gain_fraction": float(gains.get(name, 0.0) / total_gain),
            "gain_rank": int(ranks[name]),
            "missing_rate": float(np.mean(np.isnan(numeric))) if len(numeric) else 0.0,
        }

    output.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and path.name != "checksums.sha256":
            shutil.copy2(path, output / path.name)
    manifest["derived_from_run_id"] = manifest["run_id"]
    manifest["run_id"] = output.name
    manifest["runtime_patch"] = "eye-events-fusion-v2"
    _write_json(output / "training_manifest.json", manifest)
    _write_json(
        output / "feature_reference.json",
        {
            "schema_version": "1.0.0",
            "source_train_sha256": actual_hash,
            "feature_count": len(features),
            "range_contract": "q01_q99_of_fitting_train",
            "features": reference,
        },
    )
    card = (output / "MODEL_CARD.md").read_text(encoding="utf-8")
    card += "\n- Runtime derivative: adds feature-range monitoring; model weights and calibration are unchanged.\n"
    (output / "MODEL_CARD.md").write_text(card, encoding="utf-8")
    protected = sorted(path for path in output.iterdir() if path.is_file())
    (output / "checksums.sha256").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in protected),
        encoding="utf-8",
    )
    verify_bundle(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Add drift reference data to a model bundle")
    parser.add_argument("source", type=Path)
    parser.add_argument("train_csv", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(derive_reference_bundle(args.source, args.train_csv, args.output))


if __name__ == "__main__":
    main()
