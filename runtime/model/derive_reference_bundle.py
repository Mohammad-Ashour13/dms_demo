from __future__ import annotations

import argparse
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
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas and lightgbm are required to derive the reference bundle") from exc

    schema = json.loads((source / "feature_schema.json").read_text(encoding="utf-8"))
    features = list(schema["ordered_features"])
    frame = pd.read_csv(train_csv, usecols=features)
    booster = lgb.Booster(model_file=str(source / "model.txt"))
    gains = dict(zip(booster.feature_name(), booster.feature_importance(importance_type="gain")))
    total_gain = max(float(sum(gains.values())), 1e-12)
    ranked = sorted(features, key=lambda name: gains.get(name, 0.0), reverse=True)
    ranks = {name: index + 1 for index, name in enumerate(ranked)}
    reference = {}
    for name in features:
        numeric = pd.to_numeric(frame[name], errors="coerce")
        quantiles = numeric.dropna().quantile([0.01, 0.05, 0.50, 0.95, 0.99])
        reference[name] = {
            "q01": float(quantiles.loc[0.01]),
            "q05": float(quantiles.loc[0.05]),
            "median": float(quantiles.loc[0.50]),
            "q95": float(quantiles.loc[0.95]),
            "q99": float(quantiles.loc[0.99]),
            "gain": float(gains.get(name, 0.0)),
            "gain_fraction": float(gains.get(name, 0.0) / total_gain),
            "gain_rank": int(ranks[name]),
            "missing_rate": float(numeric.isna().mean()),
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
