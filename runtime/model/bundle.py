from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dms_final_system.shared.feature_contract import RUNTIME_FEATURE_VERSION, audit_features


REQUIRED_FILES = {
    "model.txt", "calibration.json", "operating_point.json", "feature_schema.json",
    "training_manifest.json", "golden_samples.json", "checksums.sha256", "MODEL_CARD.md",
}
OPTIONAL_PROTECTED_FILES = {"feature_reference.json"}


class BundleVerificationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_active_model(active_model_path: Path) -> dict:
    active_model_path = Path(active_model_path).resolve()
    payload = json.loads(active_model_path.read_text(encoding="utf-8"))
    if "bundle_path" not in payload:
        raise BundleVerificationError("active_model.json is missing bundle_path")
    target = Path(payload["bundle_path"])
    if not target.is_absolute():
        target = (active_model_path.parent / target).resolve()
    mode = str(payload.get("deployment_mode", "SHADOW")).upper()
    if mode not in {"SHADOW", "ACTIVE"}:
        raise BundleVerificationError(f"Invalid active model deployment_mode: {mode}")
    return {**payload, "bundle_dir": target, "deployment_mode": mode}


def resolve_active_bundle(active_model_path: Path) -> Path:
    return Path(read_active_model(active_model_path)["bundle_dir"])


def assert_deployment_allowed(bundle_status: str, deployment_mode: str) -> None:
    mode = str(deployment_mode).upper()
    if mode not in {"SHADOW", "ACTIVE"}:
        raise BundleVerificationError(f"Invalid deployment mode: {mode}")
    if mode == "ACTIVE" and str(bundle_status).upper() == "EXPERIMENTAL":
        raise BundleVerificationError(
            "EXPERIMENTAL model bundles are restricted to SHADOW deployment"
        )


def verify_bundle(bundle_dir: Path) -> dict:
    bundle_dir = Path(bundle_dir).resolve()
    missing = sorted(name for name in REQUIRED_FILES if not (bundle_dir / name).is_file())
    if missing:
        raise BundleVerificationError(f"Bundle files missing: {missing}")
    expected = {}
    for line in (bundle_dir / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name.strip()] = digest
    expected_names = (REQUIRED_FILES - {"checksums.sha256"}) | {
        name for name in OPTIONAL_PROTECTED_FILES if (bundle_dir / name).is_file()
    }
    if set(expected) != expected_names:
        raise BundleVerificationError(
            f"Checksum manifest coverage mismatch: expected={sorted(expected_names)}, got={sorted(expected)}"
        )
    mismatches = [name for name, digest in expected.items() if not (bundle_dir / name).is_file() or _sha256(bundle_dir / name) != digest]
    if mismatches:
        raise BundleVerificationError(f"Checksum mismatch: {mismatches}")
    schema = json.loads((bundle_dir / "feature_schema.json").read_text(encoding="utf-8"))
    if schema.get("runtime_feature_version") != RUNTIME_FEATURE_VERSION:
        raise BundleVerificationError(
            f"Runtime feature version mismatch: bundle={schema.get('runtime_feature_version')} runtime={RUNTIME_FEATURE_VERSION}"
        )
    features = schema.get("ordered_features", [])
    audit = {key: values for key, values in audit_features(features).items() if values}
    if not features or audit:
        raise BundleVerificationError(f"Invalid feature schema: {audit or 'empty order'}")
    manifest = json.loads((bundle_dir / "training_manifest.json").read_text(encoding="utf-8"))
    return {
        "bundle_dir": str(bundle_dir),
        "feature_count": len(features),
        "model_version": manifest["run_id"],
        "status": manifest["bundle_status"],
        "feature_reference_available": (bundle_dir / "feature_reference.json").is_file(),
    }
