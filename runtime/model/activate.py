from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .bundle import assert_deployment_allowed, verify_bundle
from .predictor import LightGBMRuntimePredictor


def activate(bundle_dir: Path, active_model_path: Path, deployment_mode: str = "SHADOW") -> dict:
    bundle_dir, active_model_path = Path(bundle_dir).resolve(), Path(active_model_path).resolve()
    result = verify_bundle(bundle_dir)
    deployment_mode = str(deployment_mode).upper()
    assert_deployment_allowed(result["status"], deployment_mode)
    parity = LightGBMRuntimePredictor(bundle_dir, verify=False).verify_golden_samples()
    active_model_path.parent.mkdir(parents=True, exist_ok=True)
    if active_model_path.is_file():
        shutil.copy2(active_model_path, active_model_path.with_name("active_model.previous.json"))
    try:
        relative = bundle_dir.relative_to(active_model_path.parent)
        bundle_value = str(relative)
    except ValueError:
        bundle_value = str(bundle_dir)
    temporary = active_model_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "bundle_path": bundle_value,
                "model_version": result["model_version"],
                "bundle_status": result["status"],
                "deployment_mode": deployment_mode,
                "golden_parity": parity,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(active_model_path)
    return {
        **result, **parity, "deployment_mode": deployment_mode,
        "active_model": str(active_model_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and atomically activate a DMS model bundle")
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--active-model", type=Path, required=True)
    parser.add_argument("--deployment-mode", choices=("SHADOW", "ACTIVE"), default="SHADOW")
    args = parser.parse_args()
    print(activate(args.bundle_dir, args.active_model, args.deployment_mode))


if __name__ == "__main__":
    main()
