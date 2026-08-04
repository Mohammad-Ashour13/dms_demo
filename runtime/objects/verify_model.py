from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the downloaded driver-behavior YOLO model")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--deep", action="store_true", help="Also load the model through Ultralytics")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    manifest = json.loads((bundle / "model_manifest.json").read_text(encoding="utf-8"))
    artifact = bundle / manifest["artifact"]
    actual = sha256(artifact)
    if actual != manifest["sha256"]:
        raise RuntimeError(f"Checksum mismatch: expected={manifest['sha256']} actual={actual}")
    result = {
        "model_version": manifest["model_version"],
        "artifact": str(artifact),
        "size_bytes": artifact.stat().st_size,
        "sha256": actual,
        "classes_from_manifest": manifest["source_classes"],
    }
    # A PyTorch checkpoint is a ZIP container. Inspecting the pickle bytes for
    # declared class labels verifies this artifact without executing pickle code.
    if artifact.suffix.lower() == ".pt" and zipfile.is_zipfile(artifact):
        with zipfile.ZipFile(artifact) as archive:
            pickle_names = [name for name in archive.namelist() if name.endswith("/data.pkl")]
            if not pickle_names:
                raise RuntimeError("PyTorch checkpoint has no data.pkl payload")
            payload = archive.read(pickle_names[0])
        checkpoint_classes = [
            name for name in manifest["source_classes"]
            if str(name).encode("utf-8") in payload
        ]
        missing = sorted(set(manifest["source_classes"]) - set(checkpoint_classes))
        if missing:
            raise RuntimeError(f"Checkpoint class strings are missing: {missing}")
        result["classes_verified_in_checkpoint"] = checkpoint_classes
    if args.deep:
        from .yolo_behavior import UltralyticsYoloBackend

        backend = UltralyticsYoloBackend(artifact, "cpu")
        result["classes_from_model"] = list(backend.names.values())
        missing = sorted(set(manifest["source_classes"]) - set(result["classes_from_model"]))
        if missing:
            raise RuntimeError(f"Model class mismatch: missing={missing}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
