from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the bundled YOLO11n behavior model to NCNN")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Move the generated export to this explicit directory (must not exist)",
    )
    args = parser.parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install ultralytics before exporting: pip install 'ultralytics>=8.3,<9'") from exc
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    model = YOLO(str(args.model.resolve()), task="detect")
    names = model.names
    source_classes = (
        [str(names[index]) for index in sorted(names)]
        if isinstance(names, dict)
        else [str(value) for value in names]
    )
    output = model.export(format="ncnn", imgsz=args.imgsz, half=args.half, device="cpu")
    output_dir = Path(output)
    if output_dir.is_dir():
        if args.output_dir is not None:
            requested = args.output_dir.resolve()
            if requested.exists():
                raise FileExistsError(f"Refusing to overwrite NCNN export: {requested}")
            requested.parent.mkdir(parents=True, exist_ok=True)
            output_dir = Path(shutil.move(str(output_dir), str(requested)))
        manifest = {
            "export_format": "ncnn",
            "source_model": args.model.name,
            "source_sha256": _sha256(args.model),
            "source_classes": source_classes,
            "imgsz": int(args.imgsz),
            "half": bool(args.half),
            "created_by": "dms_final_system.runtime.objects.export_ncnn",
            "exporter_versions": {
                name: _version(name) for name in ("ultralytics", "torch", "ncnn", "pnnx")
            },
            "artifacts_sha256": {
                path.name: _sha256(path)
                for path in sorted(output_dir.iterdir())
                if path.is_file() and path.suffix.lower() in {".param", ".bin"}
            },
        }
        (output_dir / "ncnn_export_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(f"NCNN export created at: {output_dir}")


if __name__ == "__main__":
    main()
