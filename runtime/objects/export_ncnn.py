from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the bundled YOLO11n behavior model to NCNN")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true")
    args = parser.parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install ultralytics before exporting: pip install 'ultralytics>=8.3,<9'") from exc
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    model = YOLO(str(args.model.resolve()), task="detect")
    output = model.export(format="ncnn", imgsz=args.imgsz, half=args.half, device="cpu")
    output_dir = Path(output)
    if output_dir.is_dir():
        manifest = {
            "export_format": "ncnn",
            "source_model": str(args.model.resolve()),
            "source_sha256": _sha256(args.model),
            "imgsz": int(args.imgsz),
            "half": bool(args.half),
            "created_by": "dms_final_system.runtime.objects.export_ncnn",
        }
        (output_dir / "ncnn_export_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(f"NCNN export created at: {output}")


if __name__ == "__main__":
    main()
