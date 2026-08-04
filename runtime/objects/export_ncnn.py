from __future__ import annotations

import argparse
from pathlib import Path


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
    print(f"NCNN export created at: {output}")


if __name__ == "__main__":
    main()
