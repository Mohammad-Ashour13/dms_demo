from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


CLASSES = ("phone", "cigarette", "drink_or_food")


def _evaluate(model_path: Path, image_size: int, frames: list[dict], thresholds: dict) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Run NCNN comparison on the export workstation with Ultralytics") from exc
    model = YOLO(str(model_path), task="detect")
    totals = Counter()
    true_positives = Counter()
    for item in frames:
        image_path = Path(item["image"])
        if not image_path.is_absolute():
            image_path = (Path(item["_manifest_dir"]) / image_path).resolve()
        truth = Counter(str(label) for label in item.get("labels", []) if label in CLASSES)
        totals.update(truth)
        result = model.predict(
            str(image_path),
            imgsz=int(image_size),
            conf=min(float(value) for value in thresholds.values()),
            iou=0.45,
            device="cpu",
            verbose=False,
        )[0]
        detected = Counter()
        if result.boxes is not None:
            for class_id, confidence in zip(
                result.boxes.cls.detach().cpu().tolist(),
                result.boxes.conf.detach().cpu().tolist(),
            ):
                label = str(model.names[int(class_id)])
                if label in thresholds and float(confidence) >= float(thresholds[label]):
                    detected[label] += 1
        for label in CLASSES:
            true_positives[label] += min(truth[label], detected[label])
    return {
        label: {
            "annotated_instances": totals[label],
            "true_positives": true_positives[label],
            "recall": true_positives[label] / totals[label] if totals[label] else None,
        }
        for label in CLASSES
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare 640 and 480 NCNN behavior recall on annotated Safee replay frames"
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--guardrail", type=float, default=0.02)
    args = parser.parse_args()
    manifest = json.loads(args.annotations.read_text(encoding="utf-8"))
    frames = list(manifest.get("frames") or [])
    if not frames:
        raise ValueError("Annotation manifest must contain a non-empty frames list")
    for item in frames:
        item["_manifest_dir"] = str(args.annotations.resolve().parent)
    thresholds = manifest.get("class_thresholds") or {
        "phone": 0.40,
        "cigarette": 0.30,
        "drink_or_food": 0.35,
    }
    if set(thresholds) != set(CLASSES):
        raise ValueError(f"class_thresholds must define exactly {list(CLASSES)}")
    reference = _evaluate(args.reference.resolve(), 640, frames, thresholds)
    candidate = _evaluate(args.candidate.resolve(), 480, frames, thresholds)
    guardrail = float(args.guardrail)
    per_class = {}
    for label in CLASSES:
        reference_recall = reference[label]["recall"]
        candidate_recall = candidate[label]["recall"]
        if reference_recall is None or candidate_recall is None:
            accepted = False
            loss = None
        else:
            loss = reference_recall - candidate_recall
            accepted = loss <= guardrail + 1e-12
        per_class[label] = {
            "reference": reference[label],
            "candidate": candidate[label],
            "recall_loss": loss,
            "accepted": accepted,
        }
    result = {
        "schema_version": "ncnn-recall-comparison-v1",
        "reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
        "annotations": str(args.annotations.resolve()),
        "guardrail": guardrail,
        "per_class": per_class,
        "candidate_accepted": all(item["accepted"] for item in per_class.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["candidate_accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
