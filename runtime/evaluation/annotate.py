from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

from .session import ANNOTATION_COLUMNS


LABEL_KEYS = {
    ord("a"): "AWAKE",
    ord("d"): "DROWSY_SIMULATED",
    ord("y"): "YAWN",
    ord("c"): "PROLONGED_EYE_CLOSURE",
    ord("b"): "BLINK",
    ord("x"): "DISTRACTION",
    ord("p"): "PHONE_USE",
    ord("e"): "EATING",
    ord("s"): "SMOKING",
    ord("u"): "UNSCORABLE",
}


def _load_existing(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle) if row.get("label")]


def _save(path: Path, rows: list[dict]) -> None:
    ordered = sorted(rows, key=lambda row: (float(row["start_sec"]), row["label"]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader()
        writer.writerows(ordered)


def annotate(session_dir: Path) -> Path:
    session_dir = Path(session_dir)
    video_path = session_dir / "session.mp4"
    manifest_path = session_dir / "session_manifest.json"
    if not video_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("session.mp4 and session_manifest.json are required")
    session_id = json.loads(manifest_path.read_text(encoding="utf-8"))["session_id"]
    annotations_path = session_dir / "annotations.csv"
    rows = _load_existing(annotations_path)
    active: dict[str, float] = {}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 15.0
    paused = True
    print("Keys: SPACE play/pause | a awake | d drowsy | b blink | y yawn | c closure")
    print("      x distraction | p phone use | e eating | s smoking | u unscorable")
    print("Press a label key at its start and the same key at its end. j/l seek -/+1s, q save and quit.")
    try:
        while True:
            if not paused:
                ok, frame = capture.read()
                if not ok:
                    break
            else:
                position = max(0, int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
                capture.set(cv2.CAP_PROP_POS_FRAMES, position)
                ok, frame = capture.read()
                if not ok:
                    break
            current_frame = max(0.0, capture.get(cv2.CAP_PROP_POS_FRAMES) - 1.0)
            now_sec = current_frame / fps
            overlay = frame.copy()
            status = ", ".join(sorted(active)) or "none"
            cv2.putText(
                overlay, f"t={now_sec:.2f}s active={status}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
            )
            cv2.imshow("DMS annotation (raw video is unchanged)", overlay)
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
            elif key == ord("j"):
                capture.set(cv2.CAP_PROP_POS_FRAMES, max(0.0, current_frame - fps))
            elif key == ord("l"):
                capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame + fps)
            elif key in LABEL_KEYS:
                label = LABEL_KEYS[key]
                if label in active:
                    start = active.pop(label)
                    if now_sec > start:
                        rows.append(
                            {
                                "session_id": session_id,
                                "start_sec": f"{start:.3f}",
                                "end_sec": f"{now_sec:.3f}",
                                "label": label,
                                "confidence": "1.0",
                                "notes": "",
                            }
                        )
                else:
                    active[label] = now_sec
    finally:
        end_sec = max(0.0, capture.get(cv2.CAP_PROP_POS_FRAMES) - 1.0) / fps
        capture.release()
        cv2.destroyAllWindows()
        for label, start in active.items():
            if end_sec > start:
                rows.append(
                    {
                        "session_id": session_id,
                        "start_sec": f"{start:.3f}", "end_sec": f"{end_sec:.3f}",
                        "label": label, "confidence": "1.0", "notes": "",
                    }
                )
        _save(annotations_path, rows)
    print(f"Saved {len(rows)} annotations to {annotations_path}")
    return annotations_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Review and annotate a DMS evaluation session")
    parser.add_argument("session_dir", type=Path)
    args = parser.parse_args()
    annotate(args.session_dir)


if __name__ == "__main__":
    main()
