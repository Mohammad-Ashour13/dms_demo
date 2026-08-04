from __future__ import annotations

import csv
import hashlib
import json
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path

import cv2

from dms_final_system.shared.contracts import FramePacket


ANNOTATION_COLUMNS = (
    "session_id", "start_sec", "end_sec", "label", "confidence", "notes"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EvaluationSessionRecorder:
    """Asynchronous raw full-session recorder with an exact frame time map."""

    def __init__(
        self,
        output_root: Path,
        session_id: str,
        telemetry,
        *,
        fps: float = 15.0,
        queue_size: int = 512,
        record_video: bool = True,
        metadata: dict | None = None,
    ):
        self.session_id = session_id
        self.session_dir = Path(output_root) / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        if (self.session_dir / "session_manifest.json").exists():
            raise FileExistsError(f"Evaluation session already exists: {self.session_dir}")
        self.telemetry = telemetry
        self.fps = float(fps)
        self.record_video = bool(record_video)
        self.queue: queue.Queue[FramePacket | None] = queue.Queue(maxsize=queue_size)
        self.video_path = self.session_dir / "session.mp4"
        self.timestamps_path = self.session_dir / "frame_timestamps.csv"
        self.annotations_path = self.session_dir / "annotations.csv"
        self.manifest_path = self.session_dir / "session_manifest.json"
        self.dropped_frames = 0
        self.recording_errors = 0
        self.written_frames = 0
        self.first_monotonic_sec: float | None = None
        self.last_monotonic_sec: float | None = None
        self.first_utc: str | None = None
        self.last_utc: str | None = None
        self.encoder = "disabled"
        self.metadata = dict(metadata or {})
        with self.annotations_path.open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=ANNOTATION_COLUMNS).writeheader()
        self._write_manifest("RECORDING")
        self.worker = threading.Thread(
            target=self._run, name="evaluation-session-recorder", daemon=True
        )
        self.worker.start()

    def update_metadata(self, **values) -> None:
        self.metadata.update(values)
        self._write_manifest("RECORDING")

    def push(self, packet: FramePacket) -> None:
        try:
            self.queue.put_nowait(packet)
        except queue.Full:
            self.dropped_frames += 1

    def _run(self) -> None:
        writer = None
        with self.timestamps_path.open("w", encoding="utf-8", newline="") as handle:
            fields = (
                "video_frame_index", "frame_id", "video_sec", "session_sec",
                "monotonic_sec", "utc_timestamp",
            )
            timestamps = csv.DictWriter(handle, fieldnames=fields)
            timestamps.writeheader()
            while True:
                packet = self.queue.get()
                if packet is None:
                    self.queue.task_done()
                    break
                try:
                    height, width = packet.frame.shape[:2]
                    if self.first_monotonic_sec is None:
                        self.first_monotonic_sec = float(packet.monotonic_sec)
                        self.first_utc = packet.utc_timestamp
                    if self.record_video and writer is None:
                        writer = cv2.VideoWriter(
                            str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                            self.fps, (width, height),
                        )
                        if not writer.isOpened():
                            raise RuntimeError("Could not start evaluation MP4 recorder")
                        self.encoder = "opencv_mp4v"
                    if writer is not None:
                        writer.write(packet.frame)
                    session_sec = float(packet.monotonic_sec) - self.first_monotonic_sec
                    video_sec = self.written_frames / self.fps
                    timestamps.writerow(
                        {
                            "video_frame_index": self.written_frames,
                            "frame_id": packet.frame_id,
                            "video_sec": f"{video_sec:.9f}",
                            "session_sec": f"{session_sec:.9f}",
                            "monotonic_sec": f"{float(packet.monotonic_sec):.9f}",
                            "utc_timestamp": packet.utc_timestamp,
                        }
                    )
                    self.written_frames += 1
                    self.last_monotonic_sec = float(packet.monotonic_sec)
                    self.last_utc = packet.utc_timestamp
                except Exception as exc:
                    self.recording_errors += 1
                    if writer is not None and not writer.isOpened():
                        writer.release()
                        writer = None
                        self.encoder = "failed"
                    self.telemetry.emit(
                        "Evaluation", "full_session_frame_failed",
                        {"error": repr(exc)}, frame_id=packet.frame_id, level="ERROR",
                    )
                finally:
                    self.queue.task_done()
        if writer is not None:
            writer.release()

    def _manifest(self, status: str) -> dict:
        duration = None
        if self.first_monotonic_sec is not None and self.last_monotonic_sec is not None:
            duration = self.last_monotonic_sec - self.first_monotonic_sec
        video_ok = self.video_path.is_file() and self.video_path.stat().st_size > 0
        return {
            "schema_version": "1.0.0",
            "session_id": self.session_id,
            "status": status,
            "created_utc": self.first_utc or datetime.now(timezone.utc).isoformat(),
            "ended_utc": self.last_utc,
            "first_monotonic_sec": self.first_monotonic_sec,
            "last_monotonic_sec": self.last_monotonic_sec,
            "duration_sec": duration,
            "video_duration_sec": self.written_frames / self.fps if self.fps else None,
            "target_fps": self.fps,
            "written_frames": self.written_frames,
            "dropped_frames": self.dropped_frames,
            "recording_errors": self.recording_errors,
            "encoder": self.encoder,
            "video_sha256": _sha256(self.video_path) if video_ok else None,
            "files": {
                "video": "session.mp4" if self.record_video else None,
                "frame_timestamps": "frame_timestamps.csv",
                "telemetry": "telemetry.jsonl",
                "annotations": "annotations.csv",
                "report": "evaluation_report.md",
            },
            **self.metadata,
        }

    def _write_manifest(self, status: str) -> None:
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self._manifest(status), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(self.manifest_path)

    def close(self) -> dict:
        self.queue.join()
        self.queue.put(None)
        self.queue.join()
        self.worker.join(timeout=15.0)
        if self.worker.is_alive():
            raise RuntimeError("Evaluation recorder did not stop cleanly")
        video_ok = (
            not self.record_video
            or (self.video_path.is_file() and self.video_path.stat().st_size > 0)
        )
        if not self.written_frames:
            status = "EMPTY"
        elif not video_ok or self.recording_errors:
            status = "RECORDER_FAILED"
        else:
            status = "READY_FOR_ANNOTATION"
        self._write_manifest(status)
        manifest = self._manifest(status)
        self.telemetry.emit(
            "Evaluation", "session_finalized",
            {
                "session_dir": str(self.session_dir),
                "written_frames": self.written_frames,
                "dropped_frames": self.dropped_frames,
                "status": status,
            },
        )
        return manifest
