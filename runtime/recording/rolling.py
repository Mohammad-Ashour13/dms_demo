from __future__ import annotations

import hashlib
import json
import queue
import shutil
import subprocess
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from dms_final_system.shared.contracts import FramePacket, IncidentRecord
from dms_final_system.runtime.integration.sinks import FilesystemOutboxSink, IncidentSink


@dataclass(slots=True)
class EncodedFrame:
    monotonic_sec: float
    utc_timestamp: str
    jpeg: bytes


@dataclass(slots=True)
class ActiveIncident:
    incident_id: str
    started_sec: float
    started_utc: str
    post_until_sec: float
    frames: list[EncodedFrame] = field(default_factory=list)
    highest_state: str = "NORMAL"
    violations: set[str] = field(default_factory=set)
    reason_codes: set[str] = field(default_factory=set)
    max_probability: float = 0.0
    summary: dict = field(default_factory=dict)
    parent_incident_id: str | None = None
    part_index: int = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RollingIncidentRecorder:
    """Compressed rolling buffer with asynchronous H.264 encoding and mp4v fallback."""

    STATE_RANK = {"UNKNOWN": -1, "NORMAL": 0, "FATIGUE_WARNING": 1, "DROWSY": 2, "CRITICAL": 3}

    def __init__(
        self,
        output_dir: Path,
        session_id: str,
        telemetry,
        sink: IncidentSink | None = None,
        pre_alert_sec=10.0,
        post_alert_sec=10.0,
        merge_gap_sec=5.0,
        max_clip_sec=60.0,
        fps=15.0,
        jpeg_quality=75,
        bitrate="1800k",
        queue_size=256,
    ):
        self.output_dir, self.session_id = Path(output_dir), session_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry, self.sink = telemetry, sink or FilesystemOutboxSink()
        self.pre_alert, self.post_alert, self.merge_gap, self.max_clip = pre_alert_sec, post_alert_sec, merge_gap_sec, max_clip_sec
        self.fps, self.jpeg_quality, self.bitrate = fps, jpeg_quality, bitrate
        self.queue: queue.Queue[FramePacket | None] = queue.Queue(maxsize=queue_size)
        self.buffer: deque[EncodedFrame] = deque()
        self.active: ActiveIncident | None = None
        self.continuation: tuple[str, int, float] | None = None
        self.lock = threading.RLock()
        self.dropped_frames = 0
        self.ffmpeg_h264_available = self._probe_ffmpeg_h264()
        self.telemetry.emit(
            "Recorder", "encoder_probe",
            {"preferred": "ffmpeg_h264", "available": self.ffmpeg_h264_available,
             "fallback": "opencv_mp4v"},
        )
        self.worker = threading.Thread(target=self._run, name="rolling-recorder", daemon=True)
        self.worker.start()

    @staticmethod
    def _probe_ffmpeg_h264() -> bool:
        if not shutil.which("ffmpeg"):
            return False
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True,
                text=True, timeout=8, check=False,
            )
            return result.returncode == 0 and "libx264" in result.stdout
        except (OSError, subprocess.SubprocessError):
            return False

    def push(self, packet: FramePacket) -> None:
        try:
            self.queue.put_nowait(packet)
        except queue.Full:
            self.dropped_frames += 1

    def trigger(self, now: float, state: str, violations: list[str], reason_codes: list[str], probability: float, summary: dict) -> str:
        with self.lock:
            if self.active is None:
                continuation = self.continuation if self.continuation and now <= self.continuation[2] else None
                if continuation:
                    parent_id, part_index, _ = continuation
                    incident_id = f"{parent_id}-part{part_index:02d}"
                    preframes = [frame for frame in self.buffer if frame.monotonic_sec >= now - 0.25]
                    self.continuation = None
                else:
                    parent_id, part_index = None, 1
                    incident_id = f"{self.session_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
                    preframes = [frame for frame in self.buffer if frame.monotonic_sec >= now - self.pre_alert]
                self.active = ActiveIncident(
                    incident_id, now, datetime.now(timezone.utc).isoformat(),
                    now + self.post_alert, list(preframes),
                    parent_incident_id=parent_id, part_index=part_index,
                )
                self.telemetry.emit("Recorder", "incident_started", {"pre_frames": len(preframes)}, monotonic_sec=now, incident_id=incident_id)
            incident = self.active
            incident.post_until_sec = max(incident.post_until_sec, now + self.post_alert)
            if self.STATE_RANK.get(state, 0) > self.STATE_RANK.get(incident.highest_state, 0):
                incident.highest_state = state
            incident.violations.update(violations)
            incident.reason_codes.update(reason_codes)
            incident.max_probability = max(incident.max_probability, probability)
            incident.summary.update(summary)
            return incident.incident_id

    def _run(self) -> None:
        while True:
            packet = self.queue.get()
            if packet is None:
                self.queue.task_done()
                break
            ok, encoded = cv2.imencode(".jpg", packet.frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not ok:
                self.telemetry.emit("Recorder", "jpeg_encode_failed", frame_id=packet.frame_id, level="ERROR")
                self.queue.task_done()
                continue
            frame = EncodedFrame(packet.monotonic_sec, packet.utc_timestamp, encoded.tobytes())
            finalize = None
            with self.lock:
                self.buffer.append(frame)
                cutoff = frame.monotonic_sec - self.pre_alert
                while self.buffer and self.buffer[0].monotonic_sec < cutoff:
                    self.buffer.popleft()
                if self.active:
                    if not self.active.frames or frame.monotonic_sec > self.active.frames[-1].monotonic_sec:
                        self.active.frames.append(frame)
                    duration = frame.monotonic_sec - self.active.frames[0].monotonic_sec
                    hit_max_duration = duration >= self.max_clip
                    if frame.monotonic_sec >= self.active.post_until_sec or hit_max_duration:
                        finalize, self.active = self.active, None
                        if hit_max_duration:
                            parent_id = finalize.parent_incident_id or finalize.incident_id
                            self.continuation = (parent_id, finalize.part_index + 1, frame.monotonic_sec + self.merge_gap)
            if finalize is not None:
                try:
                    self._finalize(finalize)
                except Exception as exc:
                    self.telemetry.emit(
                        "Recorder", "incident_finalize_failed",
                        {"incident_id": finalize.incident_id, "error": repr(exc)},
                        monotonic_sec=frame.monotonic_sec, incident_id=finalize.incident_id,
                        level="ERROR",
                    )
            self.queue.task_done()

    def _write_ffmpeg(self, path: Path, frames: list[EncodedFrame]) -> bool:
        if not self.ffmpeg_h264_available or not frames:
            return False
        first = cv2.imdecode(np.frombuffer(frames[0].jpeg, np.uint8), cv2.IMREAD_COLOR)
        if first is None:
            return False
        height, width = first.shape[:2]
        command = [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(self.fps), "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-b:v", self.bitrate,
            "-g", str(max(1, int(round(self.fps)))), "-keyint_min", str(max(1, int(round(self.fps)))), "-sc_threshold", "0",
            "-movflags", "+faststart", str(path),
        ]
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            assert process.stdin is not None
            for item in frames:
                image = cv2.imdecode(np.frombuffer(item.jpeg, np.uint8), cv2.IMREAD_COLOR)
                if image is not None:
                    process.stdin.write(image.tobytes())
            process.stdin.close()
            return process.wait(timeout=45) == 0 and path.is_file() and path.stat().st_size > 0
        except (OSError, subprocess.SubprocessError, BrokenPipeError):
            process.kill() if "process" in locals() else None
            path.unlink(missing_ok=True)
            return False

    def _write_opencv(self, path: Path, frames: list[EncodedFrame]) -> None:
        first = cv2.imdecode(np.frombuffer(frames[0].jpeg, np.uint8), cv2.IMREAD_COLOR)
        if first is None:
            raise RuntimeError("No decodable incident frames")
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("OpenCV mp4v recorder fallback failed to start")
        try:
            for item in frames:
                image = cv2.imdecode(np.frombuffer(item.jpeg, np.uint8), cv2.IMREAD_COLOR)
                if image is not None:
                    writer.write(image)
        finally:
            writer.release()

    def _finalize(self, incident: ActiveIncident) -> None:
        directory = self.output_dir / incident.incident_id
        directory.mkdir(parents=True, exist_ok=False)
        video_path, telemetry_path = directory / "video.mp4", directory / "telemetry.jsonl"
        backend = "ffmpeg_h264" if self._write_ffmpeg(video_path, incident.frames) else "opencv_mp4v"
        if backend == "opencv_mp4v":
            self._write_opencv(video_path, incident.frames)
        end_frame = incident.frames[-1] if incident.frames else None
        end_sec = end_frame.monotonic_sec if end_frame else incident.post_until_sec
        self.telemetry.export_range(telemetry_path, incident.started_sec - self.pre_alert, end_sec, incident.incident_id)
        record = IncidentRecord(
            incident.incident_id, self.session_id, incident.started_utc,
            end_frame.utc_timestamp if end_frame else datetime.now(timezone.utc).isoformat(),
            incident.highest_state, sorted(incident.violations), incident.max_probability,
            sorted(incident.reason_codes), incident.summary.get("model_version", "unknown"),
            incident.summary.get("feature_version", "unknown"), incident.summary.get("fusion_version", "unknown"),
            video_path, telemetry_path, _sha256(video_path), "PENDING",
        )
        payload = record.to_dict()
        payload.update({
            "encoder_backend": backend, "frame_count": len(incident.frames),
            "event_summary": incident.summary,
            "parent_incident_id": incident.parent_incident_id,
            "part_index": incident.part_index,
        })
        (directory / "incident.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.sink.publish(record)
        self.telemetry.emit("Recorder", "incident_finalized", {"path": str(directory), "backend": backend, "frames": len(incident.frames)}, monotonic_sec=end_sec, incident_id=incident.incident_id)

    def close(self) -> None:
        self.queue.join()
        self.queue.put(None)
        self.queue.join()
        self.worker.join(timeout=10.0)
        with self.lock:
            if self.active and self.active.frames:
                incident, self.active = self.active, None
                self._finalize(incident)
