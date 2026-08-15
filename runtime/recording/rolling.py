from __future__ import annotations

import hashlib
import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from dms_final_system.runtime.integration.sinks import FilesystemOutboxSink, IncidentSink
from dms_final_system.shared.contracts import IncidentRecord

from .frames import EncodedFrame


@dataclass(slots=True)
class ActiveIncident:
    incident_id: str
    triggered_sec: float
    started_utc: str
    post_until_sec: float
    frames: list[EncodedFrame] = field(default_factory=list)
    highest_state: str = "NORMAL"
    violations: set[str] = field(default_factory=set)
    reason_codes: set[str] = field(default_factory=set)
    trigger_kinds: set[str] = field(default_factory=set)
    max_probability: float = 0.0
    summary: dict = field(default_factory=dict)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class RollingIncidentRecorder:
    """RAM JPEG ring with fixed clips, copy-only MP4 muxing and atomic finalization."""

    STATE_RANK = {
        "UNKNOWN": -1,
        "NORMAL": 0,
        "FATIGUE_WARNING": 1,
        "DROWSY": 2,
        "CRITICAL": 3,
    }

    def __init__(
        self,
        output_dir: Path,
        session_id: str,
        telemetry,
        sink: IncidentSink | None = None,
        pre_alert_sec=5.0,
        post_alert_sec=5.0,
        merge_gap_sec=0.0,
        max_clip_sec=10.0,
        fps=15.0,
        jpeg_quality=75,
        bitrate="1800k",
        queue_size=256,
        codec="mjpeg_copy",
        require_copy_mux=False,
        reserve_free_bytes=2 * 1024 * 1024 * 1024,
        delete_oldest_when_full=True,
    ):
        del merge_gap_sec, jpeg_quality
        self.output_dir, self.session_id = Path(output_dir), str(session_id)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry, self.sink = telemetry, sink or FilesystemOutboxSink()
        self.pre_alert = float(pre_alert_sec)
        self.post_alert = float(post_alert_sec)
        self.max_clip = float(max_clip_sec)
        self.fps, self.bitrate = float(fps), str(bitrate)
        self.codec = str(codec)
        self.require_copy_mux = bool(require_copy_mux)
        self.reserve_free_bytes = int(reserve_free_bytes)
        self.delete_oldest_when_full = bool(delete_oldest_when_full)
        self.queue: queue.Queue[EncodedFrame | None] = queue.Queue(maxsize=max(1, int(queue_size)))
        self.finalize_queue: queue.Queue[ActiveIncident | None] = queue.Queue(maxsize=2)
        self.buffer: deque[EncodedFrame] = deque()
        self.active: ActiveIncident | None = None
        self.lock = threading.RLock()
        self.dropped_frames = 0
        self.finalize_dropped = 0
        self.finalized_incidents = 0
        self.deleted_incidents = 0
        self.last_started_incident_id: str | None = None
        self.last_finalized_incident_id: str | None = None
        self.last_error = ""
        self._incident_count = 0
        self.copy_mux_available = self._probe_copy_mux()
        self._recover_partial_directories()
        self._ensure_storage_reserve()
        self._incident_count = len(self._incident_directories())
        self.telemetry.emit(
            "Recorder",
            "encoder_probe",
            {
                "preferred": self.codec,
                "copy_mux_available": self.copy_mux_available,
                "require_copy_mux": self.require_copy_mux,
                "pixel_reencode": self.codec != "mjpeg_copy",
            },
        )
        if self.codec == "mjpeg_copy" and self.require_copy_mux and not self.copy_mux_available:
            raise RuntimeError("Recorder requires FFmpeg MJPEG copy-mux support")
        self.worker = threading.Thread(target=self._run, name="rolling-recorder", daemon=True)
        self.finalizer = threading.Thread(target=self._finalize_loop, name="incident-muxer", daemon=True)
        self.worker.start()
        self.finalizer.start()

    @staticmethod
    def _probe_copy_mux() -> bool:
        if not shutil.which("ffmpeg"):
            return False
        try:
            ok, encoded = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
            if not ok:
                return False
            with tempfile.TemporaryDirectory(prefix="dms-copy-mux-") as directory:
                target = Path(directory) / "probe.mp4"
                result = subprocess.run(
                    [
                        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-f", "image2pipe", "-vcodec", "mjpeg", "-framerate", "1",
                        "-i", "pipe:0", "-an", "-c:v", "copy", str(target),
                    ],
                    input=encoded.tobytes(),
                    capture_output=True,
                    timeout=8,
                    check=False,
                )
                return result.returncode == 0 and target.is_file() and target.stat().st_size > 0
        except (OSError, subprocess.SubprocessError):
            return False

    def _recover_partial_directories(self) -> None:
        for partial in sorted(self.output_dir.glob(".*.partial")):
            incident_id = partial.name[1:-8]
            final = self.output_dir / incident_id
            complete = all(
                (partial / name).is_file()
                for name in (
                    "video.mp4",
                    "video.mp4.sha256",
                    "incident.json",
                    "telemetry.jsonl",
                )
            ) and (partial / "video.mp4").stat().st_size > 0
            if complete:
                try:
                    expected = (partial / "video.mp4.sha256").read_text(
                        encoding="utf-8"
                    ).split()[0]
                    json.loads((partial / "incident.json").read_text(encoding="utf-8"))
                    complete = expected == _sha256(partial / "video.mp4")
                except (OSError, ValueError, IndexError):
                    complete = False
            if complete and not final.exists():
                partial.replace(final)
                self.telemetry.emit(
                    "Recorder", "partial_recovered", {"incident_id": incident_id, "path": str(final)}
                )
                continue
            quarantine = self.output_dir / (
                f".{incident_id}.quarantine-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            )
            if not quarantine.exists():
                partial.replace(quarantine)
                self.telemetry.emit(
                    "Recorder",
                    "partial_quarantined",
                    {"incident_id": incident_id, "path": str(quarantine)},
                    level="WARNING",
                )

    def push_encoded(self, frame: EncodedFrame) -> None:
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            self.dropped_frames += 1

    def trigger(
        self,
        now: float,
        state: str,
        violations: list[str],
        reason_codes: list[str],
        probability: float,
        summary: dict,
        *,
        start_new=True,
        trigger_kinds: list[str] | tuple[str, ...] = (),
    ) -> str | None:
        with self.lock:
            if self.active is None:
                if not start_new:
                    return None
                incident_id = (
                    f"{self.session_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
                    f"{uuid.uuid4().hex[:8]}"
                )
                preframes = [
                    frame for frame in self.buffer if frame.monotonic_sec >= now - self.pre_alert
                ]
                self.active = ActiveIncident(
                    incident_id,
                    float(now),
                    datetime.now(timezone.utc).isoformat(),
                    float(now) + self.post_alert,
                    list(preframes),
                )
                self.last_started_incident_id = incident_id
                self.last_error = ""
                self.telemetry.emit(
                    "Recorder",
                    "incident_started",
                    {"pre_frames": len(preframes), "fixed_post_sec": self.post_alert},
                    monotonic_sec=now,
                    incident_id=incident_id,
                )
            incident = self.active
            if self.STATE_RANK.get(state, 0) > self.STATE_RANK.get(incident.highest_state, 0):
                incident.highest_state = state
            incident.violations.update(violations)
            incident.reason_codes.update(reason_codes)
            incident.trigger_kinds.update(trigger_kinds)
            incident.max_probability = max(incident.max_probability, float(probability))
            incident.summary.update(summary)
            return incident.incident_id

    def _run(self) -> None:
        while True:
            frame = self.queue.get()
            if frame is None:
                self.queue.task_done()
                return
            finalize = None
            with self.lock:
                self.buffer.append(frame)
                cutoff = frame.monotonic_sec - self.pre_alert
                while self.buffer and self.buffer[0].monotonic_sec < cutoff:
                    self.buffer.popleft()
                if self.active:
                    if not self.active.frames or frame.monotonic_sec > self.active.frames[-1].monotonic_sec:
                        self.active.frames.append(frame)
                    if frame.monotonic_sec >= self.active.post_until_sec:
                        finalize, self.active = self.active, None
            if finalize is not None:
                try:
                    self.finalize_queue.put_nowait(finalize)
                except queue.Full:
                    self.finalize_dropped += 1
                    self.last_error = "finalize_queue_full"
                    self.telemetry.emit(
                        "Recorder",
                        "finalize_queue_full",
                        {"incident_id": finalize.incident_id},
                        monotonic_sec=frame.monotonic_sec,
                        incident_id=finalize.incident_id,
                        level="ERROR",
                    )
            self.queue.task_done()

    def _write_mjpeg_copy(self, path: Path, frames: list[EncodedFrame]) -> bool:
        if not self.copy_mux_available or not frames:
            return False
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-framerate",
            f"{self.fps:g}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "copy",
            "-movflags",
            "+faststart",
            str(path),
        ]
        process = None
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            assert process.stdin is not None
            for frame in frames:
                process.stdin.write(frame.jpeg)
            process.stdin.close()
            stderr = process.stderr.read() if process.stderr is not None else b""
            ok = process.wait(timeout=30) == 0 and path.is_file() and path.stat().st_size > 0
            if not ok:
                self.telemetry.emit(
                    "Recorder",
                    "copy_mux_failed",
                    {"stderr": stderr.decode("utf-8", errors="replace")[-1000:]},
                    level="ERROR",
                )
            return ok
        except (OSError, subprocess.SubprocessError, BrokenPipeError):
            if process is not None:
                process.kill()
            path.unlink(missing_ok=True)
            return False

    def _write_legacy(self, path: Path, frames: list[EncodedFrame]) -> None:
        first = cv2.imdecode(np.frombuffer(frames[0].jpeg, np.uint8), cv2.IMREAD_COLOR)
        if first is None:
            raise RuntimeError("No decodable incident frames")
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError("Legacy OpenCV recorder failed to start")
        try:
            for item in frames:
                image = cv2.imdecode(np.frombuffer(item.jpeg, np.uint8), cv2.IMREAD_COLOR)
                if image is not None:
                    writer.write(image)
        finally:
            writer.release()

    def _incident_directories(self) -> list[Path]:
        result = []
        for path in self.output_dir.iterdir():
            if path.is_dir() and not path.name.startswith(".") and (path / "incident.json").is_file():
                result.append(path)
        return sorted(result, key=lambda path: path.stat().st_mtime)

    @property
    def incident_count(self) -> int:
        return self._incident_count

    def _ensure_storage_reserve(self) -> None:
        if self.reserve_free_bytes <= 0:
            return
        usage = shutil.disk_usage(self.output_dir)
        if usage.free >= self.reserve_free_bytes:
            return
        if not self.delete_oldest_when_full:
            raise RuntimeError(
                f"Recorder free space {usage.free} is below reserve {self.reserve_free_bytes}"
            )
        for directory in self._incident_directories():
            resolved = directory.resolve()
            if resolved.parent != self.output_dir.resolve():
                continue
            shutil.rmtree(resolved)
            self.deleted_incidents += 1
            self._incident_count = max(0, self._incident_count - 1)
            self.telemetry.emit(
                "Recorder",
                "incident_deleted_for_space",
                {"path": str(resolved), "reserve_free_bytes": self.reserve_free_bytes},
                level="WARNING",
            )
            if shutil.disk_usage(self.output_dir).free >= self.reserve_free_bytes:
                return
        if shutil.disk_usage(self.output_dir).free < self.reserve_free_bytes:
            raise RuntimeError("Recorder cannot restore configured free-space reserve")

    def _finalize_loop(self) -> None:
        while True:
            incident = self.finalize_queue.get()
            if incident is None:
                self.finalize_queue.task_done()
                return
            try:
                self._finalize(incident)
            except Exception as exc:
                self.last_error = repr(exc)
                self.telemetry.emit(
                    "Recorder",
                    "incident_finalize_failed",
                    {"incident_id": incident.incident_id, "error": repr(exc)},
                    incident_id=incident.incident_id,
                    level="ERROR",
                )
            finally:
                self.finalize_queue.task_done()

    def _finalize(self, incident: ActiveIncident) -> None:
        if not incident.frames:
            raise RuntimeError("Cannot finalize an incident without frames")
        self._ensure_storage_reserve()
        final_directory = self.output_dir / incident.incident_id
        partial = self.output_dir / f".{incident.incident_id}.partial"
        if final_directory.exists() or partial.exists():
            raise FileExistsError(f"Incident directory already exists: {incident.incident_id}")
        partial.mkdir(parents=False)
        video_path = partial / "video.mp4"
        telemetry_path = partial / "telemetry.jsonl"
        if self.codec == "mjpeg_copy":
            if not self._write_mjpeg_copy(video_path, incident.frames):
                if self.require_copy_mux:
                    raise RuntimeError("Required MJPEG copy mux failed")
                self._write_legacy(video_path, incident.frames)
                encoder_backend = "opencv_mp4v_fallback"
            else:
                encoder_backend = "ffmpeg_mjpeg_copy"
        else:
            self._write_legacy(video_path, incident.frames)
            encoder_backend = "opencv_mp4v_legacy"
        first_frame, end_frame = incident.frames[0], incident.frames[-1]
        self.telemetry.export_range(
            telemetry_path,
            incident.triggered_sec - self.pre_alert,
            end_frame.monotonic_sec,
            incident.incident_id,
        )
        final_video_path = final_directory / "video.mp4"
        final_telemetry_path = final_directory / "telemetry.jsonl"
        video_sha256 = _sha256(video_path)
        checksum_path = partial / "video.mp4.sha256"
        checksum_path.write_text(
            f"{video_sha256}  video.mp4\n", encoding="utf-8"
        )
        record = IncidentRecord(
            incident.incident_id,
            self.session_id,
            first_frame.utc_timestamp,
            end_frame.utc_timestamp,
            incident.highest_state,
            sorted(incident.violations),
            incident.max_probability,
            sorted(incident.reason_codes),
            incident.summary.get("model_version", "unknown"),
            incident.summary.get("feature_version", "unknown"),
            incident.summary.get("fusion_version", "unknown"),
            final_video_path,
            final_telemetry_path,
            video_sha256,
            "LOCAL_ONLY",
        )
        payload = record.to_dict()
        payload.update(
            {
                "encoder_backend": encoder_backend,
                "pixel_reencode": encoder_backend != "ffmpeg_mjpeg_copy",
                "frame_count": len(incident.frames),
                "trigger_monotonic_sec": incident.triggered_sec,
                "actual_start_monotonic_sec": first_frame.monotonic_sec,
                "actual_end_monotonic_sec": end_frame.monotonic_sec,
                "actual_duration_sec": end_frame.monotonic_sec - first_frame.monotonic_sec,
                "trigger_kinds": sorted(incident.trigger_kinds),
                "event_summary": incident.summary,
            }
        )
        incident_path = partial / "incident.json"
        incident_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        for path in (video_path, checksum_path, telemetry_path, incident_path):
            _fsync_file(path)
        _fsync_directory(partial)
        partial.replace(final_directory)
        _fsync_directory(self.output_dir)
        self.finalized_incidents += 1
        self.last_finalized_incident_id = incident.incident_id
        self.last_error = ""
        self._incident_count += 1
        try:
            self.sink.publish(record)
        except Exception as exc:
            self.telemetry.emit(
                "Recorder",
                "incident_sink_failed",
                {"incident_id": incident.incident_id, "error": repr(exc)},
                monotonic_sec=end_frame.monotonic_sec,
                incident_id=incident.incident_id,
                level="ERROR",
            )
        self.telemetry.emit(
            "Recorder",
            "incident_finalized",
            {
                "path": str(final_directory),
                "backend": encoder_backend,
                "frames": len(incident.frames),
            },
            monotonic_sec=end_frame.monotonic_sec,
            incident_id=incident.incident_id,
        )
        self._ensure_storage_reserve()

    def close(self) -> None:
        self.queue.join()
        self.queue.put(None)
        self.queue.join()
        self.worker.join(timeout=10.0)
        with self.lock:
            if self.active and self.active.frames:
                incident, self.active = self.active, None
                self.finalize_queue.put(incident)
        self.finalize_queue.join()
        self.finalize_queue.put(None)
        self.finalize_queue.join()
        self.finalizer.join(timeout=10.0)
        if self.worker.is_alive() or self.finalizer.is_alive():
            raise RuntimeError("Recorder workers did not stop cleanly")
