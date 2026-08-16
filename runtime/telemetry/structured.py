from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable


def _json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


class StructuredTelemetry:
    """Non-blocking JSONL logger plus an in-memory tail for incident export."""

    _NORMAL_MIN_INTERVAL_SEC = {
        ("Perception", "face_signal"): 5.0,
        ("Calibration", "status"): 5.0,
        ("Events", "snapshot"): 5.0,
        ("Fusion", "decision"): 5.0,
        ("Fusion", "latency"): 10.0,
        ("Window", "feature_snapshot"): 5.0,
        ("Model", "feature_drift"): 5.0,
        ("Model", "prediction"): 5.0,
        ("BehaviorDetector", "inference"): 5.0,
        ("SeatbeltDetector", "inference"): 5.0,
        ("SeatbeltDetector", "inference_skipped"): 30.0,
    }

    def __init__(
        self,
        log_dir: Path,
        session_id: str,
        mode="NORMAL",
        rotate_bytes=20 * 1024 * 1024,
        backups=5,
        filename: str | None = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id, self.mode = session_id, mode.upper()
        self.path = self.log_dir / (filename or f"{session_id}.jsonl")
        self.pruned_log_files = self._prune_old_logs(backups) if filename is None else 0
        self.handler = RotatingFileHandler(self.path, maxBytes=rotate_bytes, backupCount=backups, encoding="utf-8")
        self.handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger = logging.getLogger(f"dms.telemetry.{session_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.handlers.clear()
        self.logger.addHandler(self.handler)
        self.queue: queue.Queue[dict | None] = queue.Queue(maxsize=10_000)
        self.tail: deque[dict] = deque()
        self.tail_lock = threading.Lock()
        self.subscribers: list[Callable[[dict[str, Any]], None]] = []
        self.subscriber_lock = threading.Lock()
        self.throttle_lock = threading.Lock()
        self.last_emitted: dict[tuple[str, str], float] = {}
        self.dropped_records = 0
        self.worker = threading.Thread(target=self._run, name="telemetry-writer", daemon=True)
        self.worker.start()

    def _prune_old_logs(self, keep: int) -> int:
        """Bound telemetry across restarts, not only within one session."""
        candidates = sorted(
            (
                path for path in self.log_dir.glob("*.jsonl*")
                if path.is_file() and not path.is_symlink() and path != self.path
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        removed = 0
        for path in candidates[max(0, int(keep)):]:
            try:
                path.unlink()
                removed += 1
            except OSError:
                continue
        return removed

    def _should_emit(
        self,
        stage: str,
        event: str,
        payload: dict,
        monotonic_sec: float,
        level: str,
        incident_id,
    ) -> bool:
        if self.mode == "DEBUG" or str(level).upper() != "INFO" or incident_id is not None:
            return True
        interval = self._NORMAL_MIN_INTERVAL_SEC.get((str(stage), str(event)))
        if interval is None or bool(payload.get("changed")):
            return True
        key = (str(stage), str(event))
        with self.throttle_lock:
            previous = self.last_emitted.get(key, float("-inf"))
            if monotonic_sec - previous < interval:
                return False
            self.last_emitted[key] = monotonic_sec
        return True

    def emit(self, stage: str, event: str, payload: dict | None = None, *, monotonic_sec: float | None = None, frame_id=None, window_id=None, incident_id=None, level="INFO") -> None:
        now_mono = time.monotonic() if monotonic_sec is None else monotonic_sec
        payload = payload or {}
        if not self._should_emit(stage, event, payload, now_mono, level, incident_id):
            return
        record = {
            "utc_timestamp": datetime.now(timezone.utc).isoformat(),
            "monotonic_sec": now_mono,
            "session_id": self.session_id,
            "frame_id": frame_id,
            "window_id": window_id,
            "incident_id": incident_id,
            "stage": stage,
            "event": event,
            "level": level,
            "payload": payload,
        }
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            self.dropped_records += 1
            return
        with self.subscriber_lock:
            subscribers = list(self.subscribers)
        for subscriber in subscribers:
            try:
                subscriber(record)
            except Exception:
                # A remote/optional sink can never interfere with the local
                # structured log or the safety loop.
                continue

    def add_subscriber(self, subscriber: Callable[[dict[str, Any]], None]) -> None:
        with self.subscriber_lock:
            if subscriber not in self.subscribers:
                self.subscribers.append(subscriber)

    def remove_subscriber(self, subscriber: Callable[[dict[str, Any]], None]) -> None:
        with self.subscriber_lock:
            try:
                self.subscribers.remove(subscriber)
            except ValueError:
                pass

    def _run(self) -> None:
        while True:
            record = self.queue.get()
            if record is None:
                self.queue.task_done()
                break
            encoded = json.dumps(record, ensure_ascii=False, default=_json_default, separators=(",", ":"))
            self.logger.info(encoded)
            with self.tail_lock:
                self.tail.append(record)
                cutoff = record["monotonic_sec"] - 75.0
                while self.tail and self.tail[0]["monotonic_sec"] < cutoff:
                    self.tail.popleft()
            self.queue.task_done()
        self.logger.removeHandler(self.handler)
        self.handler.close()

    def export_range(self, path: Path, start_sec: float, end_sec: float, incident_id: str) -> None:
        self.queue.join()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.tail_lock:
            records = [dict(item, incident_id=incident_id) for item in self.tail if start_sec <= item["monotonic_sec"] <= end_sec]
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

    def close(self) -> None:
        self.queue.join()
        self.queue.put(None)
        self.queue.join()
        self.worker.join(timeout=5.0)


class LiveStatus:
    def __init__(self, hz: float = 2.0):
        self.interval = 1.0 / max(hz, 0.1)
        self.last = float("-inf")

    def show(self, now: float, **values) -> None:
        if now - self.last < self.interval:
            return
        self.last = now
        summary = " | ".join(f"{key}={value}" for key, value in values.items())
        print(f"\r[DMS] {summary}", end="", flush=True)


def system_health() -> dict[str, float | None]:
    cpu = ram = temperature = None
    try:
        import psutil
        cpu, ram = psutil.cpu_percent(interval=None), psutil.virtual_memory().percent
    except ImportError:
        pass
    thermal = Path("/sys/class/thermal/thermal_zone0/temp")
    try:
        temperature = float(thermal.read_text().strip()) / 1000.0
    except (OSError, ValueError):
        pass
    return {"cpu_percent": cpu, "ram_percent": ram, "temperature_c": temperature}
