from __future__ import annotations

import multiprocessing as mp
import queue
import time
from dataclasses import asdict
from pathlib import Path

from dms_final_system.runtime.recording.frames import EncodedFrame


def _dashboard_entry(config: dict, incident_dir: str, status_queue, preview_queue, clients) -> None:
    from .server import run_dashboard_server

    run_dashboard_server(config, Path(incident_dir), status_queue, preview_queue, clients)


class DashboardProcess:
    """Failure-isolated dashboard with bounded, latest-only IPC."""

    def __init__(self, config, incident_dir: Path, telemetry):
        self.config = config
        self.incident_dir = Path(incident_dir)
        self.telemetry = telemetry
        self.enabled = bool(config.enabled)
        self.context = mp.get_context("spawn")
        self.status_queue = self.context.Queue(maxsize=1)
        self.preview_queue = self.context.Queue(maxsize=2)
        self.clients = self.context.Value("i", 0)
        self.safe_mode = False
        self.last_preview_sent = float("-inf")
        self.dropped_status = 0
        self.dropped_preview = 0
        self.published_preview = 0
        self.process = None

    @staticmethod
    def _put_latest(target_queue, value) -> bool:
        try:
            target_queue.put_nowait(value)
            return True
        except queue.Full:
            try:
                target_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                target_queue.put_nowait(value)
                return True
            except queue.Full:
                return False

    def start(self) -> "DashboardProcess":
        if not self.enabled:
            return self
        self.process = self.context.Process(
            target=_dashboard_entry,
            args=(
                asdict(self.config),
                str(self.incident_dir),
                self.status_queue,
                self.preview_queue,
                self.clients,
            ),
            name="safee-dashboard",
            daemon=True,
        )
        self.process.start()
        self.telemetry.emit(
            "Dashboard",
            "process_started",
            {"pid": self.process.pid, "host": self.config.host, "port": self.config.port},
        )
        return self

    def publish_status(self, status: dict) -> None:
        if not self.enabled or self.process is None or not self.process.is_alive():
            return
        if not self._put_latest(self.status_queue, status):
            self.dropped_status += 1

    def publish_preview(self, frame: EncodedFrame) -> None:
        if not self.enabled or self.process is None or not self.process.is_alive():
            return
        if self.clients.value <= 0:
            return
        fps = self.config.safe_preview_fps if self.safe_mode else self.config.preview_fps
        if frame.monotonic_sec - self.last_preview_sent < 1.0 / max(float(fps), 0.1):
            return
        self.last_preview_sent = frame.monotonic_sec
        value = (frame.frame_id, frame.monotonic_sec, frame.utc_timestamp, frame.jpeg)
        if not self._put_latest(self.preview_queue, value):
            self.dropped_preview += 1
        else:
            self.published_preview += 1

    def set_safe_mode(self, enabled: bool) -> None:
        self.safe_mode = bool(enabled)

    @property
    def active_clients(self) -> int:
        return int(self.clients.value) if self.enabled else 0

    @property
    def healthy(self) -> bool:
        return bool(not self.enabled or (self.process is not None and self.process.is_alive()))

    def close(self) -> None:
        if self.process is None:
            return
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5.0)
        self.telemetry.emit(
            "Dashboard",
            "process_stopped",
            {"exitcode": self.process.exitcode},
            level="INFO" if self.process.exitcode in {None, 0, -15} else "WARNING",
        )
