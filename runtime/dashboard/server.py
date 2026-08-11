from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from pathlib import Path


class DashboardState:
    def __init__(self, status_queue, preview_queue):
        self.status_queue = status_queue
        self.preview_queue = preview_queue
        self.lock = threading.Lock()
        self.status = {
            "schema_version": "dashboard-status-v1",
            "runtime": {"status": "STARTING"},
        }
        self.preview = None
        self.stop = threading.Event()
        self.threads = [
            threading.Thread(target=self._status_loop, name="dashboard-status-ipc", daemon=True),
            threading.Thread(target=self._preview_loop, name="dashboard-preview-ipc", daemon=True),
        ]

    def start(self) -> None:
        for thread in self.threads:
            thread.start()

    def _status_loop(self) -> None:
        while not self.stop.is_set():
            try:
                value = self.status_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            with self.lock:
                self.status = value

    def _preview_loop(self) -> None:
        while not self.stop.is_set():
            try:
                value = self.preview_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            with self.lock:
                self.preview = value

    def status_snapshot(self):
        with self.lock:
            return dict(self.status)

    def preview_snapshot(self):
        with self.lock:
            return self.preview


def _read_incidents(incident_dir: Path, limit=50) -> list[dict]:
    records = []
    if not incident_dir.is_dir():
        return records
    for path in incident_dir.iterdir():
        metadata = path / "incident.json"
        if not path.is_dir() or path.name.startswith(".") or not metadata.is_file():
            continue
        try:
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            video = path / "video.mp4"
            records.append(
                {
                    "incident_id": payload.get("incident_id", path.name),
                    "started_utc": payload.get("started_utc"),
                    "ended_utc": payload.get("ended_utc"),
                    "highest_state": payload.get("highest_state"),
                    "violations": payload.get("violations") or [],
                    "trigger_kinds": payload.get("trigger_kinds") or [],
                    "duration_sec": payload.get("actual_duration_sec"),
                    "frame_count": payload.get("frame_count"),
                    "encoder_backend": payload.get("encoder_backend"),
                    "video_size_bytes": video.stat().st_size if video.is_file() else None,
                }
            )
        except (OSError, ValueError):
            continue
    records.sort(key=lambda item: item.get("started_utc") or "", reverse=True)
    return records[: max(1, min(int(limit), 200))]


def create_app(config: dict, incident_dir: Path, state: DashboardState, clients):
    try:
        from aiohttp import web
    except ImportError as exc:
        raise RuntimeError("Dashboard requires aiohttp; install requirements-raspberry.txt") from exc

    static_dir = Path(__file__).resolve().parent / "static"

    async def index(_request):
        return web.FileResponse(static_dir / "index.html")

    async def healthz(_request):
        status = state.status_snapshot()
        updated = float(status.get("updated_monotonic_sec") or 0.0)
        stale = bool(updated and time.monotonic() - updated > 10.0)
        runtime_status = (status.get("runtime") or {}).get("status", "STARTING")
        healthy = runtime_status not in {"FAILED", "STOPPED"} and not stale
        return web.json_response(
            {"healthy": healthy, "runtime_status": runtime_status, "stale": stale},
            status=200 if healthy else 503,
        )

    async def api_status(_request):
        return web.json_response(state.status_snapshot())

    async def api_incidents(request):
        try:
            limit = int(request.query.get("limit", 50))
        except (TypeError, ValueError):
            limit = 50
        return web.json_response(
            {
                "schema_version": "incident-list-v1",
                "incidents": _read_incidents(incident_dir, limit),
            }
        )

    async def stream(request):
        with clients.get_lock():
            if clients.value >= int(config["max_clients"]):
                raise web.HTTPServiceUnavailable(text="Dashboard preview client limit reached")
            clients.value += 1
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "multipart/x-mixed-replace; boundary=frame",
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Pragma": "no-cache",
                "X-Content-Type-Options": "nosniff",
            },
        )
        await response.prepare(request)
        last_frame_id = None
        try:
            while True:
                preview = state.preview_snapshot()
                if preview is None or preview[0] == last_frame_id:
                    await asyncio.sleep(0.02)
                    continue
                frame_id, _monotonic, _utc, jpeg = preview
                last_frame_id = frame_id
                await response.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpeg)).encode("ascii")
                    + b"\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
        except (ConnectionResetError, asyncio.CancelledError, RuntimeError):
            pass
        finally:
            with clients.get_lock():
                clients.value = max(0, clients.value - 1)
        return response

    async def security_headers(_request, response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; script-src 'self'; "
            "style-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'"
        )
    app = web.Application(client_max_size=256 * 1024)
    app.on_response_prepare.append(security_headers)
    app.router.add_get("/", index)
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/api/v1/status", api_status)
    app.router.add_get("/api/v1/incidents", api_incidents)
    app.router.add_get("/stream.mjpg", stream)
    app.router.add_static("/static", static_dir, show_index=False)
    return app


def run_dashboard_server(config, incident_dir, status_queue, preview_queue, clients) -> None:
    from aiohttp import web

    state = DashboardState(status_queue, preview_queue)
    state.start()
    app = create_app(config, incident_dir, state, clients)
    web.run_app(
        app,
        host=str(config["host"]),
        port=int(config["port"]),
        print=None,
        access_log=None,
        handle_signals=False,
    )
