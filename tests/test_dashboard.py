from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import queue
import time

from aiohttp.test_utils import TestClient, TestServer

from dms_final_system.runtime.dashboard.server import DashboardState, create_app


def test_dashboard_read_only_health_status_and_incident_apis(tmp_path):
    incident = tmp_path / "incident-1"
    incident.mkdir()
    (incident / "video.mp4").write_bytes(b"video")
    (incident / "incident.json").write_text(
        json.dumps(
            {
                "incident_id": "incident-1",
                "started_utc": "2026-01-01T00:00:00Z",
                "highest_state": "DROWSY",
                "trigger_kinds": ["DROWSY"],
                "actual_duration_sec": 10.0,
            }
        ),
        encoding="utf-8",
    )
    state = DashboardState(queue.Queue(), queue.Queue())
    state.status = {
        "schema_version": "dashboard-status-v1",
        "updated_monotonic_sec": time.monotonic(),
        "runtime": {"status": "RUNNING"},
    }
    clients = mp.Value("i", 0)
    app = create_app(
        {"max_clients": 2},
        tmp_path,
        state,
        clients,
    )

    async def scenario():
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            health = await client.get("/healthz")
            assert health.status == 200
            assert health.headers["X-Frame-Options"] == "DENY"
            assert (await health.json())["healthy"] is True
            status = await client.get("/api/v1/status")
            assert (await status.json())["schema_version"] == "dashboard-status-v1"
            home = await client.get("/")
            home_text = await home.text()
            assert 'id="face-box"' in home_text
            assert 'id="face-status"' in home_text
            assert 'id="blink-rate"' in home_text
            assert 'id="behavior-overlays"' in home_text
            assert 'id="behavior-phone"' in home_text
            assert 'id="behavior-smoking"' in home_text
            assert 'id="behavior-eating"' in home_text
            script = await client.get("/static/app.js")
            script_text = await script.text()
            assert "face_bbox_normalized" in script_text
            assert "bbox_normalized" in script_text
            incidents = await client.get("/api/v1/incidents?limit=invalid")
            payload = await incidents.json()
            assert payload["schema_version"] == "incident-list-v1"
            assert payload["incidents"][0]["incident_id"] == "incident-1"
            assert payload["incidents"][0]["duration_sec"] == 10.0
            assert (await client.post("/api/v1/status")).status == 405
            paths = {resource.canonical for resource in app.router.resources()}
            assert "/stream.mjpg" in paths
        finally:
            await client.close()

    asyncio.run(scenario())
