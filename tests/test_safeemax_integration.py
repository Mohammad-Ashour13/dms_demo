from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from dms_final_system.runtime.config import load_config
from dms_final_system.runtime.integration.safeemax import (
    DeliveryResult,
    SafeemaxEventClient,
    SafeemaxEventPublisher,
)
from dms_final_system.shared.contracts import DriverState


class FakeTelemetry:
    def __init__(self):
        self.records = []

    def emit(self, stage, event, payload=None, **kwargs):
        self.records.append((stage, event, payload or {}, kwargs))


class FakeClient:
    def __init__(self):
        self.payloads = []

    def publish(self, payload):
        self.payloads.append(payload)
        return True


def _decision(state, violations=(), probability=0.8):
    return SimpleNamespace(
        driver_state=DriverState(state),
        violations=list(violations),
        smoothed_probability=probability,
    )


def test_publisher_emits_only_newly_active_alarm_types():
    client = FakeClient()
    publisher = SafeemaxEventPublisher(
        client,
        device_id="device-1",
        vehicle="vehicle-7",
        event_states=["FATIGUE_WARNING", "DROWSY", "CRITICAL"],
        event_violations=["PHONE_USE", "SMOKING", "EATING", "SEATBELT_MISSING"],
    )
    packet = SimpleNamespace(utc_timestamp="2026-08-16T12:00:00+00:00")

    assert publisher.publish_decision(packet, _decision("NORMAL"), "model-1") == []
    assert publisher.publish_decision(
        packet, _decision("DROWSY", ["PHONE_USE"], 0.91), "model-1"
    ) == ["DROWSY", "PHONE_USE"]
    assert publisher.publish_decision(
        packet, _decision("DROWSY", ["PHONE_USE"], 0.92), "model-1"
    ) == []
    assert publisher.publish_decision(packet, _decision("NORMAL"), "model-1") == []
    assert publisher.publish_decision(
        packet, _decision("NORMAL", ["PHONE_USE"], None), "model-1"
    ) == ["PHONE_USE"]

    assert len(client.payloads) == 3
    first = client.payloads[0]
    assert first["deviceId"] == "device-1"
    assert first["vehicle"] == "vehicle-7"
    assert first["alarmType"] == "DROWSY"
    assert first["severity"] == "high"
    assert first["confidence"] == 91.0
    assert first["occurredAt"] == packet.utc_timestamp
    assert len({item["eventId"] for item in client.payloads}) == 3


def test_client_persists_then_removes_acknowledged_event(tmp_path, monkeypatch):
    telemetry = FakeTelemetry()
    delivered = []

    def fake_post(self, payload):
        delivered.append(payload)
        if len(delivered) == 1:
            raise RuntimeError("temporary network failure")
        return DeliveryResult(200, True)

    monkeypatch.setattr(SafeemaxEventClient, "_post", fake_post)
    client = SafeemaxEventClient(
        "http://api.example",
        tmp_path / "outbox",
        telemetry,
        retry_initial_sec=0.01,
        retry_max_sec=0.02,
    )
    payload = {
        "eventId": "event-1",
        "deviceId": "device-1",
        "modelVersion": "model-1",
        "alarmType": "DROWSY",
        "vehicle": "vehicle-1",
        "confidence": 90,
        "occurredAt": "2026-08-16T12:00:00+00:00",
        "severity": "high",
    }
    try:
        assert client.publish(payload)
        deadline = time.monotonic() + 2.0
        while (
            len(delivered) < 2 or list((tmp_path / "outbox").glob("*.json"))
        ) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert delivered == [payload, payload]
        assert not list((tmp_path / "outbox").glob("*.json"))
        assert client.retries == 1
        assert client.duplicates == 1
        assert any(event == "event_retry" for _, event, _, _ in telemetry.records)
        assert any(event == "event_delivered" for _, event, _, _ in telemetry.records)
    finally:
        client.close()


def test_client_posts_documented_json_contract(tmp_path):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            received.append(
                {
                    "path": self.path,
                    "content_type": self.headers["Content-Type"],
                    "payload": json.loads(self.rfile.read(length)),
                }
            )
            body = json.dumps({"accepted": True, "duplicate": False}).encode()
            self.send_response(201)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    telemetry = FakeTelemetry()
    client = SafeemaxEventClient(
        f"http://127.0.0.1:{server.server_port}",
        tmp_path / "outbox",
        telemetry,
    )
    payload = {
        "eventId": "event-http-1",
        "deviceId": "device-1",
        "modelVersion": "model-1",
        "alarmType": "CRITICAL",
        "vehicle": "vehicle-1",
        "driver": None,
        "confidence": 99.5,
        "occurredAt": "2026-08-16T12:00:00+00:00",
        "severity": "critical",
    }
    try:
        assert client.publish(payload)
        deadline = time.monotonic() + 2.0
        while not received and time.monotonic() < deadline:
            time.sleep(0.01)
        assert received == [
            {
                "path": "/api/events",
                "content_type": "application/json",
                "payload": payload,
            }
        ]
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2.0)


def _write_config(path: Path, deployment_mode: str) -> None:
    path.write_text(
        json.dumps(
            {
                "deployment_mode": deployment_mode,
                "safeemax_api": {
                    "enabled": True,
                    "device_id": "device-1",
                    "vehicle": "vehicle-1",
                },
            }
        ),
        encoding="utf-8",
    )


def test_api_integration_requires_active_deployment(tmp_path):
    path = tmp_path / "runtime.json"
    _write_config(path, "SHADOW")
    with pytest.raises(ValueError, match="ACTIVE"):
        load_config(path)

    _write_config(path, "ACTIVE")
    config = load_config(path)
    assert config.safeemax_api.enabled
    assert config.safeemax_api.base_url == "http://76.13.131.115:4000"
