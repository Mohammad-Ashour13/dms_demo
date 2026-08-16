from __future__ import annotations

import json
import threading
import time
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from dms_final_system.runtime.config import load_config
from dms_final_system.runtime.integration.safeemax import (
    DeliveryResult,
    SafeemaxDeviceClient,
    SafeemaxDevicePublisher,
)
from dms_final_system.shared.contracts import DriverState, IncidentRecord


class FakeTelemetry:
    def __init__(self):
        self.records = []
        self.subscribers = []

    def emit(self, stage, event, payload=None, **kwargs):
        record = (stage, event, payload or {}, kwargs)
        self.records.append(record)

    def add_subscriber(self, subscriber):
        self.subscribers.append(subscriber)

    def remove_subscriber(self, subscriber):
        self.subscribers.remove(subscriber)


class FakeClient:
    def __init__(self):
        self.messages = []

    def publish(self, message, **kwargs):
        self.messages.append((message, kwargs))
        return True


def _decision(state, violations=(), probability=0.8):
    return SimpleNamespace(
        driver_state=DriverState(state),
        violations=list(violations),
        smoothed_probability=probability,
    )


def _event_message(message_id="event-1"):
    return {
        "id": message_id,
        "type": "event",
        "deviceId": "device-1",
        "sentAt": "2026-08-16T12:00:00+00:00",
        "data": {
            "modelVersion": "model-1",
            "alarmType": "DROWSY",
            "vehicle": "vehicle-1",
            "confidence": 90,
            "severity": "high",
        },
    }


def _pending_bundles(path: Path) -> list[Path]:
    return [
        item
        for item in path.iterdir()
        if item.is_dir() and item.name != "rejected" and not item.name.startswith(".")
    ]


def test_publisher_emits_only_newly_active_alarm_types_in_one_url_envelope():
    client = FakeClient()
    publisher = SafeemaxDevicePublisher(
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

    messages = [message for message, _ in client.messages]
    assert len(messages) == 3
    first = messages[0]
    assert first["type"] == "event"
    assert first["deviceId"] == "device-1"
    assert first["sentAt"] == packet.utc_timestamp
    assert first["data"]["vehicle"] == "vehicle-7"
    assert first["data"]["alarmType"] == "DROWSY"
    assert first["data"]["severity"] == "high"
    assert first["data"]["confidence"] == 91.0
    assert len({item["id"] for item in messages}) == 3


def test_status_is_throttled_to_configured_interval():
    client = FakeClient()
    publisher = SafeemaxDevicePublisher(
        client,
        device_id="device-1",
        vehicle="vehicle-1",
        status_interval_sec=5.0,
    )
    status = {"schema_version": "dashboard-status-v1", "runtime": {"status": "RUNNING"}}
    assert publisher.publish_status(status, monotonic_sec=10.0)
    assert not publisher.publish_status(status, monotonic_sec=14.9)
    assert publisher.publish_status(status, monotonic_sec=15.0)
    assert [item[0]["type"] for item in client.messages] == ["status", "status"]


def test_client_persists_then_removes_acknowledged_message(tmp_path, monkeypatch):
    telemetry = FakeTelemetry()
    delivered = []

    def fake_post_bundle(self, bundle):
        message = json.loads((bundle / self.MANIFEST_NAME).read_text())["message"]
        delivered.append(message)
        if len(delivered) == 1:
            raise RuntimeError("temporary network failure")
        return DeliveryResult(200, True), message

    monkeypatch.setattr(SafeemaxDeviceClient, "_post_bundle", fake_post_bundle)
    client = SafeemaxDeviceClient(
        "http://api.example/api/device-data",
        tmp_path / "outbox",
        telemetry,
        retry_initial_sec=0.01,
        retry_max_sec=0.02,
    )
    payload = _event_message()
    try:
        assert client.publish(payload)
        deadline = time.monotonic() + 2.0
        while (len(delivered) < 2 or _pending_bundles(tmp_path / "outbox")) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert delivered == [payload, payload]
        assert not _pending_bundles(tmp_path / "outbox")
        assert client.retries == 1
        assert client.duplicates == 1
        assert any(event == "message_retry" for _, event, _, _ in telemetry.records)
        assert any(event == "message_delivered" for _, event, _, _ in telemetry.records)
    finally:
        client.close()


def test_client_posts_documented_json_contract_to_single_path(tmp_path):
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
            body = json.dumps({"ok": True, "duplicate": False}).encode()
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
    client = SafeemaxDeviceClient(
        f"http://127.0.0.1:{server.server_port}/api/device-data",
        tmp_path / "outbox",
        telemetry,
    )
    payload = _event_message("event-http-1")
    try:
        assert client.publish(payload)
        deadline = time.monotonic() + 2.0
        while not received and time.monotonic() < deadline:
            time.sleep(0.01)
        assert received == [
            {
                "path": "/api/device-data",
                "content_type": "application/json",
                "payload": payload,
            }
        ]
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2.0)


def test_incident_uses_multipart_on_the_same_path(tmp_path):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            body = self.rfile.read(length)
            mime = BytesParser(policy=default).parsebytes(
                b"Content-Type: "
                + self.headers["Content-Type"].encode()
                + b"\r\nMIME-Version: 1.0\r\n\r\n"
                + body
            )
            parts = {}
            for part in mime.iter_parts():
                name = part.get_param("name", header="content-disposition")
                parts[name] = {
                    "filename": part.get_filename(),
                    "content_type": part.get_content_type(),
                    "body": part.get_payload(decode=True),
                }
            received.append({"path": self.path, "parts": parts})
            response = json.dumps({"ok": True, "duplicate": False}).encode()
            self.send_response(201)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, _format, *_args):
            return

    incident_dir = tmp_path / "incidents" / "incident-1"
    incident_dir.mkdir(parents=True)
    video = incident_dir / "video.mp4"
    telemetry_path = incident_dir / "telemetry.jsonl"
    video.write_bytes(b"video-bytes")
    telemetry_path.write_text('{"event":"decision"}\n', encoding="utf-8")
    metadata = {
        "incident_id": "incident-1",
        "video_sha256": "unused-by-transport-test",
        "highest_state": "DROWSY",
    }
    (incident_dir / "incident.json").write_text(json.dumps(metadata), encoding="utf-8")
    incident = IncidentRecord(
        "incident-1", "session-1", "start", "end", "DROWSY", [], 0.9, [],
        "model-1", "feature-1", "fusion-1", video, telemetry_path,
    )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    client = SafeemaxDeviceClient(
        f"http://127.0.0.1:{server.server_port}/api/device-data",
        tmp_path / "api-outbox",
        FakeTelemetry(),
    )
    publisher = SafeemaxDevicePublisher(
        client, device_id="device-1", vehicle="vehicle-1"
    )
    try:
        assert publisher.publish_incident(incident)
        deadline = time.monotonic() + 2.0
        while not received and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(received) == 1
        assert received[0]["path"] == "/api/device-data"
        parts = received[0]["parts"]
        assert set(parts) == {"message", "video", "telemetry"}
        message = json.loads(parts["message"]["body"])
        assert message["id"] == "incident-1"
        assert message["type"] == "incident"
        assert message["deviceId"] == "device-1"
        assert message["data"] == metadata
        assert parts["video"]["filename"] == "video.mp4"
        assert parts["video"]["body"] == b"video-bytes"
        assert parts["telemetry"]["filename"] == "telemetry.jsonl"
        assert parts["telemetry"]["body"] == telemetry_path.read_bytes()
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
    assert config.safeemax_api.endpoint_url == (
        "http://76.13.131.115:4000/api/device-data"
    )
    assert config.safeemax_api.status_interval_sec == 5.0
    assert config.safeemax_api.incident_upload_enabled
    assert config.safeemax_api.telemetry_upload_enabled
