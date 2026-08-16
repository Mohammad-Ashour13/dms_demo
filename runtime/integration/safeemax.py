from __future__ import annotations

import http.client
import json
import os
import queue
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from dms_final_system.shared.contracts import IncidentRecord


_SEVERITY_BY_ALARM = {
    "FATIGUE_WARNING": "medium",
    "DROWSY": "high",
    "CRITICAL": "critical",
    "PHONE_USE": "high",
    "SMOKING": "high",
    "EATING": "medium",
    "SEATBELT_MISSING": "high",
}
_CONFIDENCE_FLOOR = {"critical": 95.0, "high": 85.0, "medium": 70.0, "low": 50.0}
_MESSAGE_TYPES = {"event", "status", "incident", "telemetry"}


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
        separators=(",", ":"),
    ).encode("utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PermanentDeliveryError(RuntimeError):
    """The server rejected a message that retrying cannot repair."""


@dataclass(frozen=True, slots=True)
class DeliveryResult:
    status: int
    duplicate: bool = False


class SafeemaxDeviceClient:
    """Durable, non-blocking sender for the one-URL device-data API.

    Every accepted message is atomically persisted before it enters the sender
    queue. Incident attachments are hard-linked into the API outbox when
    possible, or copied when the source lives on another filesystem. This keeps
    retries valid even if the recorder later removes its original incident.
    """

    MANIFEST_NAME = "manifest.json"

    def __init__(
        self,
        endpoint_url: str,
        outbox_dir: Path,
        telemetry,
        *,
        auth_token: str = "",
        request_timeout_sec: float = 5.0,
        retry_initial_sec: float = 1.0,
        retry_max_sec: float = 30.0,
        queue_size: int = 256,
    ):
        self.endpoint_url = str(endpoint_url).strip()
        parsed = urlsplit(self.endpoint_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Device API endpoint_url must be an absolute HTTP(S) URL")
        self.outbox_dir = Path(outbox_dir)
        self.outbox_dir.mkdir(parents=True, exist_ok=True)
        self.rejected_dir = self.outbox_dir / "rejected"
        self.auth_token = str(auth_token).strip()
        self.telemetry = telemetry
        self.request_timeout_sec = float(request_timeout_sec)
        self.retry_initial_sec = float(retry_initial_sec)
        self.retry_max_sec = float(retry_max_sec)
        self.queue: queue.Queue[Path | None] = queue.Queue(maxsize=max(1, int(queue_size)))
        self.stop_event = threading.Event()
        self.delivered = 0
        self.duplicates = 0
        self.retries = 0
        self.rejected = 0
        self.queue_overflow = 0
        self.last_error = ""
        self.last_attempt_utc: str | None = None
        self.last_success_utc: str | None = None
        self.last_failure_utc: str | None = None
        self.last_http_status: int | None = None
        self.last_message_id: str | None = None
        self.last_message_type: str | None = None
        self.last_failure_permanent = False
        self._attempts: dict[Path, int] = {}
        self._recover_partials()
        self._load_pending()
        self.worker = threading.Thread(
            target=self._run, name="safeemax-device-sender", daemon=True
        )
        self.worker.start()

    @staticmethod
    def _validate_message(message: dict[str, Any]) -> None:
        required = {"id", "type", "deviceId", "sentAt", "data"}
        missing = sorted(required - message.keys())
        if missing:
            raise ValueError(f"Device API message is missing fields: {missing}")
        if not str(message["id"]).strip():
            raise ValueError("Device API message id is required")
        if str(message["type"]) not in _MESSAGE_TYPES:
            raise ValueError("Device API message type is invalid")
        if not str(message["deviceId"]).strip():
            raise ValueError("Device API deviceId is required")
        if not str(message["sentAt"]).strip():
            raise ValueError("Device API sentAt is required")
        if not isinstance(message["data"], dict):
            raise ValueError("Device API data must be an object")
        if message["type"] == "event":
            data = message["data"]
            event_required = {
                "modelVersion", "alarmType", "vehicle", "confidence", "severity"
            }
            event_missing = sorted(event_required - data.keys())
            if event_missing:
                raise ValueError(f"Device API event is missing fields: {event_missing}")
            confidence = float(data["confidence"])
            if not 0.0 <= confidence <= 100.0:
                raise ValueError("Device API event confidence must be between 0 and 100")
            if data["severity"] not in {"critical", "high", "medium", "low"}:
                raise ValueError("Device API event severity is invalid")

    def _bundle_path(self, message_id: str) -> Path:
        name = uuid.uuid5(uuid.NAMESPACE_URL, str(message_id)).hex
        return self.outbox_dir / name

    def _recover_partials(self) -> None:
        self.rejected_dir.mkdir(parents=True, exist_ok=True)
        for partial in sorted(self.outbox_dir.glob(".*.partial-*")):
            manifest_path = partial / self.MANIFEST_NAME
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                message = manifest["message"]
                self._validate_message(message)
                target = self._bundle_path(message["id"])
                attachments = manifest.get("attachments") or {}
                complete = all((partial / value["path"]).is_file() for value in attachments.values())
                if not complete:
                    raise ValueError("partial bundle has missing attachments")
                if target.exists():
                    raise ValueError("a complete bundle already exists")
                partial.replace(target)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                target = self.rejected_dir / partial.name.lstrip(".")
                if not target.exists():
                    partial.replace(target)

    def _load_pending(self) -> None:
        for path in sorted(self.outbox_dir.iterdir()):
            if path == self.rejected_dir or path.name.startswith("."):
                continue
            if not path.is_dir() or not (path / self.MANIFEST_NAME).is_file():
                continue
            try:
                self.queue.put_nowait(path)
            except queue.Full:
                self.queue_overflow += 1
                break

    @staticmethod
    def _link_or_copy(source: Path, target: Path) -> None:
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)

    def _persist_bundle(
        self,
        message: dict[str, Any],
        attachments: dict[str, tuple[Path, str, str]],
    ) -> Path:
        message_id = str(message["id"])
        target = self._bundle_path(message_id)
        if target.exists():
            existing = json.loads(
                (target / self.MANIFEST_NAME).read_text(encoding="utf-8")
            )
            if existing.get("message") != message:
                raise ValueError(f"Device API message id {message_id!r} was reused")
            return target

        temporary = self.outbox_dir / f".{target.name}.partial-{uuid.uuid4().hex}"
        temporary.mkdir(parents=False)
        attachment_manifest: dict[str, dict[str, str]] = {}
        try:
            for field_name, (source, filename, content_type) in attachments.items():
                source = Path(source)
                if not source.is_file():
                    raise ValueError(f"Device API attachment does not exist: {source}")
                stored_name = f"{field_name}-{Path(filename).name}"
                self._link_or_copy(source, temporary / stored_name)
                attachment_manifest[field_name] = {
                    "path": stored_name,
                    "filename": Path(filename).name,
                    "content_type": content_type,
                }
            manifest_path = temporary / self.MANIFEST_NAME
            manifest_path.write_bytes(
                _json_bytes({"message": message, "attachments": attachment_manifest})
            )
            with manifest_path.open("rb") as handle:
                os.fsync(handle.fileno())
            for item in attachment_manifest.values():
                with (temporary / item["path"]).open("rb") as handle:
                    os.fsync(handle.fileno())
            temporary.replace(target)
            try:
                directory_fd = os.open(self.outbox_dir, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass
            return target
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def publish(
        self,
        message: dict[str, Any],
        *,
        attachments: dict[str, tuple[Path, str, str]] | None = None,
    ) -> bool:
        self._validate_message(message)
        bundle = self._persist_bundle(message, attachments or {})
        try:
            self.queue.put_nowait(bundle)
        except queue.Full:
            self.queue_overflow += 1
            self.telemetry.emit(
                "SafeemaxAPI",
                "queue_full",
                {"message_id": message["id"], "message_type": message["type"]},
                level="WARNING",
            )
            return False
        self.telemetry.emit(
            "SafeemaxAPI",
            "message_queued",
            {"message_id": message["id"], "message_type": message["type"]},
        )
        return True

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    @staticmethod
    def _delivery_result(status: int, body: str) -> DeliveryResult:
        if status not in {200, 201}:
            if 400 <= status < 500 and status not in {408, 425, 429}:
                raise PermanentDeliveryError(f"HTTP {status}: {body[:500]}")
            raise RuntimeError(f"HTTP {status}: {body[:500]}")
        try:
            decoded = json.loads(body) if body else {}
        except json.JSONDecodeError:
            decoded = {}
        return DeliveryResult(status=status, duplicate=bool(decoded.get("duplicate", False)))

    def _post_json(self, message: dict[str, Any]) -> DeliveryResult:
        headers = {**self._headers(), "Content-Type": "application/json"}
        request = Request(
            self.endpoint_url,
            data=_json_bytes(message),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.request_timeout_sec) as response:
                return self._delivery_result(
                    int(response.status), response.read().decode("utf-8", errors="replace")
                )
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return self._delivery_result(int(exc.code), body)
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(str(exc)) from exc

    @staticmethod
    def _multipart_part_header(
        boundary: str,
        field_name: str,
        *,
        filename: str | None = None,
        content_type: str,
    ) -> bytes:
        disposition = f'Content-Disposition: form-data; name="{field_name}"'
        if filename is not None:
            safe_filename = Path(filename).name.replace('"', "")
            disposition += f'; filename="{safe_filename}"'
        return (
            f"--{boundary}\r\n{disposition}\r\n"
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")

    def _post_multipart(
        self,
        message: dict[str, Any],
        bundle: Path,
        attachments: dict[str, dict[str, str]],
    ) -> DeliveryResult:
        boundary = f"safee-{uuid.uuid4().hex}"
        message_bytes = _json_bytes(message)
        parts: list[tuple[bytes, Path | None, bytes | None]] = [
            (
                self._multipart_part_header(
                    boundary, "message", content_type="application/json"
                ),
                None,
                message_bytes,
            )
        ]
        for field_name, item in attachments.items():
            path = bundle / item["path"]
            parts.append(
                (
                    self._multipart_part_header(
                        boundary,
                        field_name,
                        filename=item["filename"],
                        content_type=item["content_type"],
                    ),
                    path,
                    None,
                )
            )
        closing = f"--{boundary}--\r\n".encode("ascii")
        content_length = len(closing)
        for header, path, inline in parts:
            content_length += len(header) + 2
            content_length += len(inline) if inline is not None else path.stat().st_size

        parsed = urlsplit(self.endpoint_url)
        connection_class = (
            http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        )
        connection = connection_class(
            parsed.hostname,
            parsed.port,
            timeout=self.request_timeout_sec,
        )
        request_path = parsed.path or "/"
        if parsed.query:
            request_path += f"?{parsed.query}"
        try:
            connection.putrequest("POST", request_path)
            for name, value in self._headers().items():
                connection.putheader(name, value)
            connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
            connection.putheader("Content-Length", str(content_length))
            connection.endheaders()
            for header, path, inline in parts:
                connection.send(header)
                if inline is not None:
                    connection.send(inline)
                else:
                    with path.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            connection.send(chunk)
                connection.send(b"\r\n")
            connection.send(closing)
            response = connection.getresponse()
            body = response.read().decode("utf-8", errors="replace")
            return self._delivery_result(int(response.status), body)
        except (OSError, TimeoutError, http.client.HTTPException) as exc:
            raise RuntimeError(str(exc)) from exc
        finally:
            connection.close()

    def _post_bundle(self, bundle: Path) -> tuple[DeliveryResult, dict[str, Any]]:
        manifest = json.loads(
            (bundle / self.MANIFEST_NAME).read_text(encoding="utf-8")
        )
        message = manifest["message"]
        self._validate_message(message)
        attachments = manifest.get("attachments") or {}
        if attachments:
            result = self._post_multipart(message, bundle, attachments)
        else:
            result = self._post_json(message)
        return result, message

    def _reject(self, bundle: Path, message: dict[str, Any], error: Exception) -> None:
        self.rejected_dir.mkdir(parents=True, exist_ok=True)
        target = self.rejected_dir / bundle.name
        if target.exists():
            target = self.rejected_dir / f"{bundle.name}-{uuid.uuid4().hex[:8]}"
        bundle.replace(target)
        self._attempts.pop(bundle, None)
        self.rejected += 1
        self.last_error = str(error)
        self.last_failure_utc = _utc_now()
        self.last_http_status = None
        self.last_failure_permanent = True
        self.telemetry.emit(
            "SafeemaxAPI",
            "message_rejected",
            {
                "message_id": message.get("id"),
                "message_type": message.get("type"),
                "error": str(error),
                "path": str(target),
            },
            level="ERROR",
        )

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                bundle = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if bundle is None:
                self.queue.task_done()
                return
            message: dict[str, Any] = {}
            try:
                if not bundle.is_dir():
                    continue
                self.last_attempt_utc = _utc_now()
                result, message = self._post_bundle(bundle)
                shutil.rmtree(bundle)
                self._attempts.pop(bundle, None)
                self.delivered += 1
                if result.duplicate:
                    self.duplicates += 1
                self.last_error = ""
                self.last_success_utc = _utc_now()
                self.last_http_status = result.status
                self.last_message_id = str(message["id"])
                self.last_message_type = str(message["type"])
                self.last_failure_permanent = False
                self.telemetry.emit(
                    "SafeemaxAPI",
                    "message_delivered",
                    {
                        "message_id": message["id"],
                        "message_type": message["type"],
                        "http_status": result.status,
                        "duplicate": result.duplicate,
                    },
                )
            except (PermanentDeliveryError, json.JSONDecodeError, ValueError, KeyError) as exc:
                self._reject(bundle, message, exc)
            except Exception as exc:
                attempt = self._attempts.get(bundle, 0) + 1
                self._attempts[bundle] = attempt
                self.retries += 1
                self.last_error = str(exc)
                self.last_failure_utc = _utc_now()
                self.last_http_status = None
                self.last_failure_permanent = False
                delay = min(
                    self.retry_max_sec,
                    self.retry_initial_sec * (2 ** min(attempt - 1, 20)),
                )
                self.telemetry.emit(
                    "SafeemaxAPI",
                    "message_retry",
                    {
                        "message_id": message.get("id"),
                        "message_type": message.get("type"),
                        "attempt": attempt,
                        "retry_in_sec": delay,
                        "error": str(exc),
                    },
                    level="WARNING",
                )
                if not self.stop_event.wait(delay):
                    try:
                        self.queue.put_nowait(bundle)
                    except queue.Full:
                        self.queue_overflow += 1
            finally:
                self.queue.task_done()

    def snapshot(self) -> dict[str, Any]:
        pending = sum(
            1
            for path in self.outbox_dir.iterdir()
            if path.is_dir() and path != self.rejected_dir and not path.name.startswith(".")
        )
        if self.last_error and self.last_failure_permanent:
            connection_status = "REJECTED"
        elif self.last_error:
            connection_status = "RETRYING"
        elif self.last_success_utc:
            connection_status = "ONLINE"
        else:
            connection_status = "CONNECTING"
        return {
            "enabled": True,
            "connection_status": connection_status,
            "endpoint_url": self.endpoint_url,
            "pending": pending,
            "delivered": self.delivered,
            "duplicates": self.duplicates,
            "retries": self.retries,
            "rejected": self.rejected,
            "queue_overflow": self.queue_overflow,
            "status_reason": self.last_error,
            "last_error": self.last_error,
            "last_attempt_utc": self.last_attempt_utc,
            "last_success_utc": self.last_success_utc,
            "last_failure_utc": self.last_failure_utc,
            "last_http_status": self.last_http_status,
            "last_message_id": self.last_message_id,
            "last_message_type": self.last_message_type,
        }

    def close(self) -> None:
        self.stop_event.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        self.worker.join(timeout=5.0)


class SafeemaxDevicePublisher:
    """Build the four documented message types for one device API URL."""

    def __init__(
        self,
        client: SafeemaxDeviceClient,
        *,
        device_id: str,
        vehicle: str,
        driver: str | None = None,
        location: str = "",
        battery: int | None = None,
        event_states: list[str] | tuple[str, ...] = (),
        event_violations: list[str] | tuple[str, ...] = (),
        status_interval_sec: float = 5.0,
    ):
        self.client = client
        self.device_id = str(device_id)
        self.vehicle = str(vehicle)
        self.driver = driver or None
        self.location = str(location)
        self.battery = battery
        self.event_states = set(event_states)
        self.event_violations = set(event_violations)
        self.status_interval_sec = max(1.0, float(status_interval_sec))
        self.active_alarm_types: set[str] = set()
        self.last_status_monotonic = float("-inf")

    def _message(
        self,
        message_type: str,
        data: dict[str, Any],
        *,
        sent_at: str | None = None,
        message_id: str | None = None,
    ) -> dict[str, Any]:
        return {
            "id": message_id or f"{self.device_id}-{message_type}-{uuid.uuid4().hex}",
            "type": message_type,
            "deviceId": self.device_id,
            "sentAt": sent_at or _utc_now(),
            "data": data,
        }

    def publish_decision(self, packet, decision, model_version: str) -> list[str]:
        state = str(decision.driver_state.value)
        current: set[str] = set()
        if state in self.event_states:
            current.add(state)
        current.update(set(map(str, decision.violations)) & self.event_violations)
        newly_active = sorted(current - self.active_alarm_types)
        self.active_alarm_types.intersection_update(current)
        sent: list[str] = []
        for alarm_type in newly_active:
            severity = _SEVERITY_BY_ALARM.get(alarm_type, "medium")
            probability = decision.smoothed_probability
            confidence = _CONFIDENCE_FLOOR[severity]
            if probability is not None:
                confidence = max(
                    confidence,
                    min(100.0, max(0.0, float(probability) * 100.0)),
                )
            data: dict[str, Any] = {
                "modelVersion": str(model_version),
                "alarmType": alarm_type,
                "vehicle": self.vehicle,
                "driver": self.driver,
                "confidence": round(confidence, 2),
                "severity": severity,
            }
            if self.location:
                data["location"] = self.location
            if self.battery is not None:
                data["battery"] = int(self.battery)
            message = self._message(
                "event",
                data,
                sent_at=str(packet.utc_timestamp),
            )
            self.client.publish(message)
            self.active_alarm_types.add(alarm_type)
            sent.append(alarm_type)
        return sent

    def publish_status(self, status: dict[str, Any], *, monotonic_sec: float) -> bool:
        if monotonic_sec - self.last_status_monotonic < self.status_interval_sec:
            return False
        self.last_status_monotonic = float(monotonic_sec)
        self.client.publish(self._message("status", status))
        return True

    def publish_telemetry(self, records: list[dict[str, Any]]) -> bool:
        if not records:
            return False
        self.client.publish(self._message("telemetry", {"records": records}))
        return True

    def publish_incident(self, incident: IncidentRecord) -> bool:
        incident_path = Path(incident.video_path).parent / "incident.json"
        metadata = json.loads(incident_path.read_text(encoding="utf-8"))
        message = self._message(
            "incident",
            metadata,
            sent_at=_utc_now(),
            message_id=incident.incident_id,
        )
        return self.client.publish(
            message,
            attachments={
                "video": (Path(incident.video_path), "video.mp4", "video/mp4"),
                "telemetry": (
                    Path(incident.telemetry_path),
                    "telemetry.jsonl",
                    "application/x-ndjson",
                ),
            },
        )


class SafeemaxIncidentSink:
    def __init__(self, publisher: SafeemaxDevicePublisher):
        self.publisher = publisher

    def publish(self, incident: IncidentRecord) -> None:
        self.publisher.publish_incident(incident)


class SafeemaxTelemetryBatcher:
    """Collect structured telemetry without blocking telemetry producers."""

    def __init__(
        self,
        telemetry,
        publisher: SafeemaxDevicePublisher,
        *,
        batch_size: int = 250,
        flush_interval_sec: float = 15.0,
        queue_size: int = 2000,
    ):
        self.telemetry = telemetry
        self.publisher = publisher
        self.batch_size = max(1, min(int(batch_size), 500))
        self.flush_interval_sec = max(1.0, float(flush_interval_sec))
        self.queue: queue.Queue[dict[str, Any] | None] = queue.Queue(
            maxsize=max(self.batch_size, int(queue_size))
        )
        self.dropped = 0
        self.worker = threading.Thread(
            target=self._run, name="safeemax-telemetry-batcher", daemon=True
        )
        self.telemetry.add_subscriber(self.push)
        self.worker.start()

    def push(self, record: dict[str, Any]) -> None:
        if record.get("stage") == "SafeemaxAPI":
            return
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1

    def _flush(self, records: list[dict[str, Any]]) -> None:
        try:
            self.publisher.publish_telemetry(records)
        except Exception as exc:
            self.telemetry.emit(
                "SafeemaxAPI",
                "telemetry_batch_queue_failed",
                {"records": len(records), "error": repr(exc)},
                level="ERROR",
            )

    def _run(self) -> None:
        records: list[dict[str, Any]] = []
        deadline = time.monotonic() + self.flush_interval_sec
        while True:
            timeout = max(0.05, deadline - time.monotonic()) if records else 0.5
            try:
                item = self.queue.get(timeout=timeout)
            except queue.Empty:
                if records and time.monotonic() >= deadline:
                    self._flush(records)
                    records = []
                deadline = time.monotonic() + self.flush_interval_sec
                continue
            if item is None:
                self.queue.task_done()
                if records:
                    self._flush(records)
                return
            records.append(item)
            self.queue.task_done()
            if len(records) >= self.batch_size:
                self._flush(records)
                records = []
                deadline = time.monotonic() + self.flush_interval_sec

    def close(self) -> None:
        self.telemetry.remove_subscriber(self.push)
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            # Make room for the stop marker; already-queued records remain in
            # the local structured telemetry log even if this batch is dropped.
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except queue.Empty:
                pass
            self.queue.put_nowait(None)
        self.worker.join(timeout=max(5.0, self.flush_interval_sec + 1.0))


# Backward-compatible import names for downstream code while the integration
# moves from the former event-only API to the one-URL device API.
SafeemaxEventClient = SafeemaxDeviceClient
SafeemaxEventPublisher = SafeemaxDevicePublisher
