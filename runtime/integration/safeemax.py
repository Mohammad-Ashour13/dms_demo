from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


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


class PermanentDeliveryError(RuntimeError):
    """The server rejected a payload that retrying cannot repair."""


@dataclass(frozen=True, slots=True)
class DeliveryResult:
    status: int
    duplicate: bool = False


class SafeemaxEventClient:
    """Durable background sender for POST /api/events.

    A payload is atomically stored before it enters the in-memory queue. Network
    failures therefore never block the AI loop and do not lose accepted events
    when the process or device restarts.
    """

    def __init__(
        self,
        base_url: str,
        outbox_dir: Path,
        telemetry,
        *,
        request_timeout_sec: float = 5.0,
        retry_initial_sec: float = 1.0,
        retry_max_sec: float = 30.0,
        queue_size: int = 256,
    ):
        self.events_url = f"{base_url.rstrip('/')}/api/events"
        self.outbox_dir = Path(outbox_dir)
        self.outbox_dir.mkdir(parents=True, exist_ok=True)
        self.rejected_dir = self.outbox_dir / "rejected"
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
        self._attempts: dict[Path, int] = {}
        self._load_pending()
        self.worker = threading.Thread(
            target=self._run, name="safeemax-event-sender", daemon=True
        )
        self.worker.start()

    def _load_pending(self) -> None:
        for path in sorted(self.outbox_dir.glob("*.json")):
            try:
                self.queue.put_nowait(path)
            except queue.Full:
                self.queue_overflow += 1
                break

    @staticmethod
    def _validate(payload: dict[str, Any]) -> None:
        required = {
            "eventId",
            "deviceId",
            "modelVersion",
            "alarmType",
            "vehicle",
            "confidence",
            "occurredAt",
            "severity",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"Safeemax event is missing fields: {missing}")
        confidence = float(payload["confidence"])
        if not 0.0 <= confidence <= 100.0:
            raise ValueError("Safeemax event confidence must be between 0 and 100")
        if payload["severity"] not in {"critical", "high", "medium", "low"}:
            raise ValueError("Safeemax event severity is invalid")

    def publish(self, payload: dict[str, Any]) -> bool:
        self._validate(payload)
        event_id = str(payload["eventId"])
        filename = f"{uuid.uuid5(uuid.NAMESPACE_URL, event_id).hex}.json"
        path = self.outbox_dir / filename
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
        try:
            directory_fd = os.open(self.outbox_dir, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Atomic rename still helps on filesystems without directory fsync.
            pass
        try:
            self.queue.put_nowait(path)
        except queue.Full:
            # The durable file remains for the next process start.
            self.queue_overflow += 1
            self.telemetry.emit(
                "SafeemaxAPI",
                "queue_full",
                {"event_id": event_id, "path": str(path)},
                level="WARNING",
            )
            return False
        self.telemetry.emit(
            "SafeemaxAPI", "event_queued", {"event_id": event_id, "alarm_type": payload["alarmType"]}
        )
        return True

    def _post(self, payload: dict[str, Any]) -> DeliveryResult:
        request = Request(
            self.events_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.request_timeout_sec) as response:
                status = int(response.status)
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if 400 <= exc.code < 500 and exc.code not in {408, 425, 429}:
                raise PermanentDeliveryError(f"HTTP {exc.code}: {body[:500]}") from exc
            raise RuntimeError(f"HTTP {exc.code}: {body[:500]}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(str(exc)) from exc
        if status not in {200, 201}:
            raise RuntimeError(f"Unexpected HTTP {status}: {body[:500]}")
        try:
            decoded = json.loads(body) if body else {}
        except json.JSONDecodeError:
            decoded = {}
        return DeliveryResult(status=status, duplicate=bool(decoded.get("duplicate", False)))

    def _reject(self, path: Path, payload: dict[str, Any], error: Exception) -> None:
        self.rejected_dir.mkdir(parents=True, exist_ok=True)
        target = self.rejected_dir / path.name
        path.replace(target)
        self._attempts.pop(path, None)
        self.rejected += 1
        self.last_error = str(error)
        self.telemetry.emit(
            "SafeemaxAPI",
            "event_rejected",
            {"event_id": payload.get("eventId"), "error": str(error), "path": str(target)},
            level="ERROR",
        )

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                path = self.queue.get(timeout=0.5)
            except queue.Empty:
                pending = next(iter(sorted(self.outbox_dir.glob("*.json"))), None)
                if pending is not None:
                    try:
                        self.queue.put_nowait(pending)
                    except queue.Full:
                        pass
                continue
            if path is None:
                self.queue.task_done()
                return
            payload: dict[str, Any] = {}
            try:
                if not path.is_file():
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                result = self._post(payload)
                path.unlink(missing_ok=True)
                self._attempts.pop(path, None)
                self.delivered += 1
                if result.duplicate:
                    self.duplicates += 1
                self.last_error = ""
                self.telemetry.emit(
                    "SafeemaxAPI",
                    "event_delivered",
                    {
                        "event_id": payload["eventId"],
                        "http_status": result.status,
                        "duplicate": result.duplicate,
                    },
                )
            except (PermanentDeliveryError, json.JSONDecodeError, ValueError) as exc:
                self._reject(path, payload, exc)
            except Exception as exc:
                attempt = self._attempts.get(path, 0) + 1
                self._attempts[path] = attempt
                self.retries += 1
                self.last_error = str(exc)
                delay = min(
                    self.retry_max_sec,
                    self.retry_initial_sec * (2 ** min(attempt - 1, 20)),
                )
                self.telemetry.emit(
                    "SafeemaxAPI",
                    "event_retry",
                    {
                        "path": str(path),
                        "attempt": attempt,
                        "retry_in_sec": delay,
                        "error": str(exc),
                    },
                    level="WARNING",
                )
                if not self.stop_event.wait(delay):
                    try:
                        self.queue.put_nowait(path)
                    except queue.Full:
                        self.queue_overflow += 1
            finally:
                self.queue.task_done()

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "pending": len(list(self.outbox_dir.glob("*.json"))),
            "delivered": self.delivered,
            "duplicates": self.duplicates,
            "retries": self.retries,
            "rejected": self.rejected,
            "queue_overflow": self.queue_overflow,
            "last_error": self.last_error,
        }

    def close(self) -> None:
        self.stop_event.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        self.worker.join(timeout=5.0)


class SafeemaxEventPublisher:
    """Turn newly-active fused risks into the API's documented event schema."""

    def __init__(
        self,
        client: SafeemaxEventClient,
        *,
        device_id: str,
        vehicle: str,
        driver: str | None = None,
        location: str = "",
        battery: int | None = None,
        event_states: list[str] | tuple[str, ...] = (),
        event_violations: list[str] | tuple[str, ...] = (),
    ):
        self.client = client
        self.device_id = str(device_id)
        self.vehicle = str(vehicle)
        self.driver = driver or None
        self.location = str(location)
        self.battery = battery
        self.event_states = set(event_states)
        self.event_violations = set(event_violations)
        self.active_alarm_types: set[str] = set()

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
                confidence = max(confidence, min(100.0, max(0.0, float(probability) * 100.0)))
            payload: dict[str, Any] = {
                "eventId": f"{self.device_id}-{uuid.uuid4().hex}",
                "deviceId": self.device_id,
                "modelVersion": str(model_version),
                "alarmType": alarm_type,
                "vehicle": self.vehicle,
                "driver": self.driver,
                "confidence": round(confidence, 2),
                "occurredAt": str(packet.utc_timestamp),
                "severity": severity,
            }
            if self.location:
                payload["location"] = self.location
            if self.battery is not None:
                payload["battery"] = int(self.battery)
            self.client.publish(payload)
            self.active_alarm_types.add(alarm_type)
            sent.append(alarm_type)
        return sent
