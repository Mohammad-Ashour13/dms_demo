# Simple Raspberry Pi Device API

Status: backend implementation contract  
Contract version: `1.0`

The Raspberry Pi sends everything to **one public URL**:

```text
POST https://your-domain.example/api/device-data
```

You may choose a different path. After the API is public, only the final full URL
must be added to the Raspberry configuration.

## 1. One message format

Events, device status, and diagnostic telemetry use `application/json` and the
same envelope:

```json
{
  "id": "pi5-cab-01-000001",
  "type": "event",
  "deviceId": "pi5-cab-01",
  "sentAt": "2026-08-16T12:00:00.123456+00:00",
  "data": {}
}
```

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `id` | string | Yes | Unique message ID; remains unchanged during retries |
| `type` | string | Yes | `event`, `status`, `incident`, or `telemetry` |
| `deviceId` | string | Yes | ID assigned to this Raspberry Pi |
| `sentAt` | string | Yes | ISO 8601 UTC timestamp |
| `data` | object | Yes | Type-specific data described below |

The server must store `id` as a unique value. If the Raspberry retries the same
message, do not create another record or another notification.

## 2. Event message

Sent immediately when a safety state or driver violation becomes active.

```json
{
  "id": "pi5-cab-01-7d09b7b8f7b54a20a11eae570e117df3",
  "type": "event",
  "deviceId": "pi5-cab-01",
  "sentAt": "2026-08-16T12:00:00.123456+00:00",
  "data": {
    "modelVersion": "20260731T140605Z_full_runtimefix1",
    "alarmType": "DROWSY",
    "vehicle": "BUS-27",
    "driver": "driver-104",
    "confidence": 91.4,
    "severity": "high",
    "location": "Damascus",
    "battery": 88
  }
}
```

Required event data:

| Field | Type | Rules |
|---|---|---|
| `modelVersion` | string | Active drowsiness model version |
| `alarmType` | string | One of the values below |
| `vehicle` | string | Vehicle ID or name |
| `confidence` | number | `0..100` |
| `severity` | string | `critical`, `high`, `medium`, or `low` |

Optional event data: `driver` (string or null), `location` (string), and
`battery` (integer `0..100`). Location and battery are configured static values
in the current runtime; the project does not currently collect GPS or live
battery readings.

Supported alerts:

| `alarmType` | Meaning | Device severity |
|---|---|---|
| `FATIGUE_WARNING` | Early fatigue warning | `medium` |
| `DROWSY` | Drowsiness detected | `high` |
| `CRITICAL` | Critical drowsiness | `critical` |
| `PHONE_USE` | Phone use | `high` |
| `SMOKING` | Smoking | `high` |
| `EATING` | Eating or drinking | `medium` |
| `SEATBELT_MISSING` | Seat belt missing | `high` |

The device sends a condition only when it becomes newly active. If it clears and
later happens again, the device sends a new message with a new `id`.

## 3. Status message

Recommended interval: once every **5 seconds** while the runtime is running.

```json
{
  "id": "pi5-cab-01-status-00001842",
  "type": "status",
  "deviceId": "pi5-cab-01",
  "sentAt": "2026-08-16T12:00:05.000000+00:00",
  "data": {
    "schema_version": "dashboard-status-v1",
    "runtime": {
      "status": "RUNNING",
      "session_id": "raspberry-pi5-live-20260816T115800Z",
      "deployment_mode": "ACTIVE",
      "uptime_sec": 125.4
    },
    "driver": {
      "state": "DROWSY",
      "target_state": "DROWSY",
      "violations": ["PHONE_USE"],
      "probability": 0.914,
      "reason_codes": ["MODEL_HIGH"],
      "face_detected": true,
      "face_quality": 0.92,
      "eye_state": "CLOSED",
      "blink_count_60s": 18
    },
    "system": {
      "temperature_c": 61.8,
      "cpu_percent": 72.1,
      "ram_percent": 43.0,
      "disk": {
        "free_bytes": 34812989440,
        "used_percent": 42.8
      }
    },
    "power": {
      "safe_mode": false,
      "reason": "NORMAL"
    }
  }
}
```

The `data` object is the same `dashboard-status-v1` snapshot already generated
by the project. It can contain these sections:

| Section | Data sent |
|---|---|
| `runtime` | Runtime status, session, deployment mode, uptime |
| `driver` | Driver state, risk probability, violations, face/eye measurements, calibration, alarm status |
| `behavior` | Phone, smoking, and eating detections and detector health |
| `seatbelt` | Seat-belt probabilities, violations, model and detector health |
| `camera` | Camera backend, resolution, FPS, dropped frames and errors |
| `performance` | AI FPS, detector FPS, latencies and queue depths |
| `system` | CPU, RAM, temperature, frequency, disk, throttling and power measurements |
| `power` | Thermal/undervoltage safe-mode state |
| `models` | Drowsiness, behavior, seat-belt, MediaPipe, feature and fusion versions |
| `recording` | Active/finalized incident counts and recorder health |
| `dashboard` | Local dashboard health and connected clients |
| `remote_api` | Pending, delivered, duplicate, retry and rejected message counts |

Measurement fields may be `null` during startup or when the operating system
cannot read a metric. The server must accept additional fields so future runtime
versions remain compatible.

The server should mark a device offline if it receives no status message for 30
seconds.

## 4. Telemetry message

Telemetry is optional diagnostic data. Alerts must still use `type: "event"`.
Send at most 500 records in one message.

```json
{
  "id": "pi5-cab-01-telemetry-00000421",
  "type": "telemetry",
  "deviceId": "pi5-cab-01",
  "sentAt": "2026-08-16T12:00:20.000000+00:00",
  "data": {
    "records": [
      {
        "utc_timestamp": "2026-08-16T12:00:19.100000+00:00",
        "monotonic_sec": 143240.1,
        "session_id": "raspberry-pi5-live-20260816T115800Z",
        "frame_id": 3321,
        "window_id": "window-661",
        "incident_id": null,
        "stage": "Fusion",
        "event": "decision",
        "level": "INFO",
        "payload": {
          "driver_state": "DROWSY",
          "violations": ["PHONE_USE"]
        }
      }
    ]
  }
}
```

The `payload` inside each telemetry record is event-specific. The server should
store it as JSON and accept unknown fields.

## 5. Incident message using the same URL

Incidents use the same `POST` URL but use `multipart/form-data` because they can
include video. The multipart fields are:

| Form field | Required | Content |
|---|---:|---|
| `message` | Yes | JSON string containing the standard envelope |
| `video` | Yes | `video.mp4` |
| `telemetry` | Yes | `telemetry.jsonl` |

The `message` field contains:

```json
{
  "id": "incident-20260816-120000-a1b2c3",
  "type": "incident",
  "deviceId": "pi5-cab-01",
  "sentAt": "2026-08-16T12:00:08.000000+00:00",
  "data": {
    "incident_id": "incident-20260816-120000-a1b2c3",
    "session_id": "raspberry-pi5-live-20260816T115800Z",
    "started_utc": "2026-08-16T11:59:55.100000+00:00",
    "ended_utc": "2026-08-16T12:00:05.100000+00:00",
    "highest_state": "DROWSY",
    "violations": ["PHONE_USE"],
    "max_model_probability": 0.914,
    "reason_codes": ["MODEL_HIGH", "PROLONGED_EYE_CLOSURE"],
    "model_version": "20260731T140605Z_full_runtimefix1",
    "feature_version": "v3-runtime-1.0.0",
    "fusion_version": "fusion-2.4.0",
    "video_sha256": "2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae",
    "contract_version": "2.4.0",
    "frame_count": 150,
    "actual_duration_sec": 10.0,
    "trigger_kinds": ["DROWSY", "PHONE_USE"],
    "event_summary": {}
  }
}
```

The server must recalculate the uploaded video's SHA-256 checksum and compare it
with `data.video_sha256`. The normal configured maximum video duration is 10
seconds. Allow at least 25 MiB for a complete multipart request.

## 6. One response format

For every message type, respond with:

```json
{
  "ok": true,
  "duplicate": false,
  "id": "pi5-cab-01-000001"
}
```

Use:

- `201 Created` when a new message is saved.
- `200 OK` when the same `id` was already saved.
- `400 Bad Request` for invalid input.
- `401 Unauthorized` or `403 Forbidden` for authentication problems.
- `413 Payload Too Large` when an incident exceeds the configured limit.
- `500` or `503` for temporary server problems.

The Raspberry stores messages locally before sending and retries temporary
network/server failures. A duplicate request must not create duplicate database
records, alerts, or notifications.

An error response can remain simple:

```json
{
  "ok": false,
  "error": "confidence must be between 0 and 100"
}
```

## 7. Authentication

The simplest secure option is one bearer token for each Raspberry Pi:

```text
Authorization: Bearer DEVICE_TOKEN
```

The token must be connected to one `deviceId`. Do not put the token in source
control. If this API is only for a temporary local demonstration, authentication
can be disabled initially, but the public production URL must use HTTPS and
authentication.

## 8. Minimal backend logic

The endpoint only needs this routing logic:

```text
if request is multipart:
    message = parse JSON form field "message"
else:
    message = parse JSON body

validate id, type, deviceId, sentAt, data
if id already exists: return 200 with duplicate=true

save message
if type == incident: save video and telemetry files and verify checksum
if type == event: create the alert/notification

return 201 with duplicate=false
```

Use a unique database constraint on `id`. Store the complete original `data`
object as JSON so the database does not need a column for every Raspberry metric.

## 9. What to send back for integration

Only send these values after the API is public:

```text
API_URL=https://your-domain.example/api/device-data
DEVICE_ID=pi5-cab-01
VEHICLE=BUS-27
AUTH=none-or-bearer
```

If authentication is enabled, tell me how the token will be installed on the
Raspberry, but do not post the real secret in a public message.

The current project sender posts the older event shape to
`<base_url>/api/events`. After receiving `API_URL`, I will adapt it to this one-URL
contract and add status, incident, and optional telemetry publishing.
