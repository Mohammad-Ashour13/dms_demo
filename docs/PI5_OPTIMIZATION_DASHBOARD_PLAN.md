# Raspberry Pi 5 Optimization and Safee Dashboard

This file is the implementation checklist, benchmark ledger, power-policy decision, risk register, and acceptance-results record for the Pi 5 deployment. Safety-critical feature order, LightGBM calibration, eye/yawn event timing, Fusion thresholds, decision semantics, and alarm behavior are unchanged.

## Implemented architecture

```text
Picamera2 (640x480 BGR, fixed 15 FPS, monotonic timestamps)
  |-- bounded latest-only AI queue --> MediaPipe 256px --> events --> LightGBM --> Fusion --> alarm
  |                                      |
  |                                      +--> bounded YOLO queue (0.50-0.75s, NCNN)
  |
  +--> bounded raw queue --> one JPEG encode --> immutable EncodedFrame
                                            |--> 5-second RAM incident ring
                                            |--> isolated dashboard preview
                                            +--> FFmpeg MJPEG-in-MP4 copy mux
```

The face/upper-body crop improves object scale; it does not claim to reduce fixed-size NCNN inference cost. Every two seconds the configured driver-context ROI is sampled so phones and hands outside the face crop can be reacquired. A context detection holds that ROI for the temporal window, allowing enough follow-up samples for confirmation instead of creating a one-frame false promise.

### Completed software checklist

- [x] Installable `dms-final-system` package with console commands, independent of checkout folder name.
- [x] Picamera2/libcamera CSI adapter with headless configuration, actual-configuration reporting, monotonic timestamps, and latest-only queues.
- [x] OpenCV retained only for laptop/live compatibility and replay.
- [x] MediaPipe processing width remains 256.
- [x] YOLO interval defaults to 0.50 seconds, is capped at 0.75 seconds, and configuration rejects an interval that cannot provide three samples in 1.5 seconds.
- [x] Periodic full driver-context reacquisition while using the face ROI.
- [x] Runtime is ready for NCNN thread counts of one, two, or three; the winning value must be selected from whole-pipeline Pi results.
- [x] One shared JPEG encode per camera frame with bounded fan-out to recording and preview.
- [x] FFmpeg MJPEG-in-MP4 `-c:v copy`; production preflight performs a real copy-mux probe and prohibits a fallback.
- [x] Fixed five-second pre/onset + five-second post clip, one clip per episode, without indefinite extension.
- [x] Trigger policy covers fatigue states and confirmed phone/smoking/eating; a yawn-only warning is suppressed.
- [x] `.partial` incident assembly, fsync, video SHA256 file, atomic directory rename, and startup recovery/quarantine.
- [x] Two-GiB reserve with oldest-finalized deletion and deletion telemetry; partial/active incidents are never retention targets.
- [x] One-second asynchronous system/firmware monitoring, current/historical throttle flags, PMIC output, reset event, CPU/RAM/RSS/temperature/frequency/disk, queues, drops, FPS, and stage p50/p95.
- [x] Immediate undervoltage/78°C safe mode; preview becomes 5 FPS and YOLO becomes 0.75 seconds. Core camera, face, event, LightGBM, Fusion, and alarm timing are unchanged.
- [x] Recovery requires 60 seconds without a current power flag and temperature at or below 72°C.
- [x] Dashboard runs in a separate process with latest-only IPC and has no CDN dependency.
- [x] Read-only endpoints: `GET /healthz`, `GET /api/v1/status`, `GET /api/v1/incidents`, and `GET /stream.mjpg`.
- [x] Responsive Safee visual system using green, charcoal, white/soft-gray surfaces, local logo, live state/violations, graphs, models, power, storage, and incidents.
- [x] Pi 5 configuration: `configs/runtime.raspberry_pi5.json`.
- [x] Transient opt-in CPU ceiling systemd unit; no `config.txt` overclock or persistent firmware setting.
- [x] Pi soak collector: `scripts/pi5_soak_benchmark.py`.
- [x] Per-class 640-vs-480 recall guardrail tool: `dms-compare-ncnn`.
- [x] Generated and checksum-protected the active 640×640 NCNN export; direct NCNN load/inference smoke test passed.
- [x] Generated and checksum-protected a separate 480×480 NCNN candidate; it is not active.
- [x] End-to-end 45-frame synthetic replay completed with MediaPipe, NCNN, Fusion, recorder hub, dashboard process, and clean shutdown; no ERROR telemetry was emitted.

### Required hardware/data work before production

- [ ] Populate an annotated Safee replay manifest with representative day/night drivers, glasses, hands, phones, cigarettes, food, and false-positive negatives.
- [ ] Accept the 480 export only when phone, cigarette, and drink/food recall each lose no more than 0.02 versus 640.
- [ ] Benchmark NCNN threads 1, 2, and 3 with the whole pipeline and dashboard open.
- [ ] Run clean-boot and two-hour soak tests at CPU ceilings 1.8, 2.0, 2.2, and 2.4 GHz.
- [ ] Enter results in the ledger below and select the lowest fully passing ceiling.
- [ ] Confirm the cable/adapter exposes a stable input under load. If no ceiling passes without current undervoltage, record a hardware blocker; do not weaken face/event/Fusion detection.
- [ ] Obtain Safee legal approval for the third-party YOLO weights and the appropriate Ultralytics license before proprietary commercial deployment.

## Bookworm Lite installation

Target: Raspberry Pi OS Bookworm 64-bit Lite. Picamera2 comes from Raspberry Pi OS packages and the virtual environment deliberately exposes system packages.

```bash
cd /opt/safee-dms
chmod +x scripts/install_pi5_bookworm.sh scripts/apply_cpu_ceiling.sh scripts/pi5_soak_benchmark.py
./scripts/install_pi5_bookworm.sh /opt/safee-dms
.venv/bin/dms-runtime --config configs/runtime.raspberry_pi5.json
```

The installer uses `python3-picamera2 --no-install-recommends`, FFmpeg, and `python3 -m venv --system-site-packages`. PyTorch and Ultralytics are not part of `requirements-raspberry.txt`.

The runtime preflight blocks startup when the active LightGBM descriptor, bundle verification, camera, behavior manifest/classes/checksums, NCNN param/bin/export manifest/image size, free storage, or real FFmpeg copy mux is invalid.

## Model export and 480 guardrail

Run these on an x86/laptop export workstation, not on the Pi production image:

```bash
dms-verify-behavior-model --bundle models/driver_behavior/luthfi_yolo11n --deep
dms-export-ncnn \
  --model models/driver_behavior/luthfi_yolo11n/best_yolo11n.pt \
  --imgsz 640 \
  --output-dir models/driver_behavior/luthfi_yolo11n/best_yolo11n_640_ncnn_model
dms-export-ncnn \
  --model models/driver_behavior/luthfi_yolo11n/best_yolo11n.pt \
  --imgsz 480 \
  --output-dir models/driver_behavior/luthfi_yolo11n/best_yolo11n_480_ncnn_candidate
dms-compare-ncnn \
  --reference models/driver_behavior/luthfi_yolo11n/best_yolo11n_640_ncnn_model \
  --candidate models/driver_behavior/luthfi_yolo11n/best_yolo11n_480_ncnn_candidate \
  --annotations evaluation/behavior_annotations.json \
  --output benchmarks/ncnn-640-vs-480.json
```

The Pi 5 profile remains pinned to 640. Changing it to 480 before `candidate_accepted: true` is prohibited.

## Power profile and benchmark ledger

Selected policy: adaptive safe mode plus the lowest measured CPU ceiling that passes every gate. The current deployed ceiling is `0` (disabled/pending measurement); this avoids guessing a limit without Pi evidence.

For each ceiling, copy `deploy/systemd/safee-dms-power.example` to `/etc/default/safee-dms-power`, set the candidate MHz value, enable `safee-dms-cpu-cap.service`, reboot cleanly, exercise repeated incidents, and collect:

For each thread trial, use a versioned copy of the Pi profile and set the top-level `ncnn_num_threads` to 1, 2, or 3. The soak command rejects a result when the reported ceiling or thread count differs from its arguments.

```bash
scripts/pi5_soak_benchmark.py \
  --label cpu-1800mhz \
  --cpu-max-mhz 1800 \
  --ncnn-threads 1 \
  --duration-sec 7200 \
  --output benchmarks/pi5/cpu-1800mhz.json
```

| CPU ceiling | NCNN threads | Clean boot | Avg / min-5s FPS | Max temp | Capture / recorder drops | Power flags | Incidents/finalization | Result |
|---:|---:|---|---|---:|---|---|---|---|
| 1.8 GHz | 1 / 2 / 3 | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| 2.0 GHz | 1 / 2 / 3 | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| 2.2 GHz | 1 / 2 / 3 | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| 2.4 GHz | 1 / 2 / 3 | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

Selection rule: choose the lowest ceiling whose best whole-pipeline thread count passes. If none pass without current undervoltage, set result to `HARDWARE_BLOCKER_18W_PATH_OR_CABLE`.

## Acceptance record

| Gate | Required | Current result |
|---|---|---|
| Existing and new automated tests | All pass | **PASS — 86 passed** |
| Golden LightGBM parity/checksums | Exact/tolerance 1e-6 | **PASS — 12 samples; raw error 0; calibrated error 1.11e-16** |
| Feature/event/Fusion/alarm replay | No semantic/timing regression | Pending annotated/golden replay run |
| Recorder | One JPEG encode; no decode/H.264; valid MJPEG MP4/SHA256; 10±0.5 sec; atomic | **PASS — automated media/identity/retention/recovery tests** |
| NCNN artifacts | Manifest/classes/source/artifact checksums; direct load | **PASS — 640 and 480 load/inference smoke tests** |
| NCNN candidate | ≤0.02 recall loss for each required behavior | Pending annotations and replay |
| Core processing | Average ≥14 FPS; no five-second window <12 FPS | Pi hardware pending |
| Drops | Capture and recorder <1% | Pi hardware pending |
| Latency | LightGBM p95 <20 ms; Fusion p95 <5 ms; staleness <0.7 sec | Pi hardware pending |
| Power/thermal | No reboot/shutdown/current undervoltage/throttle; <80°C | Pi hardware pending |
| Preview | Smooth 15 FPS normal; 5 FPS safe mode permitted | Pi hardware pending |
| Finalization | No measurable camera stall | Pi hardware pending |

Do not mark a pending gate passed from laptop-only results.

## Dashboard network boundary

The requested dashboard binds to `0.0.0.0:8080`, has no login, and is read-only. It must stay on a private vehicle/LAN segment. Do not port-forward it or expose it to the public internet. Restrict port 8080 to the intended subnet with the host/network firewall, for example:

```bash
sudo ufw allow from 192.168.1.0/24 to any port 8080 proto tcp
sudo ufw deny 8080/tcp
```

Adjust the subnet and rule order for the actual deployment network. Upload and internet synchronization remain out of scope.

## Risks and decisions

| Risk | Control / decision |
|---|---|
| 18W source or cable causes brownout | Current `get_throttled` flags enter safe mode immediately; benchmark ceilings; classify persistent failure as hardware blocker. Raspberry Pi 5's full power budget expects a 5V/5A supply. |
| Pi 5 has no hardware video encoder | Eliminate duplicate encode/decode/H.264 and copy already-compressed MJPEG packets into MP4. |
| SD-card exhaustion/corruption | Two-GiB reserve, oldest-finalized retention, audit records, fsync, SHA256, partial directory, atomic rename. |
| Slow AI/recorder/dashboard accumulates frames | Bounded queues and latest-only semantics; dashboard is a separate process. |
| Face crop misses hands/phone | Periodic driver-context ROI reacquisition. |
| 480 input loses small-object recall | Per-class two-point guardrail; 640 stays active until accepted. |
| Dashboard is unauthenticated | Read-only API, private subnet/firewall, no public exposure. |
| Experimental LightGBM bundle | SHADOW mode only; runtime refuses ACTIVE for an EXPERIMENTAL bundle. |
| Third-party model/license uncertainty | Commercial deployment blocked until Safee confirms weights rights and Ultralytics licensing. |

## systemd activation after acceptance

The checked-in service assumes `/opt/safee-dms` and user/group `safee`. Adapt those two deployment facts if necessary.

```bash
sudo install -m 0644 deploy/systemd/safee-dms.service /etc/systemd/system/
sudo install -m 0644 deploy/systemd/safee-dms-cpu-cap.service /etc/systemd/system/
sudo install -m 0644 deploy/systemd/safee-dms-power.example /etc/default/safee-dms-power
sudo systemctl daemon-reload
sudo systemctl enable --now safee-dms-cpu-cap.service safee-dms.service
```

Only set `SAFEE_CPU_MAX_MHZ` after recording the passing result in this file.

## Primary references

- [Raspberry Pi 5 software environment and absence of hardware video encode](https://www.raspberrypi.com/news/optimising-raspberry-pi-5s-software-environment/)
- [Raspberry Pi power-supply and power-budget documentation](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html)
- [Official Picamera2 manual](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf)
- [Ultralytics licensing](https://www.ultralytics.com/license)
