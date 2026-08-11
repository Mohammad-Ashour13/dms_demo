#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def fetch_status(url: str) -> dict:
    with urllib.request.urlopen(url.rstrip("/") + "/api/v1/status", timeout=2.0) as response:
        return json.load(response)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a Safee Pi 5 soak benchmark")
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--duration-sec", type=float, default=7200.0)
    parser.add_argument("--label", required=True, help="For example cpu-1800mhz")
    parser.add_argument("--cpu-max-mhz", type=int, choices=(1800, 2000, 2200, 2400), required=True)
    parser.add_argument("--ncnn-threads", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.monotonic()
    samples: list[dict] = []
    failures: list[str] = []
    while time.monotonic() - started < args.duration_sec:
        try:
            samples.append(fetch_status(args.url))
        except Exception as exc:
            failures.append(repr(exc))
        time.sleep(1.0)
    if len(samples) < 2:
        raise RuntimeError("Fewer than two dashboard samples were collected")

    interval_fps = []
    for previous, current in zip(samples, samples[1:]):
        p_perf, c_perf = previous.get("performance", {}), current.get("performance", {})
        elapsed = float(current.get("updated_monotonic_sec", 0)) - float(
            previous.get("updated_monotonic_sec", 0)
        )
        frames = int(c_perf.get("processed_frames", 0)) - int(p_perf.get("processed_frames", 0))
        if elapsed > 0:
            interval_fps.append(frames / elapsed)
    five_second_fps = [
        statistics.mean(interval_fps[index:index + 5])
        for index in range(0, max(0, len(interval_fps) - 4))
    ]

    temperatures = [
        float(value)
        for sample in samples
        if (value := (sample.get("system") or {}).get("temperature_c")) is not None
    ]
    staleness = [
        float(value)
        for sample in samples
        if (value := (sample.get("performance") or {}).get("decision_staleness_sec"))
        is not None
    ]
    current_power_flags = []
    for sample in samples:
        flags = ((sample.get("system") or {}).get("throttled") or {})
        current_power_flags.append(
            any(bool(flags.get(name)) for name in (
                "undervoltage_current", "frequency_capped_current",
                "throttled_current", "soft_temperature_limit_current",
            ))
        )
    final_perf = samples[-1].get("performance") or {}
    final_camera = samples[-1].get("camera") or {}
    final_recording = samples[-1].get("recording") or {}
    final_system = samples[-1].get("system") or {}
    final_models = samples[-1].get("models") or {}
    observed_cpu_max = final_system.get("cpu_max_frequency_mhz")
    capture_frames = max(1, int(final_perf.get("capture_frames", 0)))
    encoded_frames = max(1, int(final_perf.get("encoded_frames", 0)))
    capture_drop_rate = int(final_camera.get("dropped_frames", 0)) / capture_frames
    recorder_drop_rate = (
        int(final_recording.get("dropped_frames", 0))
        + int(final_recording.get("hub_dropped_frames", 0))
    ) / encoded_frames
    latency = final_perf.get("latency") or {}
    lightgbm_p95 = (latency.get("lightgbm") or {}).get("p95_ms")
    fusion_p95 = (latency.get("fusion") or {}).get("p95_ms")
    criteria = {
        "requested_cpu_ceiling_applied": (
            observed_cpu_max is not None
            and abs(float(observed_cpu_max) - args.cpu_max_mhz) <= 25.0
        ),
        "requested_ncnn_threads_applied": (
            int(final_models.get("ncnn_threads", -1)) == args.ncnn_threads
        ),
        "dashboard_disconnects_zero": not failures,
        "average_core_fps_at_least_14": statistics.mean(interval_fps) >= 14.0,
        "five_second_fps_at_least_12": bool(five_second_fps) and min(five_second_fps) >= 12.0,
        "capture_drop_rate_below_1_percent": capture_drop_rate < 0.01,
        "recorder_drop_rate_below_1_percent": recorder_drop_rate < 0.01,
        "lightgbm_p95_below_20_ms": lightgbm_p95 is not None and lightgbm_p95 < 20.0,
        "fusion_p95_below_5_ms": fusion_p95 is not None and fusion_p95 < 5.0,
        "decision_staleness_below_0_7_sec": bool(staleness) and max(staleness) < 0.7,
        "temperature_below_80_c": bool(temperatures) and max(temperatures) < 80.0,
        "no_current_power_or_throttle_flags": not any(current_power_flags),
    }
    result = {
        "schema_version": "pi5-soak-result-v1",
        "label": args.label,
        "profile": {
            "cpu_max_mhz": args.cpu_max_mhz,
            "ncnn_threads": args.ncnn_threads,
            "observed_cpu_max_mhz": (samples[-1].get("system") or {}).get(
                "cpu_max_frequency_mhz"
            ),
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "requested_duration_sec": args.duration_sec,
        "sample_count": len(samples),
        "dashboard_failures": failures,
        "metrics": {
            "average_core_fps": statistics.mean(interval_fps),
            "minimum_five_second_fps": min(five_second_fps) if five_second_fps else None,
            "capture_drop_rate": capture_drop_rate,
            "recorder_drop_rate": recorder_drop_rate,
            "maximum_temperature_c": max(temperatures) if temperatures else None,
            "decision_staleness_p95_sec": percentile(staleness, 0.95),
            "decision_staleness_max_sec": max(staleness) if staleness else None,
            "lightgbm_p95_ms": lightgbm_p95,
            "fusion_p95_ms": fusion_p95,
        },
        "criteria": criteria,
        "automated_gate_passed": all(criteria.values()),
        "manual_gates_required": [
            "clean boot and no reboot/shutdown",
            "repeated incident trigger/finalization",
            "no measurable camera stall at finalization",
            "smooth dashboard preview",
            "per-class annotated replay recall",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
