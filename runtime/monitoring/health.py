from __future__ import annotations

import shutil
import subprocess
import threading
import time
import copy
from pathlib import Path
from typing import Any


THROTTLED_FLAGS = {
    0: "undervoltage_current",
    1: "frequency_capped_current",
    2: "throttled_current",
    3: "soft_temperature_limit_current",
    16: "undervoltage_occurred",
    17: "frequency_capped_occurred",
    18: "throttled_occurred",
    19: "soft_temperature_limit_occurred",
}


def parse_throttled_value(value: int | str | None) -> dict[str, Any]:
    if value is None:
        numeric = None
    elif isinstance(value, str):
        text = value.strip().lower()
        if "=" in text:
            text = text.split("=", 1)[1]
        try:
            numeric = int(text, 16 if text.startswith("0x") else 10)
        except ValueError:
            numeric = None
    else:
        numeric = int(value)
    flags = {
        name: bool(numeric is not None and numeric & (1 << bit))
        for bit, name in THROTTLED_FLAGS.items()
    }
    return {
        "raw": numeric,
        "hex": f"0x{numeric:x}" if numeric is not None else None,
        **flags,
    }


def _read_float(path: Path, divisor=1.0) -> float | None:
    try:
        return float(path.read_text(encoding="utf-8").strip()) / divisor
    except (OSError, ValueError):
        return None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip("\x00\n ")
    except OSError:
        return None


class SystemMetricsCollector:
    """Low-rate Pi/system metrics without placing work in the AI frame loop."""

    def __init__(self, storage_path: Path, *, vcgencmd_timeout_sec=0.5):
        self.storage_path = Path(storage_path)
        self.vcgencmd_timeout_sec = float(vcgencmd_timeout_sec)
        self.process = None
        self.reset_event = _read_text(Path("/proc/device-tree/chosen/power/reset_event"))
        try:
            import psutil

            self.psutil = psutil
            self.process = psutil.Process()
            psutil.cpu_percent(interval=None)
            self.process.cpu_percent(interval=None)
        except ImportError:
            self.psutil = None

    def _throttled(self) -> dict[str, Any]:
        if not shutil.which("vcgencmd"):
            return parse_throttled_value(None)
        try:
            result = subprocess.run(
                ["vcgencmd", "get_throttled"],
                capture_output=True,
                text=True,
                timeout=self.vcgencmd_timeout_sec,
                check=False,
            )
            return parse_throttled_value(result.stdout if result.returncode == 0 else None)
        except (OSError, subprocess.SubprocessError):
            return parse_throttled_value(None)

    def _pmic(self) -> str | None:
        if not shutil.which("vcgencmd"):
            return None
        try:
            result = subprocess.run(
                ["vcgencmd", "pmic_read_adc"],
                capture_output=True,
                text=True,
                timeout=self.vcgencmd_timeout_sec,
                check=False,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    def collect(self) -> dict[str, Any]:
        cpu_percent = ram_percent = rss_bytes = process_cpu_percent = None
        if self.psutil is not None:
            cpu_percent = float(self.psutil.cpu_percent(interval=None))
            memory = self.psutil.virtual_memory()
            ram_percent = float(memory.percent)
            if self.process is not None:
                try:
                    rss_bytes = int(self.process.memory_info().rss)
                    process_cpu_percent = float(self.process.cpu_percent(interval=None))
                except self.psutil.Error:
                    pass
        thermal = _read_float(Path("/sys/class/thermal/thermal_zone0/temp"), 1000.0)
        frequency_khz = _read_float(
            Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
        )
        frequency_mhz = frequency_khz / 1000.0 if frequency_khz is not None else None
        maximum_frequency_khz = _read_float(
            Path("/sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq")
        )
        maximum_frequency_mhz = (
            maximum_frequency_khz / 1000.0
            if maximum_frequency_khz is not None else None
        )
        try:
            disk = shutil.disk_usage(self.storage_path)
            disk_payload = {
                "total_bytes": int(disk.total),
                "used_bytes": int(disk.used),
                "free_bytes": int(disk.free),
                "used_percent": float(disk.used / max(disk.total, 1) * 100.0),
            }
        except OSError:
            disk_payload = {
                "total_bytes": None,
                "used_bytes": None,
                "free_bytes": None,
                "used_percent": None,
            }
        return {
            "sampled_monotonic_sec": time.monotonic(),
            "cpu_percent": cpu_percent,
            "process_cpu_percent": process_cpu_percent,
            "ram_percent": ram_percent,
            "process_rss_bytes": rss_bytes,
            "temperature_c": thermal,
            "cpu_frequency_mhz": frequency_mhz,
            "cpu_max_frequency_mhz": maximum_frequency_mhz,
            "disk": disk_payload,
            "throttled": self._throttled(),
            "pmic_adc": self._pmic(),
            "reset_event": self.reset_event,
        }


class SystemMetricsSampler:
    """Run the potentially slow firmware/system probes outside the AI loop."""

    def __init__(self, collector: SystemMetricsCollector, interval_sec=1.0):
        self.collector = collector
        self.interval_sec = max(0.1, float(interval_sec))
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.value: dict[str, Any] = {
            "sampled_monotonic_sec": None,
            "temperature_c": None,
            "throttled": parse_throttled_value(None),
            "disk": {},
        }
        self.worker = threading.Thread(
            target=self._run, name="system-metrics", daemon=True
        )

    def start(self) -> "SystemMetricsSampler":
        self.worker.start()
        return self

    def _run(self) -> None:
        while not self.stop_event.is_set():
            sampled = self.collector.collect()
            with self.lock:
                self.value = sampled
            self.stop_event.wait(self.interval_sec)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return copy.deepcopy(self.value)

    def close(self) -> None:
        self.stop_event.set()
        if self.worker.is_alive():
            self.worker.join(timeout=max(2.0, self.interval_sec + 1.0))
        if self.worker.is_alive():
            raise RuntimeError("System metrics sampler did not stop cleanly")
