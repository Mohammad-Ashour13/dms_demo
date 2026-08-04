from __future__ import annotations

import math
import shutil
import struct
import subprocess
import tempfile
import threading
import wave
from abc import ABC, abstractmethod
from pathlib import Path

from dms_final_system.shared.contracts import AlarmCommand
from dms_final_system.runtime.alarm.patterns import ALARM_PATTERNS, TonePattern


class AlarmOutput(ABC):
    @abstractmethod
    def probe(self) -> tuple[bool, str]:
        """Return output availability and a human-readable backend name."""

    @abstractmethod
    def play(self, command: AlarmCommand, stop_event: threading.Event) -> str:
        """Play one command and return COMPLETED or PREEMPTED."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the currently playing pattern, if any."""

    def close(self) -> None:
        self.stop()


class NullAlarmOutput(AlarmOutput):
    def __init__(self):
        self.played: list[AlarmCommand] = []

    def probe(self) -> tuple[bool, str]:
        return True, "null"

    def play(self, command: AlarmCommand, stop_event: threading.Event) -> str:
        self.played.append(command)
        return "PREEMPTED" if stop_event.is_set() else "COMPLETED"

    def stop(self) -> None:
        return None


class LinuxAudioOutput(AlarmOutput):
    """Dependency-free PCM output using the first available Linux player."""

    BACKENDS = ("aplay", "paplay", "pw-play", "ffplay")

    def __init__(self, backend="auto", device="default", master_gain=0.40):
        self.requested_backend = str(backend).lower()
        self.device = str(device)
        self.master_gain = float(master_gain)
        self.backend: str | None = None
        self.process: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.audio_dir = Path(tempfile.gettempdir()) / "dms_alarm_audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.files: dict[str, Path] = {}

    def probe(self) -> tuple[bool, str]:
        candidates = self.BACKENDS if self.requested_backend == "auto" else (self.requested_backend,)
        self.backend = next((item for item in candidates if shutil.which(item)), None)
        if self.backend is None:
            return False, f"no supported audio player found ({', '.join(candidates)})"
        for name, pattern in ALARM_PATTERNS.items():
            path = self.audio_dir / f"{name.lower()}-{int(self.master_gain * 100):02d}.wav"
            self._write_wav(path, pattern)
            self.files[name] = path
        return True, self.backend

    def _write_wav(self, path: Path, pattern: TonePattern) -> None:
        rate = 44_100
        samples: list[int] = []
        gain = min(0.60, max(0.0, self.master_gain * pattern.gain_scale))
        for pulse in pattern.pulses:
            count = max(1, int(rate * pulse.duration_sec))
            fade = min(int(rate * 0.02), count // 2)
            for index in range(count):
                envelope = 1.0
                if fade:
                    envelope = min(1.0, index / fade, (count - 1 - index) / fade)
                value = gain * envelope * math.sin(2.0 * math.pi * pulse.frequency_hz * index / rate)
                samples.append(int(max(-1.0, min(1.0, value)) * 32767))
            samples.extend([0] * int(rate * pulse.gap_after_sec))
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    def _command(self, path: Path) -> list[str]:
        if self.backend == "aplay":
            result = ["aplay", "-q"]
            if self.device and self.device != "default":
                result += ["-D", self.device]
            return [*result, str(path)]
        if self.backend == "paplay":
            return ["paplay", str(path)]
        if self.backend == "pw-play":
            return ["pw-play", str(path)]
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]

    def play(self, command: AlarmCommand, stop_event: threading.Event) -> str:
        if self.backend is None or command.pattern not in self.files:
            raise RuntimeError("audio backend was not probed or alarm pattern is unknown")
        process = subprocess.Popen(
            self._command(self.files[command.pattern]),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        with self.lock:
            self.process = process
        while process.poll() is None:
            if stop_event.wait(0.02):
                process.terminate()
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    process.kill()
                with self.lock:
                    self.process = None
                return "PREEMPTED"
        stderr = (process.stderr.read() if process.stderr else b"").decode("utf-8", "replace").strip()
        with self.lock:
            self.process = None
        if process.returncode:
            raise RuntimeError(f"{self.backend} failed with code {process.returncode}: {stderr}")
        return "COMPLETED"

    def stop(self) -> None:
        with self.lock:
            process = self.process
        if process is not None and process.poll() is None:
            process.terminate()


class GPIOBuzzerOutput(AlarmOutput):
    """Production extension point; intentionally disabled until hardware is specified."""

    def probe(self) -> tuple[bool, str]:
        return False, "GPIO buzzer hardware is not configured"

    def play(self, command: AlarmCommand, stop_event: threading.Event) -> str:
        raise NotImplementedError("Configure a transistor-driven buzzer before enabling GPIO output")

    def stop(self) -> None:
        return None
