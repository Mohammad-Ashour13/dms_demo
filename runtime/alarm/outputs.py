from __future__ import annotations

import math
import shutil
import struct
import subprocess
import sys
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


class PlatformAudioOutput(AlarmOutput):
    """Dependency-free local WAV output for Windows and Linux."""

    LINUX_BACKENDS = ("aplay", "paplay", "pw-play", "ffplay")
    WINDOWS_BACKEND = "winsound"

    def __init__(self, backend="auto", device="default", master_gain=0.40):
        self.requested_backend = str(backend).lower()
        self.device = str(device)
        self.master_gain = float(master_gain)
        self.backend: str | None = None
        self.process: subprocess.Popen | None = None
        self.winsound = None
        self.lock = threading.Lock()
        self.audio_dir = Path(tempfile.gettempdir()) / "dms_alarm_audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.files: dict[str, Path] = {}
        self.durations: dict[str, float] = {}

    def probe(self) -> tuple[bool, str]:
        candidates = self._candidates()
        self.backend = next((item for item in candidates if self._available(item)), None)
        if self.backend is None:
            return False, f"no supported audio player found ({', '.join(candidates)})"
        for name, pattern in ALARM_PATTERNS.items():
            path = self.audio_dir / f"{name.lower()}-{int(self.master_gain * 100):02d}.wav"
            self._write_wav(path, pattern)
            self.files[name] = path
        return True, self.backend

    def _candidates(self) -> tuple[str, ...]:
        if self.requested_backend != "auto":
            return (self.requested_backend,)
        if sys.platform == "win32":
            return (self.WINDOWS_BACKEND, *self.LINUX_BACKENDS)
        return self.LINUX_BACKENDS

    def _available(self, backend: str) -> bool:
        if backend != self.WINDOWS_BACKEND:
            return bool(shutil.which(backend))
        if sys.platform != "win32":
            return False
        try:
            import winsound
        except ImportError:
            return False
        self.winsound = winsound
        return True

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
        self.durations[pattern.name] = sum(
            pulse.duration_sec + pulse.gap_after_sec for pulse in pattern.pulses
        )

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
        if self.backend == self.WINDOWS_BACKEND:
            if self.winsound is None:
                raise RuntimeError("winsound backend was not initialized")
            self.winsound.PlaySound(
                str(self.files[command.pattern]),
                self.winsound.SND_FILENAME
                | self.winsound.SND_ASYNC
                | self.winsound.SND_NODEFAULT,
            )
            if stop_event.wait(self.durations[command.pattern]):
                self.stop()
                return "PREEMPTED"
            return "COMPLETED"
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
        if self.backend == self.WINDOWS_BACKEND and self.winsound is not None:
            self.winsound.PlaySound(None, 0)
            return
        with self.lock:
            process = self.process
        if process is not None and process.poll() is None:
            process.terminate()


# Keep the old public name available for existing callers and scripts.
LinuxAudioOutput = PlatformAudioOutput


class GPIOBuzzerOutput(AlarmOutput):
    """Drive a transistor-switched buzzer without loading a GPIO directly.

    ``pin`` uses BCM numbering (GPIO18, not physical header pin 18).  An active
    buzzer is power-modulated at ``pwm_frequency_hz``; a passive buzzer uses the
    tone frequency from each pattern.  Hardware access is imported lazily so
    laptop/replay installations do not need GPIO packages.
    """

    def __init__(
        self,
        pin: int = 18,
        *,
        buzzer_type: str = "active",
        pwm_frequency_hz: float = 100.0,
        master_gain: float = 1.0,
    ):
        self.pin = int(pin)
        self.buzzer_type = str(buzzer_type).lower()
        self.pwm_frequency_hz = float(pwm_frequency_hz)
        self.master_gain = float(master_gain)
        self.device = None
        self.lock = threading.RLock()

    def probe(self) -> tuple[bool, str]:
        if self.buzzer_type not in {"active", "passive"}:
            return False, f"unsupported GPIO buzzer type: {self.buzzer_type}"
        try:
            from gpiozero import PWMOutputDevice

            self.device = PWMOutputDevice(
                self.pin,
                active_high=True,
                initial_value=0.0,
                frequency=self.pwm_frequency_hz,
            )
        except Exception as exc:
            self.device = None
            return False, f"GPIO{self.pin} unavailable: {exc}"
        return True, f"gpiozero PWM GPIO{self.pin} ({self.buzzer_type} buzzer)"

    def play(self, command: AlarmCommand, stop_event: threading.Event) -> str:
        if self.device is None:
            raise RuntimeError("GPIO buzzer was not probed")
        try:
            pattern = ALARM_PATTERNS[command.pattern]
        except KeyError as exc:
            raise RuntimeError(f"unknown GPIO alarm pattern: {command.pattern}") from exc

        level = max(0.0, min(1.0, self.master_gain * pattern.gain_scale))
        # A passive piezo needs alternating edges; 50% is its maximum useful
        # duty cycle.  An active buzzer accepts DC, so duty directly controls
        # its average power (the perceived result still depends on the module).
        duty_cycle = level if self.buzzer_type == "active" else 0.5 * level
        for pulse in pattern.pulses:
            if stop_event.is_set():
                self.stop()
                return "PREEMPTED"
            with self.lock:
                if self.buzzer_type == "passive":
                    self.device.frequency = pulse.frequency_hz
                self.device.value = duty_cycle
            if stop_event.wait(pulse.duration_sec):
                self.stop()
                return "PREEMPTED"
            self.stop()
            if pulse.gap_after_sec and stop_event.wait(pulse.gap_after_sec):
                return "PREEMPTED"
        return "COMPLETED"

    def stop(self) -> None:
        with self.lock:
            if self.device is not None:
                self.device.value = 0.0

    def close(self) -> None:
        with self.lock:
            device, self.device = self.device, None
        if device is not None:
            device.off()
            device.close()
