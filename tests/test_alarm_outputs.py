from __future__ import annotations

import sys
import threading
import types
from types import SimpleNamespace

from dms_final_system.runtime.alarm.outputs import GPIOBuzzerOutput, PlatformAudioOutput
from dms_final_system.shared.contracts import AlarmCommand, DriverState


def _command(pattern: str = "DROWSY") -> AlarmCommand:
    return AlarmCommand(
        "test-alarm", "test-episode", 0.0, DriverState.DROWSY,
        pattern, "ONSET", 1, ["TEST"], None,
    )


def test_auto_uses_winsound_on_windows_and_can_preempt(monkeypatch):
    calls = []

    def play_sound(sound, flags):
        calls.append((sound, flags))

    fake_winsound = SimpleNamespace(
        SND_FILENAME=1,
        SND_ASYNC=2,
        SND_NODEFAULT=4,
        PlaySound=play_sound,
    )
    from dms_final_system.runtime.alarm import outputs

    monkeypatch.setattr(outputs.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "winsound", fake_winsound)
    output = PlatformAudioOutput()

    ok, backend = output.probe()
    assert (ok, backend) == (True, "winsound")

    stop_event = threading.Event()
    stop_event.set()
    assert output.play(_command(), stop_event) == "PREEMPTED"
    assert calls[0][1] == 7
    assert calls[-1] == (None, 0)


def test_auto_keeps_linux_player_discovery(monkeypatch):
    from dms_final_system.runtime.alarm import outputs

    monkeypatch.setattr(outputs.sys, "platform", "linux")
    monkeypatch.setattr(outputs.shutil, "which", lambda name: "/usr/bin/paplay" if name == "paplay" else None)
    output = PlatformAudioOutput()

    assert output.probe() == (True, "paplay")


class _FakePWMOutputDevice:
    def __init__(self, pin, **kwargs):
        self.pin = pin
        self.history = []
        self.frequency = kwargs["frequency"]
        self.value = kwargs["initial_value"]
        self.closed = False

    def __setattr__(self, name, value):
        if name == "value" and "history" in self.__dict__:
            self.history.append(value)
        object.__setattr__(self, name, value)

    def off(self):
        self.value = 0.0

    def close(self):
        self.closed = True


def test_gpio_active_buzzer_scales_behavior_lower_than_critical(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "gpiozero",
        types.SimpleNamespace(PWMOutputDevice=_FakePWMOutputDevice),
    )
    output = GPIOBuzzerOutput(18, buzzer_type="active", master_gain=1.0)
    assert output.probe()[0]
    try:
        output.play(_command("DISTRACTION"), threading.Event())
        distraction_peak = max(output.device.history)
        output.device.history.clear()
        output.play(_command("CRITICAL"), threading.Event())
        assert max(output.device.history) > distraction_peak
    finally:
        output.close()


def test_gpio_buzzer_probe_fails_safely_without_gpio_access(monkeypatch):
    class BrokenPWM:
        def __init__(self, *args, **kwargs):
            raise PermissionError("gpio denied")

    monkeypatch.setitem(sys.modules, "gpiozero", types.SimpleNamespace(PWMOutputDevice=BrokenPWM))
    output = GPIOBuzzerOutput(18)
    ok, detail = output.probe()
    assert not ok
    assert "GPIO18 unavailable" in detail
