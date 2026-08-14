from __future__ import annotations

import sys
import threading
from types import SimpleNamespace

from dms_final_system.runtime.alarm.outputs import PlatformAudioOutput
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
