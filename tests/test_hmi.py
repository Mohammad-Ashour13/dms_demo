from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np

from dms_final_system.runtime.hmi.visual import VisualHMI


class _Telemetry:
    def __init__(self):
        self.records = []

    def emit(self, stage, event, payload=None, **kwargs):
        self.records.append((stage, event, payload, kwargs))


def _inputs():
    decision = SimpleNamespace(
        driver_state=SimpleNamespace(value="NORMAL"), monotonic_sec=1.0
    )
    alarm = SimpleNamespace(
        audio_playing=False, backend_health="READY", acknowledged=False
    )
    return np.zeros((80, 120, 3), dtype=np.uint8), decision, alarm


def test_hmi_highgui_runs_synchronously_on_owner_thread(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "dms_final_system.runtime.hmi.visual.cv2.imshow",
        lambda *_: calls.append(("imshow", threading.get_ident())),
    )
    monkeypatch.setattr(
        "dms_final_system.runtime.hmi.visual.cv2.waitKey", lambda *_: 0
    )
    telemetry = _Telemetry()
    hmi = VisualHMI(
        SimpleNamespace(
            enabled=True,
            show_camera=True,
            window_name="test",
            critical_flash_hz=2.0,
        ),
        telemetry,
    )

    hmi.push(*_inputs())

    assert calls == [("imshow", threading.get_ident())]
    assert not telemetry.records


def test_hmi_rejects_highgui_call_from_non_owner_thread(monkeypatch):
    monkeypatch.setattr(
        "dms_final_system.runtime.hmi.visual.cv2.imshow",
        lambda *_: (_ for _ in ()).throw(AssertionError("imshow must not run")),
    )
    telemetry = _Telemetry()
    hmi = VisualHMI(
        SimpleNamespace(
            enabled=True,
            show_camera=True,
            window_name="test",
            critical_flash_hz=2.0,
        ),
        telemetry,
    )
    thread = threading.Thread(target=lambda: hmi.push(*_inputs()))
    thread.start()
    thread.join()

    assert hmi.failed
    assert telemetry.records[0][0:2] == ("HMI", "display_failed")
