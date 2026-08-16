from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TonePulse:
    frequency_hz: float
    duration_sec: float
    gap_after_sec: float = 0.0


@dataclass(frozen=True, slots=True)
class TonePattern:
    name: str
    pulses: tuple[TonePulse, ...]
    gain_scale: float = 1.0


ALARM_PATTERNS = {
    # Short and deliberately quiet: used for phone/smoking/eating violations.
    "DISTRACTION": TonePattern(
        "DISTRACTION",
        (TonePulse(620.0, 0.12, 0.10), TonePulse(620.0, 0.12)),
        0.30,
    ),
    "DROWSY": TonePattern("DROWSY", (TonePulse(750.0, 0.45),), 0.875),
    "DROWSY_ESCALATED": TonePattern(
        "DROWSY_ESCALATED",
        (TonePulse(900.0, 0.30, 0.15), TonePulse(900.0, 0.30, 0.15), TonePulse(900.0, 0.30)),
    ),
    "CRITICAL": TonePattern(
        "CRITICAL",
        (TonePulse(1100.0, 0.70, 0.20), TonePulse(1400.0, 0.70)),
        1.25,
    ),
    "STARTUP": TonePattern("STARTUP", (TonePulse(650.0, 0.15),), 0.50),
}
