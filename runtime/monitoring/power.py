from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SafeModeState:
    enabled: bool
    changed: bool
    reason: str
    entered_monotonic_sec: float | None


class SafeModeController:
    def __init__(self, config):
        self.enabled = bool(config.enabled)
        self.enter_temperature = float(config.enter_temperature_c)
        self.exit_temperature = float(config.exit_temperature_c)
        self.recovery_sec = float(config.recovery_sec)
        self.active = False
        self.entered_at: float | None = None
        self.healthy_since: float | None = None
        self.reason = "DISABLED" if not self.enabled else "NORMAL"

    def update(self, now: float, health: dict) -> SafeModeState:
        previous = self.active
        if not self.enabled:
            return SafeModeState(False, False, "DISABLED", None)
        throttled = health.get("throttled") or {}
        temperature = health.get("temperature_c")
        undervoltage = bool(throttled.get("undervoltage_current"))
        thermal_flag = bool(throttled.get("soft_temperature_limit_current"))
        too_hot = temperature is not None and float(temperature) >= self.enter_temperature
        if undervoltage or thermal_flag or too_hot:
            self.active = True
            self.healthy_since = None
            if self.entered_at is None:
                self.entered_at = float(now)
            if undervoltage:
                self.reason = "UNDERVOLTAGE"
            elif thermal_flag:
                self.reason = "THERMAL_LIMIT"
            else:
                self.reason = "HIGH_TEMPERATURE"
        elif self.active:
            cool_enough = temperature is None or float(temperature) <= self.exit_temperature
            if cool_enough:
                if self.healthy_since is None:
                    self.healthy_since = float(now)
                if now - self.healthy_since >= self.recovery_sec:
                    self.active = False
                    self.entered_at = None
                    self.healthy_since = None
                    self.reason = "RECOVERED"
            else:
                self.healthy_since = None
        else:
            self.reason = "NORMAL"
        return SafeModeState(self.active, self.active != previous, self.reason, self.entered_at)
