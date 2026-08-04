from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from dms_final_system.shared.contracts import TemporalFeatureSnapshot


@dataclass(slots=True)
class FeatureDriftResult:
    available: bool
    model_contribution_enabled: bool
    out_of_range_count: int
    evaluated_feature_count: int
    out_of_range_fraction: float
    consecutive_bad_windows: int
    consecutive_good_windows: int
    offending_features: list[dict]
    reason: str
    changed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class FeatureDriftMonitor:
    """Stateful q01/q99 guard over the highest-gain training features."""

    def __init__(
        self,
        bundle_dir: Path,
        *,
        enabled=True,
        top_feature_count=20,
        out_of_range_fraction=0.30,
        trigger_consecutive_windows=3,
        recovery_consecutive_windows=3,
    ):
        self.enabled = bool(enabled)
        self.threshold = float(out_of_range_fraction)
        self.trigger_windows = int(trigger_consecutive_windows)
        self.recovery_windows = int(recovery_consecutive_windows)
        self.bad_windows = self.good_windows = 0
        self.model_enabled = True
        reference_path = Path(bundle_dir) / "feature_reference.json"
        self.available = self.enabled and reference_path.is_file()
        self.reference: dict[str, dict] = {}
        self.feature_names: list[str] = []
        if self.available:
            payload = json.loads(reference_path.read_text(encoding="utf-8"))
            self.reference = dict(payload.get("features", {}))
            ranked = sorted(
                self.reference,
                key=lambda name: (
                    self.reference[name].get("gain_rank", 10**9),
                    -self.reference[name].get("gain_fraction", 0.0),
                    name,
                ),
            )
            self.feature_names = ranked[: int(top_feature_count)]
            self.available = bool(self.feature_names)

    def evaluate(self, snapshot: TemporalFeatureSnapshot) -> FeatureDriftResult:
        if not self.available or not snapshot.valid:
            return FeatureDriftResult(
                available=self.available,
                model_contribution_enabled=True,
                out_of_range_count=0,
                evaluated_feature_count=0,
                out_of_range_fraction=0.0,
                consecutive_bad_windows=self.bad_windows,
                consecutive_good_windows=self.good_windows,
                offending_features=[],
                reason="REFERENCE_UNAVAILABLE" if not self.available else "INVALID_WINDOW",
            )

        values = dict(zip(snapshot.feature_names, snapshot.ordered_features))
        offending = []
        for name in self.feature_names:
            value = float(values.get(name, float("nan")))
            limits = self.reference[name]
            low, high = float(limits["q01"]), float(limits["q99"])
            if not math.isfinite(value) or value < low or value > high:
                offending.append(
                    {"feature": name, "value": value, "q01": low, "q99": high}
                )
        fraction = len(offending) / max(len(self.feature_names), 1)
        bad = fraction > self.threshold
        previous = self.model_enabled
        if bad:
            self.bad_windows += 1
            self.good_windows = 0
            if self.bad_windows >= self.trigger_windows:
                self.model_enabled = False
        else:
            self.good_windows += 1
            self.bad_windows = 0
            if self.good_windows >= self.recovery_windows:
                self.model_enabled = True
        reason = "MODEL_INPUT_OOD" if not self.model_enabled else (
            "DRIFT_PENDING" if bad else "IN_DISTRIBUTION"
        )
        return FeatureDriftResult(
            available=True,
            model_contribution_enabled=self.model_enabled,
            out_of_range_count=len(offending),
            evaluated_feature_count=len(self.feature_names),
            out_of_range_fraction=fraction,
            consecutive_bad_windows=self.bad_windows,
            consecutive_good_windows=self.good_windows,
            offending_features=offending,
            reason=reason,
            changed=previous != self.model_enabled,
        )
