from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def build_report(log_path: Path) -> dict:
    records = [json.loads(line) for line in Path(log_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    decisions = [item for item in records if item.get("stage") == "Fusion" and item.get("event") == "decision"]
    predictions = [item for item in records if item.get("stage") == "Model" and item.get("event") == "prediction"]
    incidents = [item for item in records if item.get("stage") == "Recorder" and item.get("event") == "incident_finalized"]
    events = [item for item in records if item.get("stage") == "Events" and item.get("event") == "snapshot"]
    drifts = [item for item in records if item.get("stage") == "Model" and item.get("event") == "feature_drift"]
    calibrations = [item for item in records if item.get("stage") == "Calibration" and item.get("event") == "status"]
    states = Counter(item["payload"].get("driver_state", "UNKNOWN") for item in decisions)
    targets = Counter(item["payload"].get("target_state", "UNKNOWN") for item in decisions)
    transitions = [item for item in decisions if item["payload"].get("changed")]
    probabilities = [item["payload"].get("calibrated_probability", 0.0) for item in predictions]
    perclos = [
        item["payload"].get("perclos_30s") for item in events
        if item["payload"].get("perclos_30s") is not None
    ]
    ready_calibration = next(
        (item["payload"] for item in calibrations if item["payload"].get("status") == "READY"),
        None,
    )
    return {
        "log_path": str(log_path),
        "decision_records": len(decisions),
        "prediction_records": len(predictions),
        "state_record_counts": dict(states),
        "target_state_record_counts": dict(targets),
        "normal_decision_ratio": _ratio(states.get("NORMAL", 0), len(decisions)),
        "transition_count": len(transitions),
        "transitions": [
            {
                "utc_timestamp": item["utc_timestamp"],
                "previous": item["payload"].get("previous_state"),
                "new": item["payload"].get("driver_state"),
                "target": item["payload"].get("target_state"),
                "state_entry_reason": item["payload"].get("state_entry_reason", []),
                "current_reason_codes": item["payload"].get(
                    "current_reason_codes", item["payload"].get("reason_codes", [])
                ),
            }
            for item in transitions
        ],
        "max_model_probability": max(probabilities, default=None),
        "mean_model_probability": sum(probabilities) / len(probabilities) if probabilities else None,
        "median_model_probability": median(probabilities) if probabilities else None,
        "model_contribution_enabled_ratio": _ratio(
            sum(bool(item["payload"].get("model_contribution_enabled")) for item in decisions),
            len(decisions),
        ),
        "calibration": ready_calibration,
        "events": {
            "eye_state_counts": dict(Counter(item["payload"].get("eye_state") for item in events)),
            "valid_eye_observation_ratio": _ratio(
                sum(bool(item["payload"].get("eye_signal_valid")) for item in events),
                len(events),
            ),
            "max_blink_count_60s": max(
                (item["payload"].get("blink_count_60s", 0) for item in events), default=0
            ),
            "max_yawn_count_60s": max(
                (item["payload"].get("yawn_count_60s", 0) for item in events), default=0
            ),
            "max_closure_sec": max(
                (item["payload"].get("current_closure_sec", 0.0) for item in events), default=0.0
            ),
            "max_perclos_30s": max(perclos, default=None),
        },
        "drift_reason_counts": dict(
            Counter(item["payload"].get("reason", "UNKNOWN") for item in drifts)
        ),
        "finalized_incidents": len(incidents),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a DMS replay JSONL session")
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.log_path)
    output = args.output or args.log_path.with_suffix(".report.json")
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
