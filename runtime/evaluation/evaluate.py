from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ALLOWED_LABELS = {
    "AWAKE", "DROWSY_SIMULATED", "YAWN", "PROLONGED_EYE_CLOSURE",
    "BLINK", "DISTRACTION", "PHONE_USE", "EATING", "SMOKING", "UNSCORABLE",
}
POSITIVE_LABELS = {"DROWSY_SIMULATED", "PROLONGED_EYE_CLOSURE"}
NEGATIVE_LABELS = {"AWAKE", "DISTRACTION"}


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    label: str
    identifier: str


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2 * precision * recall, precision + recall)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return records


def _load_annotations(path: Path, session_id: str) -> list[Interval]:
    if not path.is_file():
        raise FileNotFoundError(path)
    intervals = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), 1):
            if not row.get("label"):
                continue
            label = str(row["label"]).strip().upper()
            if label not in ALLOWED_LABELS:
                raise ValueError(f"Unsupported annotation label at row {index}: {label}")
            if row.get("session_id") not in {None, "", session_id}:
                raise ValueError(f"Annotation row {index} belongs to another session")
            start, end = float(row["start_sec"]), float(row["end_sec"])
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                raise ValueError(f"Invalid annotation interval at row {index}: {start}, {end}")
            intervals.append(Interval(start, end, label, f"gt-{index:04d}"))
    if not intervals:
        raise ValueError("annotations.csv contains no intervals")
    positives = [item for item in intervals if item.label in POSITIVE_LABELS]
    negatives = [item for item in intervals if item.label in NEGATIVE_LABELS]
    for positive in positives:
        for negative in negatives:
            if min(positive.end, negative.end) > max(positive.start, negative.start):
                raise ValueError(
                    f"Positive and negative annotations overlap: {positive.identifier}, {negative.identifier}"
                )
    return intervals


def _contains(intervals: list[Interval], value: float) -> bool:
    return any(item.start <= value < item.end for item in intervals)


def _merge_intervals(intervals: list[Interval], gap: float, label="PREDICTED") -> list[Interval]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda item: (item.start, item.end))
    merged: list[Interval] = []
    start, end = ordered[0].start, ordered[0].end
    for item in ordered[1:]:
        if item.start - end <= gap:
            end = max(end, item.end)
        else:
            merged.append(Interval(start, end, label, f"pred-{len(merged)+1:04d}"))
            start, end = item.start, item.end
    merged.append(Interval(start, end, label, f"pred-{len(merged)+1:04d}"))
    return merged


def _segments(grid: np.ndarray, active: np.ndarray, step: float, gap: float, label: str) -> list[Interval]:
    raw: list[Interval] = []
    start = None
    previous = None
    for timestamp, is_active in zip(grid, active):
        if is_active and start is None:
            start = float(timestamp)
        if is_active:
            previous = float(timestamp)
        elif start is not None:
            raw.append(Interval(start, float(previous) + step, label, ""))
            start = previous = None
    if start is not None:
        raw.append(Interval(start, float(previous) + step, label, ""))
    return _merge_intervals(raw, gap, label)


def _match_episodes(
    truth: list[Interval], predicted: list[Interval], tolerance: float, minimum_overlap: float
) -> tuple[list[dict], list[Interval], list[Interval]]:
    candidates = []
    for truth_index, gt in enumerate(truth):
        for prediction_index, alert in enumerate(predicted):
            overlap = min(alert.end, gt.end + tolerance) - max(alert.start, gt.start - tolerance)
            if overlap >= minimum_overlap:
                candidates.append((overlap, truth_index, prediction_index))
    used_truth, used_predicted, matches = set(), set(), []
    for overlap, truth_index, prediction_index in sorted(candidates, reverse=True):
        if truth_index in used_truth or prediction_index in used_predicted:
            continue
        used_truth.add(truth_index)
        used_predicted.add(prediction_index)
        gt, alert = truth[truth_index], predicted[prediction_index]
        matches.append(
            {
                "ground_truth_id": gt.identifier,
                "ground_truth_label": gt.label,
                "ground_truth_start_sec": gt.start,
                "ground_truth_end_sec": gt.end,
                "alert_id": alert.identifier,
                "alert_start_sec": alert.start,
                "alert_end_sec": alert.end,
                "overlap_sec": overlap,
                "delay_sec": alert.start - gt.start,
            }
        )
    missed = [item for index, item in enumerate(truth) if index not in used_truth]
    false_alerts = [item for index, item in enumerate(predicted) if index not in used_predicted]
    return matches, missed, false_alerts


def _event_metrics(
    records: list[dict], annotations: list[Interval], kind: str, label: str,
    to_video_sec, tolerance: float,
) -> dict:
    event_times = set()
    for record in records:
        if record.get("stage") == "Events" and record.get("event") == "snapshot":
            for evidence in record.get("payload", {}).get("evidence", []):
                if evidence.get("kind") == kind:
                    monotonic = float(evidence.get("monotonic_sec", record["monotonic_sec"]))
                    event_times.add(round(float(to_video_sec(monotonic)), 3))
        elif (
            record.get("stage") == "BehaviorDetector"
            and record.get("event") == "evidence_published"
            and record.get("payload", {}).get("kind") == kind
            and record.get("payload", {}).get("details", {}).get("trigger") == "ONSET"
        ):
            event_times.add(round(float(to_video_sec(float(record["monotonic_sec"]))), 3))
    truth = [item for item in annotations if item.label == label]
    matched_truth, matched_events = set(), set()
    for event_index, timestamp in enumerate(sorted(event_times)):
        for truth_index, interval in enumerate(truth):
            if truth_index in matched_truth:
                continue
            if interval.start - tolerance <= timestamp <= interval.end + tolerance:
                matched_truth.add(truth_index)
                matched_events.add(event_index)
                break
    tp = len(matched_truth)
    fp = len(event_times) - len(matched_events)
    fn = len(truth) - tp
    precision, recall = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    return {
        "kind": kind, "ground_truth_events": len(truth), "detected_events": len(event_times),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": _f1(precision, recall),
    }


def _markdown(report: dict) -> str:
    timeline = report["timeline_metrics"]
    episode = report["episode_metrics"]
    warning = report["warning_metrics"]
    quality = report["quality"]
    gate = report["acceptance_gate"]
    lines = [
        "# DMS Personal Evaluation Report", "",
        f"- Session: `{report['session_id']}`",
        f"- Result: **{gate['status']}**",
        f"- Deployment mode: **{report['deployment_mode']}**", "",
        "## Primary DROWSY / CRITICAL metrics", "",
        "| Unit | Precision | Recall | F1 | FPR | Accuracy |", "|---|---:|---:|---:|---:|---:|",
        f"| Timeline 0.5s | {timeline['precision']:.3f} | {timeline['recall']:.3f} | {timeline['f1']:.3f} | {timeline['fpr']:.3f} | {timeline['accuracy']:.3f} |",
        f"| Episodes | {episode['precision']:.3f} | {episode['recall']:.3f} | {episode['f1']:.3f} | N/A | N/A |",
        "", "## Advisory FATIGUE_WARNING (not a positive drowsiness decision)", "",
        f"- Timeline Precision / Recall / FPR: {warning['precision']:.3f} / {warning['recall']:.3f} / {warning['fpr']:.3f}",
        "", "## Episode details", "",
        f"- Ground-truth episodes: {episode['ground_truth_episodes']}",
        f"- Matched episodes: {episode['true_positive_episodes']}",
        f"- Missed episodes: {episode['false_negative_episodes']}",
        f"- False alert episodes: {episode['false_positive_episodes']}",
        f"- False alerts per awake hour: {episode['false_alerts_per_awake_hour']:.3f}",
        f"- Median detection delay: {episode['median_delay_sec']}", "",
        "## Data quality", "",
        f"- Calibration ready: {quality['calibration_ready']}",
        f"- Valid feature-window ratio: {quality['valid_window_ratio']:.3f}",
        f"- Valid eye-observation ratio: {quality['valid_eye_observation_ratio']:.3f}",
        f"- Awake NORMAL ratio: {quality['awake_normal_ratio']:.3f}",
        f"- Annotation coverage ratio: {quality['annotation_coverage_ratio']:.3f}",
        f"- Annotated/scored samples: {timeline['scored_samples']}",
        f"- False CRITICAL episodes while awake: {quality['false_critical_episodes_awake']}",
        "", "## Event and object-behavior detection", "",
        "| Kind | Ground truth | Detected | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {item['kind']} | {item['ground_truth_events']} | {item['detected_events']} | "
        f"{item['precision']:.3f} | {item['recall']:.3f} | {item['f1']:.3f} |"
        for item in report["event_metrics"]
    )
    lines.extend([
        "",
        "Object-behavior rows are pilot metrics and are not part of the drowsiness acceptance gate.",
        "",
        "## Gate checks",
        "",
    ])
    lines.extend(f"- {'PASS' if value else 'FAIL'} — {name}" for name, value in gate["checks"].items())
    lines.extend([
        "", "## Scientific limitation", "",
        "Passing this report is PERSONAL_PILOT_PASS only. A single person simulating drowsiness does not establish multi-driver production generalization.",
    ])
    return "\n".join(lines) + "\n"


def evaluate_session(session_dir: Path) -> dict:
    session_dir = Path(session_dir)
    manifest_path = session_dir / "session_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "READY_FOR_ANNOTATION":
        raise ValueError(
            f"Session is not ready for evaluation: status={manifest.get('status')}"
        )
    if manifest.get("deployment_mode") != "SHADOW":
        raise ValueError("Personal evaluation accepts SHADOW sessions only")
    session_id = manifest["session_id"]
    config = manifest.get("evaluation_config", {})
    step = float(config.get("sampling_interval_sec", 0.5))
    tolerance = float(config.get("transition_tolerance_sec", 1.0))
    merge_gap = float(config.get("alert_merge_gap_sec", 2.0))
    minimum_overlap = float(config.get("minimum_overlap_sec", 0.5))
    positive_states = set(config.get("primary_positive_states", ["DROWSY", "CRITICAL"]))
    timestamps = pd.read_csv(session_dir / "frame_timestamps.csv")
    if timestamps.empty:
        raise ValueError("frame_timestamps.csv is empty")
    first_monotonic = float(timestamps.iloc[0]["monotonic_sec"])
    playback_times = timestamps["video_sec"].to_numpy(dtype=float)
    session_times = (
        timestamps["session_sec"].to_numpy(dtype=float)
        if "session_sec" in timestamps.columns
        else playback_times
    )

    def to_video_sec(monotonic_sec: float) -> float:
        relative = float(monotonic_sec) - first_monotonic
        return float(np.interp(relative, session_times, playback_times))

    duration = float(playback_times.max())
    annotations = _load_annotations(session_dir / "annotations.csv", session_id)
    records = _read_jsonl(session_dir / "telemetry.jsonl")
    decisions = [
        item for item in records
        if item.get("stage") == "Fusion" and item.get("event") == "decision"
    ]
    if not decisions:
        raise ValueError("No Fusion decisions found in telemetry.jsonl")
    decision_times = np.asarray(
        [to_video_sec(float(item["monotonic_sec"])) for item in decisions], dtype=float
    )
    order = np.argsort(decision_times)
    decision_times = decision_times[order]
    decision_states = np.asarray(
        [decisions[index].get("payload", {}).get("driver_state", "UNKNOWN") for index in order],
        dtype=object,
    )
    decision_reasons = [
        decisions[index].get("payload", {}).get("reason_codes", []) for index in order
    ]
    grid = np.arange(0.0, duration + step * 0.5, step)
    positions = np.searchsorted(decision_times, grid, side="right") - 1
    states = np.asarray(
        [decision_states[position] if position >= 0 else "UNKNOWN" for position in positions],
        dtype=object,
    )
    reasons = [decision_reasons[position] if position >= 0 else [] for position in positions]
    positives = [item for item in annotations if item.label in POSITIVE_LABELS]
    negatives = [item for item in annotations if item.label in NEGATIVE_LABELS]
    awake_intervals = [item for item in annotations if item.label == "AWAKE"]
    unscorable = [item for item in annotations if item.label == "UNSCORABLE"]
    truth = []
    annotated = []
    for timestamp in grid:
        annotated.append(_contains(annotations, float(timestamp)))
        if _contains(unscorable, float(timestamp)):
            truth.append("EXCLUDED")
        elif _contains(positives, float(timestamp)):
            truth.append("POSITIVE")
        elif _contains(negatives, float(timestamp)):
            truth.append("NEGATIVE")
        else:
            truth.append("EXCLUDED")
    truth = np.asarray(truth, dtype=object)
    annotated = np.asarray(annotated, dtype=bool)
    predicted = np.asarray([state in positive_states for state in states], dtype=bool)
    scored = truth != "EXCLUDED"
    actual_positive, actual_negative = truth == "POSITIVE", truth == "NEGATIVE"
    tp = int(np.sum(scored & actual_positive & predicted))
    fn = int(np.sum(scored & actual_positive & ~predicted))
    fp = int(np.sum(scored & actual_negative & predicted))
    tn = int(np.sum(scored & actual_negative & ~predicted))
    precision, recall = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    fpr = _safe_div(fp, fp + tn)
    timeline_metrics = {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": _f1(precision, recall),
        "fpr": fpr, "accuracy": _safe_div(tp + tn, tp + tn + fp + fn),
        "scored_samples": int(np.sum(scored)), "sampling_interval_sec": step,
    }
    warning_predicted = states == "FATIGUE_WARNING"
    warning_tp = int(np.sum(scored & actual_positive & warning_predicted))
    warning_fn = int(np.sum(scored & actual_positive & ~warning_predicted))
    warning_fp = int(np.sum(scored & actual_negative & warning_predicted))
    warning_tn = int(np.sum(scored & actual_negative & ~warning_predicted))
    warning_precision = _safe_div(warning_tp, warning_tp + warning_fp)
    warning_recall = _safe_div(warning_tp, warning_tp + warning_fn)
    warning_metrics = {
        "tp": warning_tp, "fp": warning_fp, "fn": warning_fn, "tn": warning_tn,
        "precision": warning_precision, "recall": warning_recall,
        "f1": _f1(warning_precision, warning_recall),
        "fpr": _safe_div(warning_fp, warning_fp + warning_tn),
    }
    # Unannotated and UNSCORABLE time cannot create either a correct or false alert.
    predicted_episodes = _segments(
        grid, predicted & scored, step, merge_gap, "DROWSY_OR_CRITICAL"
    )
    truth_episodes = _merge_intervals(positives, 0.0, "GROUND_TRUTH_DROWSY")
    matches, missed, false_alerts = _match_episodes(
        truth_episodes, predicted_episodes, tolerance, minimum_overlap
    )
    episode_precision = _safe_div(len(matches), len(matches) + len(false_alerts))
    episode_recall = _safe_div(len(matches), len(matches) + len(missed))
    delays = [float(item["delay_sec"]) for item in matches]
    awake_hours = sum(item.end - item.start for item in negatives) / 3600.0
    episode_metrics = {
        "ground_truth_episodes": len(truth_episodes),
        "predicted_episodes": len(predicted_episodes),
        "true_positive_episodes": len(matches),
        "false_positive_episodes": len(false_alerts),
        "false_negative_episodes": len(missed),
        "precision": episode_precision, "recall": episode_recall,
        "f1": _f1(episode_precision, episode_recall),
        "false_alerts_per_awake_hour": _safe_div(len(false_alerts), awake_hours),
        "median_delay_sec": float(np.median(delays)) if delays else None,
        "delays_sec": delays,
    }
    calibration_records = [
        item for item in records
        if item.get("stage") == "Calibration" and item.get("event") == "status"
    ]
    calibration_ready = any(item.get("payload", {}).get("status") == "READY" for item in calibration_records)
    windows = [
        item for item in records
        if item.get("stage") == "Window" and item.get("event") == "feature_snapshot"
    ]
    valid_windows = sum(bool(item.get("payload", {}).get("valid")) for item in windows)
    event_snapshots = [
        item for item in records
        if item.get("stage") == "Events" and item.get("event") == "snapshot"
    ]
    valid_eye_snapshots = sum(
        bool(item.get("payload", {}).get("eye_signal_valid"))
        for item in event_snapshots
    )
    awake_mask = np.asarray(
        [_contains(awake_intervals, float(timestamp)) for timestamp in grid], dtype=bool
    ) & (truth != "EXCLUDED")
    awake_samples = int(np.sum(awake_mask))
    awake_normal_samples = int(np.sum(awake_mask & (states == "NORMAL")))
    critical_awake = (states == "CRITICAL") & actual_negative
    false_critical = _segments(grid, critical_awake, step, merge_gap, "FALSE_CRITICAL")
    quality = {
        "calibration_ready": calibration_ready,
        "window_records": len(windows), "valid_window_records": valid_windows,
        "valid_window_ratio": _safe_div(valid_windows, len(windows)),
        "annotation_coverage_ratio": float(np.mean(annotated)) if len(annotated) else 0.0,
        "scorable_time_ratio": float(np.mean(scored)) if len(scored) else 0.0,
        "false_critical_episodes_awake": len(false_critical),
        "evaluation_recorder_dropped_frames": int(manifest.get("dropped_frames", 0)),
        "event_snapshot_records": len(event_snapshots),
        "valid_eye_snapshot_records": valid_eye_snapshots,
        "valid_eye_observation_ratio": _safe_div(valid_eye_snapshots, len(event_snapshots)),
        "awake_samples": awake_samples,
        "awake_normal_samples": awake_normal_samples,
        "awake_normal_ratio": _safe_div(awake_normal_samples, awake_samples),
    }
    event_metrics = [
        _event_metrics(records, annotations, "BLINK", "BLINK", to_video_sec, tolerance),
        _event_metrics(records, annotations, "YAWNING", "YAWN", to_video_sec, tolerance),
        _event_metrics(
            records, annotations, "PROLONGED_EYE_CLOSURE",
            "PROLONGED_EYE_CLOSURE", to_video_sec, tolerance,
        ),
        _event_metrics(records, annotations, "PHONE_USE", "PHONE_USE", to_video_sec, tolerance),
        _event_metrics(records, annotations, "EATING", "EATING", to_video_sec, tolerance),
        _event_metrics(records, annotations, "SMOKING", "SMOKING", to_video_sec, tolerance),
    ]
    event_by_kind = {item["kind"]: item for item in event_metrics}
    median_delay = episode_metrics["median_delay_sec"]
    checks = {
        "episode_precision_at_least_0.75": episode_precision >= 0.75,
        "episode_recall_at_least_0.90": episode_recall >= 0.90,
        "episode_f1_at_least_0.80": episode_metrics["f1"] >= 0.80,
        "awake_time_fpr_below_0.10": fpr < 0.10,
        "median_delay_at_most_2s": median_delay is not None and median_delay <= 2.0,
        "zero_false_critical_while_awake": len(false_critical) == 0,
        "calibration_ready": calibration_ready,
        "valid_window_ratio_at_least_0.90": quality["valid_window_ratio"] >= 0.90,
        "valid_eye_observation_ratio_at_least_0.90": (
            quality["valid_eye_observation_ratio"] >= 0.90
        ),
        "awake_normal_ratio_at_least_0.90": quality["awake_normal_ratio"] >= 0.90,
        "blink_recall_at_least_0.85_when_annotated": (
            event_by_kind["BLINK"]["ground_truth_events"] == 0
            or event_by_kind["BLINK"]["recall"] >= 0.85
        ),
        "prolonged_closure_recall_is_1_when_annotated": (
            event_by_kind["PROLONGED_EYE_CLOSURE"]["ground_truth_events"] == 0
            or event_by_kind["PROLONGED_EYE_CLOSURE"]["recall"] == 1.0
        ),
        "annotation_coverage_at_least_0.95": quality["annotation_coverage_ratio"] >= 0.95,
    }
    report = {
        "schema_version": "1.0.0", "session_id": session_id,
        "duration_sec": duration,
        "deployment_mode": manifest.get("deployment_mode", "UNKNOWN"),
        "model_version": manifest.get("model_version"),
        "timeline_metrics": timeline_metrics, "episode_metrics": episode_metrics,
        "warning_metrics": warning_metrics,
        "quality": quality,
        "event_metrics": event_metrics,
        "acceptance_gate": {
            "status": "SESSION_PASS" if all(checks.values()) else "SESSION_FAIL",
            "checks": checks,
        },
    }
    timeline = pd.DataFrame(
        {
            "video_sec": grid, "ground_truth": truth, "fusion_state": states,
            "predicted_drowsy": predicted.astype(int), "scored": scored.astype(int),
        }
    )
    timeline.to_csv(session_dir / "evaluation_timeline.csv", index=False)
    errors = []
    errors.extend(
        {
            "error_type": "FALSE_NEGATIVE_EPISODE", "id": item.identifier,
            "start_sec": item.start, "end_sec": item.end, "label": item.label,
            "reason_codes": "NO_DROWSY_OR_CRITICAL_ALERT",
        }
        for item in missed
    )
    errors.extend(
        {
            "error_type": "FALSE_POSITIVE_EPISODE", "id": item.identifier,
            "start_sec": item.start, "end_sec": item.end, "label": item.label,
            "reason_codes": "|".join(
                sorted(
                    {
                        str(reason)
                        for timestamp, row_reasons in zip(grid, reasons)
                        if item.start <= timestamp < item.end
                        for reason in row_reasons
                    }
                )
            ),
        }
        for item in false_alerts
    )
    pd.DataFrame(
        errors,
        columns=("error_type", "id", "start_sec", "end_sec", "label", "reason_codes"),
    ).to_csv(
        session_dir / "evaluation_errors.csv", index=False
    )
    pd.DataFrame(matches).to_csv(session_dir / "episode_matches.csv", index=False)
    pd.DataFrame(report["event_metrics"]).to_csv(session_dir / "event_metrics.csv", index=False)
    (session_dir / "evaluation_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (session_dir / "evaluation_report.md").write_text(_markdown(report), encoding="utf-8")
    manifest["evaluation_status"] = report["acceptance_gate"]["status"]
    manifest["evaluation_metrics"] = "evaluation_metrics.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def aggregate_reports(reports: list[dict], output_dir: Path) -> dict:
    if not reports:
        raise ValueError("At least one session report is required")
    totals = {
        key: sum(report["timeline_metrics"][key] for report in reports)
        for key in ("tp", "fp", "fn", "tn")
    }
    precision = _safe_div(totals["tp"], totals["tp"] + totals["fp"])
    recall = _safe_div(totals["tp"], totals["tp"] + totals["fn"])
    episode_tp = sum(report["episode_metrics"]["true_positive_episodes"] for report in reports)
    episode_fp = sum(report["episode_metrics"]["false_positive_episodes"] for report in reports)
    episode_fn = sum(report["episode_metrics"]["false_negative_episodes"] for report in reports)
    episode_precision = _safe_div(episode_tp, episode_tp + episode_fp)
    episode_recall = _safe_div(episode_tp, episode_tp + episode_fn)
    delays = [delay for report in reports for delay in report["episode_metrics"]["delays_sec"]]
    fpr = _safe_div(totals["fp"], totals["fp"] + totals["tn"])
    minimum_session_recall = min(report["episode_metrics"]["recall"] for report in reports)
    total_duration = sum(float(report.get("duration_sec", 0.0)) for report in reports)
    total_awake_sec = sum(
        (report["timeline_metrics"]["tn"] + report["timeline_metrics"]["fp"])
        * report["timeline_metrics"]["sampling_interval_sec"]
        for report in reports
    )
    total_truth_episodes = episode_tp + episode_fn
    total_yawns = sum(
        next(
            (item["ground_truth_events"] for item in report["event_metrics"] if item["kind"] == "YAWNING"),
            0,
        )
        for report in reports
    )
    total_closures = sum(
        next(
            (
                item["ground_truth_events"] for item in report["event_metrics"]
                if item["kind"] == "PROLONGED_EYE_CLOSURE"
            ),
            0,
        )
        for report in reports
    )
    total_blinks = sum(
        next(
            (item["ground_truth_events"] for item in report["event_metrics"] if item["kind"] == "BLINK"),
            0,
        )
        for report in reports
    )
    blink_tp = sum(
        next((item["tp"] for item in report["event_metrics"] if item["kind"] == "BLINK"), 0)
        for report in reports
    )
    blink_fn = sum(
        next((item["fn"] for item in report["event_metrics"] if item["kind"] == "BLINK"), 0)
        for report in reports
    )
    closure_tp = sum(
        next(
            (item["tp"] for item in report["event_metrics"] if item["kind"] == "PROLONGED_EYE_CLOSURE"),
            0,
        )
        for report in reports
    )
    closure_fn = sum(
        next(
            (item["fn"] for item in report["event_metrics"] if item["kind"] == "PROLONGED_EYE_CLOSURE"),
            0,
        )
        for report in reports
    )
    awake_samples = sum(report["quality"].get("awake_samples", 0) for report in reports)
    awake_normal_samples = sum(
        report["quality"].get("awake_normal_samples", 0) for report in reports
    )
    checks = {
        "at_least_3_sessions": len(reports) >= 3,
        "at_least_45_minutes_total": total_duration >= 45 * 60,
        "at_least_30_minutes_awake": total_awake_sec >= 30 * 60,
        "at_least_20_drowsiness_episodes": total_truth_episodes >= 20,
        "at_least_10_yawns": total_yawns >= 10,
        "at_least_10_prolonged_closures": total_closures >= 10,
        "at_least_30_blinks": total_blinks >= 30,
        "blink_recall_at_least_0.85": _safe_div(blink_tp, blink_tp + blink_fn) >= 0.85,
        "prolonged_closure_recall_is_1": _safe_div(closure_tp, closure_tp + closure_fn) == 1.0,
        "episode_precision_at_least_0.75": episode_precision >= 0.75,
        "episode_recall_at_least_0.90": episode_recall >= 0.90,
        "episode_f1_at_least_0.80": _f1(episode_precision, episode_recall) >= 0.80,
        "awake_time_fpr_below_0.10": fpr < 0.10,
        "median_delay_at_most_2s": bool(delays) and float(np.median(delays)) <= 2.0,
        "zero_false_critical_while_awake": sum(
            report["quality"]["false_critical_episodes_awake"] for report in reports
        ) == 0,
        "each_session_recall_at_least_0.80": minimum_session_recall >= 0.80,
        "all_calibrations_ready": all(report["quality"]["calibration_ready"] for report in reports),
        "all_valid_window_ratios_at_least_0.90": all(
            report["quality"]["valid_window_ratio"] >= 0.90 for report in reports
        ),
        "all_valid_eye_ratios_at_least_0.90": all(
            report["quality"].get("valid_eye_observation_ratio", 0.0) >= 0.90
            for report in reports
        ),
        "awake_normal_ratio_at_least_0.90": (
            _safe_div(awake_normal_samples, awake_samples) >= 0.90
        ),
    }
    aggregate = {
        "schema_version": "1.0.0", "sessions": [report["session_id"] for report in reports],
        "timeline": {
            **totals, "precision": precision, "recall": recall, "f1": _f1(precision, recall),
            "fpr": fpr, "accuracy": _safe_div(totals["tp"] + totals["tn"], sum(totals.values())),
        },
        "episodes": {
            "tp": episode_tp, "fp": episode_fp, "fn": episode_fn,
            "precision": episode_precision, "recall": episode_recall,
            "f1": _f1(episode_precision, episode_recall),
            "median_delay_sec": float(np.median(delays)) if delays else None,
            "minimum_session_recall": minimum_session_recall,
        },
        "protocol_counts": {
            "total_duration_sec": total_duration, "awake_duration_sec": total_awake_sec,
            "drowsiness_episodes": total_truth_episodes,
            "yawns": total_yawns, "prolonged_eye_closures": total_closures,
            "blinks": total_blinks,
            "blink_recall": _safe_div(blink_tp, blink_tp + blink_fn),
            "prolonged_closure_recall": _safe_div(closure_tp, closure_tp + closure_fn),
            "awake_normal_ratio": _safe_div(awake_normal_samples, awake_samples),
        },
        "acceptance_gate": {
            "status": "PERSONAL_PILOT_PASS" if all(checks.values()) else "PERSONAL_PILOT_FAIL",
            "checks": checks,
        },
        "limitation": "Single-person simulated-drowsiness pilot; not production validation.",
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "personal_acceptance_metrics.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# DMS Personal Acceptance", "",
        f"- Sessions: {len(reports)}",
        f"- Status: **{aggregate['acceptance_gate']['status']}**",
        f"- Episode Precision / Recall / F1: **{episode_precision:.3f} / {episode_recall:.3f} / {_f1(episode_precision, episode_recall):.3f}**",
        f"- Timeline FPR: **{fpr:.3f}**",
        f"- Median delay: **{aggregate['episodes']['median_delay_sec']}**", "",
        "## Gate checks", "",
    ]
    lines.extend(
        f"- {'PASS' if value else 'FAIL'} — {name}"
        for name, value in checks.items()
    )
    lines.extend([
        "", "This is a controlled personal pilot, not multi-driver production evidence.", "",
    ])
    (output_dir / "personal_acceptance_report.md").write_text("\n".join(lines), encoding="utf-8")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate annotated DMS camera sessions")
    parser.add_argument("session_dirs", nargs="+", type=Path)
    parser.add_argument("--aggregate-output", type=Path)
    args = parser.parse_args()
    reports = [evaluate_session(path) for path in args.session_dirs]
    for path, report in zip(args.session_dirs, reports):
        print(path, report["acceptance_gate"]["status"])
    if len(reports) > 1:
        output = args.aggregate_output or Path(args.session_dirs[0]).parent / "personal_acceptance"
        aggregate = aggregate_reports(reports, output)
        print(output, aggregate["acceptance_gate"]["status"])


if __name__ == "__main__":
    main()
