from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y_true, probabilities, threshold: float) -> dict[str, float | int | list]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def threshold_sweep(
    y_true,
    probabilities,
    recall_target: float,
    precision_target: float,
    points: int = 1001,
) -> tuple[float, pd.DataFrame, bool]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pr_auc = float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    roc_auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    rows = []
    for threshold in np.linspace(0.0, 1.0, points):
        pred = p >= threshold
        positive = y == 1
        negative = ~positive
        tp = int(np.sum(pred & positive))
        fp = int(np.sum(pred & negative))
        fn = int(np.sum(~pred & positive))
        tn = int(np.sum(~pred & negative))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows.append(
            {
                "threshold": float(threshold), "precision": precision, "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                "fpr": fp / (fp + tn) if fp + tn else 0.0,
                "pr_auc": pr_auc, "roc_auc": roc_auc,
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            }
        )
    table = pd.DataFrame(rows)
    feasible = table[(table["recall"] >= recall_target) & (table["precision"] >= precision_target)]
    if not feasible.empty:
        chosen = feasible.sort_values(["f1", "fpr", "threshold"], ascending=[False, True, False]).iloc[0]
        return float(chosen["threshold"]), table, True
    # Explicit fallback: maximize F1, then recall. The manifest records that constraints failed.
    chosen = table.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]
    return float(chosen["threshold"]), table, False


def per_dataset_metrics(frame: pd.DataFrame, probabilities, threshold: float) -> pd.DataFrame:
    work = frame[["dataset_name", "label"]].copy()
    work["probability"] = probabilities
    rows = []
    for dataset, part in work.groupby("dataset_name", sort=True):
        metrics = binary_metrics(part["label"], part["probability"], threshold)
        metrics.pop("confusion_matrix", None)
        metrics["dataset_name"] = dataset
        metrics["n_windows"] = len(part)
        metrics["positive_rate"] = float(part["label"].mean())
        rows.append(metrics)
    return pd.DataFrame(rows)


def macro_and_worst_pr_auc(frame: pd.DataFrame, probabilities) -> tuple[float, float]:
    table = per_dataset_metrics(frame, probabilities, 0.5)
    values = table["pr_auc"].dropna()
    if values.empty:
        return 0.0, 0.0
    return float(values.mean()), float(values.min())
