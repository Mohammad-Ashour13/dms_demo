from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from .data import balanced_hierarchical_weights
from .metrics import macro_and_worst_pr_auc
from .splits import assert_no_group_overlap, grouped_folds


BASE_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.03,
    "n_estimators": 1500,
    "num_leaves": 15,
    "max_depth": 5,
    "min_child_samples": 80,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.2,
    "reg_lambda": 1.0,
    "verbosity": -1,
}


def make_model(params: dict[str, Any], seed: int, n_jobs: int, use_gpu: bool):
    merged = dict(BASE_PARAMS)
    merged.update(params)
    merged.update(
        random_state=seed,
        bagging_seed=seed,
        feature_fraction_seed=seed,
        data_random_seed=seed,
        deterministic=not use_gpu,
        n_jobs=n_jobs,
    )
    if use_gpu:
        merged["device_type"] = "gpu"
    return lgb.LGBMClassifier(**merged)


def cross_validated_predictions(
    frame: pd.DataFrame,
    features: list[str],
    params: dict[str, Any],
    seed: int,
    n_splits: int,
    n_jobs: int,
    use_gpu: bool,
) -> tuple[np.ndarray, list[int], list[dict]]:
    oof = np.full(len(frame), np.nan, dtype=float)
    iterations: list[int] = []
    audit_rows: list[dict] = []
    for fold, (train_idx, valid_idx) in enumerate(grouped_folds(frame, n_splits, seed)):
        assert_no_group_overlap(frame, train_idx, valid_idx)
        train_part, valid_part = frame.iloc[train_idx], frame.iloc[valid_idx]
        model = make_model(params, seed + fold, n_jobs, use_gpu)
        effective_device = "gpu" if use_gpu else "cpu"
        fallback_error = ""
        try:
            model.fit(
                train_part[features], train_part["label"],
                sample_weight=balanced_hierarchical_weights(train_part.reset_index(drop=True)),
                eval_set=[(valid_part[features], valid_part["label"])],
                eval_metric="average_precision",
                callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
            )
        except lgb.basic.LightGBMError as exc:
            if not use_gpu:
                raise
            # Colab LightGBM wheels are not always GPU-enabled; use an explicit CPU fallback.
            effective_device = "cpu_fallback"
            fallback_error = str(exc)[:500]
            model = make_model(params, seed + fold, n_jobs, False)
            model.fit(
                train_part[features], train_part["label"],
                sample_weight=balanced_hierarchical_weights(train_part.reset_index(drop=True)),
                eval_set=[(valid_part[features], valid_part["label"])],
                eval_metric="average_precision",
                callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
            )
        oof[valid_idx] = model.predict_proba(valid_part[features])[:, 1]
        best_iteration = int(model.best_iteration_ or model.n_estimators_)
        iterations.append(best_iteration)
        audit_rows.append(
            {
                "seed": seed, "fold": fold,
                "train_rows": len(train_idx), "valid_rows": len(valid_idx),
                "train_groups": train_part["group_id"].nunique(),
                "valid_groups": valid_part["group_id"].nunique(),
                "overlap_groups": 0,
                "best_iteration": best_iteration,
                "requested_device": "gpu" if use_gpu else "cpu",
                "effective_device": effective_device,
                "device_fallback_error": fallback_error,
            }
        )
    if np.isnan(oof).any():
        raise AssertionError("OOF predictions are incomplete")
    return oof, iterations, audit_rows


def cv_objective_metrics(frame: pd.DataFrame, predictions: np.ndarray) -> dict[str, float]:
    macro, worst = macro_and_worst_pr_auc(frame, predictions)
    return {"macro_pr_auc": macro, "worst_dataset_pr_auc": worst}
