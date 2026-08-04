from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn

from .bundle import export_bundle
from .calibration import fit_platt
from .config import ExperimentConfig
from .data import (
    balanced_hierarchical_weights,
    exclude_safee,
    group_safe_smoke_sample,
    prepare_frame,
    validate_schema,
)
from .features import make_feature_sets, ranked_feature_names
from .io import csv_header, read_windows, sha256_file, write_json
from .metrics import (
    binary_metrics,
    macro_and_worst_pr_auc,
    per_dataset_metrics,
    threshold_sweep,
)
from .modeling import cross_validated_predictions, make_model
from .splits import grouped_folds


def _run_id(mode: str) -> str:
    return datetime.now(timezone.utc).strftime(f"%Y%m%dT%H%M%SZ_{mode}")


def _evaluate_candidate(frame, features, params, seed, folds, config):
    predictions, iterations, audits = cross_validated_predictions(
        frame, features, params, seed, folds, config.n_jobs, config.use_gpu
    )
    macro, worst = macro_and_worst_pr_auc(frame, predictions)
    threshold, _, feasible = threshold_sweep(
        frame["label"], predictions, config.recall_target, config.precision_target, points=301
    )
    metrics = binary_metrics(frame["label"], predictions, threshold)
    return {
        "predictions": predictions,
        "iterations": iterations,
        "audits": audits,
        "macro_pr_auc": macro,
        "worst_dataset_pr_auc": worst,
        "constraints_met": feasible,
        "metrics": metrics,
    }


def _trial_params(trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 31),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 40, 180),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 3.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        "n_estimators": 2000,
    }


def _tune(frame, feature_sets, top_counts, config, folds):
    import optuna

    rows: list[dict] = []
    fit_rows: list[dict] = []
    best: dict[int, tuple[float, dict]] = {}
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for count in top_counts:
        def objective(trial):
            params = _trial_params(trial)
            result = _evaluate_candidate(
                frame, feature_sets[count], params, config.random_state, folds, config
            )
            # Feasibility dominates, followed by macro and worst-domain performance.
            value = (
                (1.0 if result["constraints_met"] else 0.0)
                + result["macro_pr_auc"]
                + 0.20 * result["worst_dataset_pr_auc"]
            )
            trial.set_user_attr("macro_pr_auc", result["macro_pr_auc"])
            trial.set_user_attr("worst_dataset_pr_auc", result["worst_dataset_pr_auc"])
            trial.set_user_attr("constraints_met", result["constraints_met"])
            for name in ("precision", "recall", "f1", "fpr", "pr_auc", "roc_auc"):
                trial.set_user_attr(name, result["metrics"][name])
            for audit in result["audits"]:
                fit_rows.append(
                    {
                        "stage": "B", "feature_count": count,
                        "trial": trial.number, **audit,
                    }
                )
            return value

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=config.optuna_trials, show_progress_bar=False)
        params = dict(study.best_trial.params)
        params["n_estimators"] = 2000
        best[count] = (float(study.best_value), params)
        for trial in study.trials:
            rows.append(
                {
                    "feature_count": count,
                    "trial": trial.number,
                    "objective": trial.value,
                    **trial.user_attrs,
                    "params": json.dumps(trial.params, sort_keys=True),
                }
            )
    return best, pd.DataFrame(rows), pd.DataFrame(fit_rows)


def _lodo(frame, features, params, seed, n_estimators, config):
    rows = []
    predictions = []
    fit_rows = []
    for dataset in sorted(frame["dataset_name"].unique()):
        train = frame[frame["dataset_name"] != dataset].reset_index(drop=True)
        held = frame[frame["dataset_name"] == dataset].reset_index(drop=True)
        if train["label"].nunique() < 2 or held["label"].nunique() < 2:
            rows.append({"held_out_dataset": dataset, "status": "NOT_BINARY", "n_windows": len(held)})
            continue
        fit_params = dict(params, n_estimators=n_estimators)
        model = make_model(fit_params, seed, config.n_jobs, config.use_gpu)
        effective_device = "gpu" if config.use_gpu else "cpu"
        fallback_error = ""
        try:
            model.fit(
                train[features], train["label"],
                sample_weight=balanced_hierarchical_weights(train),
                callbacks=[lgb.log_evaluation(0)],
            )
        except lgb.basic.LightGBMError as exc:
            effective_device = "cpu_fallback"
            fallback_error = str(exc)[:500]
            model = make_model(fit_params, seed, config.n_jobs, False)
            model.fit(
                train[features], train["label"],
                sample_weight=balanced_hierarchical_weights(train),
                callbacks=[lgb.log_evaluation(0)],
            )
        prob = model.predict_proba(held[features])[:, 1]
        macro, _ = macro_and_worst_pr_auc(held, prob)
        metric = binary_metrics(held["label"], prob, 0.5)
        metric.pop("confusion_matrix", None)
        metric.update({
            "held_out_dataset": dataset, "status": "OK", "n_windows": len(held),
            "pr_auc": macro, "positive_rate": float(held["label"].mean()),
        })
        rows.append(metric)
        pred = held[["dataset_name", "subject_id", "video_id", "label"]].copy()
        pred["lodo_probability"] = prob
        pred["held_out_dataset"] = dataset
        predictions.append(pred)
        fit_rows.append(
            {
                "stage": "LODO", "feature_count": len(features), "trial": None,
                "seed": seed, "fold": None, "held_out_dataset": dataset,
                "train_rows": len(train), "valid_rows": len(held),
                "train_groups": train["group_id"].nunique(),
                "valid_groups": held["group_id"].nunique(), "overlap_groups": 0,
                "best_iteration": n_estimators,
                "requested_device": "gpu" if config.use_gpu else "cpu",
                "effective_device": effective_device,
                "device_fallback_error": fallback_error,
            }
        )
    return (
        pd.DataFrame(rows),
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
        pd.DataFrame(fit_rows),
    )


def _decision_report(manifest: dict, comparison: pd.DataFrame, test_metrics: dict) -> str:
    status = manifest["bundle_status"]
    def metric(name: str) -> str:
        value = test_metrics.get(name)
        return "N/A" if value is None else f"{value:.3f}"

    lines = [
        "# LightGBM DMS Decision Report",
        "",
        f"- Bundle status: **{status}**",
        f"- Selected feature count: **{manifest['selected_feature_count']}**",
        f"- Selected threshold (OOF only): **{manifest['operating_threshold']:.6f}**",
        f"- OOF constraints met: **{manifest['oof_constraints_met']}**",
        f"- Executed model fits: **{manifest['execution_summary']['actual_total']} / {manifest['execution_summary']['planned_total']}**",
        f"- Test Precision / Recall / F1: **{metric('precision')} / {metric('recall')} / {metric('f1')}**",
        f"- Test FPR / PR-AUC: **{metric('fpr')} / {metric('pr_auc')}**",
        "",
        "## Decision",
        "",
    ]
    if status.startswith("CANDIDATE"):
        lines.append(
            "The window classifier passed the offline gate. It is still a candidate until replay and Raspberry incident-level acceptance pass; integrate it only through Fusion."
        )
    else:
        lines.append(
            "The offline gate failed. Keep the bundle in telemetry-only mode, inspect per-dataset/LODO errors, and do not represent it as production-ready."
        )
    lines += [
        "",
        "## Scientific caveat",
        "",
        "Labels are video-level weak labels. Window metrics therefore measure agreement with inherited labels, not frame-level clinical drowsiness ground truth.",
        "",
        "## Experiment comparison",
        "",
        "```text",
        comparison.to_string(index=False),
        "```",
        "",
    ]
    return "\n".join(lines)


def run_experiment(config: ExperimentConfig) -> Path:
    config.validate()
    run_dir = Path(config.output_path) / _run_id(config.run_mode)
    run_dir.mkdir(parents=True, exist_ok=False)

    train_header = csv_header(config.train_windows_path)
    test_header = csv_header(config.test_windows_path)
    validate_schema(train_header, test_header)
    decisions = pd.read_csv(config.feature_decisions_path)
    ranked = ranked_feature_names(decisions, set(train_header) & set(test_header))
    feature_sets = make_feature_sets(ranked, config.feature_counts)
    all_features = feature_sets[max(feature_sets)]
    meta = [
        "label", "dataset_name", "subject_id", "video_id",
        *[c for c in ("window_start_ms", "window_end_ms") if c in train_header],
    ]
    train_raw = read_windows(config.train_windows_path, list(dict.fromkeys(meta + all_features)))
    test_raw = read_windows(config.test_windows_path, list(dict.fromkeys(meta + all_features)))
    train = prepare_frame(train_raw, all_features)
    test = prepare_frame(test_raw, all_features)
    train, safee_train = exclude_safee(train, config.safee_name)
    test_main, safee_test = exclude_safee(test, config.safee_name)
    train_test_overlap = sorted(set(train["group_id"]) & set(test_main["group_id"]))
    if train_test_overlap:
        raise ValueError(f"Fixed test overlaps fitting groups: {train_test_overlap[:20]}")
    if config.run_mode == "smoke":
        train = group_safe_smoke_sample(train, config.smoke_max_groups, config.random_state)
    folds = config.n_folds_smoke if config.run_mode == "smoke" else config.n_folds_full

    comparison_rows = []
    fit_audit_rows: list[dict] = []
    stage_a_results = {}
    for count, features in feature_sets.items():
        result = _evaluate_candidate(train, features, {}, config.random_state, folds, config)
        stage_a_results[count] = result
        fit_audit_rows.extend(
            {"stage": "A", "feature_count": count, "trial": None, **audit}
            for audit in result["audits"]
        )
        comparison_rows.append(
            {
                "stage": "A", "feature_count": count,
                "macro_pr_auc": result["macro_pr_auc"],
                "worst_dataset_pr_auc": result["worst_dataset_pr_auc"],
                "precision": result["metrics"]["precision"],
                "recall": result["metrics"]["recall"],
                "f1": result["metrics"]["f1"],
                "constraints_met": result["constraints_met"],
            }
        )
    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["constraints_met", "macro_pr_auc", "worst_dataset_pr_auc", "feature_count"],
        ascending=[False, False, False, True],
    )
    top_counts = comparison.head(2)["feature_count"].astype(int).tolist()

    selected_count = int(comparison.iloc[0]["feature_count"])
    selected_params: dict = {}
    tuning = pd.DataFrame()
    if config.run_mode == "full":
        tuned, tuning, stage_b_fit_audit = _tune(
            train, feature_sets, top_counts, config, folds
        )
        fit_audit_rows.extend(stage_b_fit_audit.to_dict("records"))
        tuned_rows = []
        for count in top_counts:
            best_trial = tuning[tuning["feature_count"].eq(count)].sort_values("objective", ascending=False).iloc[0]
            tuned_rows.append(
                {
                    "stage": "B", "feature_count": count,
                    "macro_pr_auc": best_trial["macro_pr_auc"],
                    "worst_dataset_pr_auc": best_trial["worst_dataset_pr_auc"],
                    "precision": best_trial["precision"],
                    "recall": best_trial["recall"],
                    "f1": best_trial["f1"],
                    "constraints_met": bool(best_trial["constraints_met"]),
                }
            )
        comparison = pd.concat([comparison, pd.DataFrame(tuned_rows)], ignore_index=True)
        ranked_tuned = sorted(
            ((score, count, params) for count, (score, params) in tuned.items()),
            key=lambda row: (-row[0], row[1]),
        )
        best_score = ranked_tuned[0][0]
        # If scientific performance is practically tied (<0.01), prefer the smaller runtime contract.
        near_ties = [row for row in ranked_tuned if best_score - row[0] < 0.01]
        _, selected_count, selected_params = min(near_ties, key=lambda row: row[1])

    selected_features = feature_sets[selected_count]
    active_seeds = [config.random_state] if config.run_mode == "smoke" else config.seeds
    seed_predictions, best_iterations, split_audits, seed_rows = [], [], [], []
    split_manifest_rows = []
    for seed in active_seeds:
        result = _evaluate_candidate(train, selected_features, selected_params, seed, folds, config)
        seed_predictions.append(result["predictions"])
        best_iterations.extend(result["iterations"])
        split_audits.extend(result["audits"])
        fit_audit_rows.extend(
            {"stage": "C", "feature_count": selected_count, "trial": None, **audit}
            for audit in result["audits"]
        )
        for fold, (_, valid_idx) in enumerate(grouped_folds(train, folds, seed)):
            for group_id in sorted(train.iloc[valid_idx]["group_id"].unique()):
                split_manifest_rows.append({"seed": seed, "fold": fold, "group_id": group_id})
        seed_rows.append(
            {
                "seed": seed, "macro_pr_auc": result["macro_pr_auc"],
                "worst_dataset_pr_auc": result["worst_dataset_pr_auc"],
                **{k: result["metrics"][k] for k in ("precision", "recall", "f1", "fpr", "pr_auc", "roc_auc")},
            }
        )
    comparison = pd.concat(
        [comparison, pd.DataFrame([
            {"stage": "C", "feature_count": selected_count,
             "macro_pr_auc": row["macro_pr_auc"],
             "worst_dataset_pr_auc": row["worst_dataset_pr_auc"],
             "precision": row["precision"], "recall": row["recall"], "f1": row["f1"],
             "constraints_met": row["precision"] >= config.precision_target and row["recall"] >= config.recall_target,
             "seed": row["seed"]}
            for row in seed_rows
        ])],
        ignore_index=True,
    )
    oof_raw = np.mean(seed_predictions, axis=0)
    calibration = fit_platt(oof_raw, train["label"].to_numpy())
    oof_calibrated = calibration.apply(oof_raw)
    threshold, sweep, constraints_met = threshold_sweep(
        train["label"], oof_calibrated, config.recall_target, config.precision_target
    )
    oof_metrics = binary_metrics(train["label"], oof_calibrated, threshold)
    oof_per_dataset = per_dataset_metrics(train, oof_calibrated, threshold)
    n_estimators = max(20, int(np.median(best_iterations)))

    if config.run_mode == "full":
        lodo_metrics, lodo_predictions, lodo_fit_audit = _lodo(
            train, selected_features, selected_params, config.random_state, n_estimators, config
        )
        fit_audit_rows.extend(lodo_fit_audit.to_dict("records"))
    else:
        lodo_metrics = pd.DataFrame([{"status": "NOT_RUN_IN_SMOKE"}])
        lodo_predictions = pd.DataFrame()

    final_params = dict(selected_params, n_estimators=n_estimators)
    final_model = make_model(final_params, config.random_state, config.n_jobs, config.use_gpu)
    final_effective_device = "gpu" if config.use_gpu else "cpu"
    final_fallback_error = ""
    try:
        final_model.fit(
            train[selected_features], train["label"],
            sample_weight=balanced_hierarchical_weights(train),
            callbacks=[lgb.log_evaluation(0)],
        )
    except lgb.basic.LightGBMError as exc:
        final_effective_device = "cpu_fallback"
        final_fallback_error = str(exc)[:500]
        final_model = make_model(final_params, config.random_state, config.n_jobs, False)
        final_model.fit(
            train[selected_features], train["label"],
            sample_weight=balanced_hierarchical_weights(train),
            callbacks=[lgb.log_evaluation(0)],
        )
    fit_audit_rows.append(
        {
            "stage": "FINAL", "feature_count": selected_count, "trial": None,
            "seed": config.random_state, "fold": None,
            "train_rows": len(train), "valid_rows": 0,
            "train_groups": train["group_id"].nunique(), "valid_groups": 0,
            "overlap_groups": 0, "best_iteration": n_estimators,
            "requested_device": "gpu" if config.use_gpu else "cpu",
            "effective_device": final_effective_device,
            "device_fallback_error": final_fallback_error,
        }
    )

    if config.run_mode == "full":
        test_raw_probability = final_model.predict_proba(test_main[selected_features])[:, 1]
        test_probability = calibration.apply(test_raw_probability)
        test_metrics = binary_metrics(test_main["label"], test_probability, threshold)
        test_per_dataset = per_dataset_metrics(test_main, test_probability, threshold)
        lodo_ok = bool(
            not lodo_metrics.empty
            and (lodo_metrics.get("status") == "OK").all()
            and (lodo_metrics["pr_auc"] >= lodo_metrics["positive_rate"]).all()
        )
        test_domain_ok = bool(
            (test_per_dataset["pr_auc"].fillna(-1) >= test_per_dataset["positive_rate"]).all()
        )
        gate_passed = (
            constraints_met
            and test_metrics["precision"] >= config.precision_target
            and test_metrics["recall"] >= config.recall_target
            and test_metrics["f1"] >= 0.80
            and test_metrics["fpr"] < 0.10
            and lodo_ok
            and test_domain_ok
        )
        bundle_status = "CANDIDATE_PENDING_RUNTIME_ACCEPTANCE" if gate_passed else "EXPERIMENTAL"
    else:
        test_raw_probability = np.asarray([], dtype=float)
        test_probability = np.asarray([], dtype=float)
        test_metrics = {name: None for name in ("precision", "recall", "f1", "fpr", "pr_auc", "roc_auc")}
        test_metrics.update({"status": "NOT_RUN_IN_SMOKE", "confusion_matrix": []})
        test_per_dataset = pd.DataFrame([{"status": "NOT_RUN_IN_SMOKE"}])
        gate_passed = False
        lodo_ok = False
        test_domain_ok = False
        bundle_status = "EXPERIMENTAL_SMOKE"

    id_cols = [c for c in ("dataset_name", "subject_id", "video_id", "window_start_ms", "window_end_ms", "label") if c in train]
    oof_out = train[id_cols].copy()
    oof_out["raw_probability"] = oof_raw
    oof_out["calibrated_probability"] = oof_calibrated
    oof_out["prediction"] = (oof_calibrated >= threshold).astype(int)
    if config.run_mode == "full":
        test_out = test_main[id_cols].copy()
        test_out["raw_probability"] = test_raw_probability
        test_out["calibrated_probability"] = test_probability
        test_out["prediction"] = (test_probability >= threshold).astype(int)
        errors = test_out[test_out["label"] != test_out["prediction"]].copy()
        errors["error_type"] = np.where(errors["prediction"].eq(1), "FP", "FN")
    else:
        test_out = pd.DataFrame(columns=id_cols + ["raw_probability", "calibrated_probability", "prediction"])
        errors = pd.DataFrame(columns=list(test_out.columns) + ["error_type"])

    safee_stress = {"status": "NOT_PRESENT", "n_windows": 0}
    if config.run_mode == "full" and not safee_test.empty:
        safee_prob = calibration.apply(final_model.predict_proba(safee_test[selected_features])[:, 1])
        safee_stress = {
            "status": "POSITIVE_STRESS_ONLY", "n_windows": len(safee_test),
            "mean_probability": float(np.mean(safee_prob)),
            "positive_rate": float(np.mean(safee_prob >= threshold)),
        }

    fit_audit = pd.DataFrame(fit_audit_rows)
    actual_by_stage = {
        str(stage): int(count)
        for stage, count in fit_audit["stage"].value_counts().sort_index().items()
    }
    planned_by_stage = {
        "A": len(feature_sets) * folds,
        "B": len(top_counts) * config.optuna_trials * folds if config.run_mode == "full" else 0,
        "C": len(active_seeds) * folds,
        "LODO": train["dataset_name"].nunique() if config.run_mode == "full" else 0,
        "FINAL": 1,
    }
    execution_summary = {
        "planned_by_stage": planned_by_stage,
        "actual_by_stage": actual_by_stage,
        "planned_total": int(sum(planned_by_stage.values())),
        "actual_total": int(len(fit_audit)),
        "complete": int(len(fit_audit)) == int(sum(planned_by_stage.values())),
        "effective_device_counts": {
            str(device): int(count)
            for device, count in fit_audit["effective_device"].value_counts().items()
        },
        "note": "One Optuna trial contains one grouped-CV fit per fold.",
    }

    manifest = {
        "run_id": run_dir.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_mode": config.run_mode,
        "bundle_status": bundle_status,
        "selected_feature_count": selected_count,
        "selected_features": selected_features,
        "selected_params": final_params,
        "selected_seed": config.random_state,
        "seeds": active_seeds,
        "folds": folds,
        "operating_threshold": threshold,
        "oof_constraints_met": constraints_met,
        "targets": {"precision": config.precision_target, "recall": config.recall_target},
        "data_hashes": {
            "train_windows": sha256_file(config.train_windows_path),
            "test_windows": sha256_file(config.test_windows_path),
            "feature_decisions": sha256_file(config.feature_decisions_path),
        },
        "data_rows": {
            "fitting_train": len(train), "fixed_test": len(test_main),
            "excluded_safee_train": len(safee_train), "safee_positive_stress": len(safee_test),
        },
        "group_audit": {
            "train_test_overlap": 0,
            "subject_id_normalization": "integer-like aliases normalized (e.g. 001 == 1 == 1.0)",
        },
        "execution_summary": execution_summary,
        "versions": {
            "python": platform.python_version(), "lightgbm": lgb.__version__,
            "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
        },
        "safee_stress_test": safee_stress,
        "test_gate_passed": gate_passed,
        "domain_gate": {"lodo_not_below_random": lodo_ok, "test_not_below_random": test_domain_ok},
        "config": config.to_dict(),
    }

    comparison.to_csv(run_dir / "comparison.csv", index=False)
    tuning.to_csv(run_dir / "optuna_trials.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(run_dir / "seed_metrics.csv", index=False)
    pd.DataFrame(split_audits).to_csv(run_dir / "split_audit.csv", index=False)
    pd.DataFrame(split_manifest_rows).to_csv(run_dir / "split_manifest.csv", index=False)
    fit_audit.to_csv(run_dir / "fit_audit.csv", index=False)
    oof_out.to_csv(run_dir / "oof_predictions.csv", index=False)
    test_out.to_csv(run_dir / "test_predictions.csv", index=False)
    sweep.to_csv(run_dir / "threshold_sweep.csv", index=False)
    errors.to_csv(run_dir / "errors_fp_fn.csv", index=False)
    test_per_dataset.to_csv(run_dir / "metrics_per_dataset.csv", index=False)
    oof_per_dataset.to_csv(run_dir / "oof_metrics_per_dataset.csv", index=False)
    lodo_metrics.to_csv(run_dir / "lodo_metrics.csv", index=False)
    lodo_predictions.to_csv(run_dir / "lodo_predictions.csv", index=False)
    write_json(run_dir / "metrics_overall.json", {"oof": oof_metrics, "test": test_metrics, "safee": safee_stress})
    write_json(run_dir / "run_config.json", config.to_dict())
    write_json(run_dir / "execution_summary.json", execution_summary)
    export_bundle(
        run_dir / "model_bundle", final_model, selected_features, calibration,
        threshold, train, manifest,
    )
    report = _decision_report(manifest, comparison, test_metrics)
    (run_dir / "decision_report.md").write_text(report, encoding="utf-8")
    return run_dir
