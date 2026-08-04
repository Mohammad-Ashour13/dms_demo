from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from dms_training import OUTPUT_CONTRACT_VERSION
except ModuleNotFoundError:
    from dms_final_system.training_colab.dms_training import OUTPUT_CONTRACT_VERSION

try:
    from dms_training.data import (
        balanced_hierarchical_weights,
        canonical_group,
        canonical_subject_id,
    )
    from dms_training.features import make_feature_sets, ranked_feature_names
    from dms_training.splits import grouped_folds
except ModuleNotFoundError:  # Full-repository test execution.
    from dms_final_system.training_colab.dms_training.data import (
        balanced_hierarchical_weights,
        canonical_group,
        canonical_subject_id,
    )
    from dms_final_system.training_colab.dms_training.features import (
        make_feature_sets,
        ranked_feature_names,
    )
    from dms_final_system.training_colab.dms_training.splits import grouped_folds


def test_unknown_subject_falls_back_to_video():
    frame = pd.DataFrame(
        {"dataset_name": ["SUST", "NTHU"], "subject_id": ["unknown", "4"], "video_id": ["v1", "v2"]}
    )
    assert canonical_group(frame).tolist() == ["SUST::video::v1", "NTHU::4"]


def test_colab_output_contract_version():
    assert OUTPUT_CONTRACT_VERSION == 2


def test_numeric_subject_aliases_are_one_canonical_driver():
    assert [canonical_subject_id(value) for value in ("001", "1", 1, 1.0)] == ["1"] * 4
    frame = pd.DataFrame(
        {
            "dataset_name": ["NTHU"] * 4,
            "subject_id": ["005", "5", 5, 5.0],
            "video_id": ["a", "b", "c", "d"],
        }
    )
    assert canonical_group(frame).nunique() == 1


def test_feature_ranking_rejects_metadata_and_mouth_threshold_features():
    decisions = pd.DataFrame(
        {
            "Feature": ["ear_mean", "mouth_open_active_ratio", "dataset_name", "yaw_mean"],
            "Decision": ["KEEP", "KEEP", "KEEP", "REVIEW"],
            "Consensus Score": [90, 99, 100, 80],
        }
    )
    ranked = ranked_feature_names(decisions, set(decisions["Feature"]))
    assert ranked == ["ear_mean", "yaw_mean"]
    assert make_feature_sets(ranked, [1, 2])[2] == ["ear_mean", "yaw_mean"]


def test_group_folds_have_no_overlap():
    rows = []
    for label in (0, 1):
        for group in range(6):
            rows.extend({"group_id": f"g-{label}-{group}", "label": label} for _ in range(3))
    frame = pd.DataFrame(rows)
    for train_idx, valid_idx in grouped_folds(frame, 3, 42):
        assert set(frame.iloc[train_idx].group_id).isdisjoint(frame.iloc[valid_idx].group_id)


def test_hierarchical_weights_equalize_dataset_class_mass():
    frame = pd.DataFrame(
        {
            "dataset_name": ["A"] * 8 + ["B"] * 4,
            "label": [0] * 6 + [1] * 2 + [0] * 2 + [1] * 2,
            "group_id": ["a0"] * 6 + ["a1"] * 2 + ["b0"] * 2 + ["b1"] * 2,
        }
    )
    frame["w"] = balanced_hierarchical_weights(frame)
    masses = frame.groupby(["dataset_name", "label"])["w"].sum().to_numpy()
    assert np.allclose(masses, masses[0])
