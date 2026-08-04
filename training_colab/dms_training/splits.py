from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def grouped_folds(frame: pd.DataFrame, n_splits: int, seed: int):
    # A driver legitimately owns both awake and drowsy videos. StratifiedGroupKFold
    # keeps the complete driver in one fold while balancing row-level labels.
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yield from splitter.split(np.zeros(len(frame)), frame["label"], frame["group_id"])


def assert_no_group_overlap(frame: pd.DataFrame, train_idx: np.ndarray, valid_idx: np.ndarray) -> None:
    train_groups = set(frame.iloc[train_idx]["group_id"])
    valid_groups = set(frame.iloc[valid_idx]["group_id"])
    overlap = train_groups & valid_groups
    if overlap:
        raise AssertionError(f"Group leakage detected: {sorted(overlap)[:10]}")
