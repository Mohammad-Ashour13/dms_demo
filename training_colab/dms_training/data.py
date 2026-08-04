from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .feature_contract import audit_features, is_deployable_feature


REQUIRED_META = {"label", "dataset_name", "subject_id", "video_id"}
UNKNOWN_SUBJECTS = {"unknown", "nan", "none", "", "-1"}
INTEGER_LIKE_ID = re.compile(r"^[+]?0*\d+(?:\.0+)?$")


def validate_schema(train_columns: list[str], test_columns: list[str]) -> None:
    missing = sorted(REQUIRED_META - set(train_columns))
    if missing:
        raise ValueError(f"Training CSV is missing metadata columns: {missing}")
    missing_test = sorted(REQUIRED_META - set(test_columns))
    if missing_test:
        raise ValueError(f"Test CSV is missing metadata columns: {missing_test}")
    train_features = {c for c in train_columns if is_deployable_feature(c)}
    absent = sorted(train_features - set(test_columns))
    if absent:
        raise ValueError(f"Test CSV is missing deployable train features: {absent[:20]}")


def canonical_subject_id(value: object) -> str | None:
    """Normalize aliases such as ``001``, ``1`` and ``1.0`` to one driver ID."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.lower() in UNKNOWN_SUBJECTS:
        return None
    if INTEGER_LIKE_ID.fullmatch(text):
        integer_part = text.split(".", 1)[0].lstrip("+")
        return str(int(integer_part))
    return text.casefold()


def canonical_group(frame: pd.DataFrame) -> pd.Series:
    dataset = frame["dataset_name"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
    subject = frame["subject_id"].map(canonical_subject_id)
    video = frame["video_id"].fillna("unknown-video").astype(str).str.strip()
    entity = subject.where(subject.notna(), "video::" + video)
    return dataset + "::" + entity


def prepare_frame(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    required = REQUIRED_META | set(feature_names)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing[:25]}")
    audit = audit_features(feature_names)
    problems = {k: v for k, v in audit.items() if v}
    if problems:
        raise ValueError(f"Non-deployable/leaking features selected: {problems}")
    out = frame.copy()
    out["group_id"] = canonical_group(out)
    out["label"] = pd.to_numeric(out["label"], errors="raise").astype(int)
    if not set(out["label"].unique()).issubset({0, 1}):
        raise ValueError("label must be binary 0/1")
    for name in feature_names:
        out[name] = pd.to_numeric(out[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out


def exclude_safee(frame: pd.DataFrame, safee_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = frame["dataset_name"].astype(str).str.upper().eq(safee_name.upper())
    return frame.loc[~mask].reset_index(drop=True), frame.loc[mask].reset_index(drop=True)


def balanced_hierarchical_weights(frame: pd.DataFrame) -> np.ndarray:
    """Equal dataset -> class -> subject/video contribution, normalized to mean one."""
    columns = ["dataset_name", "label", "group_id"]
    work = frame[columns].reset_index(drop=True).copy()
    work["video_id"] = (
        frame["video_id"].reset_index(drop=True).fillna("unknown-video").astype(str)
        if "video_id" in frame else work["group_id"]
    )
    datasets = max(1, work["dataset_name"].nunique())
    weights = np.zeros(len(work), dtype=np.float64)
    for dataset, ds_idx in work.groupby("dataset_name", sort=False).groups.items():
        ds = work.loc[ds_idx]
        classes = max(1, ds["label"].nunique())
        for label, class_idx in ds.groupby("label", sort=False).groups.items():
            cell = work.loc[class_idx]
            groups = max(1, cell["group_id"].nunique())
            for _, group_idx in cell.groupby("group_id", sort=False).groups.items():
                subject_cell = work.loc[group_idx]
                videos = max(1, subject_cell["video_id"].nunique())
                for _, video_idx in subject_cell.groupby("video_id", sort=False).groups.items():
                    weights[np.asarray(video_idx, dtype=int)] = 1.0 / (
                        datasets * classes * groups * videos * max(1, len(video_idx))
                    )
    mean = float(weights.mean())
    if mean <= 0.0 or not np.isfinite(mean):
        raise ValueError("Could not construct finite positive sample weights")
    return weights / mean


def group_safe_smoke_sample(frame: pd.DataFrame, max_groups: int, seed: int) -> pd.DataFrame:
    summary = frame.groupby("group_id", as_index=False).agg(
        dataset_name=("dataset_name", "first"), label=("label", "max")
    )
    selected: list[str] = []
    rng = np.random.default_rng(seed)
    cells = list(summary.groupby(["dataset_name", "label"], sort=True))
    per_cell = max(2, max_groups // max(1, len(cells)))
    for _, cell in cells:
        groups = cell["group_id"].to_numpy()
        take = min(len(groups), per_cell)
        selected.extend(rng.choice(groups, size=take, replace=False).tolist())
    return frame[frame["group_id"].isin(selected)].reset_index(drop=True)
