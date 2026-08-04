from __future__ import annotations

import pandas as pd

from .feature_contract import is_deployable_feature


def ranked_feature_names(decisions: pd.DataFrame, available: set[str]) -> list[str]:
    required = {"Feature", "Decision", "Consensus Score"}
    missing = sorted(required - set(decisions.columns))
    if missing:
        raise ValueError(f"feature_decisions.csv missing columns: {missing}")
    work = decisions.copy()
    work["Feature"] = work["Feature"].astype(str)
    work["Decision"] = work["Decision"].astype(str).str.upper()
    work["Consensus Score"] = pd.to_numeric(work["Consensus Score"], errors="coerce").fillna(-1)
    work = work[work["Decision"].isin({"KEEP", "REVIEW"})]
    work = work[work["Feature"].isin(available)]
    work = work[work["Feature"].map(is_deployable_feature)]
    work["decision_rank"] = work["Decision"].map({"KEEP": 0, "REVIEW": 1})
    work = work.sort_values(
        ["decision_rank", "Consensus Score", "Feature"], ascending=[True, False, True]
    ).drop_duplicates("Feature")
    names = work["Feature"].tolist()
    if not names:
        raise ValueError("No deployable KEEP/REVIEW features remain after contract filtering")
    return names


def make_feature_sets(ranked: list[str], counts: list[int]) -> dict[int, list[str]]:
    sets: dict[int, list[str]] = {}
    for count in sorted(set(counts)):
        if len(ranked) < count:
            raise ValueError(f"Requested Top-{count}, but only {len(ranked)} deployable features exist")
        sets[count] = ranked[:count]
    return sets
