from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(slots=True)
class ExperimentConfig:
    train_windows_path: Path
    test_windows_path: Path
    feature_decisions_path: Path
    output_path: Path
    run_mode: Literal["smoke", "full"] = "smoke"
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])
    feature_counts: list[int] = field(default_factory=lambda: [7, 20, 40, 65])
    target_col: str = "label"
    dataset_col: str = "dataset_name"
    subject_col: str = "subject_id"
    video_col: str = "video_id"
    safee_name: str = "SAFEE"
    smoke_max_groups: int = 30
    n_folds_smoke: int = 3
    n_folds_full: int = 5
    optuna_trials: int = 25
    recall_target: float = 0.90
    precision_target: float = 0.75
    random_state: int = 42
    use_gpu: bool = False
    n_jobs: int = -1

    def validate(self) -> None:
        for path in (
            self.train_windows_path,
            self.test_windows_path,
            self.feature_decisions_path,
        ):
            if not Path(path).is_file():
                raise FileNotFoundError(f"Required input not found: {path}")
        if self.run_mode not in {"smoke", "full"}:
            raise ValueError("run_mode must be 'smoke' or 'full'")
        if not self.feature_counts or min(self.feature_counts) < 1:
            raise ValueError("feature_counts must contain positive integers")

    def to_dict(self) -> dict:
        data = asdict(self)
        for key in ("train_windows_path", "test_windows_path", "feature_decisions_path", "output_path"):
            data[key] = str(data[key])
        return data
