from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


EPS = 1e-7


def _logit(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


@dataclass(frozen=True, slots=True)
class PlattCalibration:
    coefficient: float
    intercept: float
    input: str = "logit_lightgbm_probability"
    fitted_from: str = "out_of_fold_train_predictions"

    def apply(self, raw_probabilities: np.ndarray) -> np.ndarray:
        z = self.coefficient * _logit(raw_probabilities) + self.intercept
        z = np.clip(z, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-z))

    def to_dict(self) -> dict:
        return asdict(self)


def fit_platt(raw_probabilities: np.ndarray, labels: np.ndarray) -> PlattCalibration:
    y = np.asarray(labels, dtype=int)
    if len(np.unique(y)) != 2:
        raise ValueError("Platt calibration requires both classes")
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(_logit(raw_probabilities).reshape(-1, 1), y)
    return PlattCalibration(float(model.coef_[0, 0]), float(model.intercept_[0]))
