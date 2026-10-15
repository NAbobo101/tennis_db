"""Temperature scaling: a one-parameter, symmetry-preserving probability calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

FloatArray = npt.NDArray[np.float64]
_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class TemperatureScaler:
    """Rescale log-odds by ``slope`` (= 1 / temperature).

    Platt scaling without an intercept: unlike full Platt scaling it keeps
    ``p(A beats B) = 1 - p(B beats A)``, which matters because the player order is arbitrary.
    """

    slope: float = 1.0

    @classmethod
    def fit(cls, p_raw: npt.ArrayLike, y: npt.ArrayLike) -> TemperatureScaler:
        """Choose the slope minimizing log-loss on held-out predictions."""
        z = logit(np.clip(np.asarray(p_raw, dtype=float), _EPS, 1 - _EPS))
        target = np.asarray(y, dtype=float)

        def loss(slope: float) -> float:
            p = np.clip(expit(slope * z), _EPS, 1 - _EPS)
            return float(-np.mean(target * np.log(p) + (1 - target) * np.log(1 - p)))

        result = minimize_scalar(loss, bounds=(0.05, 5.0), method="bounded")
        return cls(slope=float(result.x))

    def transform(self, p_raw: npt.ArrayLike) -> FloatArray:
        """Calibrated probabilities."""
        z = logit(np.clip(np.asarray(p_raw, dtype=float), _EPS, 1 - _EPS))
        return np.asarray(expit(self.slope * z), dtype=np.float64)
