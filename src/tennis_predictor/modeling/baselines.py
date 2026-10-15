"""Reference models every learned model must beat."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

FloatArray = npt.NDArray[np.float64]


class EloBaseline:
    """Probability implied by the surface-blended Elo ratings (no fitting needed)."""

    name = "elo"

    def predict_proba_a(self, X: pd.DataFrame) -> FloatArray:
        """Probability that player A wins."""
        return np.asarray(expit(X["elo_blend_logit"].to_numpy(dtype=float)), dtype=np.float64)


class RankBaseline:
    """Logistic regression on the log-rank ratio: the "better-ranked player wins" model.

    It has no intercept, so it is symmetric in the two players by construction.
    """

    name = "rank"
    columns = ("log_rank_diff", "unranked_diff")

    def __init__(self) -> None:
        self.model = LogisticRegression(fit_intercept=False, max_iter=1000)

    def fit(self, X: pd.DataFrame, y: npt.ArrayLike) -> RankBaseline:
        """Fit the two coefficients."""
        self.model.fit(X[list(self.columns)].to_numpy(dtype=float), y)
        return self

    def predict_proba_a(self, X: pd.DataFrame) -> FloatArray:
        """Probability that player A wins."""
        proba = self.model.predict_proba(X[list(self.columns)].to_numpy(dtype=float))
        return np.asarray(proba[:, 1], dtype=np.float64)
