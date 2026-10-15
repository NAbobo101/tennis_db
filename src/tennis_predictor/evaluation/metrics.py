"""Metrics for probabilistic binary forecasts."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

_EPS = 1e-15


@dataclass(frozen=True, slots=True)
class Metrics:
    """Scores of one model on one set of matches (lower is better for the losses)."""

    n: int
    log_loss: float
    brier: float
    accuracy: float
    roc_auc: float
    ece: float

    def as_dict(self) -> dict[str, float]:
        """Plain dictionary, convenient for JSON."""
        return asdict(self)


def expected_calibration_error(y: npt.ArrayLike, p: npt.ArrayLike, n_bins: int = 10) -> float:
    """Weighted mean gap between predicted and observed frequency over equal-width bins."""
    target = np.asarray(y, dtype=float)
    proba = np.asarray(p, dtype=float)
    bins = np.minimum((proba * n_bins).astype(int), n_bins - 1)
    error = 0.0
    for index in range(n_bins):
        mask = bins == index
        if mask.any():
            error += mask.mean() * abs(proba[mask].mean() - target[mask].mean())
    return float(error)


def compute_metrics(y: npt.ArrayLike, p: npt.ArrayLike) -> Metrics:
    """Log-loss, Brier score, accuracy, ROC-AUC and ECE of probabilities ``p`` for labels ``y``."""
    target = np.asarray(y, dtype=int)
    proba = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    both_classes = len(np.unique(target)) == 2
    return Metrics(
        n=len(target),
        log_loss=float(log_loss(target, proba, labels=[0, 1])),
        brier=float(brier_score_loss(target, proba)),
        accuracy=float(np.mean((proba > 0.5) == (target == 1))),
        roc_auc=float(roc_auc_score(target, proba)) if both_classes else float("nan"),
        ece=expected_calibration_error(target, proba),
    )


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Paired difference ``model - reference`` with a percentile confidence interval."""

    metric: str
    difference: float
    lower: float
    upper: float

    def as_dict(self) -> dict[str, float | str]:
        """Plain dictionary, convenient for JSON."""
        return asdict(self)


def paired_bootstrap(
    y: npt.ArrayLike,
    p_model: npt.ArrayLike,
    p_reference: npt.ArrayLike,
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> list[BootstrapResult]:
    """Bootstrap the per-match differences in log-loss and Brier score (negative = model better)."""
    target = np.asarray(y, dtype=float)
    model = np.clip(np.asarray(p_model, dtype=float), _EPS, 1 - _EPS)
    reference = np.clip(np.asarray(p_reference, dtype=float), _EPS, 1 - _EPS)

    def point_log_loss(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.asarray(-(target * np.log(p) + (1 - target) * np.log(1 - p)), dtype=np.float64)

    deltas = {
        "log_loss": point_log_loss(model) - point_log_loss(reference),
        "brier": (model - target) ** 2 - (reference - target) ** 2,
    }
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(target), size=(n_resamples, len(target)))
    alpha = (1 - confidence) / 2
    results = []
    for name, delta in deltas.items():
        samples = delta[indices].mean(axis=1)
        lower, upper = np.quantile(samples, [alpha, 1 - alpha])
        results.append(BootstrapResult(name, float(delta.mean()), float(lower), float(upper)))
    return results
