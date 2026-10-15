from __future__ import annotations

import math

import numpy as np
import pytest

from tennis_predictor.evaluation.metrics import (
    compute_metrics,
    expected_calibration_error,
    paired_bootstrap,
)


def test_compute_metrics_known_values() -> None:
    y = np.array([1, 0, 1, 0])
    p = np.array([0.8, 0.2, 0.6, 0.4])
    metrics = compute_metrics(y, p)
    assert metrics.n == 4
    assert metrics.accuracy == 1.0
    assert metrics.roc_auc == 1.0
    assert metrics.brier == pytest.approx((0.04 + 0.04 + 0.16 + 0.16) / 4)
    assert metrics.log_loss == pytest.approx(-(2 * math.log(0.8) + 2 * math.log(0.6)) / 4)
    assert metrics.as_dict()["n"] == 4


def test_single_class_has_no_auc() -> None:
    assert math.isnan(compute_metrics([1, 1], [0.6, 0.7]).roc_auc)


def test_expected_calibration_error() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=50_000)
    y = rng.random(50_000) < p
    assert expected_calibration_error(y, p) < 0.01
    assert expected_calibration_error(y, np.clip(p + 0.2, 0, 1)) > 0.1


def test_paired_bootstrap_detects_a_better_model() -> None:
    rng = np.random.default_rng(0)
    p_true = rng.uniform(0.1, 0.9, size=3000)
    y = rng.random(3000) < p_true
    results = paired_bootstrap(y, p_true, np.full(3000, 0.5), n_resamples=500)
    assert [result.metric for result in results] == ["log_loss", "brier"]
    for result in results:
        assert result.upper < 0  # the informed model is better than the coin flip
        assert result.lower <= result.difference <= result.upper
        assert result.as_dict()["metric"] == result.metric
