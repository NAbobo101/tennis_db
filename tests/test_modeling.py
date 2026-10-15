from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss

from tennis_predictor.config import Settings
from tennis_predictor.features.dataset import Dataset
from tennis_predictor.features.specs import swap_players
from tennis_predictor.features.tracker import FeatureTracker
from tennis_predictor.modeling.artifacts import (
    ArtifactMetadata,
    load_artifacts,
    load_metadata,
    load_predictions,
)
from tennis_predictor.modeling.baselines import EloBaseline, RankBaseline
from tennis_predictor.modeling.calibration import TemperatureScaler
from tennis_predictor.modeling.train import (
    make_logistic,
    positive_proba,
    symmetric_proba,
    train_models,
)

PROBABILITIES = ["p_rank", "p_elo", "p_logistic", "p_xgboost_raw", "p_xgboost"]


@pytest.fixture(scope="module")
def result(dataset: Dataset, settings: Settings):
    return train_models(dataset, settings)


def test_predictions_cover_validation_and_test(result, dataset: Dataset) -> None:
    predictions = result.predictions
    assert set(predictions["split"]) == {"val", "test"}
    assert len(predictions) == result.row_counts["val"] + result.row_counts["test"]
    assert result.row_counts["train"] == len(dataset.subset("train"))
    for column in PROBABILITIES:
        assert predictions[column].between(0, 1).all(), column


def test_models_learn_the_latent_skill(result) -> None:
    test = result.predictions.loc[result.predictions["split"] == "test"]
    coin_flip = log_loss(test["y"], np.full(len(test), 0.5))
    for column in ("p_elo", "p_logistic", "p_xgboost"):
        assert log_loss(test["y"], test[column]) < coin_flip - 0.02, column


def test_symmetric_proba_sums_to_one(result, dataset: Dataset) -> None:
    X = dataset.subset("test").X
    forward = symmetric_proba(positive_proba(result.xgboost), X)
    backward = symmetric_proba(positive_proba(result.xgboost), swap_players(X))
    np.testing.assert_allclose(forward + backward, 1.0, atol=1e-6)


def test_logistic_and_rank_are_symmetric_by_construction(dataset: Dataset) -> None:
    train = dataset.subset("train")
    X = dataset.subset("test").X
    logistic = make_logistic(1.0).fit(train.X, train.y)
    p = positive_proba(logistic)
    np.testing.assert_allclose(p(X) + p(swap_players(X)), 1.0, atol=1e-9)
    rank = RankBaseline().fit(train.X, train.y)
    np.testing.assert_allclose(
        rank.predict_proba_a(X) + rank.predict_proba_a(swap_players(X)), 1.0, atol=1e-9
    )


def test_elo_baseline_reads_the_blend_logit() -> None:
    X = pd.DataFrame({"elo_blend_logit": [0.0, np.log(3.0)]})
    np.testing.assert_allclose(EloBaseline().predict_proba_a(X), [0.5, 0.75])


def test_temperature_scaler() -> None:
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.2, 0.8, size=20_000)
    y = rng.random(20_000) < true_p
    overconfident = 1 / (1 + np.exp(-2.0 * np.log(true_p / (1 - true_p))))
    scaler = TemperatureScaler.fit(overconfident, y)
    assert scaler.slope == pytest.approx(0.5, abs=0.05)
    assert log_loss(y, scaler.transform(overconfident)) < log_loss(y, overconfident)
    np.testing.assert_allclose(scaler.transform([0.3]) + scaler.transform([0.7]), 1.0)
    assert TemperatureScaler().transform([0.25])[0] == pytest.approx(0.25)


def test_train_requires_every_split(dataset: Dataset, settings: Settings) -> None:
    empty_test = Dataset(
        dataset.X,
        dataset.y,
        dataset.meta.assign(split=dataset.meta["split"].replace("test", "val")),
    )
    with pytest.raises(ValueError, match="non-empty"):
        train_models(empty_test, settings)


def test_artifacts_round_trip(
    trained_settings: Settings, built: tuple[Dataset, FeatureTracker]
) -> None:
    directory: Path = trained_settings.model_dir
    metadata = load_metadata(directory)
    assert isinstance(metadata, ArtifactMetadata)
    assert metadata.feature_names[0] == "elo_diff"
    assert {"xgboost", "pandas", "python"} <= metadata.library_versions.keys()
    artifacts = load_artifacts(directory)
    predictions = load_predictions(directory)
    dataset, _ = built
    test = dataset.subset("test")
    reloaded = artifacts.calibrator.transform(
        symmetric_proba(positive_proba(artifacts.model), test.X)
    )
    saved = predictions.loc[predictions["split"] == "test", "p_xgboost"].to_numpy()
    np.testing.assert_allclose(reloaded, saved, atol=1e-6)


def test_missing_artifacts_raise(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="tennis train"):
        load_metadata(tmp_path)
    with pytest.raises(FileNotFoundError, match="tennis train"):
        load_predictions(tmp_path)
